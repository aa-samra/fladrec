import logging
from pathlib import Path
import random

import numpy as np
import polars as pl
import pickle as pkl
import torch
import os 
from copy import deepcopy
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, SequentialLR

from fladrec.data.sequential import TrainDataset, collate_fn, GPUSASRecDataloader
from fladrec.evaluation.eval import eval_model, get_eval_dataloader
from fladrec.models.sasrec import SASRecPlusEncoder
from fladrec.models.fl_models import TargetSASRec
from fladrec.training.cross_domain import train_sasrec_cd, evaluate_cd_model


from fladrec.utils.reproducibility import seed_everything, make_deterministic, seed_worker



torch.set_float32_matmul_precision('high')


def get_domain_data(domain_cfg, cfg):
    """Loads and preprocesses data for a single domain."""
    data_path = Path(hydra.utils.to_absolute_path(domain_cfg.path))
    
    if cfg.phase=='train':
        train_scanner = pl.scan_parquet(data_path / f'train.parquet')
    elif cfg.phase=='test':
        train_scanner = pl.concat([
            pl.scan_parquet(data_path / f'train.parquet').with_columns(pl.col("item_id").cast(pl.Int64)), 
            pl.scan_parquet(data_path / f'val.parquet').with_columns(pl.col("item_id").cast(pl.Int64))
            ])

    train_df = train_scanner.group_by('uid').agg(pl.col("item_id"), pl.col('timestamp')).collect()
    
    with open(data_path / 'item_id_to_idx.pkl', 'rb') as f:
        item_id_to_idx = pkl.load(f)
        
    num_items = len(item_id_to_idx)

    train_dataset = TrainDataset(dataset=train_df, num_items=num_items,
                                 max_seq_len=domain_cfg.max_seq_len, num_neg_items=0)
    
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
    )

    return train_df, train_dataloader, num_items, item_id_to_idx


@hydra.main(version_base=None, config_path="../config", config_name="adapt")
def main(cfg):
    # 1. Setup logging
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger()
    logger.handlers.clear()
    
    log_file = log_dir / f'{cfg.transfer.name}_adapt.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.setLevel(logging.DEBUG)

    logger.info("Starting adaptation script for %s.", cfg.transfer.name)
    logger.debug("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # 2. Setup

    try: 
        seed_everything(cfg.seed)
        make_deterministic(cfg.seed)

        # remove suffix 
        transfer_name = cfg.transfer.name.split('_')[0]
        src_domain_name, tgt_domain_name = transfer_name.split('2')

        # 3. Load Domain Specific Configs
        src_domain_cfg_path = Path(hydra.utils.get_original_cwd()) / f'config/domain/{src_domain_name}.yaml'
        tgt_domain_cfg_path = Path(hydra.utils.get_original_cwd()) / f'config/domain/{tgt_domain_name}.yaml'
        src_domain_cfg = OmegaConf.load(src_domain_cfg_path)
        tgt_domain_cfg = OmegaConf.load(tgt_domain_cfg_path)

        # 4. Load Data
        logger.info("Loading source domain data: %s", src_domain_name)
        src_train_df, src_train_dataloader, src_num_items, _ = get_domain_data(src_domain_cfg, cfg)
        src_train_dataset = TrainDataset(src_train_df, num_items=src_num_items, 
                                         max_seq_len=src_domain_cfg.max_seq_len, 
                                         num_neg_items=0)

        logger.info("Loading target domain data: %s", tgt_domain_name)
        tgt_train_df, tgt_train_dataloader, tgt_num_items, _ = get_domain_data(tgt_domain_cfg, cfg)

        if cfg.phase=='train':
            test_split = 'val'
        elif cfg.phase=='test':
            test_split = 'test'

        tgt_val_df = pl.scan_parquet(Path(hydra.utils.to_absolute_path(tgt_domain_cfg.path)) / f'{test_split}.parquet').group_by('uid').agg(pl.col("item_id"), pl.col('timestamp')).collect()
        tgt_test_df = pl.scan_parquet(Path(hydra.utils.to_absolute_path(tgt_domain_cfg.path)) / f'{test_split}.parquet').group_by('uid').agg(pl.col("item_id"), pl.col('timestamp')).collect()
        _, tgt_eval_dataloader = get_eval_dataloader(tgt_train_df, tgt_val_df,
                                                    tgt_domain_cfg.max_seq_len,
                                                    cfg.batch_size * 4,
                                                    seed=cfg.seed,
                                                    eval_mode=cfg.eval_setup.eval_mode)
        
        if cfg.get('fastloader', False):
            tgt_train_dataloader = GPUSASRecDataloader(tgt_train_dataloader, cfg.tgt_device)

        logger.info("Loading pre-trained source model...")
        
        src_model = SASRecPlusEncoder(
            num_items=src_num_items,
            embedding_dim=src_domain_cfg.hp.embedding_dim,
            num_heads=src_domain_cfg.hp.num_heads,
            num_layers=src_domain_cfg.hp.num_layers,
            dropout=cfg.transfer.hp.dropout,
            max_sequence_length=src_domain_cfg.max_seq_len,
        ).to(cfg.src_device)

        for name, param in src_model.named_parameters():
            if cfg.train_src_item_emb and 'item_embedding' in name:
                param.requires_grad = True
            elif cfg.train_src_encoder and 'encoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        src_checkpoint_path = Path(cfg.checkpoint_dir) / f'{src_domain_name}_best_model.pth'
        if not src_checkpoint_path.exists():
            logger.error("Pretrained source model checkpoint not found at: %s", src_checkpoint_path)
            raise FileNotFoundError(f"Checkpoint not found: {src_checkpoint_path}")
            
        src_model.load_state_dict(torch.load(src_checkpoint_path))
        logger.info("Pre-trained source model loaded successfully from %s", src_checkpoint_path)


        tgt_model = TargetSASRec(
            fusion_mode=cfg.fusion_mode,
            num_items=tgt_num_items,
            embedding_dim=tgt_domain_cfg.hp.embedding_dim,
            cd_emb_dim=src_domain_cfg.hp.embedding_dim,
            num_heads=tgt_domain_cfg.hp.num_heads,
            num_layers=tgt_domain_cfg.hp.num_layers,
            dropout=cfg.transfer.hp.dropout,
            max_sequence_length=tgt_domain_cfg.max_seq_len,
            proj_hidden_dim=cfg.transfer.hp.proj_hidden_dim,
            proj_num_layers=cfg.transfer.hp.proj_num_layers,
            normalize_cd=cfg.transfer.hp.normalize_cd,
        ).to(cfg.tgt_device)

        tgt_checkpoint_path = Path(cfg.checkpoint_dir) / f'{tgt_domain_name}_best_model.pth'
        if not tgt_checkpoint_path.exists():
            logger.error("Pretrained target model checkpoint not found at: %s", tgt_checkpoint_path)
            raise FileNotFoundError(f"Checkpoint not found: {tgt_checkpoint_path}")
            
        tgt_model.load_state_dict(torch.load(tgt_checkpoint_path), strict=False)
        logger.info("Pre-trained target model loaded successfully from %s", tgt_checkpoint_path)



        for name, param in tgt_model.named_parameters():
            if cfg.train_tgt_item_emb and 'item_embedding' in name:
                param.requires_grad = True
            elif 'item_embedding' in name:
                param.requires_grad = False   # Freeze item embeddings only 
            else:
                param.requires_grad = True   

        src_optimizer = torch.optim.Adam([
            {'params': src_model.parameters(), 'lr': cfg.transfer.hp.learning_rate_src},
        ])
        tgt_optimizer = torch.optim.Adam([
            {'params': tgt_model._encoder.parameters(), 'lr': cfg.transfer.hp.learning_rate_tgt},
            {'params': tgt_model._projector.parameters(), 'lr': cfg.transfer.hp.learning_rate_fuse},
        ])

        warmup_epochs = 10
        warmup_scheduler = LambdaLR(tgt_optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
        decay_scheduler = ExponentialLR(tgt_optimizer, gamma=0.99)
        tgt_scheduler = SequentialLR(tgt_optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_epochs])

        best_target_metric = 0
        best_checkpoint = None
        patience = 0

        for epoch in range(cfg.epochs):
            logger.info(f"--- Starting Epoch {epoch + 1}/{cfg.epochs} ---")
            

            train_sasrec_cd(
                src_train_dataset=src_train_dataloader.dataset,
                tgt_train_dataloader=tgt_train_dataloader,
                src_model=src_model,
                tgt_model=tgt_model, 
                src_optimizer=src_optimizer,
                tgt_optimizer=tgt_optimizer,
                src_device=cfg.src_device,
                tgt_device=cfg.tgt_device,
                num_epochs=1, 
                epoch_shift=epoch
            )
        
            logger.info("Evaluating on target domain...")
            metrics = evaluate_cd_model(
                src_model=src_model,
                tgt_model=tgt_model,
                eval_dataloader=tgt_eval_dataloader,
                src_train_dataset=src_train_dataloader.dataset,
                src_device=cfg.src_device,
                tgt_device=cfg.tgt_device,
                eval_setup=cfg.eval_setup
            )
            logger.info("Epoch %d metrics: %s", epoch + 1, str(metrics))

            validation_metric = cfg.eval_setup.validation_metric
            metric_name = validation_metric.split('@')[0]
            metric_k = int(validation_metric.split('@')[1])
            target_metric = metrics[metric_name][metric_k]

            if target_metric > 1.005 * best_target_metric:
                patience = 0
                best_target_metric = target_metric
                best_metrics = metrics
                best_tgt_checkpoint = deepcopy(tgt_model.state_dict())
                # save only trainble source params 
                best_src_checkpoint = {}
                for name, param in src_model.named_parameters():
                    if param.requires_grad:
                        best_src_checkpoint[name] = deepcopy(param.data.clone())

                logger.info(f"Found new best model with {validation_metric}: {best_target_metric:.4f}")
                checkpoint_dir = Path(cfg.checkpoint_dir)
                checkpoint_dir.mkdir(exist_ok=True)
                save_path = checkpoint_dir / f'{cfg.transfer.name}_best_model.pth'
                torch.save(best_tgt_checkpoint, save_path)

                src_save_path = checkpoint_dir / f'{cfg.transfer.name}_best_src_model.pth'
                torch.save(best_src_checkpoint, src_save_path)

                logger.info("Saved best adapted model to %s", save_path)
                logger.info("Also saved best source model to %s", src_save_path)

            else:
                patience += 1
                logger.info(f"Metric did not improve. Patience: {patience}/{cfg.max_patience}")

            if patience >= cfg.max_patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", cfg.max_patience)
                break
                
        # Test 
        logger.info("Testing best model...")

        tgt_model.load_state_dict(best_tgt_checkpoint)
        src_model.load_state_dict(best_src_checkpoint, strict=False)
        _, tgt_test_dataloader = get_eval_dataloader(tgt_train_df, tgt_test_df,
                                                    tgt_domain_cfg.max_seq_len,
                                                    cfg.batch_size * 4,
                                                    seed=cfg.seed,
                                                    eval_mode=cfg.eval_setup.eval_mode)
        metrics = evaluate_cd_model(
            src_model=src_model,
            tgt_model=tgt_model,
            eval_dataloader=tgt_test_dataloader,
            src_train_dataset=src_train_dataset,
            src_device=cfg.src_device,
            tgt_device=cfg.tgt_device,
            eval_setup=cfg.eval_setup
        )

        logger.info("Adaptation script finished for %s.", cfg.transfer.name)
        logger.info('with metrics: %s', str(metrics))
        import json
        print(json.dumps(metrics))
    except Exception as ee:
        # logger.info(ee.__str__())
        raise ee

if __name__ == "__main__":
    main()
