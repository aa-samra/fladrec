import logging
import os
from pathlib import Path
import random

import numpy as np
import polars as pl
import pickle as pkl
import torch
from torch.utils.data import DataLoader

from fladrec.data.sequential import TrainDataset, collate_fn, GPUSASRecDataloader
from fladrec.evaluation.eval import eval_model, get_eval_dataloader
from fladrec.models.sasrec import SASRecPlusEncoder
from fladrec.training.single_domain import train_sasrec

from fladrec.utils.reproducibility import seed_everything, make_deterministic, seed_worker


import hydra


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path="../config", config_name="pretrain")
def main(cfg) -> None:
    # 1. Setup logging
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger()
    logger.handlers.clear() # Clear existing handlers from basicConfig
    
    log_file = log_dir / f'{cfg.domain.name}_pretrain.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler for INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Root logger level
    logger.setLevel(logging.DEBUG)

    logger.info("Starting pre-training script.")
    logger.debug("Configuration:\n%s", cfg)

    # 2. Seed and setup
    seed_everything(cfg.seed)
    make_deterministic(cfg.seed)

    data_path = Path(hydra.utils.to_absolute_path(cfg.domain.path))
    
    logger.debug('Preprocessing data from: %s', data_path)
    train_df = pl.scan_parquet(data_path / f'train.parquet').group_by('uid').agg(pl.col("item_id"), pl.col('timestamp')).collect()
    val_df = pl.scan_parquet(data_path / f'val.parquet').group_by('uid').agg(pl.col("item_id"), pl.col('timestamp')).collect()
    test_df = pl.scan_parquet(data_path / f'test.parquet').group_by('uid').agg(pl.col("item_id"), pl.col('timestamp')).collect()
    
    with open(data_path / f'item_id_to_idx.pkl', 'rb') as f:
        item_id_to_idx = pkl.load(f)
    logger.debug('Preprocessing data has finished!')

    num_items = len(item_id_to_idx)
    
    # num_neg_items_dataset is 0 for ce loss, this is handled later
    train_dataset = TrainDataset(dataset=train_df, num_items=num_items,
                                 max_seq_len=cfg.max_seq_len, num_neg_items=0)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=1,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=4,
        pin_memory=('cuda' in cfg.device),
        # pin_memory_device=cfg.device if 'cuda' in cfg.device else None,
    )

    if cfg.get('fastloader', False):
        logger.info('Using fast data loader.')
        train_dataloader = GPUSASRecDataloader(train_dataloader, cfg.device)

    eval_df, eval_dataloader = get_eval_dataloader(train_df, val_df,
                                                   cfg.max_seq_len, 
                                                   min(1024, cfg.batch_size * 8), 
                                                   seed=cfg.seed, 
                                                   eval_mode=cfg.eval_setup.eval_mode, 
                                                   no_train=cfg.get('no_train_on_eval', False))

    hp = cfg.domain.hp
    
    sce_alpha = 2
    num_neg_items = 1
    if cfg.loss_type == 'sce':
        sce_alpha = hp.get('sce_alpha', 2)
        num_neg_items = hp.get('num_neg_items', 16)
    elif cfg.loss_type == 'bce':
        num_neg_items = hp.get('num_neg_items', 1)
    model = SASRecPlusEncoder(
        loss_type=cfg.loss_type,
        num_items=num_items,
        num_neg_items=num_neg_items,
        max_sequence_length=cfg.max_seq_len,
        sce_alpha=sce_alpha,
        embedding_dim=hp.embedding_dim,
        num_heads=hp.num_heads,
        num_layers=hp.num_layers,
        dropout=hp.dropout,
    ).to(cfg.device)
    print(model._item_embeddings.weight.data.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99 if cfg.decay_lr else 1.0)

    # 5. Training loop
    best_target_metric = 0
    best_metrics = None
    best_checkpoint = None

    logger.info("Evaluating initial model...")
    metrics = eval_model(eval_dataloader=eval_dataloader, model=model, device=cfg.device, downvote_seen=cfg.eval_setup.downvote_seen, sample_metric=cfg.eval_setup.sample_metrics)
    logger.info("Initial metrics: %s", metrics)

    patience = 0
    for epoch in range(cfg.num_epochs):
        logger.info(f"--- Starting Epoch {epoch+1}/{cfg.num_epochs} ---")
        
        # Note: If train_sasrec uses tqdm, its output will go to stderr and not be captured in the log file.
        checkpoint = train_sasrec(
            train_dataloader=train_dataloader, model=model, optimizer=optimizer,
            device=cfg.device, num_epochs=1,
            do_autocast=cfg.do_autocast, grad_accum_steps=cfg.grad_accum_steps,
            epoch_shift=epoch
        )
        model.load_state_dict(checkpoint)

        logger.info("Evaluating model after epoch %d...", epoch + 1)
        metrics = eval_model(eval_dataloader=eval_dataloader, model=model, device=cfg.device, downvote_seen=cfg.eval_setup.downvote_seen, sample_metric=cfg.eval_setup.sample_metrics)
        logger.info("Epoch %d metrics: %s", epoch + 1, metrics)
        
        validation_metric = cfg.eval_setup.validation_metric
        metric_name = validation_metric.split('@')[0]
        metric_k = int(validation_metric.split('@')[1])
        target_metric = metrics[metric_name][metric_k]

        if target_metric > 1.005 * best_target_metric:
            patience = 0
            best_target_metric = target_metric
            best_metrics = metrics
            best_checkpoint = checkpoint
            logger.info(f"Found new best model with {validation_metric}: {best_target_metric:.4f}")
            checkpoint_dir = Path(cfg.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f'{cfg.domain.name}_best_model.pth'
            torch.save(best_checkpoint, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience += 1
            logger.info(f"Metric did not improve. Patience: {patience}/10")

        scheduler.step()
        
        if patience >= 10:
            logger.info("Early stopping triggered after 10 epochs without improvement.")
            break
    
    # Test 
    logger.info("Testing best model...")

    model.load_state_dict(best_checkpoint)
    test_df, test_dataloader = get_eval_dataloader(train_df, test_df, 
                                                   cfg.max_seq_len, 
                                                   min(1024, cfg.batch_size * 8), 
                                                   seed=cfg.seed, 
                                                   eval_mode=cfg.eval_setup.eval_mode, 
                                                   no_train=cfg.get('no_train_on_eval', False))
    
    metrics = eval_model(test_dataloader, model, cfg.device, cfg.eval_setup.downvote_seen, sample_metric=cfg.eval_setup.sample_metrics)
    logger.info("Pre-training finished with metrics")
    logger.info(metrics)



if __name__ == '__main__':
    main()