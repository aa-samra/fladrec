import os
import pathlib as Path
import logging
import pickle as pkl
import polars as pl
import torch
from torch.utils.data import DataLoader
import hydra

from fladrec.data.sequential import TrainDataset, collate_fn, GPUSASRecDataloader
from fladrec.evaluation.eval import eval_model, get_eval_dataloader
from fladrec.models.sasrec import SASRecPlusEncoder
from fladrec.training.single_domain import train_sasrec


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config/sd", config_name="train")
def main(cfg):

    torch.set_float32_matmul_precision("high")

    # ----------------------------------------
    # 1. Load Dataset
    # ----------------------------------------
    data_path = Path.Path(cfg.dataset.data_dir)

    train_df = (
        pl.scan_parquet(data_path / "train.parquet")
        .group_by("uid")
        .agg(pl.col("item_id"), pl.col("timestamp"))
        .collect(engine="streaming")
    )

    val_df = (
        pl.scan_parquet(data_path / "val.parquet")
        .group_by("uid")
        .agg(pl.col("item_id"), pl.col("timestamp"))
        .collect(engine="streaming")
    )

    with open(data_path / "item_id_to_idx.pkl", "rb") as f:
        item_id_to_idx = pkl.load(f)

    num_items = len(item_id_to_idx)

    # ----------------------------------------
    # 2. Prepare Training and Eval Loaders
    # ----------------------------------------
    train_dataset = TrainDataset(
        dataset=train_df,
        num_items=num_items,
        max_seq_len=cfg.max_seq_len,
        num_neg_items=cfg.dataset.best_hp.num_neg_items,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
        pin_memory_device=cfg.device,
    )

    if cfg.fastloader:
        train_dataloader = GPUSASRecDataloader(train_dataloader, cfg.device)

    eval_df, eval_dataloader = get_eval_dataloader(
        train_df=train_df,
        eval_df=val_df,
        max_seq_len=cfg.max_seq_len,
        batch_size=min(1024, cfg.batch_size * 16),
        seed=cfg.seed,
        eval_mode=cfg.eval_mode,
    )

    # ----------------------------------------
    # 3. Build Model with Best Hyperparameters
    # ----------------------------------------
    hp = cfg.dataset.best_hp  # best hyperparameters from config

    model = SASRecPlusEncoder(
        loss_type=cfg.loss_type,
        num_items=num_items,
        num_neg_items=hp.num_neg_items,
        max_sequence_length=cfg.max_seq_len,
        sce_alpha=getattr(hp, "sce_alpha", 2),
        embedding_dim=hp.embedding_dim,
        num_heads=hp.num_heads,
        num_layers=hp.num_layers,
        loss_only_grad_accum=cfg.loss_only_grad_accum,
        grad_accum_bs=cfg.loss_only_grad_accum_bs,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.decay_lr)

    metric_name = cfg.validation_metric.split("@")[0]
    metric_k = int(cfg.validation_metric.split("@")[1])

    # ----------------------------------------
    # 4. Training Loop 
    # ----------------------------------------
    best_metric = 0
    best_checkpoint = None
    patience = 0
    max_epochs = 50

    logger.info("Starting training with best hyperparameters...")

    for epoch in range(max_epochs):

        checkpoint = train_sasrec(
            train_dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            device=cfg.device,
            num_epochs=1,
            do_autocast=cfg.do_autocast,
            grad_accum_steps=cfg.grad_accum_steps,
            epoch_shift=epoch,
        )

        model.load_state_dict(checkpoint)

        metrics = eval_model(
            eval_dataloader=eval_dataloader,
            model=model,
            device=cfg.device,
            downvote_seen=cfg.dataset.downvote_seen,
        )

        current_metric = metrics[metric_name][metric_k]

        logger.info(f"Epoch {epoch+1}/{max_epochs} — {metrics}")

        if current_metric > 1.02 * best_metric:
            best_metric = current_metric
            best_checkpoint = checkpoint
            patience = 0
            logger.info(f"New BEST! {metric_name}@{metric_k}: {best_metric:.6f}")
        else:
            patience += 1

        scheduler.step()

        if patience >= 5:
            logger.info("Early stopping triggered.")
            break

    # ----------------------------------------
    # 5. Save Best Checkpoint
    # ----------------------------------------
    dataset_name = cfg.dataset.name
    save_path = f"{cfg.checkpoint_dir}/{dataset_name}_best_sd.pth"
    os.makedirs(f"{cfg.checkpoint_dir}", exist_ok=True)

    torch.save(best_checkpoint, save_path)
    logger.info(f"Best checkpoint saved to: {save_path}")

    # ----------------------------------------
    # 6. Final Evaluation
    # ----------------------------------------
    model.load_state_dict(best_checkpoint)
    final_metrics = eval_model(
        eval_dataloader=eval_dataloader,
        model=model,
        device=cfg.device,
        downvote_seen=cfg.dataset.downvote_seen,
    )

    logger.info("FINAL EVALUATION METRICS:")
    logger.info(final_metrics)


if __name__ == "__main__":
    main()
