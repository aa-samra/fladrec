#!/usr/bin/env python3
"""
Train TargetSASRec using the best hyperparameters found by Optuna.

This script:
1. Loads dataset (train/val parquet)
2. Loads precomputed cross-domain user embeddings
3. Loads the best hyperparameters for the given fusion mode and dataset
4. Restores the checkpoint of the single-domain target SASRec model
5. Trains TargetSASRec using best params
6. Saves final checkpoint to:  <checkpoint_dir>/<dataset>_<fusion_mode>_ff.pth
"""

import logging
import os
import pickle as pkl
import torch
import polars as pl
from pathlib import Path

from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from optuna.artifacts import FileSystemArtifactStore, download_artifact
import optuna

from fladrec.data.sequential import TrainDataset, collate_fn, GPUSASRecDataloader
from fladrec.evaluation.eval import eval_model, get_eval_dataloader
from fladrec.models.fl_models import TargetSASRec
from fladrec.training.single_domain import train_sasrec



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Utility
# -----------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, dataloader, optimizer, cd_emb, cd_user_ids, device):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader)
    for batch in loop:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(batch, cd_emb=cd_emb, cd_user_ids=cd_user_ids)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    return total_loss / len(dataloader)


def wrap_for_eval(model, cd_emb, cd_user_ids):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.cd_emb = cd_emb
            self.cd_user_ids = cd_user_ids
            self.item_embeddings = model.item_embeddings
            self._num_items = model._num_items

        def forward(self, batch):
            return self.model(batch, cd_emb=self.cd_emb, cd_user_ids=self.cd_user_ids)

    return Wrapper()

def get_target_sasrec_params(cfg):
    """
    Returns the best hyperparameters for TargetSASRec for a given fusion mode.
    Reads from cfg.dataset.best_hp, where each fusion_mode has its own section.
    """

    # Root node: cfg.dataset.best_hp
    if "best_hp" not in cfg.dataset:
        raise ValueError("cfg.dataset.best_hp section missing in dataset config.")

    best_hp = cfg.dataset.best_hp

    # 1) Base hyperparameters (common for all modes)
    if "base" not in best_hp:
        raise ValueError("cfg.dataset.best_hp.base is required.")

    params = dict(best_hp.base)

    # 2) Fusion-mode–specific overrides
    fusion_mode = cfg.fusion_mode
    if fusion_mode not in best_hp:
        raise ValueError(
            f"cfg.dataset.best_hp does not provide hyperparameters for fusion mode: {fusion_mode}"
        )

    fusion_specific = best_hp[fusion_mode]
    params.update(fusion_specific)

    return params


# -----------------------------
# Main training function
# -----------------------------

@hydra.main(version_base=None, config_path="../config/ff", config_name="train")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    device = cfg.device
    dataset_name = cfg.dataset.name
    fusion_mode = cfg.fusion_mode

    # ---------------------------------
    # Load data
    # ---------------------------------
    data_dir = Path(cfg.dataset.data_dir)
    train_df = (
        pl.scan_parquet(data_dir / "train.parquet")
        .group_by("uid")
        .agg(pl.col("item_id"), pl.col("timestamp"))
        .collect(engine="streaming")
    )
    val_df = (
        pl.scan_parquet(data_dir / "val.parquet")
        .group_by("uid")
        .agg(pl.col("item_id"), pl.col("timestamp"))
        .collect(engine="streaming")
    )

    with open(data_dir / "item_id_to_idx.pkl", "rb") as f:
        item_id_to_idx = pkl.load(f)
    num_items = len(item_id_to_idx)

    # Cross-domain embeddings
    cd_path = cfg.dataset.cd_emb_path

# If the user provided only a filename → load from ../embeddings/
    if "/" not in cd_path:
        load_path = os.path.join("..", "embeddings", cd_path)
    else:
        # Otherwise treat it as a full path exactly as entered
        load_path = cd_path

    logger.info(f"Loading external embeddings from: {load_path}")
    cd_user_data = torch.load(load_path, map_location=device)
    cd_emb = cd_user_data["user.emb"].to(device).nan_to_num(0)
    cd_user_ids = cd_user_data["user.ids"].to(device)

    # Best hyperparameters
    best_params = get_target_sasrec_params(cfg)
    print(best_params)
    lr = best_params['learning_rate']
    best_params.pop('learning_rate')


    # ---------------------------------
    # Build model
    # ---------------------------------
    num_aug_tokens = best_params.get("num_cd_tokens", 0)

    model = TargetSASRec(
        fusion_mode=fusion_mode,
        cd_emb_dim=cd_emb.shape[1],
        num_items=num_items,
        cd_user_emb=cd_emb if "trainable" in fusion_mode else None,
        cd_user_ids=cd_user_ids if "trainable" in fusion_mode else None,
        **best_params,
    ).to(device)

    # initialize with best SD state
    dict_state = torch.load(f'../checkpoints/{cfg.dataset.name}_best_sd.pth', map_location=cfg.device)
    model.load_state_dict(dict_state, strict=False)

    # ---------------------------------
    # Freeze / unfreeze
    # ---------------------------------
    for name, p in model.named_parameters():
        if name in dict_state and cfg.trainable_base:
            p.requires_grad = False
        else:
            p.requires_grad = True

    # ---------------------------------
    # Datasets & Dataloaders
    # ---------------------------------
    max_len = best_params["max_sequence_length"] - num_aug_tokens

    train_dataset = TrainDataset(
        train_df, num_items=num_items, max_seq_len=max_len, num_neg_items=1
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    eval_df, eval_loader = get_eval_dataloader(
        train_df, val_df, max_len, cfg.batch_size * 4,
        seed=42, eval_mode=cfg.eval_mode
    )
    if cfg.fastloader:
        train_loader = GPUSASRecDataloader(train_loader, device=device)

    # ---------------------------------
    # Optimizer
    # ---------------------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

        # ---------------------------------
    # Training Loop (with Early Stopping)
    # ---------------------------------
    best_metric_val = -float("inf")
    epochs_no_improve = 0
    metric_name, K = cfg.validation_metric.split("@")
    K = int(K)

    patience = cfg.early_stopping.patience
    min_delta = cfg.early_stopping.min_delta

    best_ckpt_path = f"{cfg.checkpoint_dir}/{dataset_name}_{fusion_mode}_final.pth"

    for epoch in range(cfg.num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, cd_emb, cd_user_ids, device)
        logger.info(f"[Epoch {epoch+1}/{cfg.num_epochs}] Train loss = {loss:.4f}")

        # Evaluate
        wrapper = wrap_for_eval(model, cd_emb, cd_user_ids)
        metrics = eval_model(
            eval_loader,
            wrapper,
            device=device,
            downvote_seen=cfg.dataset.downvote_seen
        )
        current_val = metrics[metric_name][K]

        # Log all metrics
        metric_msg = ", ".join(
            f"{m}@{k}={v:.4f}" for m, ks in metrics.items() for k, v in ks.items()
        )
        logger.info(f"[Epoch {epoch+1}] Validation metrics: {metric_msg}")

        # -------------------------
        # Early Stopping Check
        # -------------------------
        if current_val > best_metric_val * min_delta:
            best_metric_val = current_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(
                f"✔ New best model saved (val={current_val:.4f}) → {best_ckpt_path}"
            )
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement for {epochs_no_improve}/{patience} epochs "
                f"(best={best_metric_val:.4f}, current={current_val:.4f})"
            )

            if epochs_no_improve >= patience:
                logger.info("⛔ Early stopping triggered.")
                break

    logger.info("Training complete.")
    logger.info(f"Best validation metric = {best_metric_val:.4f}")
    logger.info(f"Best model saved at: {best_ckpt_path}")

if __name__ == "__main__":
    main()
