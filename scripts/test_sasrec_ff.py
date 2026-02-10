#!/usr/bin/env python3
"""
Test TargetSASRec using the trained FF checkpoint.

This script:
1. Loads Hydra config
2. Loads train + val parquet, concatenates them
3. Loads test parquet
4. Loads precomputed cross-domain embeddings
5. Loads best FF checkpoint for TargetSASRec
6. Evaluates on the test set and prints all metrics
"""

import logging
import os
import pickle as pkl
from pathlib import Path

import torch
import polars as pl
from torch import nn

import hydra
from omegaconf import DictConfig, OmegaConf

from fladrec.evaluation.eval import eval_model, get_eval_dataloader
from fladrec.models.fl_models import TargetSASRec


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Utility functions
# -----------------------------

def load_cd_embeddings(cd_path: str, device: str):
    """
    Load CD embeddings with the same filename / path rules used in the training script.
    """
    # If user enters only filename, load from ../embeddings/
    if "/" not in cd_path:
        load_path = os.path.join("..", "embeddings", cd_path)
    else:
        load_path = cd_path

    logger.info(f"Loading external embeddings from: {load_path}")
    cd_user_data = torch.load(load_path, map_location=device)
    cd_emb = cd_user_data["user.emb"].to(device)
    cd_user_ids = cd_user_data["user.ids"].to(device)
    return cd_emb, cd_user_ids


def get_target_sasrec_params(cfg):
    """
    Same helper used in training script, included here for consistency.
    """
    if "best_hp" not in cfg.dataset:
        raise ValueError("cfg.dataset.best_hp missing.")

    best_hp = cfg.dataset.best_hp

    if "base" not in best_hp:
        raise ValueError("cfg.dataset.best_hp.base missing.")

    params = dict(best_hp.base)

    fusion_mode = cfg.fusion_mode
    if fusion_mode not in best_hp:
        raise ValueError(f"No hyperparameters defined for fusion mode: {fusion_mode}")

    params.update(best_hp[fusion_mode])
    return params


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


# -----------------------------
# Main evaluation logic
# -----------------------------

@hydra.main(version_base=None, config_path="../config/ff", config_name="test")
def main(cfg: DictConfig):

    device = cfg.device
    dataset_name = cfg.dataset.name
    fusion_mode = cfg.fusion_mode

    data_dir = Path(cfg.dataset.data_dir)

    # ---------------------------------
    # Load train + val → trainval_df
    # ---------------------------------
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
    trainval_df = pl.concat([train_df, val_df])

    # ---------------------------------
    # Load test set
    # ---------------------------------
    test_df = (
        pl.scan_parquet(data_dir / "test.parquet")
        .group_by("uid")
        .agg(pl.col("item_id"), pl.col("timestamp"))
        .collect(engine="streaming")
    )

    # ---------------------------------
    # Load item index mapping
    # ---------------------------------
    with open(data_dir / "item_id_to_idx.pkl", "rb") as f:
        item_id_to_idx = pkl.load(f)
    num_items = len(item_id_to_idx)

    # ---------------------------------
    # Load CD embeddings
    # ---------------------------------
    cd_emb, cd_user_ids = load_cd_embeddings(cfg.dataset.cd_emb_path, device=device)

    # ---------------------------------
    # Load best hyperparameters
    # ---------------------------------
    best_params = get_target_sasrec_params(cfg)
    lr = best_params.get("learning_rate")  # not used during test
    best_params.pop("learning_rate", None)

    num_aug_tokens = best_params.get("num_cd_tokens", 0)
    max_len = best_params["max_sequence_length"] - num_aug_tokens

    # ---------------------------------
    # Build model
    # ---------------------------------
    model = TargetSASRec(
        fusion_mode=fusion_mode,
        cd_emb_dim=cd_emb.shape[1],
        num_items=num_items,
        cd_user_emb=cd_emb if "trainable" in fusion_mode else None,
        cd_user_ids=cd_user_ids if "trainable" in fusion_mode else None,
        **best_params,
    ).to(device)

    # Restore SD checkpoint (as in training)
    # sd_ckpt = f"../checkpoints/{dataset_name}_best_sd.pth"
    # logger.info(f"Loading SD checkpoint: {sd_ckpt}")
    # sd_state = torch.load(sd_ckpt, map_location=device)
    # model.load_state_dict(sd_state, strict=False)

    # Restore FF fine-tuned weights
    ff_ckpt = f"{cfg.checkpoint_dir}/{dataset_name}_{fusion_mode}_final.pth"
    logger.info(f"Loading FF checkpoint: {ff_ckpt}")
    ff_state = torch.load(ff_ckpt, map_location=device)
    model.load_state_dict(ff_state, strict=False)

    # ---------------------------------
    # Evaluation dataloader
    # ---------------------------------
    eval_df, eval_loader = get_eval_dataloader(
        trainval_df,
        test_df,
        max_len,
        cfg.batch_size * 8,
        seed=cfg.seed,
        eval_mode=cfg.eval_mode
    )

    wrapper = wrap_for_eval(model, cd_emb, cd_user_ids)

    # ---------------------------------
    # Run evaluation
    # ---------------------------------
    logger.info("Running evaluation on test set...")
    metrics = eval_model(
        eval_loader,
        wrapper,
        device=device,
        downvote_seen=cfg.dataset.downvote_seen
    )

    # Log all metrics
    logger.info("=== Test Metrics ===")
    for mname, mvals in metrics.items():
        for k, v in mvals.items():
            logger.info(f"{mname}@{k} = {v:.4f}")

    logger.info("=== Testing completed ===")


if __name__ == "__main__":
    main()
