import json
import logging
import os
import pathlib as Path
import random

import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf

from fladrec.models.sasrec import SASRecPlusEncoder
from fladrec.evaluation.eval import infer_users, infer_items, get_eval_dataloader


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config/ff", config_name="extract")
def main(cfg: DictConfig):

    # -----------------------------
    # 1. Set seeds and device
    # -----------------------------
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device

    # -----------------------------
    # 2. Load dataset
    # -----------------------------
    data_path = Path.Path(cfg.dataset.data_dir)
    logger.info(f"Reading dataset from: {data_path}")

    if cfg.split == 'val':
        train_df = pl.read_parquet(data_path / 'train.parquet')
    elif cfg.split == 'test':
        train_df = pl.concat([
            pl.read_parquet(data_path / 'train.parquet'),
            pl.read_parquet(data_path / 'val.parquet')
        ])
    else:
        raise ValueError(f"Invalid split: {cfg.split}")

    # Create fake eval_df for dataloader (1 item for each user)
    src_users = train_df.select('uid').unique().to_numpy().flatten()
    fake_items = np.ones_like(src_users, dtype=np.int64).reshape(-1, 1)
    fake_tmp = train_df['timestamp'].max() * np.ones_like(src_users, dtype=np.int64) + 1

    # Sort and group train_df
    train_df = (
        train_df
        .sort("timestamp")
        .group_by("uid")
        .agg(pl.col("item_id"), pl.col("timestamp"))
    )

    void_df = pl.DataFrame({
        'uid': src_users,
        'item_id': fake_items,
        'timestamp': fake_tmp
    })

    # -----------------------------
    # 3. Build evaluation dataloader
    # -----------------------------
    eval_df, eval_dataloader = get_eval_dataloader(
        train_df=train_df,
        eval_df=void_df,
        max_seq_len=cfg.max_items,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        eval_mode='first',
    )

    # -----------------------------
    # 4. Load checkpoint
    # -----------------------------
    dataset_name = os.path.basename(cfg.dataset.data_dir)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{dataset_name}_best_sd.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Extract dims
    num_items = ckpt["_item_embeddings.weight"].shape[0] - 1
    max_seq_len = ckpt["_position_embeddings.weight"].shape[0]

    best_hp = cfg.dataset.best_hp
    logger.info(f"Using hyperparameters: {best_hp}")

    # -----------------------------
    # 5. Build and load model
    # -----------------------------
    model = SASRecPlusEncoder(
        loss_type=best_hp.get("loss_type", "sce"),
        num_items=num_items,
        max_sequence_length=max_seq_len,
        num_neg_items=best_hp.get("num_neg_items", 1),
        num_heads=best_hp["num_heads"],
        num_layers=best_hp["num_layers"],
        embedding_dim=best_hp["embedding_dim"],
        sce_alpha=best_hp.get("sce_alpha", 2),
    ).to(device)

    model.load_state_dict(ckpt)
    model.eval()

    # -----------------------------
    # 6. Extract User Embeddings
    # -----------------------------
    logger.info("Inferring user embeddings...")

    with torch.no_grad():
        user_ids, user_emb, targets = infer_users(eval_dataloader, model, device)

    logger.info(f"User embeddings shape: {user_emb.shape}")

    # -----------------------------
    # 7. Save Embeddings
    # -----------------------------
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{dataset_name}_{cfg.split}_last{cfg.max_items}.pth"
    )

    torch.save(
        {
            "user.ids": user_ids.cpu(),
            "user.emb": user_emb.cpu(),
        },
        save_path
    )

    logger.info(f"Saved embeddings to: {save_path}")


if __name__ == "__main__":
    main()
