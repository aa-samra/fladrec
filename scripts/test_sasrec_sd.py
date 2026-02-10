import json
import logging
import os
import pathlib as Path
import random
import polars as pl 

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fladrec.evaluation.eval import eval_model, get_eval_dataloader
from fladrec.models.sasrec import SASRecPlusEncoder

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config/sd", config_name="test")
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
    # 2. Load datasets
    # -----------------------------
    data_path = Path.Path(cfg.dataset.data_dir)
    logger.debug("Preprocessing data...")

    train_df = (
        pl.read_parquet(data_path / "train.parquet")
        .vstack(pl.read_parquet(data_path / "val.parquet"))
    )
    test_df = pl.read_parquet(data_path / "test.parquet")

    # Preprocess for evaluation
    train_df = train_df.sort("timestamp").group_by("uid").agg(pl.col("item_id"), pl.col("timestamp"))
    test_df = test_df.sort("timestamp").group_by("uid").agg(pl.col("item_id"), pl.col("timestamp"))

    logger.debug("Preprocessing data has finished!")

    # -----------------------------
    # 3. Prepare evaluation dataloader
    # -----------------------------
    eval_df, eval_dataloader = get_eval_dataloader(
        train_df=train_df,
        eval_df=test_df,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        eval_mode=cfg.eval_mode,
    )

    # -----------------------------
    # 4. Load the best checkpoint
    # -----------------------------
    dataset_name = os.path.basename(cfg.dataset.data_dir)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{dataset_name}_best_sd.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")

    best_hp = cfg.dataset.best_hp
    logger.info(f"Loading best checkpoint from: {checkpoint_path}")
    logger.info(f"Using hyperparameters: {best_hp}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_items = checkpoint["_item_embeddings.weight"].shape[0] - 1
    max_seq_len = checkpoint["_position_embeddings.weight"].shape[0]

    model = SASRecPlusEncoder(
        loss_type='sce',
        num_items=num_items,
        max_sequence_length=max_seq_len,
        num_neg_items=best_hp.get("num_neg_items", 1),
        num_heads=best_hp["num_heads"],
        num_layers=best_hp["num_layers"],
        embedding_dim=best_hp["embedding_dim"],
        sce_alpha=best_hp.get("sce_alpha", 2),
    ).to(device)

    model.load_state_dict(checkpoint)

    # -----------------------------
    # 5. Evaluate
    # -----------------------------
    metrics = eval_model(
        eval_dataloader=eval_dataloader,
        model=model,
        device=device,
        downvote_seen=cfg.dataset.get("downvote_seen", False),
    )

    logger.info("=== Test Metrics ===")
    for mname, mvals in metrics.items():
        for k, v in mvals.items():
            logger.info(f"{mname}@{k} = {v:.4f}")

    # -----------------------------
    # 6. Save results
    # -----------------------------
    results_path = "results.json"
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append({
        "model_info": OmegaConf.to_container(best_hp, resolve=True),
        "eval_info": {
            "dataset": cfg.dataset.data_dir,
            "batch_size": cfg.batch_size,
            "max_seq_len": cfg.max_seq_len,
            "eval_mode": cfg.eval_mode
        },
        "metrics": str(metrics),
    })

    with open(results_path, "w") as f:
        json.dump(data, f, indent=4)

    return metrics


if __name__ == "__main__":
    main()
