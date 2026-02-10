from pathlib import Path
import pickle as pkl
import polars as pl
import optuna
import os
from optuna.artifacts import FileSystemArtifactStore
from typing import Tuple, Dict, Any

def read_domain_data(
    data_path: Path, 
    test: bool=False,
) -> Tuple[pl.DataFrame, pl.DataFrame, int]:
    """
    Load domain-specific train/validation data and item index mapping.

    This function loads sequential interaction data from Parquet files,
    grouped by user (`uid`), and returns:
      - train sequences
      - validation sequences
      - number of unique items

    The Parquet files must contain at least:
      - `uid`       (user identifier)
      - `item_id`   (item identifier)
      - `timestamp` (interaction time)

    The function also loads `item_id_to_idx.pkl`, a mapping from item
    IDs to contiguous integer indices.

    Parameters
    ----------
    data_path : Path
        Path to the directory that contains:
          - `train.parquet`
          - `val.parquet`
          - `item_id_to_idx.pkl`

    Returns
    -------
    train_df : pl.DataFrame
        Polars DataFrame containing grouped training sequences:
        each row = one user, with lists of item_ids & timestamps.

    val_df : pl.DataFrame
        Polars DataFrame containing grouped validation sequences.

    num_items : int
        Total number of items (size of the item index mapping).
    """
    print("Loading data...")

    if test:
        train_df = (
            pl.concat([
                pl.scan_parquet(data_path / 'train.parquet'),
                pl.scan_parquet(data_path / 'val.parquet')])
            .group_by('uid')
            .agg(pl.col('item_id'), pl.col('timestamp'))
            .collect(engine='streaming')
        )

        val_df = (
            pl.scan_parquet(data_path / 'test.parquet')
            .group_by('uid')
            .agg(pl.col('item_id'), pl.col('timestamp'))
            .collect(engine='streaming')
        )
    else:
        train_df = (
            pl.scan_parquet(data_path / 'train.parquet')
            .group_by('uid')
            .agg(pl.col('item_id'), pl.col('timestamp'))
            .collect(engine='streaming')
        )

        val_df = (
            pl.scan_parquet(data_path / 'val.parquet')
            .group_by('uid')
            .agg(pl.col('item_id'), pl.col('timestamp'))
            .collect(engine='streaming')
        )

    with open(data_path / 'item_id_to_idx.pkl', 'rb') as f:
        item_id_to_idx = pkl.load(f)

    num_items = len(item_id_to_idx)

    return train_df, val_df, num_items

def read_optuna_study(
    study_name: str,
    storage: str,
    artifact_store: str
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Load an Optuna study, extract the best trial parameters, user attributes,
    and resolve the file path of the best model weights stored as Optuna artifacts.

    The function:
      1. Loads an Optuna study from the given RDB or storage URL.
      2. Looks up the best trial.
      3. Reads the user attribute `best_model_id`.
      4. Uses the provided artifact store directory to resolve the actual
         model weight file path.

    Parameters
    ----------
    study_name : str
        Name of the Optuna study to load.

    storage : str
        Storage URL for Optuna studies
        (e.g., `"sqlite:///optuna.db"` or a PostgreSQL DSN).

    artifact_store : str
        Path to the directory where Optuna's artifact files are stored.

    Returns
    -------
    params : Dict[str, Any]
        Hyperparameters of the best trial.

    user_attrs : Dict[str, Any]
        Additional attributes stored in the best trial, including:
          - `'best_model_id'`: ID referencing the saved model weights.

    weight_path : str
        Filesystem path pointing to the model weights artifact corresponding
        to `best_model_id`.
    """
    study = optuna.load_study(study_name=study_name, storage=storage)

    artifact_store_obj = FileSystemArtifactStore(artifact_store)
    best_trial = study.best_trial

    model_id = best_trial.user_attrs['best_model_id']
    weight_path = os.path.join(artifact_store_obj._base_path, model_id)

    return best_trial.params, best_trial.user_attrs, weight_path
