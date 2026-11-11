"""
Group-by-Unit Cross-Validation Generator for Industrial Fleet Telemetry.

Guarantees zero data leakage across train/validation splits by partitioning strictly
on machine identity (`unit_nr`). Ensures that no individual engine appears in both
training and validation sets in any fold.
"""

from typing import Generator, List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_unit_group_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    unit_col: str = "unit_nr",
    seed: Optional[int] = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate train and validation indices partitioned strictly by engine unit identity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing telemetry and engine unit identifiers.
    n_splits : int, default=5
        Number of cross-validation folds. Must be <= number of unique units.
    unit_col : str, default='unit_nr'
        Column name identifying individual engine units.
    seed : int or None, default=42
        Random seed for shuffling unit groups prior to partitioning.
        If None, preserves the original group order.

    Yields
    ------
    train_idx : np.ndarray
        Integer positional row indices for training fold.
    val_idx : np.ndarray
        Integer positional row indices for validation fold.

    Raises
    ------
    KeyError
        If `unit_col` is not present in `df`.
    ValueError
        If `n_splits` is greater than the number of unique units.
    """
    if unit_col not in df.columns:
        raise KeyError(
            f"Unit column '{unit_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    unique_units = df[unit_col].unique()
    n_units = len(unique_units)

    if n_splits <= 1:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    if n_splits > n_units:
        raise ValueError(
            f"Cannot create n_splits={n_splits} with only {n_units} unique units."
        )

    # Deterministically shuffle groups if seed is provided
    if seed is not None:
        rng = np.random.RandomState(seed)
        shuffled_units = unique_units.copy()
        rng.shuffle(shuffled_units)
        unit_to_group = {u: i for i, u in enumerate(shuffled_units)}
        groups = df[unit_col].map(unit_to_group).to_numpy()
    else:
        groups = df[unit_col].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in gkf.split(df, groups=groups):
        yield train_idx, val_idx


def get_fold_metadata(
    df: pd.DataFrame,
    n_splits: int = 5,
    unit_col: str = "unit_nr",
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """
    Extract summary metadata for each fold (unit counts, sample sizes, unit IDs).

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with metadata for each fold.
    """
    metadata = []
    splits = list(get_unit_group_kfold(df, n_splits=n_splits, unit_col=unit_col, seed=seed))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_units = set(df.iloc[train_idx][unit_col].unique())
        val_units = set(df.iloc[val_idx][unit_col].unique())

        metadata.append({
            "fold": fold_idx + 1,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_units_count": len(train_units),
            "val_units_count": len(val_units),
            "train_units": sorted(list(train_units)),
            "val_units": sorted(list(val_units)),
            "unit_overlap": len(train_units.intersection(val_units)),
        })

    return metadata

