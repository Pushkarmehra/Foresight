"""
Temporal Lifecycle Validation Splitters.

Provides temporal splitters designed for run-to-failure engine trajectories.
Evaluates model capacity to forecast future degradation using only early historical
cycles of an engine, preventing temporal lookahead leakage.
"""

from typing import Generator, Tuple, Optional
import numpy as np
import pandas as pd


def temporal_engine_split(
    df: pd.DataFrame,
    cutoff_ratio: float = 0.7,
    unit_col: str = "unit_nr",
    time_col: str = "time_cycles",
    min_train_cycles: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each engine trajectory temporally into an early history (train) and future (val).

    For each engine `i`:
        cutoff_i = max(min_train_cycles, floor(max(time_cycles_i) * cutoff_ratio))
        Train: time_cycles <= cutoff_i
        Val  : time_cycles > cutoff_i

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing engine units and temporal flight cycle counters.
    cutoff_ratio : float, default=0.7
        Fraction of each engine's lifespan to include in the training split.
        Must be strictly between 0.0 and 1.0.
    unit_col : str, default='unit_nr'
        Column name identifying individual engine units.
    time_col : str, default='time_cycles'
        Column name representing elapsed operational flight cycles.
    min_train_cycles : int, default=5
        Minimum number of early cycles guaranteed in the training split per unit.

    Returns
    -------
    train_df : pd.DataFrame
        Subset of df containing cycles up to the cutoff for each unit.
    val_df : pd.DataFrame
        Subset of df containing cycles strictly after the cutoff for each unit.

    Raises
    ------
    ValueError
        If cutoff_ratio is not in (0.0, 1.0).
    KeyError
        If unit_col or time_col are not present in df.
    """
    if not 0.0 < cutoff_ratio < 1.0:
        raise ValueError(
            f"cutoff_ratio must be strictly between 0.0 and 1.0, got {cutoff_ratio}"
        )

    for col in (unit_col, time_col):
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    if df.empty:
        return df.copy(), df.copy()

    # Calculate max cycle per unit
    max_cycles = df.groupby(unit_col)[time_col].transform("max")
    raw_cutoffs = np.floor(max_cycles * cutoff_ratio).astype(np.int32)
    # Guarantee minimum training history while not exceeding max_cycles
    effective_cutoffs = np.clip(raw_cutoffs, min_train_cycles, max_cycles - 1)

    train_mask = df[time_col] <= effective_cutoffs
    val_mask = df[time_col] > effective_cutoffs

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)

    return train_df, val_df


def expanding_window_engine_split(
    df: pd.DataFrame,
    min_train_cycles: int = 30,
    forecast_horizon: int = 30,
    step_size: int = 20,
    unit_col: str = "unit_nr",
    time_col: str = "time_cycles",
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
    """
    Generate temporal expanding-window forward-chaining splits across engine lifecycles.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame.
    min_train_cycles : int, default=30
        Initial history window required before first evaluation.
    forecast_horizon : int, default=30
        Number of forward cycles into the future to evaluate.
    step_size : int, default=20
        Number of cycles to advance the cutoff in each iteration.
    unit_col : str, default='unit_nr'
        Unit ID column.
    time_col : str, default='time_cycles'
        Cycle counter column.

    Yields
    ------
    train_slice : pd.DataFrame
        Cycles up to current cutoff.
    eval_slice : pd.DataFrame
        Cycles in (cutoff, cutoff + forecast_horizon].
    cutoff : int
        The current cutoff cycle used for the split.
    """
    max_life = int(df[time_col].max())

    current_cutoff = min_train_cycles
    while current_cutoff + forecast_horizon <= max_life:
        train_mask = df[time_col] <= current_cutoff
        eval_mask = (df[time_col] > current_cutoff) & (
            df[time_col] <= current_cutoff + forecast_horizon
        )

        train_slice = df[train_mask].copy().reset_index(drop=True)
        eval_slice = df[eval_mask].copy().reset_index(drop=True)

        if not train_slice.empty and not eval_slice.empty:
            yield train_slice, eval_slice, current_cutoff

        current_cutoff += step_size

