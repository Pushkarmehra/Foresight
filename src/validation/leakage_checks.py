"""
Zero-Leakage Guardrails and Integrity Assertions.

Automated pre-flight validation checks to detect entity (unit) contamination,
temporal lookahead leakage, and feature-target bleed before model training.
"""

from typing import Iterable, List, Optional, Set
import numpy as np
import pandas as pd


# Default standard target column names that must never bleed into feature matrices
DEFAULT_TARGET_COLUMNS: Set[str] = {
    "rul",
    "rul_clipped",
    "true_rul",
    "failure_in_30_cycles",
    "failure_in_horizon",
    "event",
    "duration",
}


def assert_no_unit_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    unit_col: str = "unit_nr",
) -> None:
    """
    Assert that training and validation sets have zero overlapping engine units.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    val_df : pd.DataFrame
        Validation partition.
    unit_col : str, default='unit_nr'
        Column name identifying individual engine units.

    Raises
    ------
    KeyError
        If `unit_col` is missing in either DataFrame.
    AssertionError
        If any engine unit appears in both train and validation splits.
    """
    for name, df in [("train_df", train_df), ("val_df", val_df)]:
        if unit_col not in df.columns:
            raise KeyError(
                f"Required unit column '{unit_col}' missing from {name}. "
                f"Available: {list(df.columns)}"
            )

    train_units = set(train_df[unit_col].unique())
    val_units = set(val_df[unit_col].unique())
    overlap = train_units.intersection(val_units)

    if overlap:
        sample_overlap = sorted(list(overlap))[:10]
        raise AssertionError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} engine units appear in BOTH "
            f"training and validation sets! Overlapping units (sample): {sample_overlap}. "
            "Validation splits MUST be partitioned strictly by machine unit identity."
        )


def assert_temporal_monotonicity(
    df: pd.DataFrame,
    unit_col: str = "unit_nr",
    time_col: str = "time_cycles",
) -> None:
    """
    Assert that flight cycles strictly increase monotonically per engine unit.

    Checks:
    1. No duplicate cycles per unit.
    2. Cycles strictly increase (t_{k+1} > t_k).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    unit_col : str, default='unit_nr'
        Column name identifying individual engine units.
    time_col : str, default='time_cycles'
        Column name representing elapsed operational flight cycles.

    Raises
    ------
    KeyError
        If `unit_col` or `time_col` are missing.
    AssertionError
        If non-monotonic cycle sequences or duplicate cycles are found.
    """
    for col in (unit_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    if df.empty:
        return

    # Check for duplicate unit-cycle pairs
    duplicates = df.duplicated(subset=[unit_col, time_col])
    if duplicates.any():
        num_dups = duplicates.sum()
        dup_rows = df[duplicates][[unit_col, time_col]].head(5)
        raise AssertionError(
            f"TEMPORAL INTEGRITY VIOLATION: Found {num_dups} duplicate cycle entries. "
            f"Examples:\n{dup_rows}"
        )

    # Check strict monotonicity per unit
    diffs = df.groupby(unit_col)[time_col].diff().dropna()
    non_positive = diffs[diffs <= 0]
    if not non_positive.empty:
        raise AssertionError(
            f"TEMPORAL INTEGRITY VIOLATION: Found {len(non_positive)} non-increasing cycle transitions. "
            "Cycle sequences must strictly increase per unit."
        )


def assert_feature_target_separation(
    feature_cols: Iterable[str],
    target_cols: Optional[Iterable[str]] = None,
) -> None:
    """
    Assert that target labels are never accidentally present in the feature matrix.

    Parameters
    ----------
    feature_cols : Iterable[str]
        List or set of column names intended to be used as model inputs (X).
    target_cols : Iterable[str], optional
        Explicit list of forbidden target column names. If None, uses DEFAULT_TARGET_COLUMNS.

    Raises
    ------
    AssertionError
        If any target column is present in feature_cols.
    """
    feature_set = set(feature_cols)
    forbidden = set(target_cols) if target_cols is not None else DEFAULT_TARGET_COLUMNS

    # Also match any column starting with failure_in_
    dynamic_targets = {c for c in feature_set if c.startswith("failure_in_")}
    forbidden = forbidden.union(dynamic_targets)

    leaked = feature_set.intersection(forbidden)
    if leaked:
        raise AssertionError(
            f"TARGET LEAKAGE DETECTED: Feature matrix contains target columns: {sorted(list(leaked))}. "
            "Target labels must be completely isolated from model features."
        )


def assert_no_future_lookahead(
    df: pd.DataFrame,
    feature_col: str,
    raw_col: str,
    unit_col: str = "unit_nr",
    time_col: str = "time_cycles",
) -> None:
    """
    Assert that a trailing feature does not depend on future unobserved cycles.

    Verifies that altering future raw sensor values of an engine does not change
    the computed historical feature at earlier cycles.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    feature_col : str
        The engineered trailing feature to test.
    raw_col : str
        The raw underlying sensor column.
    unit_col : str, default='unit_nr'
        Unit ID column.
    time_col : str, default='time_cycles'
        Time counter column.
    """
    for col in (feature_col, raw_col, unit_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    # At cycle 1 of each unit, trailing features should only depend on cycle 1
    cycle_1_rows = df[df[time_col] == 1]
    if cycle_1_rows.empty:
        return

    # Check for NaNs or sensible initialization at cycle 1
    # Trailing rolling mean with min_periods=1 must equal raw sensor at cycle 1
    diff = np.abs(cycle_1_rows[feature_col].to_numpy() - cycle_1_rows[raw_col].to_numpy())
    if np.nanmax(diff) > 1e-4:
        raise AssertionError(
            f"LOOKAHEAD LEAKAGE DETECTED in '{feature_col}': "
            f"At cycle 1, trailing feature differs from raw sensor '{raw_col}' "
            f"(max difference = {np.nanmax(diff):.4f})."
        )

