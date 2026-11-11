"""
Failure Horizon Binary Classification Target Engineering and Embargo Windows.

Transforms continuous Remaining Useful Life (RUL) trajectories into actionable
early-warning alarms for fleet maintenance dispatchers:
    failure_in_W_cycles = 1 if RUL <= W else 0

Also provides Embargo Windows (Blackout Buffers) to exclude extreme boundary
distortions in the final cycles immediately preceding catastrophic failure.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


def add_failure_horizon(
    df: pd.DataFrame,
    horizon: int = 30,
    target_col_name: Optional[str] = None,
    embargo_buffer: int = 0,
    rul_col: str = "rul",
) -> pd.DataFrame:
    """
    Construct a binary failure alarm target within a specified forecast horizon.

    Operational dispatchers cannot overhaul an engine on the exact cycle of failure.
    Instead, they require an operational lead time `horizon` (typically 30 cycles,
    representing 1-2 weeks of commercial flights) to stage spare engines, book hangar
    space, and dispatch technicians.

    Optionally applies an Embargo Window (`embargo_buffer`), filtering out the final
    few cycles (e.g. RUL <= 3) where extreme non-linear boundary shocks (rapid thermal
    runaway, sensor clipping) introduce noisy anomalies that distort classifier gradients.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing continuous RUL targets (from `add_train_rul` or `add_test_rul`).
    horizon : int, default=30
        Early warning window in flight cycles. Samples with RUL <= horizon are labeled 1.
    target_col_name : str or None, default=None
        Custom name for the binary target column. If None, defaults to `failure_in_{horizon}_cycles`.
    embargo_buffer : int, default=0
        Number of terminal cycles to blackout/filter out. If > 0, rows with
        `df[rul_col] <= embargo_buffer` are dropped from the returned DataFrame.
    rul_col : str, default='rul'
        Column name to evaluate against the horizon. Can be 'rul' or 'rul_clipped'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the new binary target column, with rows inside the
        embargo window dropped if `embargo_buffer > 0`.
    """
    if horizon <= 0:
        raise ValueError(f"Forecast horizon must be a positive integer > 0, got {horizon}")

    if embargo_buffer < 0:
        raise ValueError(f"Embargo buffer cannot be negative, got {embargo_buffer}")

    if embargo_buffer >= horizon:
        raise ValueError(
            f"Embargo buffer ({embargo_buffer}) must be strictly less than "
            f"the forecast horizon ({horizon})."
        )

    if df.empty:
        col_name = target_col_name if target_col_name else f"failure_in_{horizon}_cycles"
        df_out = df.copy()
        df_out[col_name] = pd.Series(dtype=np.int32)
        return df_out

    if rul_col not in df.columns:
        # Fallback check for rul_clipped
        if "rul_clipped" in df.columns:
            rul_col = "rul_clipped"
        else:
            raise KeyError(
                f"Required RUL column '{rul_col}' not found in DataFrame columns: {list(df.columns)}. "
                "Ensure `add_train_rul` or `add_test_rul` is called before `add_failure_horizon`."
            )

    out = df.copy()
    target_col = target_col_name if target_col_name else f"failure_in_{horizon}_cycles"

    # Vectorized binary labeling
    out[target_col] = (out[rul_col] <= horizon).astype(np.int32)

    # Apply embargo buffer blackout
    if embargo_buffer > 0:
        initial_len = len(out)
        out = out[out[rul_col] > embargo_buffer].reset_index(drop=True)
        dropped_rows = initial_len - len(out)
        # Note: Embargo drops terminal cycles where machine is already failing/broken

    return out


def get_horizon_summary(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Compute distribution and class imbalance statistics for a failure horizon target.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled DataFrame.
    target_col : str
        Target column name (e.g., 'failure_in_30_cycles').

    Returns
    -------
    dict
        Dictionary containing total samples, positive count, negative count,
        and positive prevalence ratio percentage.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    total = len(df)
    if total == 0:
        return {"total": 0, "positives": 0, "negatives": 0, "positive_pct": 0.0}

    pos = int((df[target_col] == 1).sum())
    neg = total - pos
    pos_pct = (pos / total) * 100.0

    return {
        "total": total,
        "positives": pos,
        "negatives": neg,
        "positive_pct": round(pos_pct, 2),
    }
