"""
Remaining Useful Life (RUL) Target Engineering and Piecewise Linear Clipping.

Computes physical Remaining Useful Life trajectories for C-MAPSS turbofan engine
fleets across both:
1. Run-to-failure training regimes (add_train_rul)
2. Right-censored test regimes with ground-truth back-projection (add_test_rul)

Implements the literature-standard Piecewise Linear Degradation Model (Saxena et al.,
Heimes et al.), preventing ML models from overfitting to steady-state sensor noise
during the initial healthy operational life of the engine.
"""

from typing import Optional
import numpy as np
import pandas as pd


def add_train_rul(
    df: pd.DataFrame,
    max_rul_clip: Optional[int] = 125,
    unit_col: str = "unit_nr",
    time_col: str = "time_cycles",
    rul_col: str = "rul",
    clipped_col: str = "rul_clipped",
) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for run-to-failure training data.

    In the training set, each engine unit operates until complete mechanical failure.
    The true RUL at cycle t for unit i is:
        RUL(i, t) = max(time_cycles_i) - t

    For cycles where RUL exceeds max_rul_clip, RUL is capped to reflect the
    physical reality that wear is unmeasurable during healthy initial operation:
        RUL_clipped(i, t) = min(RUL(i, t), max_rul_clip)

    Parameters
    ----------
    df : pd.DataFrame
        Training telemetry DataFrame containing at least `unit_col` and `time_col`.
    max_rul_clip : int or None, default=125
        Upper clipping ceiling for piecewise degradation. If None or <= 0, no clipping
        is applied (pure linear RUL). Literature consensus recommendation is 125.
    unit_col : str, default='unit_nr'
        Column name identifying individual engine units.
    time_col : str, default='time_cycles'
        Column name representing elapsed operational flight cycles.
    rul_col : str, default='rul'
        Target column name for raw linear RUL.
    clipped_col : str, default='rul_clipped'
        Target column name for clipped piecewise RUL.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with added `rul` and `rul_clipped` columns.
    """
    if df.empty:
        df_out = df.copy()
        df_out[rul_col] = pd.Series(dtype=np.int32)
        df_out[clipped_col] = pd.Series(dtype=np.int32)
        return df_out

    missing_cols = [col for col in [unit_col, time_col] if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns in DataFrame: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    # Vectorized max cycle calculation per unit
    max_cycles = out.groupby(unit_col)[time_col].transform("max")
    raw_rul = (max_cycles - out[time_col]).astype(np.int32)
    out[rul_col] = raw_rul

    if max_rul_clip is not None and max_rul_clip > 0:
        out[clipped_col] = raw_rul.clip(upper=max_rul_clip).astype(np.int32)
    else:
        out[clipped_col] = raw_rul.copy()

    return out


def add_test_rul(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    max_rul_clip: Optional[int] = 125,
    unit_col: str = "unit_nr",
    time_col: str = "time_cycles",
    true_rul_col: str = "true_rul",
    rul_col: str = "rul",
    clipped_col: str = "rul_clipped",
) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for right-censored test data.

    In the test set, telemetry stops at an arbitrary cycle T_last before failure occurs.
    NASA provides the true RUL at cycle T_last in `rul_df`.
    We back-project RUL to all historical cycles t <= T_last:
        RUL(i, t) = True_RUL(i) + (T_last(i) - t)

    We then apply piecewise linear clipping:
        RUL_clipped(i, t) = min(RUL(i, t), max_rul_clip)

    Parameters
    ----------
    test_df : pd.DataFrame
        Test telemetry DataFrame with elapsed flight cycles per engine.
    rul_df : pd.DataFrame
        Ground-truth RUL DataFrame for test engines. Must contain either:
        - Columns ['unit_nr', 'true_rul'], OR
        - A single column of true RUL values (1-indexed matching unit_nr).
    max_rul_clip : int or None, default=125
        Upper clipping ceiling for piecewise degradation. Literature standard is 125.
    unit_col : str, default='unit_nr'
        Column name identifying individual engine units.
    time_col : str, default='time_cycles'
        Column name representing elapsed operational flight cycles.
    true_rul_col : str, default='true_rul'
        Column name in `rul_df` storing ground truth remaining life at final test cycle.
    rul_col : str, default='rul'
        Target column name for raw linear RUL.
    clipped_col : str, default='rul_clipped'
        Target column name for clipped piecewise RUL.

    Returns
    -------
    pd.DataFrame
        A copy of `test_df` with added `rul` and `rul_clipped` columns.
    """
    if test_df.empty:
        df_out = test_df.copy()
        df_out[rul_col] = pd.Series(dtype=np.int32)
        df_out[clipped_col] = pd.Series(dtype=np.int32)
        return df_out

    missing_cols = [col for col in [unit_col, time_col] if col not in test_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns in test_df: {missing_cols}. "
            f"Available columns: {list(test_df.columns)}"
        )

    # Standardize ground truth RUL lookup
    rul_df_clean = rul_df.copy()
    if unit_col not in rul_df_clean.columns:
        if len(rul_df_clean.columns) == 1:
            rul_df_clean.columns = [true_rul_col]
            rul_df_clean[unit_col] = np.arange(1, len(rul_df_clean) + 1, dtype=np.int32)
        else:
            raise KeyError(
                f"Expected '{unit_col}' column in rul_df or a single-column DataFrame. "
                f"Found: {list(rul_df_clean.columns)}"
            )

    if true_rul_col not in rul_df_clean.columns:
        # Fall back to first non-unit_col column if true_rul_col name differs
        candidate_cols = [c for c in rul_df_clean.columns if c != unit_col]
        if candidate_cols:
            true_rul_col = candidate_cols[0]
        else:
            raise KeyError(f"Could not locate true RUL column in rul_df: {list(rul_df_clean.columns)}")

    out = test_df.copy()

    # Determine T_last for each test engine
    t_last_per_unit = out.groupby(unit_col)[time_col].transform("max")

    # Map true RUL at T_last to each row by unit_nr
    true_rul_map = rul_df_clean.set_index(unit_col)[true_rul_col].to_dict()
    mapped_true_rul = out[unit_col].map(true_rul_map)

    if mapped_true_rul.isna().any():
        missing_units = out.loc[mapped_true_rul.isna(), unit_col].unique()
        raise ValueError(f"Ground truth RUL missing for test engine units: {missing_units.tolist()}")

    # Back-project: RUL(t) = True_RUL + (T_last - t)
    raw_rul = (mapped_true_rul + (t_last_per_unit - out[time_col])).astype(np.int32)
    out[rul_col] = raw_rul

    if max_rul_clip is not None and max_rul_clip > 0:
        out[clipped_col] = raw_rul.clip(upper=max_rul_clip).astype(np.int32)
    else:
        out[clipped_col] = raw_rul.copy()

    return out
