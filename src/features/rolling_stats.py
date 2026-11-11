"""
Signal Processing: Trailing Rolling Window Statistics.

Extracts trailing rolling statistics (mean, std, min, max) for active sensors
strictly partitioned by unit_nr to prevent cross-engine contamination and lookahead leakage.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def compute_trailing_rolling_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    windows: List[int] = [5, 10, 20],
    group_col: str = "unit_nr",
) -> pd.DataFrame:
    """
    Compute trailing rolling mean, standard deviation, min, and max for active sensors.

    Parameters
    ----------
    df : pd.DataFrame
        Input telemetry containing group_col and sensor_cols.
    sensor_cols : list of str
        Names of sensor columns to compute rolling statistics for.
    windows : list of int, default [5, 10, 20]
        Window sizes in cycles (e.g., 5 cycles, 10 cycles, 20 cycles).
    group_col : str, default 'unit_nr'
        Column name identifying the unique machine/engine unit.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus engineered rolling feature columns.
    """
    df = df.copy()

    # Pre-sort by unit and time to guarantee temporal order
    if "time_cycles" in df.columns:
        df = df.sort_values(by=[group_col, "time_cycles"]).reset_index(drop=True)

    grouped = df.groupby(group_col)[sensor_cols]

    for w in windows:
        # Rolling Mean
        rolling_mean = grouped.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_mean.columns = [f"{col}_roll_mean_{w}" for col in sensor_cols]

        # Rolling Std (fill early 0-variance cycles with 0.0)
        rolling_std = grouped.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
        rolling_std.columns = [f"{col}_roll_std_{w}" for col in sensor_cols]

        # Rolling Min and Max
        rolling_min = grouped.rolling(window=w, min_periods=1).min().reset_index(level=0, drop=True)
        rolling_min.columns = [f"{col}_roll_min_{w}" for col in sensor_cols]

        rolling_max = grouped.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)
        rolling_max.columns = [f"{col}_roll_max_{w}" for col in sensor_cols]

        # Append features
        df = pd.concat([df, rolling_mean, rolling_std, rolling_min, rolling_max], axis=1)

    return df
