"""
Signal Processing: Sensor Degradation Rates & Velocity.

Computes discrete first-order rates of change and acceleration derivatives
capturing the velocity of mechanical degradation across active sensor channels.
"""

from typing import List
import pandas as pd
import numpy as np


def compute_rate_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    steps: List[int] = [3, 5],
    group_col: str = "unit_nr",
) -> pd.DataFrame:
    """
    Compute degradation velocity (rate of change) over given cycle intervals.

    rate_k(s) = (s_t - s_{t-k}) / k

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by [group_col, 'time_cycles'].
    sensor_cols : list of str
        Sensor names to compute velocity for.
    steps : list of int, default [3, 5]
        Interval in cycles over which rate of change is calculated.
    group_col : str, default 'unit_nr'

    Returns
    -------
    pd.DataFrame
        DataFrame with rate-of-change columns appended.
    """
    df = df.copy()
    grouped = df.groupby(group_col)

    rate_dfs = []
    for k in steps:
        # Rate: (current - lag_k) / k
        diff_df = grouped[sensor_cols].diff(periods=k) / float(k)
        diff_df = diff_df.fillna(0.0)
        diff_df.columns = [f"{col}_rate_{k}" for col in sensor_cols]
        rate_dfs.append(diff_df)

    if rate_dfs:
        df = pd.concat([df] + rate_dfs, axis=1)

    return df
