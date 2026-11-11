"""
Signal Processing: Historical Lag Feature Extraction.

Extracts multi-step historical lag vectors for active telemetry sensors.
"""

from typing import List
import pandas as pd
import numpy as np


def compute_lag_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    lags: List[int] = [1, 3, 5],
    group_col: str = "unit_nr",
) -> pd.DataFrame:
    """
    Compute multi-step lag features for specified sensor columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by [group_col, 'time_cycles'].
    sensor_cols : list of str
        Sensor names to compute lags for.
    lags : list of int, default [1, 3, 5]
        Cycle lag steps to extract.
    group_col : str, default 'unit_nr'

    Returns
    -------
    pd.DataFrame
        DataFrame with lag feature columns appended.
    """
    df = df.copy()
    grouped = df.groupby(group_col)

    lag_dfs = []
    for lag in lags:
        lag_df = grouped[sensor_cols].shift(lag)
        # Fill early cycles where lag does not exist with the first observed cycle
        lag_df = lag_df.bfill()
        lag_df.columns = [f"{col}_lag_{lag}" for col in sensor_cols]
        lag_dfs.append(lag_df)

    if lag_dfs:
        df = pd.concat([df] + lag_dfs, axis=1)

    return df
