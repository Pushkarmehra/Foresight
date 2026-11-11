"""
Unit and Integration Tests for Signal Processing & Feature Engineering.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.rolling_stats import compute_trailing_rolling_features
from src.features.lags import compute_lag_features
from src.features.rates import compute_rate_features


@pytest.fixture
def sample_feature_df():
    records = []
    for u in [1, 2]:
        for c in range(1, 25):
            records.append({
                "unit_nr": u,
                "time_cycles": c,
                "T30": 1500.0 + 2.0 * c,
                "Ps30": 500.0 - 1.0 * c,
            })
    return pd.DataFrame(records)


def test_trailing_rolling_features(sample_feature_df):
    sensors = ["T30", "Ps30"]
    res = compute_trailing_rolling_features(sample_feature_df, sensors, windows=[5, 10])

    assert "T30_roll_mean_5" in res.columns
    assert "T30_roll_std_5" in res.columns
    assert "Ps30_roll_min_10" in res.columns
    assert "Ps30_roll_max_10" in res.columns

    # Zero lookahead check: at cycle 1, rolling mean must equal raw value
    u1_c1 = res[(res["unit_nr"] == 1) & (res["time_cycles"] == 1)]
    assert np.isclose(u1_c1["T30_roll_mean_5"].iloc[0], u1_c1["T30"].iloc[0])


def test_lag_features(sample_feature_df):
    sensors = ["T30", "Ps30"]
    res = compute_lag_features(sample_feature_df, sensors, lags=[1, 3])

    assert "T30_lag_1" in res.columns
    assert "T30_lag_3" in res.columns
    assert "Ps30_lag_1" in res.columns

    # At cycle 5, lag 1 must equal cycle 4 value
    u1_c5 = res[(res["unit_nr"] == 1) & (res["time_cycles"] == 5)]
    u1_c4 = res[(res["unit_nr"] == 1) & (res["time_cycles"] == 4)]
    assert np.isclose(u1_c5["T30_lag_1"].iloc[0], u1_c4["T30"].iloc[0])


def test_rate_features(sample_feature_df):
    sensors = ["T30", "Ps30"]
    res = compute_rate_features(sample_feature_df, sensors, steps=[3])

    assert "T30_rate_3" in res.columns
    assert "Ps30_rate_3" in res.columns

    # T30 slope is +2.0 per cycle, so rate_3 should be (2.0 * 3) / 3 = 2.0
    u1_c10 = res[(res["unit_nr"] == 1) & (res["time_cycles"] == 10)]
    assert np.isclose(u1_c10["T30_rate_3"].iloc[0], 2.0)

