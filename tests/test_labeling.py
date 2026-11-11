"""
Unit and Integration Tests for Step 5: Target Labeling & Embargo Windows.
"""

import pytest
import pandas as pd
import numpy as np

from src.labeling.rul import add_train_rul, add_test_rul
from src.labeling.failure_horizon import add_failure_horizon, get_horizon_summary
from src.data.loader import load_dataset_bundle


@pytest.fixture
def synthetic_train_df():
    """
    Creates synthetic train data for 2 engine units:
    Unit 1: 50 cycles (max_cycle = 50)
    Unit 2: 150 cycles (max_cycle = 150)
    """
    u1 = pd.DataFrame({"unit_nr": 1, "time_cycles": np.arange(1, 51)})
    u2 = pd.DataFrame({"unit_nr": 2, "time_cycles": np.arange(1, 151)})
    return pd.concat([u1, u2], ignore_index=True)


@pytest.fixture
def synthetic_test_df():
    """
    Creates synthetic test data for 2 engines:
    Unit 1: stops at cycle 30
    Unit 2: stops at cycle 100
    """
    u1 = pd.DataFrame({"unit_nr": 1, "time_cycles": np.arange(1, 31)})
    u2 = pd.DataFrame({"unit_nr": 2, "time_cycles": np.arange(1, 101)})
    return pd.concat([u1, u2], ignore_index=True)


@pytest.fixture
def synthetic_rul_df():
    """
    Ground truth RUL at the stopping cycle:
    Unit 1 (stops at cycle 30): has 112 cycles remaining
    Unit 2 (stops at cycle 100): has 25 cycles remaining
    """
    return pd.DataFrame({"unit_nr": [1, 2], "true_rul": [112, 25]})


# ============================================================================
# Tests for add_train_rul
# ============================================================================

def test_add_train_rul_calculation(synthetic_train_df):
    labeled = add_train_rul(synthetic_train_df, max_rul_clip=125)

    assert "rul" in labeled.columns
    assert "rul_clipped" in labeled.columns

    # Unit 1 (50 cycles): at cycle 50, RUL should be 0; at cycle 1, RUL should be 49
    u1 = labeled[labeled["unit_nr"] == 1]
    assert u1.loc[u1["time_cycles"] == 50, "rul"].iloc[0] == 0
    assert u1.loc[u1["time_cycles"] == 1, "rul"].iloc[0] == 49
    assert u1.loc[u1["time_cycles"] == 1, "rul_clipped"].iloc[0] == 49

    # Unit 2 (150 cycles): at cycle 1, raw RUL = 149, clipped RUL = 125
    u2 = labeled[labeled["unit_nr"] == 2]
    assert u2.loc[u2["time_cycles"] == 1, "rul"].iloc[0] == 149
    assert u2.loc[u2["time_cycles"] == 1, "rul_clipped"].iloc[0] == 125

    # At cycle 150, raw RUL = 0, clipped RUL = 0
    assert u2.loc[u2["time_cycles"] == 150, "rul"].iloc[0] == 0
    assert u2.loc[u2["time_cycles"] == 150, "rul_clipped"].iloc[0] == 0

    # At cycle 25: raw RUL = 125, clipped RUL = 125
    assert u2.loc[u2["time_cycles"] == 25, "rul"].iloc[0] == 125
    assert u2.loc[u2["time_cycles"] == 25, "rul_clipped"].iloc[0] == 125

    # At cycle 26: raw RUL = 124, clipped RUL = 124
    assert u2.loc[u2["time_cycles"] == 26, "rul"].iloc[0] == 124
    assert u2.loc[u2["time_cycles"] == 26, "rul_clipped"].iloc[0] == 124


def test_add_train_rul_no_clip(synthetic_train_df):
    labeled = add_train_rul(synthetic_train_df, max_rul_clip=None)
    u2 = labeled[labeled["unit_nr"] == 2]
    assert u2.loc[u2["time_cycles"] == 1, "rul_clipped"].iloc[0] == 149


def test_add_train_rul_missing_cols():
    df = pd.DataFrame({"cycle": [1, 2]})
    with pytest.raises(KeyError, match="Missing required columns"):
        add_train_rul(df)


def test_add_train_rul_empty():
    df = pd.DataFrame(columns=["unit_nr", "time_cycles"])
    res = add_train_rul(df)
    assert "rul" in res.columns
    assert "rul_clipped" in res.columns
    assert len(res) == 0


# ============================================================================
# Tests for add_test_rul
# ============================================================================

def test_add_test_rul_back_projection(synthetic_test_df, synthetic_rul_df):
    labeled = add_test_rul(synthetic_test_df, synthetic_rul_df, max_rul_clip=125)

    assert "rul" in labeled.columns
    assert "rul_clipped" in labeled.columns

    # Unit 1 stops at cycle 30 with true_rul = 112
    # At cycle 30: RUL = 112 + (30 - 30) = 112
    # At cycle 1: RUL = 112 + (30 - 1) = 141; clipped = 125
    u1 = labeled[labeled["unit_nr"] == 1]
    assert u1.loc[u1["time_cycles"] == 30, "rul"].iloc[0] == 112
    assert u1.loc[u1["time_cycles"] == 30, "rul_clipped"].iloc[0] == 112
    assert u1.loc[u1["time_cycles"] == 1, "rul"].iloc[0] == 141
    assert u1.loc[u1["time_cycles"] == 1, "rul_clipped"].iloc[0] == 125

    # Unit 2 stops at cycle 100 with true_rul = 25
    # At cycle 100: RUL = 25 + (100 - 100) = 25
    # At cycle 90: RUL = 25 + (100 - 90) = 35
    u2 = labeled[labeled["unit_nr"] == 2]
    assert u2.loc[u2["time_cycles"] == 100, "rul"].iloc[0] == 25
    assert u2.loc[u2["time_cycles"] == 90, "rul"].iloc[0] == 35


def test_add_test_rul_single_column_rul_df(synthetic_test_df):
    # Single column RUL DataFrame (1-indexed implicitly)
    rul_df_single = pd.DataFrame({"true_rul": [112, 25]})
    labeled = add_test_rul(synthetic_test_df, rul_df_single)
    assert labeled.loc[labeled["unit_nr"] == 1, "rul"].iloc[-1] == 112
    assert labeled.loc[labeled["unit_nr"] == 2, "rul"].iloc[-1] == 25


def test_add_test_rul_missing_unit_ground_truth(synthetic_test_df):
    # Only provides ground truth for unit 1, not unit 2
    incomplete_rul_df = pd.DataFrame({"unit_nr": [1], "true_rul": [112]})
    with pytest.raises(ValueError, match="Ground truth RUL missing for test engine units"):
        add_test_rul(synthetic_test_df, incomplete_rul_df)


# ============================================================================
# Tests for add_failure_horizon & embargo
# ============================================================================

def test_add_failure_horizon_binary_labels(synthetic_train_df):
    labeled = add_train_rul(synthetic_train_df, max_rul_clip=125)
    horizon_df = add_failure_horizon(labeled, horizon=30)

    target_col = "failure_in_30_cycles"
    assert target_col in horizon_df.columns

    # Where RUL <= 30, target should be 1; where RUL > 30, target should be 0
    pos_mask = horizon_df["rul"] <= 30
    assert (horizon_df.loc[pos_mask, target_col] == 1).all()
    assert (horizon_df.loc[~pos_mask, target_col] == 0).all()


def test_add_failure_horizon_embargo_window(synthetic_train_df):
    labeled = add_train_rul(synthetic_train_df, max_rul_clip=125)
    embargo_df = add_failure_horizon(labeled, horizon=30, embargo_buffer=3)

    # All rows with RUL <= 3 should have been dropped
    assert (embargo_df["rul"] > 3).all()
    assert embargo_df["rul"].min() == 4


def test_failure_horizon_validation_errors(synthetic_train_df):
    labeled = add_train_rul(synthetic_train_df)

    with pytest.raises(ValueError, match="Forecast horizon must be a positive integer"):
        add_failure_horizon(labeled, horizon=0)

    with pytest.raises(ValueError, match="Embargo buffer cannot be negative"):
        add_failure_horizon(labeled, horizon=30, embargo_buffer=-1)

    with pytest.raises(ValueError, match="Embargo buffer .* must be strictly less than"):
        add_failure_horizon(labeled, horizon=30, embargo_buffer=30)


def test_horizon_summary():
    df = pd.DataFrame({"target": [0, 0, 0, 1, 1]})
    summary = get_horizon_summary(df, "target")
    assert summary["total"] == 5
    assert summary["positives"] == 2
    assert summary["negatives"] == 3
    assert summary["positive_pct"] == 40.0


# ============================================================================
# End-to-End Integration with Real C-MAPSS FD001 Data
# ============================================================================

def test_real_fd001_labeling_pipeline():
    train_df, test_df, rul_df = load_dataset_bundle("FD001")

    # 1. Train RUL
    train_labeled = add_train_rul(train_df, max_rul_clip=125)
    assert train_labeled["rul"].min() == 0
    assert train_labeled["rul_clipped"].max() == 125

    # 2. Test RUL
    test_labeled = add_test_rul(test_df, rul_df, max_rul_clip=125)
    assert test_labeled["rul_clipped"].max() <= 125
    # For every test engine, at its max cycle, RUL must equal ground truth in rul_df
    for unit in range(1, 101):
        actual_test_max = test_labeled[test_labeled["unit_nr"] == unit]["time_cycles"].max()
        last_row = test_labeled[(test_labeled["unit_nr"] == unit) & (test_labeled["time_cycles"] == actual_test_max)]
        expected_true_rul = rul_df.loc[rul_df["unit_nr"] == unit, "true_rul"].iloc[0]
        assert last_row["rul"].iloc[0] == expected_true_rul

    # 3. Horizon labeling
    train_horizon = add_failure_horizon(train_labeled, horizon=30)
    summary = get_horizon_summary(train_horizon, "failure_in_30_cycles")
    assert 10.0 <= summary["positive_pct"] <= 25.0  # Real C-MAPSS FD001 positive rate is ~15%
