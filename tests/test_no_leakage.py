"""
Unit and Integration Tests for Step 6: Zero-Leakage Validation Framework.
"""

import pytest
import numpy as np
import pandas as pd

from src.validation.group_kfold import get_unit_group_kfold, get_fold_metadata
from src.validation.time_series_split import (
    temporal_engine_split,
    expanding_window_engine_split,
)
from src.validation.leakage_checks import (
    assert_no_unit_overlap,
    assert_temporal_monotonicity,
    assert_feature_target_separation,
    assert_no_future_lookahead,
)
from src.data.loader import load_raw_train


@pytest.fixture
def synthetic_fleet_df():
    """
    Creates synthetic fleet data: 10 engines, each with 50-100 cycles.
    """
    records = []
    rng = np.random.RandomState(42)
    for u in range(1, 11):
        n_cycles = rng.randint(40, 80)
        for c in range(1, n_cycles + 1):
            records.append({
                "unit_nr": u,
                "time_cycles": c,
                "sensor_val": 500.0 + 0.1 * c + rng.normal(0, 0.5),
            })
    return pd.DataFrame(records)


# ============================================================================
# Tests for GroupKFold
# ============================================================================

def test_group_kfold_zero_unit_overlap(synthetic_fleet_df):
    """Verify that across all folds, no engine unit ever appears in both train and val."""
    n_splits = 5
    val_units_seen = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        get_unit_group_kfold(synthetic_fleet_df, n_splits=n_splits, seed=42)
    ):
        train_df = synthetic_fleet_df.iloc[train_idx]
        val_df = synthetic_fleet_df.iloc[val_idx]

        # Use assertion function
        assert_no_unit_overlap(train_df, val_df, unit_col="unit_nr")

        train_units = set(train_df["unit_nr"].unique())
        val_units = set(val_df["unit_nr"].unique())

        # Exact intersection must be empty
        assert len(train_units.intersection(val_units)) == 0

        # All units must belong to either train or val
        assert train_units.union(val_units) == set(synthetic_fleet_df["unit_nr"].unique())

        val_units_seen.extend(list(val_units))

    # Every unit must be validated exactly once
    assert sorted(val_units_seen) == sorted(list(synthetic_fleet_df["unit_nr"].unique()))


def test_get_fold_metadata(synthetic_fleet_df):
    """Test fold metadata reporting."""
    meta = get_fold_metadata(synthetic_fleet_df, n_splits=5, seed=42)
    assert len(meta) == 5
    for f in meta:
        assert f["unit_overlap"] == 0
        assert f["train_units_count"] + f["val_units_count"] == 10
        assert f["train_samples"] + f["val_samples"] == len(synthetic_fleet_df)


def test_group_kfold_validation_errors(synthetic_fleet_df):
    """Test validation errors for invalid arguments."""
    with pytest.raises(KeyError, match="not found"):
        list(get_unit_group_kfold(synthetic_fleet_df, unit_col="non_existent"))

    with pytest.raises(ValueError, match="at least 2"):
        list(get_unit_group_kfold(synthetic_fleet_df, n_splits=1))

    with pytest.raises(ValueError, match="Cannot create n_splits"):
        # Only 10 units, cannot split into 20 folds
        list(get_unit_group_kfold(synthetic_fleet_df, n_splits=20))


# ============================================================================
# Tests for Leakage Check Functions
# ============================================================================

def test_assert_no_unit_overlap_catches_leak():
    """Verify that assert_no_unit_overlap raises an error when units leak."""
    train_df = pd.DataFrame({"unit_nr": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
    # Unit 3 intentionally leaks into val_df
    val_df = pd.DataFrame({"unit_nr": [3, 5], "val": [30, 50]})

    with pytest.raises(AssertionError, match="DATA LEAKAGE DETECTED"):
        assert_no_unit_overlap(train_df, val_df, unit_col="unit_nr")


def test_assert_temporal_monotonicity():
    """Verify cycle monotonicity assertions."""
    # Valid strictly increasing
    valid_df = pd.DataFrame({
        "unit_nr": [1, 1, 1, 2, 2],
        "time_cycles": [1, 2, 3, 1, 2],
    })
    assert_temporal_monotonicity(valid_df)

    # Duplicate cycle
    dup_df = pd.DataFrame({
        "unit_nr": [1, 1, 1],
        "time_cycles": [1, 2, 2],
    })
    with pytest.raises(AssertionError, match="Found 1 duplicate cycle"):
        assert_temporal_monotonicity(dup_df)

    # Decreasing cycle (time travel)
    reverse_df = pd.DataFrame({
        "unit_nr": [1, 1, 1],
        "time_cycles": [1, 3, 2],
    })
    with pytest.raises(AssertionError, match="non-increasing cycle transitions"):
        assert_temporal_monotonicity(reverse_df)


def test_assert_feature_target_separation():
    """Verify that target leakage into feature vectors is caught."""
    clean_features = ["T30", "Ps30", "T30_rolling_mean_5", "Ps30_lag_1"]
    assert_feature_target_separation(clean_features)

    # Leaking RUL
    leaked_rul = ["T30", "rul", "Ps30"]
    with pytest.raises(AssertionError, match="TARGET LEAKAGE DETECTED"):
        assert_feature_target_separation(leaked_rul)

    # Leaking failure horizon
    leaked_horizon = ["T30", "failure_in_30_cycles"]
    with pytest.raises(AssertionError, match="TARGET LEAKAGE DETECTED"):
        assert_feature_target_separation(leaked_horizon)


# ============================================================================
# Tests for Temporal Splitters
# ============================================================================

def test_temporal_engine_split(synthetic_fleet_df):
    """Test temporal cutoff splitting per unit."""
    train_df, val_df = temporal_engine_split(
        synthetic_fleet_df, cutoff_ratio=0.7, unit_col="unit_nr", time_col="time_cycles"
    )

    assert len(train_df) + len(val_df) == len(synthetic_fleet_df)

    # For every unit, max cycle in train must be strictly less than min cycle in val
    for u in synthetic_fleet_df["unit_nr"].unique():
        u_train = train_df[train_df["unit_nr"] == u]
        u_val = val_df[val_df["unit_nr"] == u]

        assert not u_train.empty
        assert not u_val.empty
        assert u_train["time_cycles"].max() < u_val["time_cycles"].min()


def test_expanding_window_engine_split(synthetic_fleet_df):
    """Test expanding window forward chaining."""
    splits = list(
        expanding_window_engine_split(
            synthetic_fleet_df,
            min_train_cycles=20,
            forecast_horizon=10,
            step_size=10,
        )
    )

    assert len(splits) > 0
    previous_cutoff = 0
    for train_slice, eval_slice, cutoff in splits:
        assert cutoff > previous_cutoff
        assert train_slice["time_cycles"].max() <= cutoff
        assert eval_slice["time_cycles"].min() > cutoff
        assert eval_slice["time_cycles"].max() <= cutoff + 10
        previous_cutoff = cutoff


# ============================================================================
# Integration Test on Real NASA FD001 Dataset
# ============================================================================

def test_real_fd001_zero_leakage_kfold():
    """Verify 5-fold GroupKFold on the real NASA FD001 100-engine dataset."""
    df_fd001 = load_raw_train("FD001")
    assert df_fd001["unit_nr"].nunique() == 100

    folds = list(get_unit_group_kfold(df_fd001, n_splits=5, seed=42))
    assert len(folds) == 5

    all_val_units = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_df = df_fd001.iloc[train_idx]
        val_df = df_fd001.iloc[val_idx]

        # Pre-flight assertion
        assert_no_unit_overlap(train_df, val_df, unit_col="unit_nr")

        # In FD001 (100 units), 5-fold should allocate exactly 80 train / 20 val units
        assert train_df["unit_nr"].nunique() == 80
        assert val_df["unit_nr"].nunique() == 20

        all_val_units.extend(val_df["unit_nr"].unique().tolist())

    # All 100 engines validated once
    assert sorted(all_val_units) == list(range(1, 101))

