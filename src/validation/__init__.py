"""
Validation and Zero-Leakage Evaluation Module for Turbofan Fleet Telemetry.
"""

from src.validation.group_kfold import (
    get_unit_group_kfold,
    get_fold_metadata,
)
from src.validation.time_series_split import (
    temporal_engine_split,
    expanding_window_engine_split,
)
from src.validation.leakage_checks import (
    assert_no_unit_overlap,
    assert_temporal_monotonicity,
    assert_feature_target_separation,
    assert_no_future_lookahead,
    DEFAULT_TARGET_COLUMNS,
)

__all__ = [
    "get_unit_group_kfold",
    "get_fold_metadata",
    "temporal_engine_split",
    "expanding_window_engine_split",
    "assert_no_unit_overlap",
    "assert_temporal_monotonicity",
    "assert_feature_target_separation",
    "assert_no_future_lookahead",
    "DEFAULT_TARGET_COLUMNS",
]

