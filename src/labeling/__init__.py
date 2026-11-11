"""
Target Labeling, Piecewise Degradation Modeling, and Embargo Windows.
"""

from src.labeling.rul import add_train_rul, add_test_rul
from src.labeling.failure_horizon import add_failure_horizon, get_horizon_summary

__all__ = [
    "add_train_rul",
    "add_test_rul",
    "add_failure_horizon",
    "get_horizon_summary",
]
