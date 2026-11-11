"""
Layer 2 Imbalance-Aware Classifier: LightGBM.

High-efficiency gradient boosted decision tree classifier tailored for fleet scale telemetry.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


class TurbofanLightGBMClassifier:
    """
    Imbalance-Aware LightGBM Classifier for turbofan failure prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_cols: List[str] = []
        self.scale_pos_weight: float = 1.0

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "failure_in_30_cycles",
        val_df: Optional[pd.DataFrame] = None,
        scale_pos_weight: Optional[float] = None,
    ) -> "TurbofanLightGBMClassifier":
        """Fit LightGBM model."""
        self.feature_cols = list(feature_cols)

        X_train = df[self.feature_cols].fillna(0.0).to_numpy()
        y_train = df[target_col].to_numpy().astype(int)

        if scale_pos_weight is None:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            self.scale_pos_weight = float(n_neg / max(n_pos, 1))
        else:
            self.scale_pos_weight = scale_pos_weight

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )

        eval_set = None
        if val_df is not None:
            X_val = val_df[self.feature_cols].fillna(0.0).to_numpy()
            y_val = val_df[target_col].to_numpy().astype(int)
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
        )

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of failure."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = df[self.feature_cols].fillna(0.0).to_numpy()
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary failure label."""
        return (self.predict_proba(df) >= threshold).astype(int)

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "failure_in_30_cycles",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        y_true = df[target_col].to_numpy().astype(int)
        y_proba = self.predict_proba(df)
        y_pred = (y_proba >= threshold).astype(int)

        return {
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TurbofanLightGBMClassifier":
        return joblib.load(Path(filepath))

