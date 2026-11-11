"""
Layer 2 Imbalance-Aware Classifier: XGBoost.

Trains an early-warning failure classifier targeting the binary failure horizon
(failure_in_30_cycles). Implements cost-sensitive positive weighting (scale_pos_weight)
to overcome severe class imbalance and optimizes PR-AUC and ROC-AUC.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
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
import xgboost as xgb


class TurbofanXGBoostClassifier:
    """
    Imbalance-Aware XGBoost Classifier for turbofan failure prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_cols: List[str] = []
        self.scale_pos_weight: float = 1.0

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "failure_in_30_cycles",
        val_df: Optional[pd.DataFrame] = None,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: Optional[int] = 20,
    ) -> "TurbofanXGBoostClassifier":
        """
        Fit XGBoost classifier with imbalance-aware weighting.
        """
        self.feature_cols = list(feature_cols)

        X_train = df[self.feature_cols].fillna(0.0).to_numpy()
        y_train = df[target_col].to_numpy().astype(int)

        # Calculate scale_pos_weight if not explicitly provided
        if scale_pos_weight is None:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            self.scale_pos_weight = float(n_neg / max(n_pos, 1))
        else:
            self.scale_pos_weight = scale_pos_weight

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric="aucpr",
            early_stopping_rounds=early_stopping_rounds if val_df is not None else None,
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
            verbose=False,
        )

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of failure within the warning horizon.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = df[self.feature_cols].fillna(0.0).to_numpy()
        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary failure alarm using decision threshold.
        """
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "failure_in_30_cycles",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        """
        y_true = df[target_col].to_numpy().astype(int)
        y_proba = self.predict_proba(df)
        y_pred = (y_proba >= threshold).astype(int)

        pr_auc = float(average_precision_score(y_true, y_proba))
        roc_auc = float(roc_auc_score(y_true, y_proba))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm,
            "positive_prevalence": float(np.mean(y_true)),
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importance ranking."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return fi_df

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model object to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TurbofanXGBoostClassifier":
        """Load serialized model."""
        return joblib.load(Path(filepath))

