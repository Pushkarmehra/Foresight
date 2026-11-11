"""
Layer 1 Anomaly Detection: Isolation Forest.

Unsupervised anomaly detector fitted strictly on healthy operational cycles (RUL > 100)
to establish the nominal baseline flight envelope of turbofan engines. Produces a
continuous normalized anomaly score in [0.0, 1.0].
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class IsolationForestAnomalyDetector:
    """
    Unsupervised Isolation Forest trained exclusively on healthy flight regimes.

    Attributes
    ----------
    contamination : float
        Expected proportion of outliers in the nominal training population.
    random_state : int
        Seed for reproducibility.
    scaler : RobustScaler
        RobustScaler fitted on healthy feature vectors.
    model : IsolationForest
        Underlying scikit-learn IsolationForest estimator.
    feature_cols : List[str]
        Sensor feature column names used during training.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        n_estimators: int = 150,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_cols: List[str] = []
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        healthy_rul_threshold: int = 100,
        rul_col: str = "rul",
    ) -> "IsolationForestAnomalyDetector":
        """
        Fit IsolationForest strictly on healthy operational cycles.

        Parameters
        ----------
        df : pd.DataFrame
            Processed telemetry containing feature columns and RUL.
        feature_cols : List[str]
            Feature column names to include in anomaly detection.
        healthy_rul_threshold : int, default=100
            RUL threshold above which engine cycles are considered healthy/nominal.
        rul_col : str, default='rul'
            Name of RUL column.

        Returns
        -------
        self
        """
        self.feature_cols = list(feature_cols)

        # Filter for healthy regimes if RUL column is available
        if rul_col in df.columns:
            healthy_mask = df[rul_col] >= healthy_rul_threshold
            train_df = df[healthy_mask]
            if len(train_df) < 50:
                # Fallback to full df if threshold is too strict
                train_df = df
        else:
            train_df = df

        X = train_df[self.feature_cols].fillna(0.0).to_numpy()
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)

        # Calibrate score range using decision function
        raw_scores = -self.model.decision_function(X_scaled)
        self._score_min = float(np.percentile(raw_scores, 1))
        self._score_max = float(np.percentile(raw_scores, 99))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1.0

        return self

    def predict_score(
        self,
        df: pd.DataFrame,
        score_col: str = "anomaly_score",
    ) -> pd.DataFrame:
        """
        Compute normalized anomaly scores in [0.0, 1.0].

        0.0 = completely nominal / healthy
        1.0 = severe operational divergence / degradation anomaly

        Parameters
        ----------
        df : pd.DataFrame
            Telemetry containing feature_cols.
        score_col : str, default='anomaly_score'
            Name of column to store anomaly score.

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame with added `score_col`.
        """
        if not self.feature_cols:
            raise RuntimeError("Model has not been fitted yet.")

        out = df.copy()
        X = out[self.feature_cols].fillna(0.0).to_numpy()
        X_scaled = self.scaler.transform(X)

        # Negative decision function: higher means more anomalous
        raw_scores = -self.model.decision_function(X_scaled)

        # Normalize to [0.0, 1.0]
        norm_scores = (raw_scores - self._score_min) / (self._score_max - self._score_min)
        out[score_col] = np.clip(norm_scores, 0.0, 1.0).astype(np.float32)

        return out

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model and scaler to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IsolationForestAnomalyDetector":
        """Load serialized model from disk."""
        return joblib.load(Path(filepath))

