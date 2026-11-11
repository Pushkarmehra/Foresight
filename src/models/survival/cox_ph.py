"""
Layer 3 Survival Analysis: Cox Proportional Hazards Model.

Non-parametric baseline hazard estimation with parametric proportional covariates.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class TurbofanCoxPHModel:
    """
    Cox Proportional Hazards model for semi-parametric hazard ratios.
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.covariate_cols: List[str] = []
        self.coef_: Optional[np.ndarray] = None
        self._mean_x: Optional[np.ndarray] = None
        self._std_x: Optional[np.ndarray] = None

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "time_cycles",
        event_col: str = "event",
        covariate_cols: Optional[List[str]] = None,
    ) -> "TurbofanCoxPHModel":
        if covariate_cols is None:
            covariate_cols = [c for c in ["T30", "Ps30", "BPR", "T50"] if c in df.columns]

        self.covariate_cols = list(covariate_cols)

        # Summarize to unit level if multiple cycles present
        if "unit_nr" in df.columns and df.duplicated(subset=["unit_nr"]).any():
            data = df.groupby("unit_nr").agg({
                duration_col: "max",
                **{c: "last" for c in self.covariate_cols if c in df.columns}
            }).reset_index()
            data[event_col] = 1
        else:
            data = df.copy()
            if event_col not in data.columns:
                data[event_col] = 1

        X = data[self.covariate_cols].fillna(0.0).to_numpy()
        self._mean_x = np.mean(X, axis=0)
        self._std_x = np.std(X, axis=0)
        self._std_x[self._std_x < 1e-6] = 1.0
        X_norm = (X - self._mean_x) / self._std_x

        times = data[duration_col].to_numpy()
        order = np.argsort(times)[::-1]
        X_sorted = X_norm[order]

        def neg_partial_log_likelihood(beta):
            risk_scores = np.exp(np.clip(X_sorted @ beta, -10.0, 10.0))
            cum_risk = np.cumsum(risk_scores)
            log_lik = np.sum((X_sorted @ beta) - np.log(cum_risk))
            return -(log_lik - 0.5 * self.penalizer * np.sum(beta ** 2))

        init_beta = np.zeros(X.shape[1])
        res = minimize(neg_partial_log_likelihood, init_beta, method="L-BFGS-B")
        self.coef_ = res.x
        return self

    def predict_partial_hazard(self, df: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")
        X = df[self.covariate_cols].fillna(0.0).to_numpy()
        X_norm = (X - self._mean_x) / self._std_x
        return np.exp(np.clip(X_norm @ self.coef_, -10.0, 10.0))

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TurbofanCoxPHModel":
        return joblib.load(Path(filepath))

