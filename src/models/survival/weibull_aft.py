"""
Layer 3 Survival Analysis: Weibull Accelerated Failure Time (AFT) Model.

Quantifies Remaining Useful Life (RUL) with probabilistic confidence bounds.
Handles right-censored test trajectories and outputs an 80% Confidence Interval:
    [10th percentile survival, 90th percentile survival]
e.g., "Engine 24 will fail in 34-48 cycles (80% CI)".
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import weibull_min


class ScipyWeibullAFTFitter:
    """
    Parametric Weibull Accelerated Failure Time model with right-censoring support.
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.shape_: float = 1.0  # rho
        self.beta_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.covariate_cols: List[str] = []
        self._mean_x: Optional[np.ndarray] = None
        self._std_x: Optional[np.ndarray] = None

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> float:
        """Negative log-likelihood of Weibull AFT under right-censoring."""
        log_shape = params[0]
        shape = np.exp(np.clip(log_shape, -2.0, 3.0))
        intercept = params[1]
        betas = params[2:]

        # Linear predictor for scale parameter: lambda_i = exp(intercept + X @ beta)
        log_scale = intercept + X @ betas
        log_scale = np.clip(log_scale, 1.0, 10.0)
        scale = np.exp(log_scale)

        # Avoid division by zero
        safe_time = np.maximum(time, 1e-3)
        z = safe_time / scale

        # Log hazard for uncensored events: log(h(t)) = log(shape) - log(scale) + (shape - 1)*log(z)
        log_hazard = np.log(shape) - log_scale + (shape - 1.0) * np.log(np.maximum(z, 1e-6))
        # Cumulative hazard: H(t) = z^shape
        cum_hazard = np.power(np.minimum(z, 100.0), shape)

        ll = np.sum(event * log_hazard - cum_hazard)

        # L2 Regularization penalty
        penalty = 0.5 * self.penalizer * np.sum(betas ** 2)
        return float(-ll + penalty)

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
        covariate_cols: Optional[List[str]] = None,
    ) -> "ScipyWeibullAFTFitter":
        """
        Fit Weibull AFT model on durations, event indicators, and covariates.
        """
        if covariate_cols is None:
            # Default to active degradation sensors
            covariate_cols = [
                c for c in df.columns
                if c in ["T30", "Ps30", "BPR", "T50", "phi", "NRc"]
            ]
            if not covariate_cols:
                covariate_cols = [c for c in df.columns if c not in [duration_col, event_col, "unit_nr", "time_cycles"]][:5]

        self.covariate_cols = list(covariate_cols)

        time = df[duration_col].to_numpy().astype(np.float64)
        event = df[event_col].to_numpy().astype(np.float64) if event_col in df.columns else np.ones_like(time)
        X = df[self.covariate_cols].fillna(0.0).to_numpy().astype(np.float64)

        # Standardize covariates
        self._mean_x = np.mean(X, axis=0)
        self._std_x = np.std(X, axis=0)
        self._std_x[self._std_x < 1e-6] = 1.0
        X_norm = (X - self._mean_x) / self._std_x

        # Initial parameter estimates
        init_log_shape = 0.5  # shape ~ 1.6
        init_intercept = float(np.log(np.maximum(np.median(time), 10.0)))
        init_betas = np.zeros(X.shape[1])
        init_params = np.concatenate([[init_log_shape, init_intercept], init_betas])

        res = minimize(
            self._neg_log_likelihood,
            init_params,
            args=(X_norm, time, event),
            method="L-BFGS-B",
            options={"maxiter": 300},
        )

        self.shape_ = float(np.exp(np.clip(res.x[0], -2.0, 3.0)))
        self.intercept_ = float(res.x[1])
        self.beta_ = res.x[2:]

        return self

    def predict_percentiles(
        self,
        df: pd.DataFrame,
        percentiles: Tuple[float, float, float] = (0.10, 0.50, 0.90),
    ) -> np.ndarray:
        """
        Predict survival times for specified survival probabilities S(t).

        Returns
        -------
        np.ndarray of shape (N, len(percentiles))
        """
        if self.beta_ is None or self._mean_x is None:
            raise RuntimeError("Model has not been fitted.")

        X = df[self.covariate_cols].fillna(0.0).to_numpy().astype(np.float64)
        X_norm = (X - self._mean_x) / self._std_x

        log_scale = self.intercept_ + X_norm @ self.beta_
        scale = np.exp(np.clip(log_scale, 1.0, 10.0))

        out_percentiles = []
        for p in percentiles:
            # S(t) = exp(-(t/scale)^shape) = p  =>  t = scale * (-log(p))^(1/shape)
            factor = np.power(-np.log(np.clip(p, 1e-6, 1.0 - 1e-6)), 1.0 / self.shape_)
            out_percentiles.append(scale * factor)

        return np.column_stack(out_percentiles)


class TurbofanWeibullSurvivalModel:
    """
    High-level Weibull Survival Engine providing Remaining Useful Life
    estimates with 80% Confidence Intervals.
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.fitter: Optional[ScipyWeibullAFTFitter] = None
        self.covariate_cols: List[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        covariate_cols: Optional[List[str]] = None,
        duration_col: str = "time_cycles",
        event_col: str = "event",
    ) -> "TurbofanWeibullSurvivalModel":
        """
        Fit Weibull AFT on run-to-failure or right-censored engine trajectories.
        """
        data = df.copy()

        # Build terminal duration summary per engine if cycle-level data is provided
        if "unit_nr" in data.columns and data.duplicated(subset=["unit_nr"]).any():
            # Extract final cycle and failure indicator per unit
            summary = data.groupby("unit_nr").agg({
                duration_col: "max",
                **{c: "last" for c in (covariate_cols or ["T30", "Ps30", "BPR"]) if c in data.columns}
            }).reset_index()
            # If RUL is available, units with RUL == 0 are complete failure events (event=1)
            if "rul" in data.columns:
                last_rul = data.groupby("unit_nr")["rul"].min().reset_index()
                summary[event_col] = (last_rul["rul"] <= 1).astype(int)
            else:
                summary[event_col] = 1
            data = summary

        if event_col not in data.columns:
            data[event_col] = 1

        self.fitter = ScipyWeibullAFTFitter(penalizer=self.penalizer)
        self.fitter.fit(
            data,
            duration_col=duration_col,
            event_col=event_col,
            covariate_cols=covariate_cols,
        )
        self.covariate_cols = self.fitter.covariate_cols
        return self

    def predict_intervals(
        self,
        df: pd.DataFrame,
        confidence_level: float = 0.80,
        time_col: str = "time_cycles",
    ) -> pd.DataFrame:
        """
        Predict survival times and Remaining Useful Life with confidence intervals.
        """
        if self.fitter is None:
            raise RuntimeError("Model has not been fitted.")

        alpha = 1.0 - confidence_level
        p_lower = 1.0 - (alpha / 2.0)  # 0.90 survival prob -> lower cycle bound
        p_median = 0.50
        p_upper = alpha / 2.0          # 0.10 survival prob -> upper cycle bound

        # Predict total lifespan quantiles
        lifespans = self.fitter.predict_percentiles(df, percentiles=(p_lower, p_median, p_upper))

        current_time = df[time_col].to_numpy() if time_col in df.columns else np.zeros(len(df))

        # Remaining useful life = Total lifespan - current time
        rul_lower = np.maximum(0, np.round(lifespans[:, 0] - current_time)).astype(int)
        rul_median = np.maximum(0, np.round(lifespans[:, 1] - current_time)).astype(int)
        rul_upper = np.maximum(rul_lower + 1, np.round(lifespans[:, 2] - current_time)).astype(int)

        out = df.copy()
        out["rul_pred_lower_80ci"] = rul_lower
        out["rul_pred_median"] = rul_median
        out["rul_pred_upper_80ci"] = rul_upper

        # Human-readable interval string
        out["survival_ci_display"] = [
            f"Fails in {low}–{high} cycles (80% CI, median: {med})"
            for low, high, med in zip(rul_lower, rul_upper, rul_median)
        ]

        return out

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TurbofanWeibullSurvivalModel":
        return joblib.load(Path(filepath))

