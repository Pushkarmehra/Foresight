"""
FastAPI Routes for Turbofan Telemetry Serving.
"""

from pathlib import Path
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.models import SensorSnapshot, PredictionResponse, ExplainResponse
from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from src.models.classifier.xgboost_model import TurbofanXGBoostClassifier
from src.models.survival.weibull_aft import TurbofanWeibullSurvivalModel
from src.explainability.explain_prediction import explain_engine_risk
from src.health_score import calculate_health_score, get_health_tier

router = APIRouter()
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Global lazy-loaded model cache
_MODELS_CACHE: Dict[str, Any] = {}


def get_cached_models():
    """Load or retrieve cached model instances."""
    global _MODELS_CACHE
    if "loaded" not in _MODELS_CACHE:
        try:
            iso_path = MODELS_DIR / "isolation_forest.joblib"
            xgb_path = MODELS_DIR / "xgboost_model.joblib"
            surv_path = MODELS_DIR / "weibull_survival.joblib"

            if iso_path.exists() and xgb_path.exists() and surv_path.exists():
                _MODELS_CACHE["iso"] = IsolationForestAnomalyDetector.load(iso_path)
                _MODELS_CACHE["xgb"] = TurbofanXGBoostClassifier.load(xgb_path)
                _MODELS_CACHE["surv"] = TurbofanWeibullSurvivalModel.load(surv_path)
                _MODELS_CACHE["loaded"] = True
            else:
                _MODELS_CACHE["loaded"] = False
        except Exception as e:
            _MODELS_CACHE["loaded"] = False
            _MODELS_CACHE["error"] = str(e)
    return _MODELS_CACHE


def prepare_feature_row(snapshot_dict: dict, feature_cols: list) -> pd.DataFrame:
    """Efficiently construct complete feature DataFrame from snapshot dict."""
    full_dict = dict(snapshot_dict)
    for col in feature_cols:
        if col not in full_dict:
            base_col = col.split("_")[0]
            full_dict[col] = full_dict.get(base_col, 0.0)
    return pd.DataFrame([full_dict])


@router.get("/health")
def api_health():
    """Check microservice liveness and model readiness."""
    cache = get_cached_models()
    return {
        "status": "healthy",
        "service": "Foresight Turbofan Predictive Maintenance API",
        "models_loaded": cache.get("loaded", False),
        "available_endpoints": ["/health", "/predict", "/explain"],
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_engine(snapshot: SensorSnapshot):
    """
    Generate multi-layered predictive maintenance risk scoring.
    """
    cache = get_cached_models()
    if not cache.get("loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Trained models are not yet serialized in data/models/. Run `python src/pipeline.py` first."
        )

    iso_model: IsolationForestAnomalyDetector = cache["iso"]
    xgb_model: TurbofanXGBoostClassifier = cache["xgb"]
    surv_model: TurbofanWeibullSurvivalModel = cache["surv"]

    row_dict = snapshot.model_dump()
    df = prepare_feature_row(row_dict, xgb_model.feature_cols)

    # 1. Anomaly score
    df_scored = iso_model.predict_score(df)
    anomaly_score = float(df_scored["anomaly_score"].iloc[0])

    # 2. Failure probability
    failure_prob = float(xgb_model.predict_proba(df)[0])

    # 3. Survival CI
    surv_df = surv_model.predict_intervals(df, confidence_level=0.80)
    rul_med = int(surv_df["rul_pred_median"].iloc[0])
    rul_low = int(surv_df["rul_pred_lower_80ci"].iloc[0])
    rul_high = int(surv_df["rul_pred_upper_80ci"].iloc[0])
    surv_text = str(surv_df["survival_ci_display"].iloc[0])

    # 4. Health Score
    health_val = calculate_health_score(failure_prob, anomaly_score, float(rul_med))
    tier, _, hex_color = get_health_tier(health_val)

    return PredictionResponse(
        unit_nr=snapshot.unit_nr,
        time_cycles=snapshot.time_cycles,
        anomaly_score=round(anomaly_score, 4),
        failure_probability=round(failure_prob, 4),
        predicted_rul_median=rul_med,
        rul_80ci_lower=rul_low,
        rul_80ci_upper=rul_high,
        survival_display=surv_text,
        health_score=round(health_val, 1),
        health_tier=tier,
        health_color=hex_color,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain_engine(snapshot: SensorSnapshot):
    """
    Generate aerospace diagnostics identifying degrading components.
    """
    cache = get_cached_models()
    if not cache.get("loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Trained models are not yet serialized. Run `python src/pipeline.py` first."
        )

    xgb_model: TurbofanXGBoostClassifier = cache["xgb"]
    row_dict = snapshot.model_dump()
    df = prepare_feature_row(row_dict, xgb_model.feature_cols)

    diag = explain_engine_risk(xgb_model, df, top_k=3)

    return ExplainResponse(
        unit_nr=snapshot.unit_nr,
        time_cycles=snapshot.time_cycles,
        failure_probability=round(diag["failure_probability"], 4),
        risk_level=diag["risk_level"],
        affected_subsystems=diag["affected_subsystems"],
        top_contributors=diag["top_contributors"],
        diagnostic_summary=diag["diagnostic_summary"],
    )

