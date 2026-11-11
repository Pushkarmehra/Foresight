"""
FastAPI Microservice Integration Tests (In-Process Handler Validation).
"""

import pytest
from app.api.models import SensorSnapshot, PredictionResponse, ExplainResponse
from app.api.routes import api_health, predict_engine, explain_engine


def test_api_health_endpoint():
    data = api_health()
    assert data["status"] == "healthy"
    assert "available_endpoints" in data
    assert data["models_loaded"] is True


def test_api_predict_endpoint():
    snapshot = SensorSnapshot(
        unit_nr=1,
        time_cycles=120,
        setting_1=0.0,
        setting_2=0.0,
        setting_3=100.0,
        T24=642.5,
        T30=1588.0,
        T50=1405.0,
        P30=553.8,
        Nf=2388.0,
        Nc=9050.0,
        Ps30=47.2,
        phi=521.8,
        NRf=2388.0,
        NRc=8130.0,
        BPR=8.45,
        htBleed=392.0,
        W31=38.8,
        W32=23.3,
    )
    res = predict_engine(snapshot)
    assert isinstance(res, PredictionResponse)
    assert res.unit_nr == 1
    assert 0.0 <= res.failure_probability <= 1.0
    assert 0.0 <= res.anomaly_score <= 1.0
    assert 0.0 <= res.health_score <= 100.0
    assert res.health_tier in [
        "Healthy (Normal)",
        "Warning (Maintenance Scheduled)",
        "Critical (Immediate Overhaul)",
    ]


def test_api_explain_endpoint():
    snapshot = SensorSnapshot(
        unit_nr=1,
        time_cycles=120,
        T30=1605.0,
        Ps30=45.1,
        BPR=8.55,
    )
    res = explain_engine(snapshot)
    assert isinstance(res, ExplainResponse)
    assert res.unit_nr == 1
    assert res.risk_level in ["CRITICAL", "ELEVATED", "NOMINAL"]
    assert len(res.top_contributors) > 0
    assert "Fleet Advisory" in res.diagnostic_summary
