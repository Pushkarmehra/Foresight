"""
Unit and Integration Tests for Layers 1, 2, 3 Models, Explainability, and Health Scoring.
"""

import pytest
import numpy as np
import pandas as pd

from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from src.models.anomaly.autoencoder import PyTorchAutoencoderDetector
from src.models.classifier.xgboost_model import TurbofanXGBoostClassifier
from src.models.survival.weibull_aft import TurbofanWeibullSurvivalModel
from src.explainability.explain_prediction import explain_engine_risk
from src.health_score import calculate_health_score, compute_health_scores, get_health_tier


@pytest.fixture
def mock_telemetry_df():
    """Generates mock telemetry with degraded signatures."""
    rng = np.random.RandomState(42)
    records = []
    for u in range(1, 11):
        for c in range(1, 51):
            rul = 50 - c
            records.append({
                "unit_nr": u,
                "time_cycles": c,
                "rul": rul,
                "rul_clipped": min(rul, 125),
                "failure_in_30_cycles": 1 if rul <= 30 else 0,
                "T30": 1580.0 + 0.5 * c + rng.normal(0, 0.2),
                "Ps30": 555.0 - 0.2 * c + rng.normal(0, 0.1),
                "BPR": 8.4 + 0.01 * c + rng.normal(0, 0.005),
                "T50": 1400.0 + 0.3 * c + rng.normal(0, 0.2),
            })
    return pd.DataFrame(records)


def test_isolation_forest_anomaly_detector(mock_telemetry_df):
    detector = IsolationForestAnomalyDetector(n_estimators=30)
    feature_cols = ["T30", "Ps30", "BPR", "T50"]
    detector.fit(mock_telemetry_df, feature_cols, healthy_rul_threshold=30)

    scored = detector.predict_score(mock_telemetry_df)
    assert "anomaly_score" in scored.columns
    scores = scored["anomaly_score"]
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_pytorch_autoencoder_detector(mock_telemetry_df):
    detector = PyTorchAutoencoderDetector(latent_dim=2, epochs=5, batch_size=32)
    feature_cols = ["T30", "Ps30", "BPR", "T50"]
    detector.fit(mock_telemetry_df, feature_cols, healthy_rul_threshold=30)

    scored = detector.predict_score(mock_telemetry_df)
    assert "autoencoder_anomaly_score" in scored.columns
    scores = scored["autoencoder_anomaly_score"]
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_xgboost_classifier(mock_telemetry_df):
    clf = TurbofanXGBoostClassifier(n_estimators=30, max_depth=3)
    feature_cols = ["T30", "Ps30", "BPR", "T50"]
    clf.fit(mock_telemetry_df, feature_cols, target_col="failure_in_30_cycles")

    probas = clf.predict_proba(mock_telemetry_df)
    assert len(probas) == len(mock_telemetry_df)
    assert np.all((probas >= 0.0) & (probas <= 1.0))

    metrics = clf.evaluate(mock_telemetry_df)
    assert metrics["pr_auc"] >= 0.70
    assert metrics["roc_auc"] >= 0.70

    fi = clf.get_feature_importances()
    assert len(fi) == 4


def test_weibull_survival_model(mock_telemetry_df):
    surv = TurbofanWeibullSurvivalModel(penalizer=0.01)
    surv.fit(mock_telemetry_df, covariate_cols=["T30", "Ps30", "BPR"])

    intervals = surv.predict_intervals(mock_telemetry_df)
    assert "rul_pred_lower_80ci" in intervals.columns
    assert "rul_pred_upper_80ci" in intervals.columns
    assert "survival_ci_display" in intervals.columns

    # Lower bound must be <= upper bound
    assert np.all(intervals["rul_pred_lower_80ci"] <= intervals["rul_pred_upper_80ci"])


def test_health_scoring():
    # Healthy case
    healthy_score = calculate_health_score(failure_prob=0.05, anomaly_score=0.1, rul_clipped=125.0)
    assert healthy_score >= 80.0
    tier, icon, hex_color = get_health_tier(healthy_score)
    assert "Healthy" in tier

    # Critical failure case
    critical_score = calculate_health_score(failure_prob=0.95, anomaly_score=0.9, rul_clipped=5.0)
    assert critical_score <= 40.0
    tier, icon, hex_color = get_health_tier(critical_score)
    assert "Critical" in tier


def test_explain_prediction(mock_telemetry_df):
    clf = TurbofanXGBoostClassifier(n_estimators=30, max_depth=3)
    feature_cols = ["T30", "Ps30", "BPR", "T50"]
    clf.fit(mock_telemetry_df, feature_cols, target_col="failure_in_30_cycles")

    snapshot = mock_telemetry_df.iloc[[0]]
    diag = explain_engine_risk(clf, snapshot, top_k=2)

    assert "failure_probability" in diag
    assert "diagnostic_summary" in diag
    assert len(diag["top_contributors"]) == 2

