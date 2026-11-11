"""
Master Pipeline Orchestrator for Foresight Turbofan Predictive Maintenance.

Executes the complete end-to-end AI workflow:
1. Ingests processed NASA C-MAPSS telemetry.
2. Conducts 5-Fold Engine-Grouped Cross-Validation (Zero Data Leakage).
3. Trains production models across the 3-Layer Defense-in-Depth Architecture:
   - Layer 1: Isolation Forest & Autoencoder Anomaly Detectors
   - Layer 2: Imbalance-Aware XGBoost & LightGBM Classifiers
   - Layer 3: Weibull Accelerated Failure Time (AFT) Survival Model
4. Evaluates against test ground-truth.
5. Computes Unified Health Scores and aerospace domain SHAP diagnostics.
6. Serializes all model artifacts to `data/models/` and generates fleet reports.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset_bundle
from src.data.prepare_data import process_dataset
from src.validation.group_kfold import get_unit_group_kfold
from src.validation.leakage_checks import (
    assert_no_unit_overlap,
    assert_temporal_monotonicity,
    assert_feature_target_separation,
)
from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from src.models.anomaly.autoencoder import PyTorchAutoencoderDetector
from src.models.classifier.xgboost_model import TurbofanXGBoostClassifier
from src.models.classifier.lightgbm_model import TurbofanLightGBMClassifier
from src.models.survival.weibull_aft import TurbofanWeibullSurvivalModel
from src.explainability.explain_prediction import explain_engine_risk
from src.health_score import compute_health_scores, get_health_tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineOrchestrator")

MODELS_DIR = PROJECT_ROOT / "data" / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def get_model_feature_cols(df: pd.DataFrame) -> List[str]:
    """Extract all sensor, rolling stat, lag, and rate features, excluding targets/meta."""
    exclude = {
        "unit_nr", "time_cycles", "rul", "rul_clipped", "true_rul",
        "failure_in_30_cycles", "event", "duration", "anomaly_score",
        "autoencoder_anomaly_score", "failure_prob", "health_score",
        "health_tier", "health_color"
    }
    return [c for c in df.columns if c not in exclude and not c.startswith("failure_in_")]


def run_cross_validation(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
) -> Dict[str, float]:
    """Execute 5-fold engine-grouped cross validation."""
    logger.info(f"Running {n_splits}-Fold GroupKFold Cross-Validation...")
    folds = list(get_unit_group_kfold(train_df, n_splits=n_splits, seed=42))

    pr_aucs, roc_aucs, f1s = [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        tr_data = train_df.iloc[tr_idx]
        val_data = train_df.iloc[val_idx]

        assert_no_unit_overlap(tr_data, val_data)

        clf = TurbofanXGBoostClassifier(n_estimators=120, max_depth=5, learning_rate=0.05)
        clf.fit(tr_data, feature_cols, target_col="failure_in_30_cycles", val_df=val_data)
        metrics = clf.evaluate(val_data)

        pr_aucs.append(metrics["pr_auc"])
        roc_aucs.append(metrics["roc_auc"])
        f1s.append(metrics["f1_score"])

    cv_results = {
        "mean_pr_auc": float(np.mean(pr_aucs)),
        "std_pr_auc": float(np.std(pr_aucs)),
        "mean_roc_auc": float(np.mean(roc_aucs)),
        "std_roc_auc": float(np.std(roc_aucs)),
        "mean_f1": float(np.mean(f1s)),
    }
    logger.info(
        f"CV Results -> PR-AUC: {cv_results['mean_pr_auc']:.4f} ± {cv_results['std_pr_auc']:.4f} | "
        f"ROC-AUC: {cv_results['mean_roc_auc']:.4f} ± {cv_results['std_roc_auc']:.4f} | "
        f"F1: {cv_results['mean_f1']:.4f}"
    )
    return cv_results


def run_pipeline(dataset: str = "FD001", save_models: bool = True) -> Dict[str, Any]:
    """Execute complete training, evaluation, and fleet inference workflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / f"train_{dataset}_processed.parquet"
    test_path = PROCESSED_DIR / f"test_{dataset}_processed.parquet"

    if not train_path.exists() or not test_path.exists():
        logger.info(f"Processed files missing. Running process_dataset for {dataset}...")
        process_dataset(dataset)

    logger.info(f"Loading processed telemetry from {train_path.name} and {test_path.name}...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Pre-flight integrity validation
    assert_temporal_monotonicity(train_df)
    assert_temporal_monotonicity(test_df)

    feature_cols = get_model_feature_cols(train_df)
    assert_feature_target_separation(feature_cols)
    logger.info(f"Verified {len(feature_cols)} feature channels with zero target leakage.")

    # 1. Cross-Validation
    cv_metrics = run_cross_validation(train_df, feature_cols, n_splits=5)

    # 2. Layer 1: Train Anomaly Detectors
    logger.info("Training Layer 1: Isolation Forest & PyTorch Autoencoder...")
    iso_detector = IsolationForestAnomalyDetector()
    iso_detector.fit(train_df, feature_cols, healthy_rul_threshold=100)

    autoencoder = PyTorchAutoencoderDetector(latent_dim=4, epochs=15)
    autoencoder.fit(train_df, feature_cols, healthy_rul_threshold=100)

    # 3. Layer 2: Train Classifiers
    logger.info("Training Layer 2: Imbalance-Aware XGBoost & LightGBM...")
    xgb_clf = TurbofanXGBoostClassifier(n_estimators=180, max_depth=5, learning_rate=0.05)
    xgb_clf.fit(train_df, feature_cols, target_col="failure_in_30_cycles")

    lgb_clf = TurbofanLightGBMClassifier(n_estimators=180, max_depth=5, learning_rate=0.05)
    lgb_clf.fit(train_df, feature_cols, target_col="failure_in_30_cycles")

    # 4. Layer 3: Train Weibull Survival Model
    logger.info("Training Layer 3: Weibull AFT Survival Model...")
    survival_model = TurbofanWeibullSurvivalModel(penalizer=0.01)
    # Use key physical degrading sensors as covariates
    covariates = [c for c in ["T30", "Ps30", "BPR", "T50", "phi", "NRc"] if c in feature_cols]
    survival_model.fit(train_df, covariate_cols=covariates)

    # 5. Evaluate on Test Dataset
    logger.info("Evaluating all layers on Test Fleet...")
    test_eval = xgb_clf.evaluate(test_df, target_col="failure_in_30_cycles")
    logger.info(
        f"Test Fleet Metrics -> PR-AUC: {test_eval['pr_auc']:.4f} | "
        f"ROC-AUC: {test_eval['roc_auc']:.4f} | F1: {test_eval['f1_score']:.4f} | "
        f"Recall: {test_eval['recall']:.4f}"
    )

    # Generate inference predictions across test dataset
    scored_test = iso_detector.predict_score(test_df, score_col="anomaly_score")
    scored_test["failure_prob"] = xgb_clf.predict_proba(scored_test)
    scored_test = survival_model.predict_intervals(scored_test, confidence_level=0.80)
    scored_test = compute_health_scores(scored_test)

    # Extract latest snapshot per engine
    fleet_snapshots = scored_test.sort_values(["unit_nr", "time_cycles"]).groupby("unit_nr").last().reset_index()

    # 6. Save Model Artifacts
    if save_models:
        logger.info(f"Serializing all model checkpoints to {MODELS_DIR}...")
        iso_detector.save(MODELS_DIR / "isolation_forest.joblib")
        autoencoder.save(MODELS_DIR / "autoencoder.pt")
        xgb_clf.save(MODELS_DIR / "xgboost_model.joblib")
        lgb_clf.save(MODELS_DIR / "lightgbm_model.joblib")
        survival_model.save(MODELS_DIR / "weibull_survival.joblib")

        # Save feature list & metrics
        metrics_summary = {
            "dataset": dataset,
            "cv_metrics": cv_metrics,
            "test_metrics": test_eval,
            "total_features": len(feature_cols),
            "feature_cols": feature_cols,
        }
        with open(MODELS_DIR / "evaluation_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # Save Fleet Predictions Table
        output_report_path = PROCESSED_DIR / f"fleet_predictions_{dataset}.csv"
        report_cols = [
            "unit_nr", "time_cycles", "rul", "rul_clipped",
            "anomaly_score", "failure_prob", "rul_pred_lower_80ci",
            "rul_pred_median", "rul_pred_upper_80ci", "survival_ci_display",
            "health_score", "health_tier", "health_color"
        ]
        available_cols = [c for c in report_cols if c in fleet_snapshots.columns]
        fleet_snapshots[available_cols].to_csv(output_report_path, index=False)
        logger.info(f"Fleet executive report saved to {output_report_path}")

    # Generate sample explainability diagnosis on critical engine
    critical_units = fleet_snapshots[fleet_snapshots["failure_prob"] >= 0.5]
    sample_diag = {}
    if not critical_units.empty:
        sample_unit_row = critical_units.iloc[[0]]
        sample_diag = explain_engine_risk(xgb_clf, sample_unit_row)

    return {
        "dataset": dataset,
        "cv_metrics": cv_metrics,
        "test_metrics": test_eval,
        "fleet_size": len(fleet_snapshots),
        "critical_engines": int((fleet_snapshots["health_score"] < 40).sum()),
        "warning_engines": int(((fleet_snapshots["health_score"] >= 40) & (fleet_snapshots["health_score"] < 70)).sum()),
        "healthy_engines": int((fleet_snapshots["health_score"] >= 70).sum()),
        "sample_diagnosis": sample_diag,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Foresight Master ML Pipeline")
    parser.add_argument("--dataset", type=str, default="FD001", help="Dataset subset (FD001..FD004)")
    parser.add_argument("--no-save", action="store_true", help="Do not save model checkpoints")
    args = parser.parse_args()

    results = run_pipeline(dataset=args.dataset, save_models=not args.no_save)
    print("\n" + "=" * 70)
    print("🎯 FORESIGHT MASTER PIPELINE EXECUTION COMPLETED")
    print("=" * 70)
    print(f"Dataset                : {results['dataset']}")
    print(f"5-Fold CV PR-AUC       : {results['cv_metrics']['mean_pr_auc']:.4f}")
    print(f"5-Fold CV ROC-AUC      : {results['cv_metrics']['mean_roc_auc']:.4f}")
    print(f"Test Set PR-AUC        : {results['test_metrics']['pr_auc']:.4f}")
    print(f"Test Set ROC-AUC       : {results['test_metrics']['roc_auc']:.4f}")
    print(f"Fleet Monitored Units  : {results['fleet_size']}")
    print(f"🟢 Healthy Units       : {results['healthy_engines']}")
    print(f"🟡 Warning Units       : {results['warning_engines']}")
    print(f"🔴 Critical Units      : {results['critical_engines']}")
    if results.get("sample_diagnosis"):
        print("\n🔍 Sample Critical Engine Diagnostic Advisory:")
        print(results["sample_diagnosis"]["diagnostic_summary"])
    print("=" * 70)