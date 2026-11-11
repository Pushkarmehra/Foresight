"""
End-to-End Data Processing Pipeline: Raw C-MAPSS to Modeling-Ready Processed Datasets.

Executes across all benchmark subsets (FD001, FD002, FD003, FD004):
1. Ingestion: Reads space-separated raw C-MAPSS telemetry and maps all 26 columns.
2. Target Engineering:
   - Training RUL: RUL = max(time_cycles) - t
   - Piecewise Clipping: RUL_clipped = min(RUL, 125)
   - Test RUL: Back-projected from ground truth RUL vector
   - Binary Target: failure_in_30_cycles = I(RUL <= 30)
3. Sensor Selection & Pruning:
   - Identifies invariant sensor channels per dataset subset
   - Preserves all active degradation channels (14 for FD001/FD003, 21 for FD002/FD004)
4. Comprehensive Feature Engineering:
   - Trailing rolling statistics (mean, std, min, max over 5, 10, 20 cycle windows)
   - Multi-step historical lags (lags: 1, 3, 5)
   - Degradation velocity & rates of change (steps: 3, 5)
5. Persistence:
   - Saves processed matrices to data/processed/ (.parquet and .csv)
   - Optionally syncs processed parquets & CSVs to AWS S3.
"""

from pathlib import Path
import sys
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    load_dataset_bundle,
    COLUMNS,
    ACTIVE_SENSORS_FD001,
    INVARIANT_SENSORS_FD001,
)
from src.labeling.rul import add_train_rul, add_test_rul
from src.labeling.failure_horizon import add_failure_horizon
from src.features.rolling_stats import compute_trailing_rolling_features
from src.features.lags import compute_lag_features
from src.features.rates import compute_rate_features

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

ALL_SENSOR_CHANNELS = [
    c for c in COLUMNS
    if c.startswith(("T", "P", "N", "e", "p", "B", "f", "h", "W"))
]

INVARIANT_SENSORS_MAP: Dict[str, List[str]] = {
    "FD001": INVARIANT_SENSORS_FD001,
    "FD003": INVARIANT_SENSORS_FD001,
    "FD002": [],
    "FD004": [],
}


def get_sensors_for_dataset(dataset: str) -> Tuple[List[str], List[str]]:
    """
    Return (active_sensors, invariant_sensors) for a given dataset subset.
    """
    key = dataset.upper()
    invariant_sensors = INVARIANT_SENSORS_MAP.get(key, [])
    active_sensors = [s for s in ALL_SENSOR_CHANNELS if s not in invariant_sensors]
    return active_sensors, invariant_sensors


def process_dataset(
    dataset: str = "FD001",
    max_rul_clip: int = 125,
    horizon: int = 30,
    include_rolling: bool = True,
    include_lags: bool = True,
    include_rates: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform raw NASA C-MAPSS dataset into modeling-ready feature tables.
    """
    dataset = dataset.upper()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FORESIGHT: PROCESSING RAW TO PROCESSED DATA ({dataset})")
    print("=" * 70)

    # 1. Ingest Raw Data
    print(f"\n[1/4] Ingesting raw {dataset} telemetry...")
    train_df, test_df, rul_df = load_dataset_bundle(dataset)
    print(f"  -> Loaded Raw Train: {train_df.shape} ({train_df['unit_nr'].nunique()} engines)")
    print(f"  -> Loaded Raw Test : {test_df.shape} ({test_df['unit_nr'].nunique()} engines)")
    print(f"  -> Loaded True RUL : {rul_df.shape} ({len(rul_df)} ground-truth engines)")

    # 2. Target Engineering
    print(f"\n[2/4] Engineering target labels (Max RUL Clip: {max_rul_clip}, Horizon: {horizon} cycles)...")
    train_labeled = add_train_rul(train_df, max_rul_clip=max_rul_clip)
    test_labeled = add_test_rul(test_df, rul_df, max_rul_clip=max_rul_clip)

    train_labeled = add_failure_horizon(train_labeled, horizon=horizon)
    test_labeled = add_failure_horizon(test_labeled, horizon=horizon)

    target_col = f"failure_in_{horizon}_cycles"
    train_pos_ratio = (train_labeled[target_col] == 1).mean() * 100
    test_pos_ratio = (test_labeled[target_col] == 1).mean() * 100
    print(f"  -> Train Imminent Failure Rate ({target_col}): {train_pos_ratio:.2f}%")
    print(f"  -> Test Imminent Failure Rate  ({target_col}): {test_pos_ratio:.2f}%")

    # 3. Sensor Pruning & Feature Engineering
    print(f"\n[3/4] Feature engineering across active sensor channels...")
    active_sensors, invariant_sensors = get_sensors_for_dataset(dataset)
    print(f"  -> Active channels: {len(active_sensors)} ({', '.join(active_sensors)})")
    if invariant_sensors:
        print(f"  -> Invariant channels dropped: {len(invariant_sensors)} ({', '.join(invariant_sensors)})")

    # Drop zero-variance sensors
    train_clean = train_labeled.drop(columns=[col for col in invariant_sensors if col in train_labeled.columns])
    test_clean = test_labeled.drop(columns=[col for col in invariant_sensors if col in test_labeled.columns])

    # Trailing rolling stats (mean, std, min, max)
    if include_rolling:
        print("  -> Computing trailing rolling statistics (windows: 5, 10, 20)...")
        train_clean = compute_trailing_rolling_features(train_clean, active_sensors, windows=[5, 10, 20])
        test_clean = compute_trailing_rolling_features(test_clean, active_sensors, windows=[5, 10, 20])

    # Lag features (1, 3, 5)
    if include_lags:
        print("  -> Computing multi-step lag features (lags: 1, 3, 5)...")
        train_clean = compute_lag_features(train_clean, active_sensors, lags=[1, 3, 5])
        test_clean = compute_lag_features(test_clean, active_sensors, lags=[1, 3, 5])

    # Degradation rates (3, 5)
    if include_rates:
        print("  -> Computing degradation velocity & rates (steps: 3, 5)...")
        train_clean = compute_rate_features(train_clean, active_sensors, steps=[3, 5])
        test_clean = compute_rate_features(test_clean, active_sensors, steps=[3, 5])

    # Check for NaNs or Infinite values
    train_nans = train_clean.isna().sum().sum()
    test_nans = test_clean.isna().sum().sum()
    print(f"  -> Missing values check: Train NaNs={train_nans}, Test NaNs={test_nans}")

    # 4. Persistence
    print(f"\n[4/4] Saving processed datasets to: {PROCESSED_DIR}")
    train_parquet = PROCESSED_DIR / f"train_{dataset}_processed.parquet"
    test_parquet = PROCESSED_DIR / f"test_{dataset}_processed.parquet"
    train_csv = PROCESSED_DIR / f"train_{dataset}_processed.csv"
    test_csv = PROCESSED_DIR / f"test_{dataset}_processed.csv"

    train_clean.to_parquet(train_parquet, index=False)
    test_clean.to_parquet(test_parquet, index=False)
    print(f"  [OK] Saved {train_parquet.name} ({train_clean.shape[0]} rows, {train_clean.shape[1]} features)")
    print(f"  [OK] Saved {test_parquet.name} ({test_clean.shape[0]} rows, {test_clean.shape[1]} features)")

    # Also save CSV for easy quick inspection
    train_clean.to_csv(train_csv, index=False)
    test_clean.to_csv(test_csv, index=False)
    print(f"  [OK] Saved CSV exports for quick inspection")

    print("\n" + "=" * 70)
    print(f"{dataset} PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return train_clean, test_clean


def process_all_datasets(
    datasets: List[str] = ALL_DATASETS,
    max_rul_clip: int = 125,
    horizon: int = 30,
    include_rolling: bool = True,
    include_lags: bool = True,
    include_rates: bool = True,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Process all requested C-MAPSS subsets in sequence.
    """
    results = {}
    for ds in datasets:
        results[ds] = process_dataset(
            dataset=ds,
            max_rul_clip=max_rul_clip,
            horizon=horizon,
            include_rolling=include_rolling,
            include_lags=include_lags,
            include_rates=include_rates,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NASA C-MAPSS raw data to processed features.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset subset: 'all' (default, processes FD001..FD004), 'FD001', 'FD002', 'FD003', or 'FD004'",
    )
    parser.add_argument("--upload-s3", action="store_true", help="Upload processed parquets & CSVs to AWS S3")
    args = parser.parse_args()

    if args.dataset.lower() == "all":
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [args.dataset.upper()]

    for ds in datasets_to_run:
        train_processed, test_processed = process_dataset(dataset=ds)

        if args.upload_s3:
            try:
                from src.data.s3_utils import s3, S3_BUCKET_NAME
                print(f"\nSyncing {ds} processed files to AWS S3...")
                for fname in [
                    f"train_{ds}_processed.parquet",
                    f"test_{ds}_processed.parquet",
                    f"train_{ds}_processed.csv",
                    f"test_{ds}_processed.csv",
                ]:
                    local_f = PROCESSED_DIR / fname
                    if local_f.exists():
                        s3_key = f"processed/{fname}"
                        print(f"Uploading {fname} -> s3://{S3_BUCKET_NAME}/{s3_key}")
                        s3.upload_file(str(local_f), S3_BUCKET_NAME, s3_key)
                print(f"[OK] {ds} S3 sync completed successfully!")
            except Exception as e:
                print(f"Notice: S3 upload for {ds} skipped or failed: {e}")

    print("\n" + "=" * 70)
    print("ALL REQUESTED DATASETS SUCCESSFULLY PROCESSED!")
    print("=" * 70)
