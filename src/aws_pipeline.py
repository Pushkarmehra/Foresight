"""
AWS Lambda: NASA C-MAPSS Predictive Maintenance Feature Engineering Pipeline
----------------------------------------------------------------------------
Trigger : S3 ObjectCreated events on
          s3://foresight-predictive-maintenance/raw/cmapss/*.txt

Behavior:
  1. Every time a file lands under raw/cmapss/, the function lists the
     prefix and checks whether all 12 required C-MAPSS files are present
     (train/test/RUL for FD001..FD004).
  2. If any are missing, it logs which ones and exits (no-op) — it does
     NOT process partial data.
  3. If all 12 are present, it runs the full feature pipeline for each
     of the 4 sub-datasets:
       - RUL labeling (T_max - t for train, back-projection for test)
       - Piecewise RUL clipping (max 125 cycles)
       - Binary failure horizon labeling (failure_in_30_cycles)
       - Invariant-sensor detection & removal
       - Trailing rolling stats: mean, std, min, max (windows: 5, 10, 20)
       - Multi-step lag features (lags: 1, 3, 5)
       - Degradation rates / velocities (steps: 3, 5)
       - Writes train/test Parquet and CSV outputs to s3://.../processed/.

Idempotent: Re-running overwrites the processed outputs deterministically.
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# ============================================================================
# ENVIRONMENT & CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "foresight-predictive-maintenance")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "raw/cmapss/")
S3_PROCESSED_PREFIX = os.getenv("S3_PROCESSED_PREFIX", "processed/")

DEFAULT_BUCKET = S3_BUCKET_NAME
RAW_PREFIX = S3_RAW_PREFIX.rstrip("/") + "/"
PROCESSED_PREFIX = S3_PROCESSED_PREFIX.rstrip("/") + "/"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


# ============================================================================
# AWS S3 CLIENT
# ============================================================================

def get_s3_client():
    """
    Create an S3 client.

    If AWS credentials are provided in .env, they are used.
    Otherwise boto3 automatically uses the Lambda IAM execution role
    or the local AWS credential chain.
    """
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION or None
        )

    # Recommended for Lambda:
    # Use the Lambda execution role instead of hardcoding credentials.
    return boto3.client(
        "s3",
        region_name=AWS_REGION or None
    )


s3 = get_s3_client()


# ============================================================================
# C-MAPSS DATASETS & COLUMN NAMES
# ============================================================================

# C-MAPSS datasets
FD_IDS = [
    "001",
    "002",
    "003",
    "004"
]

COLUMNS = [
    "unit_nr",
    "time_cycles",

    "setting_1",
    "setting_2",
    "setting_3",

    "T2",
    "T24",
    "T30",
    "T50",

    "P2",
    "P15",
    "P30",

    "Nf",
    "Nc",

    "epr",
    "Ps30",
    "phi",

    "NRf",
    "NRc",

    "BPR",
    "farB",
    "htBleed",

    "Nf_dmd",
    "PCNfR_dmd",

    "W31",
    "W32",
]

INVARIANT_SENSORS_FD001 = [
    "T2",
    "P2",
    "P15",
    "epr",
    "farB",
    "Nf_dmd",
    "PCNfR_dmd",
    "setting_3",
]

ACTIVE_SENSORS_FD001 = [
    "T24",
    "T30",
    "T50",
    "P30",
    "Nf",
    "Nc",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "htBleed",
    "W31",
    "W32",
]

ALL_SENSOR_CHANNELS = [
    c for c in COLUMNS
    if c.startswith(("T", "P", "N", "e", "p", "B", "f", "h", "W"))
]

INVARIANT_SENSORS_MAP: Dict[str, List[str]] = {
    "001": INVARIANT_SENSORS_FD001,
    "003": INVARIANT_SENSORS_FD001,
    "002": [],
    "004": [],
}


def get_sensors_for_dataset(fd_id: str) -> Tuple[List[str], List[str]]:
    """
    Return (active_sensors, invariant_sensors) for a given dataset subset.
    - FD001 & FD003 (single condition): 14 active, 8 invariant
    - FD002 & FD004 (6 flight regimes): 21 active, 0 invariant
    """
    key = fd_id.replace("FD", "").zfill(3)
    invariant_sensors = INVARIANT_SENSORS_MAP.get(key, [])
    active_sensors = [s for s in ALL_SENSOR_CHANNELS if s not in invariant_sensors]
    return active_sensors, invariant_sensors


# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

ROLLING_WINDOWS = [5, 10, 20]
LAG_STEPS = [1, 3, 5]
RATE_STEPS = [3, 5]
RUL_CLIP = 125
FAILURE_HORIZON = 30


# ============================================================================
# GET REQUIRED FILES
# ============================================================================

def get_required_keys() -> Set[str]:
    """
    Return the 12 C-MAPSS files required before processing starts.

    4 datasets × 3 files:
        train
        test
        RUL
    """
    keys = set()

    for fd in FD_IDS:
        keys.add(
            f"{RAW_PREFIX}train_FD{fd}.txt"
        )
        keys.add(
            f"{RAW_PREFIX}test_FD{fd}.txt"
        )
        keys.add(
            f"{RAW_PREFIX}RUL_FD{fd}.txt"
        )

    return keys


# ============================================================================
# S3 HELPERS
# ============================================================================

def list_existing_keys(
    bucket: str,
    prefix: str
) -> Set[str]:
    """
    Return all S3 object keys under the specified prefix.
    """
    existing = set()

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            existing.add(
                obj["Key"]
            )

    return existing


def read_fixed_width_txt_from_s3(
    bucket: str,
    key: str,
    names: List[str]
) -> pd.DataFrame:
    """
    Read a whitespace-separated C-MAPSS TXT file from S3.
    """
    logger.info(
        "Reading s3://%s/%s",
        bucket,
        key
    )

    obj = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    raw = obj["Body"].read()

    df = pd.read_csv(
        io.BytesIO(raw),
        sep=r"\s+",
        header=None,
        names=names,
        engine="python"
    )

    return df


def save_parquet_to_s3(
    df: pd.DataFrame,
    bucket: str,
    key: str
) -> None:
    """
    Convert DataFrame to Parquet and upload it to S3.
    """
    table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )

    buffer = io.BytesIO()

    pq.write_table(
        table,
        buffer
    )

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )

    logger.info(
        "Saved Parquet -> s3://%s/%s | rows=%d | cols=%d",
        bucket,
        key,
        len(df),
        len(df.columns)
    )


def save_csv_to_s3(
    df: pd.DataFrame,
    bucket: str,
    key: str
) -> None:
    """
    Convert DataFrame to CSV and upload it to S3.
    """
    buffer = io.StringIO()

    df.to_csv(
        buffer,
        index=False
    )

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue().encode("utf-8")
    )

    logger.info(
        "Saved CSV -> s3://%s/%s | rows=%d",
        bucket,
        key,
        len(df)
    )


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def add_rolling_features(
    df: pd.DataFrame,
    sensors: List[str],
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate trailing rolling mean, standard deviation, min, and max.
    Strictly partitioned by unit_nr to prevent lookahead and cross-engine contamination.
    """
    df = df.copy()

    if "time_cycles" in df.columns:
        df = df.sort_values(
            ["unit_nr", "time_cycles"]
        ).reset_index(drop=True)

    grouped = df.groupby("unit_nr")[sensors]

    for window in windows:
        # Rolling Mean
        rolling_mean = (
            grouped.rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolling_mean.columns = [
            f"{col}_roll_mean_{window}" for col in sensors
        ]

        # Rolling Std (fill single-cycle initial with 0.0)
        rolling_std = (
            grouped.rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        rolling_std.columns = [
            f"{col}_roll_std_{window}" for col in sensors
        ]

        # Rolling Min
        rolling_min = (
            grouped.rolling(window=window, min_periods=1)
            .min()
            .reset_index(level=0, drop=True)
        )
        rolling_min.columns = [
            f"{col}_roll_min_{window}" for col in sensors
        ]

        # Rolling Max
        rolling_max = (
            grouped.rolling(window=window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        rolling_max.columns = [
            f"{col}_roll_max_{window}" for col in sensors
        ]

        df = pd.concat(
            [df, rolling_mean, rolling_std, rolling_min, rolling_max],
            axis=1
        )

    return df


def add_lag_features(
    df: pd.DataFrame,
    sensors: List[str],
    lags: List[int] = [1, 3, 5]
) -> pd.DataFrame:
    """
    Compute multi-step lag features partitioned by unit_nr.
    """
    df = df.copy()
    grouped = df.groupby("unit_nr")

    lag_dfs = []
    for lag in lags:
        lag_df = grouped[sensors].shift(lag).bfill()
        lag_df.columns = [f"{col}_lag_{lag}" for col in sensors]
        lag_dfs.append(lag_df)

    if lag_dfs:
        df = pd.concat([df] + lag_dfs, axis=1)

    return df


def add_rate_features(
    df: pd.DataFrame,
    sensors: List[str],
    steps: List[int] = [3, 5]
) -> pd.DataFrame:
    """
    Compute degradation velocity / rate of change over given cycle intervals.
    """
    df = df.copy()
    grouped = df.groupby("unit_nr")

    rate_dfs = []
    for k in steps:
        diff_df = grouped[sensors].diff(periods=k) / float(k)
        diff_df = diff_df.fillna(0.0)
        diff_df.columns = [f"{col}_rate_{k}" for col in sensors]
        rate_dfs.append(diff_df)

    if rate_dfs:
        df = pd.concat([df] + rate_dfs, axis=1)

    return df


# ============================================================================
# PROCESS ONE C-MAPSS DATASET
# ============================================================================

def process_fd_dataset(
    bucket: str,
    fd_id: str
) -> Dict[str, Any]:
    """
    Process one C-MAPSS dataset.

    Example:
        FD001
        FD002
        FD003
        FD004
    """
    train_key = (
        f"{RAW_PREFIX}train_FD{fd_id}.txt"
    )

    test_key = (
        f"{RAW_PREFIX}test_FD{fd_id}.txt"
    )

    rul_key = (
        f"{RAW_PREFIX}RUL_FD{fd_id}.txt"
    )

    logger.info("=" * 70)
    logger.info(
        "Processing C-MAPSS FD%s",
        fd_id
    )
    logger.info("=" * 70)

    # ========================================================================
    # 1. LOAD RAW DATA
    # ========================================================================

    train_df = read_fixed_width_txt_from_s3(
        bucket,
        train_key,
        COLUMNS
    )

    test_df = read_fixed_width_txt_from_s3(
        bucket,
        test_key,
        COLUMNS
    )

    rul_df = read_fixed_width_txt_from_s3(
        bucket,
        rul_key,
        ["true_rul"]
    )

    # RUL rows correspond to test engine numbers
    rul_df["unit_nr"] = np.arange(
        1,
        len(rul_df) + 1,
        dtype=np.int32
    )

    logger.info(
        "Loaded FD%s | Train=%s | Test=%s | RUL=%s",
        fd_id,
        train_df.shape,
        test_df.shape,
        rul_df.shape
    )

    # ========================================================================
    # 2. TRAINING RUL
    # ========================================================================

    # Maximum cycle reached by each engine
    max_cycles = (
        train_df
        .groupby("unit_nr")["time_cycles"]
        .transform("max")
    )

    # RUL = maximum cycle - current cycle
    train_df["rul"] = (
        max_cycles -
        train_df["time_cycles"]
    )

    # Piecewise RUL clipping
    train_df["rul_clipped"] = (
        train_df["rul"]
        .clip(upper=RUL_CLIP)
    )

    # ========================================================================
    # 3. TEST RUL
    # ========================================================================

    last_cycles = (
        test_df
        .groupby("unit_nr")["time_cycles"]
        .transform("max")
    )

    test_df = test_df.merge(
        rul_df[
            [
                "unit_nr",
                "true_rul"
            ]
        ],
        on="unit_nr",
        how="left"
    )

    # Back-project test RUL
    test_df["rul"] = (
        test_df["true_rul"] +
        (
            last_cycles -
            test_df["time_cycles"]
        )
    )

    test_df["rul_clipped"] = (
        test_df["rul"]
        .clip(upper=RUL_CLIP)
    )

    test_df.drop(
        columns=["true_rul"],
        inplace=True
    )

    # ========================================================================
    # 4. FAILURE HORIZON
    # ========================================================================

    # 1 = engine is expected to fail within 30 cycles
    # 0 = otherwise
    train_df["failure_in_30_cycles"] = (
        train_df["rul"] <= FAILURE_HORIZON
    ).astype(np.int32)

    test_df["failure_in_30_cycles"] = (
        test_df["rul"] <= FAILURE_HORIZON
    ).astype(np.int32)

    # ========================================================================
    # 5. SENSOR SELECTION & PRUNING
    # ========================================================================

    active_sensors, invariant_sensors = get_sensors_for_dataset(fd_id)

    logger.info(
        "FD%s active sensors (%d channels): %s",
        fd_id,
        len(active_sensors),
        active_sensors
    )
    if invariant_sensors:
        logger.info(
            "FD%s invariant sensors dropped (%d channels): %s",
            fd_id,
            len(invariant_sensors),
            invariant_sensors
        )

    # Remove invariant sensors
    train_df.drop(
        columns=[
            c
            for c in invariant_sensors
            if c in train_df.columns
        ],
        inplace=True
    )

    test_df.drop(
        columns=[
            c
            for c in invariant_sensors
            if c in test_df.columns
        ],
        inplace=True
    )

    # ========================================================================
    # 6. FEATURE ENGINEERING (ROLLING STATS, LAGS, RATES)
    # ========================================================================

    # Trailing rolling statistics (mean, std, min, max)
    train_processed = add_rolling_features(
        train_df,
        active_sensors,
        ROLLING_WINDOWS
    )
    test_processed = add_rolling_features(
        test_df,
        active_sensors,
        ROLLING_WINDOWS
    )

    # Multi-step lag features (1, 3, 5)
    train_processed = add_lag_features(
        train_processed,
        active_sensors,
        LAG_STEPS
    )
    test_processed = add_lag_features(
        test_processed,
        active_sensors,
        LAG_STEPS
    )

    # Degradation velocity & rates (3, 5)
    train_processed = add_rate_features(
        train_processed,
        active_sensors,
        RATE_STEPS
    )
    test_processed = add_rate_features(
        test_processed,
        active_sensors,
        RATE_STEPS
    )

    # ========================================================================
    # 7. SAVE PROCESSED DATA TO S3
    # ========================================================================

    train_parquet_key = (
        f"{PROCESSED_PREFIX}"
        f"train_FD{fd_id}_processed.parquet"
    )

    test_parquet_key = (
        f"{PROCESSED_PREFIX}"
        f"test_FD{fd_id}_processed.parquet"
    )

    train_csv_key = (
        f"{PROCESSED_PREFIX}"
        f"train_FD{fd_id}_processed.csv"
    )

    test_csv_key = (
        f"{PROCESSED_PREFIX}"
        f"test_FD{fd_id}_processed.csv"
    )

    # Parquet
    save_parquet_to_s3(
        train_processed,
        bucket,
        train_parquet_key
    )

    save_parquet_to_s3(
        test_processed,
        bucket,
        test_parquet_key
    )

    # CSV
    save_csv_to_s3(
        train_processed,
        bucket,
        train_csv_key
    )

    save_csv_to_s3(
        test_processed,
        bucket,
        test_csv_key
    )

    # ========================================================================
    # 8. STATISTICS
    # ========================================================================

    train_failure_rate = float(
        (
            train_processed[
                "failure_in_30_cycles"
            ] == 1
        ).mean() * 100
    )

    test_failure_rate = float(
        (
            test_processed[
                "failure_in_30_cycles"
            ] == 1
        ).mean() * 100
    )

    # ========================================================================
    # 9. RETURN RESULTS
    # ========================================================================

    return {
        "fd_id": fd_id,
        "train_rows": len(
            train_processed
        ),
        "test_rows": len(
            test_processed
        ),
        "feature_count":
            train_processed.shape[1],
        "active_sensor_count":
            len(active_sensors),
        "train_imminent_failure_rate_pct":
            round(
                train_failure_rate,
                2
            ),
        "test_imminent_failure_rate_pct":
            round(
                test_failure_rate,
                2
            ),
        "train_parquet_s3":
            f"s3://{bucket}/{train_parquet_key}",
        "test_parquet_s3":
            f"s3://{bucket}/{test_parquet_key}",
    }


# ============================================================================
# AWS LAMBDA HANDLER
# ============================================================================

def lambda_handler(
    event: Dict[str, Any],
    context: Any
) -> Dict[str, Any]:

    logger.info(
        "Lambda invocation received: %s",
        json.dumps(
            event,
            default=str
        )
    )

    # ========================================================================
    # 1. DETERMINE BUCKET
    # ========================================================================

    bucket = DEFAULT_BUCKET

    if (
        "Records" in event
        and len(event["Records"]) > 0
    ):
        record = event["Records"][0]

        if (
            "s3" in record
            and "bucket" in record["s3"]
        ):
            bucket = record[
                "s3"
            ].get(
                "name",
                DEFAULT_BUCKET
            )

            trigger_key = (
                record["s3"]["object"]
                .get(
                    "key",
                    "unknown"
                )
            )

            logger.info(
                "Triggered by S3 event: "
                "s3://%s/%s",
                bucket,
                trigger_key
            )

    # ========================================================================
    # 2. VALIDATE BUCKET
    # ========================================================================

    if not bucket:
        logger.error(
            "S3 bucket is not configured."
        )

        return {
            "statusCode": 500,
            "body": json.dumps({
                "message":
                    "S3 bucket is not configured. "
                    "Fill S3_BUCKET_NAME in .env or "
                    "configure the S3 trigger."
            })
        }

    # ========================================================================
    # 3. CHECK ALL 12 REQUIRED FILES
    # ========================================================================

    logger.info(
        "Checking required C-MAPSS files "
        "in s3://%s/%s",
        bucket,
        RAW_PREFIX
    )

    required_keys = (
        get_required_keys()
    )

    existing_keys = (
        list_existing_keys(
            bucket,
            RAW_PREFIX
        )
    )

    missing_keys = (
        required_keys -
        existing_keys
    )

    # ========================================================================
    # 4. WAIT IF DATA IS INCOMPLETE
    # ========================================================================

    if missing_keys:
        logger.info(
            "Waiting for %d missing file(s): %s",
            len(missing_keys),
            sorted(missing_keys)
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message":
                        f"Waiting for "
                        f"{len(missing_keys)} "
                        f"more file(s). "
                        f"Partial data is not processed.",
                    "missing_files":
                        sorted(missing_keys),
                    "existing_count":
                        len(existing_keys),
                    "required_count":
                        len(required_keys)
                },
                indent=2
            )
        }

    # ========================================================================
    # 5. PROCESS ALL FOUR DATASETS
    # ========================================================================

    logger.info(
        "All 12 C-MAPSS files found."
    )
    logger.info(
        "Starting feature engineering..."
    )

    results = []

    for fd_id in FD_IDS:
        result = process_fd_dataset(
            bucket,
            fd_id
        )
        results.append(
            result
        )

    # ========================================================================
    # 6. SUCCESS
    # ========================================================================

    logger.info(
        "All FD001-FD004 datasets "
        "successfully processed."
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message":
                    "All FD001-FD004 datasets "
                    "processed and saved to S3.",
                "bucket":
                    bucket,
                "processed_prefix":
                    PROCESSED_PREFIX,
                "results":
                    results
            },
            indent=2
        )
    }


# ============================================================================
# LOCAL TESTING & CLI RUNNER
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run C-MAPSS AWS Lambda pipeline across all raw datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset subset: 'all' (default, processes FD001..FD004), '001', '002', '003', or '004'",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TESTING C-MAPSS S3/LAMBDA PIPELINE (ALL DATASETS)")
    print("=" * 70)
    print(f"Target Bucket : {DEFAULT_BUCKET}")
    print(f"Raw Prefix    : {RAW_PREFIX}")
    print(f"Output Prefix : {PROCESSED_PREFIX}")
    print()

    if args.dataset.lower() == "all":
        # Simulate S3 event triggering full batch across all 4 datasets
        mock_event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {
                            "name": DEFAULT_BUCKET
                        },
                        "object": {
                            "key": f"{RAW_PREFIX}train_FD001.txt"
                        }
                    }
                }
            ]
        }
        response = lambda_handler(
            mock_event,
            None
        )
    else:
        target_fd = args.dataset.upper().replace("FD", "").zfill(3)
        res = process_fd_dataset(DEFAULT_BUCKET, target_fd)
        response = {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Dataset FD{target_fd} processed and saved to S3.",
                "bucket": DEFAULT_BUCKET,
                "results": [res]
            }, indent=2)
        }

    print("\nLambda Execution Result:")
    print(
        json.dumps(
            response,
            indent=2
        )
    )
