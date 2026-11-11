"""
NASA C-MAPSS Telemetry Data Loader.

Loads, maps, and validates run-to-failure datasets for the C-MAPSS
commercial turbofan engine simulation benchmark (FD001 to FD004).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Standard 26-column schema defined in Saxena et al. (PHM 2008)
COLUMNS: List[str] = [
    "unit_nr",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    "T2",        # Total temperature at fan inlet (°R)
    "T24",       # Total temperature at LPC outlet (°R)
    "T30",       # Total temperature at HPC outlet (°R)
    "T50",       # Total temperature at LPT outlet (°R)
    "P2",        # Pressure at fan inlet (psia)
    "P15",       # Total pressure in bypass-duct (psia)
    "P30",       # Total pressure at HPC outlet (psia)
    "Nf",        # Physical fan speed (rpm)
    "Nc",        # Physical core speed (rpm)
    "epr",       # Engine pressure ratio (P50/P2)
    "Ps30",      # Static pressure at HPC outlet (psia)
    "phi",       # Ratio of fuel flow to Ps30 (pps/psi)
    "NRf",       # Corrected fan speed (rpm)
    "NRc",       # Corrected core speed (rpm)
    "BPR",       # Bypass Ratio
    "farB",      # Burner fuel-air ratio
    "htBleed",   # Bleed Enthalpy
    "Nf_dmd",    # Demanded fan speed (rpm)
    "PCNfR_dmd", # Demanded corrected fan speed (rpm)
    "W31",       # HPT coolant bleed (lbm/s)
    "W32",       # LPT coolant bleed (lbm/s)
]

# Sensors with zero variance in single-condition dataset (FD001)
INVARIANT_SENSORS_FD001: List[str] = [
    "T2",
    "P2",
    "P15",
    "epr",
    "farB",
    "Nf_dmd",
    "PCNfR_dmd",
    "setting_3",
]

# Active sensors with non-zero variance showing physical degradation signatures in FD001
ACTIVE_SENSORS_FD001: List[str] = [
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

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def get_raw_data_dir() -> Path:
    """Returns the local path to data/raw directory."""
    return DEFAULT_RAW_DIR


def load_raw_train(
    dataset: str = "FD001",
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load raw training telemetry for a given C-MAPSS subset.
    
    Parameters
    ----------
    dataset : str
        Dataset identifier: 'FD001', 'FD002', 'FD003', or 'FD004'.
    data_dir : str or Path, optional
        Directory containing raw text files. Defaults to data/raw.

    Returns
    -------
    pd.DataFrame
        26-column typed DataFrame with unit numbers, cycles, settings, and sensors.
    """
    raw_path = Path(data_dir) if data_dir else get_raw_data_dir()
    file_path = raw_path / f"train_{dataset}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        dtype={"unit_nr": np.int32, "time_cycles": np.int32},
    )
    return df


def load_raw_test(
    dataset: str = "FD001",
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load raw test telemetry (right-censored) for a given C-MAPSS subset.
    """
    raw_path = Path(data_dir) if data_dir else get_raw_data_dir()
    file_path = raw_path / f"test_{dataset}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        dtype={"unit_nr": np.int32, "time_cycles": np.int32},
    )
    return df


def load_test_rul(
    dataset: str = "FD001",
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load ground-truth Remaining Useful Life values for the test engines.

    Parameters
    ----------
    dataset : str
        Dataset identifier: 'FD001', 'FD002', 'FD003', or 'FD004'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['unit_nr', 'true_rul'], indexed from 1 to N engines.
    """
    raw_path = Path(data_dir) if data_dir else get_raw_data_dir()
    file_path = raw_path / f"RUL_{dataset}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth RUL file not found: {file_path}")

    # Note: header=None and sep=r"\s+" ensures all rows are read as data (no header drop)
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["true_rul"])
    df["unit_nr"] = np.arange(1, len(df) + 1, dtype=np.int32)
    return df[["unit_nr", "true_rul"]]


def load_dataset_bundle(
    dataset: str = "FD001",
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function returning (train_df, test_df, rul_df) for a given subset.
    """
    train_df = load_raw_train(dataset, data_dir)
    test_df = load_raw_test(dataset, data_dir)
    rul_df = load_test_rul(dataset, data_dir)
    return train_df, test_df, rul_df


if __name__ == "__main__":
    print("Testing C-MAPSS loader on FD001...")
    train, test, rul = load_dataset_bundle("FD001")
    print(f"Train shape: {train.shape} (Units: {train['unit_nr'].nunique()})")
    print(f"Test shape : {test.shape}  (Units: {test['unit_nr'].nunique()})")
    print(f"RUL shape  : {rul.shape}   (Units: {len(rul)})")
    print("\nSample Train Row:")
    print(train.head(1)[["unit_nr", "time_cycles", "T24", "T30", "P30", "Nc", "Ps30", "BPR"]])
