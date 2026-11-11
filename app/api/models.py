"""
Pydantic Schemas for FastAPI Microservice.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SensorSnapshot(BaseModel):
    """Telemetry sensor snapshot for a single flight cycle."""
    unit_nr: int = Field(default=1, description="Engine identifier")
    time_cycles: int = Field(default=100, description="Flight cycle count")
    setting_1: float = Field(default=0.0)
    setting_2: float = Field(default=0.0)
    setting_3: float = Field(default=100.0)
    T24: float = Field(default=642.5, description="LPC outlet temperature (°R)")
    T30: float = Field(default=1588.0, description="HPC outlet temperature (°R)")
    T50: float = Field(default=1405.0, description="LPT outlet temperature (°R)")
    P30: float = Field(default=553.8, description="HPC outlet total pressure (psia)")
    Nf: float = Field(default=2388.0, description="Physical fan speed (rpm)")
    Nc: float = Field(default=9050.0, description="Physical core speed (rpm)")
    Ps30: float = Field(default=47.2, description="HPC outlet static pressure (psia)")
    phi: float = Field(default=521.8, description="Fuel flow to Ps30 ratio (pps/psi)")
    NRf: float = Field(default=2388.0, description="Corrected fan speed (rpm)")
    NRc: float = Field(default=8130.0, description="Corrected core speed (rpm)")
    BPR: float = Field(default=8.45, description="Bypass ratio")
    htBleed: float = Field(default=392.0, description="Bleed enthalpy")
    W31: float = Field(default=38.8, description="HPT coolant bleed (lbm/s)")
    W32: float = Field(default=23.3, description="LPT coolant bleed (lbm/s)")


class PredictionResponse(BaseModel):
    """Response containing layered multi-model predictions."""
    unit_nr: int
    time_cycles: int
    anomaly_score: float = Field(description="Layer 1 Anomaly Score [0, 1]")
    failure_probability: float = Field(description="Layer 2 P(failure in 30 cycles)")
    predicted_rul_median: int = Field(description="Layer 3 Median RUL (cycles)")
    rul_80ci_lower: int = Field(description="Layer 3 80% CI Lower Bound")
    rul_80ci_upper: int = Field(description="Layer 3 80% CI Upper Bound")
    survival_display: str
    health_score: float = Field(description="Unified Health Score [0, 100]")
    health_tier: str
    health_color: str


class ExplainResponse(BaseModel):
    """Response containing aerospace engineering diagnostics."""
    unit_nr: int
    time_cycles: int
    failure_probability: float
    risk_level: str
    affected_subsystems: List[str]
    top_contributors: List[Dict[str, Any]]
    diagnostic_summary: str

