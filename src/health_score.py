"""
Unified Fleet Health Scoring Engine.

Synthesizes outputs across the 3-Layer Defense-in-Depth Architecture:
    - Layer 1: Anomaly Score in [0, 1]
    - Layer 2: Failure Probability in [0, 1]
    - Layer 3: Remaining Useful Life (RUL)

Computes the executive Foresight Health Score:
    HealthScore = 100 * [1 - 0.50*P(fail) - 0.30*AnomalyScore - 0.20*(1 - RUL_clipped / 125)]
Clamped strictly to [0.0, 100.0].
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_health_score(
    failure_prob: float,
    anomaly_score: float,
    rul_clipped: float,
    max_rul_clip: float = 125.0,
) -> float:
    """
    Calculate unified health score for a single turbine snapshot.

    Parameters
    ----------
    failure_prob : float
        P(failure in 30 cycles) from Layer 2 classifier in [0, 1].
    anomaly_score : float
        Normalized anomaly score from Layer 1 detector in [0, 1].
    rul_clipped : float
        Clipped remaining useful life in cycles.
    max_rul_clip : float, default=125.0
        Nominal healthy baseline plateau ceiling.

    Returns
    -------
    float
        Health score clamped in [0.0, 100.0].
    """
    fp = float(np.clip(failure_prob, 0.0, 1.0))
    anom = float(np.clip(anomaly_score, 0.0, 1.0))
    rul_fraction = float(np.clip(rul_clipped / max_rul_clip, 0.0, 1.0))

    # Weighting: 50% classifier risk, 30% anomaly deviation, 20% degradation countdown
    score = 100.0 * (1.0 - (0.50 * fp) - (0.30 * anom) - (0.20 * (1.0 - rul_fraction)))
    return float(np.clip(score, 0.0, 100.0))


def compute_health_scores(
    df: pd.DataFrame,
    failure_prob_col: str = "failure_prob",
    anomaly_score_col: str = "anomaly_score",
    rul_col: str = "rul_clipped",
    output_col: str = "health_score",
    tier_col: str = "health_tier",
    color_col: str = "health_color",
    max_rul_clip: float = 125.0,
) -> pd.DataFrame:
    """
    Vectorized health score computation across fleet telemetry DataFrame.
    """
    out = df.copy()

    # Fill missing columns with safe defaults if needed
    fp = out[failure_prob_col].to_numpy() if failure_prob_col in out.columns else np.zeros(len(out))
    anom = out[anomaly_score_col].to_numpy() if anomaly_score_col in out.columns else np.zeros(len(out))
    rul = out[rul_col].to_numpy() if rul_col in out.columns else np.full(len(out), max_rul_clip)

    fp = np.clip(fp, 0.0, 1.0)
    anom = np.clip(anom, 0.0, 1.0)
    rul_fraction = np.clip(rul / max_rul_clip, 0.0, 1.0)

    scores = 100.0 * (1.0 - (0.50 * fp) - (0.30 * anom) - (0.20 * (1.0 - rul_fraction)))
    scores = np.clip(scores, 0.0, 100.0)

    out[output_col] = np.round(scores, 1)

    # Classify into operational maintenance tiers
    tiers = np.where(scores >= 70.0, "Healthy (Normal)", np.where(scores >= 40.0, "Warning (Maintenance Scheduled)", "Critical (Immediate Overhaul)"))
    colors = np.where(scores >= 70.0, "#34C759", np.where(scores >= 40.0, "#FF9500", "#FF3B30"))

    out[tier_col] = tiers
    out[color_col] = colors

    return out


def get_health_tier(score: float) -> Tuple[str, str, str]:
    """
    Return (Tier Label, Status Emoji, Hex Color) for a given health score.
    """
    if score >= 70.0:
        return "Healthy (Normal)", "🟢", "#34C759"
    elif score >= 40.0:
        return "Warning (Maintenance Scheduled)", "🟡", "#FF9500"
    else:
        return "Critical (Immediate Overhaul)", "🔴", "#FF3B30"

