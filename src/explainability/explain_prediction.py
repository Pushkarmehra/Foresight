"""
Aviation Domain Diagnostics & Explainability.

Translates mathematical Shapley values into clear physical engineering diagnostics
for airline maintenance engineers and dispatchers.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from src.explainability.shap_analysis import TurbofanExplainer


# Domain mapping from raw sensor abbreviations to aerospace mechanical components
TURBOFAN_SENSOR_METADATA: Dict[str, Dict[str, str]] = {
    "T30": {
        "name": "HPC Exit Total Temperature",
        "subsystem": "High-Pressure Compressor",
        "degradation_mode": "Aerodynamic efficiency drop & thermal stress",
    },
    "Ps30": {
        "name": "HPC Exit Static Pressure",
        "subsystem": "High-Pressure Compressor",
        "degradation_mode": "Compressor blade erosion and tip clearance loss",
    },
    "T50": {
        "name": "LPT Exit Total Temperature",
        "subsystem": "Low-Pressure Turbine",
        "degradation_mode": "Turbine blade thermal fatigue and nozzle guide vane distress",
    },
    "BPR": {
        "name": "Bypass Ratio",
        "subsystem": "Fan and Bypass Duct",
        "degradation_mode": "Core-flow choking and fan blade fouling",
    },
    "phi": {
        "name": "Fuel Flow to Static Pressure Ratio",
        "subsystem": "Combustor & Fuel Nozzles",
        "degradation_mode": "Combustor efficiency loss and thermal over-fueling",
    },
    "NRc": {
        "name": "Corrected Core Shaft Speed",
        "subsystem": "High-Pressure Spool",
        "degradation_mode": "Shaft bearing wear and spool friction drag",
    },
    "htBleed": {
        "name": "Bleed Enthalpy",
        "subsystem": "Turbine Bleed Air System",
        "degradation_mode": "Bleed valve leakage and thermal enthalpy losses",
    },
    "W31": {
        "name": "HPT Coolant Bleed Flow",
        "subsystem": "High-Pressure Turbine Cooling",
        "degradation_mode": "Cooling channel blockage or seal degradation",
    },
}


def explain_engine_risk(
    classifier,
    sensor_snapshot_df: pd.DataFrame,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Generate an aerospace engineering diagnostic explanation for a given engine reading.
    """
    failure_prob = float(classifier.predict_proba(sensor_snapshot_df)[0])
    explainer = TurbofanExplainer(classifier)
    top_contributors = explainer.get_top_contributors(sensor_snapshot_df, top_k=top_k)

    findings = []
    affected_subsystems = set()

    for item in top_contributors:
        feat_name = item["feature"].split("_")[0]  # strip rolling / lag suffixes
        meta = TURBOFAN_SENSOR_METADATA.get(feat_name, {
            "name": item["feature"],
            "subsystem": "Turbomachinery Core",
            "degradation_mode": "Sensor drift and mechanical wear",
        })

        subsystem = meta["subsystem"]
        affected_subsystems.add(subsystem)
        findings.append(
            f"{meta['name']} ({item['feature']}) contributed significantly "
            f"(SHAP impact +{item['shap_value']:.3f}), reflecting {meta['degradation_mode']}."
        )

    risk_level = "CRITICAL" if failure_prob >= 0.7 else "ELEVATED" if failure_prob >= 0.3 else "NOMINAL"

    summary_text = (
        f"Fleet Advisory [{risk_level} RISK - {failure_prob * 100:.1f}% failure probability]: "
        f"Primary physical degradation identified in {', '.join(sorted(list(affected_subsystems)))}. "
        + " ".join(findings)
    )

    return {
        "failure_probability": failure_prob,
        "risk_level": risk_level,
        "affected_subsystems": sorted(list(affected_subsystems)),
        "top_contributors": top_contributors,
        "diagnostic_summary": summary_text,
    }

