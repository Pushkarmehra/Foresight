"""
Layer 1 Anomaly Detection Models.
"""

from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from src.models.anomaly.autoencoder import PyTorchAutoencoderDetector

__all__ = [
    "IsolationForestAnomalyDetector",
    "PyTorchAutoencoderDetector",
]

