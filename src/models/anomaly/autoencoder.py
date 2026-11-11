"""
Layer 1 Anomaly Detection: PyTorch Bottleneck Autoencoder.

Compresses multi-sensor telemetry through a 4-dimensional latent bottleneck
and reconstructs healthy patterns. High Mean Squared Reconstruction Error (MSE)
flags anomalous mechanical degradation signatures.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AutoencoderNet(nn.Module):
    """Deep bottleneck autoencoder for sensor telemetry reconstruction."""

    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        # Encoder: input_dim -> 32 -> 16 -> latent_dim (4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        # Decoder: latent_dim -> 16 -> 32 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class PyTorchAutoencoderDetector:
    """
    Autoencoder Anomaly Detector wrapper managing data scaling, training,
    inference, and model persistence.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 25,
        random_seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_seed = random_seed
        self.feature_cols: List[str] = []
        self.scaler = StandardScaler()
        self.net: Optional[AutoencoderNet] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._error_p99: float = 1.0

        torch.manual_seed(self.random_seed)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        healthy_rul_threshold: int = 100,
        rul_col: str = "rul",
    ) -> "PyTorchAutoencoderDetector":
        """
        Fit autoencoder on healthy sensor cycles.
        """
        self.feature_cols = list(feature_cols)
        input_dim = len(self.feature_cols)

        # Filter for healthy regime
        if rul_col in df.columns:
            train_df = df[df[rul_col] >= healthy_rul_threshold]
            if len(train_df) < 50:
                train_df = df
        else:
            train_df = df

        X = train_df[self.feature_cols].fillna(0.0).to_numpy()
        X_scaled = self.scaler.fit_transform(X)

        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = AutoencoderNet(input_dim=input_dim, latent_dim=self.latent_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)

        self.net.train()
        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        # Calibrate reconstruction error threshold
        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            recon = self.net(tensor_x)
            mse_per_row = torch.mean((recon - tensor_x) ** 2, dim=1).cpu().numpy()
            self._error_p99 = float(np.percentile(mse_per_row, 99))
            if self._error_p99 <= 1e-6:
                self._error_p99 = 1.0

        return self

    def predict_score(
        self,
        df: pd.DataFrame,
        score_col: str = "autoencoder_anomaly_score",
    ) -> pd.DataFrame:
        """
        Compute reconstruction error normalized into [0.0, 1.0].
        """
        if self.net is None:
            raise RuntimeError("Model has not been fitted yet.")

        out = df.copy()
        X = out[self.feature_cols].fillna(0.0).to_numpy()
        X_scaled = self.scaler.transform(X)

        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            recon = self.net(tensor_x)
            mse_per_row = torch.mean((recon - tensor_x) ** 2, dim=1).cpu().numpy()

        norm_score = np.clip(mse_per_row / self._error_p99, 0.0, 1.0).astype(np.float32)
        out[score_col] = norm_score
        return out

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model weights and metadata."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "feature_cols": self.feature_cols,
            "latent_dim": self.latent_dim,
            "scaler": self.scaler,
            "_error_p99": self._error_p99,
            "state_dict": self.net.state_dict() if self.net else None,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PyTorchAutoencoderDetector":
        """Load serialized autoencoder."""
        checkpoint = torch.load(Path(filepath), map_location="cpu", weights_only=False)
        detector = cls(latent_dim=checkpoint["latent_dim"])
        detector.feature_cols = checkpoint["feature_cols"]
        detector.scaler = checkpoint["scaler"]
        detector._error_p99 = checkpoint["_error_p99"]
        detector.net = AutoencoderNet(
            input_dim=len(detector.feature_cols),
            latent_dim=detector.latent_dim,
        )
        if checkpoint["state_dict"] is not None:
            detector.net.load_state_dict(checkpoint["state_dict"])
        detector.net.eval()
        return detector

