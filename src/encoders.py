"""
Project HQ-IDS-029: Classical Feature Compressors & Autoencoders
================================================================
This module implements PyTorch classical feature compression models:
1. BottleneckMLPEncoder: Supervised / hybrid bottleneck encoder mapping
   high-dimensional network flow features (N=40) down to 4 continuous
   latent variables z in [-pi, pi]^4.
2. FlowAutoencoder: Standalone PyTorch Autoencoder (Encoder + Decoder) for
   unsupervised representation learning and dimensionality reduction via MSE loss.
"""

import os
import math
import copy
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class BottleneckMLPEncoder(nn.Module):
    """
    Classical feature compressor mapping N=40 network flow features down to
    exactly 4 continuous latent variables bounded in [-pi, pi] for quantum angle encoding.
    Architecture: Input [40] -> Linear(16) + BatchNorm1d + ReLU -> Linear(4) + Tanh * pi.
    """

    def __init__(self, input_dim: int = 40, hidden_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale Tanh [-1, 1] to [-pi, pi] for direct mapping to rotation gates
        # Handle single-sample batch normalization edge case
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x) * math.pi


class FlowAutoencoder(nn.Module):
    """
    Unsupervised feature compression Autoencoder.
    - Encoder: 40 -> 16 -> 4 (bounded in [-pi, pi])
    - Decoder: 4 -> 16 -> 40
    Trained with Mean Squared Error (MSE) reconstruction loss.
    """

    def __init__(self, input_dim: int = 40, hidden_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_linear1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_bn = nn.BatchNorm1d(hidden_dim)
        self.encoder_act = nn.ReLU()
        self.encoder_linear2 = nn.Linear(hidden_dim, latent_dim)
        self.encoder_tanh = nn.Tanh()

        # Decoder
        self.decoder_linear1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_bn = nn.BatchNorm1d(hidden_dim)
        self.decoder_act = nn.ReLU()
        self.decoder_linear2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compresses input into 4 continuous latent variables in [-pi, pi]."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.encoder_act(self.encoder_bn(self.encoder_linear1(x)))
        z = self.encoder_tanh(self.encoder_linear2(h)) * math.pi
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs original feature dimensions from 4 latent variables."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h = self.decoder_act(self.decoder_bn(self.decoder_linear1(z)))
        x_rec = self.decoder_linear2(h)
        return x_rec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        X_val: Optional[np.ndarray] = None,
        device: str = "cpu",
        random_seed: int = 42
    ) -> Dict[str, List[float]]:
        """
        Trains the Autoencoder using MSE loss with Adam optimizer.
        """
        torch.manual_seed(random_seed)
        dev = torch.device(device)
        self.to(dev)

        X_t = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32).to(dev)
        else:
            X_v = None

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(dev)
                optimizer.zero_grad()
                x_rec, _ = self.forward(batch_x)
                loss = criterion(x_rec, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            train_loss = epoch_loss / len(X_train)
            history["train_loss"].append(train_loss)

            if X_v is not None:
                self.eval()
                with torch.no_grad():
                    val_rec, _ = self.forward(X_v)
                    val_loss = criterion(val_rec, X_v).item()
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.state_dict())

        if X_v is not None:
            self.load_state_dict(best_state)

        self.eval()
        return history


if __name__ == "__main__":
    print("=" * 70)
    print("Project HQ-IDS-029: Classical Feature Compressors Self-Test")
    print("=" * 70)

    # Test BottleneckMLPEncoder
    mlp_enc = BottleneckMLPEncoder(input_dim=40, hidden_dim=16, latent_dim=4)
    dummy_x = torch.randn(16, 40)
    z_mlp = mlp_enc(dummy_x)

    print(f"[BottleneckMLPEncoder] Input shape: {dummy_x.shape} -> Latent shape: {z_mlp.shape}")
    print(f"[BottleneckMLPEncoder] Value range: [{z_mlp.min().item():.4f}, {z_mlp.max().item():.4f}] (Target: [{-math.pi:.4f}, {math.pi:.4f}])")
    assert z_mlp.shape == (16, 4), "Unexpected latent shape!"
    assert z_mlp.min().item() >= -math.pi - 1e-4 and z_mlp.max().item() <= math.pi + 1e-4

    # Test FlowAutoencoder
    ae = FlowAutoencoder(input_dim=40, hidden_dim=16, latent_dim=4)
    x_rec, z_ae = ae(dummy_x)
    print(f"[FlowAutoencoder] Input: {dummy_x.shape} -> Latent: {z_ae.shape} -> Reconstructed: {x_rec.shape}")
    assert x_rec.shape == (16, 40), "Reconstruction shape mismatch!"

    # Test Autoencoder training loop
    X_dummy_np = np.random.randn(200, 40).astype(np.float32)
    history = ae.fit(X_dummy_np, epochs=5, batch_size=32)
    print(f"[FlowAutoencoder] Fit test completed. Final train loss: {history['train_loss'][-1]:.4f}")

    print("\nAll feature compressor modules verified successfully!")
