"""
Project HQ-IDS-029: Anomaly Detection Architectures (Classical & Quantum)
========================================================================
This module implements:
1. ClassicalAutoencoder (CAE):
   A 3-layer deep autoencoder (39 -> 16 -> 4 -> 16 -> 39) trained with MSE loss
   strictly on benign network flows. Anomaly score is defined as the per-sample
   reconstruction mean squared error.

2. QuantumAutoencoder (QAE):
   Based on Romero et al. (2017) quantum data compression:
   - An input flow vector x in R^39 is encoded into 4 latent variables z in [-pi, pi]^4.
   - z is angle-embedded into a 4-qubit register (|q0, q1, q2, q3>).
   - A parameterized unitary U(theta) acts across all 4 qubits to compress the
     state into 2 latent qubits (q0, q1) while driving the 2 trash qubits (q2, q3)
     into the reference vacuum state |00>.
   - Trained on benign flows by minimizing trash infidelity: L = 1 - P(|00>_trash).
   - Anomaly score: 1 - P(|00>_trash). Unseen zero-day intrusions violate the benign
     manifold compression, generating significant excitation leakage into the trash
     qubits (P(|00>) drops, infidelity spikes).
"""

import os
import sys
import math
import copy
import time
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from src import determinism as _determinism  # noqa: F401  (pins threads; see src/determinism.py)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

# Ensure repo root is accessible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ClassicalAutoencoder(nn.Module):
    """
    Classical Bottleneck Autoencoder for Unsupervised Anomaly Detection.
    Architecture:
      Input (39) -> Linear(16) -> BatchNorm -> ReLU -> Linear(4) -> Tanh*pi (Latent Bottleneck)
      Latent (4) -> Linear(16) -> BatchNorm -> ReLU -> Linear(39) (Reconstructed Input)
    """

    def __init__(self, input_dim: int = 39, hidden_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x) * math.pi

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def fit_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 25,
        lr: float = 0.003,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = torch.device(device)
        self.to(dev)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        best_state = None

        self.train()
        for epoch in range(epochs):
            total_train_loss = 0.0
            n_samples = 0
            for batch in train_loader:
                X_batch = batch[0].to(dev)
                optimizer.zero_grad()
                x_rec, _ = self.forward(X_batch)
                loss = criterion(x_rec, X_batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(X_batch)
                n_samples += len(X_batch)

            epoch_train_loss = total_train_loss / max(n_samples, 1)
            history["train_loss"].append(epoch_train_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch in val_loader:
                        X_val = batch[0].to(dev)
                        x_rec, _ = self.forward(X_val)
                        loss = criterion(x_rec, X_val)
                        val_loss += loss.item() * len(X_val)
                        val_samples += len(X_val)
                epoch_val_loss = val_loss / max(val_samples, 1)
                history["val_loss"].append(epoch_val_loss)
                self.train()

                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    best_state = copy.deepcopy(self.state_dict())

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        return history

    def compute_anomaly_scores(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Computes per-sample Mean Squared Error (MSE) reconstruction anomaly scores.
        """
        self.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.tensor(X[i : i + batch_size], dtype=torch.float32)
                x_rec, _ = self.forward(batch_X)
                # Per-sample MSE: (B, D) -> (B,)
                mse = torch.mean((batch_X - x_rec) ** 2, dim=1).cpu().numpy()
                scores.append(mse)
        return np.concatenate(scores, axis=0)


class QuantumAutoencoder(nn.Module):
    """
    Quantum Autoencoder (Romero et al. 2017) with Classical Encoder Front-End.
    - Compresses n=4 qubits into k=2 latent qubits (wires 0, 1).
    - Steering wires 2 and 3 (trash qubits) to state |00>.
    - Loss & Anomaly score: Infidelity = 1 - P(|00>_trash).
    """

    def __init__(
        self,
        classical_encoder: nn.Module,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend: str = "default.qubit",
    ):
        super().__init__()
        self.encoder = classical_encoder
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Device setup
        self.dev = qml.device(backend, wires=n_qubits)

        # Parameter initialization for QAE unitary: (n_layers, n_qubits, 3)
        init_weights = torch.empty(n_layers, n_qubits, 3).uniform_(0.0, 2.0 * math.pi)
        self.weights = nn.Parameter(init_weights)

        # Build QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            # 1. State preparation via angle embedding on all 4 qubits
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # 2. Parameterized unitary encoder U(theta)
            for l in range(weights.shape[0]):
                for i in range(self.n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                # Entangling gates across all pairs to capture multi-qubit correlations
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])
                qml.CNOT(wires=[0, 2])
                qml.CNOT(wires=[1, 3])

            # 3. Measure trash state probabilities on wires 2, 3
            # Index 0 corresponds to computational basis outcome |00>
            return qml.probs(wires=[2, 3])

        self.circuit = _circuit

    def forward_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Runs the quantum autoencoder circuit on batch of 4D latent vectors.
        Returns trash state probability distribution: shape (batch_size, 4).
        Index 0 is P(|00>).
        """
        probs = [self.circuit(x, self.weights) for x in z]
        return torch.stack(probs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input flows through classical encoder then quantum circuit.
        """
        z = self.encoder.encode(x)
        return self.forward_latent(z)

    def fit_model(
        self,
        train_loader: DataLoader,
        epochs: int = 12,
        lr: float = 0.02,
        device: str = "cpu",
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        """
        Trains the Quantum Autoencoder unitary weights strictly on benign flows
        to maximize P(|00>_trash), i.e., minimize trash infidelity (1 - P(|00>)).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        optimizer = torch.optim.Adam([self.weights], lr=lr)

        history = {"train_loss": [], "p00_fidelity": []}

        for epoch in range(epochs):
            total_loss = 0.0
            total_p00 = 0.0
            n_samples = 0

            for batch in train_loader:
                X_batch = batch[0]
                optimizer.zero_grad()

                # Extract latent representation with frozen classical encoder
                with torch.no_grad():
                    z_batch = self.encoder.encode(X_batch)

                probs = self.forward_latent(z_batch)
                # Probability of |00> is at index 0
                p00 = probs[:, 0]
                infidelity = 1.0 - p00
                loss = torch.mean(infidelity)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(X_batch)
                total_p00 += torch.mean(p00).item() * len(X_batch)
                n_samples += len(X_batch)

            epoch_loss = total_loss / max(n_samples, 1)
            epoch_p00 = total_p00 / max(n_samples, 1)
            history["train_loss"].append(epoch_loss)
            history["p00_fidelity"].append(epoch_p00)

        return history

    def compute_anomaly_scores(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Computes Quantum Infidelity Anomaly Scores: Score(x) = 1 - P(|00>_trash).
        """
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.tensor(X[i : i + batch_size], dtype=torch.float32)
                z = self.encoder.encode(batch_X)
                probs = self.forward_latent(z)
                p00 = probs[:, 0].cpu().numpy()
                infidelity = 1.0 - p00
                scores.append(infidelity)
        return np.concatenate(scores, axis=0)


if __name__ == "__main__":
    print("Testing ClassicalAutoencoder and QuantumAutoencoder models...")
    cae = ClassicalAutoencoder(input_dim=39, hidden_dim=16, latent_dim=4)
    dummy_x = torch.randn(8, 39)
    x_rec, z = cae(dummy_x)
    print(f"Classical AE: Input {dummy_x.shape} -> Latent {z.shape} -> Rec {x_rec.shape}")
    assert x_rec.shape == dummy_x.shape

    qae = QuantumAutoencoder(classical_encoder=cae, n_qubits=4, n_layers=2)
    probs = qae(dummy_x)
    print(f"Quantum AE: Trash Probs shape {probs.shape}, Mean P(|00>): {probs[:, 0].mean().item():.4f}")
    assert probs.shape == (8, 4)
    print("All model self-tests PASSED successfully!")
