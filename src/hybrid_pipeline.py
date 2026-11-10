"""
Project HQ-IDS-029: End-to-End Hybrid Quantum-Classical Pipeline
================================================================
Combines:
  Input [40] -> BottleneckMLPEncoder [4] -> PennyLaneVQCModule [4] -> Linear Head [2]

Provides a unified interface:
- fit(X_train, y_train, X_val=None, y_val=None, epochs=20, lr=0.01, batch_size=64)
- predict(X) -> np.ndarray
- predict_proba(X) -> np.ndarray
- measure_latency(X_sample, n_runs=100) -> float (ms/sample)
- get_quantum_expectation_values(X) -> np.ndarray
"""

import os
import time
import copy
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Suppress font cache / matplotlib warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
# Add repo root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import BottleneckMLPEncoder
from src.quantum_circuits import PennyLaneVQCModule, QiskitVQCBuilder


class _HybridQNNModule(nn.Module):
    """
    Composite PyTorch module linking classical encoder, quantum variational circuit,
    and linear classification readout head.

    `quantum_layer` selects the ablation variant:
      * "vqc"     - full hybrid QNN: trained 4-qubit VQC between encoder and head
      * "identity"- Control A: the quantum layer is bypassed, the latent angles
                    z in [-pi, pi]^4 feed the linear head directly
      * "frozen"  - Control B: an untrained, randomly initialised VQC whose
                    parameters are frozen (requires_grad=False); the encoder and
                    head still train
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 16,
        num_qubits: int = 4,
        num_layers: int = 2,
        output_dim: int = 2,
        ansatz_type: str = "RealAmplitudes",
        feature_map: str = "AngleEmbedding",
        feature_map_reps: int = 1,
        quantum_layer: str = "vqc",
        output_mode: str = "expectations",
    ):
        super().__init__()
        self.quantum_layer = quantum_layer
        self.output_mode = output_mode
        self.encoder = BottleneckMLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=num_qubits
        )

        if quantum_layer == "vqc":
            self.vqc = PennyLaneVQCModule(
                num_qubits=num_qubits,
                num_layers=num_layers,
                ansatz_type=ansatz_type,
                feature_map=feature_map,
                feature_map_reps=feature_map_reps,
                output_mode=output_mode,
            )
        elif quantum_layer == "identity":
            self.vqc = nn.Identity()
        elif quantum_layer == "frozen":
            self.vqc = PennyLaneVQCModule(
                num_qubits=num_qubits,
                num_layers=num_layers,
                ansatz_type=ansatz_type,
                feature_map=feature_map,
                feature_map_reps=feature_map_reps,
                output_mode=output_mode,
            )
            for parameter in self.vqc.parameters():
                parameter.requires_grad = False
        else:
            raise ValueError(f"Unsupported quantum_layer: {quantum_layer}")

        head_dim = 2**num_qubits if (quantum_layer != "identity" and output_mode == "bitstrings") else num_qubits
        self.head = nn.Linear(head_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Classical feature compression: input -> 4 latent angles in [-pi, pi]
        z = self.encoder(x)
        # 2. Parameterized Quantum Circuit: 4 -> 4 expectations or 16 probabilities
        q_out = self.vqc(z)
        # 3. Classical readout head -> class logits
        logits = self.head(q_out)
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 4 bottleneck latent rotation angles z in [-pi, pi]^4."""
        return self.encoder(x)

    def extract_latent_and_quantum(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        q_expvals = self.vqc(z)
        return z, q_expvals


class HybridQNNClassifier:
    """
    Production wrapper for the Hybrid Quantum Neural Network Intrusion Detector.
    Matches the unified baseline interface (fit, predict, predict_proba, measure_latency).

    `quantum_layer` selects the ablation variant ("vqc", "identity", "frozen") while
    keeping the encoder, optimiser, schedule and readout head identical, so that any
    measured difference is attributable to the quantum layer alone.
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 16,
        num_qubits: int = 4,
        num_layers: int = 2,
        output_dim: int = 2,
        ansatz_type: str = "RealAmplitudes",
        feature_map: str = "AngleEmbedding",
        feature_map_reps: int = 1,
        quantum_layer: str = "vqc",
        output_mode: str = "expectations",
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        epochs: int = 20,
        random_state: int = 42,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.ansatz_type = ansatz_type
        self.feature_map = feature_map
        self.feature_map_reps = feature_map_reps
        self.quantum_layer = quantum_layer
        self.output_mode = output_mode
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = torch.device(device)

        # Reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = _HybridQNNModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_qubits=num_qubits,
            num_layers=num_layers,
            output_dim=output_dim,
            ansatz_type=ansatz_type,
            feature_map=feature_map,
            feature_map_reps=feature_map_reps,
            quantum_layer=quantum_layer,
            output_mode=output_mode,
        ).to(self.device)

        self.criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> "HybridQNNClassifier":
        torch.manual_seed(self.random_state)
        num_epochs = epochs or self.epochs

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_v = torch.tensor(y_val, dtype=torch.long).to(self.device)
        else:
            X_v, y_v = None, None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            train_loss = epoch_loss / len(X_train)

            val_loss_str = ""
            if X_v is not None and y_v is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_v)
                    val_loss = self.criterion(val_logits, y_v).item()
                val_loss_str = f" | Val Loss: {val_loss:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
                print(f"[HybridQNN] Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {train_loss:.4f}{val_loss_str}")

        if X_v is not None and y_v is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            # Evaluate in batches if X is large
            if len(X_t) > 256:
                all_probs = []
                eval_loader = DataLoader(TensorDataset(X_t), batch_size=256, shuffle=False)
                for (batch_x,) in eval_loader:
                    logits = self.model(batch_x)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    all_probs.append(probs)
                return np.vstack(all_probs)
            else:
                logits = self.model(X_t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                return probs

    def measure_latency(self, X_sample: np.ndarray, n_runs: int = 200, warmup_runs: int = 50) -> float:
        """
        Measures single-sample end-to-end inference latency (classical encoding + quantum VQC + readout)
        in milliseconds (ms/sample).
        Includes at least 50 warmup iterations to purge JIT/cache overhead.
        """
        self.model.eval()
        x_tensor = torch.tensor(X_sample.reshape(1, -1), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Warmup (at least 50 passes to purge JIT/cache overhead)
            for _ in range(warmup_runs):
                _ = self.model(x_tensor)

            # Benchmark
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = self.model(x_tensor)
            t1 = time.perf_counter()

        latency_ms = ((t1 - t0) / n_runs) * 1000.0
        return latency_ms

    def get_quantum_expectation_values(self, X: np.ndarray) -> np.ndarray:
        """Extracts the 4-qubit Pauli-Z expectation values for representation analysis."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            _, q_expvals = self.model.extract_latent_and_quantum(X_t)
            return q_expvals.cpu().numpy()

    def get_latent_features(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the 4 bottleneck latent rotation angles z in [-pi, pi]^4.

        These are the data-encoding angles fed to the quantum feature map, and are
        what the Qiskit Aer shot-based backends replay.
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model.encode(X_t).cpu().numpy()

    def get_variational_weights(self) -> np.ndarray:
        """Returns the flattened trainable VQC weights, ordered as Qiskit's ansatz parameters."""
        return self.model.vqc.qnn.weights.detach().cpu().numpy().reshape(-1)

    def classify_from_expectations(self, q_expvals: np.ndarray) -> np.ndarray:
        """
        Applies the trained classical readout head to externally supplied (e.g.
        shot-based Aer or M3-mitigated) Pauli-Z expectation values.
        """
        self.model.eval()
        with torch.no_grad():
            q = torch.tensor(q_expvals, dtype=torch.float32).to(self.device)
            return torch.softmax(self.model.head(q), dim=-1).cpu().numpy()


if __name__ == "__main__":
    print("=" * 70)
    print("Project HQ-IDS-029: Hybrid Quantum-Classical Pipeline Self-Test")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    X_tr = np.random.randn(128, 40).astype(np.float32)
    y_tr = np.random.choice([0, 1], size=128)
    X_val = np.random.randn(32, 40).astype(np.float32)
    y_val = np.random.choice([0, 1], size=32)
    X_test = np.random.randn(32, 40).astype(np.float32)
    y_test = np.random.choice([0, 1], size=32)

    hybrid_model = HybridQNNClassifier(
        input_dim=40,
        hidden_dim=16,
        num_qubits=4,
        num_layers=2,
        epochs=3,
        batch_size=32,
        random_state=42
    )

    print("\n--- Training Hybrid QNN (3 test epochs) ---")
    hybrid_model.fit(X_tr, y_tr, X_val, y_val, epochs=3, verbose=True)

    preds = hybrid_model.predict(X_test)
    probs = hybrid_model.predict_proba(X_test)
    expvals = hybrid_model.get_quantum_expectation_values(X_test[:5])
    latency = hybrid_model.measure_latency(X_test[0], n_runs=20)

    print(f"\n[HybridQNN] Test Predictions: {preds}")
    print(f"[HybridQNN] Test Probabilities Shape: {probs.shape}")
    print(f"[HybridQNN] Sample Quantum Expectation Values:\n{expvals}")
    print(f"[HybridQNN] Single-sample latency: {latency:.4f} ms/sample")

    print("\nHybrid Quantum-Classical pipeline verified successfully!")
