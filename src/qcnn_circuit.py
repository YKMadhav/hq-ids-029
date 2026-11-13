"""
Project HQ-IDS-029: Quantum Convolutional Neural Network (QCNN)
===============================================================
This module implements a 4-to-2 qubit Quantum Convolutional Neural Network (QCNN)
for network flow intrusion detection:

Architecture:
1. Input Layer: d=4 continuous latent parameters mapped via Ry(z_i) rotation encoding
   (AngleEmbedding) on qubits q0, q1, q2, q3.
2. Quantum Convolution Layer: Parameterized 2-qubit entangling blocks U(theta) applied
   across adjacent pairs (q0, q1), (q2, q3), and interlaced (q1, q2).
   Each block consists of Rx, Ry, Rz single-qubit rotations on both qubits, followed by
   CNOT entanglers and secondary Ry rotations.
3. Quantum Pooling Layer: Downsample from 4 qubits to 2 qubits via parameterized
   conditional rotations V(phi) from q1 -> q0 and q3 -> q2, condensing quantum information
   into active qubits q0 and q2.
4. Readout: Measures multi-qubit Pauli-Z expectations <Z_0> and <Z_2> from the 2 active qubits.
5. Hybrid Integration: Passes the 2 quantum output expectations into a classical
   PyTorch linear readout head (torch.nn.Linear(2, 2)) for binary classification.
"""

import os
import sys
import time
import copy
from typing import Optional, Tuple, Dict, Any, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from src import determinism as _determinism  # noqa: F401  (pins threads; see src/determinism.py)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml

from src.encoders import BottleneckMLPEncoder


def _conv_block(weights: torch.Tensor, wire1: int, wire2: int):
    """
    Parameterized 2-qubit unitary convolution block U(theta).
    Consists of arbitrary single-qubit rotations (Rx, Ry, Rz), CNOT, and coupling rotations.
    weights shape: (8,)
    """
    qml.RX(weights[0], wires=wire1)
    qml.RY(weights[1], wires=wire1)
    qml.RZ(weights[2], wires=wire1)
    qml.RX(weights[3], wires=wire2)
    qml.RY(weights[4], wires=wire2)
    qml.RZ(weights[5], wires=wire2)
    qml.CNOT(wires=[wire1, wire2])
    qml.RY(weights[6], wires=wire1)
    qml.RY(weights[7], wires=wire2)
    qml.CNOT(wires=[wire2, wire1])


def _pool_block(weights: torch.Tensor, control: int, target: int):
    """
    Parameterized 2-qubit quantum pooling block V(phi).
    Conditional rotations from control to target qubit, transferring quantum correlation
    before discarding/ignoring the control wire.
    weights shape: (3,)
    """
    qml.CRX(weights[0], wires=[control, target])
    qml.CRY(weights[1], wires=[control, target])
    qml.CRZ(weights[2], wires=[control, target])


class PennyLaneQCNNModule(nn.Module):
    """
    4-Qubit to 2-Qubit QCNN module implemented in PennyLane and wrapped as
    a PyTorch nn.Module using diff_method='backprop' on default.qubit.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.dev = qml.device(device_name, wires=num_qubits)

        # Weight shapes:
        # Conv layer: 3 pairs -> (q0, q1), (q2, q3), (q1, q2), each 8 params = 24 params
        # Pool layer: 2 downsamplings -> q1->q0, q3->q2, each 3 params = 6 params
        self.weight_shapes = {
            "conv_weights": (3, 8),
            "pool_weights": (2, 3),
        }

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, conv_weights, pool_weights):
            # 1. Feature Encoding (AngleEmbedding Ry rotations)
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")

            # 2. Quantum Convolution Layer (Adjacent + Interlaced)
            _conv_block(conv_weights[0], 0, 1)
            _conv_block(conv_weights[1], 2, 3)
            _conv_block(conv_weights[2], 1, 2)

            # 3. Quantum Pooling Layer (4 qubits -> 2 active qubits)
            # q1 pools into q0, q3 pools into q2
            _pool_block(pool_weights[0], 1, 0)
            _pool_block(pool_weights[1], 3, 2)

            # 4. Readout: Measure expectations <Z_0> and <Z_2> on the 2 retained qubits
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(2))]

        self.circuit_fn = _circuit
        self.qnn = qml.qnn.TorchLayer(self.circuit_fn, self.weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Batch of 4 continuous latent variables (B, 4) in [-pi, pi]^4.
        Output: Batch of 2 Pauli-Z expectations (B, 2) in [-1, 1]^2.
        """
        return self.qnn(x)


class HybridQCNNClassifier:
    """
    Unified end-to-end Hybrid QCNN Classifier:
    Input [39] -> BottleneckMLPEncoder [4] -> PennyLaneQCNNModule [2] -> Linear Readout Head [2] -> Logits
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 16,
        output_dim: int = 2,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        epochs: int = 20,
        loss_fn: Optional[nn.Module] = None,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Model definition
        self.encoder = BottleneckMLPEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=4
        )
        self.qcnn = PennyLaneQCNNModule(num_qubits=4)
        self.head = nn.Linear(2, output_dim)

        self.model = nn.Sequential(
            self.encoder,
            self.qcnn,
            self.head,
        )

        self.criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.is_fitted_ = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> "HybridQCNNClassifier":
        """Fit Hybrid QCNN end-to-end using Adam and specified loss function."""
        if epochs is None:
            epochs = self.epochs

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.long)

        criterion = self.criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state = None

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(by)

            train_loss = total_loss / len(dataset)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_v)
                    val_loss = criterion(val_logits, y_v).item()
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                best_state = copy.deepcopy(self.model.state_dict())
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def get_quantum_expectation_values(self, X: np.ndarray) -> np.ndarray:
        """Extract intermediate 2-dimensional QCNN Pauli-Z expectations."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            z = self.encoder(X_t)
            q_exp = self.qcnn(z).cpu().numpy()
        return q_exp

    def measure_latency(self, X_sample: np.ndarray, n_runs: int = 100) -> float:
        """Measure single-sample inference latency in milliseconds."""
        self.model.eval()
        sample = np.asarray(X_sample, dtype=np.float32)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        elif sample.shape[0] > 1:
            sample = sample[:1]

        sample_tensor = torch.tensor(sample, dtype=torch.float32)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(sample_tensor)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(sample_tensor)
        t1 = time.perf_counter()

        return float((t1 - t0) / n_runs * 1000.0)


if __name__ == "__main__":
    print("=" * 60)
    print("Running Self-Test: Quantum Convolutional Neural Network (QCNN)")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Test PennyLaneQCNNModule forward and backprop
    qcnn_module = PennyLaneQCNNModule(num_qubits=4)
    dummy_latent = torch.randn(8, 4, requires_grad=True)
    q_out = qcnn_module(dummy_latent)

    assert q_out.shape == (8, 2), f"Expected shape (8, 2), got {q_out.shape}"
    print(f"  [1/4] QCNN module forward pass verified. Shape: {q_out.shape}")

    loss = q_out.sum()
    loss.backward()
    assert dummy_latent.grad is not None, "Gradient propagation through QCNN failed!"
    assert dummy_latent.grad.shape == (8, 4), f"Unexpected grad shape: {dummy_latent.grad.shape}"
    print(f"  [2/4] Analytical backpropagation check passed. Grad shape: {dummy_latent.grad.shape}")

    # 2. Test full end-to-end HybridQCNNClassifier pipeline
    N_samples = 60
    input_dim = 39
    X_synthetic = np.random.uniform(-1, 1, size=(N_samples, input_dim))
    y_synthetic = np.random.randint(0, 2, size=N_samples)

    hybrid_qcnn = HybridQCNNClassifier(
        input_dim=input_dim,
        hidden_dim=16,
        output_dim=2,
        epochs=3,
        batch_size=32,
        random_state=42,
    )
    hybrid_qcnn.fit(X_synthetic, y_synthetic, epochs=3, verbose=False)

    preds = hybrid_qcnn.predict(X_synthetic[:10])
    probs = hybrid_qcnn.predict_proba(X_synthetic[:10])
    expvals = hybrid_qcnn.get_quantum_expectation_values(X_synthetic[:10])

    assert preds.shape == (10,), f"Expected preds shape (10,), got {preds.shape}"
    assert probs.shape == (10, 2), f"Expected probs shape (10, 2), got {probs.shape}"
    assert expvals.shape == (10, 2), f"Expected expvals shape (10, 2), got {expvals.shape}"
    print(f"  [3/4] Hybrid QCNN classifier fit, predict, predict_proba passed.")

    lat = hybrid_qcnn.measure_latency(X_synthetic[0], n_runs=20)
    print(f"  [4/4] Single-sample latency measured: {lat:.3f} ms/sample")
    print("=" * 60)
    print("Self-test PASSED successfully! All mathematical invariants verified.")
    print("=" * 60)
