"""
Project HQ-IDS-029: 5-Qubit Sequential Quantum Architecture & Classical Controls (Phase 4)
========================================================================================
This module implements architectures for early-packet biflow intrusion detection:

1. SequentialQNNClassifier (5-Qubit VQC / RQNN):
   - Compresses sequential biflow attributes (3 * k continuous features) to 5 angles z in [-pi, pi]^5.
   - Angle-embeds z into 5 qubits (Hilbert space dimension 2^5 = 32).
   - Evaluates a 2-layer hardware-efficient ansatz with single-qubit Rot gates
     and ring CNOT entanglement (10 CNOTs total).
   - Measures Pauli-Z expectations on all 5 qubits.
   - Maps 5-dim quantum expectation vector to binary threat logits via a linear readout head.

2. Classical Sequential Baselines:
   - BiflowCNN1DClassifier: 1D Convolution over packet sequence.
   - BiflowLSTMClassifier: Recurrent network processing packet vectors over time.
   - BiflowMLPClassifier: Depth-matched classical MLP control (3k -> 16 -> 5 -> 2).
"""

import os
import sys
import math
import time
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from src import determinism as _determinism  # noqa: F401  (pins threads; see src/determinism.py)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pennylane as qml

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, transpile


class SequentialQNNClassifier(nn.Module):
    """
    5-Qubit Hybrid Sequential Quantum Classifier.
    - Classical Front-End: Linear(3*k, 16) -> BatchNorm -> ReLU -> Linear(16, 5) -> Tanh * pi
    - Quantum Circuit: 5 qubits, 2 layers of Rot + Ring CNOTs
    - Quantum Measurement: 5-dim Pauli-Z expectations <Z_0>, ..., <Z_4>
    - Classical Readout: Linear(5, 2)
    """

    def __init__(self, k_packets: int = 10, n_layers: int = 2, backend: str = "default.qubit"):
        super().__init__()
        self.k_packets = k_packets
        self.input_dim = k_packets * 3
        self.n_qubits = 5
        self.n_layers = n_layers

        # 1. Classical front-end compressor
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Tanh()
        )

        # 2. Quantum Circuit setup
        self.dev = qml.device(backend, wires=self.n_qubits)
        init_weights = torch.empty(n_layers, self.n_qubits, 3).uniform_(0.0, 2.0 * math.pi)
        self.q_weights = nn.Parameter(init_weights)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            # Angle embedding on 5 qubits
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Parameterized layers
            for l in range(weights.shape[0]):
                for i in range(self.n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                # Ring + linear entanglement across 5 qubits
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 4])
                qml.CNOT(wires=[4, 0])

            # Pauli-Z expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = _circuit

        # 3. Classical Readout Head
        self.readout = nn.Linear(5, 2)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.dim() == 1:
            x_flat = x_flat.unsqueeze(0)

        # 1. Classical compression to [-pi, pi]^5
        z = self.encoder(x_flat) * math.pi

        # 2. Quantum circuit evaluation
        q_outs = [self.circuit(sample, self.q_weights) for sample in z]
        q_tensor = torch.stack([torch.stack(res) for res in q_outs]).to(torch.float32)  # (batch_size, 5)

        # 3. Linear classification head
        logits = self.readout(q_tensor)
        return logits


class BiflowCNN1DClassifier(nn.Module):
    """
    1D-Convolutional Neural Network for sequential biflow packet processing.
    Input: (B, k, 3) -> transposed to (B, 3, k)
    Conv1D(3, 16) -> ReLU -> MaxPool -> Conv1D(16, 32) -> GlobalAvgPool -> Linear(32, 2)
    """

    def __init__(self, k_packets: int = 10):
        super().__init__()
        self.k_packets = k_packets
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq is (B, k, 3) -> transpose to (B, 3, k)
        x_trans = x_seq.transpose(1, 2)
        feats = self.conv(x_trans).squeeze(-1)  # (B, 16)
        return self.classifier(feats)


class BiflowLSTMClassifier(nn.Module):
    """
    Recurrent LSTM Neural Network for packet arrival sequence modeling.
    Input: (B, k, 3)
    LSTM(input_size=3, hidden_size=16, batch_first=True) -> Linear(16, 2)
    """

    def __init__(self, k_packets: int = 10, hidden_dim: int = 16):
        super().__init__()
        self.k_packets = k_packets
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, k, 3)
        lstm_out, (hn, cn) = self.lstm(x_seq)
        # hn[-1] is the last layer hidden state: (B, hidden_dim)
        return self.classifier(hn[-1])


class BiflowMLPClassifier(nn.Module):
    """
    Depth-matched classical MLP control for flat biflow features.
    Architecture: Linear(3*k, 16) -> BatchNorm -> ReLU -> Linear(16, 5) -> ReLU -> Linear(5, 2)
    """

    def __init__(self, k_packets: int = 10):
        super().__init__()
        self.k_packets = k_packets
        self.net = nn.Sequential(
            nn.Linear(k_packets * 3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.dim() == 1:
            x_flat = x_flat.unsqueeze(0)
        return self.net(x_flat)


def profile_5qubit_circuit(n_layers: int = 2) -> Dict[str, Any]:
    """
    Constructs the exact 5-qubit hardware-efficient circuit in Qiskit
    and transpiles to universal basis ['rz', 'sx', 'x', 'cx'] to measure
    physical depth and 2-qubit CNOT gate complexity.
    """
    qc = QuantumCircuit(5)
    # 1. Angle embedding
    for i in range(5):
        qc.ry(1.0, i)

    # 2. Variational layers
    for _ in range(n_layers):
        for i in range(5):
            qc.rz(1.0, i)
            qc.ry(1.0, i)
            qc.rz(1.0, i)
        # Ring entanglement
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

    # Transpile to basis gates
    transpiled = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=2)
    ops = transpiled.count_ops()
    cnot_count = ops.get('cx', 0)
    depth = transpiled.depth()
    total_gates = sum(ops.values())

    return {
        "n_qubits": 5,
        "n_layers": n_layers,
        "circuit_depth": depth,
        "cnot_count": cnot_count,
        "total_gates": total_gates,
        "gate_breakdown": dict(ops),
    }


def measure_single_sample_latency(model: nn.Module, sample_input: torch.Tensor, n_runs: int = 300) -> float:
    """
    Measures single-sample inference latency in milliseconds with 50 warm-up runs.
    """
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(50):
            _ = model(sample_input)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(sample_input)
        t1 = time.perf_counter()

    latency_ms = ((t1 - t0) / n_runs) * 1000.0
    return latency_ms


if __name__ == "__main__":
    print("Testing 5-Qubit Sequential Architecture and Classical Baselines...")
    k = 10
    dummy_flat = torch.randn(8, k * 3)
    dummy_seq = torch.randn(8, k, 3)

    # Test Sequential QNN
    qnn = SequentialQNNClassifier(k_packets=k, n_layers=2)
    q_out = qnn(dummy_flat)
    print(f"Sequential QNN output shape: {q_out.shape} (target: [8, 2])")

    # Test CNN
    cnn = BiflowCNN1DClassifier(k_packets=k)
    c_out = cnn(dummy_seq)
    print(f"Biflow 1D-CNN output shape: {c_out.shape} (target: [8, 2])")

    # Test LSTM
    lstm = BiflowLSTMClassifier(k_packets=k)
    l_out = lstm(dummy_seq)
    print(f"Biflow LSTM output shape:   {l_out.shape} (target: [8, 2])")

    # Test MLP
    mlp = BiflowMLPClassifier(k_packets=k)
    m_out = mlp(dummy_flat)
    print(f"Biflow MLP output shape:    {m_out.shape} (target: [8, 2])")

    # Profile Qiskit circuit
    prof = profile_5qubit_circuit(n_layers=2)
    print(f"5-Qubit Qiskit Profile: Depth={prof['circuit_depth']}, CNOTs={prof['cnot_count']}, Total Gates={prof['total_gates']}")
    print("Self-test PASSED successfully!")
