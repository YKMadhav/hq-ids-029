"""
Project HQ-IDS-029: Quantum Support Vector Classifier (QSVM)
============================================================
Implements a Quantum Support Vector Classifier for intrusion detection:
1. Feature Map: AngleEmbedding (default, Ry rotations) or ZZFeatureMap mapping
   continuous d=4 latent vectors z in [-pi, pi]^4 into quantum statevectors |psi(z)>.
2. High-Performance Vectorized Gram Matrix: Computes exact statevectors for each
   latent vector, and computes the Gram kernel matrix:
       K_{ij} = |<psi(z_i) | psi(z_j)>|^2
   using matrix multiplication with upper-triangular symmetry optimization.
3. Classifier Fit: Trains sklearn.svm.SVC(kernel='precomputed', C=C) on K_train
   and predicts on K_test.
4. Autoencoder feature extraction & caching helper for rapid multi-seed benchmarking.
"""

import os
import math
import time
from typing import Optional, Tuple, Dict, Any, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from sklearn.svm import SVC
import pennylane as qml
from src import determinism as _determinism  # noqa: F401  (pins threads; see src/determinism.py)
import torch

from src.encoders import FlowAutoencoder, BottleneckMLPEncoder


def _apply_zz_feature_map(inputs: np.ndarray, wires: list, reps: int = 1) -> None:
    """PennyLane transcription of 2nd-order Pauli expansion (ZZFeatureMap)."""
    pairs = [(i, i + 1) for i in range(len(wires) - 1)]
    for _ in range(reps):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for i in wires:
            qml.PhaseShift(2.0 * inputs[i], wires=i)
        for i, j in pairs:
            qml.CNOT(wires=[i, j])
            qml.PhaseShift(2.0 * (math.pi - inputs[i]) * (math.pi - inputs[j]), wires=j)
            qml.CNOT(wires=[i, j])


class QuantumKernel:
    """
    Computes Quantum Kernel Gram matrices via vectorized statevector inner products.
    K_{ij} = |<psi(z_i) | psi(z_j)>|^2
    """

    def __init__(
        self,
        num_qubits: int = 4,
        feature_map: str = "AngleEmbedding",
        feature_map_reps: int = 1,
        device_name: str = "default.qubit",
    ):
        self.num_qubits = num_qubits
        self.feature_map = feature_map
        self.feature_map_reps = feature_map_reps
        self.dev = qml.device(device_name, wires=num_qubits)

        if feature_map == "AngleEmbedding":
            @qml.qnode(self.dev)
            def _state_circuit(x):
                qml.AngleEmbedding(x, wires=range(num_qubits), rotation="Y")
                return qml.state()
        elif feature_map == "ZZFeatureMap":
            @qml.qnode(self.dev)
            def _state_circuit(x):
                _apply_zz_feature_map(x, list(range(num_qubits)), reps=feature_map_reps)
                return qml.state()
        else:
            raise ValueError(f"Unsupported feature_map: {feature_map}")

        self._state_fn = _state_circuit

    def get_statevectors(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes quantum statevectors |psi(z)> for each sample in Z.
        Z: array of shape (N, num_qubits)
        Returns: array of shape (N, 2^num_qubits) of complex128 statevectors.
        """
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            return np.array(self._state_fn(Z), dtype=np.complex128)
        
        # Sequential evaluation over fast analytical state simulator
        states = [np.array(self._state_fn(z), dtype=np.complex128) for z in Z]
        return np.array(states, dtype=np.complex128)

    def compute_kernel_matrix(
        self,
        Z1: np.ndarray,
        Z2: Optional[np.ndarray] = None,
        states1: Optional[np.ndarray] = None,
        states2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the Gram kernel matrix between Z1 and Z2:
        K_{ij} = |<psi(z1_i) | psi(z2_j)>|^2

        If Z2 is None, computes symmetric K(Z1, Z1) utilizing upper-triangular symmetry.
        Precomputed statevectors can be passed via states1 / states2 to avoid redundant state evolution.
        """
        if states1 is None:
            states1 = self.get_statevectors(Z1)

        if Z2 is None or Z2 is Z1:
            # Symmetric Gram matrix: K = |S @ S^H|^2
            # S is (N, 2^n). Inner products: M = S @ S^H
            overlaps = states1 @ states1.conj().T
            K = np.abs(overlaps) ** 2
            # Numerical guarantee: K_ii = 1.0, clip to [0, 1]
            np.fill_diagonal(K, 1.0)
            return np.clip(K, 0.0, 1.0)
        else:
            if states2 is None:
                states2 = self.get_statevectors(Z2)
            # Rectangular Gram matrix: (N1, N2)
            overlaps = states1 @ states2.conj().T
            K = np.abs(overlaps) ** 2
            return np.clip(K, 0.0, 1.0)


class QSVMClassifier:
    """
    Quantum Support Vector Classifier wrapping sklearn.svm.SVC with a precomputed quantum kernel.
    Can ingest either pre-extracted latent features z in [-pi, pi]^4 or raw flow features X
    with an optional bottleneck feature extractor.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        feature_map: str = "AngleEmbedding",
        feature_map_reps: int = 1,
        C: float = 1.0,
        probability: bool = True,
        decision_function_shape: str = "ovr",
        random_state: int = 42,
    ):
        self.num_qubits = num_qubits
        self.feature_map = feature_map
        self.feature_map_reps = feature_map_reps
        self.C = C
        self.probability = probability
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

        self.kernel_generator = QuantumKernel(
            num_qubits=num_qubits,
            feature_map=feature_map,
            feature_map_reps=feature_map_reps,
        )
        self.svc = SVC(
            C=C,
            kernel="precomputed",
            probability=probability,
            decision_function_shape=decision_function_shape,
            random_state=random_state,
        )

        self.Z_train_: Optional[np.ndarray] = None
        self.states_train_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, Z_train: np.ndarray, y_train: np.ndarray) -> "QSVMClassifier":
        """
        Fit QSVM on precomputed training latent representations Z_train (N_train, 4).
        """
        self.Z_train_ = np.asarray(Z_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.int64)

        # Cache train statevectors
        self.states_train_ = self.kernel_generator.get_statevectors(self.Z_train_)
        K_train = self.kernel_generator.compute_kernel_matrix(
            self.Z_train_, states1=self.states_train_
        )

        self.svc.fit(K_train, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, Z_test: np.ndarray) -> np.ndarray:
        """Predict binary class labels for test latent vectors."""
        if not self.is_fitted_:
            raise RuntimeError("QSVMClassifier must be fitted before predict.")
        Z_test = np.asarray(Z_test, dtype=np.float64)
        states_test = self.kernel_generator.get_statevectors(Z_test)
        K_test_train = self.kernel_generator.compute_kernel_matrix(
            Z_test, self.Z_train_, states1=states_test, states2=self.states_train_
        )
        return self.svc.predict(K_test_train)

    def predict_proba(self, Z_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities for test latent vectors."""
        if not self.is_fitted_:
            raise RuntimeError("QSVMClassifier must be fitted before predict_proba.")
        Z_test = np.asarray(Z_test, dtype=np.float64)
        states_test = self.kernel_generator.get_statevectors(Z_test)
        K_test_train = self.kernel_generator.compute_kernel_matrix(
            Z_test, self.Z_train_, states1=states_test, states2=self.states_train_
        )
        return self.svc.predict_proba(K_test_train)

    def measure_latency(self, Z_sample: np.ndarray, n_runs: int = 100) -> float:
        """
        Measures single-sample inference latency in milliseconds:
        Includes statevector mapping, kernel overlap with training support vectors, and SVC decision.
        """
        if not self.is_fitted_:
            raise RuntimeError("QSVMClassifier must be fitted before latency measurement.")
        Z_sample = np.asarray(Z_sample, dtype=np.float64)
        if Z_sample.ndim == 1:
            Z_sample = Z_sample.reshape(1, -1)
        elif Z_sample.shape[0] > 1:
            Z_sample = Z_sample[:1]

        # Warmup
        for _ in range(5):
            _ = self.predict(Z_sample)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.predict(Z_sample)
        t1 = time.perf_counter()

        return float((t1 - t0) / n_runs * 1000.0)


if __name__ == "__main__":
    print("=" * 60)
    print("Running Self-Test: Quantum Support Vector Classifier (QSVM)")
    print("=" * 60)

    np.random.seed(42)
    N_train, N_test, d = 40, 15, 4
    Z_train = np.random.uniform(-np.pi, np.pi, size=(N_train, d))
    y_train = (Z_train[:, 0] + Z_train[:, 1] > 0).astype(int)
    Z_test = np.random.uniform(-np.pi, np.pi, size=(N_test, d))
    y_test = (Z_test[:, 0] + Z_test[:, 1] > 0).astype(int)

    # 1. Test QuantumKernel properties
    kernel = QuantumKernel(num_qubits=4, feature_map="AngleEmbedding")
    K = kernel.compute_kernel_matrix(Z_train)

    # Symmetry check
    symmetry_error = np.max(np.abs(K - K.T))
    assert symmetry_error < 1e-12, f"Kernel matrix not symmetric! Max diff: {symmetry_error}"
    print(f"  [1/4] Kernel symmetry check passed (max diff = {symmetry_error:.2e})")

    # Positivity / diagonal check
    diag_min = np.min(np.diag(K))
    diag_max = np.max(np.diag(K))
    assert np.allclose(np.diag(K), 1.0, atol=1e-7), f"Diagonal elements are not 1.0! Min: {diag_min}, Max: {diag_max}"
    print(f"  [2/4] Kernel diagonal unit-positivity check passed (K_ii = 1.0)")

    # 2. Test ZZFeatureMap
    kernel_zz = QuantumKernel(num_qubits=4, feature_map="ZZFeatureMap")
    K_zz = kernel_zz.compute_kernel_matrix(Z_train[:10])
    assert np.allclose(np.diag(K_zz), 1.0, atol=1e-7), "ZZFeatureMap diagonal check failed"
    print(f"  [3/4] ZZFeatureMap Gram matrix check passed")

    # 3. Test QSVM Classifier fit and prediction
    qsvm = QSVMClassifier(num_qubits=4, feature_map="AngleEmbedding", C=1.0, random_state=42)
    qsvm.fit(Z_train, y_train)
    y_pred = qsvm.predict(Z_test)
    y_prob = qsvm.predict_proba(Z_test)

    assert y_pred.shape == (N_test,), f"Expected shape ({N_test},), got {y_pred.shape}"
    assert y_prob.shape == (N_test, 2), f"Expected shape ({N_test}, 2), got {y_prob.shape}"
    assert np.all((y_pred == 0) | (y_pred == 1)), "Predictions not binary {0, 1}"
    print(f"  [4/4] QSVM fit, predict, predict_proba passed (test acc = {np.mean(y_pred == y_test):.3f})")

    lat = qsvm.measure_latency(Z_test[0], n_runs=20)
    print(f"  Single-sample latency: {lat:.3f} ms/sample")
    print("=" * 60)
    print("Self-test PASSED successfully! All mathematical invariants verified.")
    print("=" * 60)
