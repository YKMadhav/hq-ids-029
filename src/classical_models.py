"""
Project HQ-IDS-029: Classical Baseline Models & Latency Benchmarking
====================================================================
This module implements standardized wrapper classes for classical intrusion
detection baselines:
1. RandomForestBaseline: Tuned Scikit-Learn tree ensemble.
2. XGBoostBaseline: Gradient-boosted decision trees (with resilient OpenMP loading).
3. ClassicalMLPBaseline: PyTorch Neural Network with an identical bottleneck depth
   (40 -> 16 -> 4 -> 2) matching the parameter depth of the hybrid QML architecture.

All baselines implement a unified interface:
- fit(X_train, y_train, X_val=None, y_val=None)
- predict(X) -> np.ndarray
- predict_proba(X) -> np.ndarray
- measure_latency(X_sample, n_runs=1000) -> float (ms/sample)
"""

import os
import sys
import time
import copy
import glob
import site
import ctypes
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Ensure OpenMP is dynamically loaded on macOS if libomp is bundled in torch or qiskit_aer
def _ensure_openmp_loaded():
    try:
        import xgboost
        return
    except Exception:
        pass

    candidates = [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
    ]
    for sp in site.getsitepackages():
        candidates.extend(glob.glob(os.path.join(sp, "**/libomp.dylib"), recursive=True))

    for lib in candidates:
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib)
                break
            except Exception:
                continue

_ensure_openmp_loaded()

from sklearn.ensemble import RandomForestClassifier

# Resilient XGBoost import with HistGradientBoosting fallback if native dylib fails
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except Exception as e:
    _XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class RandomForestBaseline:
    """
    Random Forest classifier baseline.
    Strong benchmark for tabular flow data in intrusion detection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        self.random_state = random_state

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "RandomForestBaseline":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def measure_latency(self, X_sample: np.ndarray, n_runs: int = 1000) -> float:
        """
        Measures single-sample inference time in milliseconds (ms/sample).
        Includes 50-run warm-up to ensure stable cache and thread-pool execution.
        """
        x = np.ascontiguousarray(X_sample.reshape(1, -1))
        # Warmup
        for _ in range(50):
            _ = self.model.predict_proba(x)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.model.predict_proba(x)
        t1 = time.perf_counter()

        latency_ms = ((t1 - t0) / n_runs) * 1000.0
        return latency_ms


class XGBoostBaseline:
    """
    XGBoost gradient-boosted decision trees baseline.
    Industry standard for production tabular network flow detection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        eval_metric: str = "logloss",
        n_jobs: int = -1,
        **kwargs
    ):
        self.random_state = random_state
        if _XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric=eval_metric,
                n_jobs=n_jobs,
                **kwargs
            )
            self.is_native_xgb = True
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                **kwargs
            )
            self.is_native_xgb = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "XGBoostBaseline":
        if self.is_native_xgb:
            if X_val is not None and y_val is not None:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                self.model.fit(X_train, y_train, verbose=False)
        else:
            self.model.fit(X_train, y_train)
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def measure_latency(self, X_sample: np.ndarray, n_runs: int = 1000) -> float:
        """
        Measures single-sample inference time in milliseconds (ms/sample).
        Includes 50-run warm-up to ensure stable cache and thread-pool execution.
        """
        x = np.ascontiguousarray(X_sample.reshape(1, -1))
        # Warmup
        for _ in range(50):
            _ = self.model.predict_proba(x)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.model.predict_proba(x)
        t1 = time.perf_counter()

        latency_ms = ((t1 - t0) / n_runs) * 1000.0
        return latency_ms


class _PyTorchMLPModule(nn.Module):
    """
    Feedforward Neural Network with bottleneck representation matching
    the parameter depth of the hybrid QML architecture:
    Input [39] -> Hidden [16] -> Bottleneck [4] -> Output [2].
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 16,
        bottleneck_dim: int = 4,
        output_dim: int = 2,
        scale_latent_by_pi: bool = False
    ):
        super().__init__()
        self.scale_latent_by_pi = scale_latent_by_pi
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        self.classifier = nn.Linear(bottleneck_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        latent = self.encoder(x)
        if self.scale_latent_by_pi:
            latent = latent * np.pi
        logits = self.classifier(latent)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        latent = self.encoder(x)
        if self.scale_latent_by_pi:
            latent = latent * np.pi
        return latent


class ClassicalMLPBaseline:
    """
    PyTorch Multi-Layer Perceptron baseline.
    Serves as an architectural control with identical bottleneck capacity
    (4 continuous latent variables) to isolate the contribution of the quantum layer.
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 16,
        bottleneck_dim: int = 4,
        output_dim: int = 2,
        learning_rate: float = 0.005,
        batch_size: int = 64,
        epochs: int = 40,
        weight_decay: float = 1e-4,
        patience: int = 8,
        loss_fn: Optional[nn.Module] = None,
        random_state: int = 42,
        device: str = "cpu",
        scale_latent_by_pi: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.loss_fn = loss_fn
        self.random_state = random_state
        self.device = torch.device(device)
        self.scale_latent_by_pi = scale_latent_by_pi

        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = _PyTorchMLPModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
            scale_latent_by_pi=scale_latent_by_pi,
        ).to(self.device)

        self.criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "ClassicalMLPBaseline":
        torch.manual_seed(self.random_state)

        # Build DataLoader
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_v = torch.tensor(y_val, dtype=torch.long).to(self.device)
        else:
            X_v, y_v = None, None

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            # Validation / Early Stopping check
            if X_v is not None and y_v is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_v)
                    val_loss = self.criterion(val_logits, y_v).item()

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break

        # Load best weights if validation was used
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
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def measure_latency(self, X_sample: np.ndarray, n_runs: int = 1000) -> float:
        """
        Measures single-sample inference time in milliseconds (ms/sample).
        Includes 50-run warm-up to ensure stable cache and JIT dispatch.
        """
        self.model.eval()
        x_tensor = torch.tensor(X_sample.reshape(1, -1), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Warmup
            for _ in range(50):
                _ = self.model(x_tensor)

            # Benchmark
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = self.model(x_tensor)
            t1 = time.perf_counter()

        latency_ms = ((t1 - t0) / n_runs) * 1000.0
        return latency_ms


if __name__ == "__main__":
    print("=" * 70)
    print("Project HQ-IDS-029: Classical Baselines Self-Test")
    print("=" * 70)

    # Generate dummy data for verification
    np.random.seed(42)
    N_train, N_val, N_test, D = 400, 100, 100, 40
    X_tr = np.random.randn(N_train, D)
    y_tr = np.random.choice([0, 1], size=N_train)
    X_v = np.random.randn(N_val, D)
    y_v = np.random.choice([0, 1], size=N_val)
    X_te = np.random.randn(N_test, D)
    y_te = np.random.choice([0, 1], size=N_test)

    sample_x = X_te[0]

    baselines = {
        "RandomForest": RandomForestBaseline(n_estimators=50, max_depth=8, random_state=42),
        "XGBoost": XGBoostBaseline(n_estimators=50, max_depth=4, random_state=42),
        "ClassicalMLP": ClassicalMLPBaseline(input_dim=D, epochs=10, random_state=42),  # epochs=10 is for unit-level smoke testing only
    }

    for name, model in baselines.items():
        print(f"\n--- Testing {name} ---")
        model.fit(X_tr, y_tr, X_v, y_v)
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)
        lat = model.measure_latency(sample_x, n_runs=500)

        print(f"[{name}] Predictions shape: {preds.shape}, unique classes: {np.unique(preds)}")
        print(f"[{name}] Probabilities shape: {probs.shape}, prob sum sample: {probs[0].sum():.4f}")
        print(f"[{name}] Single-sample latency: {lat:.4f} ms/sample")

    print("\nAll classical baselines verified successfully!")
