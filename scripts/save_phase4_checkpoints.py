"""Train leakage-safe Phase 4 setup-header checkpoints.

This script trains only the conservative k=2 CSV setup-header proxy. It does
not train or overwrite models that purport to observe longer packet prefixes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.biflow_loader import PHASE4_PROTOCOL, get_sequential_biflow_dataloaders
from src.models.sequential_qnn import (
    BiflowCNN1DClassifier,
    BiflowLSTMClassifier,
    BiflowMLPClassifier,
    SequentialQNNClassifier,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
SCALER_BUNDLE = CHECKPOINT_DIR / "phase4_setup_scaler.joblib"
MANIFEST_PATH = CHECKPOINT_DIR / "phase4_setup_manifest.json"
CHECKPOINT_NAMES = {
    "ClassicalMLP": "phase4_setup_classical_mlp_k2.pt",
    "CNN1D": "phase4_setup_cnn1d_k2.pt",
    "LSTM": "phase4_setup_lstm_k2.pt",
    "SequentialQNN": "phase4_setup_sequential_qnn_k2.pt",
}


def _train(model: nn.Module, train_loader, use_sequence: bool, epochs: int, lr: float) -> None:
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for X_flat, X_seq, labels in train_loader:
            optimiser.zero_grad()
            logits = model(X_seq if use_sequence else X_flat)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()


def train_and_save_checkpoints() -> None:
    # Set both framework seeds before loader construction, splitting, or model creation.
    np.random.seed(42)
    torch.manual_seed(42)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, _, _, scaler, metadata = get_sequential_biflow_dataloaders(
        batch_size=64, k_packets=2, random_seed=42
    )

    scaler_bundle = {"scaler": scaler, "metadata": metadata}
    joblib.dump(scaler_bundle, SCALER_BUNDLE)
    print(f"Saved provenance-tagged frozen scaler to {SCALER_BUNDLE}")

    models = (
        ("ClassicalMLP", BiflowMLPClassifier(k_packets=2), False, 6, 0.008),
        ("CNN1D", BiflowCNN1DClassifier(k_packets=2), True, 6, 0.008),
        ("LSTM", BiflowLSTMClassifier(k_packets=2), True, 6, 0.008),
    )
    for name, model, use_sequence, epochs, learning_rate in models:
        print(f"Training {name} on the k=2 setup-header proxy...")
        _train(model, train_loader, use_sequence, epochs, learning_rate)
        path = CHECKPOINT_DIR / CHECKPOINT_NAMES[name]
        torch.save(model.state_dict(), path)
        print(f"Saved {path}")

    qnn = SequentialQNNClassifier(k_packets=2, n_layers=2)
    print("Training SequentialQNN on the k=2 setup-header proxy...")
    # Match the public benchmark's full-training-partition recipe; using a
    # private 1,200-row subset made checkpoint and table provenance ambiguous.
    _train(qnn, train_loader, False, 6, 0.008)
    qnn_path = CHECKPOINT_DIR / CHECKPOINT_NAMES["SequentialQNN"]
    torch.save(qnn.state_dict(), qnn_path)
    print(f"Saved {qnn_path}")

    manifest = {
        "protocol": PHASE4_PROTOCOL,
        "k_packets": 2,
        "seed": 42,
        "scaler_bundle": str(SCALER_BUNDLE),
        "checkpoints": CHECKPOINT_NAMES,
        "scaler_metadata": metadata,
        "training_recipe": {"epochs": 6, "learning_rate": 0.008, "weight_decay": 0.0001},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint provenance manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    train_and_save_checkpoints()
