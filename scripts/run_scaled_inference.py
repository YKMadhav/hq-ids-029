"""Run Phase 4.5 inference once over the official UNSW-NB15 test partition."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.biflow_loader import PHASE4_PROTOCOL
from src.models.sequential_qnn import (
    BiflowCNN1DClassifier,
    BiflowLSTMClassifier,
    BiflowMLPClassifier,
    SequentialQNNClassifier,
)
from src.scaled_loader import get_scaled_inference_dataloaders


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
MANIFEST_PATH = CHECKPOINT_DIR / "phase4_setup_manifest.json"
RESULTS_DIR = REPO_ROOT / "results"


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint manifest: {MANIFEST_PATH}. "
            "Train the leakage-safe Phase 4 k=2 checkpoints first."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PHASE4_PROTOCOL or manifest.get("k_packets") != 2:
        raise ValueError("Rejected checkpoint manifest with incompatible Phase 4 provenance.")
    return manifest


def load_model(model_cls, filename: str, **kwargs) -> nn.Module:
    path = CHECKPOINT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = model_cls(k_packets=2, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def run_model_inference(model: nn.Module, loader, use_sequence: bool) -> tuple[np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for X_flat, X_seq, _ in loader:
            logits = model(X_seq if use_sequence else X_flat)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    elapsed = time.perf_counter() - start
    return np.concatenate(predictions), elapsed


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, elapsed_seconds: float) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_samples = len(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "total_inference_time_sec": float(elapsed_seconds),
        "latency_per_sample_ms": float(elapsed_seconds * 1000.0 / n_samples),
        "throughput_flows_per_sec": float(n_samples / max(elapsed_seconds, 1e-6)),
    }


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    manifest = load_manifest()
    loader, y_true, metadata = get_scaled_inference_dataloaders(batch_size=512, random_seed=42)
    print(
        f"Evaluating {metadata['n_samples']} unique official test flows exactly once "
        f"(attacks={metadata['threat_count']}, benign={metadata['benign_count']})."
    )

    model_specs = (
        ("ClassicalMLP", BiflowMLPClassifier, False, {}),
        ("CNN1D", BiflowCNN1DClassifier, True, {}),
        ("LSTM", BiflowLSTMClassifier, True, {}),
        ("SequentialQNN", SequentialQNNClassifier, False, {"n_layers": 2}),
    )
    summary: dict[str, Any] = {
        "metadata": {
            "protocol": PHASE4_PROTOCOL,
            "seed": 42,
            "n_samples": metadata["n_samples"],
            "unique_source_flows": metadata["unique_source_flows"],
            "source_flow_id_sha256": metadata["source_flow_id_sha256"],
            "testing_source": metadata["testing_source"],
            "threat_count": metadata["threat_count"],
            "benign_count": metadata["benign_count"],
            "threat_ratio": metadata["threat_ratio"],
            "horizons": [2],
            "checkpoint_manifest": str(MANIFEST_PATH),
            "scaler_metadata": metadata["scaler_metadata"],
        },
        "setup_header_proxy_k2": {},
    }
    predictions: dict[str, Any] = {
        "source_flow_id": metadata["source_flow_ids"],
        "true_label": y_true,
    }
    for name, model_cls, use_sequence, kwargs in model_specs:
        model = load_model(model_cls, manifest["checkpoints"][name], **kwargs)
        y_pred, elapsed = run_model_inference(model, loader, use_sequence)
        summary["setup_header_proxy_k2"][name] = compute_metrics(y_true, y_pred, elapsed)
        predictions[f"pred_{name.lower()}_k2"] = y_pred
        print(f"{name}: accuracy={summary['setup_header_proxy_k2'][name]['accuracy']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "scaled_inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(predictions).to_csv(RESULTS_DIR / "scaled_predictions.csv", index=False)
    print(f"Wrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
