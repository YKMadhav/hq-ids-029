"""Benchmark the leakage-safe, CSV-backed Phase 4 setup-header proxy.

This is not a raw-PCAP packet-prefix benchmark. It reports only k=2 header
observations available during TCP connection setup and makes no k>2 claim.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.biflow_loader import PHASE4_PROTOCOL, get_sequential_biflow_dataloaders
from src.trivial_baselines import MajorityClassBaseline, SingleSplitBaseline, LookupTableBaseline
from src.models.sequential_qnn import (
    BiflowCNN1DClassifier,
    BiflowLSTMClassifier,
    BiflowMLPClassifier,
    SequentialQNNClassifier,
    measure_single_sample_latency,
    profile_5qubit_circuit,
)


SEEDS = [42, 43, 44, 45, 46]
K_PACKETS = 2
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def train_eval_model(
    model: nn.Module, train_loader, test_loader, use_sequence: bool, epochs: int, lr: float
) -> dict[str, float]:
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
    model.eval()
    observed, predicted = [], []
    with torch.no_grad():
        for X_flat, X_seq, labels in test_loader:
            logits = model(X_seq if use_sequence else X_flat)
            observed.extend(labels.numpy())
            predicted.extend(torch.argmax(logits, dim=1).numpy())
    y_true, y_pred = np.asarray(observed), np.asarray(predicted)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
    }


def _confusion_metrics(observed, predicted) -> dict[str, float]:
    y_true, y_pred = np.asarray(observed), np.asarray(predicted)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
    }


def _collect(loader):
    xs, ys = [], []
    for X_flat, _, labels in loader:
        xs.append(X_flat.numpy())
        ys.append(labels.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def run_phase4_benchmark() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    latencies: dict[str, float] = {}
    qiskit_profile = profile_5qubit_circuit(n_layers=2)

    for seed in SEEDS:
        # Seeds are set before the data split and every model is instantiated.
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_loader, _, test_loader, _, metadata = get_sequential_biflow_dataloaders(
            batch_size=64, k_packets=K_PACKETS, random_seed=seed
        )
        X_flat, X_seq, _ = next(iter(test_loader))

        # Reference detectors on the identical 6-number inputs. The lookup table is the
        # empirical-Bayes ceiling for exact-match features: if every neural model lands on
        # it, the task is information-limited by these 5 header fields, not by the model.
        Xtr_np, ytr_np = _collect(train_loader)
        Xte_np, yte_np = _collect(test_loader)
        seen = set(map(tuple, np.round(Xtr_np, 6)))
        overlap = float(np.mean([tuple(r) in seen for r in np.round(Xte_np, 6)]))
        for ref_name, ref in (
            ("MajorityClass(flag-all)", MajorityClassBaseline(random_state=seed)),
            ("SingleSplitStump", SingleSplitBaseline(random_state=seed)),
            ("LookupTable(exact-match)", LookupTableBaseline()),
        ):
            ref.fit(Xtr_np, ytr_np)
            records.append({"model": ref_name, "k_packets": 2, "seed": seed,
                            **_confusion_metrics(yte_np, ref.predict(Xte_np)),
                            "test_vectors_seen_in_train": overlap})
        models = (
            ("ClassicalMLP", BiflowMLPClassifier(k_packets=2), False, 6, 0.008, X_flat[:1]),
            ("CNN1D", BiflowCNN1DClassifier(k_packets=2), True, 6, 0.008, X_seq[:1]),
            ("LSTM", BiflowLSTMClassifier(k_packets=2), True, 6, 0.008, X_seq[:1]),
            ("SequentialQNN", SequentialQNNClassifier(k_packets=2, n_layers=2), False, 6, 0.008, X_flat[:1]),
        )
        for name, model, use_sequence, epochs, lr, latency_input in models:
            metrics = train_eval_model(model, train_loader, test_loader, use_sequence, epochs, lr)
            records.append({"model": name, "k_packets": 2, "seed": seed, **metrics})
            if seed == 42:
                latencies[name] = float(measure_single_sample_latency(model, latency_input))

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_DIR / "phase4_setup_header_results.csv", index=False)
    summary: dict[str, Any] = {
        "metadata": {
            "protocol": PHASE4_PROTOCOL,
            "seed_values": SEEDS,
            "k_packets": 2,
            "claim_boundary": "CSV setup-header proxy; not a chronological PCAP packet-prefix result.",
            "training_source": metadata["training_source"],
            "testing_source": metadata["testing_source"],
            "testing_id_sha256": metadata["testing_id_sha256"],
            "qiskit_hardware_profile": qiskit_profile,
            "training_recipe": "All models: full official training partition, 6 epochs, Adam lr=0.008, weight_decay=1e-4.",
        },
        "models": {},
    }
    for name, subset in results.groupby("model"):
        summary["models"][name] = {
            "accuracy_mean": float(subset["accuracy"].mean()),
            "accuracy_std": float(subset["accuracy"].std()),
            "f1_macro_mean": float(subset["f1_macro"].mean()),
            "recall_mean": float(subset["recall"].mean()),
            "fpr_mean": float(subset["fpr"].mean()),
            "latency_ms_seed42": latencies.get(name),
        }
    (RESULTS_DIR / "phase4_setup_header_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved leakage-safe Phase 4 results for {len(results)} runs to {RESULTS_DIR}")


if __name__ == "__main__":
    run_phase4_benchmark()
