"""
Project HQ-IDS-029: Step 2 - Hybrid Quantum Neural Network Simulation
======================================================================
Trains and evaluates the 4-qubit Hybrid Quantum Neural Network (HQ-IDS) on the
real UNSW-NB15 flow sample.

Architecture:

    x in R^39  --encoder-->  z in [-pi, pi]^4  --VQC-->  <Z> in [-1, 1]^4  --head-->  yhat in R^2

  * encoder : Dense(16) + BatchNorm + ReLU -> Dense(4) + Tanh * pi
  * VQC     : 4 qubits, ZZFeatureMap(4, reps=1) + RealAmplitudes(4, reps=2, linear)
              profiled by Qiskit at depth 18, 12 CNOT, 16 parameters
  * head    : Dense(4 -> 2)

Protocol: 5 seeds [42..46], deterministic stratified 70/10/20 splits, 20 epochs
per seed, Adam, analytical backpropagation (diff_method="backprop").

Outputs:
  results/hybrid_all_seeds.csv  one row per seed
  results/metrics.csv           appends the aggregated HybridQNN row
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    RESULTS_DIR,
    compute_metrics,
    load_config,
    load_flow_data,
    log_decision,
    log_event,
    preprocess_split,
    stratified_split,
)
from src.hybrid_pipeline import HybridQNNClassifier  # noqa: E402
from src.quantum_circuits import (  # noqa: E402
    QiskitVQCBuilder,
    verify_pennylane_qiskit_equivalence,
)

METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "fpr",
    "latency_ms",
]


def run_hybrid_experiment(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    test_size = config.get("dataset", {}).get("test_size", 0.2)
    val_size = config.get("dataset", {}).get("val_size", 0.1)
    latency_runs = config.get("baselines", {}).get("latency_runs", 1000)

    qc_cfg = config.get("quantum_circuit", {})
    num_qubits = qc_cfg.get("num_qubits", 4)
    reps = qc_cfg.get("reps", 2)
    ansatz_type = qc_cfg.get("ansatz", "RealAmplitudes")
    feature_map = qc_cfg.get("feature_map", "ZZFeatureMap")
    feature_map_reps = qc_cfg.get("feature_map_reps", 1)
    entanglement = qc_cfg.get("entanglement", "linear")

    hyb_cfg = config.get("hybrid_classifier", {})
    lr = hyb_cfg.get("learning_rate", 0.01)
    batch_size = hyb_cfg.get("batch_size", 64)
    epochs = hyb_cfg.get("epochs", 20)

    # 1. Profile circuit complexity with Qiskit, for the trained circuit and for the
    #    ZZFeatureMap hardware-reference configuration.
    print(f"[Circuit Profiling] Building {feature_map}({num_qubits}) + "
          f"RealAmplitudes({num_qubits}, reps={reps}, {entanglement})")
    builder = QiskitVQCBuilder(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
        feature_map_type=feature_map,
    )
    circuit_metrics = builder.get_circuit_metrics()

    reference = QiskitVQCBuilder(
        num_qubits=num_qubits, reps=reps, entanglement=entanglement,
        feature_map_type="ZZFeatureMap",
    ).get_circuit_metrics()
    reference_metrics = {
        "depth": reference["depth"],
        "cnot_count": reference["cnot_count"],
        "num_parameters": reference["num_parameters"],
    }

    # Cross-backend parity: the PennyLane training circuit must equal the Qiskit
    # circuit that the Step 3 Aer NISQ simulation replays.
    deviation = verify_pennylane_qiskit_equivalence(
        num_qubits=num_qubits, reps=reps, n_samples=8, seed=42, feature_map=feature_map
    )
    print(f"[Circuit Profiling] Trained circuit ({feature_map}): depth {circuit_metrics['depth']} | "
          f"CNOTs {circuit_metrics['cnot_count']} | params {circuit_metrics['num_parameters']} "
          f"({circuit_metrics['feature_map_parameters']} data + {circuit_metrics['ansatz_parameters']} "
          f"variational) | gates {circuit_metrics['ops']}")
    print(f"[Circuit Profiling] Hardware reference (ZZFeatureMap): depth {reference_metrics['depth']} | "
          f"CNOTs {reference_metrics['cnot_count']} | params {reference_metrics['num_parameters']}")
    print(f"[Circuit Profiling] PennyLane/Qiskit <Z_i> parity: {deviation:.3e}")
    if deviation > 1e-5:
        raise AssertionError(
            f"PennyLane and Qiskit circuits disagree by {deviation:.3e}; Aer replay would be invalid."
        )

    for label, metrics, expected_cfg in (
        ("trained", circuit_metrics, qc_cfg.get("trained_circuit", {})),
        ("reference", reference_metrics, qc_cfg.get("hardware_reference_circuit", {})),
    ):
        for key, expected in expected_cfg.items():
            if expected is not None and metrics[key] != expected:
                raise AssertionError(
                    f"Circuit profiling mismatch on the {label} circuit: "
                    f"{key}={metrics[key]}, expected {expected}."
                )

    # 2. Ingest the real UNSW-NB15 flows.
    raw_df = load_flow_data(config)
    print(f"[Dataset] Using all {len(raw_df)} real flows "
          f"({raw_df['label'].mean():.2%} malicious), full 70/10/20 partitioning per seed")

    detailed_records: List[Dict[str, Any]] = []

    for seed_idx, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 66)
        print(f"  HYBRID QNN {seed_idx}/{len(seeds)}  (seed={seed})")
        print("=" * 66)

        train_df, val_df, test_df = stratified_split(raw_df, seed, test_size, val_size)
        _, arrays = preprocess_split(train_df, val_df, test_df, config)
        X_train, y_train = arrays["train"]
        X_val, y_val = arrays["val"]
        X_test, y_test = arrays["test"]

        model = HybridQNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=config.get("classical_encoder", {}).get("hidden_dim", 16),
            num_qubits=num_qubits,
            num_layers=reps,
            output_dim=2,
            ansatz_type=ansatz_type,
            feature_map=feature_map,
            feature_map_reps=feature_map_reps,
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs,
            random_state=seed,
        )

        start = time.perf_counter()
        model.fit(X_train, y_train, X_val, y_val, epochs=epochs, verbose=True)
        train_time = time.perf_counter() - start

        y_prob = model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        latency_ms = model.measure_latency(X_test[0], n_runs=latency_runs)

        print(f"  [seed {seed}] Acc {metrics['accuracy']:.4f} | F1-Macro {metrics['f1_macro']:.4f} | "
              f"ROC-AUC {metrics['roc_auc']:.4f} | FPR {metrics['fpr']:.4f} | "
              f"Latency {latency_ms:.4f} ms | Train {train_time:.2f}s")

        detailed_records.append({
            "seed": seed,
            "model": "HybridQNN",
            "n_features": X_train.shape[1],
            **metrics,
            "feature_map": feature_map,
            "circuit_depth": circuit_metrics["depth"],
            "cnot_count": circuit_metrics["cnot_count"],
            "num_parameters": circuit_metrics["num_parameters"],
            "reference_depth": reference_metrics["depth"],
            "reference_cnot_count": reference_metrics["cnot_count"],
            "latency_ms": float(latency_ms),
            "train_time_sec": float(train_time),
        })

    detailed_df = pd.DataFrame(detailed_records)

    summary_row: Dict[str, Any] = {
        "model": "HybridQNN",
        "n_features": int(detailed_df["n_features"].iloc[0]),
        "feature_map": feature_map,
        "circuit_depth": circuit_metrics["depth"],
        "cnot_count": circuit_metrics["cnot_count"],
        "num_parameters": circuit_metrics["num_parameters"],
        "reference_depth": reference_metrics["depth"],
        "reference_cnot_count": reference_metrics["cnot_count"],
    }
    for column in METRIC_COLUMNS:
        summary_row[f"{column}_mean"] = float(detailed_df[column].mean())
        summary_row[f"{column}_std"] = float(detailed_df[column].std(ddof=1))
    hybrid_summary_df = pd.DataFrame([summary_row])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    detailed_path = os.path.join(RESULTS_DIR, "hybrid_all_seeds.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"\n[Results] Per-seed hybrid runs -> {detailed_path}")

    # Append to the shared metrics table, replacing any previous HybridQNN row.
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        existing = existing[existing["model"] != "HybridQNN"]
        combined = pd.concat([existing, hybrid_summary_df], ignore_index=True)
    else:
        combined = hybrid_summary_df
    combined.to_csv(metrics_path, index=False)
    print(f"[Results] Combined benchmark table -> {metrics_path}")

    return hybrid_summary_df, detailed_df, circuit_metrics, reference_metrics


def main() -> None:
    print("=" * 70)
    print("Project HQ-IDS-029: Step 2 - Hybrid Quantum Neural Network Simulation")
    print("=" * 70)

    config = load_config()
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    log_event("ACTION", f"Started Step 2 hybrid QNN training over seeds {seeds} on real UNSW-NB15 flows.")

    summary, detailed, circuit_metrics, reference_metrics = run_hybrid_experiment(config)

    log_event("VERIFICATION",
              f"Step 2 complete: {len(detailed)} hybrid QNN runs saved to results/hybrid_all_seeds.csv; "
              f"trained circuit depth {circuit_metrics['depth']}, CNOTs {circuit_metrics['cnot_count']}, "
              f"parameters {circuit_metrics['num_parameters']}; ZZ reference depth "
              f"{reference_metrics['depth']}, CNOTs {reference_metrics['cnot_count']}.")

    print("\n" + "=" * 70)
    print("  5-SEED AGGREGATED HYBRID QNN RESULTS")
    print("=" * 70)
    print(summary[[
        "model", "n_features",
        "accuracy_mean", "accuracy_std",
        "f1_macro_mean", "f1_macro_std",
        "f1_weighted_mean", "f1_weighted_std",
        "roc_auc_mean", "roc_auc_std",
        "fpr_mean", "fpr_std",
        "latency_ms_mean", "latency_ms_std",
        "circuit_depth", "cnot_count", "num_parameters",
        "reference_depth", "reference_cnot_count",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
