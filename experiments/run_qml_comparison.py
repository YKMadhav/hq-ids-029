"""
Project HQ-IDS-029: Master Multi-Paradigm QML Benchmark Suite
=============================================================
Unified, 5-seed comparative benchmark across all 3 QML architectures,
classical baselines, and ablation controls:

Evaluated Models (7 total):
1. XGBoost Classifier (Native gradient-boosted tree baseline)
2. Random Forest Classifier (Bagged tree ensemble baseline)
3. Depth-Matched Classical MLP (39 -> 16 -> 4 -> 2)
4. Hybrid VQC (Bottleneck Encoder + 4-qubit VQC + Readout Head)
5. Hybrid QCNN (Bottleneck Encoder + 4-qubit QCNN Conv/Pool + Readout Head)
6. QSVM (Bottleneck Latent + Quantum Kernel + Precomputed SVC)
7. Control A (Quantum Bypassed) (Bottleneck Encoder + Direct Linear Head)

Protocol:
- Seeds: [42, 43, 44, 45, 46]
- Stratified 70/10/20 train/val/test splits on UNSW-NB15 flow dataset (N=5,000)
- Zero data leakage: preprocessor fitted strictly on training partition
- Feature caching: Bottleneck Autoencoder fit once per seed for QSVM
- Latency benchmarking: Single-sample inference latency (ms/sample)

Outputs:
- results/qml_comparison_results.csv (Full per-seed, per-model evaluation)
- results/qml_comparison_summary.json (Aggregated mean +/- std and execution metadata)
- results/figures/qml_architecture_comparison.png (Grouped publication comparison chart)
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.common import (
    load_config,
    load_flow_data,
    stratified_split,
    preprocess_split,
    compute_metrics,
    RESULTS_DIR,
    log_event,
)
from src.classical_models import RandomForestBaseline, XGBoostBaseline, ClassicalMLPBaseline
from src.hybrid_pipeline import HybridQNNClassifier
from src.qcnn_circuit import HybridQCNNClassifier
from src.qsvm_classifier import QSVMClassifier
from src.encoders import FlowAutoencoder

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_NAMES = [
    "XGBoost",
    "RandomForest",
    "ClassicalMLP",
    "HybridVQC",
    "HybridQCNN",
    "QSVM",
    "Control_A_Bypassed",
]

DISPLAY_NAMES = {
    "XGBoost": "XGBoost",
    "RandomForest": "Random Forest",
    "ClassicalMLP": "Classical MLP",
    "HybridVQC": "Hybrid VQC",
    "HybridQCNN": "Hybrid QCNN",
    "QSVM": "QSVM (Kernel)",
    "Control_A_Bypassed": "Control A (Bypassed)",
}

PALETTE = {
    "XGBoost": "#4DBBD5",
    "RandomForest": "#E64B35",
    "ClassicalMLP": "#2878B5",
    "HybridVQC": "#00A087",
    "HybridQCNN": "#91D1C2",
    "QSVM": "#8491B4",
    "Control_A_Bypassed": "#7E6148",
}


def run_benchmark():
    print("=" * 80)
    print("HQ-IDS-029: MASTER MULTI-PARADIGM QML BENCHMARK SUITE")
    print("Variational (VQC) vs. Convolutional (QCNN) vs. Kernel (QSVM) vs. Baselines")
    print("=" * 80)

    config = load_config()
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    raw_df = load_flow_data(config)
    print(f"Loaded dataset: {len(raw_df)} flows, seeds: {seeds}")

    results_rows = []
    start_total_time = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed} ({seed_idx + 1}/{len(seeds)})] Generating Stratified Partitions ---")
        train_df, val_df, test_df = stratified_split(raw_df, seed=seed)
        preprocessor, arrays = preprocess_split(train_df, val_df, test_df, config)

        X_train, y_train = arrays["train"]
        X_val, y_val = arrays["val"]
        X_test, y_test = arrays["test"]

        # -------------------------------------------------------------
        # 1. Classical Baselines (XGBoost, Random Forest, Classical MLP)
        # -------------------------------------------------------------
        print(f"[{seed}] Fitting 1/7: XGBoost...")
        t0 = time.time()
        xgb = XGBoostBaseline(
            n_estimators=config.get("baselines", {}).get("xgboost", {}).get("n_estimators", 100),
            max_depth=config.get("baselines", {}).get("xgboost", {}).get("max_depth", 6),
            learning_rate=config.get("baselines", {}).get("xgboost", {}).get("learning_rate", 0.1),
            random_state=seed,
        )
        xgb.fit(X_train, y_train, X_val, y_val)
        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        lat = xgb.measure_latency(X_test[0], n_runs=100)
        results_rows.append({
            "seed": seed, "model": "XGBoost", "family": "Classical Ensemble",
            **m, "latency_ms": lat, "train_time_sec": time.time() - t0
        })

        print(f"[{seed}] Fitting 2/7: Random Forest...")
        t0 = time.time()
        rf = RandomForestBaseline(
            n_estimators=config.get("baselines", {}).get("random_forest", {}).get("n_estimators", 100),
            max_depth=config.get("baselines", {}).get("random_forest", {}).get("max_depth", 12),
            random_state=seed,
        )
        rf.fit(X_train, y_train, X_val, y_val)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        lat = rf.measure_latency(X_test[0], n_runs=100)
        results_rows.append({
            "seed": seed, "model": "RandomForest", "family": "Classical Ensemble",
            **m, "latency_ms": lat, "train_time_sec": time.time() - t0
        })

        print(f"[{seed}] Fitting 3/7: Depth-Matched Classical MLP (39->16->4->2)...")
        t0 = time.time()
        mlp = ClassicalMLPBaseline(
            input_dim=X_train.shape[1],
            hidden_dim=config.get("baselines", {}).get("mlp", {}).get("hidden_dim", 16),
            bottleneck_dim=config.get("baselines", {}).get("mlp", {}).get("bottleneck_dim", 4),
            output_dim=2,
            learning_rate=config.get("baselines", {}).get("mlp", {}).get("learning_rate", 0.005),
            batch_size=config.get("baselines", {}).get("mlp", {}).get("batch_size", 64),
            epochs=config.get("baselines", {}).get("mlp", {}).get("epochs", 40),
            random_state=seed,
        )
        mlp.fit(X_train, y_train, X_val, y_val)
        y_pred = mlp.predict(X_test)
        y_prob = mlp.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        lat = mlp.measure_latency(X_test[0], n_runs=100)
        results_rows.append({
            "seed": seed, "model": "ClassicalMLP", "family": "Classical Neural Net",
            **m, "latency_ms": lat, "train_time_sec": time.time() - t0
        })

        # -------------------------------------------------------------
        # 2. Hybrid VQC (Variational QML)
        # -------------------------------------------------------------
        print(f"[{seed}] Fitting 4/7: Hybrid VQC...")
        t0 = time.time()
        vqc = HybridQNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=config.get("classical_encoder", {}).get("hidden_dim", 16),
            num_qubits=config.get("quantum_circuit", {}).get("num_qubits", 4),
            num_layers=config.get("quantum_circuit", {}).get("reps", 2),
            output_dim=2,
            ansatz_type=config.get("quantum_circuit", {}).get("ansatz", "RealAmplitudes"),
            feature_map=config.get("quantum_circuit", {}).get("feature_map", "AngleEmbedding"),
            feature_map_reps=config.get("quantum_circuit", {}).get("feature_map_reps", 1),
            quantum_layer="vqc",
            learning_rate=config.get("hybrid_classifier", {}).get("learning_rate", 0.01),
            batch_size=config.get("hybrid_classifier", {}).get("batch_size", 64),
            epochs=config.get("hybrid_classifier", {}).get("epochs", 20),
            random_state=seed,
        )
        vqc.fit(X_train, y_train, X_val, y_val, epochs=20, verbose=False)
        y_pred = vqc.predict(X_test)
        y_prob = vqc.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        lat = vqc.measure_latency(X_test[0], n_runs=100)
        results_rows.append({
            "seed": seed, "model": "HybridVQC", "family": "Variational QML",
            **m, "latency_ms": lat, "train_time_sec": time.time() - t0
        })

        # -------------------------------------------------------------
        # 3. Hybrid QCNN (Convolutional & Pooling QML)
        # -------------------------------------------------------------
        print(f"[{seed}] Fitting 5/7: Hybrid QCNN (Conv + Pooling)...")
        t0 = time.time()
        qcnn = HybridQCNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=config.get("classical_encoder", {}).get("hidden_dim", 16),
            output_dim=2,
            learning_rate=config.get("hybrid_classifier", {}).get("learning_rate", 0.01),
            batch_size=config.get("hybrid_classifier", {}).get("batch_size", 64),
            epochs=config.get("hybrid_classifier", {}).get("epochs", 20),
            random_state=seed,
        )
        qcnn.fit(X_train, y_train, X_val, y_val, epochs=20, verbose=False)
        y_pred = qcnn.predict(X_test)
        y_prob = qcnn.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        lat = qcnn.measure_latency(X_test[0], n_runs=100)
        results_rows.append({
            "seed": seed, "model": "HybridQCNN", "family": "Convolutional QML",
            **m, "latency_ms": lat, "train_time_sec": time.time() - t0
        })

        # -------------------------------------------------------------
        # 4. QSVM (Kernel-Based QML)
        # -------------------------------------------------------------
        print(f"[{seed}] Fitting 6/7: QSVM (Autoencoder Caching + Quantum Kernel)...")
        t0 = time.time()
        # Fit autoencoder once per seed on training partition
        ae = FlowAutoencoder(
            input_dim=X_train.shape[1],
            hidden_dim=config.get("classical_encoder", {}).get("hidden_dim", 16),
            latent_dim=4,
        )
        ae.fit(X_train, epochs=25, batch_size=64, lr=0.002, X_val=X_val, random_seed=seed)
        ae.eval()
        with torch.no_grad():
            Z_train = ae.encode(torch.tensor(X_train, dtype=torch.float32)).cpu().numpy()
            Z_test = ae.encode(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()

        qsvm = QSVMClassifier(
            num_qubits=4,
            feature_map="AngleEmbedding",
            C=1.0,
            probability=True,
            random_state=seed,
        )
        qsvm.fit(Z_train, y_train)
        y_pred = qsvm.predict(Z_test)
        y_prob = qsvm.predict_proba(Z_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        lat = qsvm.measure_latency(Z_test[0], n_runs=100)
        results_rows.append({
            "seed": seed, "model": "QSVM", "family": "Kernel-based QML",
            **m, "latency_ms": lat, "train_time_sec": time.time() - t0
        })

        # -------------------------------------------------------------
        # 5. Control A (Quantum Bypassed)
        # -------------------------------------------------------------
        print(f"[{seed}] Fitting 7/7: Control A (Quantum Bypassed)...")
        t0 = time.time()
        bypassed = HybridQNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=config.get("classical_encoder", {}).get("hidden_dim", 16),
            num_qubits=4,
            output_dim=2,
            quantum_layer="identity",
            learning_rate=config.get("hybrid_classifier", {}).get("learning_rate", 0.01),
            batch_size=config.get("hybrid_classifier", {}).get("batch_size", 64),
            epochs=config.get("hybrid_classifier", {}).get("epochs", 20),
            random_state=seed,
        )
        bypassed.fit(X_train, y_train, X_val, y_val, epochs=20, verbose=False)
        y_pred = bypassed.predict(X_test)
        y_prob = bypassed.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        lat = bypassed.measure_latency(X_test[0], n_runs=100)
        results_rows.append({
            "seed": seed, "model": "Control_A_Bypassed", "family": "Ablation Control",
            **m, "latency_ms": lat, "train_time_sec": time.time() - t0
        })

    # Save full raw CSV results
    results_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(RESULTS_DIR, "qml_comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved full per-seed results to: {csv_path}")

    # Compute summary statistics (mean +/- std)
    summary_dict: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": float(time.time() - start_total_time),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "models": {},
    }

    metrics_list = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "fpr", "latency_ms"]

    for model_name in MODEL_NAMES:
        sub_df = results_df[results_df["model"] == model_name]
        family = sub_df["family"].iloc[0]
        stats = {"family": family}
        for met in metrics_list:
            vals = sub_df[met].values
            stats[met] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "formatted": f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}",
            }
        summary_dict["models"][model_name] = stats

    json_path = os.path.join(RESULTS_DIR, "qml_comparison_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved summary JSON to: {json_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Model':<22} | {'Accuracy':<17} | {'Macro F1':<17} | {'FPR':<17} | {'Latency (ms)':<15}")
    print("-" * 95)
    for model_name in MODEL_NAMES:
        s = summary_dict["models"][model_name]
        dname = DISPLAY_NAMES.get(model_name, model_name)
        print(f"{dname:<22} | {s['accuracy']['formatted']:<17} | {s['f1_macro']['formatted']:<17} | {s['fpr']['formatted']:<17} | {s['latency_ms']['formatted']:<15}")
    print("=" * 95)

    # Generate Publication Figure
    _generate_comparison_figure(results_df, summary_dict)

    log_event(
        "run_qml_comparison.py",
        "Completed 5-seed master benchmark across all 3 QML paradigms (VQC, QCNN, QSVM), classical baselines, and ablation controls.",
    )


def _generate_comparison_figure(df: pd.DataFrame, summary: Dict[str, Any]):
    """
    Generates publication figure results/figures/qml_architecture_comparison.png
    comparing Accuracy, Macro F1, FPR, and Inference Latency across all 7 models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FAFAFA")

    # Order of models
    models = MODEL_NAMES
    x_pos = np.arange(len(models))
    bar_width = 0.55

    # 1. Accuracy
    ax1 = axes[0, 0]
    ax1.set_facecolor("#FFFFFF")
    acc_means = [summary["models"][m]["accuracy"]["mean"] * 100 for m in models]
    acc_stds = [summary["models"][m]["accuracy"]["std"] * 100 for m in models]
    colors = [PALETTE[m] for m in models]

    bars1 = ax1.bar(x_pos, acc_means, yerr=acc_stds, capsize=4, width=bar_width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax1.set_ylim(91.0, 96.0)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Multi-Paradigm Classification Accuracy", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=30, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars1, acc_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, mean + 0.35, f"{mean:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 2. Macro F1
    ax2 = axes[0, 1]
    ax2.set_facecolor("#FFFFFF")
    f1_means = [summary["models"][m]["f1_macro"]["mean"] * 100 for m in models]
    f1_stds = [summary["models"][m]["f1_macro"]["std"] * 100 for m in models]

    bars2 = ax2.bar(x_pos, f1_means, yerr=f1_stds, capsize=4, width=bar_width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax2.set_ylim(89.0, 95.0)
    ax2.set_ylabel("Macro F1-Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Multi-Paradigm Threat Macro F1-Score", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=30, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars2, f1_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, mean + 0.35, f"{mean:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 3. False Positive Rate (FPR) - Lower is better
    ax3 = axes[1, 0]
    ax3.set_facecolor("#FFFFFF")
    fpr_means = [summary["models"][m]["fpr"]["mean"] * 100 for m in models]
    fpr_stds = [summary["models"][m]["fpr"]["std"] * 100 for m in models]

    bars3 = ax3.bar(x_pos, fpr_means, yerr=fpr_stds, capsize=4, width=bar_width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax3.set_ylim(8.0, 22.0)
    ax3.set_ylabel("False Positive Rate (%) [Lower = Better]", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Operational Alert Fatigue (FPR)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=30, ha="right", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars3, fpr_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, mean + 0.5, f"{mean:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 4. Latency (log scale)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#FFFFFF")
    lat_means = [summary["models"][m]["latency_ms"]["mean"] for m in models]
    lat_stds = [summary["models"][m]["latency_ms"]["std"] for m in models]

    bars4 = ax4.bar(x_pos, lat_means, yerr=lat_stds, capsize=4, width=bar_width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax4.set_yscale("log")
    ax4.set_ylabel("Latency (ms/sample, log-scale)", fontsize=11, fontweight="bold")
    ax4.set_title("(d) Single-Sample Inference Latency", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=30, ha="right", fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars4, lat_means):
        ax4.text(bar.get_x() + bar.get_width() / 2, mean * 1.25, f"{mean:.2f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("HQ-IDS-029: Unified Multi-Paradigm QML vs. Classical Intrusion Detection Suite", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "qml_architecture_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated publication comparison figure: {fig_path}")


if __name__ == "__main__":
    run_benchmark()
