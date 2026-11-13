"""
Project HQ-IDS-029: Phase 2 - Multi-Class Threat Taxonomy Benchmark
===================================================================
Benchmarks all 7 model paradigms across the 10-class UNSW-NB15 taxonomy:
  0: Normal
  1: Fuzzers
  2: Analysis
  3: Backdoors
  4: DoS
  5: Exploits
  6: Generic
  7: Reconnaissance
  8: Shellcode
  9: Worms

Protocol:
- Seeds: [42, 43, 44, 45, 46]
- Stratified 70/10/20 split based on attack_cat targets
- Ingestion with zero leakage: FlowPreprocessor fitted strictly on 70% train split
- Class-Weighted Focal Loss (gamma=2.0) applied to neural models to counteract the 532:1 imbalance
- Generates:
  * results/multiclass_summary.json (comprehensive macro/weighted F1, accuracy, per-class breakdown)
  * results/figures/multiclass_confusion_matrix.png (10x10 normalized confusion matrix heatmaps)
  * results/figures/per_class_f1_breakdown.png (grouped bar chart comparing per-class F1)
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
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.common import (
    load_config,
    load_flow_data,
    RESULTS_DIR,
    log_event,
)
from src.data_loader import (
    FlowPreprocessor,
    ATTACK_CAT_MAPPING,
    ATTACK_CAT_NAMES,
)
from src.losses import ClassWeightedFocalLoss
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
    "HybridVQC_Bitstrings",
    "HybridVQC_Expectations",
    "HybridQCNN",
    "QSVM",
]

DISPLAY_NAMES = {
    "XGBoost": "XGBoost",
    "RandomForest": "Random Forest",
    "ClassicalMLP": "Classical MLP",
    "HybridVQC_Bitstrings": "Hybrid VQC (Bitstrings)",
    "HybridVQC_Expectations": "Hybrid VQC (Expectations)",
    "HybridQCNN": "Hybrid QCNN",
    "QSVM": "QSVM (Kernel)",
}

PALETTE = {
    "XGBoost": "#4DBBD5",
    "RandomForest": "#E64B35",
    "ClassicalMLP": "#2878B5",
    "HybridVQC_Bitstrings": "#00A087",
    "HybridVQC_Expectations": "#008B8B",
    "HybridQCNN": "#91D1C2",
    "QSVM": "#8491B4",
}


def map_attack_categories(values: pd.Series) -> np.ndarray:
    """Map labels losslessly; unknown attack families must stop the benchmark."""
    cleaned = values.astype(str).str.strip()
    mapped = cleaned.map(ATTACK_CAT_MAPPING)
    if mapped.isna().any():
        unknown = sorted(cleaned[mapped.isna()].unique())
        raise ValueError(f"Unknown attack categories; refusing to map them to Normal: {unknown}")
    return mapped.to_numpy(dtype=np.int64)


def stratified_multiclass_split(
    frame: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/10/20 train/val/test split preserving attack_cat distribution."""
    # Ensure targets are defined
    y = map_attack_categories(frame["attack_cat"])
    
    # Stratified split with fallback if extremely small classes have fewer than 2 samples
    train_val_df, test_df = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=y
    )
    y_tv = map_attack_categories(train_val_df["attack_cat"])
    val_rel_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_rel_size, random_state=seed, stratify=y_tv
    )
    return train_df, val_df, test_df


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive multi-class metrics across all 10 threat categories."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    p_per_class = precision_score(y_true, y_pred, average=None, labels=list(range(10)), zero_division=0)
    r_per_class = recall_score(y_true, y_pred, average=None, labels=list(range(10)), zero_division=0)
    f_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(10)), zero_division=0)

    per_class = {}
    for i, name in enumerate(ATTACK_CAT_NAMES):
        per_class[name] = {
            "precision": float(p_per_class[i]),
            "recall": float(r_per_class[i]),
            "f1": float(f_per_class[i]),
        }

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist(),
    }


def run_multiclass_benchmark():
    print("=" * 85)
    print("HQ-IDS-029: PHASE 2 - MULTI-CLASS THREAT TAXONOMY BENCHMARK")
    print("10-Class UNSW-NB15 Triage across Classical, Variational, Convolutional & Kernel QML")
    print("=" * 85)

    config = load_config()
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    raw_df = load_flow_data(config)
    print(f"Loaded dataset: {len(raw_df)} flows, seeds: {seeds}")

    # Track results across seeds
    all_seed_results = {model: [] for model in MODEL_NAMES}
    cumulative_cms = {model: np.zeros((10, 10), dtype=np.int64) for model in MODEL_NAMES}

    start_total_time = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed} ({seed_idx + 1}/{len(seeds)})] Multi-Class Stratified Partitioning ---")
        train_df, val_df, test_df = stratified_multiclass_split(raw_df, seed=seed)

        # Preprocessor with multi_class=True
        preprocessor = FlowPreprocessor(
            feature_range=(-np.pi, np.pi),
            log_transform_skewed=True,
            skew_threshold=1.0,
            multi_class=True,
        )
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_val, y_val = preprocessor.transform(val_df)
        X_test, y_test = preprocessor.transform(test_df)

        print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Compute dynamic class weights for Focal Loss
        class_counts = np.bincount(y_train, minlength=10).astype(float)
        class_counts = np.maximum(class_counts, 1.0)  # Avoid division by zero
        total_train = len(y_train)
        weights_arr = total_train / (10.0 * class_counts)
        weights_arr = weights_arr / np.mean(weights_arr)
        weights_tensor = torch.tensor(weights_arr, dtype=torch.float32)

        focal_loss_fn = ClassWeightedFocalLoss(class_weights=weights_tensor, gamma=2.0)

        # -------------------------------------------------------------
        # 1. XGBoost Multi-Class (multi:softprob)
        # -------------------------------------------------------------
        print(f"[{seed}] Model 1/7: XGBoost (10-class multi:softprob)...")
        t0 = time.time()
        xgb = XGBoostBaseline(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=10,
            random_state=seed,
        )
        xgb.fit(X_train, y_train, X_val, y_val)
        y_pred = xgb.predict(X_test)
        lat = xgb.measure_latency(X_test[0], n_runs=50)
        m = compute_multiclass_metrics(y_test, y_pred)
        m["latency_ms"] = lat
        m["train_time_sec"] = time.time() - t0
        all_seed_results["XGBoost"].append(m)
        cumulative_cms["XGBoost"] += np.array(m["confusion_matrix"])

        # -------------------------------------------------------------
        # 2. Random Forest Multi-Class
        # -------------------------------------------------------------
        print(f"[{seed}] Model 2/7: Random Forest (10-class)...")
        t0 = time.time()
        rf = RandomForestBaseline(n_estimators=100, max_depth=12, random_state=seed)
        rf.fit(X_train, y_train, X_val, y_val)
        y_pred = rf.predict(X_test)
        lat = rf.measure_latency(X_test[0], n_runs=50)
        m = compute_multiclass_metrics(y_test, y_pred)
        m["latency_ms"] = lat
        m["train_time_sec"] = time.time() - t0
        all_seed_results["RandomForest"].append(m)
        cumulative_cms["RandomForest"] += np.array(m["confusion_matrix"])

        # -------------------------------------------------------------
        # 3. Classical MLP Multi-Class with Focal Loss (39 -> 16 -> 4 -> 10)
        # -------------------------------------------------------------
        print(f"[{seed}] Model 3/7: Classical MLP (39->16->4->10 with Focal Loss)...")
        t0 = time.time()
        mlp = ClassicalMLPBaseline(
            input_dim=X_train.shape[1],
            hidden_dim=16,
            bottleneck_dim=4,
            output_dim=10,
            learning_rate=0.005,
            batch_size=64,
            epochs=35,
            loss_fn=focal_loss_fn,
            random_state=seed,
        )
        mlp.fit(X_train, y_train, X_val, y_val)
        y_pred = mlp.predict(X_test)
        lat = mlp.measure_latency(X_test[0], n_runs=50)
        m = compute_multiclass_metrics(y_test, y_pred)
        m["latency_ms"] = lat
        m["train_time_sec"] = time.time() - t0
        all_seed_results["ClassicalMLP"].append(m)
        cumulative_cms["ClassicalMLP"] += np.array(m["confusion_matrix"])

        # -------------------------------------------------------------
        # 4. Hybrid VQC (Bitstring Mode: 16 Bitstrings -> 10 Classes)
        # -------------------------------------------------------------
        print(f"[{seed}] Model 4/7: Hybrid VQC Mode B (16 Bitstring Probs -> 10 Logits)...")
        t0 = time.time()
        vqc_bits = HybridQNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=16,
            num_qubits=4,
            num_layers=2,
            output_dim=10,
            ansatz_type="RealAmplitudes",
            feature_map="AngleEmbedding",
            feature_map_reps=1,
            quantum_layer="vqc",
            output_mode="bitstrings",
            loss_fn=focal_loss_fn,
            learning_rate=0.01,
            batch_size=64,
            epochs=20,
            random_state=seed,
        )
        vqc_bits.fit(X_train, y_train, X_val, y_val, epochs=20, verbose=False)
        y_pred = vqc_bits.predict(X_test)
        lat = vqc_bits.measure_latency(X_test[0], n_runs=50)
        m = compute_multiclass_metrics(y_test, y_pred)
        m["latency_ms"] = lat
        m["train_time_sec"] = time.time() - t0
        all_seed_results["HybridVQC_Bitstrings"].append(m)
        cumulative_cms["HybridVQC_Bitstrings"] += np.array(m["confusion_matrix"])

        # -------------------------------------------------------------
        # 5. Hybrid VQC (Expectation Mode: 4 Expectations -> 10 Classes)
        # -------------------------------------------------------------
        print(f"[{seed}] Model 5/7: Hybrid VQC Mode A (4 Pauli-Z Expectations -> 10 Logits)...")
        t0 = time.time()
        vqc_exp = HybridQNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=16,
            num_qubits=4,
            num_layers=2,
            output_dim=10,
            ansatz_type="RealAmplitudes",
            feature_map="AngleEmbedding",
            feature_map_reps=1,
            quantum_layer="vqc",
            output_mode="expectations",
            loss_fn=focal_loss_fn,
            learning_rate=0.01,
            batch_size=64,
            epochs=20,
            random_state=seed,
        )
        vqc_exp.fit(X_train, y_train, X_val, y_val, epochs=20, verbose=False)
        y_pred = vqc_exp.predict(X_test)
        lat = vqc_exp.measure_latency(X_test[0], n_runs=50)
        m = compute_multiclass_metrics(y_test, y_pred)
        m["latency_ms"] = lat
        m["train_time_sec"] = time.time() - t0
        all_seed_results["HybridVQC_Expectations"].append(m)
        cumulative_cms["HybridVQC_Expectations"] += np.array(m["confusion_matrix"])

        # -------------------------------------------------------------
        # 6. Hybrid QCNN Multi-Class (2 Active Qubits -> 10 Classes)
        # -------------------------------------------------------------
        print(f"[{seed}] Model 6/7: Hybrid QCNN (2 Active Qubits -> 10 Logits)...")
        t0 = time.time()
        qcnn = HybridQCNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=16,
            output_dim=10,
            learning_rate=0.01,
            batch_size=64,
            epochs=20,
            loss_fn=focal_loss_fn,
            random_state=seed,
        )
        qcnn.fit(X_train, y_train, X_val, y_val, epochs=20, verbose=False)
        y_pred = qcnn.predict(X_test)
        lat = qcnn.measure_latency(X_test[0], n_runs=50)
        m = compute_multiclass_metrics(y_test, y_pred)
        m["latency_ms"] = lat
        m["train_time_sec"] = time.time() - t0
        all_seed_results["HybridQCNN"].append(m)
        cumulative_cms["HybridQCNN"] += np.array(m["confusion_matrix"])

        # -------------------------------------------------------------
        # 7. QSVM Multi-Class (decision_function_shape='ovr')
        # -------------------------------------------------------------
        print(f"[{seed}] Model 7/7: QSVM (Autoencoder Caching + Multi-Class OVR Kernel)...")
        t0 = time.time()
        ae = FlowAutoencoder(input_dim=X_train.shape[1], hidden_dim=16, latent_dim=4)
        ae.fit(X_train, epochs=25, batch_size=64, lr=0.002, X_val=X_val, random_seed=seed)
        ae.eval()
        with torch.no_grad():
            Z_train = ae.encode(torch.tensor(X_train, dtype=torch.float32)).cpu().numpy()
            Z_test = ae.encode(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()

        qsvm = QSVMClassifier(
            num_qubits=4,
            feature_map="AngleEmbedding",
            C=1.0,
            probability=False,
            decision_function_shape="ovr",
            random_state=seed,
        )
        qsvm.fit(Z_train, y_train)
        y_pred = qsvm.predict(Z_test)
        lat = qsvm.measure_latency(Z_test[0], n_runs=50)
        m = compute_multiclass_metrics(y_test, y_pred)
        m["latency_ms"] = lat
        m["train_time_sec"] = time.time() - t0
        all_seed_results["QSVM"].append(m)
        cumulative_cms["QSVM"] += np.array(m["confusion_matrix"])

    # -------------------------------------------------------------
    # Aggregate Summary Across 5 Seeds
    # -------------------------------------------------------------
    summary_dict: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": float(time.time() - start_total_time),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "classes": ATTACK_CAT_NAMES,
        "models": {},
    }

    scalar_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "latency_ms"]

    for model_name in MODEL_NAMES:
        seed_runs = all_seed_results[model_name]
        m_summary = {}
        for sm in scalar_metrics:
            vals = [run[sm] for run in seed_runs]
            m_summary[sm] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "formatted": f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}",
            }

        # Aggregate per-class metrics
        per_class_summary = {}
        for cat_name in ATTACK_CAT_NAMES:
            p_vals = [run["per_class"][cat_name]["precision"] for run in seed_runs]
            r_vals = [run["per_class"][cat_name]["recall"] for run in seed_runs]
            f_vals = [run["per_class"][cat_name]["f1"] for run in seed_runs]
            per_class_summary[cat_name] = {
                "precision": f"{np.mean(p_vals):.4f} +/- {np.std(p_vals, ddof=1):.4f}",
                "recall": f"{np.mean(r_vals):.4f} +/- {np.std(r_vals, ddof=1):.4f}",
                "f1": f"{np.mean(f_vals):.4f} +/- {np.std(f_vals, ddof=1):.4f}",
                "f1_mean": float(np.mean(f_vals)),
            }

        m_summary["per_class"] = per_class_summary
        m_summary["confusion_matrix_sum"] = cumulative_cms[model_name].tolist()
        summary_dict["models"][model_name] = m_summary

    # Save summary JSON
    json_path = os.path.join(RESULTS_DIR, "multiclass_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\nSaved Multi-Class Summary JSON to: {json_path}")

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Model Paradigm':<26} | {'Accuracy':<17} | {'Macro F1':<17} | {'Weighted F1':<17} | {'Latency (ms)':<15}")
    print("-" * 105)
    for model_name in MODEL_NAMES:
        s = summary_dict["models"][model_name]
        dname = DISPLAY_NAMES.get(model_name, model_name)
        print(f"{dname:<26} | {s['accuracy']['formatted']:<17} | {s['f1_macro']['formatted']:<17} | {s['f1_weighted']['formatted']:<17} | {s['latency_ms']['formatted']:<15}")
    print("=" * 105)

    # Generate Figures
    _generate_multiclass_confusion_figure(cumulative_cms)
    _generate_per_class_f1_figure(summary_dict)

    log_event(
        "run_multiclass_benchmark.py",
        "Completed 10-class multi-class threat taxonomy benchmark with Class-Weighted Focal Loss across 7 paradigms.",
    )


def _generate_multiclass_confusion_figure(cumulative_cms: Dict[str, np.ndarray]):
    """
    Generates results/figures/multiclass_confusion_matrix.png
    comparing 10x10 normalized confusion matrix heatmaps between
    XGBoost, Hybrid VQC (Bitstrings), and Classical MLP.
    """
    models_to_plot = [
        ("XGBoost", "XGBoost Classifier"),
        ("HybridVQC_Bitstrings", "Hybrid VQC (16 Bitstring Readout)"),
        ("ClassicalMLP", "Classical MLP (Bottleneck Control)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#FAFAFA")

    for idx, (m_key, title) in enumerate(models_to_plot):
        ax = axes[idx]
        cm = cumulative_cms[m_key]
        # Normalize row-wise (true label)
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=ATTACK_CAT_NAMES,
            yticklabels=ATTACK_CAT_NAMES if idx == 0 else False,
            ax=ax,
            annot_kws={"size": 7.5},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Category", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("True Category", fontsize=10, fontweight="bold")
        ax.set_xticklabels(ATTACK_CAT_NAMES, rotation=45, ha="right", fontsize=8)
        if idx == 0:
            ax.set_yticklabels(ATTACK_CAT_NAMES, rotation=0, fontsize=8)

    plt.suptitle("HQ-IDS-029: 10-Class Threat Triage Normalized Confusion Matrices (Aggregated 5 Seeds, N=5,000)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "multiclass_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated confusion matrix figure: {fig_path}")


def _generate_per_class_f1_figure(summary: Dict[str, Any]):
    """
    Generates results/figures/per_class_f1_breakdown.png
    comparing per-class F1-scores across all 10 threat categories.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    n_classes = len(ATTACK_CAT_NAMES)
    x = np.arange(n_classes)
    total_width = 0.82
    width = total_width / len(MODEL_NAMES)

    for i, m_key in enumerate(MODEL_NAMES):
        f1_means = [summary["models"][m_key]["per_class"][c]["f1_mean"] * 100 for c in ATTACK_CAT_NAMES]
        offset = (i - len(MODEL_NAMES) / 2) * width + width / 2
        ax.bar(
            x + offset,
            f1_means,
            width=width,
            label=DISPLAY_NAMES[m_key],
            color=PALETTE[m_key],
            edgecolor="#222222",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_CAT_NAMES, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Per-Class F1-Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("HQ-IDS-029: Per-Class F1 Performance across 10 UNSW-NB15 Threat Categories", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "per_class_f1_breakdown.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated per-class F1 figure: {fig_path}")


if __name__ == "__main__":
    run_multiclass_benchmark()
