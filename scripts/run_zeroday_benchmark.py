"""
Project HQ-IDS-029: Phase 3 Unsupervised Zero-Day Anomaly Detection Benchmark
=============================================================================
This script executes a rigorous 5-seed benchmark comparing:
1. Classical Autoencoder (CAE)
2. Quantum Autoencoder (QAE, Romero et al. 2017)
3. Hybrid Reconstruction Ensemble (CAE + QAE)

Pipeline Protocol:
- Training: 100% Benign ('Normal') network flows. Models learn strictly the normal traffic manifold.
- Preprocessor: Fit strictly on benign training flows.
- Validation: Benign + Known Attack ('Exploits') to calibrate anomaly decision thresholds (tau_95 and tau_opt).
- Testing: Benign + Completely Unseen Zero-Day Threats ('Worms', 'Backdoor', 'Shellcode', 'Analysis', 'DoS', 'Fuzzers', 'Reconnaissance').
- Metrics: ROC-AUC, PR-AUC, Macro-F1, Precision, Recall, FPR, Specificity, and per-category zero-day detection rates across 5 seeds [42, 43, 44, 45, 46].
"""

import os
import sys
import math
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.anomaly_loader import get_zeroday_anomaly_dataloaders
from src.anomaly_models import ClassicalAutoencoder, QuantumAutoencoder

SEEDS = [42, 43, 44, 45, 46]


def evaluate_anomaly_detector(
    scores_train: np.ndarray,
    scores_val: np.ndarray,
    y_val: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    test_cats: List[str],
) -> Dict[str, Any]:
    """
    Computes comprehensive anomaly detection metrics:
    - ROC-AUC and PR-AUC
    - Thresholds calibrated from validation:
      * tau_95: 95th percentile of benign validation scores (targeting ~5% FPR)
      * tau_opt: threshold maximizing validation F1 score
    - Precision, Recall (TPR), FPR, Specificity, Macro F1, Binary F1
    - Per-category zero-day detection breakdown
    """
    # 1. Discrimination capacity (threshold-independent)
    roc_auc = float(roc_auc_score(y_test, scores_test))
    pr_auc = float(average_precision_score(y_test, scores_test))

    # 2. Calibration of decision thresholds on validation
    val_benign_scores = scores_val[y_val == 0]
    tau_95 = float(np.percentile(val_benign_scores, 95))

    # Optimal F1 threshold search on validation
    threshold_candidates = np.percentile(scores_val, np.linspace(5, 95, 91))
    best_tau_val = float(threshold_candidates[0])
    best_f1_val = -1.0
    for tau in threshold_candidates:
        preds = (scores_val >= tau).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1_val:
            best_f1_val = f1
            best_tau_val = float(tau)

    # Use tau_95 as primary operational threshold (standard SOC 95% benign percentile)
    tau = tau_95
    y_pred = (scores_test >= tau).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1_binary = float(f1_score(y_test, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    fpr = float(fp / max(fp + tn, 1))
    specificity = float(tn / max(tn + fp, 1))

    # Evaluation with optimal validation threshold as secondary reference
    y_pred_opt = (scores_test >= best_tau_val).astype(int)
    f1_macro_opt = float(f1_score(y_test, y_pred_opt, average="macro", zero_division=0))
    recall_opt = float(recall_score(y_test, y_pred_opt, zero_division=0))
    fpr_opt = float(np.sum((y_test == 0) & (y_pred_opt == 1)) / max(np.sum(y_test == 0), 1))

    # Per-category zero-day detection breakdown
    test_cats_arr = np.array(test_cats)
    unique_cats = sorted(list(set(test_cats)))
    per_category = {}
    for cat in unique_cats:
        cat_mask = (test_cats_arr == cat)
        cat_total = int(np.sum(cat_mask))
        if cat_total == 0:
            continue
        cat_flagged = int(np.sum(y_pred[cat_mask]))
        detection_rate = float(cat_flagged / cat_total)
        per_category[cat] = {
            "total_samples": cat_total,
            "flagged_samples": cat_flagged,
            "detection_rate": detection_rate,
        }

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tau_95": tau_95,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_binary": f1_binary,
        "f1_macro": f1_macro,
        "fpr": fpr,
        "specificity": specificity,
        "tau_opt_val": best_tau_val,
        "f1_macro_opt": f1_macro_opt,
        "recall_opt": recall_opt,
        "fpr_opt": fpr_opt,
        "per_category": per_category,
    }


def run_benchmark():
    print("=" * 80)
    print("PROJECT HQ-IDS-029: PHASE 3 ZERO-DAY UNSUPERVISED ANOMALY DETECTION BENCHMARK")
    print(f"Seeds: {SEEDS} | Paradigms: Classical Autoencoder, Quantum Autoencoder, Hybrid Ensemble")
    print("=" * 80)

    os.makedirs("results/figures", exist_ok=True)

    all_seed_results = []
    # Data storage for multi-seed curves and distribution plots
    plot_data_first_seed = {}
    roc_curves_data = {"ClassicalAutoencoder": [], "QuantumAutoencoder": [], "HybridEnsemble": []}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n[Seed {seed} ({seed_idx + 1}/{len(SEEDS)})] Initializing Zero-Day Data Partition...")
        train_loader, val_loader, test_loader, preprocessor, meta = get_zeroday_anomaly_dataloaders(
            data_path="./data/real_flows.csv",
            known_attack_cat="Exploits",
            n_train_benign=1100,
            n_val_benign=150,
            n_val_known_attack=150,
            batch_size=64,
            random_seed=seed,
        )

        input_dim = meta["input_dim"]
        print(f"  Train: {meta['n_train']} Benign flows | Val: {meta['n_val']} (Benign + {meta['val_known_attack']})")
        print(f"  Test: {meta['n_test']} flows (Benign + Unseen Zero-Days: {meta['zero_day_categories']})")

        # Extract numpy arrays for evaluation
        X_train = np.array([batch[0].numpy() for batch in train_loader], dtype=object)
        X_train = np.concatenate(X_train, axis=0)

        X_val_list, y_val_list = [], []
        for batch in val_loader:
            X_val_list.append(batch[0].numpy())
            y_val_list.append(batch[1].numpy())
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        X_test_list, y_test_list, test_cats_list = [], [], []
        for batch in test_loader:
            X_test_list.append(batch[0].numpy())
            y_test_list.append(batch[1].numpy())
            test_cats_list.extend(batch[2])
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # -------------------------------------------------------------
        # 1. Classical Autoencoder (CAE)
        # -------------------------------------------------------------
        print("  Training Classical Autoencoder (39 -> 16 -> 4 -> 16 -> 39)...")
        t0 = time.time()
        cae = ClassicalAutoencoder(input_dim=input_dim, hidden_dim=16, latent_dim=4)
        cae_history = cae.fit_model(
            train_loader,
            epochs=25,
            lr=0.003,
            device="cpu",
            seed=seed,
        )
        t_cae = time.time() - t0
        print(f"  CAE Train Complete ({t_cae:.1f}s) | Final Loss: {cae_history['train_loss'][-1]:.5f}")

        # Compute CAE anomaly scores (MSE)
        cae_train_scores = cae.compute_anomaly_scores(X_train)
        cae_val_scores = cae.compute_anomaly_scores(X_val)
        cae_test_scores = cae.compute_anomaly_scores(X_test)

        cae_metrics = evaluate_anomaly_detector(
            cae_train_scores, cae_val_scores, y_val, cae_test_scores, y_test, test_cats_list
        )
        cae_metrics["model"] = "ClassicalAutoencoder"
        cae_metrics["seed"] = seed
        cae_metrics["train_time_sec"] = t_cae
        all_seed_results.append(cae_metrics)
        print(f"    CAE Test: ROC-AUC={cae_metrics['roc_auc']:.4f} | PR-AUC={cae_metrics['pr_auc']:.4f} | "
              f"Recall={cae_metrics['recall']:.4f} | FPR={cae_metrics['fpr']:.4f} | F1-Macro={cae_metrics['f1_macro']:.4f}")

        # -------------------------------------------------------------
        # 2. Quantum Autoencoder (QAE)
        # -------------------------------------------------------------
        print("  Training Quantum Autoencoder (Romero et al. 4-Qubit Compression)...")
        t0 = time.time()
        qae = QuantumAutoencoder(classical_encoder=cae, n_qubits=4, n_layers=2)
        qae_history = qae.fit_model(
            train_loader,
            epochs=10,
            lr=0.02,
            device="cpu",
            seed=seed,
        )
        t_qae = time.time() - t0
        print(f"  QAE Train Complete ({t_qae:.1f}s) | Final Infidelity: {qae_history['train_loss'][-1]:.5f} | "
              f"Mean P(|00>): {qae_history['p00_fidelity'][-1]:.4f}")

        # Compute QAE anomaly scores (Trash state Infidelity = 1 - P(|00>))
        qae_train_scores = qae.compute_anomaly_scores(X_train)
        qae_val_scores = qae.compute_anomaly_scores(X_val)
        qae_test_scores = qae.compute_anomaly_scores(X_test)

        qae_metrics = evaluate_anomaly_detector(
            qae_train_scores, qae_val_scores, y_val, qae_test_scores, y_test, test_cats_list
        )
        qae_metrics["model"] = "QuantumAutoencoder"
        qae_metrics["seed"] = seed
        qae_metrics["train_time_sec"] = t_qae
        all_seed_results.append(qae_metrics)
        print(f"    QAE Test: ROC-AUC={qae_metrics['roc_auc']:.4f} | PR-AUC={qae_metrics['pr_auc']:.4f} | "
              f"Recall={qae_metrics['recall']:.4f} | FPR={qae_metrics['fpr']:.4f} | F1-Macro={qae_metrics['f1_macro']:.4f}")

        # -------------------------------------------------------------
        # 3. Hybrid Ensemble (CAE + QAE)
        # -------------------------------------------------------------
        # Standardize validation scores to scale [0, 1] for balanced combination
        cae_val_mean, cae_val_std = np.mean(cae_val_scores), np.std(cae_val_scores) + 1e-8
        qae_val_mean, qae_val_std = np.mean(qae_val_scores), np.std(qae_val_scores) + 1e-8

        hyb_train_scores = 0.5 * ((cae_train_scores - cae_val_mean) / cae_val_std) + \
                           0.5 * ((qae_train_scores - qae_val_mean) / qae_val_std)
        hyb_val_scores = 0.5 * ((cae_val_scores - cae_val_mean) / cae_val_std) + \
                         0.5 * ((qae_val_scores - qae_val_mean) / qae_val_std)
        hyb_test_scores = 0.5 * ((cae_test_scores - cae_val_mean) / cae_val_std) + \
                          0.5 * ((qae_test_scores - qae_val_mean) / qae_val_std)

        hyb_metrics = evaluate_anomaly_detector(
            hyb_train_scores, hyb_val_scores, y_val, hyb_test_scores, y_test, test_cats_list
        )
        hyb_metrics["model"] = "HybridEnsemble"
        hyb_metrics["seed"] = seed
        hyb_metrics["train_time_sec"] = t_cae + t_qae
        all_seed_results.append(hyb_metrics)
        print(f"    HYB Test: ROC-AUC={hyb_metrics['roc_auc']:.4f} | PR-AUC={hyb_metrics['pr_auc']:.4f} | "
              f"Recall={hyb_metrics['recall']:.4f} | FPR={hyb_metrics['fpr']:.4f} | F1-Macro={hyb_metrics['f1_macro']:.4f}")

        # Store curves for plotting
        fpr_c, tpr_c, _ = roc_curve(y_test, cae_test_scores)
        fpr_q, tpr_q, _ = roc_curve(y_test, qae_test_scores)
        fpr_h, tpr_h, _ = roc_curve(y_test, hyb_test_scores)
        roc_curves_data["ClassicalAutoencoder"].append((fpr_c, tpr_c, cae_metrics["roc_auc"]))
        roc_curves_data["QuantumAutoencoder"].append((fpr_q, tpr_q, qae_metrics["roc_auc"]))
        roc_curves_data["HybridEnsemble"].append((fpr_h, tpr_h, hyb_metrics["roc_auc"]))

        if seed_idx == 0:
            plot_data_first_seed = {
                "y_test": y_test,
                "test_cats": test_cats_list,
                "cae_test_scores": cae_test_scores,
                "qae_test_scores": qae_test_scores,
                "hyb_test_scores": hyb_test_scores,
                "cae_tau": cae_metrics["tau_95"],
                "qae_tau": qae_metrics["tau_95"],
            }

    # -------------------------------------------------------------
    # Statistical Aggregation
    # -------------------------------------------------------------
    results_df = pd.DataFrame(all_seed_results)
    results_df.to_csv("results/zeroday_results.csv", index=False)

    summary: Dict[str, Any] = {
        "metadata": {
            "seeds": SEEDS,
            "train_set": "100% Benign (Normal, n=1100)",
            "val_set": "Benign (n=150) + Known Attack (Exploits, n=150)",
            "test_set": "Benign (n=347) + Unseen Zero-Days (n=1304)",
            "unseen_zero_day_categories": meta["zero_day_categories"],
            "threshold_rule": "tau_95: 95th percentile of benign validation reconstruction error",
        },
        "models": {},
    }

    numeric_cols = [
        "roc_auc", "pr_auc", "accuracy", "precision", "recall",
        "f1_binary", "f1_macro", "fpr", "specificity", "train_time_sec"
    ]

    for model_name in ["ClassicalAutoencoder", "QuantumAutoencoder", "HybridEnsemble"]:
        sub_df = results_df[results_df["model"] == model_name]
        model_summary = {}
        for col in numeric_cols:
            vals = sub_df[col].values
            model_summary[f"{col}_mean"] = float(np.mean(vals))
            model_summary[f"{col}_std"] = float(np.std(vals))

        # Per-category detection rate aggregation
        per_cat_summary = {}
        first_entry = sub_df.iloc[0]["per_category"]
        for cat in first_entry.keys():
            rates = [row["per_category"][cat]["detection_rate"] for _, row in sub_df.iterrows()]
            totals = [row["per_category"][cat]["total_samples"] for _, row in sub_df.iterrows()]
            per_cat_summary[cat] = {
                "samples": int(totals[0]),
                "detection_rate_mean": float(np.mean(rates)),
                "detection_rate_std": float(np.std(rates)),
            }
        model_summary["per_category"] = per_cat_summary
        summary["models"][model_name] = model_summary

    with open("results/zeroday_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 3 ZERO-DAY BENCHMARK SUMMARY (5-Seed Mean ± Std):")
    print("=" * 80)
    for model_name, m_stats in summary["models"].items():
        print(f"[{model_name}]")
        print(f"  ROC-AUC:   {m_stats['roc_auc_mean']:.4f} ± {m_stats['roc_auc_std']:.4f}")
        print(f"  PR-AUC:    {m_stats['pr_auc_mean']:.4f} ± {m_stats['pr_auc_std']:.4f}")
        print(f"  Recall:    {m_stats['recall_mean']:.4f} ± {m_stats['recall_std']:.4f} (Zero-Day Detection Rate)")
        print(f"  FPR:       {m_stats['fpr_mean']:.4f} ± {m_stats['fpr_std']:.4f}")
        print(f"  Macro-F1:  {m_stats['f1_macro_mean']:.4f} ± {m_stats['f1_macro_std']:.4f}")
        print(f"  Train Time:{m_stats['train_time_sec_mean']:.1f}s")
        print("  Zero-Day Detection Breakdown:")
        for cat, c_info in m_stats["per_category"].items():
            if cat != "Normal":
                print(f"    - {cat:<15}: {c_info['detection_rate_mean']*100:.1f}% ± {c_info['detection_rate_std']*100:.1f}% (n={c_info['samples']})")
        print(f"    - Normal (FPR)  : {m_stats['per_category']['Normal']['detection_rate_mean']*100:.1f}%\n")

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    print("Generating Publication Figures...")
    generate_figures(plot_data_first_seed, roc_curves_data, summary)
    print("Benchmark complete. Artifacts saved to results/.")


def generate_figures(
    plot_data: Dict[str, Any],
    roc_curves: Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]],
    summary: Dict[str, Any],
):
    """
    Renders high-resolution scientific figures conforming to IEEE style.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Figure 1: Reconstruction Loss / Infidelity Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    cats = np.array(plot_data["test_cats"])
    benign_mask = (cats == "Normal")
    rare_attacks_mask = np.isin(cats, ["Worms", "Shellcode", "Analysis", "Backdoor"])
    volumetric_mask = np.isin(cats, ["DoS", "Fuzzers", "Reconnaissance"])

    # Panel A: Classical Autoencoder MSE
    ax1 = axes[0]
    cae_scores = plot_data["cae_test_scores"]
    ax1.hist(cae_scores[benign_mask], bins=35, alpha=0.6, color="#2ca02c", density=True, label="Benign (Normal)")
    ax1.hist(cae_scores[rare_attacks_mask], bins=35, alpha=0.6, color="#d62728", density=True, label="Rare Zero-Days (Worms/Shellcode/Backdoor)")
    ax1.hist(cae_scores[volumetric_mask], bins=35, alpha=0.5, color="#1f77b4", density=True, label="Volumetric Zero-Days (DoS/Fuzzers/Recon)")
    ax1.axvline(plot_data["cae_tau"], color="black", linestyle="--", linewidth=1.8, label=f"Decision Threshold $\\tau_{{95}}$ ({plot_data['cae_tau']:.3f})")
    ax1.set_title("(a) Classical Autoencoder Reconstruction MSE", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Reconstruction Loss $\\mathcal{L}_{\\mathrm{MSE}}$", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel B: Quantum Autoencoder Trash Infidelity
    ax2 = axes[1]
    qae_scores = plot_data["qae_test_scores"]
    ax2.hist(qae_scores[benign_mask], bins=35, alpha=0.6, color="#2ca02c", density=True, label="Benign (Normal)")
    ax2.hist(qae_scores[rare_attacks_mask], bins=35, alpha=0.6, color="#d62728", density=True, label="Rare Zero-Days (Worms/Shellcode/Backdoor)")
    ax2.hist(qae_scores[volumetric_mask], bins=35, alpha=0.5, color="#9467bd", density=True, label="Volumetric Zero-Days (DoS/Fuzzers/Recon)")
    ax2.axvline(plot_data["qae_tau"], color="black", linestyle="--", linewidth=1.8, label=f"Decision Threshold $\\tau_{{95}}$ ({plot_data['qae_tau']:.3f})")
    ax2.set_title("(b) Quantum Autoencoder Trash Infidelity $1 - P(|00\\rangle)$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Quantum Trash Infidelity $\\mathcal{L}_{\\mathrm{QAE}}$", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig_path1 = "results/figures/reconstruction_loss_distribution.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path1}")

    # -------------------------------------------------------------
    # Figure 2: Zero-Day ROC Curves with Confidence Bands
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    model_colors = {
        "ClassicalAutoencoder": "#1f77b4",
        "QuantumAutoencoder": "#9467bd",
        "HybridEnsemble": "#d62728",
    }
    model_labels = {
        "ClassicalAutoencoder": "Classical Autoencoder (CAE)",
        "QuantumAutoencoder": "Quantum Autoencoder (QAE)",
        "HybridEnsemble": "Hybrid Ensemble (CAE + QAE)",
    }

    base_fpr = np.linspace(0, 1, 200)

    for model_name, runs in roc_curves.items():
        tprs = []
        aucs = []
        for (fpr_curve, tpr_curve, auc_val) in runs:
            # Interpolate TPR across common FPR grid
            tpr_interp = np.interp(base_fpr, fpr_curve, tpr_curve)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(auc_val)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(
            base_fpr,
            mean_tpr,
            color=model_colors[model_name],
            linewidth=2.2,
            label=f"{model_labels[model_name]} (AUC = {mean_auc:.4f} $\\pm$ {std_auc:.4f})",
        )
        ax.fill_between(
            base_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color=model_colors[model_name],
            alpha=0.15,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("Zero-Day Anomaly Detection ROC Curves (5-Seed Benchmark)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    fig_path2 = "results/figures/zeroday_roc_curve.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path2}")

    # -------------------------------------------------------------
    # Figure 3: Per-Category Zero-Day Detection Rate
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    cats_to_plot = [c for c in summary["metadata"]["unseen_zero_day_categories"] if c in summary["models"]["ClassicalAutoencoder"]["per_category"]]
    cats_to_plot = sorted(cats_to_plot)

    x = np.arange(len(cats_to_plot))
    width = 0.26

    cae_rates = [summary["models"]["ClassicalAutoencoder"]["per_category"][c]["detection_rate_mean"] * 100 for c in cats_to_plot]
    cae_errs = [summary["models"]["ClassicalAutoencoder"]["per_category"][c]["detection_rate_std"] * 100 for c in cats_to_plot]

    qae_rates = [summary["models"]["QuantumAutoencoder"]["per_category"][c]["detection_rate_mean"] * 100 for c in cats_to_plot]
    qae_errs = [summary["models"]["QuantumAutoencoder"]["per_category"][c]["detection_rate_std"] * 100 for c in cats_to_plot]

    hyb_rates = [summary["models"]["HybridEnsemble"]["per_category"][c]["detection_rate_mean"] * 100 for c in cats_to_plot]
    hyb_errs = [summary["models"]["HybridEnsemble"]["per_category"][c]["detection_rate_std"] * 100 for c in cats_to_plot]

    ax.bar(x - width, cae_rates, width, yerr=cae_errs, capsize=4, label="Classical AE", color="#1f77b4", alpha=0.85)
    ax.bar(x, qae_rates, width, yerr=qae_errs, capsize=4, label="Quantum AE (Romero et al.)", color="#9467bd", alpha=0.85)
    ax.bar(x + width, hyb_rates, width, yerr=hyb_errs, capsize=4, label="Hybrid Ensemble", color="#d62728", alpha=0.85)

    ax.set_ylabel("Zero-Day Detection Rate (%)", fontsize=11)
    ax.set_title("Unseen Threat Detection Rates by Attack Family ($\tau_{95}$ Threshold)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    # Add sample counts to x tick labels
    xtick_labels = [f"{c}\n(n={summary['models']['ClassicalAutoencoder']['per_category'][c]['samples']})" for c in cats_to_plot]
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylim([0, 105])
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    fig_path3 = "results/figures/zeroday_detection_by_category.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path3}")


if __name__ == "__main__":
    run_benchmark()
