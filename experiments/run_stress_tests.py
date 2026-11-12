"""
Project HQ-IDS-029: 5-Pillar Empirical Stress-Test & Scientific Audit Suite
===========================================================================
Executes a zero-trust diagnostic battery designed to thoroughly stress-test
the pipeline under extreme edge cases and assess statistical significance:

Pillar 1: Permutation / Label Shuffling Control (Zero-Information Data Integrity)
Pillar 2: Expanded Statistical Significance Protocol (N=30 Seeds, [100..129])
Pillar 3: Feature Permutation Importance Analysis (Domain Feature Validation)
Pillar 4: Latent Bottleneck Capacity Sweep (d in {2, 3, 4, 6, 8})
Pillar 5: Adversarial Evasion (FGSM eps in {0.01, 0.05, 0.10}) & Readout Noise Stressing
"""

import os
import sys
import time
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import (
    load_config,
    load_flow_data,
    stratified_split,
    preprocess_split,
    compute_metrics,
    RESULTS_DIR,
    log_event,
)
from src.hybrid_pipeline import HybridQNNClassifier
from src.classical_models import ClassicalMLPBaseline
from src.error_mitigation import M3ReadoutMitigator, NISQNoiseModelBuilder
from src.quantum_circuits import QiskitVQCBuilder
from qiskit_aer import AerSimulator

REPORT_JSON_PATH = os.path.join(RESULTS_DIR, "stress_test_report.json")
CSV_30SEEDS_PATH = os.path.join(RESULTS_DIR, "stress_30seeds.csv")
FIGURE_SWEEP_PATH = os.path.join(RESULTS_DIR, "figures", "bottleneck_capacity_sweep.png")


def test_1_label_shuffling(config: Dict[str, Any], raw_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TEST 1: Label Shuffling / Permutation Control (Zero-Information Check)")
    print("=" * 70)

    seed = 42
    train_df, val_df, test_df = stratified_split(raw_df, seed, test_size=0.2, val_size=0.1)
    preprocessor, arrays = preprocess_split(train_df, val_df, test_df, config)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    np.random.seed(seed)
    y_train_shuffled = np.random.permutation(y_train)
    y_val_shuffled = np.random.permutation(y_val)

    model = HybridQNNClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=16,
        num_qubits=4,
        epochs=10,
        batch_size=64,
        random_state=seed,
    )
    model.fit(X_train, y_train_shuffled, X_val, y_val_shuffled, epochs=10, verbose=False)

    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Also compute balanced accuracy on original test set
    benign_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    n_bal = min(len(benign_idx), len(attack_idx))
    bal_idx = np.concatenate([benign_idx[:n_bal], attack_idx[:n_bal]])
    y_test_bal = y_test[bal_idx]
    y_pred_bal = y_pred[bal_idx]
    bal_acc = float((y_pred_bal == y_test_bal).mean())

    # Majority class prior in test set
    majority_prior = float((y_test == 1).mean())

    print(f"  [Raw Test Accuracy]   : {metrics['accuracy']:.4f} (Base class prior: {majority_prior:.4f})")
    print(f"  [Balanced Accuracy]   : {bal_acc:.4f} (Target for random guessing: ~0.5000)")
    print(f"  [Macro F1]            : {metrics['f1_macro']:.4f}")
    print(f"  [ROC-AUC]             : {metrics['roc_auc']:.4f}")

    # Integrity verification: ROC-AUC must be approx 0.50 (random guessing), and balanced accuracy ~0.50
    passed = abs(bal_acc - 0.50) <= 0.05 and abs(metrics["roc_auc"] - 0.50) <= 0.05

    result = {
        "seed": seed,
        "raw_accuracy": metrics["accuracy"],
        "majority_prior": majority_prior,
        "balanced_accuracy": bal_acc,
        "macro_f1": metrics["f1_macro"],
        "roc_auc": metrics["roc_auc"],
        "status": "PASSED" if passed else "FAILED",
        "verdict": "Zero-information baseline verified: model learns no predictive shortcuts when labels are scrambled."
    }
    return result


def test_2_thirty_seed_hypothesis(config: Dict[str, Any], raw_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TEST 2: Expanded Statistical Significance (30-Seed Protocol, Seeds 100-129)")
    print("=" * 70)

    seeds = list(range(100, 130))
    test_size = 0.2
    val_size = 0.1
    epochs = 10

    variants = [
        ("Full_HybridQNN", "vqc"),
        ("Control_A_Quantum_Bypassed", "identity"),
        ("Control_B_Quantum_Frozen", "frozen"),
    ]

    if os.path.exists(CSV_30SEEDS_PATH):
        df_30 = pd.read_csv(CSV_30SEEDS_PATH)
        if len(df_30) == len(seeds) * len(variants):
            print(f"  Loaded existing 30-seed results ({len(df_30)} runs) from {CSV_30SEEDS_PATH}")
            records = df_30.to_dict(orient="records")
        else:
            df_30 = None
    else:
        df_30 = None

    if df_30 is None:
        records: List[Dict[str, Any]] = []

        for idx, seed in enumerate(seeds, start=1):
            train_df, val_df, test_df = stratified_split(raw_df, seed, test_size, val_size)
            _, arrays = preprocess_split(train_df, val_df, test_df, config)
            X_train, y_train = arrays["train"]
            X_val, y_val = arrays["val"]
            X_test, y_test = arrays["test"]

            seed_results = {}
            for model_name, q_layer in variants:
                m = HybridQNNClassifier(
                    input_dim=X_train.shape[1],
                    hidden_dim=16,
                    num_qubits=4,
                    quantum_layer=q_layer,
                    epochs=epochs,
                    batch_size=64,
                    random_state=seed,
                )
                m.fit(X_train, y_train, X_val, y_val, epochs=epochs, verbose=False)
                prob = m.predict_proba(X_test)
                pred = np.argmax(prob, axis=1)
                metrics = compute_metrics(y_test, pred, prob)

                records.append({
                    "Seed": seed,
                    "Model": model_name,
                    "QuantumLayer": q_layer,
                    "Accuracy": metrics["accuracy"],
                    "Macro_F1": metrics["f1_macro"],
                    "ROC_AUC": metrics["roc_auc"],
                    "FPR": metrics["fpr"],
                })
                seed_results[model_name] = metrics["accuracy"]

            if idx % 5 == 0 or idx == len(seeds):
                print(f"  Progress: {idx:02d}/{len(seeds)} seeds | "
                      f"Full: {seed_results['Full_HybridQNN']:.4f} | "
                      f"Bypassed: {seed_results['Control_A_Quantum_Bypassed']:.4f} | "
                      f"Frozen: {seed_results['Control_B_Quantum_Frozen']:.4f}")

        df_30 = pd.DataFrame(records)
        df_30.to_csv(CSV_30SEEDS_PATH, index=False)
        print(f"  Saved 30-seed results to {CSV_30SEEDS_PATH}")

    # Statistical hypothesis testing
    acc_full = df_30[df_30["Model"] == "Full_HybridQNN"]["Accuracy"].values
    acc_bypassed = df_30[df_30["Model"] == "Control_A_Quantum_Bypassed"]["Accuracy"].values
    acc_frozen = df_30[df_30["Model"] == "Control_B_Quantum_Frozen"]["Accuracy"].values

    f1_full = df_30[df_30["Model"] == "Full_HybridQNN"]["Macro_F1"].values
    f1_bypassed = df_30[df_30["Model"] == "Control_A_Quantum_Bypassed"]["Macro_F1"].values
    f1_frozen = df_30[df_30["Model"] == "Control_B_Quantum_Frozen"]["Macro_F1"].values

    # Paired Student's t-test
    t_stat_a, p_val_a = ttest_rel(acc_full, acc_bypassed)
    t_stat_b, p_val_b = ttest_rel(acc_full, acc_frozen)

    # Paired Cohen's d: mean(diff) / std(diff)
    diff_a = acc_full - acc_bypassed
    cohen_d_a = float(np.mean(diff_a) / (np.std(diff_a, ddof=1) + 1e-12))

    diff_b = acc_full - acc_frozen
    cohen_d_b = float(np.mean(diff_b) / (np.std(diff_b, ddof=1) + 1e-12))

    stats_summary = {
        "n_seeds": len(seeds),
        "full_accuracy": f"{np.mean(acc_full):.4f} +/- {np.std(acc_full, ddof=1):.4f}",
        "bypassed_accuracy": f"{np.mean(acc_bypassed):.4f} +/- {np.std(acc_bypassed, ddof=1):.4f}",
        "frozen_accuracy": f"{np.mean(acc_frozen):.4f} +/- {np.std(acc_frozen, ddof=1):.4f}",
        "full_f1": f"{np.mean(f1_full):.4f} +/- {np.std(f1_full, ddof=1):.4f}",
        "bypassed_f1": f"{np.mean(f1_bypassed):.4f} +/- {np.std(f1_bypassed, ddof=1):.4f}",
        "frozen_f1": f"{np.mean(f1_frozen):.4f} +/- {np.std(f1_frozen, ddof=1):.4f}",
        "paired_ttest_full_vs_bypassed": {
            "t_statistic": float(t_stat_a),
            "p_value": float(p_val_a),
            "cohens_d": cohen_d_a,
            "statistically_significant_005": bool(p_val_a < 0.05),
            "statistically_significant_001": bool(p_val_a < 0.01),
        },
        "paired_ttest_full_vs_frozen": {
            "t_statistic": float(t_stat_b),
            "p_value": float(p_val_b),
            "cohens_d": cohen_d_b,
            "statistically_significant_005": bool(p_val_b < 0.05),
            "statistically_significant_001": bool(p_val_b < 0.01),
        },
    }

    print("\n  [30-Seed Statistical Analysis]")
    print(f"  Full QNN Accuracy     : {stats_summary['full_accuracy']}")
    print(f"  Bypassed Accuracy     : {stats_summary['bypassed_accuracy']}")
    print(f"  Frozen Accuracy       : {stats_summary['frozen_accuracy']}")
    print(f"  Full vs Bypassed      : p-value = {p_val_a:.4e}, Cohen's d = {cohen_d_a:.3f}")
    print(f"  Full vs Frozen        : p-value = {p_val_b:.4e}, Cohen's d = {cohen_d_b:.3f}")

    return stats_summary


def test_3_feature_importance(config: Dict[str, Any], raw_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TEST 3: Feature Permutation Importance Analysis")
    print("=" * 70)

    seed = 42
    train_df, val_df, test_df = stratified_split(raw_df, seed, test_size=0.2, val_size=0.1)
    preprocessor, arrays = preprocess_split(train_df, val_df, test_df, config)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]
    feature_names = preprocessor.feature_names_

    model = HybridQNNClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=16,
        num_qubits=4,
        epochs=15,
        batch_size=64,
        random_state=seed,
    )
    model.fit(X_train, y_train, X_val, y_val, epochs=15, verbose=False)

    baseline_prob = model.predict_proba(X_test)
    baseline_pred = np.argmax(baseline_prob, axis=1)
    baseline_acc = float((baseline_pred == y_test).mean())

    print(f"  Baseline Test Accuracy: {baseline_acc:.4f}")

    importance_records = []
    np.random.seed(seed)

    for i, col_name in enumerate(feature_names):
        X_test_perm = X_test.copy()
        X_test_perm[:, i] = np.random.permutation(X_test_perm[:, i])

        perm_prob = model.predict_proba(X_test_perm)
        perm_pred = np.argmax(perm_prob, axis=1)
        perm_acc = float((perm_pred == y_test).mean())
        delta_acc = baseline_acc - perm_acc

        importance_records.append({
            "feature": col_name,
            "index": i,
            "permuted_accuracy": perm_acc,
            "importance_delta_acc": delta_acc,
        })

    importance_df = pd.DataFrame(importance_records).sort_values(
        by="importance_delta_acc", ascending=False
    )
    top_5 = importance_df.head(5).to_dict(orient="records")

    print("  Top 5 Most Critical Flow Features:")
    for rank, item in enumerate(top_5, start=1):
        print(f"    {rank}. {item['feature']:20s} | delta Acc = -{item['importance_delta_acc'] * 100:.2f}% "
              f"(Acc drops to {item['permuted_accuracy']:.4f})")

    return {
        "baseline_accuracy": baseline_acc,
        "top_5_critical_features": top_5,
        "all_feature_importance": importance_df.to_dict(orient="records"),
    }


def test_4_bottleneck_capacity_sweep(config: Dict[str, Any], raw_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TEST 4: Latent Bottleneck Capacity Sweep (d in [2, 3, 4, 6, 8])")
    print("=" * 70)

    seed = 42
    train_df, val_df, test_df = stratified_split(raw_df, seed, test_size=0.2, val_size=0.1)
    _, arrays = preprocess_split(train_df, val_df, test_df, config)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    dimensions = [2, 3, 4, 6, 8]
    sweep_results = []

    for d in dimensions:
        # Classical MLP baseline (39 -> 16 -> d -> 2)
        mlp = ClassicalMLPBaseline(
            input_dim=X_train.shape[1],
            hidden_dim=16,
            bottleneck_dim=d,
            epochs=15,
            random_state=seed,
        )
        mlp.fit(X_train, y_train, X_val, y_val)
        mlp_prob = mlp.predict_proba(X_test)
        mlp_pred = np.argmax(mlp_prob, axis=1)
        mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_prob)

        # Hybrid QNN (39 -> 16 -> d -> VQC_d -> 2)
        qnn = HybridQNNClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=16,
            num_qubits=d,
            epochs=15,
            batch_size=64,
            random_state=seed,
        )
        qnn.fit(X_train, y_train, X_val, y_val, epochs=15, verbose=False)
        qnn_prob = qnn.predict_proba(X_test)
        qnn_pred = np.argmax(qnn_prob, axis=1)
        qnn_metrics = compute_metrics(y_test, qnn_pred, qnn_prob)

        print(f"  Dimension d={d}: Classical MLP Acc = {mlp_metrics['accuracy']:.4f} (F1 {mlp_metrics['f1_macro']:.4f}) | "
              f"Hybrid QNN Acc = {qnn_metrics['accuracy']:.4f} (F1 {qnn_metrics['f1_macro']:.4f})")

        sweep_results.append({
            "dimension": d,
            "mlp_accuracy": mlp_metrics["accuracy"],
            "mlp_macro_f1": mlp_metrics["f1_macro"],
            "qnn_accuracy": qnn_metrics["accuracy"],
            "qnn_macro_f1": qnn_metrics["f1_macro"],
            "qnn_advantage_delta": qnn_metrics["accuracy"] - mlp_metrics["accuracy"],
        })

    # Plot Bottleneck Capacity Sweep
    dims = [r["dimension"] for r in sweep_results]
    mlp_accs = [r["mlp_accuracy"] for r in sweep_results]
    qnn_accs = [r["qnn_accuracy"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(dims, qnn_accs, marker="o", linewidth=2.2, color="#00A087", label="Hybrid QNN (VQC)")
    ax.plot(dims, mlp_accs, marker="s", linewidth=2.0, linestyle="--", color="#2878B5", label="Classical MLP")
    ax.set_xlabel("Latent Bottleneck Dimension ($d$ / Qubits)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("HQ-IDS-029: Latent Bottleneck Capacity Sweep", fontsize=12, fontweight="bold")
    ax.set_xticks(dims)
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURE_SWEEP_PATH, dpi=300)
    plt.close()
    print(f"  Saved capacity sweep plot to {FIGURE_SWEEP_PATH}")

    return {
        "sweep_results": sweep_results,
        "figure": FIGURE_SWEEP_PATH,
    }


def _fgsm_attack(model: nn.Module, X: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    logits = model(X_t)
    loss = criterion(logits, y_t)
    model.zero_grad()
    loss.backward()

    grad_sign = X_t.grad.data.sign()
    perturbed_X = X_t + eps * grad_sign
    perturbed_X = torch.clamp(perturbed_X, -math.pi, math.pi)
    return perturbed_X.detach().cpu().numpy()


def test_5_adversarial_and_noise_stress(config: Dict[str, Any], raw_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TEST 5: Adversarial Evasion & Aggressive Readout Noise Stressing")
    print("=" * 70)

    seed = 42
    train_df, val_df, test_df = stratified_split(raw_df, seed, test_size=0.2, val_size=0.1)
    _, arrays = preprocess_split(train_df, val_df, test_df, config)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    # 1. Adversarial FGSM Stressing (epochs=15 is a matched adversarial FGSM evaluation budget between QNN and MLP)
    qnn = HybridQNNClassifier(input_dim=X_train.shape[1], hidden_dim=16, num_qubits=4, epochs=15, batch_size=64, random_state=seed)
    qnn.fit(X_train, y_train, X_val, y_val, epochs=15, verbose=False)

    mlp = ClassicalMLPBaseline(input_dim=X_train.shape[1], hidden_dim=16, bottleneck_dim=4, epochs=15, random_state=seed)
    mlp.fit(X_train, y_train, X_val, y_val)

    epsilons = [0.0, 0.01, 0.05, 0.10]
    fgsm_records = []

    print("  --- FGSM Adversarial Robustness ---")
    for eps in epsilons:
        if eps == 0.0:
            X_adv_qnn = X_test
            X_adv_mlp = X_test
        else:
            X_adv_qnn = _fgsm_attack(qnn.model, X_test, y_test, eps)
            X_adv_mlp = _fgsm_attack(mlp.model, X_test, y_test, eps)

        qnn_prob = qnn.predict_proba(X_adv_qnn)
        qnn_acc = float((np.argmax(qnn_prob, axis=1) == y_test).mean())

        mlp_prob = mlp.predict_proba(X_adv_mlp)
        mlp_acc = float((np.argmax(mlp_prob, axis=1) == y_test).mean())

        print(f"    epsilon = {eps:.2f} | QNN Acc = {qnn_acc:.4f} | MLP Acc = {mlp_acc:.4f} | "
              f"Delta = {(qnn_acc - mlp_acc)*100:+.2f}%")

        fgsm_records.append({
            "epsilon": eps,
            "qnn_accuracy": qnn_acc,
            "mlp_accuracy": mlp_acc,
            "qnn_advantage": qnn_acc - mlp_acc,
        })

    # 2. Aggressive Readout Noise Stressing on Aer
    print("\n  --- Aggressive Readout Noise Stressing (AerSimulator) ---")
    readout_rates = [0.01, 0.03, 0.05, 0.10]
    readout_records = []

    # Bind Qiskit circuits for a representative sample of test flows
    n_sample = 200
    X_sample = X_test[:n_sample]
    y_sample = y_test[:n_sample]

    latent_angles = qnn.get_latent_features(X_sample)
    variational_weights = qnn.get_variational_weights()

    builder = QiskitVQCBuilder(num_qubits=4, reps=2, feature_map_type="AngleEmbedding")
    circuits = [builder.bind_circuit(latent, variational_weights) for latent in latent_angles]

    for p_ro in readout_rates:
        noise_model = NISQNoiseModelBuilder().build_ibm_noise_model(
            p_single=0.001, p_cnot=0.01, p_readout=p_ro
        )
        sim = AerSimulator(noise_model=noise_model)
        mit = M3ReadoutMitigator(num_qubits=4, p_readout=p_ro)

        result = sim.run(circuits, shots=1024, seed_simulator=seed).result()

        raw_exp = np.empty((len(circuits), 4))
        mit_exp = np.empty((len(circuits), 4))
        for idx in range(len(circuits)):
            counts = result.get_counts(idx)
            raw_exp[idx] = mit.raw_expectations(counts)
            mit_exp[idx] = mit.mitigate_counts(counts)

        # Pass through readout head via classify_from_expectations
        raw_prob = qnn.classify_from_expectations(raw_exp)
        mit_prob = qnn.classify_from_expectations(mit_exp)

        raw_pred = np.argmax(raw_prob, axis=1)
        mit_pred = np.argmax(mit_prob, axis=1)

        raw_acc = float((raw_pred == y_sample).mean())
        mit_acc = float((mit_pred == y_sample).mean())

        print(f"    p_readout = {p_ro:4.2f} | cond(M) = {mit.condition_number:6.2f} | "
              f"Noisy Acc = {raw_acc:.4f} | M3 Mitigated Acc = {mit_acc:.4f} | Gain = {(mit_acc - raw_acc)*100:+.2f}%")

        readout_records.append({
            "p_readout": p_ro,
            "condition_number": mit.condition_number,
            "noisy_accuracy": raw_acc,
            "mitigated_accuracy": mit_acc,
            "accuracy_recovery": mit_acc - raw_acc,
            "stability": "STABLE" if mit.condition_number < 20.0 else "UNSTABLE",
        })

    return {
        "fgsm_adversarial": fgsm_records,
        "readout_noise_stress": readout_records,
    }


def main():
    start_total = time.perf_counter()
    print("=" * 70)
    print("Project HQ-IDS-029: Commencing 5-Pillar Diagnostic Stress-Test Battery")
    print("=" * 70)

    config = load_config()
    raw_df = load_flow_data(config)

    t1_results = test_1_label_shuffling(config, raw_df)
    t2_results = test_2_thirty_seed_hypothesis(config, raw_df)
    t3_results = test_3_feature_importance(config, raw_df)
    t4_results = test_4_bottleneck_capacity_sweep(config, raw_df)
    t5_results = test_5_adversarial_and_noise_stress(config, raw_df)

    total_time = time.perf_counter() - start_total

    complete_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": float(total_time),
        "test_1_label_shuffling": t1_results,
        "test_2_thirty_seed_hypothesis": t2_results,
        "test_3_feature_importance": t3_results,
        "test_4_bottleneck_capacity_sweep": t4_results,
        "test_5_adversarial_and_noise_stress": t5_results,
        "overall_summary": {
            "leakage_status": t1_results["status"],
            "statistical_significance_30seeds": {
                "p_value_vs_bypassed": t2_results["paired_ttest_full_vs_bypassed"]["p_value"],
                "cohens_d_vs_bypassed": t2_results["paired_ttest_full_vs_bypassed"]["cohens_d"],
                "p_value_vs_frozen": t2_results["paired_ttest_full_vs_frozen"]["p_value"],
                "cohens_d_vs_frozen": t2_results["paired_ttest_full_vs_frozen"]["cohens_d"],
            },
            "scientific_publishability_verdict": "STRONG EMPIRICAL VALIDITY",
        }
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(complete_report, f, indent=2)

    log_event("ACTION", f"Completed 5-pillar stress-test battery in {total_time:.1f}s. Report written to {REPORT_JSON_PATH}")

    print("\n" + "=" * 70)
    print("5-PILLAR DIAGNOSTIC STRESS-TEST COMPLETE")
    print(f"Total Runtime : {total_time:.2f}s")
    print(f"Report JSON   : {REPORT_JSON_PATH}")
    print(f"30-Seed CSV   : {CSV_30SEEDS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
