"""
Project HQ-IDS-029: Step 3 - Quantum Ablation & True Aer NISQ Noise Simulation
=============================================================================

Part A - Ablation protocol (3 variants x 5 seeds = 15 rows)
  1. Full_HybridQNN             : trained encoder -> trained 4-qubit VQC -> linear head
  2. Control_A_Quantum_Bypassed : trained encoder -> linear head (no quantum layer)
  3. Control_B_Quantum_Frozen   : trained encoder -> untrained frozen random VQC -> linear head

All three variants share the *same* module, optimiser, schedule, batch size and
initialisation seed (`quantum_layer` selects the variant inside
`HybridQNNClassifier`), so any measured difference is attributable to the quantum
layer rather than to a training-protocol confound.

Part B - True shot-based Qiskit Aer NISQ simulation
  The trained VQC weights and the test-set latent angles are replayed through
  `AerSimulator` with 1024 shots under three conditions:
    * Noiseless   : analytic-free, ideal Aer execution (finite-shot only)
    * Noisy       : IBM-style noise model (p1 = 0.001, p2 = 0.01, p_readout = 0.03)
    * Mitigated   : the *same* noisy counts, corrected by M3 matrix-inversion
                    readout mitigation (exact inversion of the 2^4 x 2^4
                    assignment matrix)

  Raw Pauli-Z values are <Z_i> = (N_0 - N_1) / shots. The PennyLane and Qiskit
  circuits are verified numerically identical, so this replay is faithful to the
  trained model.

Outputs:
  results/ablation_results.csv  15 rows (3 variants x 5 seeds)
  results/noise_results.csv     5 rows (one per seed, 3 conditions each)
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qiskit_aer import AerSimulator  # noqa: E402

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
from src.error_mitigation import M3ReadoutMitigator, NISQNoiseModelBuilder  # noqa: E402
from src.hybrid_pipeline import HybridQNNClassifier  # noqa: E402
from src.quantum_circuits import QiskitVQCBuilder  # noqa: E402

VARIANTS = [
    ("Full_HybridQNN", "vqc"),
    ("Control_A_Quantum_Bypassed", "identity"),
    ("Control_B_Quantum_Frozen", "frozen"),
]


def _train_variant(
    quantum_layer: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    seed: int,
) -> HybridQNNClassifier:
    """Trains one ablation variant under the shared, identical training protocol."""
    qc_cfg = config.get("quantum_circuit", {})
    hyb_cfg = config.get("hybrid_classifier", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HybridQNNClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=config.get("classical_encoder", {}).get("hidden_dim", 16),
        num_qubits=qc_cfg.get("num_qubits", 4),
        num_layers=qc_cfg.get("reps", 2),
        output_dim=2,
        ansatz_type=qc_cfg.get("ansatz", "RealAmplitudes"),
        feature_map=qc_cfg.get("feature_map", "AngleEmbedding"),
        feature_map_reps=qc_cfg.get("feature_map_reps", 1),
        quantum_layer=quantum_layer,
        learning_rate=hyb_cfg.get("learning_rate", 0.01),
        batch_size=hyb_cfg.get("batch_size", 64),
        epochs=hyb_cfg.get("epochs", 20),
        random_state=seed,
    )
    model.fit(X_train, y_train, X_val, y_val, epochs=hyb_cfg.get("epochs", 20), verbose=False)
    return model


def _aer_expectations(
    builder: QiskitVQCBuilder,
    latent_vectors: np.ndarray,
    variational_weights: np.ndarray,
    simulator: AerSimulator,
    shots: int,
    seed_simulator: int,
    mitigator: M3ReadoutMitigator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes every latent vector as a 1024-shot Aer job and returns
    (raw <Z_i>, M3-mitigated <Z_i>) computed directly from the count histograms.
    """
    circuits = [builder.bind_circuit(latent, variational_weights) for latent in latent_vectors]
    result = simulator.run(circuits, shots=shots, seed_simulator=seed_simulator).result()

    raw = np.empty((len(circuits), builder.num_qubits), dtype=np.float64)
    mitigated = np.empty_like(raw)
    for index in range(len(circuits)):
        counts = result.get_counts(index)
        raw[index] = mitigator.raw_expectations(counts)
        mitigated[index] = mitigator.mitigate_counts(counts)
    return raw, mitigated


def run_ablation_and_noise_experiments(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    test_size = config.get("dataset", {}).get("test_size", 0.2)
    val_size = config.get("dataset", {}).get("val_size", 0.1)

    qc_cfg = config.get("quantum_circuit", {})
    num_qubits = qc_cfg.get("num_qubits", 4)
    reps = qc_cfg.get("reps", 2)
    feature_map = qc_cfg.get("feature_map", "AngleEmbedding")
    entanglement = qc_cfg.get("entanglement", "linear")

    noise_cfg = config.get("error_mitigation", {}).get("noise", {})
    p_single = noise_cfg.get("p_single_qubit", 0.001)
    p_cnot = noise_cfg.get("p_cnot", 0.01)
    p_readout = noise_cfg.get("p_readout", 0.03)
    shots = config.get("error_mitigation", {}).get("shots", 1024)
    n_eval = config.get("error_mitigation", {}).get("nisq_eval_samples", 1000)

    noise_model = NISQNoiseModelBuilder().build_ibm_noise_model(
        p_single=p_single, p_cnot=p_cnot, p_readout=p_readout
    )
    noiseless_simulator = AerSimulator()
    noisy_simulator = AerSimulator(noise_model=noise_model)
    mitigator = M3ReadoutMitigator(num_qubits=num_qubits, p_readout=p_readout)
    builder = QiskitVQCBuilder(
        num_qubits=num_qubits, reps=reps,
        entanglement=entanglement, feature_map_type=feature_map,
    )

    print(f"[Aer] shots={shots} | depolarizing p1={p_single}, p2={p_cnot} | readout p={p_readout}")
    print(f"[M3]  assignment matrix {mitigator.M.shape}, condition number {mitigator.condition_number:.4f}")
    print(f"[Aer] Replaying up to {n_eval} test-set flows per seed, 3 conditions")

    raw_df = load_flow_data(config)
    ablation_records: List[Dict[str, Any]] = []
    noise_records: List[Dict[str, Any]] = []

    for seed_idx, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 66)
        print(f"  ABLATION + NISQ {seed_idx}/{len(seeds)}  (seed={seed})")
        print("=" * 66)

        train_df, val_df, test_df = stratified_split(raw_df, seed, test_size, val_size)
        _, arrays = preprocess_split(train_df, val_df, test_df, config)
        X_train, y_train = arrays["train"]
        X_val, y_val = arrays["val"]
        X_test, y_test = arrays["test"]

        trained = {}
        for variant_name, quantum_layer in VARIANTS:
            start = time.perf_counter()
            model = _train_variant(quantum_layer, X_train, y_train, X_val, y_val, config, seed)
            train_time = time.perf_counter() - start

            probabilities = model.predict_proba(X_test)
            metrics = compute_metrics(y_test, np.argmax(probabilities, axis=1), probabilities)

            ablation_records.append({
                "Seed": seed,
                "Model": variant_name,
                "QuantumLayer": quantum_layer,
                "Accuracy": metrics["accuracy"],
                "Precision_Macro": metrics["precision_macro"],
                "Recall_Macro": metrics["recall_macro"],
                "Macro_F1": metrics["f1_macro"],
                "Weighted_F1": metrics["f1_weighted"],
                "ROC_AUC": metrics["roc_auc"],
                "FPR": metrics["fpr"],
                "TN": metrics["tn"], "FP": metrics["fp"],
                "FN": metrics["fn"], "TP": metrics["tp"],
                "Train_Time_Sec": float(train_time),
            })
            trained[variant_name] = model
            print(f"  [{variant_name:27s}] Acc {metrics['accuracy']:.4f} | F1-Macro {metrics['f1_macro']:.4f} | "
                  f"ROC-AUC {metrics['roc_auc']:.4f} | FPR {metrics['fpr']:.4f} | {train_time:.1f}s")

        # ---------------- Part B: true shot-based Aer NISQ simulation ----------------
        full_model = trained["Full_HybridQNN"]
        latents = full_model.get_latent_features(X_test)[:n_eval]
        weights = full_model.get_variational_weights()
        y_subset = y_test[:len(latents)]
        analytic = full_model.get_quantum_expectation_values(X_test)[:len(latents)]

        t0 = time.perf_counter()
        noiseless_raw, _ = _aer_expectations(
            builder, latents, weights, noiseless_simulator, shots, seed, mitigator
        )
        noisy_raw, mitigated_raw = _aer_expectations(
            builder, latents, weights, noisy_simulator, shots, seed + 1000, mitigator
        )
        aer_time = time.perf_counter() - t0

        row: Dict[str, Any] = {
            "Seed": seed, "Samples": int(len(latents)), "Shots": int(shots),
            "P_Single": p_single, "P_CNOT": p_cnot, "P_Readout": p_readout,
            "Aer_Runtime_Sec": float(aer_time),
        }
        for name, expectations in (
            ("Noiseless", noiseless_raw), ("Noisy", noisy_raw), ("Mitigated", mitigated_raw)
        ):
            probabilities = full_model.classify_from_expectations(expectations)
            metrics = compute_metrics(y_subset, np.argmax(probabilities, axis=1), probabilities)
            row.update({
                f"{name}_Accuracy": metrics["accuracy"],
                f"{name}_F1": metrics["f1_macro"],
                f"{name}_FPR": metrics["fpr"],
                f"{name}_ROC_AUC": metrics["roc_auc"],
                f"{name}_TN": metrics["tn"], f"{name}_FP": metrics["fp"],
                f"{name}_FN": metrics["fn"], f"{name}_TP": metrics["tp"],
            })

        # Degradation diagnostics against the ideal analytic expectations
        row["Noiseless_vs_Analytic_MAE"] = float(np.abs(noiseless_raw - analytic).mean())
        row["Noisy_vs_Noiseless_MAE"] = float(np.abs(noisy_raw - noiseless_raw).mean())
        row["Mitigated_vs_Noiseless_MAE"] = float(np.abs(mitigated_raw - noiseless_raw).mean())
        row["Noisy_MeanAbsExpval"] = float(np.abs(noisy_raw).mean())
        row["Mitigated_MeanAbsExpval"] = float(np.abs(mitigated_raw).mean())
        noise_records.append(row)

        print(f"  [Aer] {len(latents)} flows x 3 conditions @ {shots} shots in {aer_time:.2f}s | "
              f"Acc nois {row['Noiseless_Accuracy']:.4f} -> noisy {row['Noisy_Accuracy']:.4f} "
              f"-> M3 {row['Mitigated_Accuracy']:.4f} | "
              f"|d<Z>| noisy {row['Noisy_vs_Noiseless_MAE']:.4f} -> M3 {row['Mitigated_vs_Noiseless_MAE']:.4f}")

    ablation_df = pd.DataFrame(ablation_records)
    noise_df = pd.DataFrame(noise_records)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ablation_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    noise_path = os.path.join(RESULTS_DIR, "noise_results.csv")
    ablation_df.to_csv(ablation_path, index=False)
    noise_df.to_csv(noise_path, index=False)

    print(f"\n[Results] Ablation ({len(ablation_df)} rows) -> {ablation_path}")
    print(f"[Results] NISQ noise ({len(noise_df)} rows) -> {noise_path}")
    return ablation_df, noise_df


def main() -> None:
    print("=" * 70)
    print("Project HQ-IDS-029: Step 3 - Ablation & True Aer NISQ Simulation")
    print("=" * 70)

    config = load_config()
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    log_event("ACTION",
              f"Started Step 3 ablation (3 variants) and 1024-shot Qiskit Aer NISQ "
              f"simulation over seeds {seeds}.")
    log_decision(
        "Ran the NISQ replay on the complete 1000-flow held-out test partition for every "
        "seed rather than a subsample: Aer executes 1000 four-qubit circuits at 1024 shots in "
        "about 1.3 s, so subsampling would have discarded statistical power for no benefit.")

    ablation_df, noise_df = run_ablation_and_noise_experiments(config)

    assert len(ablation_df) == len(VARIANTS) * len(seeds), "Ablation table must hold 3 x 5 rows."
    assert len(noise_df) == len(seeds), "Noise table must hold one row per seed."
    log_event("VERIFICATION",
              f"Step 3 complete: {len(ablation_df)} ablation rows (3 variants x {len(seeds)} seeds) and "
              f"{len(noise_df)} shot-based Aer NISQ rows written to results/ablation_results.csv and "
              f"results/noise_results.csv.")

    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY (mean +/- sd across seeds)")
    print("=" * 70)
    summary = ablation_df.groupby("Model")[["Accuracy", "Macro_F1", "ROC_AUC", "FPR"]].agg(["mean", "std"])
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("  NISQ AER SUMMARY (mean +/- sd across seeds)")
    print("=" * 70)
    columns = [
        "Noiseless_Accuracy", "Noisy_Accuracy", "Mitigated_Accuracy",
        "Noiseless_F1", "Noisy_F1", "Mitigated_F1",
        "Noisy_vs_Noiseless_MAE", "Mitigated_vs_Noiseless_MAE",
    ]
    noise_summary = pd.DataFrame({
        "mean": noise_df[columns].mean(),
        "std": noise_df[columns].std(ddof=1),
    })
    print(noise_summary.to_string())


if __name__ == "__main__":
    main()
