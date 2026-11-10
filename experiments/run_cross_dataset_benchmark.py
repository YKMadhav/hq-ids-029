"""Cross-Dataset Domain Generalization Benchmark: UNSW-NB15 -> CIC-IDS2017.

Evaluates whether 4-qubit quantum Hilbert space representations provide superior
generalization stability across network domain shifts compared to depth-matched
classical neural networks and tree ensembles.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]
FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "smean", "dmean", "sinpkt", "dinpkt", "swin", "dwin"
]
INPUT_DIM = len(FEATURES)  # 12
LATENT_DIM = 4             # 4 qubits
NUM_CLASSES = 2
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.01

REPO_ROOT = Path(__file__).resolve().parents[1]
UNSW_PATH = REPO_ROOT / "data" / "unsw_aligned_5k.csv"
CIC_PATH = REPO_ROOT / "data" / "cic_aligned_5k.csv"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Quantum Circuit & Hybrid Architectures
# -----------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=LATENT_DIM)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(LATENT_DIM), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(LATENT_DIM))
    return [qml.expval(qml.PauliZ(i)) for i in range(LATENT_DIM)]

class ClassicalBottleneckEncoder(nn.Module):
    def __init__(self, in_features=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * np.pi  # scaled to [-pi, pi]

class HybridVQCModel(nn.Module):
    def __init__(self, in_features=INPUT_DIM, latent_dim=LATENT_DIM, num_layers=2):
        super().__init__()
        self.encoder = ClassicalBottleneckEncoder(in_features, latent_dim)
        weight_shapes = {"weights": (num_layers, latent_dim, 3)}
        self.qnode = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)
        self.head = nn.Linear(latent_dim, NUM_CLASSES)

    def forward(self, x):
        z = self.encoder(x)
        q_out = self.qnode(z)
        logits = self.head(q_out)
        return logits

class ClassicalMLPModel(nn.Module):
    def __init__(self, in_features=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = ClassicalBottleneckEncoder(in_features, latent_dim)
        self.head = nn.Linear(latent_dim, NUM_CLASSES)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)

# -----------------------------------------------------------------------------
# Preprocessing Helper (Zero Leakage)
# -----------------------------------------------------------------------------
class CrossPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

    def fit_transform(self, df):
        vals = np.log1p(np.maximum(df[FEATURES].values.astype(float), 0.0))
        scaled = self.scaler.fit_transform(vals)
        return np.clip(scaled, -np.pi, np.pi)

    def transform(self, df):
        vals = np.log1p(np.maximum(df[FEATURES].values.astype(float), 0.0))
        scaled = self.scaler.transform(vals)
        return np.clip(scaled, -np.pi, np.pi)

# -----------------------------------------------------------------------------
# Model Evaluation Helper
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    acc = float(np.mean(y_true == y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {"accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec, "roc_auc": auc, "fpr": fpr}

# -----------------------------------------------------------------------------
# Main Benchmark Routine
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STARTING CROSS-DATASET BENCHMARK: UNSW-NB15 -> CIC-IDS2017")
    print(f"Features: {INPUT_DIM} aligned continuous transport metrics")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    unsw_df = pd.read_csv(UNSW_PATH)
    cic_df = pd.read_csv(CIC_PATH)
    y_cic = cic_df["label"].values.astype(int)

    all_records = []

    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 70% Train, 10% Val, 20% Test on UNSW
        n_samples = len(unsw_df)
        indices = np.random.permutation(n_samples)
        n_train = int(0.70 * n_samples)
        n_val = int(0.10 * n_samples)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_df = unsw_df.iloc[train_idx]
        val_df = unsw_df.iloc[val_idx]
        test_df = unsw_df.iloc[test_idx]

        # Fit preprocessor strictly on UNSW Train
        prep = CrossPreprocessor()
        X_train = prep.fit_transform(train_df)
        y_train = train_df["label"].values.astype(int)

        X_val = prep.transform(val_df)
        y_val = val_df["label"].values.astype(int)

        X_test_unsw = prep.transform(test_df)
        y_test_unsw = test_df["label"].values.astype(int)

        # Transform CIC test set with frozen UNSW scaler
        X_cic = prep.transform(cic_df)

        # PyTorch DataLoaders
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        # ---------------------------------------------------------------------
        # 1. XGBoost
        # ---------------------------------------------------------------------
        print("  Evaluating XGBoost...")
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed, eval_metric="logloss")
        xgb_clf.fit(X_train, y_train)

        # In-Domain
        xgb_preds_in = xgb_clf.predict(X_test_unsw)
        xgb_probs_in = xgb_clf.predict_proba(X_test_unsw)[:, 1]
        m_xgb_in = compute_metrics(y_test_unsw, xgb_preds_in, xgb_probs_in)

        # Cross-Domain
        xgb_preds_cross = xgb_clf.predict(X_cic)
        xgb_probs_cross = xgb_clf.predict_proba(X_cic)[:, 1]
        m_xgb_cross = compute_metrics(y_cic, xgb_preds_cross, xgb_probs_cross)

        retention_xgb = (m_xgb_cross["f1_macro"] / m_xgb_in["f1_macro"]) * 100.0 if m_xgb_in["f1_macro"] > 0 else 0.0

        all_records.append({
            "seed": seed, "model": "XGBoost", "paradigm": "Classical Tree Ensemble",
            "in_accuracy": m_xgb_in["accuracy"], "in_f1": m_xgb_in["f1_macro"], "in_fpr": m_xgb_in["fpr"], "in_auc": m_xgb_in["roc_auc"],
            "cross_accuracy": m_xgb_cross["accuracy"], "cross_f1": m_xgb_cross["f1_macro"], "cross_fpr": m_xgb_cross["fpr"], "cross_auc": m_xgb_cross["roc_auc"],
            "f1_retention_pct": retention_xgb
        })

        # ---------------------------------------------------------------------
        # 2. Random Forest
        # ---------------------------------------------------------------------
        print("  Evaluating Random Forest...")
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed)
        rf_clf.fit(X_train, y_train)

        rf_preds_in = rf_clf.predict(X_test_unsw)
        rf_probs_in = rf_clf.predict_proba(X_test_unsw)[:, 1]
        m_rf_in = compute_metrics(y_test_unsw, rf_preds_in, rf_probs_in)

        rf_preds_cross = rf_clf.predict(X_cic)
        rf_probs_cross = rf_clf.predict_proba(X_cic)[:, 1]
        m_rf_cross = compute_metrics(y_cic, rf_preds_cross, rf_probs_cross)

        retention_rf = (m_rf_cross["f1_macro"] / m_rf_in["f1_macro"]) * 100.0 if m_rf_in["f1_macro"] > 0 else 0.0

        all_records.append({
            "seed": seed, "model": "Random Forest", "paradigm": "Classical Tree Ensemble",
            "in_accuracy": m_rf_in["accuracy"], "in_f1": m_rf_in["f1_macro"], "in_fpr": m_rf_in["fpr"], "in_auc": m_rf_in["roc_auc"],
            "cross_accuracy": m_rf_cross["accuracy"], "cross_f1": m_rf_cross["f1_macro"], "cross_fpr": m_rf_cross["fpr"], "cross_auc": m_rf_cross["roc_auc"],
            "f1_retention_pct": retention_rf
        })

        # ---------------------------------------------------------------------
        # 3. Classical MLP Baseline (12 -> 16 -> 4 -> 2)
        # ---------------------------------------------------------------------
        print("  Evaluating Classical MLP...")
        mlp_net = ClassicalMLPModel(INPUT_DIM, LATENT_DIM)
        opt_mlp = torch.optim.Adam(mlp_net.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()

        mlp_net.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                opt_mlp.zero_grad()
                out = mlp_net(bx)
                loss = crit(out, by)
                loss.backward()
                opt_mlp.step()

        mlp_net.eval()
        with torch.no_grad():
            out_in = mlp_net(torch.tensor(X_test_unsw, dtype=torch.float32))
            mlp_probs_in = torch.softmax(out_in, dim=1)[:, 1].numpy()
            mlp_preds_in = torch.argmax(out_in, dim=1).numpy()
            m_mlp_in = compute_metrics(y_test_unsw, mlp_preds_in, mlp_probs_in)

            out_cross = mlp_net(torch.tensor(X_cic, dtype=torch.float32))
            mlp_probs_cross = torch.softmax(out_cross, dim=1)[:, 1].numpy()
            mlp_preds_cross = torch.argmax(out_cross, dim=1).numpy()
            m_mlp_cross = compute_metrics(y_cic, mlp_preds_cross, mlp_probs_cross)

        retention_mlp = (m_mlp_cross["f1_macro"] / m_mlp_in["f1_macro"]) * 100.0 if m_mlp_in["f1_macro"] > 0 else 0.0

        all_records.append({
            "seed": seed, "model": "Classical MLP", "paradigm": "Classical Neural Network",
            "in_accuracy": m_mlp_in["accuracy"], "in_f1": m_mlp_in["f1_macro"], "in_fpr": m_mlp_in["fpr"], "in_auc": m_mlp_in["roc_auc"],
            "cross_accuracy": m_mlp_cross["accuracy"], "cross_f1": m_mlp_cross["f1_macro"], "cross_fpr": m_mlp_cross["fpr"], "cross_auc": m_mlp_cross["roc_auc"],
            "f1_retention_pct": retention_mlp
        })

        # ---------------------------------------------------------------------
        # 4. Hybrid VQC (12 -> 16 -> 4 -> 4-Qubit VQC -> 2)
        # ---------------------------------------------------------------------
        print("  Evaluating 4-Qubit Hybrid VQC...")
        vqc_net = HybridVQCModel(INPUT_DIM, LATENT_DIM, num_layers=2)
        opt_vqc = torch.optim.Adam(vqc_net.parameters(), lr=LR)

        vqc_net.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                opt_vqc.zero_grad()
                out = vqc_net(bx)
                loss = crit(out, by)
                loss.backward()
                opt_vqc.step()

        vqc_net.eval()
        with torch.no_grad():
            out_in = vqc_net(torch.tensor(X_test_unsw, dtype=torch.float32))
            vqc_probs_in = torch.softmax(out_in, dim=1)[:, 1].numpy()
            vqc_preds_in = torch.argmax(out_in, dim=1).numpy()
            m_vqc_in = compute_metrics(y_test_unsw, vqc_preds_in, vqc_probs_in)

            out_cross = vqc_net(torch.tensor(X_cic, dtype=torch.float32))
            vqc_probs_cross = torch.softmax(out_cross, dim=1)[:, 1].numpy()
            vqc_preds_cross = torch.argmax(out_cross, dim=1).numpy()
            m_vqc_cross = compute_metrics(y_cic, vqc_preds_cross, vqc_probs_cross)

        retention_vqc = (m_vqc_cross["f1_macro"] / m_vqc_in["f1_macro"]) * 100.0 if m_vqc_in["f1_macro"] > 0 else 0.0

        all_records.append({
            "seed": seed, "model": "Hybrid VQC", "paradigm": "Variational QML",
            "in_accuracy": m_vqc_in["accuracy"], "in_f1": m_vqc_in["f1_macro"], "in_fpr": m_vqc_in["fpr"], "in_auc": m_vqc_in["roc_auc"],
            "cross_accuracy": m_vqc_cross["accuracy"], "cross_f1": m_vqc_cross["f1_macro"], "cross_fpr": m_vqc_cross["fpr"], "cross_auc": m_vqc_cross["roc_auc"],
            "f1_retention_pct": retention_vqc
        })

    # -------------------------------------------------------------------------
    # Aggregation and Reporting
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(all_records)
    csv_out = RESULTS_DIR / "cross_dataset_results.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\nSaved raw trials to {csv_out}")

    summary = {}
    models = df_results["model"].unique()
    for m in models:
        sub = df_results[df_results["model"] == m]
        summary[m] = {
            "paradigm": str(sub["paradigm"].iloc[0]),
            "in_accuracy_mean": float(sub["in_accuracy"].mean()),
            "in_accuracy_std": float(sub["in_accuracy"].std()),
            "in_f1_mean": float(sub["in_f1"].mean()),
            "in_f1_std": float(sub["in_f1"].std()),
            "in_fpr_mean": float(sub["in_fpr"].mean()),
            "in_fpr_std": float(sub["in_fpr"].std()),
            "cross_accuracy_mean": float(sub["cross_accuracy"].mean()),
            "cross_accuracy_std": float(sub["cross_accuracy"].std()),
            "cross_f1_mean": float(sub["cross_f1"].mean()),
            "cross_f1_std": float(sub["cross_f1"].std()),
            "cross_fpr_mean": float(sub["cross_fpr"].mean()),
            "cross_fpr_std": float(sub["cross_fpr"].std()),
            "f1_retention_mean": float(sub["f1_retention_pct"].mean()),
            "f1_retention_std": float(sub["f1_retention_pct"].std())
        }

    json_out = RESULTS_DIR / "cross_dataset_summary.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_out}")

    # -------------------------------------------------------------------------
    # Visual Artifact Generation
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(models))
    width = 0.35

    in_f1s = [summary[m]["in_f1_mean"] * 100 for m in models]
    in_errs = [summary[m]["in_f1_std"] * 100 for m in models]
    cross_f1s = [summary[m]["cross_f1_mean"] * 100 for m in models]
    cross_errs = [summary[m]["cross_f1_std"] * 100 for m in models]

    ax1.bar(x - width/2, in_f1s, width, yerr=in_errs, label="In-Domain (UNSW-NB15)", color="#2b5c8f", capsize=4)
    ax1.bar(x + width/2, cross_f1s, width, yerr=cross_errs, label="Cross-Domain (CIC-IDS2017)", color="#d73027", capsize=4)
    ax1.set_ylabel("Macro F1-Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("In-Domain vs. Cross-Domain Generalization", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    retentions = [summary[m]["f1_retention_mean"] for m in models]
    ret_errs = [summary[m]["f1_retention_std"] for m in models]
    bars = ax2.bar(models, retentions, yerr=ret_errs, color=["#4575b4", "#74add1", "#fdae61", "#313695"], capsize=4)
    ax2.set_ylabel("F1 Retention Ratio (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Cross-Domain Transfer Stability", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, retentions):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = FIGURES_DIR / "cross_dataset_generalization.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated publication figure: {plot_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("CROSS-DATASET GENERALIZATION BENCHMARK SUMMARY (5 SEEDS)")
    print("=" * 80)
    print(f"{'Model':<15} | {'In-Domain F1':<18} | {'Cross-Domain F1':<18} | {'F1 Retention':<15}")
    print("-" * 80)
    for m in models:
        s = summary[m]
        print(f"{m:<15} | {s['in_f1_mean']*100:.2f}% ± {s['in_f1_std']*100:.2f}% | {s['cross_f1_mean']*100:.2f}% ± {s['cross_f1_std']*100:.2f}% | {s['f1_retention_mean']:.2f}% ± {s['f1_retention_std']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
