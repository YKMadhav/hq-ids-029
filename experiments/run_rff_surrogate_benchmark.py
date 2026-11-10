"""Classical Random Fourier Feature (RFF) Surrogate Control Benchmark.

Tests the hypothesis:
'Does a 4-qubit parameterized variational quantum circuit provide any discriminative
or domain-generalization capability beyond a classical Fourier feature surrogate
with matched input dimensions and frequency spectrum?'

Evaluates across 5 seeds:
1. Classical MLP (Direct Linear Bottleneck)
2. Classical RFF (Random Fourier Features Surrogate, frozen frequencies)
3. Trainable Fourier Network (Trainable Fourier series surrogate)
4. 4-Qubit Hybrid VQC (Parameterized Quantum Circuit)
5. Quantum Bypassed Control (Linear Head on latent features)

Measures In-Domain (UNSW-NB15) and Cross-Domain (CIC-IDS2017) transfer stability.
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
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]
FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "smean", "dmean", "sinpkt", "dinpkt", "swin", "dwin"
]
INPUT_DIM = len(FEATURES)  # 12
LATENT_DIM = 4             # 4 latent variables / 4 qubits
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
# Quantum Circuit Architecture
# -----------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=LATENT_DIM)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(LATENT_DIM), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(LATENT_DIM))
    return [qml.expval(qml.PauliZ(i)) for i in range(LATENT_DIM)]

# -----------------------------------------------------------------------------
# Neural Architectures & Classical Surrogates
# -----------------------------------------------------------------------------
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

class RandomFourierFeatureLayer(nn.Module):
    """Classical RFF Surrogate (Rahimi & Recht 2007, Gil-Fuster et al. 2024).

    Maps z in [-pi, pi]^d to d Fourier features cos(W z + b), bounded in [-1, 1],
    matching the exact output range of Pauli-Z expectation values.
    """
    def __init__(self, in_features=LATENT_DIM, out_features=LATENT_DIM, sigma=1.0):
        super().__init__()
        # Random projection matrix sampled from Gaussian
        self.register_buffer("W", torch.randn(out_features, in_features) * sigma)
        self.register_buffer("b", torch.rand(out_features) * 2 * np.pi)

    def forward(self, x):
        proj = torch.matmul(x, self.W.t()) + self.b
        return torch.cos(proj)

class TrainableFourierLayer(nn.Module):
    """Trainable Fourier Series Surrogate.

    Frequencies W and phase offsets b are updated via gradient descent,
    analogous to variational rotation parameters in a PQC.
    """
    def __init__(self, in_features=LATENT_DIM, out_features=LATENT_DIM):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.rand(out_features) * 2 * np.pi)

    def forward(self, x):
        proj = torch.matmul(x, self.W.t()) + self.b
        return torch.cos(proj)

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
        return self.head(q_out)

class RFFSurrogateModel(nn.Module):
    def __init__(self, in_features=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = ClassicalBottleneckEncoder(in_features, latent_dim)
        self.rff = RandomFourierFeatureLayer(latent_dim, latent_dim)
        self.head = nn.Linear(latent_dim, NUM_CLASSES)

    def forward(self, x):
        z = self.encoder(x)
        f_out = self.rff(z)
        return self.head(f_out)

class TrainableFourierModel(nn.Module):
    def __init__(self, in_features=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = ClassicalBottleneckEncoder(in_features, latent_dim)
        self.tfl = TrainableFourierLayer(latent_dim, latent_dim)
        self.head = nn.Linear(latent_dim, NUM_CLASSES)

    def forward(self, x):
        z = self.encoder(x)
        f_out = self.tfl(z)
        return self.head(f_out)

class ClassicalMLPModel(nn.Module):
    def __init__(self, in_features=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = ClassicalBottleneckEncoder(in_features, latent_dim)
        self.head = nn.Linear(latent_dim, NUM_CLASSES)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)

# -----------------------------------------------------------------------------
# Evaluation Helpers
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

def measure_latency_ms(model, x_sample, n_runs=100):
    model.eval()
    t = torch.tensor(x_sample[:1], dtype=torch.float32)
    # 20 warm-up passes
    with torch.no_grad():
        for _ in range(20):
            _ = model(t)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(t)
    return float((time.perf_counter() - t0) / n_runs * 1000.0)

# -----------------------------------------------------------------------------
# Main Benchmark Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("STARTING CLASSICAL RFF SURROGATE BENCHMARK (OPTION B)")
    print("Comparing 4-Qubit Hybrid VQC against Classical Fourier Surrogates")
    print(f"Features: {INPUT_DIM} | Latent Dim: {LATENT_DIM} | Seeds: {SEEDS}")
    print("=" * 75)

    unsw_df = pd.read_csv(UNSW_PATH)
    cic_df = pd.read_csv(CIC_PATH)
    y_cic = cic_df["label"].values.astype(int)

    all_records = []

    model_factories = {
        "Classical MLP": lambda: ClassicalMLPModel(INPUT_DIM, LATENT_DIM),
        "Classical RFF": lambda: RFFSurrogateModel(INPUT_DIM, LATENT_DIM),
        "Trainable Fourier": lambda: TrainableFourierModel(INPUT_DIM, LATENT_DIM),
        "Hybrid VQC": lambda: HybridVQCModel(INPUT_DIM, LATENT_DIM, num_layers=2)
    }

    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        n_samples = len(unsw_df)
        indices = np.random.permutation(n_samples)
        n_train = int(0.70 * n_samples)
        n_val = int(0.10 * n_samples)

        train_df = unsw_df.iloc[indices[:n_train]]
        val_df = unsw_df.iloc[indices[n_train:n_train + n_val]]
        test_df = unsw_df.iloc[indices[n_train + n_val:]]

        prep = CrossPreprocessor()
        X_train = prep.fit_transform(train_df)
        y_train = train_df["label"].values.astype(int)

        X_val = prep.transform(val_df)
        y_val = val_df["label"].values.astype(int)

        X_test_unsw = prep.transform(test_df)
        y_test_unsw = test_df["label"].values.astype(int)

        X_cic = prep.transform(cic_df)

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        for name, factory in model_factories.items():
            print(f"  Training and evaluating: {name}...")
            model = factory()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            # Train
            model.train()
            for epoch in range(EPOCHS):
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

            # Latency
            lat_ms = measure_latency_ms(model, X_test_unsw, n_runs=100 if "VQC" in name else 500)

            # In-Domain (UNSW)
            model.eval()
            with torch.no_grad():
                out_in = model(torch.tensor(X_test_unsw, dtype=torch.float32))
                probs_in = torch.softmax(out_in, dim=1)[:, 1].numpy()
                preds_in = torch.argmax(out_in, dim=1).numpy()
                m_in = compute_metrics(y_test_unsw, preds_in, probs_in)

                # Cross-Domain (CIC)
                out_cross = model(torch.tensor(X_cic, dtype=torch.float32))
                probs_cross = torch.softmax(out_cross, dim=1)[:, 1].numpy()
                preds_cross = torch.argmax(out_cross, dim=1).numpy()
                m_cross = compute_metrics(y_cic, preds_cross, probs_cross)

            retention = (m_cross["f1_macro"] / m_in["f1_macro"]) * 100.0 if m_in["f1_macro"] > 0 else 0.0

            all_records.append({
                "seed": seed,
                "model": name,
                "latency_ms": lat_ms,
                "in_accuracy": m_in["accuracy"],
                "in_f1": m_in["f1_macro"],
                "in_fpr": m_in["fpr"],
                "in_auc": m_in["roc_auc"],
                "cross_accuracy": m_cross["accuracy"],
                "cross_f1": m_cross["f1_macro"],
                "cross_fpr": m_cross["fpr"],
                "cross_auc": m_cross["roc_auc"],
                "f1_retention_pct": retention
            })

    # -------------------------------------------------------------------------
    # Aggregation & Statistical Tests
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(all_records)
    csv_out = RESULTS_DIR / "rff_surrogate_results.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\nSaved raw trial metrics to: {csv_out}")

    models = list(model_factories.keys())
    summary = {}
    for m in models:
        sub = df_results[df_results["model"] == m]
        summary[m] = {
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
            "f1_retention_std": float(sub["f1_retention_pct"].std()),
            "latency_ms_mean": float(sub["latency_ms"].mean()),
            "latency_ms_std": float(sub["latency_ms"].std())
        }

    # Hypothesis Testing: Hybrid VQC vs Classical Surrogates
    vqc_in_f1 = df_results[df_results["model"] == "Hybrid VQC"]["in_f1"].values
    vqc_cross_f1 = df_results[df_results["model"] == "Hybrid VQC"]["cross_f1"].values

    stats_report = {}
    for baseline in ["Classical MLP", "Classical RFF", "Trainable Fourier"]:
        b_in = df_results[df_results["model"] == baseline]["in_f1"].values
        b_cross = df_results[df_results["model"] == baseline]["cross_f1"].values

        t_in, p_in = stats.ttest_rel(vqc_in_f1, b_in)
        t_cross, p_cross = stats.ttest_rel(vqc_cross_f1, b_cross)

        stats_report[f"VQC_vs_{baseline}"] = {
            "in_domain": {"t_statistic": float(t_in), "p_value": float(p_in), "significant_p05": bool(p_in < 0.05)},
            "cross_domain": {"t_statistic": float(t_cross), "p_value": float(p_cross), "significant_p05": bool(p_cross < 0.05)}
        }

    summary["hypothesis_tests"] = stats_report

    json_out = RESULTS_DIR / "rff_surrogate_summary.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregated summary & statistical tests to: {json_out}")

    # -------------------------------------------------------------------------
    # Publication Visualization
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(models))
    width = 0.35

    in_f1s = [summary[m]["in_f1_mean"] * 100 for m in models]
    in_errs = [summary[m]["in_f1_std"] * 100 for m in models]
    cross_f1s = [summary[m]["cross_f1_mean"] * 100 for m in models]
    cross_errs = [summary[m]["cross_f1_std"] * 100 for m in models]

    ax1.bar(x - width/2, in_f1s, width, yerr=in_errs, label="In-Domain (UNSW-NB15)", color="#2b5c8f", capsize=4)
    ax1.bar(x + width/2, cross_f1s, width, yerr=cross_errs, label="Cross-Domain (CIC-IDS2017)", color="#d73027", capsize=4)
    ax1.set_ylabel("Macro F1-Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Classical Fourier Surrogates vs. Quantum VQC", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.5, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    retentions = [summary[m]["f1_retention_mean"] for m in models]
    ret_errs = [summary[m]["f1_retention_std"] for m in models]
    bars = ax2.bar(models, retentions, yerr=ret_errs, color=["#4575b4", "#74add1", "#abd9e9", "#313695"], capsize=4)
    ax2.set_ylabel("F1 Retention Ratio (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Domain Generalization Retention Ratio", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, retentions):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = FIGURES_DIR / "rff_quantum_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated publication figure: {plot_path}")

    # Summary Table
    print("\n" + "=" * 90)
    print("RFF SURROGATE VS QUANTUM VQC BENCHMARK SUMMARY (5 SEEDS)")
    print("=" * 90)
    print(f"{'Model':<20} | {'In-Domain F1':<18} | {'Cross-Domain F1':<18} | {'Retention':<12} | {'Latency':<10}")
    print("-" * 90)
    for m in models:
        s = summary[m]
        print(f"{m:<20} | {s['in_f1_mean']*100:.2f}% ± {s['in_f1_std']*100:.2f}% | {s['cross_f1_mean']*100:.2f}% ± {s['cross_f1_std']*100:.2f}% | {s['f1_retention_mean']:.2f}% | {s['latency_ms_mean']:.3f} ms")
    print("=" * 90)

    print("\n--- Paired Student's t-test Results (VQC vs Baselines) ---")
    for k, v in stats_report.items():
        print(f"  {k}:")
        print(f"    In-Domain:    t = {v['in_domain']['t_statistic']:.4f}, p = {v['in_domain']['p_value']:.4f} (Significant: {v['in_domain']['significant_p05']})")
        print(f"    Cross-Domain: t = {v['cross_domain']['t_statistic']:.4f}, p = {v['cross_domain']['p_value']:.4f} (Significant: {v['cross_domain']['significant_p05']})")
    print("=" * 90)

if __name__ == "__main__":
    main()
