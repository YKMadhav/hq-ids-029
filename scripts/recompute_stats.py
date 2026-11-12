#!/usr/bin/env python3
"""
scripts/recompute_stats.py

Re-derives all hypothesis tests, effect sizes (Cohen's d), TOST equivalence tests,
and family-wise error rate corrections (Holm-Bonferroni) directly from current
results CSV artifacts across all project phases.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Computes paired Cohen's d: mean(diff) / std(diff, ddof=1)."""
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def tost_paired(x: np.ndarray, y: np.ndarray, bound: float = 0.01):
    """
    Two One-Sided Tests (TOST) for paired samples.
    H0: |mu_diff| >= bound vs H1: -bound < mu_diff < bound
    Returns (mean_diff, se, p_tost).
    """
    diff = x - y
    n = len(diff)
    mean_diff = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    if se < 1e-12:
        return mean_diff, se, 0.0
    t1 = (mean_diff + bound) / se
    p1 = 1.0 - stats.t.cdf(t1, df=n - 1)
    t2 = (mean_diff - bound) / se
    p2 = stats.t.cdf(t2, df=n - 1)
    return mean_diff, se, float(max(p1, p2))


def holm_bonferroni(p_values):
    """Holm-Bonferroni family-wise error rate step-down correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (m - rank))
        adjusted[orig_idx] = adj_p
    for i in range(1, m):
        prev_idx = indexed[i - 1][0]
        curr_idx = indexed[i][0]
        if adjusted[curr_idx] < adjusted[prev_idx]:
            adjusted[curr_idx] = adjusted[prev_idx]
    return adjusted


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def recompute_ablation_5seeds():
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    print_section("1. 5-SEED ABLATION HYPOTHESIS TESTS (N=5)")
    print("Models: Full_HybridQNN, Control_A_Quantum_Bypassed, Control_B_Quantum_Frozen")
    
    metrics = ["Accuracy", "Macro_F1", "ROC_AUC", "FPR"]
    rows = []
    p_vals_all = []
    
    for metric in metrics:
        full = df[df["Model"] == "Full_HybridQNN"].sort_values("Seed")[metric].values
        byp = df[df["Model"] == "Control_A_Quantum_Bypassed"].sort_values("Seed")[metric].values
        froz = df[df["Model"] == "Control_B_Quantum_Frozen"].sort_values("Seed")[metric].values
        
        t_a, p_a = stats.ttest_rel(full, byp)
        d_a = cohens_d_paired(full, byp)
        _, _, p_tost_a = tost_paired(full, byp, bound=0.01)
        
        t_b, p_b = stats.ttest_rel(full, froz)
        d_b = cohens_d_paired(full, froz)
        _, _, p_tost_b = tost_paired(full, froz, bound=0.01)
        
        rows.append({
            "Metric": metric,
            "Full_mean_sd": f"{np.mean(full):.4f} +/- {np.std(full, ddof=1):.4f}",
            "Byp_mean_sd": f"{np.mean(byp):.4f} +/- {np.std(byp, ddof=1):.4f}",
            "Froz_mean_sd": f"{np.mean(froz):.4f} +/- {np.std(froz, ddof=1):.4f}",
            "t_Byp": t_a, "p_Byp": p_a, "d_Byp": d_a, "p_tost_Byp": p_tost_a,
            "t_Froz": t_b, "p_Froz": p_b, "d_Froz": d_b, "p_tost_Froz": p_tost_b,
        })
        p_vals_all.extend([p_a, p_b])

    adj_p = holm_bonferroni(p_vals_all)
    idx = 0
    for r in rows:
        r["p_Byp_Holm"] = adj_p[idx]
        r["p_Froz_Holm"] = adj_p[idx + 1]
        idx += 2

    res_df = pd.DataFrame(rows)
    for _, r in res_df.iterrows():
        print(f"\nMetric: {r['Metric']}")
        print(f"  Means: Full = {r['Full_mean_sd']} | Bypassed = {r['Byp_mean_sd']} | Frozen = {r['Froz_mean_sd']}")
        print(f"  Full vs Bypassed: t={r['t_Byp']:+.4f}, p_raw={r['p_Byp']:.4f}, p_Holm={r['p_Byp_Holm']:.4f}, Cohen's d={r['d_Byp']:+.4f}, TOST p(±0.01)={r['p_tost_Byp']:.4f}")
        print(f"  Full vs Frozen:   t={r['t_Froz']:+.4f}, p_raw={r['p_Froz']:.4f}, p_Holm={r['p_Froz_Holm']:.4f}, Cohen's d={r['d_Froz']:+.4f}, TOST p(±0.01)={r['p_tost_Froz']:.4f}")


def recompute_stress_30seeds():
    csv_path = os.path.join(RESULTS_DIR, "stress_30seeds.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    print_section("2. EXPANDED 30-SEED ABLATION HYPOTHESIS TESTS (N=30)")
    
    for metric in ["Accuracy", "Macro_F1"]:
        full = df[df["Model"] == "Full_HybridQNN"].sort_values("Seed")[metric].values
        byp = df[df["Model"] == "Control_A_Quantum_Bypassed"].sort_values("Seed")[metric].values
        froz = df[df["Model"] == "Control_B_Quantum_Frozen"].sort_values("Seed")[metric].values
        
        t_a, p_a = stats.ttest_rel(full, byp)
        d_a = cohens_d_paired(full, byp)
        _, _, p_tost_a = tost_paired(full, byp, bound=0.01)
        
        t_b, p_b = stats.ttest_rel(full, froz)
        d_b = cohens_d_paired(full, froz)
        _, _, p_tost_b = tost_paired(full, froz, bound=0.01)
        
        print(f"\nMetric: {metric} (N=30)")
        print(f"  Full:     {np.mean(full):.4f} +/- {np.std(full, ddof=1):.4f}")
        print(f"  Bypassed: {np.mean(byp):.4f} +/- {np.std(byp, ddof=1):.4f}")
        print(f"  Frozen:   {np.mean(froz):.4f} +/- {np.std(froz, ddof=1):.4f}")
        print(f"  Full vs Bypassed: t={t_a:+.4f}, p={p_a:.4f}, Cohen's d={d_a:+.4f}, TOST p(±0.01)={p_tost_a:.4e}")
        print(f"  Full vs Frozen:   t={t_b:+.4f}, p={p_b:.4f}, Cohen's d={d_b:+.4f}, TOST p(±0.01)={p_tost_b:.4e}")


def recompute_rff_surrogate():
    csv_path = os.path.join(RESULTS_DIR, "rff_surrogate_results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    print_section("3. CLASSICAL FOURIER SURROGATES VS HYBRID VQC (N=5)")
    
    vqc_in = df[df["model"] == "Hybrid VQC"].sort_values("seed")["in_f1"].values
    vqc_cross = df[df["model"] == "Hybrid VQC"].sort_values("seed")["cross_f1"].values
    
    models = ["Classical MLP", "Classical RFF", "Trainable Fourier"]
    p_in_list = []
    p_cross_list = []
    
    print("\n--- In-Domain Macro F1 (UNSW-NB15) ---")
    for m in models:
        m_in = df[df["model"] == m].sort_values("seed")["in_f1"].values
        t, p = stats.ttest_rel(vqc_in, m_in)
        d = cohens_d_paired(vqc_in, m_in)
        p_in_list.append(p)
        print(f"  VQC ({np.mean(vqc_in):.4f}) vs {m} ({np.mean(m_in):.4f}): t={t:+.4f}, p_raw={p:.4f}, Cohen's d={d:+.4f}")
        
    print("\n--- Cross-Domain Macro F1 (CIC-IDS2017) ---")
    for m in models:
        m_cross = df[df["model"] == m].sort_values("seed")["cross_f1"].values
        t, p = stats.ttest_rel(vqc_cross, m_cross)
        d = cohens_d_paired(vqc_cross, m_cross)
        p_cross_list.append(p)
        print(f"  VQC ({np.mean(vqc_cross):.4f}) vs {m} ({np.mean(m_cross):.4f}): t={t:+.4f}, p_raw={p:.4f}, Cohen's d={d:+.4f}")
        
    print("\nHolm-Bonferroni Corrected P-values:")
    adj_in = holm_bonferroni(p_in_list)
    adj_cross = holm_bonferroni(p_cross_list)
    for m, p_r, p_a in zip(models, p_in_list, adj_in):
        print(f"  In-Domain VQC vs {m}: p_raw={p_r:.4f} -> p_adj={p_a:.4f}")
    for m, p_r, p_a in zip(models, p_cross_list, adj_cross):
        print(f"  Cross-Domain VQC vs {m}: p_raw={p_r:.4f} -> p_adj={p_a:.4f}")


def recompute_zeroday():
    csv_path = os.path.join(RESULTS_DIR, "zeroday_results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    print_section("4. ZERO-DAY UNSUPERVISED ANOMALY DETECTION (UN-LEAKED, N=5)")
    
    cae_roc = df[df["model"] == "ClassicalAutoencoder"].sort_values("seed")["roc_auc"].values
    qae_roc = df[df["model"] == "QuantumAutoencoder"].sort_values("seed")["roc_auc"].values
    hyb_roc = df[df["model"] == "HybridEnsemble"].sort_values("seed")["roc_auc"].values
    
    cae_pr = df[df["model"] == "ClassicalAutoencoder"].sort_values("seed")["pr_auc"].values
    hyb_pr = df[df["model"] == "HybridEnsemble"].sort_values("seed")["pr_auc"].values
    
    t_roc_cae, p_roc_cae = stats.ttest_rel(hyb_roc, cae_roc)
    d_roc_cae = cohens_d_paired(hyb_roc, cae_roc)
    
    t_roc_qae, p_roc_qae = stats.ttest_rel(hyb_roc, qae_roc)
    d_roc_qae = cohens_d_paired(hyb_roc, qae_roc)
    
    t_pr, p_pr = stats.ttest_rel(hyb_pr, cae_pr)
    d_pr = cohens_d_paired(hyb_pr, cae_pr)
    
    print(f"CAE ROC-AUC:    {np.mean(cae_roc):.4f} +/- {np.std(cae_roc, ddof=1):.4f}")
    print(f"QAE ROC-AUC:    {np.mean(qae_roc):.4f} +/- {np.std(qae_roc, ddof=1):.4f}")
    print(f"Hybrid ROC-AUC: {np.mean(hyb_roc):.4f} +/- {np.std(hyb_roc, ddof=1):.4f}")
    print(f"  Hybrid vs CAE ROC-AUC: t={t_roc_cae:+.4f}, p={p_roc_cae:.4f}, Cohen's d={d_roc_cae:+.4f} (Statistically Significant: {p_roc_cae < 0.05})")
    print(f"  Hybrid vs QAE ROC-AUC: t={t_roc_qae:+.4f}, p={p_roc_qae:.4f}, Cohen's d={d_roc_qae:+.4f} (Statistically Significant: {p_roc_qae < 0.05})")
    print(f"\nCAE PR-AUC:     {np.mean(cae_pr):.4f} +/- {np.std(cae_pr, ddof=1):.4f}")
    print(f"Hybrid PR-AUC:  {np.mean(hyb_pr):.4f} +/- {np.std(hyb_pr, ddof=1):.4f}")
    print(f"  Hybrid vs CAE PR-AUC:  t={t_pr:+.4f}, p={p_pr:.4f}, Cohen's d={d_pr:+.4f} (Statistically Significant: {p_pr < 0.05})")


def recompute_m3_noise():
    csv_path = os.path.join(RESULTS_DIR, "noise_results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    print_section("5. M3 READOUT MITIGATION AND EXPECTATION DRIFT (N=5)")
    
    noisy_an = df["Noisy_vs_Analytic_MAE"].values
    mit_an = df["Mitigated_vs_Analytic_MAE"].values
    rec_an = (noisy_an - mit_an) / noisy_an * 100.0
    
    noisy_nl = df["Noisy_vs_Noiseless_MAE"].values
    mit_nl = df["Mitigated_vs_Noiseless_MAE"].values
    rec_nl = (noisy_nl - mit_nl) / noisy_nl * 100.0
    
    print("Per-Seed Breakdown:")
    for _, r in df.iterrows():
        s = int(r["Seed"])
        n_a = r["Noisy_vs_Analytic_MAE"]
        m_a = r["Mitigated_vs_Analytic_MAE"]
        rec = (n_a - m_a) / n_a * 100.0
        print(f"  Seed {s}: Noisy_MAE={n_a:.4f}, Mitigated_MAE={m_a:.4f}, Drift Recovery={rec:.1f}% | Noisy_Acc={r['Noisy_Accuracy']:.4f}, Mit_Acc={r['Mitigated_Accuracy']:.4f}")
        
    print(f"\nSummary (Analytic Baseline):")
    print(f"  Noisy vs Analytic MAE:     {np.mean(noisy_an):.4f} +/- {np.std(noisy_an, ddof=1):.4f} [range: {np.min(noisy_an):.4f} - {np.max(noisy_an):.4f}]")
    print(f"  Mitigated vs Analytic MAE: {np.mean(mit_an):.4f} +/- {np.std(mit_an, ddof=1):.4f} [range: {np.min(mit_an):.4f} - {np.max(mit_an):.4f}]")
    print(f"  Analytic Recovery %:       {np.mean(rec_an):.2f}% +/- {np.std(rec_an, ddof=1):.2f}%")
    print(f"\nSummary (Finite-Shot Noiseless Baseline):")
    print(f"  Noisy vs Noiseless MAE:     {np.mean(noisy_nl):.4f} +/- {np.std(noisy_nl, ddof=1):.4f}")
    print(f"  Mitigated vs Noiseless MAE: {np.mean(mit_nl):.4f} +/- {np.std(mit_nl, ddof=1):.4f}")
    print(f"  Finite-Shot Recovery %:     {np.mean(rec_nl):.2f}% +/- {np.std(rec_nl, ddof=1):.2f}%")


def main():
    print("=" * 80)
    print("  PROJECT HQ-IDS-029: RECOMPUTED STATISTICAL AUDIT REPORT")
    print("=" * 80)
    recompute_ablation_5seeds()
    recompute_stress_30seeds()
    recompute_rff_surrogate()
    recompute_zeroday()
    recompute_m3_noise()
    print("\n" + "=" * 80)
    print("  END OF STATISTICAL AUDIT REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
