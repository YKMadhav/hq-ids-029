"""
Generate every number and table used in the manuscript from results/*.csv|json and data/.

Nothing in paper/paper.tex may be typed by hand: results enter only through
``paper/generated_numbers.tex`` (LaTeX macros) and ``paper/tab_*.tex`` written here, and
``tests/test_paper_numbers.py`` fails if the manuscript contains a literal result number.

Usage: python scripts/generate_paper_numbers.py
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "paper")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

MACROS: dict = {}


def macro(name: str, value, fmt: str = "{:.2f}") -> None:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def pct(x: float) -> float:
    return 100.0 * x


def pm(mean: float, sd: float, digits: int = 2) -> str:
    return f"{pct(mean):.{digits}f}\\,$\\pm$\\,{pct(sd):.{digits}f}"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RES, name))


def write(name: str, text: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as handle:
        handle.write(text)


def booktabs(header: list, rows: list, align: str) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


LABELS = {
    "Majority": "Majority class (train prior)", "Stump": "One-split stump", "RandomForest": "Random forest",
    "XGBoost": "XGBoost", "ClassicalMLP": "Classical MLP", "HybridQNN": "Hybrid QNN (trained VQC)",
    "QNN_Bypassed": "Hybrid, VQC bypassed", "QNN_Frozen": "Hybrid, VQC frozen random",
}


def model_table(frame: pd.DataFrame, order: list) -> str:
    rows = []
    for m in order:
        g = frame[frame.model == m]
        if g.empty:
            continue
        sd = lambda c: g[c].std(ddof=1) if len(g) > 1 else 0.0
        rows.append([LABELS.get(m, m), pm(g.accuracy.mean(), sd("accuracy")), pm(g.balanced_accuracy.mean(), sd("balanced_accuracy")),
                     f"{g.roc_auc.mean():.3f}", f"{g.mcc.mean():.3f}", pm(g.fpr.mean(), sd("fpr"), 1)])
    return booktabs(["Model", "Acc.\\ (\\%)", "Bal.\\ acc.\\ (\\%)", "AUC", "MCC", "FPR (\\%)"], rows, "lrrrrr")


def main() -> None:
    # ---------------------------------------------------------------- data audit
    from common import deduplicate_flows, stratified_split  # noqa: E402

    sample = pd.read_csv(os.path.join(REPO, "data", "real_flows.csv"))
    uniq = deduplicate_flows(sample)
    macro("sampleRows", len(sample), "{:,}".replace(",", "\\,") if False else "{}")
    macro("sampleUnique", len(uniq), "{}")
    macro("sampleDupRows", len(sample) - len(uniq), "{}")
    cols = [c for c in sample.columns if c not in ("id", "label", "attack_cat")]
    tr, _, te = stratified_split(sample, 42, deduplicate=False)
    seen = set(map(tuple, tr[cols].astype(str).values))
    twin = sum(tuple(r) in seen for r in te[cols].astype(str).values)
    macro("sampleTwinTest", twin, "{}")
    macro("sampleTestRows", len(te), "{}")
    macro("sampleTwinPct", pct(twin / len(te)), "{:.1f}")

    heldout = json.load(open(os.path.join(RES, "official_heldout_summary_official_30k.json")))["partitions"]
    for key, name in [("official_train_rows", "offTrainRows"), ("train_pool_rows", "offTrainPool"), ("official_test_rows", "offTestRows"),
                      ("test_rows_final", "offTestFinal")]:
        macro(name, heldout[key], "{}")
    macro("offTrainAttackPct", pct(heldout["official_train_attack_rate"]), "{:.1f}")
    macro("offPoolAttackPct", pct(heldout["train_attack_rate"]), "{:.1f}")
    macro("offTestAttackPct", pct(heldout["test_attack_rate"]), "{:.1f}")
    macro("offTestTwinPct", pct(1 - heldout["test_rows_final"] / heldout["official_test_rows"]), "{:.1f}")
    macro("offTrainDupPct", pct(1 - heldout["train_pool_rows"] / heldout["official_train_rows"]), "{:.1f}")

    # parameter counts (architecture facts, computed rather than typed)
    from src.hybrid_pipeline import _HybridQNNModule

    counts = {q: _HybridQNNModule(input_dim=39, hidden_dim=16, num_qubits=4, num_layers=2, quantum_layer=q) for q in ("vqc", "identity", "frozen")}
    trainable = {q: sum(p.numel() for p in m.parameters() if p.requires_grad) for q, m in counts.items()}
    macro("paramsHqnn", trainable["vqc"], "{}")
    macro("paramsMlp", trainable["identity"], "{}")
    macro("paramsVqc", trainable["vqc"] - trainable["identity"], "{}")

    # ---------------------------------------------------------------- primary protocol
    p30 = read("official_heldout_results_official_30k.csv")
    full = read("official_heldout_results_official_full.csv")
    abl = read("official_heldout_results_official_30k_ablation.csv")
    boot = read("official_heldout_bootstrap_official_30k.csv")

    write("tab_primary.tex", model_table(p30, ["Majority", "Stump", "RandomForest", "XGBoost", "ClassicalMLP", "HybridQNN"]))
    write("tab_primary_fulltrain.tex", model_table(full, ["Majority", "Stump", "RandomForest", "XGBoost", "ClassicalMLP"]))
    for m, tag in [("HybridQNN", "Hqnn"), ("ClassicalMLP", "Mlp"), ("XGBoost", "Xgb"), ("RandomForest", "Rf"), ("Stump", "Stump"), ("Majority", "Maj")]:
        g = p30[p30.model == m]
        macro(f"acc{tag}", pct(g.accuracy.mean()))
        macro(f"accSd{tag}", pct(g.accuracy.std(ddof=1)))
        macro(f"auc{tag}", g.roc_auc.mean(), "{:.3f}")
        macro(f"fpr{tag}", pct(g.fpr.mean()), "{:.1f}")
    gf = full[full.model == "ClassicalMLP"]
    macro("accMlpFull", pct(gf.accuracy.mean()))
    macro("accXgbFull", pct(full[full.model == "XGBoost"].accuracy.mean()))

    b = boot[boot.model == "HybridQNN"]
    macro("bootDiffHqnnMlp", pct(b.acc_diff.mean()))
    macro("bootExcludeZero", int(((b.ci_low > 0) | (b.ci_high < 0)).sum()), "{}")
    macro("bootSeeds", len(b), "{}")

    # ablation on the primary protocol (+ bootstrap vs MLP)
    write("tab_ablation_official.tex", model_table(abl, ["ClassicalMLP", "HybridQNN", "QNN_Bypassed", "QNN_Frozen"]))
    for m, tag in [("QNN_Bypassed", "Byp"), ("QNN_Frozen", "Frz")]:
        macro(f"acc{tag}", pct(abl[abl.model == m].accuracy.mean()))
        macro(f"accSd{tag}", pct(abl[abl.model == m].accuracy.std(ddof=1)))

    # ---------------------------------------------------------------- 5k-sample protocol
    s30 = read("stress_30seeds.csv")
    rows = []
    for m, label in [("Full_HybridQNN", "Hybrid QNN (trained VQC)"), ("Control_A_Quantum_Bypassed", "VQC bypassed"),
                     ("Control_B_Quantum_Frozen", "VQC frozen random")]:
        g = s30[s30.Model == m]
        rows.append([label, pm(g.Accuracy.mean(), g.Accuracy.std(ddof=1)), f"{g.ROC_AUC.mean():.3f}", pm(g.FPR.mean(), g.FPR.std(ddof=1), 1)])
    write("tab_sample30.tex", booktabs(["Variant", "Acc.\\ (\\%)", "AUC", "FPR (\\%)"], rows, "lrrr"))
    rep = json.load(open(os.path.join(RES, "stress_test_report.json")))["test_2_thirty_seed_hypothesis"]
    macro("pFullByp", rep["paired_ttest_full_vs_bypassed"]["p_value"], "{:.3f}")
    macro("pFullFrz", rep["paired_ttest_full_vs_frozen"]["p_value"], "{:.3f}")
    macro("dFullByp", rep["paired_ttest_full_vs_bypassed"]["cohens_d"], "{:.2f}")
    macro("dFullFrz", rep["paired_ttest_full_vs_frozen"]["cohens_d"], "{:.2f}")

    base = read("metrics.csv")
    for key, tag in [("MajorityClass(flag-all)", "SMaj"), ("SingleSplitStump", "SStump"), ("HybridQNN", "SHqnn"), ("ClassicalMLP", "SMlp"),
                     ("XGBoost", "SXgb"), ("RandomForest", "SRf")]:
        r = base[base.model == key]
        if len(r):
            macro(f"acc{tag}", pct(float(r.accuracy_mean.iloc[0])))

    # ---------------------------------------------------------------- leakage screens (5k sample, seed 42)
    stress = json.load(open(os.path.join(RES, "stress_test_report.json")))
    t1 = stress["test_1_label_shuffling"]
    macro("shuffleN", t1["n_shuffles"], "{}")
    macro("shuffleBal", pct(t1["balanced_accuracy_max"]), "{:.1f}")
    macro("shuffleAuc", t1["roc_auc"], "{:.3f}")
    macro("shuffleAucSe", t1["roc_auc_se"], "{:.3f}")
    macro("shuffleAucLo", min(t1["roc_auc_per_shuffle"]), "{:.2f}")
    macro("shuffleAucHi", max(t1["roc_auc_per_shuffle"]), "{:.2f}")
    macro("sentinelAuc", t1["screen_sentinel_auc"], "{:.2f}")
    top = sorted(t1["single_feature_auc"], key=lambda r: -r["single_feature_auc"])[:3]
    macro("topFeatOne", top[0]["feature"].replace("_", "\\_"))
    macro("topFeatOneAuc", top[0]["single_feature_auc"], "{:.3f}")
    macro("topFeatTwo", top[1]["feature"].replace("_", "\\_"))
    macro("topFeatTwoAuc", top[1]["single_feature_auc"], "{:.3f}")
    imp = stress["test_3_feature_importance"]["top_5_critical_features"]
    for tag, r in zip(("One", "Two", "Three"), imp[:3]):
        macro(f"permFeat{tag}", r["feature"].replace("_", "\\_"))
        macro(f"permDrop{tag}", pct(r["importance_delta_acc"]), "{:.1f}")
    th = json.load(open(os.path.join(RES, "thread_sensitivity.json")))
    macro("threadAccOne", pct(th["accuracy_1_thread"]), "{:.2f}")
    macro("threadAccTwo", pct(th["accuracy_2_threads"]), "{:.2f}")
    macro("threadBitIdentical", "yes" if th["bit_identical_at_fixed_threads"] else "no")
    macro("threadSumPDiff", abs(th["runs"]["1"][0]["sum_p"] - th["runs"]["2"][0]["sum_p"]), "{:.4f}")

    # ---------------------------------------------------------------- zero-day
    zd = read("zeroday_results.csv")
    rows = []
    for m, label in [("ClassicalAutoencoder", "Classical autoencoder"), ("QuantumAutoencoder", "Quantum autoencoder"), ("HybridEnsemble", "Hybrid ensemble")]:
        g = zd[zd.model == m]
        rows.append([label, f"{g.roc_auc.mean():.3f}\\,$\\pm$\\,{g.roc_auc.std(ddof=1):.3f}", pm(g.recall.mean(), g.recall.std(ddof=1), 1),
                     pm(g.fpr.mean(), g.fpr.std(ddof=1), 1)])
    write("tab_zeroday.tex", booktabs(["Detector", "AUC", "Recall (\\%)", "FPR (\\%)"], rows, "lrrr"))

    # ---------------------------------------------------------------- noise / mitigation
    nz = read("noise_results.csv")
    macro("noiseAccDelta", pct((nz.Noisy_Accuracy - nz.Noiseless_Accuracy).abs().max()), "{:.2f}")
    macro("maeNoisyLo", nz.Noisy_vs_Analytic_MAE.min(), "{:.3f}")
    macro("maeNoisyHi", nz.Noisy_vs_Analytic_MAE.max(), "{:.3f}")
    macro("maeMitLo", nz.Mitigated_vs_Analytic_MAE.min(), "{:.3f}")
    macro("maeMitHi", nz.Mitigated_vs_Analytic_MAE.max(), "{:.3f}")
    macro("maeShot", nz.Noiseless_vs_Analytic_MAE.mean(), "{:.3f}")
    macro("mitRecoveryPct", pct(1 - (nz.Mitigated_vs_Analytic_MAE.sum() / nz.Noisy_vs_Analytic_MAE.sum())), "{:.0f}")
    macro("noiseSamples", int(nz.Samples.iloc[0]), "{}")
    macro("noiseShots", int(nz.Shots.iloc[0]), "{}")
    macro("noisePReadout", float(nz.P_Readout.iloc[0]), "{:.2f}")

    # ---------------------------------------------------------------- phase 4 (optional)
    p4_path = os.path.join(RES, "phase4_setup_header_results.csv")
    if os.path.exists(p4_path):
        p4 = pd.read_csv(p4_path)
        rows = []
        for m in p4.model.unique():
            g = p4[p4.model == m]
            rows.append([m.replace("_", "\\_"), pm(g.accuracy.mean(), g.accuracy.std(ddof=1)), pm(g.recall.mean(), g.recall.std(ddof=1), 1),
                         pm(g.fpr.mean(), g.fpr.std(ddof=1), 1)])
        write("tab_phase4.tex", booktabs(["Model", "Acc.\\ (\\%)", "Recall (\\%)", "FPR (\\%)"], rows, "lrrr"))
        if "test_vectors_seen_in_train" in p4:
            macro("pFourSeenPct", pct(p4.test_vectors_seen_in_train.dropna().mean()), "{:.1f}")

    # ---------------------------------------------------------------- multiclass (5k sample)
    mc = json.load(open(os.path.join(RES, "multiclass_summary.json")))["models"]
    rows = []
    for m, label in [("XGBoost", "XGBoost"), ("RandomForest", "Random forest"), ("ClassicalMLP", "Classical MLP"),
                     ("HybridVQC_Bitstrings", "Hybrid VQC (bitstrings)"), ("HybridQCNN", "Hybrid QCNN"), ("QSVM", "QSVM")]:
        v = mc[m]
        rows.append([label, pm(v["accuracy"]["mean"], v["accuracy"]["std"]), pm(v["f1_macro"]["mean"], v["f1_macro"]["std"])])
    write("tab_multiclass.tex", booktabs(["Model", "Acc.\\ (\\%)", "Macro-F1 (\\%)"], rows, "lrr"))

    # ------------------------------------------------- quantum-advantage search (qadv/)
    qa = os.path.join(RES, "qadv")
    g1 = pd.read_csv(os.path.join(qa, "item1_geometric.csv")); g12 = pd.read_csv(os.path.join(qa, "item1_scale12.csv"))
    macro("gdLo", g1["g_C_to_Q_min"].min(), "{:.1f}"); macro("gdHi", g1["g_C_to_Q_min"].max(), "{:.1f}")
    macro("gdTwelveLo", g12["g_C_to_Q_min"].min(), "{:.1f}"); macro("gdTwelveHi", g12["g_C_to_Q_min"].max(), "{:.1f}")
    macro("gdRefLo", g1["g_classical_ref"].min(), "{:.1f}"); macro("gdRefHi", g1["g_classical_ref"].max(), "{:.1f}")
    macro("gdSqrtN", g1["sqrtN"].iloc[0], "{:.1f}")
    zu = pd.read_csv(os.path.join(qa, "item4_zeroday.csv")); zc = pd.read_csv(os.path.join(qa, "item4_zeroday_cic.csv"))
    QK, NOENT = "OCSVM-QuantumKernel(8q,bw=1.0)", "OCSVM-ProductStateKernel(8q,no-entanglement,bw=1.0)"
    mu = zu.groupby("detector")[["auc", "fpr_at_95tpr"]].mean(); mc_ = zc.groupby("detector")[["auc", "fpr_at_95tpr"]].mean()
    macro("zdQkAuc", mu.loc[QK, "auc"], "{:.3f}"); macro("zdQkFpr", mu.loc[QK, "fpr_at_95tpr"], "{:.2f}")
    macro("zdKnnAuc", mu.loc["kNN(k=10, 8d PCA)", "auc"], "{:.3f}"); macro("zdKnnFpr", mu.loc["kNN(k=10, 8d PCA)", "fpr_at_95tpr"], "{:.2f}")
    macro("zdLapAuc", mu.loc["OCSVM-Laplacian(8d,gamma=10.0)", "auc"], "{:.3f}")
    macro("zdIfFpr", mu.loc["IsolationForest", "fpr_at_95tpr"], "{:.2f}")
    macro("zdNoEntAuc", mu.loc[NOENT, "auc"], "{:.2f}")
    macro("zdMpsAuc", mu.loc["MPS-BornMachine(8d,chi=12)", "auc"], "{:.2f}")
    qk_cic = max(mc_.loc[[d for d in mc_.index if d.startswith("OCSVM-QuantumKernel")], "auc"])
    rbf_cic = max(mc_.loc[[d for d in mc_.index if d.startswith("OCSVM-RBF(8d")], "auc"])
    macro("zdCicQkAuc", qk_cic, "{:.3f}"); macro("zdCicRbfAuc", rbf_cic, "{:.3f}")
    macro("zdSeeds", zu["seed"].nunique(), "{:d}"); macro("zdFamilies", zu["family"].nunique(), "{:d}")
    ZL = [("kNN(k=10, 8d PCA)", "kNN, 8-d PCA"), ("OCSVM-Laplacian(8d,gamma=10.0)", "Laplacian-kernel OCSVM, 8-d"),
          ("OCSVM-RBF(8d,gamma=30.0)", "RBF OCSVM, 8-d"), (QK, "Quantum-kernel OCSVM, 8 qubits"),
          ("IsolationForest", "Isolation forest, 39-d"), ("Autoencoder", "Autoencoder (39-16-4)"),
          ("MPS-BornMachine(8d,chi=12)", "MPS Born machine (quantum-inspired)"), (NOENT, "Product-state kernel (no entanglement)")]
    rows = [[lab, f"{mu.loc[d,'auc']:.3f}", f"{mu.loc[d,'fpr_at_95tpr']:.2f}", f"{mc_.loc[d,'auc']:.3f}" if d in mc_.index else "--",
             f"{mc_.loc[d,'fpr_at_95tpr']:.2f}" if d in mc_.index else "--"] for d, lab in ZL]
    write("tab_qadv_zeroday.tex", booktabs(["Detector (benign-only training)", "UNSW AUC", "UNSW FPR@95", "CIC AUC", "CIC FPR@95"], rows, "lrrrr"))
    fs = pd.read_csv(os.path.join(qa, "item3_fewshot.csv")); pv = fs.pivot_table(index="model", columns="k", values="auc")
    ks = list(pv.columns)
    rows = [[m] + [f"{pv.loc[m,k]:.3f}" for k in ks] for m in pv.index]
    write("tab_qadv_fewshot.tex", booktabs(["Model"] + [f"$k={k}$" for k in ks], rows, "l" + "r" * len(ks)))
    macro("fsReps", fs["rep"].nunique(), "{:d}")
    qk_fs, rbf_fs = pv.loc["QuantumKernelSVM(8q)"], pv.loc["RBF-SVM(8d,gamma=0.3)"]
    macro("fsQkOne", qk_fs[1], "{:.2f}"); macro("fsRbfOne", rbf_fs[1], "{:.2f}")
    macro("fsQkTwenty", qk_fs[20], "{:.2f}"); macro("fsRbfTwenty", rbf_fs[20], "{:.2f}")

    # ------------------------------------------------- post-quantum TLS handshake sizes (measured)
    hs = pd.read_csv(os.path.join(RES, "qadv", "handshake_sizes.csv")); hs = hs[hs["ok"]]
    hm = hs.groupby(["cert", "group"])[["c2s", "s2c", "first_c2s"]].mean()
    macro("hsReps", int(hs.groupby(["cert", "group"]).size().min()), "{:d}")
    macro("hsChClassical", hm.loc[("ECDSA-P256", "X25519"), "first_c2s"], "{:.0f}")
    macro("hsChHybrid", hm.loc[("ECDSA-P256", "X25519MLKEM768"), "first_c2s"], "{:.0f}")
    macro("hsChMax", hm.loc[("ECDSA-P256", "MLKEM1024"), "first_c2s"], "{:.0f}")
    macro("hsChRatio", hm.loc[("ECDSA-P256", "X25519MLKEM768"), "first_c2s"] / hm.loc[("ECDSA-P256", "X25519"), "first_c2s"], "{:.1f}")
    macro("hsSrvClassical", hm.loc[("ECDSA-P256", "X25519"), "s2c"], "{:.0f}")
    macro("hsSrvHybridEcdsa", hm.loc[("ECDSA-P256", "X25519MLKEM768"), "s2c"], "{:.0f}")
    macro("hsSrvHybridMldsa", hm.loc[("ML-DSA-65", "X25519MLKEM768"), "s2c"], "{:.0f}")
    macro("hsSrvRatio", hm.loc[("ML-DSA-65", "X25519MLKEM768"), "s2c"] / hm.loc[("ECDSA-P256", "X25519"), "s2c"], "{:.0f}")
    macro("hsStdMax", hs.groupby(["cert", "group"])["c2s"].std().max(), "{:.0f}")
    rows = []
    for g in ["X25519", "secp256r1", "X25519MLKEM768", "SecP256r1MLKEM768", "MLKEM512", "MLKEM768", "MLKEM1024"]:
        rows.append([g, f"{hm.loc[('ECDSA-P256', g), 'c2s']:.0f}"] + [f"{hm.loc[(c, g), 's2c']:.0f}" for c in ("ECDSA-P256", "RSA-2048", "ML-DSA-44", "ML-DSA-65")])
    write("tab_handshake.tex", booktabs(["Key-exchange group", "Client$\\to$server", "ECDSA-P256", "RSA-2048", "ML-DSA-44", "ML-DSA-65"], rows, "lrrrrr"))

    body = "% AUTO-GENERATED by scripts/generate_paper_numbers.py -- do not edit.\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(MACROS.items()))
    write("generated_numbers.tex", body)
    with open(os.path.join(RES, "PAPER_NUMBERS.json"), "w", encoding="utf-8") as handle:
        json.dump(MACROS, handle, indent=2, sort_keys=True)
    print(f"wrote {len(MACROS)} macros and tables to {OUT}")


if __name__ == "__main__":
    main()
