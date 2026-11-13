"""
Manuscript-vs-results regression test for HQ-IDS-029.

Parses key numeric claims directly from paper_draft.tex and PAPER_MANUSCRIPT_IEEE.md
using regular expressions, and asserts that every extracted number matches the value
recomputed directly from results/*.csv and results/*.json within a tolerance of 0.0005.

If any assertion fails, the manuscript text is out of sync with the underlying evidence.
"""
import re
import json
import math
import numpy as np
import pandas as pd
import pytest

TOL = 0.0005  # 0.05 pp absolute tolerance

pytestmark = pytest.mark.manuscript_sync


# ---------------------------------------------------------------------------
# Helpers for ground truth computation
# ---------------------------------------------------------------------------

def _mean(values):
    return float(np.mean(values))

def _std(values, ddof=1):
    return float(np.std(values, ddof=ddof))


# ---------------------------------------------------------------------------
# Fixtures for Results Data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ablation_data():
    df = pd.read_csv("results/ablation_results.csv")
    out = {}
    for model in df["Model"].unique():
        rows = df[df["Model"] == model].sort_values("Seed")
        out[model] = {
            "mean_acc": _mean(rows["Accuracy"].values),
            "std_acc": _std(rows["Accuracy"].values),
            "mean_fpr": _mean(rows["FPR"].values),
            "std_fpr": _std(rows["FPR"].values),
            "mean_f1": _mean(rows["Macro_F1"].values),
            "std_f1": _std(rows["Macro_F1"].values),
        }
    return out


@pytest.fixture(scope="module")
def stress_30_data():
    df = pd.read_csv("results/stress_30seeds.csv")
    out = {}
    for model in df["Model"].unique():
        rows = df[df["Model"] == model].sort_values("Seed")
        out[model] = {
            "mean_acc": _mean(rows["Accuracy"].values),
            "std_acc": _std(rows["Accuracy"].values),
            "mean_f1": _mean(rows["Macro_F1"].values),
            "std_f1": _std(rows["Macro_F1"].values),
        }
    return out


@pytest.fixture(scope="module")
def zeroday_data():
    with open("results/zeroday_summary.json") as f:
        d = json.load(f)
    return {
        "hybrid_roc_auc_mean": float(d["models"]["HybridEnsemble"]["roc_auc_mean"]),
        "cae_roc_auc_mean": float(d["models"]["ClassicalAutoencoder"]["roc_auc_mean"]),
    }


# ---------------------------------------------------------------------------
# Fixtures for Manuscript Text
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tex_text():
    with open("paper_draft.tex", "r") as f:
        return f.read()


@pytest.fixture(scope="module")
def md_text():
    with open("PAPER_MANUSCRIPT_IEEE.md", "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Tests: 5-seed Ablation Claims (Parsed from TeX & MD)
# ---------------------------------------------------------------------------

def test_tex_5seed_vqc_accuracy(tex_text, ablation_data):
    """Regex-extract Full VQC 5-seed accuracy from paper_draft.tex and verify against CSV."""
    m = re.search(r"Full Hybrid VQC attains an accuracy of \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", tex_text)
    assert m is not None, "Failed to regex-match 5-seed Full VQC accuracy in paper_draft.tex"
    claimed_mean, claimed_std = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    actual_mean = ablation_data["Full_HybridQNN"]["mean_acc"]
    actual_std = ablation_data["Full_HybridQNN"]["std_acc"]
    assert abs(claimed_mean - actual_mean) <= TOL, f"TeX VQC acc mean {claimed_mean} != {actual_mean}"
    assert abs(claimed_std - actual_std) <= TOL, f"TeX VQC acc std {claimed_std} != {actual_std}"


def test_tex_5seed_ctrla_accuracy(tex_text, ablation_data):
    """Regex-extract Control A 5-seed accuracy from paper_draft.tex and verify against CSV."""
    m = re.search(r"compared to \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%(?:\$)? for Control A", tex_text)
    assert m is not None, "Failed to regex-match 5-seed Control A accuracy in paper_draft.tex"
    claimed_mean, claimed_std = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    actual_mean = ablation_data["Control_A_Quantum_Bypassed"]["mean_acc"]
    actual_std = ablation_data["Control_A_Quantum_Bypassed"]["std_acc"]
    assert abs(claimed_mean - actual_mean) <= TOL, f"TeX Control A acc mean {claimed_mean} != {actual_mean}"
    assert abs(claimed_std - actual_std) <= TOL, f"TeX Control A acc std {claimed_std} != {actual_std}"


def test_tex_5seed_ctrlb_accuracy(tex_text, ablation_data):
    """Regex-extract Control B 5-seed accuracy from paper_draft.tex and verify against CSV."""
    m = re.search(r"and \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%(?:\$)? for Control B", tex_text)
    assert m is not None, "Failed to regex-match 5-seed Control B accuracy in paper_draft.tex"
    claimed_mean, claimed_std = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    actual_mean = ablation_data["Control_B_Quantum_Frozen"]["mean_acc"]
    actual_std = ablation_data["Control_B_Quantum_Frozen"]["std_acc"]
    assert abs(claimed_mean - actual_mean) <= TOL, f"TeX Control B acc mean {claimed_mean} != {actual_mean}"
    assert abs(claimed_std - actual_std) <= TOL, f"TeX Control B acc std {claimed_std} != {actual_std}"


def test_tex_5seed_vqc_fpr(tex_text, ablation_data):
    """Regex-extract Full VQC 5-seed FPR from paper_draft.tex and verify against CSV."""
    m = re.search(r"FPR of \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", tex_text)
    assert m is not None, "Failed to regex-match 5-seed VQC FPR in paper_draft.tex"
    claimed_mean, claimed_std = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    actual_mean = ablation_data["Full_HybridQNN"]["mean_fpr"]
    actual_std = ablation_data["Full_HybridQNN"]["std_fpr"]
    assert abs(claimed_mean - actual_mean) <= TOL, f"TeX VQC FPR mean {claimed_mean} != {actual_mean}"
    assert abs(claimed_std - actual_std) <= TOL, f"TeX VQC FPR std {claimed_std} != {actual_std}"


def test_md_5seed_vqc_accuracy(md_text, ablation_data):
    """Regex-extract Full VQC 5-seed accuracy from PAPER_MANUSCRIPT_IEEE.md and verify against CSV."""
    m = re.search(r"Full Hybrid VQC attains an accuracy of \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", md_text)
    assert m is not None, "Failed to regex-match 5-seed Full VQC accuracy in PAPER_MANUSCRIPT_IEEE.md"
    claimed_mean, claimed_std = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    actual_mean = ablation_data["Full_HybridQNN"]["mean_acc"]
    actual_std = ablation_data["Full_HybridQNN"]["std_acc"]
    assert abs(claimed_mean - actual_mean) <= TOL
    assert abs(claimed_std - actual_std) <= TOL


def test_md_5seed_ctrla_accuracy(md_text, ablation_data):
    """Regex-extract Control A 5-seed accuracy from PAPER_MANUSCRIPT_IEEE.md and verify against CSV."""
    m = re.search(r"compared to \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%(?:\$)? for Control A", md_text)
    assert m is not None, "Failed to regex-match 5-seed Control A accuracy in PAPER_MANUSCRIPT_IEEE.md"
    claimed_mean, claimed_std = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    actual_mean = ablation_data["Control_A_Quantum_Bypassed"]["mean_acc"]
    actual_std = ablation_data["Control_A_Quantum_Bypassed"]["std_acc"]
    assert abs(claimed_mean - actual_mean) <= TOL
    assert abs(claimed_std - actual_std) <= TOL


def test_md_5seed_ctrlb_accuracy(md_text, ablation_data):
    """Regex-extract Control B 5-seed accuracy from PAPER_MANUSCRIPT_IEEE.md and verify against CSV."""
    m = re.search(r"and \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%(?:\$)? for Control B", md_text)
    assert m is not None, "Failed to regex-match 5-seed Control B accuracy in PAPER_MANUSCRIPT_IEEE.md"
    claimed_mean, claimed_std = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    actual_mean = ablation_data["Control_B_Quantum_Frozen"]["mean_acc"]
    actual_std = ablation_data["Control_B_Quantum_Frozen"]["std_acc"]
    assert abs(claimed_mean - actual_mean) <= TOL
    assert abs(claimed_std - actual_std) <= TOL


# ---------------------------------------------------------------------------
# Tests: 30-seed Stress Claims (Parsed from TeX & MD)
# ---------------------------------------------------------------------------

def test_tex_30seed_accuracies(tex_text, stress_30_data):
    """Regex-extract 30-seed accuracies from paper_draft.tex and verify against stress_30seeds.csv."""
    m_vqc = re.search(r"Full Hybrid VQC achieves \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", tex_text)
    m_a = re.search(r"Control A \(Bypassed\) achieves \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", tex_text)
    m_b = re.search(r"Control B \(Frozen\) achieves \$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", tex_text)
    assert m_vqc and m_a and m_b, "Failed to regex-match 30-seed cohort accuracies in paper_draft.tex"
    
    assert abs(float(m_vqc.group(1))/100.0 - stress_30_data["Full_HybridQNN"]["mean_acc"]) <= TOL
    assert abs(float(m_vqc.group(2))/100.0 - stress_30_data["Full_HybridQNN"]["std_acc"]) <= TOL
    assert abs(float(m_a.group(1))/100.0 - stress_30_data["Control_A_Quantum_Bypassed"]["mean_acc"]) <= TOL
    assert abs(float(m_a.group(2))/100.0 - stress_30_data["Control_A_Quantum_Bypassed"]["std_acc"]) <= TOL
    assert abs(float(m_b.group(1))/100.0 - stress_30_data["Control_B_Quantum_Frozen"]["mean_acc"]) <= TOL
    assert abs(float(m_b.group(2))/100.0 - stress_30_data["Control_B_Quantum_Frozen"]["std_acc"]) <= TOL


def test_md_30seed_accuracies(md_text, stress_30_data):
    """Regex-extract 30-seed accuracies from PAPER_MANUSCRIPT_IEEE.md and verify against stress_30seeds.csv."""
    m_vqc = re.search(r"Full Hybrid VQC:\s+\$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", md_text)
    m_a = re.search(r"Control A \(Bypassed\):\s+\$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", md_text)
    m_b = re.search(r"Control B \(Frozen\):\s+\$([0-9.]+)\\%\s+\\pm\s+([0-9.]+)\\%", md_text)
    assert m_vqc and m_a and m_b, "Failed to regex-match 30-seed cohort accuracies in PAPER_MANUSCRIPT_IEEE.md"
    
    assert abs(float(m_vqc.group(1))/100.0 - stress_30_data["Full_HybridQNN"]["mean_acc"]) <= TOL
    assert abs(float(m_vqc.group(2))/100.0 - stress_30_data["Full_HybridQNN"]["std_acc"]) <= TOL
    assert abs(float(m_a.group(1))/100.0 - stress_30_data["Control_A_Quantum_Bypassed"]["mean_acc"]) <= TOL
    assert abs(float(m_a.group(2))/100.0 - stress_30_data["Control_A_Quantum_Bypassed"]["std_acc"]) <= TOL
    assert abs(float(m_b.group(1))/100.0 - stress_30_data["Control_B_Quantum_Frozen"]["mean_acc"]) <= TOL
    assert abs(float(m_b.group(2))/100.0 - stress_30_data["Control_B_Quantum_Frozen"]["std_acc"]) <= TOL


# ---------------------------------------------------------------------------
# Test: Control A Exact Standalone Reproduction
# ---------------------------------------------------------------------------

def test_matched_mlp_reproduces_control_a():
    """Matched MLP (patience=20) must reproduce Control A on all 5 seeds to 1e-6."""
    matched = pd.read_csv("results/matched_mlp_results.csv").sort_values("seed")
    abl = pd.read_csv("results/ablation_results.csv")
    ctrl_a = abl[abl["Model"] == "Control_A_Quantum_Bypassed"].sort_values("Seed")
    for metric_m, metric_a in [("accuracy", "Accuracy"), ("f1_macro", "Macro_F1"),
                                 ("roc_auc", "ROC_AUC"), ("fpr", "FPR")]:
        max_diff = max(abs(a - b) for a, b in
                       zip(matched[metric_m].tolist(), ctrl_a[metric_a].tolist()))
        assert max_diff < 1e-6, (
            f"Matched MLP vs Control A: max abs diff on {metric_m} = {max_diff:.2e} >= 1e-6"
        )


# ---------------------------------------------------------------------------
# Test: Report & README Zero-Day Claims Parsed Against zeroday_summary.json
# ---------------------------------------------------------------------------

def test_readme_and_report_zeroday_claims(zeroday_data):
    """Regex-extract zero-day claims from README.md and PROJECT_FINAL_REPORT.md and verify against json."""
    with open("README.md", "r") as f:
        readme = f.read()
    with open("PROJECT_FINAL_REPORT.md", "r") as f:
        report = f.read()

    m_readme = re.search(r"Hybrid Ensemble \$([0-9.]+)", readme)
    assert m_readme is not None, "Failed to regex-match Hybrid Ensemble in README.md"
    assert abs(float(m_readme.group(1)) - zeroday_data["hybrid_roc_auc_mean"]) <= TOL

    m_report = re.search(r"Hybrid Ensemble\s*\|\s*([0-9.]+)", report)
    assert m_report is not None, "Failed to regex-match Hybrid Ensemble in PROJECT_FINAL_REPORT.md"
    assert abs(float(m_report.group(1)) - zeroday_data["hybrid_roc_auc_mean"]) <= TOL
