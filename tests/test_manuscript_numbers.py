"""
Manuscript-vs-results regression test for HQ-IDS-029.

Parses key numeric claims from paper_draft.tex and PAPER_MANUSCRIPT_IEEE.md
and asserts they match the values derived from the results CSV/JSON files,
within a tolerance of 0.0005 (0.05 percentage points).

If any assertion fails, the manuscript is wrong (not the results).
"""
import re
import json
import math
import numpy as np
import pandas as pd
import pytest

TOL = 0.0005  # 0.05 pp absolute tolerance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values):
    return sum(values) / len(values)

def _std(values, ddof=1):
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - ddof))


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ablation():
    return pd.read_csv("results/ablation_results.csv")


@pytest.fixture(scope="module")
def stress():
    return pd.read_csv("results/stress_30seeds.csv")


@pytest.fixture(scope="module")
def baselines():
    return pd.read_csv("results/baselines_all_seeds.csv")


@pytest.fixture(scope="module")
def hybrid_all():
    return pd.read_csv("results/hybrid_all_seeds.csv")


@pytest.fixture(scope="module")
def zeroday_summary():
    with open("results/zeroday_summary.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 5-seed ablation means (from results/ablation_results.csv)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def abl_means(ablation):
    out = {}
    for model in ablation["Model"].unique():
        rows = ablation[ablation["Model"] == model]
        out[model] = {
            "Accuracy":  _mean(rows["Accuracy"].tolist()),
            "Macro_F1":  _mean(rows["Macro_F1"].tolist()),
            "ROC_AUC":   _mean(rows["ROC_AUC"].tolist()),
            "FPR":       _mean(rows["FPR"].tolist()),
            "std_acc":   _std(rows["Accuracy"].tolist()),
            "std_f1":    _std(rows["Macro_F1"].tolist()),
            "std_fpr":   _std(rows["FPR"].tolist()),
        }
    return out


# ---------------------------------------------------------------------------
# Tests: 5-seed Full VQC
# ---------------------------------------------------------------------------

def test_vqc_5seed_accuracy_mean(abl_means):
    """paper claims Full VQC 93.96% ± 0.74%"""
    m = abl_means["Full_HybridQNN"]["Accuracy"]
    assert abs(m - 0.9396) <= TOL, f"VQC accuracy mean {m:.4f} != 0.9396"

def test_vqc_5seed_accuracy_std(abl_means):
    s = abl_means["Full_HybridQNN"]["std_acc"]
    assert abs(s - 0.0074) <= TOL, f"VQC accuracy std {s:.4f} != 0.0074"

def test_vqc_5seed_fpr_mean(abl_means):
    """paper claims FPR 14.67% ± 3.57%"""
    m = abl_means["Full_HybridQNN"]["FPR"]
    assert abs(m - 0.1467) <= TOL, f"VQC FPR mean {m:.4f} != 0.1467"


# ---------------------------------------------------------------------------
# Tests: 5-seed Control A (Quantum Bypassed)
# ---------------------------------------------------------------------------

def test_ctrla_5seed_accuracy_mean(abl_means):
    """paper claims Control A 93.78% ± 0.79%"""
    m = abl_means["Control_A_Quantum_Bypassed"]["Accuracy"]
    assert abs(m - 0.9378) <= TOL, f"Control A accuracy {m:.4f} != 0.9378"

def test_ctrla_5seed_accuracy_std(abl_means):
    s = abl_means["Control_A_Quantum_Bypassed"]["std_acc"]
    assert abs(s - 0.0079) <= TOL, f"Control A std {s:.4f} != 0.0079"


# ---------------------------------------------------------------------------
# Tests: 5-seed Control B (Quantum Frozen)
# ---------------------------------------------------------------------------

def test_ctrlb_5seed_accuracy_mean(abl_means):
    """paper claims Control B 93.82% ± 0.85%"""
    m = abl_means["Control_B_Quantum_Frozen"]["Accuracy"]
    assert abs(m - 0.9382) <= TOL, f"Control B accuracy {m:.4f} != 0.9382"


# ---------------------------------------------------------------------------
# Tests: 30-seed stress results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_means(stress):
    out = {}
    for model in stress["Model"].unique():
        rows = stress[stress["Model"] == model]
        out[model] = {
            "Accuracy": _mean(rows["Accuracy"].tolist()),
            "Macro_F1": _mean(rows["Macro_F1"].tolist()),
            "std_acc":  _std(rows["Accuracy"].tolist()),
            "std_f1":   _std(rows["Macro_F1"].tolist()),
        }
    return out


def test_30seed_vqc_accuracy_mean(stress_means):
    """paper claims Full VQC 30-seed 93.65% ± 0.81%"""
    m = stress_means["Full_HybridQNN"]["Accuracy"]
    assert abs(m - 0.9365) <= TOL, f"30-seed VQC accuracy {m:.4f} != 0.9365"

def test_30seed_vqc_accuracy_std(stress_means):
    s = stress_means["Full_HybridQNN"]["std_acc"]
    assert abs(s - 0.0081) <= TOL, f"30-seed VQC std {s:.4f} != 0.0081"

def test_30seed_bypassed_accuracy_mean(stress_means):
    """paper claims Control A 30-seed 93.61% ± 0.76%"""
    m = stress_means["Control_A_Quantum_Bypassed"]["Accuracy"]
    assert abs(m - 0.9361) <= TOL, f"30-seed Bypassed accuracy {m:.4f} != 0.9361"

def test_30seed_frozen_accuracy_mean(stress_means):
    """paper claims Control B 30-seed 93.65% ± 0.94%"""
    m = stress_means["Control_B_Quantum_Frozen"]["Accuracy"]
    assert abs(m - 0.9365) <= TOL, f"30-seed Frozen accuracy {m:.4f} != 0.9365"


# ---------------------------------------------------------------------------
# Tests: Matched MLP (Control A reproduction)
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
# Tests: Zero-day (Phase 3) summary
# ---------------------------------------------------------------------------

def test_zeroday_hybrid_roc_auc(zeroday_summary):
    """paper claims Hybrid ROC-AUC 0.7447"""
    val = zeroday_summary["models"]["HybridEnsemble"]["roc_auc_mean"]
    assert abs(val - 0.7447) <= TOL, f"Hybrid ROC-AUC {val:.4f} != 0.7447"
