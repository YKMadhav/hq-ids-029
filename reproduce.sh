#!/usr/bin/env bash
# ==============================================================================
# Project HQ-IDS-029 (Quaiprobe) Full Reproduction Pipeline
# ==============================================================================
# Usage:
#   ./reproduce.sh               # full pipeline
#   REPRO_QUICK=1 ./reproduce.sh # skip long experiments; run only stats + tests
#
# Dataset: data/real_flows.csv (5,000 UNSW-NB15 flows, included in repo).
#   Full UNSW-NB15 CSVs for Phase 4 proxy:
#     data/UNSW_NB15_training-set.csv
#     data/UNSW_NB15_testing-set.csv
#   Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
#
# Phase 4 requires the official CSVs and is run only when they are present.
# Historical reports are never overwritten unless REGENERATE_REPORT=1 is set.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "========================================================================"
echo "HQ-IDS-029: Commencing Forensic Verification and Pipeline Reproduction"
echo "========================================================================"

# ---------------------------------------------------------------------------
# 1. Create / activate a virtual environment and install dependencies
# ---------------------------------------------------------------------------
# The pinned stack needs Python 3.11: pandas==2.0.3 has no wheels for >=3.12
# (installing on 3.12/3.13 fails with a source-build error).
if [ -n "${PYTHON_BIN:-}" ]; then
    BASE_PYTHON="$PYTHON_BIN"
elif command -v python3.11 &>/dev/null; then
    BASE_PYTHON="python3.11"
else
    BASE_PYTHON="python3"
fi
BASE_VERSION="$("$BASE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$BASE_VERSION" != "3.11" ]; then
    echo "ERROR: Python 3.11 is required (found $BASE_VERSION via $BASE_PYTHON)." >&2
    echo "       Set PYTHON_BIN=/path/to/python3.11 (the pins, e.g. pandas==2.0.3, do not install on >=3.12)." >&2
    exit 1
fi

# Results are only portable across machines when thread counts are pinned (src/determinism.py).
unset HQIDS_ALLOW_MULTITHREAD

VENV_DIR="$REPO_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[Setup] Creating virtual environment at .venv using $BASE_PYTHON ..."
    "$BASE_PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python3"

echo "[Setup] Installing / verifying dependencies from requirements.txt ..."
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt

# ---------------------------------------------------------------------------
# 2. Pytest leakage / preprocessing suite
# ---------------------------------------------------------------------------
echo ""
echo "[Step 1] Running pytest leakage and preprocessing verification ..."
"$PYTHON" -m pytest -v -m "not manuscript_sync"

# Manuscript-vs-results sync is reported, not gated: the manuscripts quote the pre-correction
# numbers (duplicate-inflated, thread-dependent) and are marked historical in SCIENTIFIC_STATUS.md.
# These tests are expected to FAIL after a corrected rerun until the paper is rewritten from
# the regenerated artifacts; a failure here is the tripwire that says so.
"$PYTHON" -m pytest -q -m manuscript_sync 2>&1 | tail -3 || true

# ---------------------------------------------------------------------------
# 3. Recompute statistics
# ---------------------------------------------------------------------------
echo ""
echo "[Step 2] Verifying statistical tests and TOST equivalence ..."
"$PYTHON" scripts/recompute_stats.py

# ---------------------------------------------------------------------------
# 4. Matched-MLP reproduction (always runs; it is fast)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 3] Reproducing matched-MLP Control A results ..."
"$PYTHON" scripts/run_matched_mlp.py

# ---------------------------------------------------------------------------
# 5. Full experiments (skipped in REPRO_QUICK mode)
# ---------------------------------------------------------------------------
if [ "${REPRO_QUICK:-0}" = "1" ]; then
    echo ""
    echo "[REPRO_QUICK=1] Skipping long experiments."
else
    echo ""
    echo "[Step 4] Running baselines (writes results/metrics.csv) ..."
    "$PYTHON" experiments/run_baselines.py

    echo ""
    echo "[Step 4b] Running the HybridQNN 5-seed benchmark (appends the HybridQNN row to metrics.csv) ..."
    echo "          (run_baselines.py overwrites metrics.csv, so this MUST come after it)"
    "$PYTHON" experiments/run_hybrid_sim.py

    echo ""
    echo "[Step 5] Running QML comparison benchmark ..."
    "$PYTHON" experiments/run_qml_comparison.py

    echo ""
    echo "[Step 6] Running ablation and NISQ noise / M3 mitigation study ..."
    "$PYTHON" experiments/run_ablation.py

    if [ -f data/aligned_data_manifest.json ]; then
        echo ""
        echo "[Step 7] Running RFF surrogate benchmark ..."
        "$PYTHON" experiments/run_rff_surrogate_benchmark.py

        echo ""
        echo "[Step 8] Running cross-dataset benchmark ..."
        "$PYTHON" experiments/run_cross_dataset_benchmark.py
    else
        echo ""
        echo "[Step 7/8] SKIPPED: data/aligned_data_manifest.json is absent. The aligned UNSW/CIC CSVs have"
        echo "           no generating code or verified field semantics (see data/ALIGNED_DATA_PROVENANCE.md);"
        echo "           the RFF-surrogate and cross-dataset claims are therefore withdrawn, not reproduced."
    fi

    echo ""
    echo "[Step 9] Running zero-day anomaly benchmark ..."
    "$PYTHON" scripts/run_zeroday_benchmark.py

    echo ""
    echo "[Step 10] Running stress tests (30 seeds) ..."
    "$PYTHON" experiments/run_stress_tests.py

    echo ""
    echo "[Step 11] Running multiclass benchmark ..."
    "$PYTHON" scripts/run_multiclass_benchmark.py

    if [ -f data/UNSW_NB15_training-set.csv ] && [ -f data/UNSW_NB15_testing-set.csv ]; then
        echo ""
        echo "[Step 12] Running Phase 4 CSV setup-header proxy ..."
        "$PYTHON" scripts/run_phase4_earlypacket_benchmark.py
    else
        echo "[Step 12] Skipping Phase 4: official UNSW training/testing CSVs are absent."
    fi

    if [ -f data/UNSW_NB15_testing-set.csv ]; then
        echo ""
        echo "[Step 13] Official-test-set held-out evaluation (all models, 5 seeds) ..."
        "$PYTHON" scripts/run_official_heldout_eval.py
        "$PYTHON" scripts/run_official_heldout_eval.py --dedup-test --tag _dedup_test \
            --models Majority,Stump,RandomForest,XGBoost,ClassicalMLP
    else
        echo "[Step 13] SKIPPED: data/UNSW_NB15_testing-set.csv is absent."
    fi

    echo ""
    echo "[Step 13b] Official-protocol benchmarks (primary results for the paper) ..."
    if [ -f data/UNSW_NB15_training-set.csv ] && [ -f data/UNSW_NB15_testing-set.csv ]; then
        "$PYTHON" scripts/run_official_heldout_eval.py --train-source official --max-train 30000 --tag _official_30k
        "$PYTHON" scripts/run_official_heldout_eval.py --train-source official --max-train 30000 \
            --models ClassicalMLP,HybridQNN,QNN_Bypassed,QNN_Frozen --tag _official_30k_ablation
        "$PYTHON" scripts/run_official_heldout_eval.py --train-source official \
            --models Majority,Stump,RandomForest,XGBoost,ClassicalMLP --tag _official_full
        "$PYTHON" scripts/check_thread_sensitivity.py > /dev/null
        "$PYTHON" scripts/generate_paper_numbers.py
        "$PYTHON" -m pytest -q tests/test_paper_numbers.py
    else
        echo "           SKIPPED: official training/testing CSVs absent (data/UNSW_NB15_*-set.csv)."
    fi

    echo ""
    echo "[Step 14] Writing run manifest (versions, thread pinning, data hashes) ..."
    "$PYTHON" scripts/write_run_manifest.py

    if [ "${REGENERATE_REPORT:-0}" = "1" ]; then
        echo ""
        echo "[Optional] Regenerating PROJECT_FINAL_REPORT.md from current artifacts ..."
        "$PYTHON" experiments/generate_plots_and_report.py
    else
        echo "[Optional] Historical report left untouched (set REGENERATE_REPORT=1 to replace it)."
    fi
fi

# ---------------------------------------------------------------------------
# 6. Compile LaTeX manuscript
# ---------------------------------------------------------------------------
# The manuscript is historical (see SCIENTIFIC_STATUS.md); compiling it is optional and must
# never turn a finished experimental run into a failed one.
if command -v pdflatex &>/dev/null && command -v kpsewhich &>/dev/null && [ -n "$(kpsewhich IEEEtran.cls 2>/dev/null)" ]; then
    echo ""
    echo "[Final] Compiling IEEE manuscript (paper_draft.tex) ..."
    pdflatex -interaction=nonstopmode paper_draft.tex > /dev/null
    pdflatex -interaction=nonstopmode paper_draft.tex 2>&1 | grep -E "Output written|Warning|Error"

    PAGES=$("$PYTHON" -c "
import re
with open('paper_draft.log') as f:
    text = f.read()
m = re.findall(r'Output written on paper_draft\.pdf \((\d+) pages', text)
print(m[-1] if m else 'Unknown')
")
    echo "Manuscript pages: ${PAGES}"
    if [ "${PAGES}" -le 6 ]; then
        echo "PASS: manuscript <= 6 pages."
    else
        echo "WARN: manuscript exceeds 6 pages (${PAGES} pages)."
    fi
else
    echo "[Final] pdflatex or IEEEtran.cls not available; skipping optional PDF compilation."
fi

echo ""
echo "========================================================================"
echo "HQ-IDS-029: Reproduction pipeline complete."
echo "========================================================================"
