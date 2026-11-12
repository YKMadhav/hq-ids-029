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
# Checkpoints: pre-trained weights in models/checkpoints/ are included in repo.
#   They are only needed by run_scaled_inference.py and Phase 4 scripts.
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
BASE_PYTHON=""
if [ -x "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" ]; then
    BASE_PYTHON="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
elif [ -x "/usr/local/bin/python3" ]; then
    BASE_PYTHON="/usr/local/bin/python3"
else
    BASE_PYTHON="python3"
fi

VENV_DIR="$REPO_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[Setup] Creating virtual environment at .venv using $BASE_PYTHON ..."
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
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
"$PYTHON" -m pytest -v

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
    echo "[REPRO_QUICK=1] Skipping long experiments (ablation, baselines, stress tests)."
else
    echo ""
    echo "[Step 4] Running baselines ..."
    "$PYTHON" experiments/run_baselines.py

    echo ""
    echo "[Step 5] Running QML comparison benchmark ..."
    "$PYTHON" experiments/run_qml_comparison.py

    echo ""
    echo "[Step 6] Running ablation and NISQ noise / M3 mitigation study ..."
    "$PYTHON" experiments/run_ablation.py

    echo ""
    echo "[Step 7] Running RFF surrogate benchmark ..."
    "$PYTHON" experiments/run_rff_surrogate_benchmark.py

    echo ""
    echo "[Step 8] Running cross-dataset benchmark ..."
    "$PYTHON" experiments/run_cross_dataset_benchmark.py

    echo ""
    echo "[Step 9] Running zero-day anomaly benchmark ..."
    "$PYTHON" scripts/run_zeroday_benchmark.py

    echo ""
    echo "[Step 10] Running stress tests (30 seeds) ..."
    "$PYTHON" experiments/run_stress_tests.py

    echo ""
    echo "[Step 11] Generating plots and report ..."
    "$PYTHON" experiments/generate_plots_and_report.py
fi

# ---------------------------------------------------------------------------
# 6. Compile LaTeX manuscript
# ---------------------------------------------------------------------------
if command -v pdflatex &>/dev/null; then
    echo ""
    echo "[Step 12] Compiling IEEE manuscript (paper_draft.tex) ..."
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
    echo "[Step 12] pdflatex not found; skipping PDF compilation."
fi

echo ""
echo "========================================================================"
echo "HQ-IDS-029: Reproduction pipeline complete."
echo "========================================================================"
