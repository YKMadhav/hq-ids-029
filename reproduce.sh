#!/usr/bin/env bash
# ==============================================================================
# Project HQ-IDS-029 (Quaiprobe) Full Reproduction Pipeline
# ==============================================================================
# Usage: ./reproduce.sh
# ==============================================================================

set -euo pipefail

echo "========================================================================"
echo "HQ-IDS-029: Commencing Forensic Verification and Pipeline Reproduction"
echo "========================================================================"

# Select python binary
if [ -x "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" ]; then
    PYTHON="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
else
    PYTHON="python3"
fi

# Step 1: Run forensic leakage test suite
echo ""
echo "[Step 1/4] Running pytest leakage and preprocessing verification..."
${PYTHON} -m pytest -v

# Step 2: Verify ablation statistics, Holm-Bonferroni, Cohen's d, and TOST
echo ""
echo "[Step 2/4] Verifying statistical tests and TOST equivalence..."
${PYTHON} scripts/recompute_stats.py

# Step 3: Compile LaTeX IEEE manuscript
echo ""
echo "[Step 3/4] Compiling IEEE conference manuscript (paper_draft.tex)..."
pdflatex -interaction=nonstopmode paper_draft.tex > /dev/null
pdflatex -interaction=nonstopmode paper_draft.tex | grep -E "Output written|Warning|Error"

# Step 4: Verify strictly <= 6 pages
echo ""
echo "[Step 4/4] Validating manuscript page count constraint..."
PAGES=$(${PYTHON} -c "
with open('paper_draft.log') as f:
    text = f.read()
import re
m = re.findall(r'Output written on paper_draft\.pdf \((\d+) pages', text)
print(m[-1] if m else 'Unknown')
")

echo "Total manuscript pages: ${PAGES}"
if [ "${PAGES}" -le 6 ]; then
    echo "PASS: Manuscript satisfies strict <= 6-page IEEE conference limit."
else
    echo "FAIL: Manuscript exceeds 6 pages (${PAGES} pages detected)."
    exit 1
fi

echo ""
echo "========================================================================"
echo "HQ-IDS-029: Full Pipeline Reproduction and Verification Successful!"
echo "========================================================================"
