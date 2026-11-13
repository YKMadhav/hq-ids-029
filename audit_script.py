"""Validate the actual Phase 4 scaled-inference artifact schema.

This is intentionally a validator, not an experiment runner: it refuses absent
labels/predictions instead of printing a misleading success message.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "scaled_inference_summary.json"
PREDICTIONS = ROOT / "results" / "scaled_predictions.csv"


def main() -> None:
    if not SUMMARY.is_file() or not PREDICTIONS.is_file():
        raise FileNotFoundError("Run scripts/run_scaled_inference.py before validating its artifacts.")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    frame = pd.read_csv(PREDICTIONS)
    label_column = "true_label" if "true_label" in frame else "label"
    if label_column not in frame:
        raise ValueError("scaled_predictions.csv has no ground-truth 'true_label' or 'label' column.")
    prediction_columns = [column for column in frame if column.startswith("pred_") or column.endswith("_pred")]
    if not prediction_columns:
        raise ValueError("scaled_predictions.csv has no '*_pred' columns.")
    if not set(np.unique(frame[label_column])).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1 for this validator.")
    print(f"Summary model entries: {sorted(summary.get('models', {}))}")
    for column in prediction_columns:
        prediction = frame[column]
        if not set(np.unique(prediction)).issubset({0, 1}):
            raise ValueError(f"{column} is not binary.")
        accuracy = float((prediction == frame[label_column]).mean())
        negatives = frame[label_column] == 0
        fpr = float(((prediction == 1) & negatives).sum() / negatives.sum()) if negatives.any() else 0.0
        print(f"{column}: accuracy={accuracy:.4f}, fpr={fpr:.4f}")


if __name__ == "__main__":
    main()
