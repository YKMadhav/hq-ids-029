"""
Project HQ-IDS-029: Shared Experiment Utilities
==============================================
Common helpers used by every script in `experiments/`:
- configuration loading
- deterministic, stratified 70/10/20 partitioning of the real UNSW-NB15 flows
- leakage-free preprocessing (the scaler is fitted on the train partition only)
- the shared metric block (accuracy, macro/weighted F1, ROC-AUC, FPR)
- timestamped audit logging to EXECUTION_LOG.md

Keeping these in one place guarantees every experiment uses the identical split
and metric definitions, so cross-experiment numbers are directly comparable.
"""

import os
import sys
import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import determinism as _determinism  # noqa: E402,F401  (pins threads before any torch work)
from src.data_loader import FlowPreprocessor  # noqa: E402

CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "config.yaml")
EXECUTION_LOG = os.path.join(REPO_ROOT, "EXECUTION_LOG.md")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_dataset_path(config: Dict[str, Any]) -> str:
    raw = config.get("dataset", {}).get("path", "./data/real_flows.csv")
    return os.path.join(REPO_ROOT, raw.lstrip("./"))


def load_flow_data(config: Dict[str, Any]) -> pd.DataFrame:
    path = resolve_dataset_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required real dataset is missing: {path}. Run src/fetch_real_dataset.py first."
        )
    return pd.read_csv(path)


IDENTIFIER_AND_LABEL_COLUMNS = ("id", "label", "attack_cat", "attack_category", "binary_label", "clean_cat")


def deduplicate_flows(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows whose complete feature vector (every column except the record id and
    the label columns) already occurred earlier in the frame.

    Why: in the 5,000-flow UNSW-NB15 sample 571 rows are exact feature-vector
    duplicates of another row. Under a random split ~17 % of test flows then have an
    identical twin in the training partition (166 / 1000 at seed 42), which is
    memorisation credit, not generalisation. The first occurrence is kept, so the
    result does not depend on the split seed. Rows whose duplicates carry
    conflicting labels are kept once with the label of the first occurrence.
    """
    feature_cols = [c for c in frame.columns if c not in IDENTIFIER_AND_LABEL_COLUMNS]
    return frame.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)


def stratified_split(
    frame: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    deduplicate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic stratified 70 / 10 / 20 train / validation / test partition.

    ``deduplicate=True`` (default) removes exact feature-vector duplicates BEFORE
    splitting so no test flow has an identical twin in train. Pass False only to
    reproduce the historical protocol.
    """
    if deduplicate:
        frame = deduplicate_flows(frame)
    train_val_df, test_df = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame["label"]
    )
    val_rel_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_rel_size, random_state=seed, stratify=train_val_df["label"]
    )
    return train_df, val_df, test_df


def preprocess_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[FlowPreprocessor, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Fits the preprocessor strictly on the train partition and applies it to all
    three partitions, so no validation/test statistic leaks into training.
    """
    dataset_cfg = config.get("dataset", {})
    preprocessor = FlowPreprocessor(
        feature_range=tuple(dataset_cfg.get("scaler_range", [-np.pi, np.pi])),
        log_transform_skewed=dataset_cfg.get("log_transform_skewed", True),
        skew_threshold=dataset_cfg.get("skew_threshold", 1.0),
    )
    arrays = {}
    arrays["train"] = preprocessor.fit_transform(train_df)
    arrays["val"] = preprocessor.transform(val_df)
    arrays["test"] = preprocessor.transform(test_df)
    return preprocessor, arrays


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Shared metric block.
    FPR = FP / (FP + TN) is the benign-class false alarm rate.
    `y_prob` may be either a 1-D positive-class score or a (N, 2) probability matrix.
    """
    scores = np.asarray(y_prob)
    if scores.ndim == 2:
        if scores.shape[1] != 2:
            raise ValueError(f"Expected 2 probability columns, got {scores.shape[1]}")
        scores = scores[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        # Accuracy depends on class prevalence (a flag-everything detector scores the
        # attack rate); balanced accuracy and MCC do not, so report them alongside it.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "test_attack_rate": float(np.mean(np.asarray(y_true) == 1)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def aggregate_by_model(
    frame: pd.DataFrame,
    model_column: str,
    metric_columns: list,
) -> pd.DataFrame:
    """Mean and sample standard deviation of each metric, grouped by model."""
    rows = []
    for model_name, group in frame.groupby(model_column, sort=False):
        row: Dict[str, Any] = {model_column: model_name}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def log_event(category: str, message: str) -> None:
    """Appends a timestamped audit entry to EXECUTION_LOG.md."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EXECUTION_LOG, "a", encoding="utf-8") as handle:
        handle.write(f"- `[{timestamp}] [{category}]` {message}\n")


def log_decision(message: str) -> None:
    log_event("AUTONOMOUS CHOICE", message)
