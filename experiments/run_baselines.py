"""
Project HQ-IDS-029: Step 1 - Classical Baseline Benchmark
=========================================================
Evaluates three classical intrusion detectors on the real UNSW-NB15 flow sample:

  1. RandomForest   (n_estimators=100, max_depth=12)
  2. XGBoost        (n_estimators=100, max_depth=6, learning_rate=0.1)
  3. ClassicalMLP   (39 -> 16 -> 4 -> 2, bottleneck depth matched to the hybrid QNN)

Protocol: 5 seeds [42, 43, 44, 45, 46]; each seed gets its own deterministic
stratified 70/10/20 train/validation/test partition; the preprocessor is fitted on
the train partition only.

Reported metrics: Accuracy, Macro Precision/Recall, Macro F1, Weighted F1,
ROC-AUC, False Positive Rate (FP/(FP+TN)) and single-sample inference latency.

Outputs:
  results/metrics.csv            aggregated (mean +/- std) rows, one per model
  results/baselines_all_seeds.csv one row per (seed, model)
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    RESULTS_DIR,
    aggregate_by_model,
    compute_metrics,
    deduplicate_flows,
    load_config,
    load_flow_data,
    log_decision,
    log_event,
    preprocess_split,
    stratified_split,
)
from src.classical_models import (  # noqa: E402
    ClassicalMLPBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from src.trivial_baselines import MajorityClassBaseline, SingleSplitBaseline  # noqa: E402

METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "balanced_accuracy",
    "mcc",
    "test_attack_rate",
    "fpr",
    "latency_ms",
]


def run_baselines_benchmark(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    test_size = config.get("dataset", {}).get("test_size", 0.2)
    val_size = config.get("dataset", {}).get("val_size", 0.1)
    latency_runs = config.get("baselines", {}).get("latency_runs", 1000)

    rf_cfg = config.get("baselines", {}).get("random_forest", {})
    xgb_cfg = config.get("baselines", {}).get("xgboost", {})
    mlp_cfg = config.get("baselines", {}).get("mlp", {})

    raw_df = load_flow_data(config)
    print(f"[Benchmark] Ingested {len(raw_df)} real UNSW-NB15 flows "
          f"({int((raw_df['label'] == 1).sum())} attack / {int((raw_df['label'] == 0).sum())} benign, "
          f"attack ratio {raw_df['label'].mean():.2%})")

    n_unique = len(deduplicate_flows(raw_df))
    print(f"[Benchmark] After exact-duplicate removal: {n_unique} unique flows "
          f"({len(raw_df) - n_unique} duplicates dropped before splitting)")

    detailed_records: List[Dict[str, Any]] = []
    feature_dim = None

    for seed_idx, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 66)
        print(f"  BASELINE BENCHMARK {seed_idx}/{len(seeds)}  (seed={seed})")
        print("=" * 66)

        train_df, val_df, test_df = stratified_split(raw_df, seed, test_size, val_size)
        _, arrays = preprocess_split(train_df, val_df, test_df, config)
        X_train, y_train = arrays["train"]
        X_val, y_val = arrays["val"]
        X_test, y_test = arrays["test"]
        feature_dim = X_train.shape[1]

        print(f"[seed {seed}] train={len(X_train)} val={len(X_val)} test={len(X_test)} "
              f"| model input dim={feature_dim}")

        models = {
            # Reference detectors: every headline number must be read against these.
            "MajorityClass(flag-all)": MajorityClassBaseline(random_state=seed),
            "SingleSplitStump": SingleSplitBaseline(random_state=seed),
            "RandomForest": RandomForestBaseline(
                n_estimators=rf_cfg.get("n_estimators", 100),
                max_depth=rf_cfg.get("max_depth", 12),
                random_state=seed,
            ),
            "XGBoost": XGBoostBaseline(
                n_estimators=xgb_cfg.get("n_estimators", 100),
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.1),
                random_state=seed,
            ),
            "ClassicalMLP": ClassicalMLPBaseline(
                input_dim=feature_dim,
                hidden_dim=mlp_cfg.get("hidden_dim", 16),
                bottleneck_dim=mlp_cfg.get("bottleneck_dim", 4),
                output_dim=mlp_cfg.get("output_dim", 2),
                learning_rate=mlp_cfg.get("learning_rate", 0.005),
                batch_size=mlp_cfg.get("batch_size", 64),
                epochs=mlp_cfg.get("epochs", 40),
                patience=mlp_cfg.get("patience", 8),
                random_state=seed,
            ),
        }

        for model_name, model in models.items():
            start = time.perf_counter()
            model.fit(X_train, y_train, X_val, y_val)
            train_time = time.perf_counter() - start

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            metrics = compute_metrics(y_test, y_pred, y_prob)

            latency_ms = model.measure_latency(X_test[0], n_runs=latency_runs)

            print(
                f"  [{model_name:24s}] Acc {metrics['accuracy']:.4f} | "
                f"F1-Macro {metrics['f1_macro']:.4f} | ROC-AUC {metrics['roc_auc']:.4f} | "
                f"FPR {metrics['fpr']:.4f} | Latency {latency_ms:.4f} ms | Train {train_time:.2f}s"
            )

            detailed_records.append({
                "seed": seed,
                "model": model_name,
                "n_features": feature_dim,
                **metrics,
                "latency_ms": float(latency_ms),
                "train_time_sec": float(train_time),
            })

    detailed_df = pd.DataFrame(detailed_records)
    summary_df = aggregate_by_model(detailed_df, "model", METRIC_COLUMNS)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "metrics.csv")
    detailed_path = os.path.join(RESULTS_DIR, "baselines_all_seeds.csv")
    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)

    print(f"\n[Benchmark] Aggregated metrics -> {summary_path}")
    print(f"[Benchmark] Per-seed trials     -> {detailed_path}")
    return summary_df, detailed_df


def main() -> None:
    print("=" * 70)
    print("Project HQ-IDS-029: Step 1 - Classical Baseline Benchmark")
    print("=" * 70)

    config = load_config()
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    log_event("ACTION",
              f"Started Step 1 classical baseline benchmark over seeds {seeds} on real UNSW-NB15 flows.")

    summary, detailed = run_baselines_benchmark(config)

    log_event("VERIFICATION",
              f"Step 1 complete: {len(detailed)} per-seed trials ({detailed['model'].nunique()} models x "
              f"{len(seeds)} seeds) written to results/baselines_all_seeds.csv and results/metrics.csv.")

    print("\n" + "=" * 70)
    print("  5-SEED AGGREGATED CLASSICAL BASELINE RESULTS")
    print("=" * 70)
    print(summary[[
        "model",
        "accuracy_mean", "accuracy_std",
        "f1_macro_mean", "f1_macro_std",
        "f1_weighted_mean", "f1_weighted_std",
        "roc_auc_mean", "roc_auc_std",
        "fpr_mean", "fpr_std",
        "latency_ms_mean", "latency_ms_std",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
