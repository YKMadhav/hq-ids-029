"""
Held-out evaluation on the official UNSW-NB15 testing set.
==========================================================
The headline benchmark re-splits one 5,000-flow sample, so its test fold has only
~886 flows (SE of accuracy ~ 0.8 points) and shares captures, attack mix and
near-duplicate rows with the training fold. This script gives a larger, differently
distributed check:

  train   : the de-duplicated 5,000-flow sample, MINUS every flow that also appears
            (identical feature vector) in the official testing set
  test    : the official testing set (82,332 flows) MINUS every flow that appears in
            the sample. Repeated flows INSIDE the test set are kept (repeated traffic is
            real traffic; they cannot leak into training because train/test are
            asserted disjoint). ``--dedup-test`` reports the stricter variant that
            also collapses within-test duplicates - it changes the attack prevalence
            drastically (the official test set is ~1/3 duplicate attack vectors)
  seeds   : 42-46 (they change the 90/10 train/validation split and initialisation;
            the test set is the same for every seed, so this measures training
            variance only)

Caveat that must travel with the numbers: the official training/testing CSVs are a
random partition of the same captures, so this is a larger and prevalence-shifted
test, NOT a temporal or cross-network one. Accuracy and FPR depend on the attack
prevalence, which differs between the sample and the official test; read AUC,
balanced accuracy and MCC for the model comparison.

Usage:
    python scripts/run_official_heldout_eval.py [--official-test PATH]
        [--models Majority,Stump,RandomForest,XGBoost,ClassicalMLP,HybridQNN]
        [--max-test N]   # optional fixed stratified subsample for speed

Outputs: results/official_heldout_results.csv, results/official_heldout_summary.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

from common import (  # noqa: E402
    RESULTS_DIR,
    compute_metrics,
    deduplicate_flows,
    load_config,
    load_flow_data,
    log_event,
    IDENTIFIER_AND_LABEL_COLUMNS,
)
from src.classical_models import ClassicalMLPBaseline, RandomForestBaseline, XGBoostBaseline  # noqa: E402
from src.data_loader import FlowPreprocessor  # noqa: E402
from src.hybrid_pipeline import HybridQNNClassifier  # noqa: E402
from src.trivial_baselines import MajorityClassBaseline, SingleSplitBaseline  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ALL_MODELS = ["Majority", "Stump", "RandomForest", "XGBoost", "ClassicalMLP", "HybridQNN", "QNN_Bypassed", "QNN_Frozen"]


def row_hashes(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in frame.columns if c not in IDENTIFIER_AND_LABEL_COLUMNS]
    return pd.util.hash_pandas_object(frame[cols], index=False)


def build_partitions(sample: pd.DataFrame, official_test: pd.DataFrame, max_test: int = 0, dedup_test: bool = False):
    sample_hash_all = set(row_hashes(sample))
    official_hash = set(row_hashes(official_test))

    unique_sample = deduplicate_flows(sample)
    train_pool = unique_sample[~row_hashes(unique_sample).isin(official_hash)].reset_index(drop=True)

    test = official_test[~row_hashes(official_test).isin(sample_hash_all)].reset_index(drop=True)
    n_test_before_dedup = len(test)
    if dedup_test:
        test = deduplicate_flows(test)

    if max_test and len(test) > max_test:
        test, _ = train_test_split(test, train_size=max_test, random_state=0, stratify=test["label"])
        test = test.reset_index(drop=True)

    # Hard guarantee: no identical feature vector on both sides.
    assert not set(row_hashes(train_pool)) & set(row_hashes(test)), "train/test overlap survived filtering"

    info = {
        "sample_rows": int(len(sample)),
        "sample_unique_rows": int(len(unique_sample)),
        "train_pool_rows": int(len(train_pool)),
        "train_attack_rate": float(train_pool["label"].mean()),
        "official_test_rows": int(len(official_test)),
        "official_test_attack_rate": float(official_test["label"].mean()),
        "test_rows_after_overlap_removal": int(n_test_before_dedup),
        "test_rows_final": int(len(test)),
        "test_attack_rate": float(test["label"].mean()),
        "max_test_subsample": int(max_test),
        "within_test_deduplicated": bool(dedup_test),
    }
    return train_pool, test, info


def build_official_partitions(official_train: pd.DataFrame, official_test: pd.DataFrame, dedup_test: bool = False):
    """Primary protocol: official training partition -> official testing partition.

    train : official training set, exact feature-vector duplicates collapsed, minus every flow
            whose feature vector also occurs in the official testing set
    test  : official testing set minus every flow whose feature vector occurs in the training
            set (within-test repeats are kept unless ``dedup_test``)
    """
    train_hash_all = set(row_hashes(official_train))
    test_hash_all = set(row_hashes(official_test))
    unique_train = deduplicate_flows(official_train)
    train_pool = unique_train[~row_hashes(unique_train).isin(test_hash_all)].reset_index(drop=True)
    test = official_test[~row_hashes(official_test).isin(train_hash_all)].reset_index(drop=True)
    if dedup_test:
        test = deduplicate_flows(test)
    assert not set(row_hashes(train_pool)) & set(row_hashes(test)), "train/test overlap survived filtering"
    info = {
        "protocol": "official_train_to_official_test",
        "official_train_rows": int(len(official_train)),
        "official_train_attack_rate": float(official_train["label"].mean()),
        "train_pool_rows": int(len(train_pool)),
        "train_attack_rate": float(train_pool["label"].mean()),
        "official_test_rows": int(len(official_test)),
        "official_test_attack_rate": float(official_test["label"].mean()),
        "test_rows_final": int(len(test)),
        "test_attack_rate": float(test["label"].mean()),
        "within_test_deduplicated": bool(dedup_test),
    }
    return train_pool, test, info


def make_model(name: str, seed: int, input_dim: int, config: dict):
    base = config.get("baselines", {})
    hyb = config.get("hybrid_classifier", {})
    qc = config.get("quantum_circuit", {})
    if name == "Majority":
        return MajorityClassBaseline(random_state=seed)
    if name == "Stump":
        return SingleSplitBaseline(random_state=seed)
    if name == "RandomForest":
        rf = base.get("random_forest", {})
        return RandomForestBaseline(n_estimators=rf.get("n_estimators", 100), max_depth=rf.get("max_depth", 12), random_state=seed)
    if name == "XGBoost":
        xg = base.get("xgboost", {})
        return XGBoostBaseline(n_estimators=xg.get("n_estimators", 100), max_depth=xg.get("max_depth", 6),
                               learning_rate=xg.get("learning_rate", 0.1), random_state=seed)
    if name == "ClassicalMLP":
        m = base.get("mlp", {})
        return ClassicalMLPBaseline(input_dim=input_dim, hidden_dim=m.get("hidden_dim", 16),
                                    bottleneck_dim=m.get("bottleneck_dim", 4), output_dim=m.get("output_dim", 2),
                                    learning_rate=m.get("learning_rate", 0.005), batch_size=m.get("batch_size", 64),
                                    epochs=m.get("epochs", 40), patience=m.get("patience", 8), random_state=seed)
    if name in ("HybridQNN", "QNN_Bypassed", "QNN_Frozen"):
        layer = {"HybridQNN": "vqc", "QNN_Bypassed": "identity", "QNN_Frozen": "frozen"}[name]
        return HybridQNNClassifier(
            input_dim=input_dim, hidden_dim=config.get("classical_encoder", {}).get("hidden_dim", 16),
            num_qubits=qc.get("num_qubits", 4), num_layers=qc.get("reps", 2), output_dim=2,
            ansatz_type=qc.get("ansatz", "RealAmplitudes"), feature_map=qc.get("feature_map", "AngleEmbedding"),
            feature_map_reps=qc.get("feature_map_reps", 1), quantum_layer=layer, learning_rate=hyb.get("learning_rate", 0.01),
            batch_size=hyb.get("batch_size", 64), epochs=hyb.get("epochs", 20), random_state=seed)
    raise ValueError(f"Unknown model {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official-test", default=os.path.join(REPO_ROOT, "data", "UNSW_NB15_testing-set.csv"))
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--dedup-test", action="store_true", help="also collapse duplicates inside the test set")
    parser.add_argument("--tag", default="", help="suffix for output file names")
    parser.add_argument("--train-source", choices=["sample", "official"], default="sample",
                        help="'official' trains on the official training CSV (primary protocol)")
    parser.add_argument("--official-train", default=os.path.join(REPO_ROOT, "data", "UNSW_NB15_training-set.csv"))
    parser.add_argument("--max-train", type=int, default=0,
                        help="stratified training subsample per seed (0 = all); same for every model")
    args = parser.parse_args()

    if not os.path.exists(args.official_test):
        print(f"SKIP: official testing set not found at {args.official_test}")
        return 0

    config = load_config()
    official = pd.read_csv(args.official_test, encoding="utf-8-sig")
    if args.train_source == "official":
        if not os.path.exists(args.official_train):
            print(f"SKIP: official training set not found at {args.official_train}")
            return 0
        official_train = pd.read_csv(args.official_train, encoding="utf-8-sig")
        train_pool, test, info = build_official_partitions(official_train, official, args.dedup_test)
        if args.max_test and len(test) > args.max_test:
            test, _ = train_test_split(test, train_size=args.max_test, random_state=0, stratify=test["label"])
            test = test.reset_index(drop=True)
            info["test_rows_final"], info["max_test_subsample"] = int(len(test)), int(args.max_test)
    else:
        sample = load_flow_data(config)
        train_pool, test, info = build_partitions(sample, official, args.max_test, args.dedup_test)
    info["max_train_per_seed"] = int(args.max_train)
    print("[Partitions]", json.dumps(info, indent=1))

    models = [m for m in args.models.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    records = []
    records_boot = []
    for seed in seeds:
        pool = train_pool
        if args.max_train and len(pool) > args.max_train:
            pool, _ = train_test_split(pool, train_size=args.max_train, random_state=seed, stratify=pool["label"])
        train_df, val_df = train_test_split(pool, test_size=0.1, random_state=seed, stratify=pool["label"])
        pre = FlowPreprocessor(
            feature_range=tuple(config["dataset"].get("scaler_range", [-np.pi, np.pi])),
            log_transform_skewed=config["dataset"].get("log_transform_skewed", True),
            skew_threshold=config["dataset"].get("skew_threshold", 1.0),
        )
        X_train, y_train = pre.fit_transform(train_df)          # scaler fitted on train only
        X_val, y_val = pre.transform(val_df)
        X_test, y_test = pre.transform(test)
        print(f"\n[seed {seed}] train={len(X_train)} val={len(X_val)} test={len(X_test)} dim={X_train.shape[1]}")

        seed_preds = {}
        for name in models:
            start = time.perf_counter()
            model = make_model(name, seed, X_train.shape[1], config)
            model.fit(X_train, y_train, X_val, y_val)
            y_prob = model.predict_proba(X_test)
            y_pred = np.argmax(y_prob, axis=1) if np.ndim(y_prob) == 2 else (y_prob > 0.5).astype(int)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            elapsed = time.perf_counter() - start
            print(f"  [{name:12s}] acc {metrics['accuracy']:.4f} bal-acc {metrics['balanced_accuracy']:.4f} "
                  f"AUC {metrics['roc_auc']:.4f} MCC {metrics['mcc']:.4f} FPR {metrics['fpr']:.4f} ({elapsed:.1f}s)")
            seed_preds[name] = y_pred
            records.append({"seed": seed, "model": name, **metrics, "seconds": elapsed})

        # Paired bootstrap over TEST FLOWS (accuracy difference vs the classical MLP, same seed).
        # Captures test-sampling uncertainty only; training variance is the across-seed SD.
        if "ClassicalMLP" in seed_preds:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(y_test), size=(1000, len(y_test)), dtype=np.int32)
            ref_correct = (seed_preds["ClassicalMLP"] == y_test).astype(np.float32)
            for name, pred in seed_preds.items():
                if name == "ClassicalMLP":
                    continue
                diff = ((pred == y_test).astype(np.float32) - ref_correct)[idx].mean(axis=1)
                records_boot.append({"seed": seed, "model": name, "vs": "ClassicalMLP",
                                     "acc_diff": float((pred == y_test).mean() - ref_correct.mean()),
                                     "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5))})
            del idx

    frame = pd.DataFrame(records)
    if records_boot:
        pd.DataFrame(records_boot).to_csv(os.path.join(RESULTS_DIR, f"official_heldout_bootstrap{args.tag}.csv"), index=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frame.to_csv(os.path.join(RESULTS_DIR, f"official_heldout_results{args.tag}.csv"), index=False)
    cols = ["accuracy", "balanced_accuracy", "roc_auc", "mcc", "fpr"]
    summary = {"partitions": info, "caveat": "random partition of the same captures; not temporal/cross-network; "
                                             "accuracy/FPR are prevalence-dependent", "models": {}}
    for name, group in frame.groupby("model", sort=False):
        summary["models"][name] = {c: {"mean": float(group[c].mean()), "std": float(group[c].std(ddof=1)) if len(group) > 1 else 0.0}
                                   for c in cols}
    with open(os.path.join(RESULTS_DIR, f"official_heldout_summary{args.tag}.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log_event("VERIFICATION", f"Official held-out evaluation: {len(frame)} runs on {info['test_rows_final']} unseen official-test flows.")
    print("\n" + frame.groupby("model", sort=False)[cols].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
