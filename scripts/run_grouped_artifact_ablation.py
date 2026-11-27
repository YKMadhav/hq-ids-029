"""
Grouped-split and testbed-artifact ablation (audit items: grouped protocol,
explicit testbed-artifact ablation).

Two orthogonal factors are crossed on the 5,000-flow UNSW-NB15 sample:

  protocol  : "random"         - the historical stratified 70/10/20 split (5 seeds)
              "family_holdout" - leave-one-attack-family-out: every flow of one
                                 attack family is removed from train/val and
                                 placed in test together with 20% of benign
                                 flows. Measures generalisation to an UNSEEN
                                 attack family (recall) at the benign FPR.
  features  : "full"           - the 39 historical features
              "no_artifacts"   - drops fields that are properties of the UNSW-NB15
                                 testbed rather than of the traffic (TTL values,
                                 TCP window sizes, TCP base sequence numbers,
                                 ct_state_ttl, is_sm_ips_ports).

The artifact list is a STUDY ASSUMPTION (see ARTIFACT_FEATURES) and is not a
validated set. Random re-splits quantify split variability only.

Models: Random Forest, XGBoost, matched-protocol MLP (Control A protocol), and
the 4-qubit Hybrid VQC. Output: results/grouped_artifact_results.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments.common import (  # noqa: E402
    RESULTS_DIR, compute_metrics, load_config, load_flow_data, log_event,
    preprocess_split, stratified_split,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from src.classical_models import (  # noqa: E402
    ClassicalMLPBaseline, RandomForestBaseline, XGBoostBaseline,
)
from src.hybrid_pipeline import HybridQNNClassifier  # noqa: E402

ARTIFACT_FEATURES = ["sttl", "dttl", "swin", "dwin", "stcpb", "dtcpb",
                     "ct_state_ttl", "is_sm_ips_ports"]
HOLDOUT_FAMILIES = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"]
SEEDS = [42, 43, 44, 45, 46]
OUT = os.path.join(RESULTS_DIR, "grouped_artifact_results.csv")  # corrected (de-duplicated) protocol


def make_splits(raw, protocol):
    """Yield (split_id, seed, train_df, val_df, test_df, held_out_family)."""
    if protocol == "random":
        for s in SEEDS:
            tr, va, te = stratified_split(raw, s)
            yield f"seed{s}", s, tr, va, te, "all"
    else:
        for fam in HOLDOUT_FAMILIES:
            seed = 42
            benign = raw[raw.attack_cat == "Normal"]
            ben_train, ben_test = train_test_split(benign, test_size=0.2, random_state=seed)
            held = raw[raw.attack_cat == fam]
            pool = pd.concat([ben_train, raw[(raw.attack_cat != "Normal") & (raw.attack_cat != fam)]])
            tr, va = train_test_split(pool, test_size=1 / 8, random_state=seed, stratify=pool["label"])
            te = pd.concat([ben_test, held])
            yield f"holdout_{fam}", seed, tr, va, te, fam


def build_models(n_feat, cfg, seed):
    m = cfg["baselines"]["matched_mlp"]
    hq = cfg["hybrid_classifier"]
    qc = cfg["quantum_circuit"]
    return {
        "RandomForest": RandomForestBaseline(random_state=seed, n_jobs=2, **{
            k: cfg["baselines"]["random_forest"][k] for k in ("n_estimators", "max_depth")}),
        "XGBoost": XGBoostBaseline(random_state=seed, n_jobs=2, **{
            k: cfg["baselines"]["xgboost"][k] for k in ("n_estimators", "max_depth", "learning_rate")}),
        "MatchedMLP": ClassicalMLPBaseline(
            input_dim=n_feat, hidden_dim=m["hidden_dim"], bottleneck_dim=m["bottleneck_dim"],
            output_dim=2, learning_rate=m["learning_rate"], batch_size=m["batch_size"],
            epochs=m["epochs"], weight_decay=m["weight_decay"], patience=m["patience"],
            scale_latent_by_pi=True, random_state=seed),
        "HybridVQC": HybridQNNClassifier(
            input_dim=n_feat, hidden_dim=cfg["classical_encoder"]["hidden_dim"],
            num_qubits=qc["num_qubits"], num_layers=qc["reps"], output_dim=2,
            ansatz_type=qc["ansatz"], feature_map=qc["feature_map"],
            feature_map_reps=qc["feature_map_reps"], learning_rate=hq["learning_rate"],
            batch_size=hq["batch_size"], epochs=hq["epochs"], random_state=seed),
    }


def main():
    cfg = load_config()
    raw = load_flow_data(cfg)
    # De-duplicate on the NO-ARTIFACT feature view so that neither feature arm has an
    # identical twin across train/test (dropping columns can create new duplicates).
    from experiments.common import IDENTIFIER_AND_LABEL_COLUMNS
    cols = [c for c in raw.columns if c not in IDENTIFIER_AND_LABEL_COLUMNS and c not in ARTIFACT_FEATURES]
    n0 = len(raw)
    raw = raw.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    print(f"De-duplicated {n0} -> {len(raw)} flows (same rows used by both feature arms)")
    only = os.environ.get("ONLY_MODELS")
    only = set(only.split(",")) if only else None
    rows = []
    if os.path.exists(OUT):  # resumable
        rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["protocol"], r["features"], r["split"], r["model"]) for r in rows}

    for protocol in ("random", "family_holdout"):
        for features in ("full", "no_artifacts"):
            for split_id, seed, tr, va, te, fam in make_splits(raw, protocol):
                drop = ARTIFACT_FEATURES if features == "no_artifacts" else []
                tr, va, te = (d.drop(columns=drop) for d in (tr, va, te))
                _, arr = preprocess_split(tr, va, te, cfg)
                Xtr, ytr = arr["train"]; Xva, yva = arr["val"]; Xte, yte = arr["test"]
                for name, model in build_models(Xtr.shape[1], cfg, seed).items():
                    if only and name not in only:
                        continue
                    if (protocol, features, split_id, name) in done:
                        continue
                    t0 = time.time()
                    if name == "HybridVQC":
                        model.fit(Xtr, ytr, Xva, yva, epochs=cfg["hybrid_classifier"]["epochs"], verbose=False)
                    else:
                        model.fit(Xtr, ytr, Xva, yva)
                    prob = model.predict_proba(Xte)
                    pred = np.argmax(prob, axis=1) if prob.ndim == 2 else (prob > .5).astype(int)
                    m = compute_metrics(yte, pred, prob)
                    m["attack_recall"] = m["tp"] / max(m["tp"] + m["fn"], 1)
                    rows.append({"protocol": protocol, "features": features, "split": split_id,
                                 "held_out_family": fam, "seed": seed, "model": name,
                                 "n_features": Xtr.shape[1], "n_train": len(tr), "n_test": len(te),
                                 **m, "train_sec": round(time.time() - t0, 1)})
                    pd.DataFrame(rows).to_csv(OUT, index=False)
                    print(f"{protocol:15s} {features:12s} {split_id:22s} {name:12s} "
                          f"acc={m['accuracy']:.4f} recall={m['attack_recall']:.4f} "
                          f"fpr={m['fpr']:.4f} auc={m['roc_auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    log_event("GROUPED/ARTIFACT ABLATION",
              f"Wrote {len(rows)} rows to results/grouped_artifact_results.csv "
              f"(random vs leave-one-family-out; full vs no_artifacts features).")


if __name__ == "__main__":
    main()
