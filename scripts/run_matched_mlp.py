"""Matched-protocol MLP control: same recipe as the hybrid VQC
(lr=0.01, 20 epochs, wd=0, tanh*pi latent). Writes results/matched_mlp_results.csv."""
import os, sys, time
import numpy as np
import pandas as pd

import torch
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.common import (load_config, load_flow_data, stratified_split,
                                preprocess_split, compute_metrics, RESULTS_DIR)
from src.classical_models import ClassicalMLPBaseline


def main():
    config = load_config()
    seeds = config.get("baselines", {}).get("seeds", [42, 43, 44, 45, 46])
    cfg = config["baselines"]["matched_mlp"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=cfg.get('patience', 20))
    args = parser.parse_args()
    cfg['patience'] = args.patience
    raw_df = load_flow_data(config)
    rows = []
    for seed in seeds:
        train_df, val_df, test_df = stratified_split(raw_df, seed=seed)
        _, arrays = preprocess_split(train_df, val_df, test_df, config)
        X_train, y_train = arrays["train"]
        X_val, y_val = arrays["val"]
        X_test, y_test = arrays["test"]

        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        mlp = ClassicalMLPBaseline(
            input_dim=X_train.shape[1],
            hidden_dim=cfg["hidden_dim"],
            bottleneck_dim=cfg["bottleneck_dim"],
            output_dim=2,
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            weight_decay=cfg["weight_decay"],
            patience=cfg["patience"],
            scale_latent_by_pi=True,
            random_state=seed,
        )
        mlp.fit(X_train, y_train, X_val, y_val)
        y_pred = mlp.predict(X_test)
        y_prob = mlp.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        rows.append({"seed": seed, "model": "Matched_MLP",
                     "family": "Classical Control", **m,
                     "train_time_sec": time.time() - t0})
        print(f"[{seed}] acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
              f"auc={m['roc_auc']:.4f} fpr={m['fpr']:.4f}  "
              f"({time.time() - t0:.1f}s)")
    out = os.path.join(RESULTS_DIR, "matched_mlp_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
