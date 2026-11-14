"""Measure how much the thread count alone changes a seeded HybridQNN run.

Writes results/thread_sensitivity.json. Each setting is run twice in fresh processes to also
confirm run-to-run determinism at a fixed thread count."""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import sys, json, numpy as np, torch
sys.path.insert(0, %(repo)r); sys.path.insert(0, %(repo)r + '/experiments')
n = int(sys.argv[1]); torch.set_num_threads(n)
from common import load_config, load_flow_data, stratified_split, preprocess_split
from src.hybrid_pipeline import HybridQNNClassifier
cfg = load_config(); df = load_flow_data(cfg)
tr, va, te = stratified_split(df, 42)
_, a = preprocess_split(tr, va, te, cfg)
(Xtr, ytr), (Xva, yva), (Xte, yte) = a['train'], a['val'], a['test']
m = HybridQNNClassifier(input_dim=Xtr.shape[1], hidden_dim=16, num_qubits=4, epochs=3, batch_size=64, random_state=42)
m.fit(Xtr, ytr, Xva, yva, epochs=3, verbose=False)
p = m.predict_proba(Xte)[:, 1]
print(json.dumps({'threads': n, 'accuracy': float(((p > .5) == yte).mean()), 'sum_p': float(p.sum())}))
"""


def run(threads: int) -> dict:
    env = dict(os.environ, HQIDS_ALLOW_MULTITHREAD="1")
    out = subprocess.check_output([sys.executable, "-c", CHILD % {"repo": REPO}, str(threads)], env=env, text=True, cwd=REPO,
                                  stderr=subprocess.DEVNULL)
    return json.loads(out.strip().splitlines()[-1])


if __name__ == "__main__":
    runs = {t: [run(t), run(t)] for t in (1, 2)}
    result = {
        "epochs": 3, "seed": 42,
        "runs": {str(t): r for t, r in runs.items()},
        "bit_identical_at_fixed_threads": all(r[0]["sum_p"] == r[1]["sum_p"] for r in runs.values()),
        "accuracy_1_thread": runs[1][0]["accuracy"], "accuracy_2_threads": runs[2][0]["accuracy"],
    }
    with open(os.path.join(REPO, "results", "thread_sensitivity.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
