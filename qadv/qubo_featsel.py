"""Direction 3: QUBO feature selection - QAOA (exact simulation) vs classical solvers. See QUBO_PREREGISTRATION.md."""
import os, sys, time, itertools
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
from experiments.common import load_config, load_flow_data, deduplicate_flows, IDENTIFIER_AND_LABEL_COLUMNS  # noqa

ALPHA, LAM, SHOTS = 1.0, 0.5, 1000
SIZES, N_INST, QAOA_P, RESTARTS = (10, 14, 18), 8, (1, 2, 3), 5
OUT = os.path.join(REPO, "results", "qadv", "qubo_featsel_results.csv")


def build_instance(rel, red, idx, k):
    r, R = rel[idx], red[np.ix_(idx, idx)]
    n = len(idx)
    N = 1 << n
    bits = ((np.arange(N)[:, None] >> np.arange(n)) & 1).astype(np.float32)  # bit i of state s = feature i
    s = bits.sum(1)
    E = -(bits @ r) + ALPHA * 0.5 * np.einsum("si,ij,sj->s", bits, np.triu(R, 1) + np.triu(R, 1).T, bits) + LAM * (s - k) ** 2
    return E.astype(np.float64)


def energy_of(E, state):
    return E[state]


# ---------------- QAOA (exact statevector) ----------------
def apply_mixer(psi, n, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        v = psi.reshape(-1, 2, 1 << q)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = s * a + c * b
    return psi


def qaoa_state(params, E, n, p):
    psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=np.complex128)
    for l in range(p):
        psi *= np.exp(-1j * params[l] * E)
        psi = apply_mixer(psi, n, params[p + l])
    return psi


def run_qaoa(E, n, p, rng):
    Es = (E - E.min()) / (E.max() - E.min() + 1e-12)  # normalised cost for angle scale
    nfev = [0]

    def f(x):
        nfev[0] += 1
        psi = qaoa_state(x, Es, n, p)
        return float(np.real(np.vdot(psi, Es * psi)))
    best = None
    for _ in range(RESTARTS):
        x0 = np.concatenate([rng.uniform(0, 2, p), rng.uniform(0, np.pi / 2, p)])
        res = minimize(f, x0, method="COBYLA", options={"maxiter": 60 * p, "rhobeg": 0.4})
        if best is None or res.fun < best.fun:
            best = res
    prob = np.abs(qaoa_state(best.x, Es, n, p)) ** 2
    prob /= prob.sum()
    shots = rng.choice(1 << n, size=SHOTS, p=prob)
    return shots, nfev[0]


# ---------------- classical solvers (operate on energy table = oracle for the QUBO) ----------------
def flip_energy(E, s, i):
    return E[s ^ (1 << i)]


def run_sa(E, n, budget_s, rng):
    best_s, t0, ev = 0, time.time(), 0
    states = []
    while time.time() - t0 < budget_s:
        s = int(rng.integers(0, 1 << n)); e = E[s]; T0 = 1.0
        steps = 400
        for t in range(steps):
            T = T0 * (1 - t / steps) + 1e-3
            i = int(rng.integers(0, n)); e2 = E[s ^ (1 << i)]; ev += 1
            if e2 < e or rng.random() < np.exp(-(e2 - e) / T):
                s, e = s ^ (1 << i), e2
        states.append(s)
        if time.time() - t0 >= budget_s: break
    return np.array(states), ev


def run_tabu(E, n, budget_s, rng):
    t0, ev, found = time.time(), 0, []
    while time.time() - t0 < budget_s:
        s = int(rng.integers(0, 1 << n)); tabu = {}
        best_s = s
        for it in range(120):
            cands = [(E[s ^ (1 << i)], i) for i in range(n) if tabu.get(i, -1) < it]; ev += len(cands)
            e2, i = min(cands); s ^= (1 << i); tabu[i] = it + 5
            if E[s] < E[best_s]: best_s = s
        found.append(best_s)
    return np.array(found), ev


def run_random(E, n, budget_s, rng):
    # random sampling with the same number of energy evaluations SA would use in the budget (time-matched)
    t0, found = time.time(), []
    while time.time() - t0 < budget_s:
        found.append(int(rng.integers(0, 1 << n)))
    return np.array(found), len(found)


def run_greedy(E, n):
    s = 0
    while True:
        cands = [(E[s ^ (1 << i)], i) for i in range(n)]; e2, i = min(cands)
        if e2 >= E[s]: return np.array([s]), 1
        s ^= (1 << i)


def metrics(E, samples):
    emin, emax = E.min(), E.mean()
    best = E[samples].min()
    return {"p_opt": float((E[samples] <= emin + 1e-9).mean()), "found_opt": bool(best <= emin + 1e-9),
            "approx_ratio": float((emax - best) / (emax - emin + 1e-12)), "mean_E_gap": float(E[samples].mean() - emin)}


def downstream(df_tr, df_te, feats, seeds=(0, 1, 2, 3, 4)):
    from xgboost import XGBClassifier
    accs = []
    for s in seeds:
        m = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=2, random_state=s, verbosity=0)
        m.fit(df_tr[feats].values, df_tr["label"].values); accs.append((m.predict(df_te[feats].values) == df_te["label"].values).mean())
    return float(np.mean(accs))


def main():
    cfg = load_config(); raw = deduplicate_flows(load_flow_data(cfg))
    num = [c for c in raw.columns if c not in IDENTIFIER_AND_LABEL_COLUMNS and pd.api.types.is_numeric_dtype(raw[c])]
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(raw)); ntr = int(.7 * len(raw)); tr, te = raw.iloc[perm[:ntr]], raw.iloc[perm[ntr:]]
    X = tr[num].values.astype(float)
    rel = mutual_info_classif(X, tr["label"].values, random_state=0); rel = rel / rel.max()
    red = np.abs(spearmanr(X).statistic); red = np.nan_to_num(red); np.fill_diagonal(red, 0)
    rows = []
    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT); rows = old.to_dict("records"); done = {(r["n"], r["instance"], r["method"]) for r in rows}
    for n in SIZES:
        for inst in range(N_INST):
            rng = np.random.default_rng(1000 * n + inst)
            idx = np.sort(rng.choice(len(num), n, replace=False)); k = n // 3
            E = build_instance(rel, red, idx, k); opt = int(E.argmin())
            # QAOA first (determines the wall-clock budget for the classical solvers)
            qtime = 0.0; qaoa_out = {}
            for p in QAOA_P:
                if (n, inst, f"QAOA_p{p}") in done: continue
                t0 = time.time(); shots, nfev = run_qaoa(E, n, p, rng); dt = time.time() - t0
                qaoa_out[p] = (shots, dt, nfev)
            budget = max([v[1] for v in qaoa_out.values()] + [0.5]) if qaoa_out else 1.0
            res = {f"QAOA_p{p}": (s, dt, nf) for p, (s, dt, nf) in qaoa_out.items()}
            for name, fn in (("SA", run_sa), ("Tabu", run_tabu), ("Random", run_random)):
                if (n, inst, name) in done: continue
                t0 = time.time(); s, ev = fn(E, n, budget, rng); res[name] = (s, time.time() - t0, ev)
            if (n, inst, "Greedy") not in done:
                t0 = time.time(); s, ev = run_greedy(E, n); res["Greedy"] = (s, time.time() - t0, ev)
            for name, (s, dt, ev) in res.items():
                best_state = int(s[np.argmin(E[s])]); feats = [num[idx[i]] for i in range(n) if (best_state >> i) & 1] or [num[idx[0]]]
                m = metrics(E, s)
                rows.append({"n": n, "instance": inst, "k": k, "method": name, "seconds": round(dt, 3), "evals": ev,
                             "n_selected": len(feats), "downstream_acc": downstream(tr, te, feats, seeds=(0, 1)), **m})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(n, inst, {r["method"]: (round(r["approx_ratio"], 3), r["found_opt"]) for r in rows if r["n"] == n and r["instance"] == inst},
                  f"budget={budget:.1f}s", flush=True)


if __name__ == "__main__":
    main()
