"""Item 1: geometric-difference gate (Huang et al. 2021). Can ANY of these quantum kernels
differ from the best classical kernel enough to allow an advantage on this data?"""
import json, itertools, numpy as np
from qadv.common import load_unsw, seed_everything
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

def hadamard(n):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2); M = np.array([[1.0]])
    for _ in range(n): M = np.kron(M, H)
    return M

def zz_states(X, reps=2, bw=1.0):
    """Havlicek-style ZZ feature map statevectors, exact simulation. X in [0,pi]."""
    N, n = X.shape; dim = 2 ** n; H = hadamard(n)
    bits = ((np.arange(dim)[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    Z = 1 - 2 * bits  # (dim,n) eigenvalues
    pairs = list(itertools.combinations(range(n), 2))
    psi = np.tile(H[:, 0].astype(complex), (N, 1))  # H^n|0>
    Xb = bw * X
    for r in range(reps):
        phase = Xb @ Z.T
        for i, j in pairs:
            phase = phase + ((np.pi - Xb[:, i]) * (np.pi - Xb[:, j]))[:, None] * (Z[:, i] * Z[:, j])[None, :]
        psi = psi * np.exp(1j * phase)
        if r < reps - 1: psi = psi @ H.T
    return psi

def fidelity_kernel(A, B=None):
    B = A if B is None else B
    return np.abs(A.conj() @ B.T) ** 2

def sqrtm_psd(K):
    w, V = np.linalg.eigh(K); w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T

def geo_diff(Kc, Kq, lam=1e-3):
    """g(C||Q) = sqrt(||sqrt(Kq)(Kc+lam I)^-1 sqrt(Kq)||_inf), kernels trace-normalised to N."""
    N = len(Kc); Kc = Kc * N / np.trace(Kc); Kq = Kq * N / np.trace(Kq)
    S = sqrtm_psd(Kq)
    M = S @ np.linalg.solve(Kc + lam * N * np.eye(N), S)
    return float(np.sqrt(np.linalg.norm(M, 2)))

def classical_kernels(X):
    ks = {f'rbf_g{g}': rbf_kernel(X, gamma=g) for g in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)}
    ks['linear'] = X @ X.T + 1e-9
    ks['poly3'] = polynomial_kernel(X, degree=3, gamma=1.0 / X.shape[1])
    return ks

def main():
    X, y, cat, num = load_unsw()
    out = {'config': {'n_train': 600, 'lam': 1e-3, 'seeds': 5}, 'runs': []}
    for nq in (4, 8):
        for bw in (0.5, 1.0):
            for seed in range(5):
                rng = np.random.RandomState(seed); seed_everything(seed)
                idx = rng.choice(len(X), 600, replace=False)
                pca = PCA(nq, random_state=seed).fit(X[idx])
                sc = MinMaxScaler((0, np.pi)).fit(pca.transform(X[idx]))
                Z = np.clip(sc.transform(pca.transform(X[idx])), 0, np.pi)
                Kq = fidelity_kernel(zz_states(Z, 2, bw))
                kc = classical_kernels(Z)
                g = {k: geo_diff(K, Kq) for k, K in kc.items()}
                gback = {k: geo_diff(Kq, K) for k, K in kc.items()}
                # scale reference: two classical kernels against each other
                gref = geo_diff(kc['rbf_g0.3'], kc['rbf_g1.0'])
                best = min(g, key=g.get)
                out['runs'].append({'qubits': nq, 'bw': bw, 'seed': seed, 'g_C_to_Q_min': g[best],
                                    'best_classical': best, 'g_Q_to_C_min': min(gback.values()),
                                    'g_classical_ref': gref, 'sqrtN': float(np.sqrt(600)),
                                    'eff_rank_Kq': float(np.exp(-sum(p*np.log(p) for p in np.clip(np.linalg.eigvalsh(Kq/np.trace(Kq)),1e-12,None)))),
                                    })
    import pandas as pd
    df = pd.DataFrame(out['runs']); df.to_csv('results/qadv/item1_geometric.csv', index=False)
    print(df.groupby(['qubits', 'bw'])[['g_C_to_Q_min', 'g_Q_to_C_min', 'g_classical_ref', 'eff_rank_Kq']].mean().round(2))
    print('sqrt(N)=', round(np.sqrt(600), 1))
    # Performance on the same representation (does a different kernel even help?)
    rows = []
    for nq in (4, 8):
        for seed in range(3):
            rng = np.random.RandomState(100 + seed); perm = rng.permutation(len(X))
            tr, te = perm[:1500], perm[1500:4500]
            pca = PCA(nq, random_state=seed).fit(X[tr]); sc = MinMaxScaler((0, np.pi)).fit(pca.transform(X[tr]))
            f = lambda a: np.clip(sc.transform(pca.transform(X[a])), 0, np.pi)
            Ztr, Zte = f(tr), f(te)
            Qtr, Qte = zz_states(Ztr, 2, 0.5), zz_states(Zte, 2, 0.5)
            m = SVC(kernel='precomputed', C=10).fit(fidelity_kernel(Qtr), y[tr])
            s = m.decision_function(fidelity_kernel(Qte, Qtr))
            rows.append({'qubits': nq, 'seed': seed, 'model': 'QuantumKernelSVM', 'auc': roc_auc_score(y[te], s), 'bal_acc': balanced_accuracy_score(y[te], s > 0)})
            for gma in (0.1, 0.3, 1.0):
                m = SVC(kernel='rbf', gamma=gma, C=10).fit(Ztr, y[tr]); s = m.decision_function(Zte)
                rows.append({'qubits': nq, 'seed': seed, 'model': f'RBF-SVM(g={gma})', 'auc': roc_auc_score(y[te], s), 'bal_acc': balanced_accuracy_score(y[te], s > 0)})
    pf = pd.DataFrame(rows); pf.to_csv('results/qadv/item1_performance.csv', index=False)
    print(pf.groupby(['qubits', 'model'])[['auc', 'bal_acc']].mean().round(4))

if __name__ == '__main__':
    main()
