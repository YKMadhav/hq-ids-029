import numpy as np, pandas as pd
from qadv.common import load_unsw, seed_everything
from qadv.item1_geometric import zz_states, fidelity_kernel, geo_diff, classical_kernels
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
X, y, cat, _ = load_unsw(); rows = []
for nq in (12,):
    for bw in (0.5, 1.0):
        for seed in range(3):
            rng = np.random.RandomState(seed); idx = rng.choice(len(X), 500, replace=False)
            pca = PCA(nq, random_state=seed).fit(X[idx]); Z = MinMaxScaler((0, np.pi)).fit_transform(pca.transform(X[idx]))
            Kq = fidelity_kernel(zz_states(Z, 2, bw)); kc = classical_kernels(Z)
            g = {k: geo_diff(K, Kq) for k, K in kc.items()}; b = min(g, key=g.get)
            rows.append({'qubits': nq, 'bw': bw, 'seed': seed, 'g_C_to_Q_min': g[b], 'best_classical': b, 'sqrtN': 500 ** .5})
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/qadv/item1_scale12.csv', index=False)
