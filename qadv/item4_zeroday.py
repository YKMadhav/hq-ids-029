"""Items 4 + 2: leave-one-family-out zero-day detection. Train on BENIGN ONLY, score unseen
attack families. Metric: AUC and FPR at 95% TPR (SOC metric). Detectors trained once per seed."""
import sys, json, numpy as np, pandas as pd, torch, torch.nn as nn
from qadv.common import load_unsw, load_cic, seed_everything, fpr_at_tpr
from qadv.item1_geometric import zz_states, fidelity_kernel
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

N_TRAIN, N_BEN_TEST, CAP = 3000, 3000, 2000

def _pk(A, B, bw):
    K = np.ones((len(A), len(B)))
    for j in range(A.shape[1]):
        K *= np.cos(bw * (A[:, j][:, None] - B[:, j][None, :]) / 2) ** 2
    return K

class MPSBorn(nn.Module):
    def __init__(self, n_sites, n_bins, chi, seed):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        eye = torch.eye(chi).expand(n_sites, n_bins, chi, chi)
        self.A = nn.Parameter(eye + 0.1 * torch.randn(n_sites, n_bins, chi, chi, generator=g))
        self.l = nn.Parameter(torch.randn(chi, generator=g)); self.r = nn.Parameter(torch.randn(chi, generator=g))
        self.n = n_sites
    def log_psi2(self, S):  # S: (N, n) long
        v = self.l.expand(S.shape[0], -1)
        for i in range(self.n):
            v = torch.einsum('nc,ncd->nd', v, self.A[i][S[:, i]])
        return torch.log((v @ self.r) ** 2 + 1e-30)
    def log_Z(self):
        chi = self.l.shape[0]; w = torch.kron(self.l, self.l)
        for i in range(self.n):
            E = torch.einsum('sab,scd->acbd', self.A[i], self.A[i]).reshape(chi * chi, chi * chi)
            w = w @ E
        return torch.log(w @ torch.kron(self.r, self.r))
    def nll(self, S): return -(self.log_psi2(S) - self.log_Z()).mean()
    def score(self, S):
        with torch.no_grad(): return (-(self.log_psi2(S) - self.log_Z())).numpy()

def bin_features(Ztr, Z, n_bins):
    out = np.zeros(Z.shape, dtype=np.int64)
    for j in range(Z.shape[1]):
        edges = np.quantile(Ztr[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[:, j] = np.searchsorted(edges, Z[:, j])
    return out

def run_seed(seed, X, y, cat):
    rng = np.random.RandomState(seed); seed_everything(seed)
    ben = np.where(y == 0)[0]; rng.shuffle(ben)
    tr, bte = ben[:N_TRAIN], ben[N_TRAIN:N_TRAIN + N_BEN_TEST]
    fams = [c for c in np.unique(cat) if c != 'Normal']
    ev_idx, ev_fam = [bte], ['Normal'] * len(bte)
    for f in fams:
        ii = np.where(cat == f)[0]; ii = rng.choice(ii, min(CAP, len(ii)), replace=False)
        ev_idx.append(ii); ev_fam += [f] * len(ii)
    ev = np.concatenate(ev_idx); ev_fam = np.array(ev_fam)
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xev = sc.transform(X[tr]), np.clip(sc.transform(X[ev]), -1, 2)
    scores = {}
    scores['IsolationForest'] = -IsolationForest(n_estimators=200, random_state=seed).fit(Xtr).score_samples(Xev)
    nn_ = NearestNeighbors(n_neighbors=10).fit(Xtr); scores['kNN(k=10)'] = nn_.kneighbors(Xev)[0].mean(1)
    mu, cov = Xtr.mean(0), np.cov(Xtr.T) + 1e-6 * np.eye(Xtr.shape[1]); P = np.linalg.inv(cov)
    d = Xev - mu; scores['Mahalanobis'] = np.einsum('ij,jk,ik->i', d, P, d)
    scores['OCSVM-RBF(39d)'] = -OneClassSVM(gamma='scale', nu=0.05).fit(Xtr).decision_function(Xev)
    # autoencoder (matched small size)
    ae = nn.Sequential(nn.Linear(Xtr.shape[1], 16), nn.ReLU(), nn.Linear(16, 4), nn.Tanh(), nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, Xtr.shape[1]))
    opt = torch.optim.Adam(ae.parameters(), 3e-3); T = torch.tensor(Xtr, dtype=torch.float32)
    for ep in range(150):
        perm = torch.randperm(len(T))
        for b in range(0, len(T), 128):
            xb = T[perm[b:b + 128]]; opt.zero_grad(); l = ((ae(xb) - xb) ** 2).mean(); l.backward(); opt.step()
    with torch.no_grad(): E = torch.tensor(Xev, dtype=torch.float32); scores['Autoencoder'] = ((ae(E) - E) ** 2).mean(1).numpy()
    # matched 8-d representation: classical vs quantum kernel one-class SVM
    pca = PCA(8, random_state=seed).fit(Xtr); s2 = MinMaxScaler((0, np.pi)).fit(pca.transform(Xtr))
    f8 = lambda a: np.clip(s2.transform(pca.transform(a)), 0, np.pi)
    Ztr, Zev = f8(Xtr), f8(Xev)
    for g in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        scores[f'OCSVM-RBF(8d,gamma={g})'] = -OneClassSVM(gamma=g, nu=0.05).fit(Ztr).decision_function(Zev)
    for bw in (0.25, 0.5, 1.0):
        Qtr, Qev = zz_states(Ztr, 2, bw), zz_states(Zev, 2, bw)
        ocq = OneClassSVM(kernel='precomputed', nu=0.05).fit(fidelity_kernel(Qtr))
        scores[f'OCSVM-QuantumKernel(8q,bw={bw})'] = -ocq.decision_function(fidelity_kernel(Qev, Qtr))
    from sklearn.metrics.pairwise import laplacian_kernel
    for g in (1.0, 3.0, 10.0):
        scores[f'OCSVM-Laplacian(8d,gamma={g})'] = -OneClassSVM(kernel='precomputed', nu=0.05).fit(laplacian_kernel(Ztr, gamma=g)).decision_function(laplacian_kernel(Zev, Ztr, gamma=g))
    for bw in (1.0, 2.0):  # entanglement-free product-state kernel: classically trivial
        pk = lambda A, B, bw=bw: np.prod([np.cos(bw * (A[:, j][:, None] - B[:, j][None, :]) / 2) ** 2 for j in range(A.shape[1])], axis=0) if False else _pk(A, B, bw)
        scores[f'OCSVM-ProductStateKernel(8q,no-entanglement,bw={bw})'] = -OneClassSVM(kernel='precomputed', nu=0.05).fit(pk(Ztr, Ztr)).decision_function(pk(Zev, Ztr))
    scores['kNN(k=10, 8d PCA)'] = NearestNeighbors(n_neighbors=10).fit(Ztr).kneighbors(Zev)[0].mean(1)
    scores['IsolationForest(8d PCA)'] = -IsolationForest(n_estimators=200, random_state=seed).fit(Ztr).score_samples(Zev)
    # quantum-inspired tensor-network Born machine (classical)
    S_tr = torch.tensor(bin_features(Ztr, Ztr, 8)); S_ev = torch.tensor(bin_features(Ztr, Zev, 8))
    for chi in (6, 12):
        seed_everything(seed); m = MPSBorn(8, 8, chi, seed); opt = torch.optim.Adam(m.parameters(), 0.02)
        for it in range(250):
            opt.zero_grad(); l = m.nll(S_tr); l.backward(); opt.step()
        scores[f'MPS-BornMachine(8d,chi={chi})'] = m.score(S_ev)
    rows = []
    isb = ev_fam == 'Normal'
    for name, s in scores.items():
        s = np.nan_to_num(s, nan=0.0, posinf=1e12)
        for f in fams:
            sel = isb | (ev_fam == f); yy = (ev_fam[sel] != 'Normal').astype(int)
            rows.append({'seed': seed, 'detector': name, 'family': f, 'n_family': int((ev_fam == f).sum()),
                         'auc': roc_auc_score(yy, s[sel]), 'fpr_at_95tpr': fpr_at_tpr(yy, s[sel])})
    return rows

def main():
    ds = sys.argv[2] if len(sys.argv) > 2 else 'unsw'
    X, y, cat, _ = (load_cic() if ds == 'cic' else load_unsw()); rows = []
    OUT = 'results/qadv/item4_zeroday.csv' if ds == 'unsw' else f'results/qadv/item4_zeroday_{ds}.csv'
    for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
        rows += run_seed(seed, X, y, cat); print('seed', seed, 'done', flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby('detector')[['auc', 'fpr_at_95tpr']].mean().round(3).sort_values('auc', ascending=False))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(df.pivot_table(index='family', columns='detector', values='auc').round(2).to_string())

if __name__ == '__main__': main()
