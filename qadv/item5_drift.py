"""Item 5: temporal drift. Train benign-only detectors on the FIRST day's benign flows, then score every later day
(benign + attacks of that day). Reports AUC / FPR@95TPR per day and a benign-shift measure (domain-classifier AUC).
Input: data/cic2018_sample.csv from scripts/make_cic2018_sample.py (columns: day, Timestamp, Label, features)."""
import sys, numpy as np, pandas as pd
from qadv.common import seed_everything, fpr_at_tpr
from qadv.item1_geometric import zz_states, fidelity_kernel
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

def main(path='data/cic2018_sample.csv', seeds=5):
    d = pd.read_csv(path, parse_dates=['Timestamp']); feats = [c for c in d.columns if c not in ('day', 'Timestamp', 'Label')]
    order = d.groupby('day')['Timestamp'].min().sort_values().index.tolist(); d['y'] = (d['Label'] != 'Benign').astype(int)
    X = np.log1p(d[feats].clip(lower=0).to_numpy(float)); rows = []
    base = d['day'] == order[0]
    for seed in range(seeds):
        seed_everything(seed); rng = np.random.RandomState(seed)
        btr = np.where(base & (d.y == 0))[0]; btr = rng.permutation(btr); tr, hold = btr[: len(btr) * 2 // 3], btr[len(btr) * 2 // 3:]
        sc = MinMaxScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); pca = PCA(8, random_state=seed).fit(Xtr)
        s2 = MinMaxScaler((0, np.pi)).fit(pca.transform(Xtr)); f8 = lambda a: np.clip(s2.transform(pca.transform(np.clip(sc.transform(a), -1, 2))), 0, np.pi)
        Ztr = f8(X[tr]); Qtr = zz_states(Ztr, 2, 0.5)
        dets = {'IsolationForest': lambda A, Z, Q: -iso.score_samples(sc.transform(A)),
                'kNN(8d PCA)': lambda A, Z, Q: knn.kneighbors(Z)[0].mean(1),
                'OCSVM-RBF(8d,gamma=3)': lambda A, Z, Q: -rbf.decision_function(Z),
                'OCSVM-QuantumKernel(8q,bw=0.5)': lambda A, Z, Q: -oq.decision_function(fidelity_kernel(Q, Qtr))}
        iso = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr); knn = NearestNeighbors(n_neighbors=10).fit(Ztr)
        rbf = OneClassSVM(gamma=3.0, nu=0.05).fit(Ztr); oq = OneClassSVM(kernel='precomputed', nu=0.05).fit(fidelity_kernel(Qtr))
        for day in order:
            sel = np.where(d['day'] == day)[0] if day != order[0] else np.concatenate([hold, np.where((d['day'] == day) & (d.y == 1))[0]])
            yy = d.y.to_numpy()[sel]
            if yy.min() == yy.max(): continue
            Z = f8(X[sel]); Q = zz_states(Z, 2, 0.5)
            for name, fn in dets.items():
                s = fn(X[sel], Z, Q); rows.append({'seed': seed, 'day': day, 'detector': name, 'auc': roc_auc_score(yy, s), 'fpr_at_95tpr': fpr_at_tpr(yy, s)})
            bsel = sel[yy == 0]
            if day != order[0] and len(bsel) > 50:  # benign shift: can a classifier tell day-k benign from base-day benign?
                Xd = np.vstack([Ztr[:len(bsel)], f8(X[bsel])]); yd = np.r_[np.zeros(min(len(Ztr), len(bsel))), np.ones(len(bsel))][: len(Xd)]
                Xd = Xd[: len(yd)]; p = cross_val_predict(RandomForestClassifier(100, random_state=seed, n_jobs=1), Xd, yd, cv=3, method='predict_proba')[:, 1]
                rows.append({'seed': seed, 'day': day, 'detector': 'benign_shift_domain_AUC', 'auc': roc_auc_score(yd, p), 'fpr_at_95tpr': np.nan})
    r = pd.DataFrame(rows); r.to_csv('results/qadv/item5_drift.csv', index=False)
    print(r.pivot_table(index='day', columns='detector', values='auc').reindex(order).round(3).to_string())
if __name__ == '__main__': main(*(sys.argv[1:2] or ['data/cic2018_sample.csv']), seeds=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
