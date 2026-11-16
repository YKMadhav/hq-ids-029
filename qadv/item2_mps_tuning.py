"""Item 2 follow-up: give the quantum-inspired MPS Born machine a fair tuning budget (bins, bond dim, steps,
input dimension) on UNSW zero-day, same splits as item4. Selection is on the evaluation families (optimistic)."""
import sys, numpy as np, pandas as pd, torch
from qadv.common import load_unsw, seed_everything, fpr_at_tpr
from qadv.item4_zeroday import MPSBorn, bin_features, N_TRAIN, N_BEN_TEST, CAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

CONFIGS = [(d, b, c, st) for d in (4, 8) for b in (8, 16) for c in (6, 16) for st in (250, 600)]
def main(nseeds):
    X, y, cat, _ = load_unsw(); fams = [c for c in np.unique(cat) if c != 'Normal']; rows = []
    for seed in range(nseeds):
        rng = np.random.RandomState(seed); seed_everything(seed)
        ben = np.where(y == 0)[0]; rng.shuffle(ben); tr, bte = ben[:N_TRAIN], ben[N_TRAIN:N_TRAIN + N_BEN_TEST]
        ev_idx, ev_fam = [bte], ['Normal'] * len(bte)
        for f in fams:
            ii = np.where(cat == f)[0]; ii = rng.choice(ii, min(CAP, len(ii)), replace=False); ev_idx.append(ii); ev_fam += [f] * len(ii)
        ev = np.concatenate(ev_idx); ev_fam = np.array(ev_fam); isb = ev_fam == 'Normal'
        sc = MinMaxScaler().fit(X[tr]); Xtr, Xev = sc.transform(X[tr]), np.clip(sc.transform(X[ev]), -1, 2)
        def evaluate(name, s):
            for f in fams:
                sel = isb | (ev_fam == f); yy = (ev_fam[sel] != 'Normal').astype(int)
                rows.append({'seed': seed, 'model': name, 'family': f, 'auc': roc_auc_score(yy, s[sel]), 'fpr_at_95tpr': fpr_at_tpr(yy, s[sel])})
        for d in (4, 8):
            pca = PCA(d, random_state=seed).fit(Xtr); s2 = MinMaxScaler((0, np.pi)).fit(pca.transform(Xtr))
            f8 = lambda a: np.clip(s2.transform(pca.transform(a)), 0, np.pi); Ztr, Zev = f8(Xtr), f8(Xev)
            evaluate(f'kNN(k=10, {d}d PCA)', NearestNeighbors(n_neighbors=10).fit(Ztr).kneighbors(Zev)[0].mean(1))
            for (dd, b, c, st) in [k for k in CONFIGS if k[0] == d]:
                S_tr = torch.tensor(bin_features(Ztr, Ztr, b)); S_ev = torch.tensor(bin_features(Ztr, Zev, b))
                seed_everything(seed); m = MPSBorn(d, b, c, seed); opt = torch.optim.Adam(m.parameters(), 0.02)
                for _ in range(st): opt.zero_grad(); m.nll(S_tr).backward(); opt.step()
                evaluate(f'MPS(d={d},bins={b},chi={c},steps={st})', m.score(S_ev))
        print('seed', seed, flush=True); pd.DataFrame(rows).to_csv('results/qadv/item2_mps_tuning.csv', index=False)
    df = pd.DataFrame(rows); print(df.groupby('model')[['auc', 'fpr_at_95tpr']].mean().round(3).sort_values('auc', ascending=False).to_string())
if __name__ == '__main__': main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
