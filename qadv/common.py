import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/hq-ids-029')
from src.determinism import seed_everything  # noqa
from sklearn.preprocessing import MinMaxScaler

DROP = {'id', 'label', 'attack_cat'}

def load_unsw(path='data/UNSW_NB15_training-set.csv', dedup=True):
    d = pd.read_csv('/home/claude/hq-ids-029/' + path)
    if dedup:
        cols = [c for c in d.columns if c not in ('id',)]
        d = d.drop_duplicates(subset=cols).reset_index(drop=True)
    num = [c for c in d.select_dtypes('number').columns if c not in DROP]
    X = np.log1p(d[num].clip(lower=0).to_numpy(dtype=float))
    return X, d['label'].to_numpy(), d['attack_cat'].astype(str).to_numpy(), num

def fpr_at_tpr(y, score, tpr=0.95):
    """FPR when threshold set so that TPR (attack = 1, higher score = more anomalous) >= tpr."""
    pos = np.sort(score[y == 1]); neg = score[y == 0]
    thr = np.quantile(pos, 1 - tpr)
    return float((neg >= thr).mean())
