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


def load_cic(path='data/cicids2017_raw.csv'):
    d = pd.read_csv('/home/claude/hq-ids-029/' + path, low_memory=False)
    d = d.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates().reset_index(drop=True)
    num = [c for c in d.select_dtypes('number').columns if c != 'Label']
    X = np.log1p(d[num].clip(lower=0).to_numpy(dtype=float))
    cat = np.where(d['Label'] == 'BENIGN', 'Normal', d['Label'].astype(str))
    return X, (cat != 'Normal').astype(int), cat, num


def load_pqc(dir_='data/pqc_oav'):
    """CIC-PQC_OAV v1 handshake features. 'family' = testbed scenario; normal scenarios -> 'Normal'.
    Exact duplicate feature vectors removed."""
    base = '/home/claude/hq-ids-029/' + dir_
    f = pd.read_csv(base + '/ml_features_and_labels.csv'); m = pd.read_csv(base + '/scenario_manifest.csv')
    m['ID'] = m['ID'].str.replace('.pcap', '', regex=False)
    d = f.merge(m[['ID', 'scenario_id', 'anomaly_label']], on='ID', how='left')
    assert d['scenario_id'].notna().all()
    feat = [c for c in f.columns if c not in ('label', 'split', 'taxonomy', 'ID')]
    Xf = d[feat].astype(float).fillna(0.0)  # missing handshake fields (e.g. no cert parsed) -> 0
    d = d.loc[~Xf.round(6).duplicated()].reset_index(drop=True); Xf = Xf.loc[Xf.index.isin(Xf.round(6).drop_duplicates().index)].reset_index(drop=True)
    X = np.log1p(Xf.clip(lower=0).to_numpy())
    cat = np.where(d['anomaly_label'] == 'Normal', 'Normal', d['scenario_id'].astype(str))
    return X, (cat != 'Normal').astype(int), cat, feat
