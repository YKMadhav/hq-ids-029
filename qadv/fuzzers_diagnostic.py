"""Why do all models miss unseen Fuzzers? (UNSW-NB15 5k sample, de-duplicated, XGBoost, 5 seeds)
(a) Is Fuzzers separable from Normal at all when labelled? (5-fold CV AUC)
(b) Recall on held-out Fuzzers as k labelled Fuzzers are added to training (k=0..all) at fixed FPR~benign test.
(c) Which features distinguish missed Fuzzers from detected attacks / from Normal (standardised mean difference)."""
import os, sys
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
from experiments.common import load_config, load_flow_data, deduplicate_flows, IDENTIFIER_AND_LABEL_COLUMNS
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

raw = deduplicate_flows(load_flow_data(load_config()))
num = [c for c in raw.columns if c not in IDENTIFIER_AND_LABEL_COLUMNS and pd.api.types.is_numeric_dtype(raw[c])]
mk = lambda s: XGBClassifier(n_estimators=100, max_depth=4, n_jobs=2, random_state=s, verbosity=0)
fz, nm = raw[raw.attack_cat == 'Fuzzers'], raw[raw.attack_cat == 'Normal']
print('n fuzzers', len(fz), 'n normal', len(nm))
# (a)
sub = pd.concat([fz, nm]); y = (sub.attack_cat == 'Fuzzers').astype(int).values
p = cross_val_predict(mk(0), sub[num].values, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
print('(a) Fuzzers vs Normal supervised CV AUC = %.3f' % roc_auc_score(y, p))
# (b)
rows = []
for seed in range(5):
    ben_tr, ben_te = train_test_split(nm, test_size=.2, random_state=seed)
    fz_tr, fz_te = train_test_split(fz, test_size=.5, random_state=seed)
    others = raw[(raw.attack_cat != 'Normal') & (raw.attack_cat != 'Fuzzers')]
    for k in (0, 5, 20, 50, len(fz_tr)):
        tr = pd.concat([ben_tr, others, fz_tr.sample(min(k, len(fz_tr)), random_state=seed)])
        m = mk(seed).fit(tr[num].values, (tr.label).values)
        pb = m.predict_proba(ben_te[num].values)[:, 1]; pf = m.predict_proba(fz_te[num].values)[:, 1]
        thr = np.quantile(pb, .90)  # 10% FPR operating point
        rows.append({'seed': seed, 'k_fuzzers': k, 'auc': roc_auc_score(np.r_[np.zeros(len(pb)), np.ones(len(pf))], np.r_[pb, pf]),
                     'recall_at_10pct_fpr': float((pf > thr).mean()), 'recall_at_0.5': float((pf > .5).mean())})
r = pd.DataFrame(rows).groupby('k_fuzzers').mean().drop(columns='seed').round(3); print('(b)\n', r.to_string()); r.to_csv(os.path.join(REPO, 'results/qadv/fuzzers_learning_curve.csv'))
# (c) features: Fuzzers vs Normal and vs Exploits, standardised mean difference on raw-scale medians
sd = raw[num].std().replace(0, 1)
d1 = ((fz[num].median() - nm[num].median()) / sd).abs().sort_values(ascending=False).head(8)
ex = raw[raw.attack_cat == 'Exploits']; d2 = ((fz[num].median() - ex[num].median()) / sd).abs().sort_values(ascending=False).head(8)
print('(c) top |median diff|/sd, Fuzzers vs Normal:', d1.round(2).to_dict()); print('    Fuzzers vs Exploits:', d2.round(2).to_dict())
# fuzzers vs normal on the stump feature
for c in ('sttl', 'ct_state_ttl', 'dur', 'sbytes'):
    print(c, 'Fuzzers median/Normal median/Exploits median:', fz[c].median(), nm[c].median(), ex[c].median())
