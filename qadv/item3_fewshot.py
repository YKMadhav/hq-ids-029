"""Item 3: few-shot detection of a NEW attack family. Train = 1500 benign + k labelled examples of the
target family (k = 1,5,20,100). Test = held-out target-family flows vs held-out benign. Hyperparameters
are fixed a priori (RBF gamma=1.0, quantum-kernel bw=0.5, C=10), NOT tuned on test data."""
import sys, numpy as np, pandas as pd
from qadv.common import load_unsw, seed_everything, fpr_at_tpr
from qadv.item1_geometric import zz_states, fidelity_kernel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel

KS = (1, 5, 20, 100); NB, NTE = 1500, 1500

def main(reps):
    X, y, cat, _ = load_unsw(); fams = [c for c in np.unique(cat) if c != 'Normal']; rows = []
    for rep in range(reps):
        rng = np.random.RandomState(rep); seed_everything(rep)
        ben = rng.permutation(np.where(y == 0)[0]); btr, bte = ben[:NB], ben[NB:NB + NTE]
        sc = MinMaxScaler().fit(X[btr]); pca = PCA(8, random_state=rep).fit(sc.transform(X[btr]))
        s2 = MinMaxScaler((0, np.pi)).fit(pca.transform(sc.transform(X[btr])))
        f8 = lambda a: np.clip(s2.transform(pca.transform(np.clip(sc.transform(X[a]), -1, 2))), 0, np.pi)
        for fam in fams:
            fi = rng.permutation(np.where(cat == fam)[0]); n_te = min(NTE, max(len(fi) - max(KS), 20))
            fte = fi[max(KS):max(KS) + n_te] if len(fi) > max(KS) + 20 else fi[max(KS) // 2:]
            if len(fte) < 20: continue
            for k in KS:
                ftr = fi[:k]
                if set(ftr) & set(fte): continue
                tr = np.concatenate([btr, ftr]); te = np.concatenate([bte, fte])
                ytr = np.r_[np.zeros(len(btr)), np.ones(len(ftr))]; yte = np.r_[np.zeros(len(bte)), np.ones(len(fte))]
                Ztr, Zte = f8(tr), f8(te); Qtr, Qte = zz_states(Ztr, 2, 0.5), zz_states(Zte, 2, 0.5)
                res = {}
                res['QuantumKernelSVM(8q)'] = SVC(kernel='precomputed', C=10, class_weight='balanced').fit(fidelity_kernel(Qtr), ytr).decision_function(fidelity_kernel(Qte, Qtr))
                res['RBF-SVM(8d,gamma=1)'] = SVC(C=10, gamma=1.0, class_weight='balanced').fit(Ztr, ytr).decision_function(Zte)
                res['RBF-SVM(8d,gamma=0.3)'] = SVC(C=10, gamma=0.3, class_weight='balanced').fit(Ztr, ytr).decision_function(Zte)
                res['LogReg(8d)'] = LogisticRegression(C=10, class_weight='balanced', max_iter=2000).fit(Ztr, ytr).decision_function(Zte)
                res['kNN(k=min(5,k))'] = KNeighborsClassifier(min(5, k), weights='distance').fit(Ztr, ytr).predict_proba(Zte)[:, 1]
                for name, s in res.items():
                    rows.append({'rep': rep, 'family': fam, 'k': k, 'model': name, 'n_test_attack': len(fte),
                                 'auc': roc_auc_score(yte, s), 'fpr_at_95tpr': fpr_at_tpr(yte.astype(int), s)})
        print('rep', rep, flush=True); pd.DataFrame(rows).to_csv('results/qadv/item3_fewshot.csv', index=False)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index='model', columns='k', values='auc').round(3))
    print(df.pivot_table(index='model', columns='k', values='fpr_at_95tpr').round(3))

if __name__ == '__main__': main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
