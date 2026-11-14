"""
Project HQ-IDS-029: Trivial reference detectors
===============================================
Any headline accuracy must be read against these. They share the interface of the
other baselines (fit / predict / predict_proba / measure_latency) so they run
through the identical split, preprocessing and metric code.

* MajorityClassBaseline  - flags everything as the most common training class.
  Its accuracy equals the attack prevalence; its ROC-AUC is 0.5 by construction.
* SingleSplitBaseline    - one depth-1 decision stump: the single best
  feature/threshold chosen on the training partition. If a model barely beats this,
  the model has learned one threshold, not network behaviour.
"""

import time
from typing import Optional

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier


class _TrivialBase:
    def fit(self, X_train, y_train, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def measure_latency(self, x_single, n_runs: int = 1000) -> float:
        x = np.asarray(x_single).reshape(1, -1)
        start = time.perf_counter()
        for _ in range(n_runs):
            self.model.predict_proba(x)
        return (time.perf_counter() - start) / n_runs * 1000.0


class MajorityClassBaseline(_TrivialBase):
    def __init__(self, random_state: int = 42):
        self.model = DummyClassifier(strategy="prior", random_state=random_state)


class SingleSplitBaseline(_TrivialBase):
    def __init__(self, random_state: int = 42):
        self.model = DecisionTreeClassifier(max_depth=1, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        super().fit(X_train, y_train, X_val, y_val)
        tree = self.model.tree_
        self.split_feature_index_ = int(tree.feature[0])
        self.split_threshold_ = float(tree.threshold[0])
        return self


class LookupTableBaseline:
    """Exact-match lookup: majority training label per unique feature vector (rounded to 6
    decimals); unseen vectors get the global training majority. For a low-cardinality feature
    set this is the empirical ceiling of any model that sees only those features."""

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        keys = [tuple(r) for r in np.round(np.asarray(X_train, dtype=np.float64), 6)]
        counts = {}
        for key, label in zip(keys, np.asarray(y_train).astype(int)):
            c = counts.setdefault(key, [0, 0])
            c[label] += 1
        self.table_ = {k: int(v[1] > v[0]) for k, v in counts.items()}
        self.default_ = int(np.mean(y_train) >= 0.5)
        return self

    def predict(self, X):
        return np.array([self.table_.get(tuple(r), self.default_) for r in np.round(np.asarray(X, dtype=np.float64), 6)])
