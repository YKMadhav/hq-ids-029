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
