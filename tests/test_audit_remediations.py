import numpy as np
import pandas as pd
import pytest

from scripts.run_multiclass_benchmark import map_attack_categories
from src.biflow_loader import _protocol_code
from src.error_mitigation import MatrixInversionReadoutMitigator


def test_matrix_inversion_uses_a_real_simplex_projection():
    projected = MatrixInversionReadoutMitigator._project_to_simplex(
        np.array([1.2, -0.1, -0.1])
    )
    np.testing.assert_allclose(projected, [1.0, 0.0, 0.0])
    assert np.isclose(projected.sum(), 1.0)
    assert (projected >= 0).all()


def test_phase4_protocol_codes_do_not_collide_for_arp_and_hopopt():
    encoded = _protocol_code(pd.Series(["arp", "hopopt"]))
    assert encoded[0] != encoded[1]


def test_unknown_multiclass_label_is_not_silently_mapped_to_normal():
    with pytest.raises(ValueError, match="Unknown attack categories"):
        map_attack_categories(pd.Series(["Normal", "FutureAttackFamily"]))


# --- Added in the executed forensic pass -------------------------------------------------

import os

import torch

from experiments.common import deduplicate_flows, stratified_split
from src.data_loader import FlowPreprocessor
from src.trivial_baselines import MajorityClassBaseline, SingleSplitBaseline


def _toy_frame(n=400, n_dup=80, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"id": np.arange(n), "a": rng.normal(size=n), "b": rng.integers(0, 5, n).astype(float)})
    frame["label"] = (frame["a"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    twins = frame.sample(n_dup, random_state=1).copy()
    twins["id"] = np.arange(n, n + n_dup)
    return pd.concat([frame, twins], ignore_index=True)


def test_dedup_removes_train_test_twins():
    frame = _toy_frame()
    train, val, test = stratified_split(frame, seed=42)
    cols = ["a", "b"]
    key = lambda d: set(map(tuple, d[cols].round(12).values))
    assert not key(train) & key(test)
    assert not key(train) & key(val)
    assert not key(val) & key(test)
    # the historical protocol demonstrably leaks twins on the same data
    h_train, _, h_test = stratified_split(frame, seed=42, deduplicate=False)
    assert key(h_train) & key(h_test)
    assert len(deduplicate_flows(frame)) == 400


def test_constant_train_column_maps_to_midpoint_not_saturation():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "const": [7.0] * 4, "label": [0, 1, 0, 1]})
    test = pd.DataFrame({"a": [2.0, 3.0], "const": [9.0, 100.0], "label": [0, 1]})
    pre = FlowPreprocessor(log_transform_skewed=False)
    X_train, _ = pre.fit_transform(train)
    X_test, _ = pre.transform(test)
    assert pre.constant_features_ == ["const"]
    idx = pre.feature_names_.index("const")
    np.testing.assert_allclose(X_train[:, idx], 0.0)
    np.testing.assert_allclose(X_test[:, idx], 0.0)  # was +/-pi (saturated) before the fix


def test_threads_are_pinned_for_portable_results():
    if os.environ.get("HQIDS_ALLOW_MULTITHREAD") == "1":
        pytest.skip("explicit opt-out")
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


def test_trivial_baselines_behave_as_references():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 1] > 0.2).astype(int)
    majority = MajorityClassBaseline().fit(X, y)
    assert set(majority.predict(X)) == {int(np.bincount(y).argmax())}
    stump = SingleSplitBaseline().fit(X, y)
    assert stump.split_feature_index_ == 1
    assert (stump.predict(X) == y).mean() > 0.95


def test_seed_before_construction_makes_initial_weights_reproducible():
    from src.anomaly_models import ClassicalAutoencoder
    from src.determinism import seed_everything

    def weights():
        seed_everything(7)
        m = ClassicalAutoencoder(input_dim=39, hidden_dim=16, latent_dim=4)
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert torch.equal(weights(), weights())


def test_sequential_qnn_batched_circuit_matches_per_sample_reference():
    from src.models.sequential_qnn import SequentialQNNClassifier

    torch.manual_seed(0)
    model = SequentialQNNClassifier(k_packets=2, n_layers=2)
    z = (torch.rand(16, 5) * 2 - 1) * 3.14159
    torch.testing.assert_close(model._quantum_features(z), model._quantum_features_per_sample(z), atol=1e-6, rtol=0)
    torch.testing.assert_close(model._quantum_features(z[:1]), model._quantum_features_per_sample(z[:1]), atol=1e-6, rtol=0)
