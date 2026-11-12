"""
Regression and Forensic Leakage Test Suite for HQ-IDS-029.

Verifies:
1. Exact feature dimension equals 39 (no leakage of labels or metadata).
2. Target columns (id, proto, service, state, attack_cat, binary_label, clean_cat) strictly absent.
3. Preprocessing scalers are fitted strictly on the training partition.
4. Anomaly detection loader features match input_dim == 39.
"""

import os
import yaml
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_flow_dataframe, FlowPreprocessor
from experiments.common import stratified_split, preprocess_split
from src.anomaly_loader import get_zeroday_anomaly_dataloaders


@pytest.fixture(scope="module")
def config():
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def raw_df(config):
    data_path = config.get("dataset", {}).get("path", "data/real_flows.csv")
    assert os.path.exists(data_path), f"Dataset file {data_path} not found"
    return load_flow_dataframe(data_path)


def test_raw_data_loading_and_columns(raw_df):
    """Verify raw dataset contains expected records and target column."""
    assert len(raw_df) == 5000, f"Expected 5,000 flows, got {len(raw_df)}"
    assert "binary_label" in raw_df.columns or "label" in raw_df.columns


def test_feature_dimension_is_strictly_39(raw_df, config):
    """Assert preprocessed feature matrix has exactly 39 continuous features."""
    train_df, val_df, test_df = stratified_split(raw_df, seed=42, test_size=0.2, val_size=0.1)
    preprocessor, split_data = preprocess_split(train_df, val_df, test_df, config)

    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    assert X_train.shape[1] == 39, f"X_train feature count must be 39, got {X_train.shape[1]}"
    assert X_val.shape[1] == 39, f"X_val feature count must be 39, got {X_val.shape[1]}"
    assert X_test.shape[1] == 39, f"X_test feature count must be 39, got {X_test.shape[1]}"
    assert len(preprocessor.feature_names_) == 39


def test_target_and_metadata_columns_strictly_excluded(raw_df, config):
    """Ensure no target or high-cardinality nominal columns survive in features."""
    forbidden_cols = {
        "id", "proto", "service", "state", "attack_cat", 
        "binary_label", "clean_cat", "label"
    }
    
    train_df, val_df, test_df = stratified_split(raw_df, seed=42, test_size=0.2, val_size=0.1)
    skew_thresh = config.get("dataset", {}).get("skew_threshold", 1.0)
    preprocessor = FlowPreprocessor(skew_threshold=skew_thresh)
    preprocessor.fit(train_df)

    surviving_forbidden = forbidden_cols.intersection(set(preprocessor.feature_names_))
    assert len(surviving_forbidden) == 0, f"Leaked forbidden columns found: {surviving_forbidden}"


def test_scaler_fitted_strictly_on_train_split(raw_df, config):
    """Verify that test partition statistics do NOT influence the fitted scaler."""
    train_df, val_df, test_df = stratified_split(raw_df, seed=42, test_size=0.2, val_size=0.1)
    skew_thresh = config.get("dataset", {}).get("skew_threshold", 1.0)
    
    # Fit preprocessor on normal train partition
    prep1 = FlowPreprocessor(skew_threshold=skew_thresh)
    prep1.fit(train_df)
    mins1 = prep1.scaler.data_min_.copy()
    maxs1 = prep1.scaler.data_max_.copy()

    # Create an artificially corrupted test partition with massive outliers
    corrupted_test_df = test_df.copy()
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    corrupted_test_df[numeric_cols[0]] = 99999999.0

    # Fit a second preprocessor on the same train partition (test data is only transformed)
    prep2 = FlowPreprocessor(skew_threshold=skew_thresh)
    prep2.fit(train_df)
    _ = prep2.transform(corrupted_test_df)
    mins2 = prep2.scaler.data_min_
    maxs2 = prep2.scaler.data_max_

    # Fitted parameters must be bitwise identical
    np.testing.assert_array_equal(mins1, mins2, err_msg="Scaler min was influenced by test data!")
    np.testing.assert_array_equal(maxs1, maxs2, err_msg="Scaler max was influenced by test data!")


def test_anomaly_loader_feature_dimension_is_39(config):
    """Verify Phase 3 unsupervised zero-day loader strictly strips binary_label."""
    data_path = config.get("dataset", {}).get("path", "data/real_flows.csv")
    train_loader, val_loader, test_loader, prep, meta = get_zeroday_anomaly_dataloaders(
        data_path=data_path, random_seed=42
    )

    X_train = train_loader.dataset.X
    y_train = train_loader.dataset.y
    X_val = val_loader.dataset.X
    X_test = test_loader.dataset.X

    assert X_train.shape[1] == 39, f"Phase 3 X_train features must be 39, got {X_train.shape[1]}"
    assert X_val.shape[1] == 39, f"Phase 3 X_val features must be 39, got {X_val.shape[1]}"
    assert X_test.shape[1] == 39, f"Phase 3 X_test features must be 39, got {X_test.shape[1]}"
    assert len(prep.feature_names_) == 39
    # Train set must be 100% benign
    assert bool((y_train == 0).all()), "Phase 3 train set must contain strictly benign flows (y=0)"
