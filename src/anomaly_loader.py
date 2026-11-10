"""
Project HQ-IDS-029: Zero-Day Anomaly Detection Data Ingestion Pipeline
======================================================================
This module partitions the UNSW-NB15 dataset strictly for unsupervised
anomaly detection and zero-day intrusion benchmarking:

1. Training Set: 100% Benign (Normal) network flows.
   - The preprocessor (scaler & log transforms) is fitted STRICTLY on this set.
   - Zero exposure to attack flows during training (zero target leakage).
2. Validation Set: Benign + 1 Known Attack Category (e.g., 'Exploits').
   - Used for unsupervised reconstruction threshold calibration (e.g., 95th percentile
     of benign reconstruction error or validation F1 optimization).
3. Test Set: Benign + Entirely Unseen Zero-Day Intrusions
   - Contains rare and unseen zero-day attack classes:
     'Worms', 'Backdoor', 'Shellcode', 'Analysis', 'DoS', 'Fuzzers', 'Reconnaissance'.
   - Evaluates whether the quantum/classical autoencoder flags novel signatures
     purely from reconstruction infidelity without prior signature exposure.
"""

import os
import sys
import math
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    load_flow_dataframe,
    FlowPreprocessor,
    FlowDataset,
    METADATA_COLUMNS,
    LABEL_COLUMNS,
)


class ZeroDayFlowDataset(Dataset):
    """
    Dataset wrapper holding feature vectors, binary anomaly labels (0=Benign, 1=Anomaly),
    and original attack category strings for granular evaluation.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, categories: List[str]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.categories = list(categories)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.X[idx], self.y[idx], self.categories[idx]


def get_zeroday_anomaly_dataloaders(
    data_path: str = "./data/real_flows.csv",
    known_attack_cat: str = "Exploits",
    n_train_benign: int = 1100,
    n_val_benign: int = 150,
    n_val_known_attack: int = 150,
    batch_size: int = 64,
    random_seed: int = 42,
    feature_range: Tuple[float, float] = (-math.pi, math.pi),
) -> Tuple[DataLoader, DataLoader, DataLoader, FlowPreprocessor, Dict[str, Any]]:
    """
    Constructs train/val/test data loaders for unsupervised zero-day anomaly detection.
    
    Guarantees:
    - Train split is 100% Benign ('Normal').
    - Scalers & skewness transformations fit strictly on the train split.
    - Test split contains unseen zero-day attacks completely absent from train and val.
    """
    df = load_flow_dataframe(data_path=data_path, require_real=True)

    # Normalize category names
    cat_col = "attack_cat" if "attack_cat" in df.columns else "attack_category"
    df["clean_cat"] = df[cat_col].astype(str).str.strip()
    # Normalize Backdoors -> Backdoor
    df["clean_cat"] = df["clean_cat"].replace({"Backdoors": "Backdoor"})

    # Separate benign and attack pools
    rng = np.random.default_rng(random_seed)
    
    benign_df = df[df["clean_cat"] == "Normal"].copy()
    benign_indices = rng.permutation(len(benign_df))
    benign_shuffled = benign_df.iloc[benign_indices]

    if len(benign_shuffled) < (n_train_benign + n_val_benign + 100):
        raise ValueError(
            f"Not enough benign samples ({len(benign_shuffled)}) for requested split "
            f"({n_train_benign} train + {n_val_benign} val)."
        )

    # 1. Train Benign
    train_benign_df = benign_shuffled.iloc[:n_train_benign].copy()
    train_benign_df["binary_label"] = 0

    # 2. Val Benign
    val_benign_df = benign_shuffled.iloc[n_train_benign : n_train_benign + n_val_benign].copy()
    val_benign_df["binary_label"] = 0

    # 3. Test Benign
    test_benign_df = benign_shuffled.iloc[n_train_benign + n_val_benign :].copy()
    test_benign_df["binary_label"] = 0

    # 4. Val Known Attack
    known_pool = df[df["clean_cat"] == known_attack_cat].copy()
    known_indices = rng.permutation(len(known_pool))
    val_known_df = known_pool.iloc[known_indices[:n_val_known_attack]].copy()
    val_known_df["binary_label"] = 1

    # Val combine & shuffle
    val_combined = pd.concat([val_benign_df, val_known_df], axis=0)
    val_combined = val_combined.iloc[rng.permutation(len(val_combined))].reset_index(drop=True)

    # 5. Test Unseen Zero-Day Attacks
    # Exclude Normal and the known attack category
    zero_day_cats = [
        c for c in df["clean_cat"].unique()
        if c not in ["Normal", known_attack_cat, "Generic"]
    ]
    # We include all rare zero-day threats: Worms, Backdoor, Shellcode, Analysis, DoS, Fuzzers, Reconnaissance
    test_attacks_df = df[df["clean_cat"].isin(zero_day_cats)].copy()
    test_attacks_df["binary_label"] = 1

    # Test combine & shuffle
    test_combined = pd.concat([test_benign_df, test_attacks_df], axis=0)
    test_combined = test_combined.iloc[rng.permutation(len(test_combined))].reset_index(drop=True)

    # Strict feature extraction & preprocessing fit strictly on train_benign_df
    preprocessor = FlowPreprocessor(feature_range=feature_range, multi_class=False)
    X_train, _ = preprocessor.fit_transform(train_benign_df)
    y_train = np.zeros(len(X_train), dtype=np.int64)

    X_val, _ = preprocessor.transform(val_combined)
    y_val = val_combined["binary_label"].values.astype(np.int64)
    val_cats = val_combined["clean_cat"].tolist()

    X_test, _ = preprocessor.transform(test_combined)
    y_test = test_combined["binary_label"].values.astype(np.int64)
    test_cats = test_combined["clean_cat"].tolist()

    # Create PyTorch datasets
    train_ds = ZeroDayFlowDataset(X_train, y_train, train_benign_df["clean_cat"].tolist())
    val_ds = ZeroDayFlowDataset(X_val, y_val, val_cats)
    test_ds = ZeroDayFlowDataset(X_test, y_test, test_cats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    meta = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "val_known_attack": known_attack_cat,
        "zero_day_categories": zero_day_cats,
        "test_cat_counts": pd.Series(test_cats).value_counts().to_dict(),
        "input_dim": X_train.shape[1],
        "feature_names": preprocessor.feature_names_,
    }

    return train_loader, val_loader, test_loader, preprocessor, meta


if __name__ == "__main__":
    print("Testing zero-day anomaly data loader...")
    tr, va, te, prep, meta = get_zeroday_anomaly_dataloaders(random_seed=42)
    print(f"Train size: {meta['n_train']} (100% Benign)")
    print(f"Val size: {meta['n_val']} (Benign + {meta['val_known_attack']})")
    print(f"Test size: {meta['n_test']} (Benign + Unseen Zero-Days: {meta['zero_day_categories']})")
    print(f"Test class distribution: {meta['test_cat_counts']}")
    print("Self-test completed successfully.")
