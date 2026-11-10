"""Official-partition Phase 4.5 inference loader.

Phase 4.5 evaluates each record in the official UNSW-NB15 testing partition
exactly once. It never resamples, jitters, synthesises, or refits preprocessing
on that partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.biflow_loader import (
    OFFICIAL_TESTING_CSV,
    PHASE4_PROTOCOL,
    SequentialBiflowDataset,
    _normalise_labels,
    _require_official_partition,
    dataframe_id_fingerprint,
    extract_setup_header_sequences,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
SCALER_BUNDLE_PATH = CHECKPOINT_DIR / "phase4_setup_scaler.joblib"


def load_phase4_scaler_bundle(scaler_path: str | Path = SCALER_BUNDLE_PATH):
    """Load and validate a scaler trained by the leakage-safe checkpoint job."""

    path = Path(scaler_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Phase 4 setup scaler bundle: {path}. "
            "Run scripts/save_phase4_checkpoints.py after supplying the official "
            "training and testing partitions."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or set(bundle) != {"scaler", "metadata"}:
        raise ValueError(
            "Rejected scaler artifact: Phase 4.5 accepts only the new "
            "provenance-tagged phase4_setup_scaler.joblib bundle, not legacy scalers."
        )
    scaler, metadata = bundle["scaler"], bundle["metadata"]
    if metadata.get("protocol") != PHASE4_PROTOCOL:
        raise ValueError("Rejected scaler artifact with an incompatible Phase 4 protocol.")
    if metadata.get("input_flat_dim") != 6 or getattr(scaler, "n_features_in_", None) != 6:
        raise ValueError("Rejected scaler artifact with an incompatible feature dimension.")
    return scaler, metadata


def get_scaled_inference_dataloaders(
    data_path: str | Path = OFFICIAL_TESTING_CSV,
    batch_size: int = 256,
    random_seed: int = 42,
    scaler_path: str | Path = SCALER_BUNDLE_PATH,
    n_samples: int | None = None,
):
    """Build a loader over the unmodified official test partition.

    The n_samples argument remains only to fail loudly for old callers: it may
    be omitted or equal the full official partition size, never used to sample
    a subset or create copies.
    """

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    scaler, scaler_metadata = load_phase4_scaler_bundle(scaler_path)
    test_frame = _require_official_partition(data_path, "testing")
    if n_samples is not None and n_samples != len(test_frame):
        raise ValueError(
            f"Phase 4.5 evaluates all {len(test_frame)} official test flows exactly once; "
            f"n_samples={n_samples} is not permitted."
        )

    raw = extract_setup_header_sequences(test_frame)
    labels = _normalise_labels(test_frame["label"])
    transformed = scaler.transform(raw)
    if not np.isfinite(transformed).all():
        raise ValueError("Frozen scaler produced non-finite official-test features.")
    dataset = SequentialBiflowDataset(np.clip(transformed, 0.0, 1.0), labels, k_packets=2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    meta: dict[str, Any] = {
        "protocol": PHASE4_PROTOCOL,
        "n_samples": int(len(test_frame)),
        "unique_source_flows": int(test_frame["id"].nunique()),
        "source_flow_ids": test_frame["id"].to_numpy(copy=True),
        "source_flow_id_sha256": dataframe_id_fingerprint(test_frame),
        "testing_source": str(Path(data_path).expanduser().resolve()),
        "threat_count": int(np.sum(labels == 1)),
        "benign_count": int(np.sum(labels == 0)),
        "threat_ratio": float(np.mean(labels)),
        "batch_size": int(batch_size),
        "scaler_metadata": scaler_metadata,
    }
    return loader, labels, meta


if __name__ == "__main__":
    inference_loader, y_true, metadata = get_scaled_inference_dataloaders(batch_size=512)
    print(
        f"Loaded {metadata['n_samples']} unique official test flows "
        f"({metadata['threat_count']} attacks, {metadata['benign_count']} benign)."
    )
    X_flat, X_seq, y_batch = next(iter(inference_loader))
    print(f"First batch: flat={X_flat.shape}, sequence={X_seq.shape}, labels={y_batch.shape}")
