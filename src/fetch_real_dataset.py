"""Fetch and normalize a reproducible UNSW-NB15 binary-flow benchmark sample.

The output deliberately retains the source's 42 predictive fields plus
``attack_cat`` and binary ``label``. FlowPreprocessor performs the training-only
numeric transforms and removes the categorical metadata during each experiment.
"""

from pathlib import Path
import hashlib
import os
from urllib.request import urlopen

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "real_flows.csv"
SAMPLE_SIZE = 5_000
RANDOM_STATE = 42


def download_unsw_nb15() -> pd.DataFrame:
    """Download a user-supplied HTTPS source with mandatory SHA-256 verification.

    We deliberately do not retry with TLS verification disabled or use an
    unaudited personal mirror. Set UNSW_NB15_SOURCE_URL and
    UNSW_NB15_SOURCE_SHA256 from the official dataset release notes.
    """
    source_url = os.environ.get("UNSW_NB15_SOURCE_URL")
    expected_sha256 = os.environ.get("UNSW_NB15_SOURCE_SHA256", "").lower()
    if not source_url or not expected_sha256:
        raise RuntimeError(
            "Set UNSW_NB15_SOURCE_URL and UNSW_NB15_SOURCE_SHA256 to an official, "
            "versioned UNSW-NB15 CSV before fetching."
        )
    if not source_url.startswith("https://") or len(expected_sha256) != 64:
        raise ValueError("Dataset source must be HTTPS and SHA-256 must be 64 hexadecimal characters.")
    with urlopen(source_url, timeout=60) as response:
        payload = response.read()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Dataset SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}.")
    from io import BytesIO
    return pd.read_csv(BytesIO(payload))


def normalize_unsw_nb15(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and ensure the official binary label is an integer."""
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "label" not in frame.columns:
        raise ValueError("UNSW-NB15 source does not contain the required binary 'label' column.")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int64").clip(0, 1)
    if "attack_cat" not in frame.columns:
        frame["attack_cat"] = np.where(frame["label"].eq(0), "Normal", "Attack")
    frame = frame.dropna(subset=["label"]).replace([np.inf, -np.inf], np.nan)
    if frame.shape[1] < 42:
        raise ValueError(f"Expected 40+ flow features plus labels; received {frame.shape[1]} columns.")
    if frame["label"].nunique() != 2:
        raise ValueError("The normalized sample must contain both benign and attack flows.")
    return frame


def make_stratified_sample(frame: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Return a deterministic, class-proportional subset sized for QNN simulation."""
    if len(frame) <= sample_size:
        return frame.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return (frame.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(
                n=round(sample_size * len(group) / len(frame)), random_state=RANDOM_STATE))
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True))


def fetch_real_dataset(output_path: Path = OUTPUT_PATH) -> Path:
    source = normalize_unsw_nb15(download_unsw_nb15())
    normalized = make_stratified_sample(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    print(f"Saved {len(normalized)} UNSW-NB15 flows with {len(normalized.columns) - 2} source features to {output_path}")
    print(f"Binary label distribution: {normalized['label'].value_counts().sort_index().to_dict()}")
    return output_path


if __name__ == "__main__":
    fetch_real_dataset()
