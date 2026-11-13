"""Leakage-safe Phase 4 setup-header data loader.

UNSW-NB15 CSV records are connection summaries, not packet captures. They
cannot support a claim about arbitrary chronological packet prefixes. This
module therefore implements only a conservative two-observation connection-
setup header proxy: [proto, source TTL, source TCP window] and [proto,
destination TTL, destination TCP window]. A literal k>2 packet-prefix
experiment requires raw PCAP parsing and is refused here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src import determinism as _determinism  # noqa: F401  (pins threads; see src/determinism.py)
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAINING_CSV = REPO_ROOT / "data" / "UNSW_NB15_training-set.csv"
OFFICIAL_TESTING_CSV = REPO_ROOT / "data" / "UNSW_NB15_testing-set.csv"
PHASE4_PROTOCOL = "tcp_setup_header_proxy_v1"
SETUP_PACKET_FEATURES = (("proto", "sttl", "swin"), ("proto", "dttl", "dwin"))
POST_HOC_FORBIDDEN_COLUMNS = frozenset(
    {"sbytes", "dbytes", "spkts", "dpkts", "sloss", "dloss"}
)


class SequentialBiflowDataset(Dataset):
    """Two directional setup-header observations represented as a 2 x 3 sequence."""

    def __init__(self, X_flat: np.ndarray, labels: np.ndarray, k_packets: int = 2):
        if k_packets != 2:
            raise ValueError(
                "CSV-backed Phase 4 supports only k_packets=2. "
                "Use raw PCAP parsing for a literal packet-prefix horizon."
            )
        if X_flat.ndim != 2 or X_flat.shape[1] != 6:
            raise ValueError("Expected six setup-header values per flow.")
        if len(X_flat) != len(labels):
            raise ValueError("Feature and label counts differ.")
        self.X_flat = torch.as_tensor(X_flat, dtype=torch.float32)
        self.X_seq = self.X_flat.reshape(-1, 2, 3)
        self.y = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_flat[idx], self.X_seq[idx], self.y[idx]


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    return out


def _normalise_labels(values: pd.Series) -> np.ndarray:
    if values.dtype == object:
        text = values.astype(str).str.strip().str.lower()
        mapped = text.map(
            {
                "0": 0,
                "1": 1,
                "normal": 0,
                "benign": 0,
                "attack": 1,
                "malicious": 1,
            }
        )
        if mapped.isna().any():
            raise ValueError("Unsupported label values in official UNSW-NB15 partition.")
        return mapped.to_numpy(dtype=np.int64)
    return pd.to_numeric(values, errors="raise").astype(np.int64).to_numpy()


def _require_official_partition(csv_path: str | Path, partition_name: str) -> pd.DataFrame:
    """Load an official partition and reject sample or synthetic substitutes."""

    path = Path(csv_path).expanduser().resolve()
    expected_name = (
        "UNSW_NB15_training-set.csv"
        if partition_name == "training"
        else "UNSW_NB15_testing-set.csv"
    )
    if path.name != expected_name:
        raise ValueError(
            f"Phase 4 requires the official {partition_name} partition named "
            f"{expected_name}; received {path.name!r}."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing official UNSW-NB15 {partition_name} partition: {path}. "
            "Do not substitute data/real_flows.csv or a resampled dataset."
        )

    frame = _normalise_columns(pd.read_csv(path))
    required = {"id", "label", "proto", "sttl", "dttl", "swin", "dwin"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required setup-header columns: {missing}")
    if frame["id"].isna().any() or not frame["id"].is_unique:
        raise ValueError(f"{path} must contain one unique, non-null id per source flow.")
    if frame.empty:
        raise ValueError(f"{path} contains no flows.")
    return frame


def _protocol_code(values: pd.Series) -> np.ndarray:
    """Encode protocol without fitting a data-dependent categorical encoder.

    Values recognised by the static IANA protocol registry use their protocol
    numbers. Other UNSW textual protocol labels map to one fixed -1 bucket,
    rather than deriving an encoder vocabulary from an evaluation partition.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float32)
    protocol_map = {
        "hopopt": 0.0,
        "icmp": 1.0,
        "igmp": 2.0,
        "ggp": 3.0,
        "ipv4": 4.0,
        "tcp": 6.0,
        "egp": 8.0,
        "igp": 9.0,
        "udp": 17.0,
        "ipv6": 41.0,
        "rsvp": 46.0,
        "gre": 47.0,
        "esp": 50.0,
        "ah": 51.0,
        "icmpv6": 58.0,
        "eigrp": 88.0,
        "ospf": 89.0,
        "sctp": 132.0,
        # ARP is an Ethernet EtherType, not IP protocol 0 (HOPOPT).  Keep it
        # distinct from the IANA IP protocol registry rather than creating an
        # ordinal collision with hopopt.
        "arp": -2.0,
        "any": 255.0,
    }
    mapped = values.astype(str).str.strip().str.lower().map(protocol_map).fillna(-1.0)
    return mapped.to_numpy(dtype=np.float32)


def extract_setup_header_sequences(frame: pd.DataFrame) -> np.ndarray:
    """Return the only CSV-based Phase 4 representation permitted by policy.

    This function uses a literal feature allow-list. It does not inspect byte,
    packet-count, or loss totals, so end-of-flow statistics cannot be derived
    into a purported early packet sequence.
    """

    normalised = _normalise_columns(frame)
    required = {feature for observation in SETUP_PACKET_FEATURES for feature in observation}
    missing = sorted(required.difference(normalised.columns))
    if missing:
        raise ValueError(f"Missing temporal allow-list columns: {missing}")
    proto = _protocol_code(normalised["proto"])
    source_ttl = pd.to_numeric(normalised["sttl"], errors="raise").to_numpy(dtype=np.float32)
    source_window = pd.to_numeric(normalised["swin"], errors="raise").to_numpy(dtype=np.float32)
    destination_ttl = pd.to_numeric(normalised["dttl"], errors="raise").to_numpy(dtype=np.float32)
    destination_window = pd.to_numeric(normalised["dwin"], errors="raise").to_numpy(dtype=np.float32)
    sequence = np.column_stack(
        (proto, source_ttl, source_window, proto, destination_ttl, destination_window)
    ).astype(np.float32, copy=False)
    if not np.isfinite(sequence).all():
        raise ValueError("Setup-header allow-list contains non-finite values.")
    return sequence


def extract_biflow_sequences(frame: pd.DataFrame, random_seed: int = 42) -> np.ndarray:
    """Compatibility alias; randomness is intentionally unused."""

    del random_seed
    return extract_setup_header_sequences(frame)


def dataframe_id_fingerprint(frame: pd.DataFrame) -> str:
    ordered_ids = frame["id"].astype(str).sort_values(kind="stable")
    return hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()


def build_scaler_metadata(training_frame: pd.DataFrame, source_path: Path) -> dict[str, Any]:
    return {
        "protocol": PHASE4_PROTOCOL,
        "feature_names": [
            "packet_1_proto",
            "packet_1_sttl",
            "packet_1_swin",
            "packet_2_proto",
            "packet_2_dttl",
            "packet_2_dwin",
        ],
        "input_flat_dim": 6,
        "training_source": str(source_path.resolve()),
        "n_training_flows": int(len(training_frame)),
        "training_id_sha256": dataframe_id_fingerprint(training_frame),
    }


def get_sequential_biflow_dataloaders(
    data_path: str | Path = OFFICIAL_TRAINING_CSV,
    test_data_path: str | Path = OFFICIAL_TESTING_CSV,
    batch_size: int = 128,
    k_packets: int = 2,
    random_seed: int = 42,
):
    """Return train/validation loaders and an untouched official test loader.

    The scaler is fitted only on a seeded split of the official training
    partition. The test partition is neither sampled nor used for fitting.
    """

    if k_packets != 2:
        raise ValueError(
            "CSV-backed Phase 4 supports only k_packets=2; "
            "larger horizons require raw chronological PCAP records."
        )
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    training_path = Path(data_path)
    testing_path = Path(test_data_path)
    training_frame = _require_official_partition(training_path, "training")
    testing_frame = _require_official_partition(testing_path, "testing")
    # UNSW official CSV partitions restart their local id numbering in the
    # testing file. Each partition is required to be internally unique;
    # comparing local row identifiers across files would falsely reject the
    # canonical 175,341/82,332 split.

    X_all = extract_setup_header_sequences(training_frame)
    y_all = _normalise_labels(training_frame["label"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.10, random_state=random_seed, stratify=y_all
    )
    X_test = extract_setup_header_sequences(testing_frame)
    y_test = _normalise_labels(testing_frame["label"])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    train_dataset = SequentialBiflowDataset(np.clip(X_train, 0.0, 1.0), y_train, k_packets)
    val_dataset = SequentialBiflowDataset(np.clip(X_val, 0.0, 1.0), y_val, k_packets)
    test_dataset = SequentialBiflowDataset(np.clip(X_test, 0.0, 1.0), y_test, k_packets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = build_scaler_metadata(training_frame, training_path)
    metadata.update(
        {
            "testing_source": str(testing_path.resolve()),
            "n_testing_flows": int(len(testing_frame)),
            "testing_id_sha256": dataframe_id_fingerprint(testing_frame),
        }
    )
    return train_loader, val_loader, test_loader, scaler, metadata


if __name__ == "__main__":
    print(get_sequential_biflow_dataloaders()[-1])
