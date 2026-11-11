"""
Project HQ-IDS-029: Network Flow Data Loader & Preprocessing Engine
====================================================================
This module implements:
1. Synthetic realistic network flow generation mimicking real-world NIDS datasets
   (such as CIC-IDS2017 and UNSW-NB15) with realistic heavy-tailed distributions.
2. Ingestion of the real UNSW-NB15 flow sample at ./data/real_flows.csv.
3. Metadata stripping to prevent shortcut learning and data leakage.
4. Log1p transformations on skewed traffic statistics (bytes, durations, packet counts).
5. Continuous feature normalization scaled to [-pi, pi] for quantum angle encoding.
6. PyTorch Dataset & DataLoader interfaces for hybrid quantum-classical training.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader


# Standard 40 flow feature names modeled after CIC-IDS2017 / UNSW-NB15.
# NOTE: the real UNSW-NB15 sample at ./data/real_flows.csv exposes 44 raw columns;
# after metadata stripping (id, proto, service, state, attack_cat) the model input
# dimensionality is inferred at run time rather than hard-coded.
FLOW_FEATURE_COLUMNS = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    "fwd_header_length",
    "bwd_header_length",
    "fwd_packets_per_sec",
    "bwd_packets_per_sec",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "syn_flag_count",
    "rst_flag_count",
    "ack_flag_count",
    "down_up_ratio"
]

# Non-predictive, high-cardinality metadata. Dropping these columns is mandatory:
# retaining them lets the detector memorise per-flow identifiers instead of
# learning intrusion behaviour (i.e. shortcut learning / target leakage).
#   - Synthetic schema (SyntheticFlowGenerator)
SYNTHETIC_METADATA_COLUMNS = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp", "attack_category",
]
#   - Real UNSW-NB15 schema
#     `id` is a unique per-flow record identifier: keeping it trivially separates
#     the classes and drives every metric to 1.0.
#     `proto` / `service` / `state` are nominal protocol metadata.
#     `attack_cat` is the ground-truth attack family, i.e. the label itself.
UNSW_METADATA_COLUMNS = ["id", "proto", "service", "state", "attack_cat", "clean_cat"]

METADATA_COLUMNS = SYNTHETIC_METADATA_COLUMNS + UNSW_METADATA_COLUMNS

# Ground-truth / supervision columns that must never enter the feature matrix.
LABEL_COLUMNS = ["label", "attack_category", "attack_cat", "binary_label", "clean_cat"]

# Canonical 10-class UNSW-NB15 attack category taxonomy
ATTACK_CAT_MAPPING = {
    "Normal": 0,
    "Fuzzers": 1,
    "Analysis": 2,
    "Backdoors": 3,
    "Backdoor": 3,  # Normalize singular/plural
    "DoS": 4,
    "Exploits": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9,
}
ATTACK_CAT_NAMES = [
    "Normal",
    "Fuzzers",
    "Analysis",
    "Backdoors",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]


class SyntheticFlowGenerator:
    """
    Generates synthetic network flow records exhibiting realistic multi-modal
    statistical properties (log-normal byte distributions, Poisson arrivals,
    and distinct benign vs. attack behavioral signatures).
    """

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

    def generate(self, n_samples: int = 5000, attack_ratio: float = 0.3) -> pd.DataFrame:
        """
        Synthesize n_samples of network flows with the given attack proportion.
        Attack classes include 'DDoS', 'PortScan', and 'BruteForce'.
        """
        n_attack = int(n_samples * attack_ratio)
        n_benign = n_samples - n_attack

        # Generate Benign samples
        benign_data = self._generate_benign_flows(n_benign)

        # Generate Attack samples (split between DDoS, PortScan, BruteForce)
        n_ddos = n_attack // 2
        n_portscan = n_attack // 4
        n_bruteforce = n_attack - n_ddos - n_portscan

        ddos_data = self._generate_ddos_flows(n_ddos)
        portscan_data = self._generate_portscan_flows(n_portscan)
        bruteforce_data = self._generate_bruteforce_flows(n_bruteforce)

        # Combine into a single DataFrame
        df = pd.concat([benign_data, ddos_data, portscan_data, bruteforce_data], ignore_index=True)
        # Shuffle records
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        return df

    def _generate_benign_flows(self, n: int) -> pd.DataFrame:
        """Generates standard benign web/DNS/streaming traffic patterns."""
        rng = self.rng
        data: Dict[str, np.ndarray] = {}

        # Metadata
        data["src_ip"] = [f"192.168.1.{rng.integers(10, 250)}" for _ in range(n)]
        data["dst_ip"] = [f"10.0.0.{rng.integers(1, 50)}" for _ in range(n)]
        data["src_port"] = rng.integers(1024, 65535, size=n)
        data["dst_port"] = rng.choice([80, 443, 53, 8080, 22], size=n, p=[0.4, 0.4, 0.15, 0.03, 0.02])
        data["protocol"] = rng.choice([6, 17], size=n, p=[0.85, 0.15])  # TCP / UDP
        data["timestamp"] = np.linspace(1700000000, 1700003600, n)

        # Flow duration (milliseconds, log-normal: typical 50ms - 10s)
        duration = np.exp(rng.normal(5.0, 1.2, size=n))
        data["flow_duration"] = np.clip(duration, 1.0, 120000.0)

        # Forward/Backward packets (typical 5 to 50 packets)
        data["total_fwd_packets"] = np.clip(rng.poisson(12, size=n), 1, 500)
        data["total_bwd_packets"] = np.clip(rng.poisson(15, size=n), 0, 500)

        # Packet lengths and bytes
        fwd_mean_len = rng.uniform(40.0, 600.0, size=n)
        bwd_mean_len = rng.uniform(200.0, 1200.0, size=n)
        data["fwd_packet_length_min"] = rng.uniform(40.0, 64.0, size=n)
        data["fwd_packet_length_max"] = fwd_mean_len * rng.uniform(1.2, 2.5, size=n)
        data["fwd_packet_length_mean"] = fwd_mean_len
        data["fwd_packet_length_std"] = fwd_mean_len * rng.uniform(0.1, 0.4, size=n)

        data["bwd_packet_length_min"] = rng.uniform(40.0, 64.0, size=n)
        data["bwd_packet_length_max"] = bwd_mean_len * rng.uniform(1.2, 2.0, size=n)
        data["bwd_packet_length_mean"] = bwd_mean_len
        data["bwd_packet_length_std"] = bwd_mean_len * rng.uniform(0.1, 0.5, size=n)

        data["total_fwd_bytes"] = data["total_fwd_packets"] * data["fwd_packet_length_mean"]
        data["total_bwd_bytes"] = data["total_bwd_packets"] * data["bwd_packet_length_mean"]

        # Rates
        sec_duration = np.maximum(data["flow_duration"] / 1000.0, 0.001)
        data["flow_bytes_per_sec"] = (data["total_fwd_bytes"] + data["total_bwd_bytes"]) / sec_duration
        data["flow_packets_per_sec"] = (data["total_fwd_packets"] + data["total_bwd_packets"]) / sec_duration
        data["fwd_packets_per_sec"] = data["total_fwd_packets"] / sec_duration
        data["bwd_packets_per_sec"] = data["total_bwd_packets"] / sec_duration

        # Inter-arrival times (IAT in ms)
        data["flow_iat_mean"] = np.clip(rng.exponential(20.0, size=n), 0.1, 2000.0)
        data["flow_iat_std"] = data["flow_iat_mean"] * rng.uniform(0.3, 0.9, size=n)
        data["flow_iat_max"] = data["flow_iat_mean"] * rng.uniform(2.0, 5.0, size=n)
        data["flow_iat_min"] = np.clip(data["flow_iat_mean"] * rng.uniform(0.01, 0.1, size=n), 0.001, 10.0)

        data["fwd_iat_total"] = data["total_fwd_packets"] * data["flow_iat_mean"]
        data["fwd_iat_mean"] = data["flow_iat_mean"] * 1.1
        data["fwd_iat_std"] = data["flow_iat_std"] * 1.1
        data["bwd_iat_total"] = data["total_bwd_packets"] * data["flow_iat_mean"]
        data["bwd_iat_mean"] = data["flow_iat_mean"] * 0.9
        data["bwd_iat_std"] = data["flow_iat_std"] * 0.9

        # Flags and headers
        data["fwd_psh_flags"] = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        data["bwd_psh_flags"] = rng.choice([0, 1], size=n, p=[0.6, 0.4])
        data["fwd_urg_flags"] = np.zeros(n, dtype=int)
        data["bwd_urg_flags"] = np.zeros(n, dtype=int)
        data["fwd_header_length"] = data["total_fwd_packets"] * 20
        data["bwd_header_length"] = data["total_bwd_packets"] * 20
        data["syn_flag_count"] = rng.choice([1, 2], size=n, p=[0.9, 0.1])
        data["rst_flag_count"] = rng.choice([0, 1], size=n, p=[0.95, 0.05])
        data["ack_flag_count"] = data["total_bwd_packets"]
        data["down_up_ratio"] = np.clip(data["total_bwd_packets"] / np.maximum(data["total_fwd_packets"], 1), 0.1, 10.0)

        total_pkts = data["total_fwd_packets"] + data["total_bwd_packets"]
        total_bytes = data["total_fwd_bytes"] + data["total_bwd_bytes"]
        data["packet_length_mean"] = total_bytes / total_pkts
        data["packet_length_std"] = rng.uniform(50.0, 300.0, size=n)
        data["packet_length_variance"] = data["packet_length_std"] ** 2

        df = pd.DataFrame(data)
        df["attack_category"] = "Benign"
        df["label"] = 0
        return df

    def _generate_ddos_flows(self, n: int) -> pd.DataFrame:
        """Generates volumetric DDoS / SYN flood signatures."""
        rng = self.rng
        data: Dict[str, np.ndarray] = {}

        data["src_ip"] = [f"172.16.{rng.integers(1, 255)}.{rng.integers(1, 255)}" for _ in range(n)]
        data["dst_ip"] = ["10.0.0.1" for _ in range(n)]  # Single target victim
        data["src_port"] = rng.integers(1024, 65535, size=n)
        data["dst_port"] = rng.choice([80, 443], size=n)
        data["protocol"] = np.full(n, 6)  # TCP
        data["timestamp"] = np.linspace(1700001000, 1700001500, n)

        # Very high packet rate, small duration, asymmetric packets
        data["flow_duration"] = np.clip(rng.exponential(250.0, size=n), 1.0, 5000.0)
        data["total_fwd_packets"] = np.clip(rng.poisson(80, size=n), 10, 2000)
        data["total_bwd_packets"] = np.clip(rng.poisson(2, size=n), 0, 5)  # Few or no ACKs from overwhelmed target

        # Small payload (SYN packets often have zero payload)
        data["fwd_packet_length_min"] = np.full(n, 40.0)
        data["fwd_packet_length_max"] = np.full(n, 60.0)
        data["fwd_packet_length_mean"] = rng.uniform(40.0, 60.0, size=n)
        data["fwd_packet_length_std"] = rng.uniform(0.0, 5.0, size=n)

        data["bwd_packet_length_min"] = np.full(n, 0.0)
        data["bwd_packet_length_max"] = np.full(n, 54.0)
        data["bwd_packet_length_mean"] = rng.uniform(0.0, 54.0, size=n)
        data["bwd_packet_length_std"] = rng.uniform(0.0, 10.0, size=n)

        data["total_fwd_bytes"] = data["total_fwd_packets"] * data["fwd_packet_length_mean"]
        data["total_bwd_bytes"] = data["total_bwd_packets"] * data["bwd_packet_length_mean"]

        sec_duration = np.maximum(data["flow_duration"] / 1000.0, 0.001)
        data["flow_bytes_per_sec"] = (data["total_fwd_bytes"] + data["total_bwd_bytes"]) / sec_duration
        data["flow_packets_per_sec"] = (data["total_fwd_packets"] + data["total_bwd_packets"]) / sec_duration
        data["fwd_packets_per_sec"] = data["total_fwd_packets"] / sec_duration
        data["bwd_packets_per_sec"] = data["total_bwd_packets"] / sec_duration

        # Extremely low IAT (flooding)
        data["flow_iat_mean"] = np.clip(rng.exponential(1.5, size=n), 0.01, 15.0)
        data["flow_iat_std"] = data["flow_iat_mean"] * rng.uniform(0.1, 0.3, size=n)
        data["flow_iat_max"] = data["flow_iat_mean"] * 3.0
        data["flow_iat_min"] = np.full(n, 0.001)

        data["fwd_iat_total"] = data["total_fwd_packets"] * data["flow_iat_mean"]
        data["fwd_iat_mean"] = data["flow_iat_mean"]
        data["fwd_iat_std"] = data["flow_iat_std"]
        data["bwd_iat_total"] = data["total_bwd_packets"] * 10.0
        data["bwd_iat_mean"] = np.full(n, 5.0)
        data["bwd_iat_std"] = np.full(n, 2.0)

        data["fwd_psh_flags"] = np.zeros(n, dtype=int)
        data["bwd_psh_flags"] = np.zeros(n, dtype=int)
        data["fwd_urg_flags"] = np.zeros(n, dtype=int)
        data["bwd_urg_flags"] = np.zeros(n, dtype=int)
        data["fwd_header_length"] = data["total_fwd_packets"] * 20
        data["bwd_header_length"] = data["total_bwd_packets"] * 20
        data["syn_flag_count"] = data["total_fwd_packets"]  # SYN flood
        data["rst_flag_count"] = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        data["ack_flag_count"] = data["total_bwd_packets"]
        data["down_up_ratio"] = np.clip(data["total_bwd_packets"] / np.maximum(data["total_fwd_packets"], 1), 0.0, 0.1)

        total_pkts = data["total_fwd_packets"] + data["total_bwd_packets"]
        total_bytes = data["total_fwd_bytes"] + data["total_bwd_bytes"]
        data["packet_length_mean"] = total_bytes / total_pkts
        data["packet_length_std"] = rng.uniform(0.0, 10.0, size=n)
        data["packet_length_variance"] = data["packet_length_std"] ** 2

        df = pd.DataFrame(data)
        df["attack_category"] = "DDoS"
        df["label"] = 1
        return df

    def _generate_portscan_flows(self, n: int) -> pd.DataFrame:
        """Generates reconnaissance / stealth port scan flows."""
        rng = self.rng
        data: Dict[str, np.ndarray] = {}

        data["src_ip"] = ["192.168.1.105" for _ in range(n)]  # Single probing scanner
        data["dst_ip"] = ["10.0.0.5" for _ in range(n)]
        data["src_port"] = rng.integers(40000, 60000, size=n)
        data["dst_port"] = rng.integers(1, 1024, size=n)  # Sweeping destination ports
        data["protocol"] = np.full(n, 6)
        data["timestamp"] = np.linspace(1700002000, 1700002500, n)

        # Fast 1-2 packet probes, zero payload, RST return
        data["flow_duration"] = np.clip(rng.exponential(20.0, size=n), 0.1, 100.0)
        data["total_fwd_packets"] = np.full(n, 1)  # Single SYN probe
        data["total_bwd_packets"] = rng.choice([0, 1], size=n, p=[0.3, 0.7])  # RST/ACK or dropped

        data["fwd_packet_length_min"] = np.full(n, 40.0)
        data["fwd_packet_length_max"] = np.full(n, 40.0)
        data["fwd_packet_length_mean"] = np.full(n, 40.0)
        data["fwd_packet_length_std"] = np.full(n, 0.0)

        data["bwd_packet_length_min"] = np.full(n, 40.0)
        data["bwd_packet_length_max"] = np.full(n, 40.0)
        data["bwd_packet_length_mean"] = np.full(n, 40.0)
        data["bwd_packet_length_std"] = np.full(n, 0.0)

        data["total_fwd_bytes"] = np.full(n, 40.0)
        data["total_bwd_bytes"] = data["total_bwd_packets"] * 40.0

        sec_duration = np.maximum(data["flow_duration"] / 1000.0, 0.0001)
        data["flow_bytes_per_sec"] = (data["total_fwd_bytes"] + data["total_bwd_bytes"]) / sec_duration
        data["flow_packets_per_sec"] = (data["total_fwd_packets"] + data["total_bwd_packets"]) / sec_duration
        data["fwd_packets_per_sec"] = data["total_fwd_packets"] / sec_duration
        data["bwd_packets_per_sec"] = data["total_bwd_packets"] / sec_duration

        data["flow_iat_mean"] = np.clip(rng.exponential(5.0, size=n), 0.01, 50.0)
        data["flow_iat_std"] = np.zeros(n)
        data["flow_iat_max"] = data["flow_iat_mean"]
        data["flow_iat_min"] = data["flow_iat_mean"]

        data["fwd_iat_total"] = np.zeros(n)
        data["fwd_iat_mean"] = np.zeros(n)
        data["fwd_iat_std"] = np.zeros(n)
        data["bwd_iat_total"] = np.zeros(n)
        data["bwd_iat_mean"] = np.zeros(n)
        data["bwd_iat_std"] = np.zeros(n)

        data["fwd_psh_flags"] = np.zeros(n, dtype=int)
        data["bwd_psh_flags"] = np.zeros(n, dtype=int)
        data["fwd_urg_flags"] = np.zeros(n, dtype=int)
        data["bwd_urg_flags"] = np.zeros(n, dtype=int)
        data["fwd_header_length"] = np.full(n, 20)
        data["bwd_header_length"] = data["total_bwd_packets"] * 20
        data["syn_flag_count"] = np.full(n, 1)
        data["rst_flag_count"] = data["total_bwd_packets"]  # Closed port reset
        data["ack_flag_count"] = np.zeros(n, dtype=int)
        data["down_up_ratio"] = data["total_bwd_packets"].astype(float)

        data["packet_length_mean"] = np.full(n, 40.0)
        data["packet_length_std"] = np.zeros(n)
        data["packet_length_variance"] = np.zeros(n)

        df = pd.DataFrame(data)
        df["attack_category"] = "PortScan"
        df["label"] = 1
        return df

    def _generate_bruteforce_flows(self, n: int) -> pd.DataFrame:
        """Generates repeated SSH/FTP authentication brute-force flows."""
        rng = self.rng
        data: Dict[str, np.ndarray] = {}

        data["src_ip"] = ["192.168.1.200" for _ in range(n)]
        data["dst_ip"] = ["10.0.0.22" for _ in range(n)]
        data["src_port"] = rng.integers(30000, 60000, size=n)
        data["dst_port"] = rng.choice([22, 21], size=n, p=[0.8, 0.2])  # SSH / FTP
        data["protocol"] = np.full(n, 6)
        data["timestamp"] = np.linspace(1700003000, 1700003500, n)

        # Moderate duration, fixed exchange cadence
        data["flow_duration"] = np.clip(rng.normal(1200.0, 200.0, size=n), 200.0, 3000.0)
        data["total_fwd_packets"] = np.clip(rng.poisson(18, size=n), 8, 40)
        data["total_bwd_packets"] = np.clip(rng.poisson(14, size=n), 6, 35)

        data["fwd_packet_length_min"] = np.full(n, 40.0)
        data["fwd_packet_length_max"] = rng.uniform(150.0, 350.0, size=n)
        data["fwd_packet_length_mean"] = rng.uniform(80.0, 160.0, size=n)
        data["fwd_packet_length_std"] = rng.uniform(20.0, 60.0, size=n)

        data["bwd_packet_length_min"] = np.full(n, 40.0)
        data["bwd_packet_length_max"] = rng.uniform(100.0, 250.0, size=n)
        data["bwd_packet_length_mean"] = rng.uniform(60.0, 120.0, size=n)
        data["bwd_packet_length_std"] = rng.uniform(15.0, 50.0, size=n)

        data["total_fwd_bytes"] = data["total_fwd_packets"] * data["fwd_packet_length_mean"]
        data["total_bwd_bytes"] = data["total_bwd_packets"] * data["bwd_packet_length_mean"]

        sec_duration = np.maximum(data["flow_duration"] / 1000.0, 0.001)
        data["flow_bytes_per_sec"] = (data["total_fwd_bytes"] + data["total_bwd_bytes"]) / sec_duration
        data["flow_packets_per_sec"] = (data["total_fwd_packets"] + data["total_bwd_packets"]) / sec_duration
        data["fwd_packets_per_sec"] = data["total_fwd_packets"] / sec_duration
        data["bwd_packets_per_sec"] = data["total_bwd_packets"] / sec_duration

        data["flow_iat_mean"] = np.clip(rng.normal(80.0, 15.0, size=n), 10.0, 300.0)
        data["flow_iat_std"] = data["flow_iat_mean"] * 0.4
        data["flow_iat_max"] = data["flow_iat_mean"] * 2.5
        data["flow_iat_min"] = np.clip(data["flow_iat_mean"] * 0.1, 0.1, 10.0)

        data["fwd_iat_total"] = data["total_fwd_packets"] * data["flow_iat_mean"]
        data["fwd_iat_mean"] = data["flow_iat_mean"]
        data["fwd_iat_std"] = data["flow_iat_std"]
        data["bwd_iat_total"] = data["total_bwd_packets"] * data["flow_iat_mean"]
        data["bwd_iat_mean"] = data["flow_iat_mean"]
        data["bwd_iat_std"] = data["flow_iat_std"]

        data["fwd_psh_flags"] = rng.choice([1, 2, 3], size=n)
        data["bwd_psh_flags"] = rng.choice([1, 2, 3], size=n)
        data["fwd_urg_flags"] = np.zeros(n, dtype=int)
        data["bwd_urg_flags"] = np.zeros(n, dtype=int)
        data["fwd_header_length"] = data["total_fwd_packets"] * 20
        data["bwd_header_length"] = data["total_bwd_packets"] * 20
        data["syn_flag_count"] = np.ones(n, dtype=int)
        data["rst_flag_count"] = rng.choice([0, 1], size=n, p=[0.2, 0.8])  # Failed auth closes connection
        data["ack_flag_count"] = data["total_bwd_packets"]
        data["down_up_ratio"] = np.clip(data["total_bwd_packets"] / np.maximum(data["total_fwd_packets"], 1), 0.5, 2.0)

        total_pkts = data["total_fwd_packets"] + data["total_bwd_packets"]
        total_bytes = data["total_fwd_bytes"] + data["total_bwd_bytes"]
        data["packet_length_mean"] = total_bytes / total_pkts
        data["packet_length_std"] = rng.uniform(30.0, 70.0, size=n)
        data["packet_length_variance"] = data["packet_length_std"] ** 2

        df = pd.DataFrame(data)
        df["attack_category"] = "BruteForce"
        df["label"] = 1
        return df


class FlowPreprocessor:
    """
    Cleans, transforms, and normalizes flow data.
    - Eliminates metadata (flow ids, protocol labels, timestamps, attack family)
      to eliminate shortcut learning and target leakage.
    - Applies log1p to heavily skewed positive continuous metrics.
    - Normalizes continuous features to [-pi, pi] for quantum angle rotation.
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (-math.pi, math.pi),
        log_transform_skewed: bool = True,
        skew_threshold: float = 1.0,
        multi_class: bool = False,
    ):
        self.feature_range = feature_range
        self.log_transform_skewed = log_transform_skewed
        self.skew_threshold = skew_threshold
        self.multi_class = multi_class
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.skewed_features_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "FlowPreprocessor":
        """Fits skewness detectors and MinMaxScaler on training data only."""
        # Filter out metadata and label columns
        features_df = self._extract_features(df)
        self.feature_names_ = list(features_df.columns)

        # Detect highly skewed features
        if self.log_transform_skewed:
            self.skewed_features_ = []
            for col in self.feature_names_:
                series = features_df[col].dropna()
                # Apply skew check only on strictly non-negative features
                if (series >= 0).all() and series.skew() > self.skew_threshold:
                    self.skewed_features_.append(col)

        # Copy and transform before fitting scaler
        transformed_df = features_df.copy()
        if self.log_transform_skewed:
            for col in self.skewed_features_:
                transformed_df[col] = np.log1p(np.maximum(transformed_df[col].values, 0.0))

        # Handle any residual NaNs or Infs
        transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Fit MinMaxScaler
        self.scaler.fit(transformed_df.values)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms features into scaled numpy array bounded in [-pi, pi],
        and returns (X, y).
        If self.multi_class is True, y is 10-class ordinal index from ATTACK_CAT_MAPPING.
        """
        features_df = self._extract_features(df)
        if self.multi_class:
            if "attack_cat" in df.columns:
                labels = df["attack_cat"].map(lambda c: ATTACK_CAT_MAPPING.get(str(c).strip(), 0)).values.astype(np.int64)
            elif "attack_category" in df.columns:
                labels = df["attack_category"].map(lambda c: ATTACK_CAT_MAPPING.get(str(c).strip(), 0)).values.astype(np.int64)
            elif "label" in df.columns:
                labels = df["label"].values.astype(np.int64)
            else:
                labels = np.zeros(len(df), dtype=np.int64)
        else:
            labels = df["label"].values.astype(np.int64) if "label" in df.columns else np.zeros(len(df), dtype=np.int64)

        transformed_df = features_df.copy()
        if self.log_transform_skewed:
            for col in self.skewed_features_:
                transformed_df[col] = np.log1p(np.maximum(transformed_df[col].values, 0.0))

        transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = self.scaler.transform(transformed_df.values)
        # Ensure values are strictly clamped to range
        X_scaled = np.clip(X_scaled, self.feature_range[0], self.feature_range[1])
        return X_scaled, labels

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strips metadata and label columns, returning the numeric feature matrix only."""
        drop_cols = [c for c in METADATA_COLUMNS + LABEL_COLUMNS if c in df.columns]
        features = df.drop(columns=drop_cols)
        # Retain only numeric columns
        return features.select_dtypes(include=[np.number])


class FlowDataset(Dataset):
    """PyTorch Dataset wrapper for network flow vectors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_flow_dataframe(
    data_path: str = "./data/real_flows.csv",
    require_real: bool = True,
    n_synthetic_samples: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Loads the real UNSW-NB15 flow sample, or synthesises a fallback corpus when
    `require_real` is False and the file is absent.

    Returns the raw DataFrame *including* the metadata and label columns so that
    each experiment can perform its own stratified split before preprocessing.
    """
    if data_path and os.path.exists(data_path):
        print(f"[DataLoader] Ingesting real flow dataset from: {data_path}")
        return pd.read_csv(data_path)

    if require_real:
        raise FileNotFoundError(
            f"Required real dataset is missing: {data_path}. Run src/fetch_real_dataset.py first."
        )

    print(f"[DataLoader] Generating {n_synthetic_samples} synthetic flow records (seed={random_seed})...")
    df = SyntheticFlowGenerator(random_seed=random_seed).generate(n_samples=n_synthetic_samples)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_flows.csv", index=False)
    return df


def get_network_flow_dataloaders(
    data_path: Optional[str] = None,
    n_synthetic_samples: int = 5000,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 64,
    random_seed: int = 42,
    feature_range: Tuple[float, float] = (-math.pi, math.pi),
) -> Tuple[DataLoader, DataLoader, DataLoader, FlowPreprocessor]:
    """
    End-to-end data ingestion pipeline:
    - Loads existing dataset if data_path exists, otherwise synthesizes realistic flow traffic.
    - Splits into Train, Validation, and Test sets using stratified sampling.
    - Fits preprocessor strictly on Train, then transforms Val and Test to prevent data leakage.
    - Returns PyTorch DataLoaders and the fitted preprocessor.
    """
    df = load_flow_dataframe(
        data_path=data_path or "data/real_flows.csv",
        require_real=False,
        n_synthetic_samples=n_synthetic_samples,
        random_seed=random_seed,
    )

    # Stratified Train/Val/Test split
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df["label"]
    )
    val_rel_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_rel_size, random_state=random_seed, stratify=train_val_df["label"]
    )

    print(
        f"[DataLoader] Split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} | "
        f"Attack ratio: {df['label'].mean():.2%}"
    )

    # Preprocessing: Fit strictly on train_df
    preprocessor = FlowPreprocessor(feature_range=feature_range)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    # Construct Datasets & DataLoaders
    train_dataset = FlowDataset(X_train, y_train)
    val_dataset = FlowDataset(X_val, y_val)
    test_dataset = FlowDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, preprocessor


if __name__ == "__main__":
    print("=" * 70)
    print("Project HQ-IDS-029: Running Flow Data Loader Self-Test")
    print("=" * 70)

    real_df = load_flow_dataframe("./data/real_flows.csv", require_real=False)
    train_ld, val_ld, test_ld, prep = get_network_flow_dataloaders(
        data_path="./data/real_flows.csv",
        n_synthetic_samples=1000,
        batch_size=32,
        random_seed=42,
    )

    X_sample, y_sample = next(iter(train_ld))
    print(f"\n[Verification] Raw DataFrame shape: {real_df.shape}")
    print(f"[Verification] Retained feature names ({len(prep.feature_names_)}): {prep.feature_names_}")
    for banned in ("id", "proto", "service", "state", "attack_cat"):
        assert banned not in prep.feature_names_, f"Leakage: metadata column '{banned}' survived stripping!"
    print("[Verification] Metadata strip OK (no id/proto/service/state/attack_cat columns).")
    print(f"[Verification] Batch X shape: {X_sample.shape} (Features={X_sample.shape[1]})")
    print(f"[Verification] Batch y shape: {y_sample.shape} (Class counts: Benign={int((y_sample==0).sum())}, Attack={int((y_sample==1).sum())})")
    print(f"[Verification] Value bounds: Min={X_sample.min().item():.4f}, Max={X_sample.max().item():.4f} (Target: [-pi, pi] = [{-math.pi:.4f}, {math.pi:.4f}])")
    print(f"[Verification] Detected {len(prep.skewed_features_)} skewed features requiring log1p transform.")
    print(f"[Verification] Skewed features: {prep.skewed_features_}")
    print("\nSelf-test PASSED successfully!")
