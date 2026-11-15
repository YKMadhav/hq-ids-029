# Data sources for the next experiments (checked 2026-11-15)

"Retrieved" = downloaded from this sandbox. The sandbox network only allows package registries and GitHub, so most hosts are blocked from here; you can download them yourself and upload them.

## Post-quantum TLS (item 6)
| Source | What it is | Status |
|---|---|---|
| CIC-PQC_OAV v1, UNB: https://www.unb.ca/cic/datasets/pqc-oav-dataset-2025.html | 40,010 labelled TLS 1.3 sessions (about 87.5% normal, 12.5% anomalous) with classical and ML-KEM/ML-DSA handshakes; 32 features + JSONL + scenario manifest. **Synthetic testbed**, no raw pcaps, timestamps not stated. Download: http://205.174.165.80/CICDataset/CIC-PQC_OAV%20v1/ | **Blocked here (host not allow-listed). Best match for UC-029; you need to download it.** |
| Telefonica/Public_Performance_Quantum_Safe (GitHub), paper arXiv 2603.11006 | Containerised testbed + `pcap_layer_analysis.py` for classical, hybrid and pure-PQC TLS 1.3 handshakes. No pcaps released. | Retrieved (code only, 276 KB). Can be used to generate captures. |
| CipherIQ/pqc-flow (GitHub, GPL-3.0) | Passive PQC detector for TLS/SSH/QUIC pcaps. Tool, not data. | Not cloned. |
| arXiv 2608.22683 "Colossus with Feet of Clay" | Encrypted-traffic classifiers under hybrid PQC; code at an anonymous.4open.science link, data release not stated. | Not retrieved. |
| Cloudflare Radar post-quantum page | Aggregate PQ adoption statistics, not packets. | Not retrieved. |

## Time-ordered / drift data (item 5)
| Source | Notes | Status |
|---|---|---|
| CSE-CIC-IDS2018 (UNB): https://www.unb.ca/cic/datasets/ids-2018.html and BCCC mirror https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/large-scale-intrusion-detection-dataset-bccc-cse-cic-ids2018/ | Multi-day captures with attack days and timestamps in the original CSVs. Large. | Blocked here. |
| CIC-IDS2017 original CSVs (UNB) | Per-day files with Timestamp column; our project copy is a 56,661-flow sample with no timestamps. | Blocked here. |
| CTU-13: https://www.stratosphereips.org/datasets-ctu13 ; GitHub CSV conversion https://github.com/imfaisalmalik/CTU13-CSV-Dataset | 13 botnet scenarios. The GitHub CSVs merge attack/normal and drop scenario and time. | Stratosphere: not retrieved. GitHub CSV: **retrieved** (38,898 attack + 53,314 normal flows, CICFlowMeter features, no timestamps). |

## Catalogues
- Continuous survey of intrusion-detection datasets: https://arxiv.org/html/2408.02521v1
- Würzburg NIDS dataset list: https://www.informatik.uni-wuerzburg.de/datascience/research/datasets/nids-ds/
