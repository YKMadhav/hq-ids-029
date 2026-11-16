# Quantum-advantage search for UC-029 (UNSW-NB15, de-duplicated) - findings

Scope: exact classical simulation, 4-8 qubits, UNSW-NB15 training CSV de-duplicated. Code in `qadv/`, raw results in `results/qadv/`.
Hyperparameter grids were chosen with knowledge of the evaluation families, so every "best of grid" number is optimistic - for quantum and classical detectors alike.

## Item 1 - Geometric-difference gate (Huang et al. 2021), 600 training flows, 5 seeds
|   qubits |   bw |   g_C_to_Q_min |   g_Q_to_C_min |   g_classical_ref |   eff_rank_Kq |
|---------:|-----:|---------------:|---------------:|------------------:|--------------:|
|        4 |  0.5 |           3.54 |           2.22 |              2.51 |         26.84 |
|        4 |  1   |           4.58 |           3.1  |              2.51 |         42.88 |
|        8 |  0.5 |           2.59 |           2.4  |              2.19 |        287.16 |
|        8 |  1   |           3.06 |           3.39 |              2.19 |        387.57 |

sqrt(N) = 24.5 is the scale at which a kernel advantage becomes possible. Measured g(C->Q) = 2.6-4.6, close to the 2.2-2.5 between two *classical* RBF kernels. No room for a kernel advantage on this representation.

## Item 4 / 2 - Leave-one-family-out zero-day, benign-only training (10 seeds x 9 families)
AUC (higher better) and FPR at 95% TPR (lower better; the SOC metric):

| detector                                            |   auc |   fpr_at_95tpr |
|:----------------------------------------------------|------:|---------------:|
| kNN(k=10, 8d PCA)                                   | 0.881 |          0.339 |
| OCSVM-Laplacian(8d,gamma=10.0)                      | 0.88  |          0.371 |
| OCSVM-QuantumKernel(8q,bw=1.0)                      | 0.874 |          0.468 |
| OCSVM-RBF(8d,gamma=30.0)                            | 0.866 |          0.468 |
| IsolationForest                                     | 0.858 |          0.334 |
| Mahalanobis                                         | 0.842 |          0.433 |
| Autoencoder                                         | 0.731 |          0.699 |
| MPS-BornMachine(8d,chi=12)                          | 0.701 |          0.72  |
| OCSVM-ProductStateKernel(8q,no-entanglement,bw=1.0) | 0.572 |          0.797 |

Paired difference, quantum kernel (bw=1.0) minus classical, bootstrap over seeds:
- vs kNN on the same 8-d input: AUC -0.007 [-0.016, +0.001]; FPR@95TPR +0.129 [+0.094, +0.173] (quantum worse).
- vs Laplacian-kernel OCSVM (gamma=10): AUC -0.006 [-0.016, +0.002]; FPR +0.097 (quantum worse).
- vs RBF-OCSVM (gamma=30): AUC +0.008 [-0.001, +0.017]; FPR -0.000 [-0.036, +0.046] (tie).
- vs Isolation Forest (39-d): AUC +0.016 [+0.005, +0.025]; FPR +0.134 (higher AUC, much worse FPR).
- vs the same circuit without entanglement: AUC +0.303; FPR -0.329 (entanglement matters for this kernel, but a local classical kernel recovers the same performance).

A first look suggested the quantum kernel was best (AUC 0.874 vs RBF 0.70-0.78); that was an artefact of an RBF grid whose best value sat at the grid edge. Extending the classical grid removed the gap. The tensor-network Born machine (AUC 0.70) was weaker than simple distance methods in this untuned configuration (8 bins, chi 6-12, 250 steps).

## Item 3 - Few-shot detection of a new family (6 repetitions, hyperparameters fixed before testing)
AUC by number k of labelled attack examples:

| model                 |     1 |     5 |    20 |   100 |
|:----------------------|------:|------:|------:|------:|
| LogReg(8d)            | 0.744 | 0.886 | 0.935 | 0.944 |
| QuantumKernelSVM(8q)  | 0.684 | 0.803 | 0.883 | 0.947 |
| RBF-SVM(8d,gamma=0.3) | 0.749 | 0.903 | 0.955 | 0.978 |
| RBF-SVM(8d,gamma=1)   | 0.703 | 0.847 | 0.918 | 0.97  |
| kNN(k=min(5,k))       | 0.55  | 0.747 | 0.867 | 0.952 |

FPR at 95% TPR:

| model                 |     1 |     5 |    20 |   100 |
|:----------------------|------:|------:|------:|------:|
| LogReg(8d)            | 0.745 | 0.453 | 0.239 | 0.174 |
| QuantumKernelSVM(8q)  | 0.832 | 0.758 | 0.628 | 0.209 |
| RBF-SVM(8d,gamma=0.3) | 0.694 | 0.41  | 0.224 | 0.057 |
| RBF-SVM(8d,gamma=1)   | 0.813 | 0.686 | 0.531 | 0.098 |
| kNN(k=min(5,k))       | 1     | 1     | 1     | 0.511 |

The quantum-kernel SVM is below the RBF-SVM at k = 1, 5, 20 and roughly equal at k = 100.

## Item 8 - Fault-tolerant cost of Grover signature search (resource model; assumptions mine)
Assumed: 2^30 signatures, ~1,100 logical qubits, ~1.3e8 T gates for ONE search. Surface-code model, superconducting error 4.9e-4: 1.2-1.8 million physical qubits and 6.2-7.5 hours per single query (two published logical-error fits). A classical hash lookup answers in microseconds. Oracle size and quantum-memory needs are my assumptions and the model is optimistic on time: order-of-magnitude "not competitive", not a measurement.

## Not tested
- Item 5 (drift) / temporal split: UNSW CSVs have no timestamps; CIC-IDS2017 needed.
- Items 6, 7, 9: need datasets not in the project.
- Variational circuits at scale, noisy hardware, more than 8 qubits.

## Conclusion
On classical flow features, with exact simulation up to 8 qubits, no quantum advantage appears in kernel geometry, zero-day detection or few-shot learning; every apparent win disappeared against a tuned classical control. This is a result about this data and these models, not a proof for all quantum methods.


## Replication and scaling checks
**CIC-IDS2017 sample (56,661 flows, 6 families: Bot, BruteForce, DoS, Infiltration, PortScan, WebAttack; 6 seeds), same protocol as Item 4:**

| detector                       |   auc |   fpr_at_95tpr |
|:-------------------------------|------:|---------------:|
| OCSVM-RBF(8d,gamma=10.0)       | 0.902 |          0.387 |
| OCSVM-Laplacian(8d,gamma=3.0)  | 0.897 |          0.391 |
| OCSVM-QuantumKernel(8q,bw=0.5) | 0.87  |          0.402 |
| IsolationForest                | 0.848 |          0.413 |
| kNN(k=10, 8d PCA)              | 0.847 |          0.413 |
| OCSVM-QuantumKernel(8q,bw=1.0) | 0.842 |          0.474 |
| Autoencoder                    | 0.816 |          0.417 |
| MPS-BornMachine(8d,chi=12)     | 0.706 |          0.577 |

The best quantum-kernel setting (AUC 0.870) is below a tuned RBF one-class SVM (0.902) on the same 8-d input, so the UNSW conclusion replicates on a second dataset. This file has no timestamps or day labels, so drift (Item 5) still cannot be tested.

**Item 1 at 12 qubits (500 flows, 3 seeds); sqrt(N) = 22.4:**

|   bw |   mean |   min |   max |
|-----:|-------:|------:|------:|
|  0.5 |   2.29 |  2.1  |  2.65 |
|  1   |   2.33 |  2.02 |  2.64 |

The geometric difference does not grow with qubit count (2.0-2.6 at 12 qubits vs 2.6-4.6 at 4-8), so there is no sign of a regime opening up as the circuit gets larger, within what exact simulation reaches.

## Measured post-quantum TLS 1.3 handshake sizes (OpenSSL 3.5.0, loopback)
`qadv/handshake_sizes.py`; raw rows in `results/qadv/handshake_sizes.csv`. A counting TCP proxy between `s_client` and `s_server` (tickets off), 3 repetitions per cell; byte counts are deterministic (standard deviation 0-1 byte). Requires an OpenSSL >= 3.5 build (set `OSSL` to its binary; the system 3.0.13 has no ML-KEM).

- First client flight: 217 B (X25519) vs 1,393 B (X25519MLKEM768), 6.4x; 1,745 B for ML-KEM-1024.
- Server flight with an ECDSA-P256 certificate: 781 B (X25519) vs 1,869 B (X25519MLKEM768); with ML-DSA-65: 10,231 B (13x the classical case).
- A size threshold on the first client flight separates PQ from classical key exchange perfectly **for one client implementation**. Real clients vary in extensions and padding; this is a protocol measurement, not an intrusion-detection result.
- The `first_s2c` column in the CSV is a socket read size (buffer artefact), not a message size; use `s2c` totals.

## CIC-PQC_OAV v1 (user-supplied, synthetic testbed): leave-one-scenario-out, normal-only training
`ml_features_and_labels.csv` + `scenario_manifest.csv` (join on ID after stripping `.pcap`); 40,010 sessions, 38,494 after exact-duplicate removal; 11 anomaly scenarios; 6 seeds; missing handshake fields set to 0 (missingness stays visible as a value). `raw_sessions.jsonl` was not available and is not needed. Same detectors and protocol as Item 4; the data are not committed (licence not stated; place them in `data/pqc_oav/`).

- Best classical detector: AUC 0.934 (Isolation Forest, 8-d PCA). Best quantum-kernel setting: 0.907; the setting that was best on UNSW (bandwidth 1.0) is the worst here, 0.810.
- Entanglement-free product-state kernel: 0.923, so entanglement is not needed on this data.
- 9 of 11 scenarios are separated almost perfectly (AUC >= 0.99) by some detector; hybrid-PKI (best AUC 0.64) and high-loss (0.70) stay near chance for every detector.
- Caveat: synthetic, scenario-structured data; this says little about real traffic. Normal sessions come from 10 fixed configurations, which makes one-class detection easy.
- Result: no quantum advantage here either.
