# Project HQ-IDS-029: NISQ-Compatible Hybrid Quantum-Classical Network Intrusion Detection System

## 1. Overview
Project **HQ-IDS-029** (`hq-ids-029`) investigates and implements a resource-constrained, NISQ-compatible Hybrid Quantum-Classical Network Intrusion Detection System. Operating within a strict hardware budget of $\le 4$ to $6$ qubits, the architecture integrates:
- High-throughput classical network flow preprocessing and skew reduction.
- Classical compression via bottleneck encoders (PyTorch Autoencoder / MLP) reducing high-dimensional network flow representations down to 4 latent continuous parameters $z \in [-\pi, \pi]^4$.
- Parameterized Quantum Circuits (PQCs) utilizing entangling feature maps (`ZZFeatureMap` / data re-uploading) and variational ansatzes (`RealAmplitudes` / `TwoLocal`) measured via Pauli-$Z$ expectation values.
- Classical decision heads for binary threat detection and zero-day anomaly classification.
- Rigorous ablation controls, classical tree ensemble baselines (Random Forest, XGBoost), and simulated NISQ noise with readout error mitigation.

## 2. Theoretical & Literature Backbone
This codebase operationalizes key insights from foundational literature in QML network security:
1. **Abreu et al. (QuantumNetSec / QML-IDS)**: Benchmarking VQC, QSVM, and QCNN against flow datasets (UNSW-NB15, CICIDS2017, CICIoT23).
2. **Chen et al. (JHU)**: Compressed feature representations ($\le 4$ qubits) via classical MLP encoders + VQC with M3 readout mitigation and XY8 dynamical decoupling on noisy NISQ hardware.
3. **Vishvaa et al. (Q-NIDS)**: Critical benchmarking demonstrating that classical tree ensembles (Random Forest, XGBoost) remain formidable baselines, mandating strict ablation controls to prevent attribution of classical post-processing power to "quantum advantage."
4. **Al Malwi et al. & Chinthalagattu (IIT Madras)**: Generative QCBMs/QGANs and Quantum Autoencoders for zero-day threat detection.
5. **Al Mazroa et al. (RHQ-CNN)**: Residual classical connections to avoid barren plateaus in variational circuits.
6. **Spadari et al. (University of Napoli Federico II / Verona)**: Early traffic classification using transport-level header sequences (Payload Length, Direction, TCP Window Size) across the first 10 packets of bi-directional flows (biflows).

## 3. Repository Architecture
```text
hq-ids-029/
├── README.md               # Project overview and reproduction guidelines
├── requirements.txt        # Python dependency manifest
├── EXECUTION_LOG.md        # Chronological audit trail of all actions and autonomous decisions
├── configs/
│   └── config.yaml         # Centralized hyperparameters, seeds, and qubit budgets
├── data/                   # Raw and processed flow datasets (CSV/PCAP/Parquet)
├── notebooks/              # Exploratory data analysis and circuit visualization
├── src/
│   ├── data_loader.py      # Flow data generation, cleaning, log scaling, and normalization
│   ├── classical_models.py # RF, XGBoost, MLP baseline classifiers
│   ├── encoders.py         # PyTorch Autoencoder and Bottleneck MLP compressors
│   ├── quantum_circuits.py # Qiskit and PennyLane feature maps and variational ansatzes
│   ├── hybrid_pipeline.py  # End-to-end PyTorch-QML neural network module
│   └── error_mitigation.py # M3 readout mitigation and dynamical decoupling routines
├── experiments/
│   ├── run_baselines.py    # Classical baseline benchmarking suite
│   ├── run_hybrid_sim.py   # Hybrid quantum simulator benchmarking
│   └── run_ablation.py     # Rigorous ablation studies (quantum bypassed/frozen)
└── results/
    ├── figures/            # ROC-AUC curves, confusion matrices, circuit diagrams
    └── metrics.csv         # Structured comparative metrics log
```

## 4. Setup & Quickstart
```bash
# Clone or navigate to the repository
cd hq-ids-029

# Install dependencies
pip install -r requirements.txt

# Run initial pipeline verification with synthetic dummy traffic
python3 src/data_loader.py
```

## 5. Reproducing the Full Benchmark

Run the four stages in order. Every stage writes CSV artifacts to `results/` and
appends an audit entry to `EXECUTION_LOG.md`.

```bash
# Stage 1: classical baselines (Random Forest, XGBoost, Classical MLP) x 5 seeds
python3 experiments/run_baselines.py

# Stage 2: Hybrid QNN on the real UNSW-NB15 test splits x 5 seeds
python3 experiments/run_hybrid_sim.py

# Stage 3: 3-variant quantum-layer ablation + true shot-based Qiskit Aer
#          NISQ simulation (1024 shots, depolarising + readout noise, M3 mitigation)
python3 experiments/run_ablation.py

# Stage 4: publication figures + PROJECT_FINAL_REPORT.md, generated from the CSVs
python3 experiments/generate_plots_and_report.py
```

**Notes on the environment.** The project targets Python 3.10 with PyTorch,
PennyLane, and Qiskit/Aer. If the default interpreter lacks `torch`, point `python3`
at the interpreter that has it.

**Data.** `data/real_flows.csv` holds 5,000 real UNSW-NB15 flows (3,403 attack /
1,597 benign). `src/data_loader.py` strips the non-predictive columns
(`id`, `proto`, `service`, `state`, `attack_cat`) and retains **39** continuous
features; the scaler is fit on the train partition only.

**Results at a glance** (5 seeds, mean accuracy): XGBoost 0.9452, Random Forest
0.9430, Hybrid QNN 0.9396, Classical MLP 0.9352. The Hybrid QNN achieves the lowest
false positive rate (0.1467) of all four. See `PROJECT_FINAL_REPORT.md` for the
full methodology, ablation, and NISQ/M3 analysis.

**Key artifacts:**
- `results/metrics.csv` — aggregated benchmark, one row per model
- `results/ablation_results.csv` — 15 rows (3 variants x 5 seeds)
- `results/noise_results.csv` — 5 rows of 1024-shot Aer NISQ metrics
- `results/figures/` — latency, ablation, confusion matrix, and noise figures
