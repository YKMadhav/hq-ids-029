# HQ-IDS-029 handoff (written 2026-11-27, paste into a new Claude session)

## 0. How to use this
Repo: https://github.com/YKMadhav/hq-ids-029 (public). **Work on branch `forensic-rerun-corrected` + PR #1
(`claude/forensic-additions` -> `forensic-rerun-corrected`). Do NOT treat `main` as current**: main is the old, leaky
protocol (last commit 13 Nov). A new session must `git fetch origin forensic-rerun-corrected claude/forensic-additions`.
Read first: `SCIENTIFIC_STATUS.md`, `results/qadv/FINDINGS.md`, `README.md` header, this file.
Owner's goal: find ANY area where the quantum layer genuinely helps. Honest result so far: none found. The owner wants
exploration of new directions, not a justification of the existing story. Be skeptical, tag confidence, do not flatter.
Owner style: challenge assumptions first, uncomfortable truth first, tag claims [Certain]/[Likely]/[Guessing].
Session limits: GitHub pushes work via git to new branches; branch DELETION and some GitHub writes are blocked by the
proxy; PR creation works via `gh api repos/YKMadhav/hq-ids-029/pulls` (GraphQL blocked). 2 CPU cores only.
Stale/hook noise: a stop hook repeatedly claims "15 unpushed commits" - it is wrong (checked with git ls-remote).

## 1. Project in one paragraph
Hybrid quantum-classical network IDS on UNSW-NB15 (5,000-flow sample `data/real_flows.csv`, 39 features, binary label).
Classical encoder -> 4 latent angles -> 4-qubit variational circuit (PennyLane/Qiskit, exact simulation) -> classifier.
Part of an Amaravati Quantum Valley / QAIC program (attached PDF was the team-lead portal manual; not project content).

## 2. History
- Months 1-3: built autonomously by Antigravity (owner suspected hallucination). Audit found a fabricated "Matched MLP"
  number (removed 12 Nov), a zero-day target leak (patched), unsourced citations (removed).
- `main` (last commit 13 Nov): old protocol, results NOT citable per its own SCIENTIFIC_STATUS.md.
- `forensic-rerun-corrected` (to 17 Nov, work done in another account/session): the real corrected rerun. Found
  571/5000 duplicate feature vectors -> 17% of test flows had a training twin; de-duplication added
  (`experiments/common.py::deduplicate_flows`); official UNSW train->test evaluation; threads pinned to 1
  (`src/determinism.py`, bit-reproducible); paper regenerated (`paper/paper.tex`, `scripts/generate_paper_numbers.py`,
  `results/PAPER_NUMBERS.json`); large quantum-advantage search in `qadv/`.

## 3. Authoritative results (corrected protocol; from SCIENTIFIC_STATUS.md / FINDINGS.md)
- Headline (5 seeds, ~886 test flows, 64.5% attack): flag-everything 0.6445; one-feature stump 0.9126; RF 0.9318;
  XGBoost 0.9325; MLP 0.9284; HybridQNN 0.9269. Quantum is 1.4 pts above ONE threshold on ONE feature (sttl/ct_state_ttl).
- 30-seed ablation: full 0.9247, quantum bypassed 0.9270, quantum frozen 0.9273 -> quantum layer slightly WORSE (p=0.015,
  uncorrected; split variance only). Old "TOST equivalence / quantum helps" claims are unsupported.
- Official UNSW test (80,717 flows): XGB 0.854, RF 0.836, MLP 0.825, Hybrid 0.818, stump 0.762; all models have 30-40% FPR.
- Zero-day (benign-only training) AE-based: AUC ~0.67-0.72, recall 6-14%. Weak.
- qadv search (exact simulation, 4-12 qubits): geometric difference gate shows no kernel-advantage room; quantum-kernel
  one-class SVM <= tuned RBF/Laplacian/kNN on UNSW, CIC-IDS2017 and CIC-PQC_OAV; few-shot quantum-kernel SVM below RBF-SVM;
  MPS Born machine weak (AUC 0.70-0.71); Grover signature search not competitive (resource model); PQ-TLS handshake
  size is a trivially classical separator. Every apparent early win vanished against a tuned classical control.

## 4. What this session added (all on `claude/forensic-additions`, PR #1)
1. `scripts/run_grouped_artifact_ablation.py` -> `results/grouped_artifact_results.csv` (80 rows; de-duplicated
   protocol, 1 thread). Random-split arm reproduces headline (RF .9318, XGB .9325, Hybrid .9269).
   - Dropping 8 ASSUMED testbed-artifact fields (sttl,dttl,swin,dwin,stcpb,dtcpb,ct_state_ttl,is_sm_ips_ports; list is my
     assumption) costs all models ~0.4-0.6 acc pts -> artifacts are not the main driver.
   - Leave-one-attack-family-out (1 seed/family): DoS, Recon, Generic ~0.99-1.0 recall for all models, Exploits 0.84-0.99,
     Fuzzers only 0.26-0.41 for EVERY model incl. VQC. VQC never better on any family.
2. `scripts/verify_aligned_data.py` + `data/aligned_data_manifest.json` + `data/ALIGNED_DATA_PROVENANCE.md`: all 12 fields of
   `unsw_aligned_5k.csv` / `cic_aligned_5k.csv` reproduced exactly from raw. UNSW aligned = real_flows (sinpkt/dinpkt /1000).
   CIC mapping: dur=FlowDuration/1e6; rate=(spkts+dpkts)/max(dur,1e-4); smean/dmean=Fwd/Bwd Packet Length Mean;
   sinpkt/dinpkt=Fwd/Bwd IAT Mean/1e6; swin/dwin=clip(Init_Win*255/65535,0,255). 581/5000 CIC labels ambiguous in raw.
   **Semantic mismatch (kept `field_semantics_verified=false`, RFF benchmark stays blocked)**: UNSW rate=(pkts-1)/dur vs CIC
   pkts/dur; UNSW swin only {0,172,255} vs CIC continuous; IAT definitions differ; CIC class order mirrors UNSW (3403/1597).
   Owner decision: do NOT accept the mismatch; treat cross-dataset as exploratory; do not flip the flag to bypass the guard.
3. Environment recipe that works: `uv venv --python 3.11`; install pinned `requirements.txt` in two steps (light deps, then
   torch/pennylane/qiskit in background - one big install times out); `pytest` ran 19 tests on old tree.
4. Mistakes to avoid: an old-protocol rerun I did earlier (results on a now-stale branch `claude/audit-followup-2026-11-27`,
   undeletable by the session; owner should delete it) is WRONG protocol; ignore it. The earlier "VQC collapses on unseen
   Generic attacks" finding was a leaky-protocol artifact. Earlier README claim "formally proves" lives in
   EXECUTION_LOG.md not README.
5. Not done: Phase 4 early-packet proxy rerun (SequentialQNN training >25 min on 2 cores - run on a bigger machine);
   temporal split (UNSW has no timestamps); regenerating manuscripts; manuscript-number tests expectedly fail on old branch.

## 5. Recommended next directions (ordered; goal = find where quantum could genuinely help)
Principle: classical flow features + <=12 simulated qubits has been exhausted. Next, change the SETTING, not tune the circuit.
A. Quantum-native data (highest value). Network-derived data with real quantum structure: e.g. QKD/quantum-network
   telemetry, quantum-channel tomography/noise-fingerprint anomaly detection, PQC handshake/side-channel traces (timing,
   power), quantum-random-number-generator health tests. Quantum models can plausibly win where the DATA is quantum.
   Need a dataset or a physically-motivated simulator (Qiskit Aer noise models) - define the task first.
B. Constrained-resource regimes where small models matter: parameter/FLOP-matched comparisons at very small budgets
   (e.g. <=50 parameters, edge/line-rate), measuring accuracy per parameter and latency, with matched classical controls.
   Honest test of "quantum as compact model class". Use the parameter-matched Fourier/RFF controls (needs item 3 resolved).
C. Combinatorial/optimization sub-tasks of an IDPS (not classification): rule/feature-subset selection, alert correlation,
   signature-set cover as QUBO; compare QAOA/annealing-style (simulated) against strong classical solvers (CP-SAT, SA,
   tabu) on instance families with equal wall-clock. Expect classical to win at small n; look at scaling trends only.
D. Fix the evaluation before inventing models: temporal split on CIC-IDS2017 (day-wise) or CSE-CIC-IDS2018 (sampler exists:
   `qadv` CIC-2018 sampler/drift experiment on branch), grouped CV with multiple seeds per held-out family for Fuzzers
   (all models ~0.3 recall: hardest, most informative failure). Report FPR@95%TPR, not accuracy.
E. Quantum reservoir / data re-uploading with higher expressivity tested for Fuzzers specifically; dequantization check
   (does a Random-Fourier-Features model with matched frequency spectrum reproduce it?) before claiming anything.
F. Noise as a feature: test whether hardware noise (shot noise, readout error) acts as regularisation on tiny datasets;
   run few-shot (k<=20) on real IBM hardware only after a simulated noisy result looks different from classical.
G. If nothing wins: write the rigorous negative-result/benchmark paper (already supported by data): 4-qubit VQC IDS is
   no better than a bypass, a frozen random circuit, or a one-threshold stump after leakage removal.
Immediate concrete steps for the next agent: (1) merge/review PR #1; (2) decide A vs B vs C with the owner (ask which data
or resources exist); (3) pick one, write the task + classical controls + success criterion BEFORE running; (4) pin threads,
dedup, 5+ seeds, bootstrap CIs, tune classical controls as hard as the quantum model; (5) log in EXECUTION_LOG.md and
update SCIENTIFIC_STATUS.md; push to a new `claude/...` branch and open a PR.

## 6. Open decisions for the owner
- Which direction (A/B/C) and what data/hardware access exists (IBM account? QKD/PQC traces? bigger CPU?).
- Delete stale branch `claude/audit-followup-2026-11-27` (session cannot).
- Whether to regenerate manuscripts and fix README on the forensic branch, and whether to keep the cross-dataset section.
- Confirm/edit the testbed-artifact feature list.
