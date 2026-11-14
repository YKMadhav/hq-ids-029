# Scientific status and result provenance

**Do not treat the committed CSVs, figures, reports, or manuscripts as
publication-ready evidence.** They were generated under an earlier protocol and
are retained only as historical artifacts until the corrected pipeline has been
rerun.

The repository now enforces or documents the following boundaries:

- Label permutation is a negative control, not a leakage proof. The stress test
  also reports held-out single-feature AUCs and fails on a near-perfect feature.
- Matrix inversion uses an assumed simulator assignment matrix. It is not M3,
  IBM-calibrated mitigation, or a hardware result.
- The CSV Phase 4 setup-header task is a two-observation proxy, not a raw-PCAP
  packet-prefix study. Its protocol code has no ARP/HOPOPT collision.
- The Fourier experiments are exploratory controls, not parameter- or
  spectrum-matched proofs of equivalence.
- Random re-splits quantify split variability, not population uncertainty. A
  grouped or temporal protocol and an explicit testbed-artifact ablation are
  required before making deployment or generalization claims.

Before external use, regenerate all artifacts from a pinned environment, record
dataset hashes and source-generation code, run Phase 4 and multiclass explicitly,
and update the manuscripts only from the regenerated outputs.

---

## Regenerated results (corrected protocol)

Produced by `reproduce.sh` on Python 3.11, torch 2.6.0, 1 pinned thread
(`results/RUN_MANIFEST.json`). Two independent full runs gave bit-identical CSVs
(timing columns excluded) after the zero-day seeding fix. The manuscripts, README and
`PROJECT_FINAL_REPORT.md` still quote the OLD numbers and are out of date by about
1.3 accuracy points; the `manuscript_sync` tests fail on purpose to say so.

What changed the numbers:
1. **Exact-duplicate leakage removed.** 571 of 5,000 sample rows are duplicate feature
   vectors; 166 of 1,000 test flows at seed 42 had an identical training twin. The
   splitter now de-duplicates first (`experiments/common.py`).
2. **Thread count pinned.** 1 vs 2 torch threads give bit-different outputs for the same seed
   (`results/thread_sensitivity.json`). On the current de-duplicated protocol this did not
   change any 3-epoch test prediction, but on the earlier non-de-duplicated split it moved
   accuracy 93.3 % -> 93.7 % at seed 42, so `src/determinism.py` pins 1 thread as a precaution.
   (An earlier version of this note stated the accuracy effect without this qualification.)
3. **Zero-day autoencoders** drew initial weights before seeding; now seeded.
4. **Label-shuffle gate** was mis-calibrated (null AUC scatters 0.27-0.70 across
   shuffles, balanced accuracy is exactly 0.50); it now uses 10 shuffles.

### Headline benchmark (5 seeds, de-duplicated, about 886 test flows, 64.5 % attack)

| Model | Accuracy | Balanced acc. | ROC-AUC |
|---|---|---|---|
| Flag-everything | 0.6445 | 0.500 | 0.500 |
| One-split stump (best single feature) | 0.9126 ± 0.0044 | 0.878 | 0.878 |
| RandomForest | 0.9318 ± 0.0087 | 0.910 | 0.983 |
| XGBoost | 0.9325 ± 0.0096 | 0.918 | 0.983 |
| ClassicalMLP | 0.9284 ± 0.0086 | 0.904 | 0.977 |
| HybridQNN | 0.9269 ± 0.0055 | 0.903 | 0.977 |

The quantum model is 1.4 points above one threshold on one feature and statistically
indistinguishable from the classical MLP. The stump's best feature is `sttl`/
`ct_state_ttl` (single-feature AUC about 0.88).

### 30-seed ablation (re-splits of the same 5,000 flows; p-values describe split variance only)

Full HybridQNN 0.9247 ± 0.0083, quantum bypassed 0.9270 ± 0.0092, quantum frozen
0.9273 ± 0.0081. Full vs bypassed: p = 0.0151, d = -0.47; full vs frozen: p = 0.0074,
d = -0.53 (uncorrected). The quantum layer is slightly WORSE, not equivalent. The old
"TOST equivalence / quantum helps" language is not supported.

### Official UNSW-NB15 testing set, 80,717 unseen flows (54.5 % attack, train/test disjoint)

| Model | Accuracy | AUC | FPR |
|---|---|---|---|
| Flag-everything | 0.545 | 0.500 | 1.00 |
| Stump | 0.762 | 0.739 | 0.52 |
| RandomForest | 0.836 | 0.971 | 0.35 |
| XGBoost | 0.854 | 0.975 | 0.30 |
| ClassicalMLP | 0.825 | 0.951 | 0.36 |
| HybridQNN | 0.818 | 0.941 | 0.38 |

Every model has 30-40 % false-positive rate at the default threshold: unusable as an
IDS. This is a random partition of the same captures, not a temporal or cross-network
test. Collapsing within-test duplicates (`*_dedup_test.csv`) lowers accuracy further
because the official test set is about one-third duplicate attack vectors.

### Other regenerated artifacts
- Zero-day (train on benign only): ROC-AUC 0.674 (CAE), 0.684 (QAE), 0.718 (hybrid);
  recall 6.5-13.9 % at 3.9-7.1 % FPR. Weak.
- Readout mitigation: classification accuracy is unchanged by simulated noise; the
  expectation-value error versus analytic falls from 0.069-0.077 to 0.032-0.037.
- Multiclass benchmark: re-run on the de-duplicated data; not analysed here.

### Still NOT done
- **Phase 4** needs `UNSW_NB15_training-set.csv`, which was not available; only the
  stored predictions were validated (all four models about 78.6 % accuracy, 47.4 % FPR).
- **RFF surrogate and cross-dataset** claims are withdrawn: the aligned CSVs have no
  generating code and no verified field semantics (no manifest, so the steps are skipped).
- No temporal or grouped split exists in these CSVs (no timestamps or host IDs).
- Manuscripts, README and report have not been rewritten from these numbers.
