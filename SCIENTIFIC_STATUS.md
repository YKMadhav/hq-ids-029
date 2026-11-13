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
