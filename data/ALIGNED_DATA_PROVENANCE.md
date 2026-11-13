# Aligned cross-dataset data is not yet reproducible

`unsw_aligned_5k.csv` and `cic_aligned_5k.csv` are historical artifacts. This
repository does not contain their extraction or field-alignment code, and shared
column names do not by themselves demonstrate shared semantics across Argus and
CICFlowMeter.

Accordingly, `experiments/run_rff_surrogate_benchmark.py` now requires an
untracked `aligned_data_manifest.json` before it will run. The manifest must
contain SHA-256 values for both CSVs, source dataset versions, extraction and
mapping code references, and `field_semantics_verified: true` after a human
review. Do not manufacture that declaration merely to bypass the guard.
