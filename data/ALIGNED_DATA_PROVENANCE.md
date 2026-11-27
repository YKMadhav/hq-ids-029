# Aligned cross-dataset data: provenance and semantic status

`unsw_aligned_5k.csv` and `cic_aligned_5k.csv` are historical artifacts whose
original extraction code was not retained. As of 2026-11-27 every field of both
files has been **reverse-engineered and machine-verified** against raw sources by
`scripts/verify_aligned_data.py`, which writes `data/aligned_data_manifest.json`
(SHA-256 of both CSVs and both sources, exact field mappings, verification counts).

## What is verified
- UNSW aligned = `data/real_flows.csv` rows, 12 columns copied as-is except
  `sinpkt`/`dinpkt` divided by 1000 (ms to s). 5000/5000 rows.
- CIC aligned: all 5000 rows map to rows of the raw CICIDS2017 sample; all 12
  fields are reproduced exactly (formulas in the manifest). Of the rows, 4419 have
  an unambiguous raw label and all 4419 agree; 581 share a feature key with both
  benign and attack raw rows, so their label cannot be confirmed from features.

## What is NOT verified (why `field_semantics_verified` stays `false`)
1. `rate`: UNSW is (pkts-1)/dur; CIC here is pkts/max(dur, 100 us).
2. `swin`/`dwin`: UNSW takes only {0, 172, 255}; CIC is `Init_Win_bytes` rescaled by
   255/65535 and clipped. Different quantities on a common range.
3. `sinpkt`/`dinpkt`: Argus interpacket times vs CICFlowMeter Fwd/Bwd IAT means.
4. The CIC label sequence equals the UNSW aligned sequence row for row (3403
   attack / 1597 benign): matched sampling, not the natural class prior.

Until a human reviewer decides these are acceptable (or the mapping is changed),
`experiments/run_rff_surrogate_benchmark.py` stays blocked and cross-dataset F1
numbers must be described as an exploratory, semantically mismatched transfer.
Do not set `field_semantics_verified: true` merely to bypass the guard.
The raw CICIDS2017 file used (56,661 rows) is a sample, not the full dataset.
