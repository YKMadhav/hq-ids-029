"""
Rebuild-and-verify provenance for the aligned cross-dataset CSVs.

Checks that every row of data/cic_aligned_5k.csv and data/unsw_aligned_5k.csv can
be regenerated from the raw sources, records the exact field mappings, and writes
data/aligned_data_manifest.json.

Inputs (not tracked in git; pass paths on the command line):
    --cic-raw   CICIDS2017 flow CSV with the 78 CICFlowMeter columns + Label
    --unsw-real data/real_flows.csv (default)

IMPORTANT: this script NEVER sets field_semantics_verified=true. That flag is a
human judgement; see the "semantic_findings" block that is written to the manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KEY = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
       "Total Length of Fwd Packets", "Total Length of Bwd Packets"]
FEATURES = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
            "smean", "dmean", "sinpkt", "dinpkt", "swin", "dwin"]
RATE_FLOOR_S = 1e-4
WIN_SCALE = 255.0 / 65535.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_unsw(aligned: pd.DataFrame, real: pd.DataFrame) -> dict:
    out = {}
    for c in FEATURES:
        scale = 1 / 1000.0 if c in ("sinpkt", "dinpkt") else 1.0
        out[c] = bool(np.allclose(aligned[c].values, real[c].values * scale, rtol=1e-9, atol=1e-12))
    out["label"] = bool((aligned["label"].values == real["label"].values).all())
    pk = real.spkts + real.dpkts
    m = real.dur > 0
    out["rate_is_(pkts-1)/dur"] = bool(np.isclose(((pk - 1) / real.dur.where(m))[m], real.rate[m], rtol=1e-4).all())
    out["swin_distinct_values"] = sorted(map(float, real.swin.unique()))
    return out


def verify_cic(aligned: pd.DataFrame, raw: pd.DataFrame) -> dict:
    raw = raw.copy()
    raw["k"] = list(zip(*[raw[c] for c in KEY]))
    raw["y"] = (raw["Label"] != "BENIGN").astype(int)
    groups = raw.groupby("k")
    n = len(aligned)
    unique = ambiguous = missing = label_ok = 0
    ok = {c: 0 for c in ["smean", "dmean", "sinpkt", "dinpkt", "swin", "dwin"]}
    for _, r in aligned.iterrows():
        k = (round(r.dur * 1e6), r.spkts, r.dpkts, r.sbytes, r.dbytes)
        if k not in groups.groups:
            missing += 1
            continue
        g = groups.get_group(k)
        if g["y"].nunique() > 1:
            ambiguous += 1
        else:
            unique += 1
            label_ok += int(g["y"].iloc[0] == r.label)
        cand = {
            "smean": g["Fwd Packet Length Mean"].values,
            "dmean": g["Bwd Packet Length Mean"].values,
            "sinpkt": g["Fwd IAT Mean"].values * 1e-6,
            "dinpkt": g["Bwd IAT Mean"].values * 1e-6,
            "swin": np.clip(g["Init_Win_bytes_forward"].clip(lower=0).values * WIN_SCALE, 0, 255),
            "dwin": np.clip(g["Init_Win_bytes_backward"].clip(lower=0).values * WIN_SCALE, 0, 255),
        }
        for c, v in cand.items():
            ok[c] += int(np.isclose(v, r[c], rtol=1e-6, atol=1e-9).any())
    est = (aligned.spkts + aligned.dpkts) / np.maximum(aligned.dur, RATE_FLOOR_S)
    return {
        "rows": n, "no_raw_match": missing, "label_ambiguous_in_raw": ambiguous,
        "label_unique_match": unique, "label_agrees_with_raw": label_ok,
        "fields_reproduced_of_rows": ok,
        "rate_formula_reproduced_rows": int(np.isclose(est, aligned.rate, rtol=1e-6).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic-raw", required=True)
    ap.add_argument("--unsw-real", default=str(REPO / "data" / "real_flows.csv"))
    a = ap.parse_args()

    cic_p, unsw_p = REPO / "data" / "cic_aligned_5k.csv", REPO / "data" / "unsw_aligned_5k.csv"
    cic, unsw = pd.read_csv(cic_p), pd.read_csv(unsw_p)
    cic_raw = pd.read_csv(a.cic_raw)
    real = pd.read_csv(a.unsw_real)

    u, c = verify_unsw(unsw, real), verify_cic(cic, cic_raw)
    same_seq = bool((cic.label.values == unsw.label.values).all())

    manifest = {
        "sha256": {"unsw_aligned_5k.csv": sha256(unsw_p), "cic_aligned_5k.csv": sha256(cic_p)},
        "sources": {
            "unsw": {"file": "data/real_flows.csv", "sha256": sha256(Path(a.unsw_real)),
                     "mapping": "12 columns copied as-is; sinpkt,dinpkt divided by 1000 (ms->s)"},
            "cic": {"file": Path(a.cic_raw).name, "sha256": sha256(Path(a.cic_raw)),
                    "mapping": {
                        "dur": "Flow Duration / 1e6", "spkts": "Total Fwd Packets",
                        "dpkts": "Total Backward Packets", "sbytes": "Total Length of Fwd Packets",
                        "dbytes": "Total Length of Bwd Packets",
                        "rate": f"(spkts+dpkts)/max(dur,{RATE_FLOOR_S})",
                        "smean": "Fwd Packet Length Mean", "dmean": "Bwd Packet Length Mean",
                        "sinpkt": "Fwd IAT Mean / 1e6", "dinpkt": "Bwd IAT Mean / 1e6",
                        "swin": "clip(Init_Win_bytes_forward*255/65535, 0, 255)",
                        "dwin": "clip(Init_Win_bytes_backward*255/65535, 0, 255)",
                        "label": "Label != BENIGN"}},
        },
        "unsw_verification": u,
        "cic_verification": c,
        "cic_label_sequence_identical_to_unsw_aligned": same_seq,
        "extraction_code": "scripts/verify_aligned_data.py documents and verifies the mapping; "
                           "the original sampling code that produced the 5k CIC subset was not retained.",
        "semantic_findings": [
            "UNSW rate = (spkts+dpkts-1)/dur (Argus); CIC rate here = (spkts+dpkts)/max(dur,100us). Definitions differ.",
            "UNSW swin/dwin are quantised TCP window values (observed {0,172,255}); CIC swin/dwin are "
            "Init_Win_bytes rescaled by 255/65535 and clipped, i.e. continuous. Not the same quantity.",
            "UNSW sinpkt/dinpkt are interarrival means in ms (converted to s); CIC uses Fwd/Bwd IAT Mean, "
            "which are not Argus source/destination interpacket arrival times.",
            "CIC label sequence equals the UNSW aligned label sequence row-for-row (3403 attack/1597 benign): "
            "the CIC subset was drawn to mirror the UNSW class order. Features still match raw rows with "
            "agreeing labels, so this is matched sampling, not label copying, but class prior is NOT natural.",
        ],
        "field_semantics_verified": False,
    }
    out = REPO / "data" / "aligned_data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("unsw_verification", "cic_verification")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
