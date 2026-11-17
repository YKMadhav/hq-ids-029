"""Run on YOUR machine on the downloaded CSE-CIC-IDS2018 'Processed Traffic Data for ML Algorithms' files.
Streams each file in chunks (no need to load 6 GB), drops repeated header rows / inf / NaN, keeps the common numeric
features + Timestamp + Label, samples N rows per file (half attack, half benign when available) in TIME ORDER, and writes
ONE small CSV you can upload (default ~3,000 rows per file, about 20 MB for 10 files).
Usage: python scripts/make_cic2018_sample.py cic-ids2018-csvs/ cic2018_sample.csv [rows_per_file]"""
import sys, glob, os, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
ID_COLS = {'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port'}
rng = np.random.RandomState(0); parts = []
for f in sorted(glob.glob(os.path.join(src, '*.csv'))):
    if os.path.abspath(f) == os.path.abspath(out): continue  # never read our own output
    day = os.path.basename(f).split('_')[0]; keep_b, keep_a = [], []
    for ch in pd.read_csv(f, chunksize=200_000, low_memory=False):
        ch = ch[ch['Label'] != 'Label']
        ch['Timestamp'] = pd.to_datetime(ch['Timestamp'], errors='coerce', dayfirst=True); ch = ch.dropna(subset=['Timestamp'])
        num = [c for c in ch.columns if c not in ID_COLS | {'Timestamp', 'Label'}]
        ch[num] = ch[num].apply(pd.to_numeric, errors='coerce'); ch = ch.replace([np.inf, -np.inf], np.nan).dropna()
        isb = ch['Label'] == 'Benign'
        for part, store in ((ch[isb], keep_b), (ch[~isb], keep_a)):  # cap candidates per chunk, keeps memory flat
            if len(part): store.append(part.sample(n=min(len(part), 4000), random_state=0)[['Timestamp', 'Label'] + num])
    b = pd.concat(keep_b) if keep_b else pd.DataFrame(); a = pd.concat(keep_a) if keep_a else pd.DataFrame()
    nb, na = (N // 2, N // 2) if len(a) else (N, 0)
    pick = [x.sample(min(len(x), n), random_state=0) for x, n in ((b, nb), (a, na)) if len(x)]
    d = pd.concat(pick).sort_values('Timestamp'); d.insert(0, 'day', day); parts.append(d); print(day, len(d), d.Label.value_counts().to_dict())
al = pd.concat(parts); common = [c for c in al.columns if al[c].notna().all()]
al[common].to_csv(out, index=False, float_format='%.6g'); print('wrote', out, al[common].shape, round(os.path.getsize(out) / 1e6, 1), 'MB')
