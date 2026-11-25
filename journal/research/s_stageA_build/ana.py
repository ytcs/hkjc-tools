"""ana.py: milestone numbers for the re-submission (no races beyond the milestone cells).
- label churn U vs masked cell for the three 0312 cases (core and main, video and final);
- 20260301_ST_2 #10/#11 rows over 12-52 s: median prob and share prob < 0.9, both arms, video and final;
- 20231001_ST_1 at 71-75 s: rank of #2 by product position among all runners, and the list row;
- validation gates of every cell against main's."""
import json
from pathlib import Path

import numpy as np
import polars as pl

W = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
M = Path("/home/ytc/Projects/hkjc-tools/data_cache/s_ms/runs/main")
MAIN = {"20240511_ST_7": "pod_B3g/recon", "20241201_ST_10": "pod_B3g/recon", "20251126_HV_3": "pod_B3g/recon",
        "20241020_ST_9": "pod_B3g/recon", "20231001_ST_1": "product", "20240505_ST_10": "product",
        "20260301_ST_2": "product", "20250226_HV_8": "product", "20260419_ST_4": "product", "20240612_HV_8": "product"}

print("churn U vs masked (rows labelled in either run whose labels differ)")
for k, c in (("20231001_ST_1", "lg1"), ("20240505_ST_10", "lg0"), ("20260301_ST_2", "lg1")):
    out = []
    for name, root in (("core", W), ("main", M)):
        for f in ("det_labels.npz", "det_labels_final.npz"):
            a = np.load(root / "U" / k / f)["labels"]
            b = np.load(root / c / k / f)["labels"]
            m = (a > 0) | (b > 0)
            out.append(f"{name} {f[:-4]} {100 * np.mean(a[m] != b[m]):.1f}%")
    print(" ", k, c, " | ".join(out))

print("20260301_ST_2 rows labelled #10 or #11, 12-52 s")
for c in ("U", "lg1"):
    T = np.load(W / c / "20260301_ST_2/tracks.npz", allow_pickle=True)
    for f in ("det_labels.npz", "det_labels_final.npz"):
        z = np.load(W / c / "20260301_ST_2" / f)
        m = np.isin(z["labels"], [10, 11]) & (T["t"] >= 12) & (T["t"] <= 52)
        p = z["prob"][m]
        print(f"  {c} {f[:-4]}: n {m.sum()}, median prob {np.median(p):.3f}, share < 0.9 {np.mean(p < 0.9):.3f}")

print("20231001_ST_1, #2's rank among runners by product position")
for c in ("U", "lg1"):
    P = pl.read_parquet(W / c / "20231001_ST_1/paths.parquet")
    for t in (71.0, 73.0, 75.0):
        q = P.filter((pl.col("t") - t).abs() < 0.051).sort("s", descending=True)
        hs = q["horse_no"].to_list()
        print(f"  {c} t={t}: order {hs}, #2 rank {hs.index(2) + 1 if 2 in hs else None}")

print("gates (video-only), core vs main")
for d in sorted(p for p in (W / "U").iterdir() if p.is_dir()):
    vj = d / "validation.json"
    mj = Path("/home/ytc/Projects/hkjc-tools/data_cache") / MAIN.get(d.name, "product") / d.name / "validation.json"
    if not vj.exists() or not mj.exists():
        continue
    a, b = json.load(open(vj)), json.load(open(mj))
    keys = ("sectional_mae", "finish_rho", "behind_winner_mae", "sectional_coverage")
    print(f"  {d.name}: " + ", ".join(f"{k} {b.get(k, np.nan):.3f} -> {a.get(k, np.nan):.3f}" for k in keys))
