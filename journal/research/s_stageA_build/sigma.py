"""sigma.py: the official interior boundary residuals over the 60 dev races' U video paths, for OFFICIAL_INTERIOR_SD.
Per race: the U cell's paths_video.parquet from the first source that has it (in order: runs/p1after = 9a94af5,
runs/p1before = 6e7faff, runs/tune3048 = 3048af2, runs/dev9460 = 9460adb, runs/main = main). Per interior boundary:
e = video s at the horse's official time minus the boundary's s; r = e - the boundary's median e. Class: 400 m to go
vs the others. Robust sd s0 = 1.4826 MAD of r (pooled per class); residuals beyond 3 s0 are excluded and the robust
sd of the rest is the value; the excluded share is reported."""
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data import race_info
from hkjc.data.race import official_boundaries

R = Path("/home/ytc/Projects/hkjc-tools/data_cache/s_stageA/runs")
SRC = [("p1after", "9a94af5"), ("p1before", "6e7faff"), ("tune3048", "3048af2"), ("dev9460", "9460adb"),
       ("main", "main")]
dev = [ln.split("\t")[0] for ln in open("/home/ytc/Projects/hkjc-tools/journal/research/s_stageA/dev_list.tsv")
       if not ln.startswith("#")]
res = {"400": [], "other": []}
used = {}
for key in dev:
    src = next(((d, c) for d, c in SRC if (R / d / "U" / key / "paths_video.parquet").exists()), None)
    if src is None:
        continue
    used[key] = src[1]
    d, v, n = key.split("_")
    info = race_info(d, v, int(n))
    bnd, cum = official_boundaries(info)
    D = info.distance
    P = pl.read_parquet(R / src[0] / "U" / key / "paths_video.parquet")
    G = {h: g.sort("t") for (h,), g in P.group_by("horse_no")}
    for j, bd in enumerate(bnd):
        if bd <= 0:
            continue
        e = [float(np.interp(c[j], G[h]["t"].to_numpy(), G[h]["s"].to_numpy())) - (D - bd)
             for h, c in cum.items() if h in G and np.isfinite(c[j])]
        if len(e) < 3:
            continue
        e = np.array(e)
        res["400" if abs(bd - 400) < 1 else "other"].extend((e - np.median(e)).tolist())
codes = {}
for k, c in used.items():
    codes.setdefault(c, []).append(k)
print(f"races: {len(used)} of {len(dev)}; by code: " + ", ".join(f"{c} {len(ks)}" for c, ks in codes.items()))
for cls, r in res.items():
    r = np.array(r)
    s0 = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) <= 3 * s0
    s1 = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
    print(f"class {cls}: n {len(r)}, robust sd all {s0:.3f} m; beyond 3 sd {100 * (~keep).mean():.1f} % "
          f"({int((~keep).sum())}); robust sd of the rest {s1:.3f} m; plain sd of the rest {r[keep].std():.3f} m")
