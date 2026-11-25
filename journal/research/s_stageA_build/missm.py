"""missm.py CELLDIR KEY: each horse's miss of its official interior boundary times under the cell's video product
(paths_video.parquet): e = video s at the official time minus the boundary's s (m); b = the boundary's median e (the
shared render metric bias the final removes); r = e - b, the horse-specific miss the 0.6 m official sd has to absorb."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data import race_info
from hkjc.data.race import official_boundaries

celldir, key = Path(sys.argv[1]), sys.argv[2]
d, v, n = key.split("_")
info = race_info(d, v, int(n))
bnd, cum = official_boundaries(info)
D = info.distance
P = pl.read_parquet(celldir / key / "paths_video.parquet")
rows = {}
for j, bd in enumerate(bnd):
    if bd <= 0:
        continue
    e = {}
    for h, c in cum.items():
        p = P.filter(pl.col("horse_no") == h).sort("t")
        if not len(p) or not np.isfinite(c[j]):
            continue
        e[h] = float(np.interp(c[j], p["t"].to_numpy(), p["s"].to_numpy())) - (D - bd)
    b = float(np.median(list(e.values())))
    for h, x in e.items():
        rows.setdefault(h, {})[bd] = (x, x - b)
    r = np.array([x - b for x in e.values()])
    print(f"{key} {celldir.name} boundary {bd:.0f} m to go (t ~ {np.median([cum[h][j] for h in e]):.1f} s): bias b "
          f"{b:+.2f} m; |r| median {np.median(np.abs(r)):.2f}, max {np.abs(r).max():.2f} m; robust sd "
          f"{1.4826 * np.median(np.abs(r - np.median(r))):.2f} m")
bds = sorted({bd for r in rows.values() for bd in r}, reverse=True)
print("horse " + " ".join(f"r@{bd:.0f}" for bd in bds))
for h in sorted(rows):
    print(f"#{h:<4d} " + " ".join(f"{rows[h][bd][1]:+6.2f}" if bd in rows[h] else "   nan" for bd in bds))
