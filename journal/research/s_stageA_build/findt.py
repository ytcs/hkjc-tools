"""findt.py RUNDIR: per dev race (dev_list.tsv only), the video product's crossing time minus the official time at
each official boundary (median over horses), the finish last. Shows how the render's distance error near the line
differs from the boundary before it."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data import race_info
from hkjc.data.race import official_boundaries
from hkjc.recon.validate import crossing

root = Path(sys.argv[1])
dev = [ln.split("\t") for ln in open("/home/ytc/Projects/hkjc-tools/journal/research/s_stageA/dev_list.tsv")
       if not ln.startswith("#")]
for key, _blk, style in [(x[0], x[1], x[2].strip()) for x in dev]:
    f = root / key / "paths_video.parquet"
    if not f.exists():
        continue
    d, v, n = key.split("_")
    info = race_info(d, v, int(n))
    bnd, cum = official_boundaries(info)
    P = pl.read_parquet(f)
    D = info.distance
    med = []
    for j, bd in enumerate(bnd):
        r = []
        for h, c in cum.items():
            p = P.filter(pl.col("horse_no") == h).sort("t")
            if not len(p) or not np.isfinite(c[j]):
                continue
            t, s, sd = p["t"].to_numpy(), p["s"].to_numpy(), p["sd_s"].to_numpy()
            tr = crossing(t, D - s, sd, bd)
            if np.isfinite(tr):
                r.append(tr - c[j])
        med.append(np.median(r) if len(r) >= 3 else np.nan)
    print(f"{key} {style:4s} D {D:5.0f} bnd {[int(b) for b in bnd]} dt " + " ".join(f"{x:+.2f}" for x in med))
