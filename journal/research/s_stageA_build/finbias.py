"""finbias.py RUNDIR: per dev race, the render metric bias (pipeline._metric_bias's per-boundary median of
(crossing - official) x speed, m) at the finish minus at the last boundary before it, from the video product;
summarised per render style on the tune and check halves (dev_tune.txt)."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data import race_info
from hkjc.data.race import official_boundaries
from hkjc.recon.validate import crossing

root = Path(sys.argv[1])
here = Path(__file__).parent
tune = set(open(here / "dev_tune.txt").read().split())
dev = [ln.split("\t") for ln in open("/home/ytc/Projects/hkjc-tools/journal/research/s_stageA/dev_list.tsv")
       if not ln.startswith("#")]
res = {}
for key, style in [(x[0], x[2].strip()) for x in dev]:
    f = root / key / "paths_video.parquet"
    if not f.exists():
        continue
    d, v, n = key.split("_")
    info = race_info(d, v, int(n))
    bnd, cum = official_boundaries(info)
    P = pl.read_parquet(f)
    D = info.distance
    bs = []
    for j, bd in enumerate(bnd[-2:]):
        jj = len(bnd) - 2 + j
        r = []
        for h, c in cum.items():
            p = P.filter(pl.col("horse_no") == h).sort("t")
            if not len(p) or not np.isfinite(c[jj]):
                continue
            t, s, sd = p["t"].to_numpy(), p["s"].to_numpy(), p["sd_s"].to_numpy()
            tr = crossing(t, D - s, sd, bd)
            if np.isfinite(tr):
                vv = np.interp(tr, t[1:], np.diff(s) / np.diff(t))
                r.append((tr - c[jj]) * vv)
        bs.append(np.median(r) if len(r) >= 3 else np.nan)
    if np.isfinite(bs).all():
        res[key] = (style, "tune" if key in tune else "check", bs[1] - bs[0])
        print(key, style, res[key][1], f"{bs[0]:+.2f} {bs[1]:+.2f} delta {bs[1] - bs[0]:+.2f} m")
for style in ("silk", "disc"):
    for half in ("tune", "check"):
        x = np.array([r[2] for r in res.values() if r[0] == style and r[1] == half])
        print(f"{style} {half}: n {len(x)} median {np.median(x):+.2f} m mean {x.mean():+.2f} sd {x.std():.2f}")
