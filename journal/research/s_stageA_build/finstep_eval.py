"""finstep_eval.py: on main's F cells (finish withheld), the finish |dt| as run and with the final product's s moved
by FINISH_STEP at the line (dt - step / v); pooled median and p90 per half of the dev list."""
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data import race_info
from hkjc.data.race import official_boundaries
from hkjc.recon.validate import crossing

STEP = {"silk": 1.07, "disc": -0.50}
root = Path("/home/ytc/Projects/hkjc-tools/data_cache/s_stageA/runs/main/F")
tune = set(open(Path(__file__).parent / "dev_tune.txt").read().split())
dev = [ln.split("\t") for ln in open("/home/ytc/Projects/hkjc-tools/journal/research/s_stageA/dev_list.tsv")
       if not ln.startswith("#")]
acc = {}
for key, style in [(x[0], x[2].strip()) for x in dev]:
    f = root / key / "paths.parquet"
    if not f.exists():
        continue
    d, v, n = key.split("_")
    info = race_info(d, v, int(n))
    bnd, cum = official_boundaries(info)
    P = pl.read_parquet(f)
    D = info.distance
    for h, c in cum.items():
        p = P.filter(pl.col("horse_no") == h).sort("t")
        if not len(p) or not np.isfinite(c[-1]):
            continue
        t, s = p["t"].to_numpy(), p["s"].to_numpy()
        tr = crossing(t, D - s, np.zeros_like(s), 0.0)
        if not np.isfinite(tr):
            continue
        vv = np.interp(tr, t[1:], np.diff(s) / np.diff(t))
        half = "tune" if key in tune else "check"
        acc.setdefault((style, half), []).append((abs(tr - c[-1]), abs(tr - c[-1] - STEP[style] / vv)))
for k, x in sorted(acc.items()):
    x = np.array(x)
    print(k, len(x), "median |dt| %.3f -> %.3f   p90 %.3f -> %.3f" % (
        np.median(x[:, 0]), np.median(x[:, 1]), np.percentile(x[:, 0], 90), np.percentile(x[:, 1], 90)))
