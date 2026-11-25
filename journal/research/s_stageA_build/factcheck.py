"""factcheck.py KEY T0 T1 PATHS...: arbitration of labellings by the Stage C TV/patrol facts (render-independent).
Lateral patrol facts in [T0, T1] whose rail sense is determined (|p_sense - 0.5| >= 0.3) and whose two tracks are each
identified with P >= 0.8 (softmax of the track's silk log-likelihood over the race's runners, in runner order) state
"horse a nearer the rail than horse b" with probability p_sense; along TV facts state "a ahead of b" (conf).
For each labelling's paths file (horse_no, t, s, lat), the share of such facts its paths agree with (lat smaller =
nearer the rail; s larger = ahead), each weighted by its probability."""
import sys

import numpy as np
import polars as pl

from hkjc.data import race_info

key, t0, t1, files = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
PAIR = {int(x) for x in __import__("os").environ.get("PAIR", "").split(",") if x}  # only facts among these horses
d, v, n = key.split("_")
nos = [int(r.no) for r in race_info(d, v, int(n)).runners][:14]
F = pl.read_parquet(f"/home/ytc/Projects/hkjc-tools/data_cache/s_stageC/facts/{key}.parquet")
F = F.filter((pl.col("t") >= t0) & (pl.col("t") <= t1))


def ident(ll):
    x = np.asarray(ll, float)[:len(nos)]
    p = np.exp(x - x.max())
    p /= p.sum()
    i = int(p.argmax())
    return nos[i], float(p[i])


facts = []
for r in F.iter_rows(named=True):
    ha, pa = ident(r["ll_a"])
    hb, pb = ident(r["ll_b"])
    if ha == hb or pa < 0.8 or pb < 0.8:
        continue
    if PAIR and {ha, hb} != PAIR and not (len(PAIR) > 2 and ha in PAIR and hb in PAIR):
        continue
    if r["kind"] == "lateral" and abs(r["p_sense"] - 0.5) >= 0.3:
        facts.append(("lat", r["t"], ha, hb, r["p_sense"]))
    elif r["kind"] == "along":
        facts.append(("along", r["t"], ha, hb, r["conf"]))
print(f"{key} {t0}-{t1} s: {len(facts)} usable facts "
      f"({sum(f[0] == 'lat' for f in facts)} lateral, {sum(f[0] == 'along' for f in facts)} along)")
for fn in files:
    P = pl.read_parquet(fn)
    G = {h: g.sort("t") for (h,), g in P.group_by("horse_no")}
    agree = tot = 0.0
    for kind, t, ha, hb, p in facts:
        if ha not in G or hb not in G:
            continue
        col = "lat" if kind == "lat" else "s"
        xa = np.interp(t, G[ha]["t"].to_numpy(), G[ha][col].to_numpy())
        xb = np.interp(t, G[hb]["t"].to_numpy(), G[hb][col].to_numpy())
        claim = (xa < xb) if kind == "lat" else (xa > xb)  # lat: a nearer the rail; along: a ahead
        w = abs(p - 0.5) * 2 if kind == "lat" else p
        sense = p >= 0.5 if kind == "lat" else True
        agree += w * (claim == sense)
        tot += w
    print(f"  {fn.split('/runs/')[-1]}: weighted agreement {agree / max(tot, 1e-9):.3f} over weight {tot:.1f}")
