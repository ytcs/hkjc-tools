"""Scorer check failure follow-up (declared stop rule): is it the scorer, the 40-race sample or the HEAD replay?
(a) 0272 rows (s_score/sub.parquet, sub_flag.py method) restricted to these 40 races; (b) this scorer on the stored
product paths of the same races; (c) this scorer on off_U (HEAD replay)."""
import sys

import numpy as np
import polars as pl

sys.argv = ["score.py", "none"]
src = open("journal/research/s_subobs/score.py").read().rsplit("\n{", 1)[0]
exec(src)  # noqa: S102 - the scorer's functions

S = pl.read_parquet("data_cache/s_score/sub.parquet")
keys = {k: (s, p) for s, k, p, _j, _n in JOBS}
H = []
for (st, k), g in S.group_by(["set", "key"]):
    if k not in keys or keys[k][0] != st:
        continue
    for (h,), gh in g.group_by(["horse"]):
        tt = {r["point"]: (r["t_off"], r["t_c"]) for r in gh.iter_rows(named=True)}
        for a, b in ALLH:
            if a in tt and b in tt and tt[a][1] is not None and tt[b][1] is not None:
                ot, pt = tt[b][0] - tt[a][0], tt[b][1] - tt[a][1]
                if np.isfinite(ot) and np.isfinite(pt) and ot > 5:
                    H.append(dict(key=k, half=f"{a:.0f}-{b:.0f}", ot=ot, dT=pt - ot))
H = pl.DataFrame(H)
m = H.group_by("key", "half").agg(pl.len().alias("n_"), pl.col("dT").median().alias("m"))
H = H.join(m, on=["key", "half"]).filter(pl.col("n_") >= 3).with_columns(frac=(pl.col("dT") - pl.col("m")).abs() / pl.col("ot"))
print(f"(a) 0272 rows, these races: p90 {100 * p90(H['frac'].to_numpy()):.3f} % n {H.height} races {H['key'].n_unique()}")
for lab, fn in (("(b) stored products", lambda k: keys[k][1] + "/paths.parquet"),
                ("(c) off_U replay", lambda k: f"data_cache/s_subobs/runs/off_U/{k}/paths.parquet")):
    rows = []
    for k in RACE:
        P = pl.read_parquet(fn(k), columns=["horse_no", "t", "s"]).sort("horse_no", "t")
        G = {int(h): (g["t"].to_numpy(), g["s"].to_numpy()) for (h,), g in P.group_by(["horse_no"])}
        rows += halves(G, k, ALLH)
    x = pl.DataFrame(rows)
    print(f"{lab}: p90 {100 * p90(x['frac'].to_numpy()):.3f} % n {x.height}; by set",
          x.join(pl.DataFrame([dict(race=k, set=r['set']) for k, r in RACE.items()]), on="race")
          .group_by("set").agg((pl.col("frac").quantile(0.9) * 100).round(3)).rows())
