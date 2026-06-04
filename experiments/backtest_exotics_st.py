"""Exotic-pool backtest at Sha Tin — does the structural model beat the market where
the crowd is weakest (quinella / trio)?

Two tests, both leak-free walk-forward (abilities from train only):
1. PREDICTION (proper score): per race, the model's vs market's probability of the
   *realised* quinella (top-2 set) and trio (top-3 set), from each model's simulated
   finishing-order distribution. Mean −log panel — does the structural model predict the
   actual exotic outcome better than the win-market expansion?
2. VALUE / ROI: bet each model's top-k combinations, settle with the **actual declared
   dividends**. Does model-driven exotic selection pay better than market-driven?

Market orders = Plackett-Luce sim of the de-vigged win probs. Needs the dividend
cache (experiments/build_dividends_st.py).

Run: uv run --group bayes python experiments/backtest_exotics_st.py
"""
from __future__ import annotations

from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.calibration import distance_band, fit_energetics_population
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.plackett_luce import PlackettLuceModel
from hkjc.model.traffic_jax import sample_orders_jax
from hkjc.simulation import RaceSimulator
from hkjc.track import track_for

from backtest_traffic_st import BEND, _horse_residuals, _surface, load_st

CACHE = Path("data_cache")
N = 6000


def _topset_counts(orders, k):
    """Counter of frozenset(top-k) over the order samples (orders: (size, n) indices)."""
    return Counter(frozenset(row[:k].tolist()) for row in orders)


def _prob(counter, key, size, n, k):
    return (counter.get(key, 0) + 1.0 / comb(n, k)) / (size + 1.0)


def predict_fold(train, test, div):
    pop = fit_energetics_population(train, rating_col="rtg")
    res = _horse_residuals(train, pop)

    def lam(re, n):
        return re.tau2 / (re.tau2 + re.sigma2 / n) if n > 0 else 0.0

    rows = []
    for race_id, g in test.group_by("race_id"):
        g = g.sort("runner")
        n = g.height
        if n < 6:
            continue
        dist = int(g["distance"][0])
        band = distance_band(dist)
        rtg = g["rtg"].to_numpy().astype(float)
        runners = g["runner"].to_numpy()
        vc = np.array([pop.v_c.fixed(band, rtg[i]) + lam(pop.v_c, res.get(h, (0, 0, 0))[0]) * res.get(h, (0, 0, 0))[1]
                       for i, h in enumerate(g["horse_id"].to_list())])
        wp = np.array([pop.w_prime.fixed(band, rtg[i]) + lam(pop.w_prime, res.get(h, (0, 0, 0))[0]) * res.get(h, (0, 0, 0))[2]
                       for i, h in enumerate(g["horse_id"].to_list())])
        draw = g["draw"].to_numpy().astype(float)
        geom = track_for("ST", dist, _surface(dist))
        seed = abs(hash(race_id[0])) % (2**31)
        mo = sample_orders_jax(vc, wp, draw, dist, track=geom, course="A", size=N,
                               bend_curvature=BEND, seed=seed)
        ko = RaceSimulator(PlackettLuceModel(g["market_win_prob"].to_numpy() /
                                             g["market_win_prob"].sum()), n_samples=N, seed=seed).orders
        # realised top-2/top-3 as index sets (into the runner-sorted g)
        fp = g["finish_pos"].to_numpy()
        order_idx = np.argsort(fp)                       # idx of 1st,2nd,3rd,...
        if np.any(np.isnan(fp.astype(float))) or fp.min() < 1:
            continue
        real2 = frozenset(order_idx[:2].tolist())
        real3 = frozenset(order_idx[:3].tolist())
        def rkey(idx_set):                               # canonical runner-number string
            return ",".join(map(str, sorted(int(runners[i]) for i in idx_set)))

        rec = {"race_id": race_id[0], "n": n}
        for tag, k, real in (("q2", 2, real2), ("q3", 3, real3)):
            mc, kc = _topset_counts(mo, k), _topset_counts(ko, k)
            rec[f"m_{tag}"] = _prob(mc, real, N, n, k)       # P(realised) — index-set lookup
            rec[f"k_{tag}"] = _prob(kc, real, N, n, k)
            rec[f"u_{tag}"] = 1.0 / comb(n, k)
            rec[f"m_pick_{tag}"] = rkey(mc.most_common(1)[0][0])   # top-1 pick as string
            rec[f"k_pick_{tag}"] = rkey(kc.most_common(1)[0][0])
            rec[f"real_{tag}"] = rkey(real)
        # dividends for this race
        d = div.filter(pl.col("race_id") == race_id[0])
        rec["qin"] = _div(d, "QIN")
        rec["tri"] = _div(d, "TRI")
        rows.append(rec)
    return rows


def _div(d, pool):
    r = d.filter(pl.col("pool") == pool)
    return float(r["decimal"][0]) if r.height else None


def main():
    df = load_st()
    div = (pl.read_parquet(CACHE / "dividends_st.parquet")
           .with_columns((pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id")))
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    rows = []
    for f in folds:
        tr, te = df.filter(pl.col("date").is_in(f.train)), df.filter(pl.col("date").is_in(f.test))
        if tr.height and te.height:
            rows += predict_fold(tr, te, div)
    panel = pl.DataFrame(rows)
    print(f"exotic panel: {panel.height} races")

    print("\n=== PREDICTION — mean −log P(realised exotic), lower better ===")
    for tag, name in (("q2", "QUINELLA (top-2)"), ("q3", "TRIO (top-3)")):
        mm = float(-np.log(panel[f"m_{tag}"].to_numpy()).mean())
        kk = float(-np.log(panel[f"k_{tag}"].to_numpy()).mean())
        uu = float(-np.log(panel[f"u_{tag}"].to_numpy()).mean())
        share = (uu - mm) / (uu - kk) if uu > kk else float("nan")
        print(f"  {name:18s} traffic={mm:.3f}  market={kk:.3f}  uniform={uu:.3f}  "
              f"| traffic captures {share:.0%} of market skill")

    print("\n=== VALUE — ROI of betting the top-1 combination, settled on real dividends ===")
    for tag, pool, name in (("q2", "qin", "QUINELLA"), ("q3", "tri", "TRIO")):
        sub = panel.filter(pl.col(pool).is_not_null())
        for who in ("m", "k"):
            stake = sub.height
            ret = sub.filter(pl.col(f"{who}_pick_{tag}") == pl.col(f"real_{tag}"))[pool].sum()
            hit = (sub[f"{who}_pick_{tag}"] == sub[f"real_{tag}"]).sum()
            roi = (ret - stake) / stake if stake else float("nan")
            label = "traffic" if who == "m" else "market"
            print(f"  {name:9s} {label:7s}: bets={stake} hits={hit} "
                  f"({hit/stake:.1%})  return={ret:.0f}  ROI={roi:+.1%}")


if __name__ == "__main__":
    main()
