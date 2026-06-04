"""Win vs PLACE backtest at Sha Tin — is the structural model relatively better on place?

P(win) is a knife-edge: 1st and 2nd are split-seconds apart, so a model of genuine
strength is measured against mostly-noise. P(top-3) averages over that — it leans on
whole-field ordering, which is what the energetics/geometry model actually produces.
Thesis: the structural model's *gap to the market* should be smaller on place than on
win. Same leak-free walk-forward as the win backtest (population + abilities from
train only); the market place benchmark is the Plackett-Luce expansion of the market
win probabilities.

Run: uv run --group bayes python experiments/backtest_place_st.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.calibration import distance_band, fit_energetics_population
from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.plackett_luce import PlackettLuceModel
from hkjc.model.traffic_jax import topk_probs, win_probs
from hkjc.simulation import RaceSimulator
from hkjc.track import track_for

from backtest_traffic_st import BEND, N_SAMPLES, _horse_residuals, _surface, load_st

PL_SAMPLES = 8000


def predict_fold(train, test):
    pop = fit_energetics_population(train, rating_col="rtg")
    res = _horse_residuals(train, pop)

    def lam(re, n):
        return re.tau2 / (re.tau2 + re.sigma2 / n) if n > 0 else 0.0

    out = []
    for race_id, g in test.group_by("race_id"):
        g = g.sort("runner")
        dist = int(g["distance"][0])
        band = distance_band(dist)
        rtg = g["rtg"].to_numpy().astype(float)
        vc, wp = [], []
        for i, h in enumerate(g["horse_id"].to_list()):
            n, u_vc, u_w = res.get(h, (0, 0.0, 0.0))
            vc.append(pop.v_c.fixed(band, rtg[i]) + lam(pop.v_c, n) * u_vc)
            wp.append(pop.w_prime.fixed(band, rtg[i]) + lam(pop.w_prime, n) * u_w)
        vc, wp = np.array(vc), np.array(wp)
        draw = g["draw"].to_numpy().astype(float)
        nf = len(vc)
        pp = 3 if nf >= 7 else (2 if nf >= 4 else 1)        # HKJC paying places
        geom = track_for("ST", dist, _surface(dist))
        seed = abs(hash(race_id[0])) % (2**31)
        m_win = win_probs(vc, wp, draw, dist, track=geom, course="A", size=N_SAMPLES,
                          bend_curvature=BEND, seed=seed)
        m_plc = topk_probs(vc, wp, draw, dist, k=pp, track=geom, course="A",
                           size=N_SAMPLES, bend_curvature=BEND, seed=seed)
        # market: Plackett-Luce expansion of the de-vigged win probs -> win + place
        mkt = g["market_win_prob"].to_numpy()
        mkt = mkt / mkt.sum()
        sim = RaceSimulator(PlackettLuceModel(mkt), n_samples=PL_SAMPLES, seed=seed)
        k_plc = np.array([sim.prob(sim.places(i, k=pp)) for i in range(len(mkt))])
        out.append(g.with_columns(
            pl.Series("m_win", m_win), pl.Series("m_plc", m_plc),
            pl.Series("k_win", mkt), pl.Series("k_plc", k_plc)))
    return pl.concat(out)


def main():
    df = load_st()
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    panel = pl.concat([predict_fold(df.filter(pl.col("date").is_in(f.train)),
                                    df.filter(pl.col("date").is_in(f.test)))
                       for f in folds if df.filter(pl.col("date").is_in(f.train)).height])
    panel = panel.with_columns(
        pl.when(pl.col("field_size") >= 7).then(3).when(pl.col("field_size") >= 4).then(2)
        .otherwise(0).alias("paying_places"))
    panel = panel.with_columns(
        (pl.col("m_plc") / pl.col("m_plc").sum().over("race_id") * pl.col("paying_places")).alias("m_plc_n"),
        (pl.col("k_plc") / pl.col("k_plc").sum().over("race_id") * pl.col("paying_places")).alias("k_plc_n"),
        (pl.col("paying_places") / pl.col("field_size")).alias("u_plc"))
    n_races = panel["race_id"].n_unique()
    print(f"ST panel: {panel.height} runners, {n_races} races")

    def binlogloss(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    placed = panel["placed"].to_numpy()
    print("\n=== WIN  (race-win log-loss, lower better) ===")
    for nm, c in (("traffic", "m_win"), ("market", "k_win"), ("uniform", "u_plc")):
        col = panel["m_win"] if c == "m_win" else (panel["k_win"] if c == "k_win" else (1.0 / panel["field_size"]))
        s = panel.with_columns(pl.Series("p", col))
        ll = harness.score_predictions(s, prob_col="p", outcome_col="won", race_col="race_id")["race_win_log_loss"]
        print(f"  {nm:8s} {ll:.4f}")

    print("\n=== PLACE / top-3  (per-runner binary log-loss + Brier, lower better) ===")
    rows = {"traffic": panel["m_plc_n"].to_numpy(), "market": panel["k_plc_n"].to_numpy(),
            "uniform": panel["u_plc"].to_numpy()}
    res = {}
    for nm, p in rows.items():
        p = np.clip(p, 1e-6, 0.999)
        ll = binlogloss(p, placed)
        br = float(((p - placed) ** 2).mean())
        res[nm] = ll
        print(f"  {nm:8s} logloss={ll:.4f}  brier={br:.4f}")
    print("\n  GAP TO MARKET (model − market; smaller = structural model closer):")
    # win gap
    sw = panel.with_columns(pl.Series("p", panel["m_win"]))
    win_t = harness.score_predictions(sw, prob_col="p", outcome_col="won", race_col="race_id")["race_win_log_loss"]
    sw = panel.with_columns(pl.Series("p", panel["k_win"]))
    win_m = harness.score_predictions(sw, prob_col="p", outcome_col="won", race_col="race_id")["race_win_log_loss"]
    win_u = harness.score_predictions(panel.with_columns(pl.Series("p", 1.0 / panel["field_size"])),
                                      prob_col="p", outcome_col="won", race_col="race_id")["race_win_log_loss"]
    print(f"  WIN:   traffic−market = {win_t - win_m:+.4f}   (as fraction of uniform→market skill: "
          f"{(win_t - win_m) / (win_u - win_m):.2f})")
    print(f"  PLACE: traffic−market = {res['traffic'] - res['market']:+.4f}   (fraction of skill: "
          f"{(res['traffic'] - res['market']) / (res['uniform'] - res['market']):.2f})")

    print("\n=== PLACE gap by trip ===")
    p2 = panel.with_columns(pl.when(pl.col("distance") <= 1200).then(pl.lit("sprint"))
                            .when(pl.col("distance") <= 1650).then(pl.lit("mile"))
                            .otherwise(pl.lit("route")).alias("trip"))
    for trip in ("sprint", "mile", "route"):
        sub = p2.filter(pl.col("trip") == trip)
        yy = sub["placed"].to_numpy()
        t = binlogloss(np.clip(sub["m_plc_n"].to_numpy(), 1e-6, .999), yy)
        m = binlogloss(np.clip(sub["k_plc_n"].to_numpy(), 1e-6, .999), yy)
        print(f"  {trip:7s} races={sub['race_id'].n_unique():3d}  traffic={t:.4f} market={m:.4f}  gap={t-m:+.4f}")


if __name__ == "__main__":
    main()
