"""Flexible race-win model: gradient-boosted utility + softmax-over-race (a GBM conditional logit).

The pivot's payoff hinges on STRUCTURE as well as features: the linear conditional logit caps ~43%
of market and is numerically unstable on collinear feature sets. A GBM learns nonlinear effects and
interactions (draw x distance x venue, form x field size, weight x class) the linear model cannot, and
is stable. We train LightGBM with a binary `won` objective, take the raw margin as each horse's utility,
and softmax it WITHIN each race to get calibrated win probs (the choice structure the market shares).
Walk-forward, leak-free trailing features only (no odds). Reports % of market vs the linear baseline.

Run: uv run --with lightgbm python experiments/gbm_model.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.calibration import distance_band
from hkjc.eval.walkforward import walk_forward_folds

CACHE = Path("data_cache")


def build() -> pl.DataFrame:
    d = (pl.read_parquet(CACHE / "results_rich.parquet")
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("Pla").cast(pl.Int64, strict=False),
                       pl.col("WinOdds").cast(pl.Float64, strict=False),
                       pl.col("Dr").cast(pl.Int64, strict=False).alias("draw"),
                       pl.col("ActWt").cast(pl.Float64, strict=False).alias("act_wt"),
                       pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("decl_wt"),
                       pl.col("RaceDistance").cast(pl.Int64, strict=False).alias("dist"))
         .drop_nulls(["Pla", "WinOdds", "draw", "dist"]))
    rid = ["date", "Venue", "RaceNumber"]
    bandmap = {dd: distance_band(dd) for dd in d["dist"].unique().to_list()}
    d = d.with_columns(
        pl.len().over(rid).alias("field_size"),
        (pl.col("Pla") == 1).cast(pl.Int8).alias("won"),
        (1.0 / pl.col("WinOdds")).alias("mraw"),
        pl.col("dist").replace_strict(bandmap, default=0).alias("dband"),
        (pl.col("Venue") == "HV").cast(pl.Int8).alias("is_hv"),
        pl.col("RunningPosition").str.split(" ").list.first().cast(pl.Float64, strict=False).alias("early"))
    d = d.with_columns(
        (pl.col("mraw") / pl.col("mraw").sum().over(rid)).alias("market_win_prob"),
        (1.0 - (pl.col("Pla") - 1) / (pl.col("field_size") - 1)).alias("rf"),
        ((pl.col("early") - 1) / (pl.col("field_size") - 1)).alias("early_norm"),
        (pl.col("draw").cast(pl.Float64) / pl.col("field_size")).alias("draw_norm"),
        (pl.col("date") + "_" + pl.col("Venue") + "_" + pl.col("RaceNumber").cast(pl.Utf8)).alias("race_id"))

    def trail_mean(col, key):
        return ((pl.col(col).cum_sum().over(key) - pl.col(col))
                / (pl.col(col).cum_count().over(key) - 1))

    d = d.sort(["HorseID", "d"]).with_columns(
        pl.col("rf").shift(1).ewm_mean(half_life=2.0, ignore_nulls=True).over("HorseID").alias("form_ew"),
        trail_mean("rf", "HorseID").alias("form_career"),
        trail_mean("early_norm", "HorseID").alias("style_early"),
        (pl.col("decl_wt") - pl.col("decl_wt").shift(1).over("HorseID")).alias("bwt_chg"),
        (pl.col("d") - pl.col("d").shift(1).over("HorseID")).dt.total_days().alias("days_since"),
        pl.col("d").cum_count().over("HorseID").alias("career_runs"))
    d = d.sort(["Jockey", "d"]).with_columns(trail_mean("won", "Jockey").alias("jky_sr"))
    d = d.sort(["Trainer", "d"]).with_columns(trail_mean("won", "Trainer").alias("trn_sr"))

    gj = float(d["won"].mean())
    fills = {"form_ew": 0.5, "form_career": 0.5, "style_early": 0.5, "jky_sr": gj, "trn_sr": gj,
             "bwt_chg": 0.0, "days_since": 30.0}
    for c, v in fills.items():
        d = d.with_columns(pl.col(c).fill_nan(None).fill_null(v))
    return d


import os
# Full kitchen-sink set OVERFIT (40% < linear logit's 43%): the GBM split on weak features
# (weight/days/trainer/style) that add ~0 OOS. LEAN keeps only the signals that survived
# walk-forward, letting the GBM test whether INTERACTIONS among them (draw x dist x venue,
# form x field) beat the linear model. Toggle with GBM_FEATURES=full.
LEAN = ["form_ew", "jky_sr", "draw", "draw_norm", "field_size", "dist", "is_hv"]
FULL = ["form_ew", "form_career", "jky_sr", "trn_sr", "draw", "draw_norm", "field_size",
        "dist", "is_hv", "act_wt", "decl_wt", "bwt_chg", "days_since", "style_early", "career_runs"]
FEATURES = FULL if os.environ.get("GBM_FEATURES") == "full" else LEAN


def softmax_by_race(df: pl.DataFrame, margin: np.ndarray) -> pl.DataFrame:
    df = df.with_columns(pl.Series("m", margin))
    return df.with_columns(
        (pl.col("m") - pl.col("m").max().over("race_id")).exp().alias("_e")
    ).with_columns((pl.col("_e") / pl.col("_e").sum().over("race_id")).alias("p"))


def main():
    import lightgbm as lgb

    d = build()
    dates = sorted(d["date"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races, {len(folds)} folds, "
          f"{len(FEATURES)} features\n")

    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, max_depth=5,
                  lambda_l2=1.0, verbose=-1)

    def race_loss(pred):
        out = {}
        for rid, g in pred.group_by("race_id"):
            p = np.clip(g["p"].to_numpy().astype(float), 1e-12, None); p = p / p.sum()
            won = g["won"].to_numpy().astype(bool)
            if won.any():
                out[rid[0]] = -np.log(p[won][0])
        return out

    gbm_loss, mkt_loss, uni_loss = {}, {}, {}
    for fold in folds:
        tr = d.filter(pl.col("date").is_in(fold.train))
        te = d.filter(pl.col("date").is_in(fold.test))
        if te.is_empty():
            continue
        dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), label=tr["won"].to_numpy())
        booster = lgb.train(params, dtr, num_boost_round=300)
        margin = booster.predict(te.select(FEATURES).to_numpy(), raw_score=True)
        pred = softmax_by_race(te, margin)
        gbm_loss.update(race_loss(pred))
        for rid, g in te.group_by("race_id"):
            won = g["won"].to_numpy().astype(bool)
            if won.any():
                m = g["market_win_prob"].to_numpy().astype(float); m = np.clip(m / m.sum(), 1e-12, 1)
                mkt_loss[rid[0]] = -np.log(m[won][0]); uni_loss[rid[0]] = np.log(len(won))
    ks = sorted(set(gbm_loss) & set(mkt_loss))
    g = np.array([gbm_loss[k] for k in ks]); m = np.array([mkt_loss[k] for k in ks])
    u = np.array([uni_loss[k] for k in ks])
    floor, market, gbm = u.mean(), m.mean(), g.mean()
    print(f"  {'model':22s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    for nm, ll in [("uniform (floor)", floor), ("GBM (softmax/race)", gbm), ("MARKET (ceiling)", market)]:
        print(f"  {nm:22s} {ll:>9.4f} {floor-ll:>+9.4f} {(floor-ll)/(floor-market):>8.0%}")

    imp = sorted(zip(FEATURES, booster.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])
    print("\n  top feature gains (last fold):")
    for nm, v in imp[:10]:
        print(f"    {nm:14s} {v:>10.0f}")


if __name__ == "__main__":
    main()
