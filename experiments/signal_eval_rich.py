"""At-scale signal evaluation: which condition/skill signals beat form, vs the market?

The 140-race form scan was data-starved (couldn't see jockey/trainer). results_rich gives
3197 races / 5 seasons / 86 jockeys (~450 rides each), so trailing rates are well-estimated.
We build leak-free signals and run a walk-forward conditional logit: baseline = form (trailing
relative-finish); then baseline + each signal. Skill = nats/race vs uniform; ceiling = market.
A signal earns mechanistic coupling only if it adds OOS skill on top of form here.

Run: uv run python experiments/signal_eval_rich.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")


def build() -> pl.DataFrame:
    d = (pl.read_parquet(CACHE / "results_rich.parquet")
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("Pla").cast(pl.Int64, strict=False),
                       pl.col("WinOdds").cast(pl.Float64, strict=False))
         .drop_nulls(["Pla", "WinOdds"]))
    rid = ["date", "Venue", "RaceNumber"]
    d = d.with_columns(
        pl.len().over(rid).alias("field_size"),
        (pl.col("Pla") == 1).cast(pl.Int8).alias("won"),
        (1.0 / pl.col("WinOdds")).alias("mraw"),
        pl.col("RunningPosition").str.split(" ").list.first().cast(pl.Float64, strict=False).alias("early"))
    d = d.with_columns(
        (pl.col("mraw") / pl.col("mraw").sum().over(rid)).alias("market_win_prob"),
        (1.0 - (pl.col("Pla") - 1) / (pl.col("field_size") - 1)).alias("rf"),
        ((pl.col("early") - 1) / (pl.col("field_size") - 1)).alias("early_norm"),
        (pl.col("date") + "_" + pl.col("Venue") + "_" + pl.col("RaceNumber").cast(pl.Utf8)).alias("race_id"))

    def trail_mean(col, key):
        return ((pl.col(col).cum_sum().over(key) - pl.col(col))
                / (pl.col(col).cum_count().over(key) - 1))

    d = d.sort(["HorseID", "d"]).with_columns(
        trail_mean("rf", "HorseID").alias("form"),
        trail_mean("early_norm", "HorseID").alias("style_early"),
        (pl.col("DeclarHorseWt") - pl.col("DeclarHorseWt").shift(1).over("HorseID")).alias("bwt_chg"),
        (pl.col("d") - pl.col("d").shift(1).over("HorseID")).dt.total_days().alias("days_since"))
    d = d.sort(["Jockey", "d"]).with_columns(trail_mean("won", "Jockey").alias("jky_sr"))
    d = d.sort(["Trainer", "d"]).with_columns(trail_mean("won", "Trainer").alias("trn_sr"))

    gj = float(d["won"].mean())
    # trail_mean is 0/0 = NaN (not null) on each entity's first run — clear NaN before filling
    d = d.with_columns(pl.col(c).fill_nan(None) for c in ["form", "style_early", "jky_sr", "trn_sr"])
    d = d.with_columns(
        pl.col("form").fill_null(0.5), pl.col("style_early").fill_null(0.5),
        pl.col("jky_sr").fill_null(gj), pl.col("trn_sr").fill_null(gj),
        pl.col("bwt_chg").fill_null(0.0),
        pl.col("days_since").fill_null(pl.col("days_since").median()))
    for c in ["form", "style_early", "jky_sr", "trn_sr", "bwt_chg", "days_since",
              "ActWt", "DeclarHorseWt"]:
        s = d[c].std()
        d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))
    return d


def main():
    d = build()
    dates = sorted(d["date"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races, {len(dates)} meetings, {len(folds)} folds\n")

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(d, folds, fp, date_col="date", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    def mkt(col):
        fm = harness.walk_forward_evaluate(
            d, folds, lambda tr, te: te.with_columns((pl.col(col) /
                pl.col(col).sum().over("race_id")).alias("q")),
            date_col="date", prob_col="q", outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    # uniform floor + market ceiling
    d2 = d.with_columns((1.0 / pl.col("field_size")).alias("uni"))
    floor = harness.aggregate(harness.walk_forward_evaluate(
        d2, folds, lambda tr, te: te.with_columns(pl.col("uni").alias("q")),
        date_col="date", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")
    market = mkt("market_win_prob")
    base = skill(["z_form"])

    def line(name, ll):
        sk = floor - ll
        share = sk / (floor - market) if floor > market else float("nan")
        print(f"  {name:26s} {ll:>9.4f} {sk:>+9.4f} {share:>8.0%}")

    print(f"  {'model':26s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    line("uniform (floor)", floor)
    line("form (baseline)", base)
    for name, col in [("+ jockey strike rate", "z_jky_sr"), ("+ trainer strike rate", "z_trn_sr"),
                      ("+ body weight", "z_DeclarHorseWt"), ("+ body-weight change", "z_bwt_chg"),
                      ("+ carried weight", "z_ActWt"), ("+ days since", "z_days_since"),
                      ("+ running style (early)", "z_style_early")]:
        line(name, skill(["z_form", col]))
    line("form + jockey + trainer", skill(["z_form", "z_jky_sr", "z_trn_sr"]))
    line("form + ALL signals", skill(["z_form", "z_jky_sr", "z_trn_sr", "z_DeclarHorseWt",
                                      "z_bwt_chg", "z_ActWt", "z_days_since", "z_style_early"]))
    line("MARKET (ceiling)", market)


if __name__ == "__main__":
    main()
