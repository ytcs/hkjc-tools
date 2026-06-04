"""Does a MARGIN-based speed-figure form beat the coarse ordinal (relative-finish) form?

Ordinal form throws away margin: a 12-horse blanket finish and a strung-out 20-length rout
map to identical ranks. FinishTime (per-horse seconds) lets us build a real speed figure —
speed = distance/time, standardized leak-free within each (venue, distance-band, going) cohort
so a slow track or a sprint doesn't masquerade as ability, then trailing-averaged per horse.

We A/B it against the ordinal form in the same walk-forward conditional logit as signal_eval_rich:
  form_ordinal   (baseline)   vs   form_sf   vs   both.
A win for form_sf means margin information is real skill the ABM's FORM_GAIN should ingest.

Run: uv run python experiments/speed_figure_form.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.calibration import distance_band
from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")


def build() -> pl.DataFrame:
    d = (pl.read_parquet(CACHE / "results_rich.parquet")
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("Pla").cast(pl.Int64, strict=False),
                       pl.col("WinOdds").cast(pl.Float64, strict=False),
                       pl.col("FinishTime").cast(pl.Float64, strict=False),
                       pl.col("RaceDistance").cast(pl.Int64, strict=False))
         .drop_nulls(["Pla", "WinOdds"]))
    rid = ["date", "Venue", "RaceNumber"]
    band = {dd: distance_band(dd) for dd in d["RaceDistance"].drop_nulls().unique().to_list()}
    d = d.with_columns(
        pl.len().over(rid).alias("field_size"),
        (pl.col("Pla") == 1).cast(pl.Int8).alias("won"),
        (1.0 / pl.col("WinOdds")).alias("mraw"),
        pl.col("RaceDistance").replace_strict(band, default=None).alias("dband"),
        (pl.col("RaceDistance") / pl.col("FinishTime")).alias("speed"))
    d = d.with_columns(
        (pl.col("mraw") / pl.col("mraw").sum().over(rid)).alias("market_win_prob"),
        (1.0 - (pl.col("Pla") - 1) / (pl.col("field_size") - 1)).alias("rf"),
        (pl.col("date") + "_" + pl.col("Venue") + "_" + pl.col("RaceNumber").cast(pl.Utf8)).alias("race_id"))

    # leak-free EXPANDING per-cohort speed standardization (no current race in its own stats).
    # First rows of a cohort have count<2 -> variance 0/NaN; guard, else cum_sum below propagates
    # the NaN forward and poisons the whole trailing column. Undefined cohort -> neutral 0.
    cohort = ["Venue", "dband", "Going"]
    d = d.sort("d")
    csum = pl.col("speed").cum_sum().over(cohort) - pl.col("speed")
    ccnt = pl.col("speed").cum_count().over(cohort) - 1
    cmean = csum / ccnt
    csq = (pl.col("speed") ** 2).cum_sum().over(cohort) - pl.col("speed") ** 2
    cvar = (csq / ccnt) - cmean ** 2
    cstd = pl.when(cvar > 1e-9).then(cvar.sqrt()).otherwise(None)
    sf = pl.when(ccnt >= 2).then((pl.col("speed") - cmean) / cstd).otherwise(None)
    d = d.with_columns(sf.alias("sf_perf"))
    d = d.with_columns(pl.col("sf_perf").fill_nan(None).fill_null(0.0))

    # WITHIN-RACE standardized time margin (faster than the race's own mean = higher). Pace- and
    # going-immune by construction (every runner shared them), unlike the cohort figure above.
    rt_mean = pl.col("FinishTime").mean().over(rid)
    rt_std = pl.col("FinishTime").std().over(rid)
    d = d.with_columns(
        pl.when(rt_std > 1e-9).then(-(pl.col("FinishTime") - rt_mean) / rt_std)
        .otherwise(0.0).alias("margin_perf"))

    def trail_mean(col, key):
        return ((pl.col(col).cum_sum().over(key) - pl.col(col))
                / (pl.col(col).cum_count().over(key) - 1))

    d = d.sort(["HorseID", "d"]).with_columns(
        trail_mean("rf", "HorseID").alias("form"),
        trail_mean("sf_perf", "HorseID").alias("form_sf"),
        trail_mean("margin_perf", "HorseID").alias("form_margin"))
    # clear 0/0 NaN (first run) before fill
    d = d.with_columns(pl.col(c).fill_nan(None) for c in ["form", "form_sf", "form_margin"])
    d = d.with_columns(pl.col("form").fill_null(0.5),
                       pl.col("form_sf").fill_null(0.0),
                       pl.col("form_margin").fill_null(0.0))
    for c in ["form", "form_sf", "form_margin"]:
        s = d[c].std()
        d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))
    return d


def main():
    d = build()
    nft = d["FinishTime"].null_count()
    dates = sorted(d["date"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races, {len(dates)} meetings, "
          f"{len(folds)} folds  (FinishTime nulls {nft})\n")

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(d, folds, fp, date_col="date", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    d2 = d.with_columns((1.0 / pl.col("field_size")).alias("uni"))
    floor = harness.aggregate(harness.walk_forward_evaluate(
        d2, folds, lambda tr, te: te.with_columns(pl.col("uni").alias("q")),
        date_col="date", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")
    market = harness.aggregate(harness.walk_forward_evaluate(
        d, folds, lambda tr, te: te.with_columns((pl.col("market_win_prob") /
            pl.col("market_win_prob").sum().over("race_id")).alias("q")),
        date_col="date", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")

    def line(name, ll):
        sk = floor - ll
        share = sk / (floor - market) if floor > market else float("nan")
        print(f"  {name:30s} {ll:>9.4f} {sk:>+9.4f} {share:>8.0%}")

    print(f"  {'model':30s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    line("uniform (floor)", floor)
    line("form ordinal (baseline)", skill(["z_form"]))
    line("form_sf (cohort speed fig)", skill(["z_form_sf"]))
    line("form_margin (within-race)", skill(["z_form_margin"]))
    line("ordinal + margin", skill(["z_form", "z_form_margin"]))
    line("ordinal + sf + margin", skill(["z_form", "z_form_sf", "z_form_margin"]))
    line("MARKET (ceiling)", market)


if __name__ == "__main__":
    main()
