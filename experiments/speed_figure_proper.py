"""Track-variant speed figure (Beyer-style) — the principled version of the failed naive margin form.

The naive within-race time-margin failed because standardising within a race removes the cross-race
CLASS signal (everyone in a slow weak race looks equal). A real speed figure keeps it: raw speed minus
a DISTANCE PAR (expected speed at the trip) minus a DAILY TRACK VARIANT (how fast the whole card ran —
strips going/rail/wind, the step the naive version lacked). What remains is a class- and condition-
neutral performance level that flags well-handicapped horses (fast absolute time in a strong race) that
finishing RANK misses. We trail-EW it per horse and test it as an ADD over EW rank-form + jockey + draw.

Leak-free: a PAST race's variant/par use that race's fully-known results; the per-horse trailing EW
excludes today. Distance par = expanding mean speed by distance.

Run: uv run python experiments/speed_figure_proper.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

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
                       pl.col("Dr").cast(pl.Int64, strict=False).alias("draw"),
                       pl.col("RaceDistance").cast(pl.Int64, strict=False).alias("dist"))
         .drop_nulls(["Pla", "WinOdds", "draw", "dist", "FinishTime"])
         .filter(pl.col("FinishTime") > 0))
    rid = ["date", "Venue", "RaceNumber"]
    d = d.with_columns(
        pl.len().over(rid).alias("field_size"),
        (pl.col("Pla") == 1).cast(pl.Int8).alias("won"),
        (1.0 / pl.col("WinOdds")).alias("mraw"),
        (pl.col("dist") / pl.col("FinishTime")).alias("speed"))
    d = d.with_columns(
        (pl.col("mraw") / pl.col("mraw").sum().over(rid)).alias("market_win_prob"),
        (1.0 - (pl.col("Pla") - 1) / (pl.col("field_size") - 1)).alias("rf"),
        (pl.col("draw").cast(pl.Float64) / pl.col("field_size")).alias("draw_norm"),
        (pl.col("date") + "_" + pl.col("Venue") + "_" + pl.col("RaceNumber").cast(pl.Utf8)).alias("race_id"))

    # distance par (expanding mean speed by exact distance, leak-free over prior runs)
    d = d.sort("d")
    par = (pl.col("speed").cum_sum().over("dist") - pl.col("speed")) / (pl.col("speed").cum_count().over("dist") - 1)
    d = d.with_columns(par.alias("par"))
    d = d.with_columns((pl.col("speed") - pl.col("par")).alias("excess"))
    # daily track variant: mean excess over the whole card (date, venue) — known for any PAST race
    d = d.with_columns(pl.col("excess").mean().over(["date", "Venue"]).alias("variant"))
    d = d.with_columns((pl.col("excess") - pl.col("variant")).alias("sfig"))  # speed figure (m/s above par+variant)

    def trail_mean(col, key):
        return ((pl.col(col).cum_sum().over(key) - pl.col(col))
                / (pl.col(col).cum_count().over(key) - 1))

    d = d.sort(["HorseID", "d"]).with_columns(
        pl.col("rf").shift(1).ewm_mean(half_life=2.0, ignore_nulls=True).over("HorseID").alias("form_ew"),
        pl.col("sfig").shift(1).ewm_mean(half_life=2.0, ignore_nulls=True).over("HorseID").alias("form_speed"),
        trail_mean("sfig", "HorseID").alias("form_speed_career"))
    d = d.sort(["Jockey", "d"]).with_columns(trail_mean("won", "Jockey").alias("jky_sr"))

    gj = float(d["won"].mean())
    for c, v in [("form_ew", 0.5), ("jky_sr", gj), ("form_speed", 0.0), ("form_speed_career", 0.0)]:
        d = d.with_columns(pl.col(c).fill_nan(None).fill_null(v))
    for c in ["form_ew", "jky_sr", "draw_norm", "form_speed", "form_speed_career"]:
        s = d[c].std()
        d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))
    return d


def main():
    d = build()
    folds = walk_forward_folds(sorted(d["date"].unique().to_list()), n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races\n")

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(d, folds, fp, date_col="date", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    floor = harness.aggregate(harness.walk_forward_evaluate(
        d.with_columns((1.0 / pl.col("field_size")).alias("uni")), folds,
        lambda tr, te: te.with_columns(pl.col("uni").alias("q")),
        date_col="date", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")
    market = harness.aggregate(harness.walk_forward_evaluate(
        d, folds, lambda tr, te: te.with_columns((pl.col("market_win_prob") /
            pl.col("market_win_prob").sum().over("race_id")).alias("q")),
        date_col="date", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")

    def line(name, ll):
        sk = floor - ll
        print(f"  {name:34s} {ll:>9.4f} {sk:>+9.4f} {sk/(floor-market):>8.0%}")

    base = ["z_form_ew", "z_jky_sr", "z_draw_norm"]
    print(f"  {'model':34s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    line("EW form + jockey + draw (base)", skill(base))
    line("speed-fig form alone", skill(["z_form_speed"]))
    line("  base + speed-fig (EW)", skill(base + ["z_form_speed"]))
    line("  base + speed-fig (career)", skill(base + ["z_form_speed_career"]))
    line("MARKET (ceiling)", market)


if __name__ == "__main__":
    main()
