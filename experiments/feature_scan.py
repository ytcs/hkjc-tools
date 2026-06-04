"""Next orthogonal features beyond EW form + jockey: draw, conditions-suitability, field-pace.

Pivoted to a flexible feature model (it beats the ABM and the ABM adds nothing). signal_eval_rich
already showed weight/body-weight/days-since/trainer/running-style add ~0 beyond form, and rating is
neutralised by the handicap (corr with winning +0.036, 58% null). The UNTESTED orthogonal signals are
DRAW (barrier x venue x distance — not handicap-equalised) and CONDITIONS-SUITABILITY (a horse's edge
at TODAY's distance/going). We test each as a marginal add over the EW-form+jockey baseline (41% of
market on 3248 results_rich races) in the same walk-forward conditional logit, reporting % of market.

Key draw featurization: empirical leak-free trailing win-rate by (venue, distance-band, draw) minus
fair (1/field) — encodes the draw x conditions interaction without a nonlinear model.

Run: uv run python experiments/feature_scan.py
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
                       pl.col("Dr").cast(pl.Int64, strict=False).alias("draw"),
                       pl.col("RaceDistance").cast(pl.Int64, strict=False).alias("dist"))
         .drop_nulls(["Pla", "WinOdds", "draw", "dist"]))
    rid = ["date", "Venue", "RaceNumber"]
    bandmap = {dd: distance_band(dd) for dd in d["dist"].unique().to_list()}
    d = d.with_columns(
        pl.len().over(rid).alias("field_size"),
        (pl.col("Pla") == 1).cast(pl.Int8).alias("won"),
        (1.0 / pl.col("WinOdds")).alias("mraw"),
        pl.col("dist").replace_strict(bandmap, default=None).alias("dband"),
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

    # form (EW) + jockey baseline
    d = d.sort(["HorseID", "d"]).with_columns(
        pl.col("rf").shift(1).ewm_mean(half_life=2.0, ignore_nulls=True).over("HorseID").alias("form_ew"),
        trail_mean("early_norm", "HorseID").alias("style_early"))
    d = d.sort(["Jockey", "d"]).with_columns(trail_mean("won", "Jockey").alias("jky_sr"))

    # DRAW BIAS: leak-free expanding win-rate by (venue, dband, draw), minus fair (1/field)
    d = d.sort("d")
    grp = ["Venue", "dband", "draw"]
    wr = (pl.col("won").cum_sum().over(grp) - pl.col("won")) / (pl.col("won").cum_count().over(grp) - 1)
    d = d.with_columns((wr - 1.0 / pl.col("field_size")).alias("draw_bias"))

    # CONDITIONS-SUITABILITY: distance-matched trailing form (|Δdist|<=100m prior runs), and going-match
    # done per horse in numpy (weights depend on today's distance) — Gaussian distance kernel
    parts = []
    for _, g in d.sort(["HorseID", "d"]).group_by("HorseID", maintain_order=True):
        g = g.sort("d")
        dd = g["dist"].to_numpy().astype(float); rf = g["rf"].to_numpy().astype(float)
        n = len(rf); out = np.full(n, np.nan)
        for i in range(1, n):
            w = np.exp(-((dd[:i] - dd[i]) ** 2) / (2 * 120.0 ** 2))
            sw = w.sum()
            if sw > 1e-9:
                out[i] = (w @ rf[:i]) / sw
        parts.append(g.with_columns(pl.Series("dist_form", out)))
    d = pl.concat(parts)

    gj = float(d["won"].mean())
    for c, fill in [("form_ew", 0.5), ("style_early", 0.5), ("jky_sr", gj), ("dist_form", 0.5),
                    ("draw_bias", 0.0)]:
        d = d.with_columns(pl.col(c).fill_nan(None).fill_null(fill))
    for c in ["form_ew", "jky_sr", "dist_form", "draw_bias", "draw_norm", "style_early"]:
        s = d[c].std()
        d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))
    return d


def main():
    d = build()
    dates = sorted(d["date"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races, {len(folds)} folds\n")

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

    base = ["z_form_ew", "z_jky_sr"]
    print(f"  {'model':34s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    line("EW + jockey (baseline)", skill(base))
    line("  + draw_bias (empirical)", skill(base + ["z_draw_bias"]))
    line("  + draw_norm (raw)", skill(base + ["z_draw_norm"]))
    line("  + dist_form (suitability)", skill(base + ["z_dist_form"]))
    line("  + style_early", skill(base + ["z_style_early"]))
    line("EW+jky + draw_bias + dist_form", skill(base + ["z_draw_bias", "z_dist_form"]))
    line("ALL", skill(base + ["z_draw_bias", "z_draw_norm", "z_dist_form", "z_style_early"]))
    line("MARKET (ceiling)", market)


if __name__ == "__main__":
    main()
