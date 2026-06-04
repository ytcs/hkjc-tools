"""Is form better featurized as CONDITIONAL ability (suitability + recency) than a scalar career mean?

The current `form` averages a horse's relative-finish over ALL past races, blind to (a) whether
today's DISTANCE suits it (a 1200m sprinter is not the same horse at 2000m) and (b) RECENCY
(form cycles with fitness/age; a career mean is stale). The market prices both heavily. Both are
pure featurizations of the same public results and mechanistically couplable (effective v_c rises
at a horse's preferred trip), not tacked-on covariates.

We A/B several form featurizations in the same walk-forward conditional logit as signal_eval_rich:
  career-mean (baseline) | last-3 | exp-weighted | distance-conditional | combos.
A variant earns ABM coupling only if it adds OOS skill on top of (or beats) the career mean.

Run: uv run python experiments/conditional_form.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")
SIGMA_D = 150.0  # m: distance-similarity kernel width


def _dist_conditional(dates, dist, rf, sigma=SIGMA_D):
    """For each race, Gaussian-distance-weighted mean of STRICTLY-PRIOR rf (leak-free). 0 if none."""
    n = len(rf)
    out = np.zeros(n)
    for i in range(n):
        if i == 0:
            continue
        w = np.exp(-((dist[:i] - dist[i]) ** 2) / (2 * sigma ** 2))
        sw = w.sum()
        out[i] = (w @ rf[:i]) / sw if sw > 1e-9 else 0.0
    return out


def build() -> pl.DataFrame:
    d = (pl.read_parquet(CACHE / "results_rich.parquet")
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("Pla").cast(pl.Int64, strict=False),
                       pl.col("WinOdds").cast(pl.Float64, strict=False),
                       pl.col("RaceDistance").cast(pl.Int64, strict=False))
         .drop_nulls(["Pla", "WinOdds"]))
    rid = ["date", "Venue", "RaceNumber"]
    d = d.with_columns(
        pl.len().over(rid).alias("field_size"),
        (pl.col("Pla") == 1).cast(pl.Int8).alias("won"),
        (1.0 / pl.col("WinOdds")).alias("mraw"))
    d = d.with_columns(
        (pl.col("mraw") / pl.col("mraw").sum().over(rid)).alias("market_win_prob"),
        (1.0 - (pl.col("Pla") - 1) / (pl.col("field_size") - 1)).alias("rf"),
        (pl.col("date") + "_" + pl.col("Venue") + "_" + pl.col("RaceNumber").cast(pl.Utf8)).alias("race_id"))

    d = d.sort(["HorseID", "d"])

    def trail_mean(col, key):
        return ((pl.col(col).cum_sum().over(key) - pl.col(col))
                / (pl.col(col).cum_count().over(key) - 1))

    # career mean (baseline) + recency: last-3 rolling mean (shifted), exp-weighted (shifted)
    d = d.with_columns(
        trail_mean("rf", "HorseID").alias("form"),
        pl.col("rf").shift(1).rolling_mean(3, min_samples=1).over("HorseID").alias("form_l3"),
        pl.col("rf").shift(1).ewm_mean(half_life=2.0, ignore_nulls=True).over("HorseID").alias("form_ew"))

    # distance-conditional: per-horse Gaussian-weighted prior rf vs today's distance
    parts = []
    for _, g in d.group_by("HorseID", maintain_order=True):
        g = g.sort("d")
        dd = _dist_conditional(g["d"].to_numpy(), g["RaceDistance"].to_numpy().astype(float),
                               g["rf"].to_numpy().astype(float))
        parts.append(g.with_columns(pl.Series("form_dist", dd)))
    d = pl.concat(parts)

    d = d.with_columns(pl.col(c).fill_nan(None) for c in ["form", "form_l3", "form_ew", "form_dist"])
    d = d.with_columns(pl.col(c).fill_null(0.5) for c in ["form", "form_l3", "form_ew", "form_dist"])
    for c in ["form", "form_l3", "form_ew", "form_dist"]:
        s = d[c].std()
        d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))
    return d


def main():
    d = build()
    dates = sorted(d["date"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races, {len(dates)} meetings, "
          f"{len(folds)} folds  (sigma_d={SIGMA_D})\n")

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
    line("form career-mean (baseline)", skill(["z_form"]))
    line("form last-3", skill(["z_form_l3"]))
    line("form exp-weighted", skill(["z_form_ew"]))
    line("form distance-conditional", skill(["z_form_dist"]))
    line("career + last-3", skill(["z_form", "z_form_l3"]))
    line("career + dist-conditional", skill(["z_form", "z_form_dist"]))
    line("career + l3 + ew + dist", skill(["z_form", "z_form_l3", "z_form_ew", "z_form_dist"]))
    line("MARKET (ceiling)", market)


if __name__ == "__main__":
    main()
