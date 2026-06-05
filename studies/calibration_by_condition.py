"""Calibration-by-condition — the instrument the residual diagnosis lacks (race-level effects).

The market-residual regression is BLIND to race-level covariates (field size, distance, going,
class): they're constants within a race and vanish under within-race renormalisation. So a
systematic miscalibration by condition is invisible to it. This computes, on full-scale OOS probit
predictions (fast), the model's % of market and a confidence ratio WITHIN buckets of field size /
distance / going — to SEE where the single global speed_noise is wrong before adding any structure
(parsimony: only structure the noise if a bucket is systematically mis-concentrated).

Confidence ratio = mean(model top prob) / (winner-is-favourite rate). >1 => over-confident
(model's favourite too strong vs how often the model's pick wins); <1 => under-confident.

Run: uv run python studies/calibration_by_condition.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load
from hkjc.backtesting import walk_forward_folds
from hkjc.backtesting.scoring import race_win_log_loss
from hkjc.model.fit import fit_predict_traffic

CACHE = Path("data_cache")
SIM = os.environ.get("SIM", "wald")
SIGMA_T = float(os.environ.get("SIGMA_T", "2.5"))


def collect(df, jky_mean) -> pl.DataFrame:
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    parts = []
    for fold in folds:
        tr = df.filter(pl.col("date").is_in(fold.train))
        te = df.filter(pl.col("date").is_in(fold.test))
        if tr.height and te.height:
            parts.append(fit_predict_traffic(tr, te, jky_mean=jky_mean, sim=SIM,
                                             sigma_t=SIGMA_T, n_samples=20000))
    return pl.concat(parts)


def _bucket_report(p, by, label):
    rows = []
    for key, g in p.group_by(by):
        if g["race_id"].n_unique() < 40:
            continue
        gm = g.with_columns(
            (pl.col("p") / pl.col("p").sum().over("race_id")).alias("mp"),
            (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("kp"),
            (1.0 / pl.col("field_size")).alias("up"))
        m = race_win_log_loss(gm, prob_col="mp", winner_col="won", race_col="race_id")
        k = race_win_log_loss(gm, prob_col="kp", winner_col="won", race_col="race_id")
        u = race_win_log_loss(gm, prob_col="up", winner_col="won", race_col="race_id")
        pct = (u - m) / (u - k) if u > k else float("nan")
        # confidence: mean top model prob vs how often the model's top pick wins
        top = gm.group_by("race_id").agg(pl.col("mp").max().alias("topp"),
                                         pl.col("won").filter(pl.col("mp") == pl.col("mp").max()).max().alias("hit"))
        conf = top["topp"].mean() / max(top["hit"].mean(), 1e-9)
        rows.append((key[0] if isinstance(key, tuple) else key, g["race_id"].n_unique(), pct, conf))
    print(f"\n{label}: (conf>1 = over-confident, <1 = under-confident)")
    print(f"  {'bucket':>16s} {'races':>6s} {'%market':>8s} {'conf':>6s}")
    for b, nr, pct, conf in sorted(rows, key=lambda r: str(r[0])):
        print(f"  {str(b):>16s} {nr:>6d} {pct:>7.0%} {conf:>6.2f}")


def main():
    df, jky = load("ST")
    p = collect(df, jky)
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
                        pl.col("Venue").cast(pl.Utf8).alias("venue"),
                        pl.col("Going").cast(pl.Utf8).alias("going"))
          .select("date", "venue", "race_no", "runner", "going"))
    p = p.join(rr, on=["date", "venue", "race_no", "runner"], how="left")
    p = p.with_columns(
        pl.when(pl.col("field_size") <= 8).then(pl.lit("<=8"))
        .when(pl.col("field_size") <= 11).then(pl.lit("9-11")).otherwise(pl.lit("12+")).alias("fs_bin"),
        pl.col("distance").alias("dist"))
    print(f"OOS {p['race_id'].n_unique()} races (probit, full-scale)")
    _bucket_report(p, "fs_bin", "by FIELD SIZE")
    _bucket_report(p, "dist", "by DISTANCE")
    _bucket_report(p, "going", "by GOING")


if __name__ == "__main__":
    main()
