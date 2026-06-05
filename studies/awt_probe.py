"""Diagnose the AWT (all-weather, 1650m) failure — calibration-by-condition put it at 13% market.

AWT has different surface physics (dirt/all-weather vs turf): different achievable velocities, draw
bias, kickback, and — crucially — a horse's ability may NOT transfer between surfaces (turf
specialists vs AWT specialists). The turf-calibrated model pools AWT in. This decomposes the cause:

  1. Is the model's ABILITY signal useless on AWT? -> rank-corr of model v_c_eb vs winning, AWT vs turf.
  2. Is it NOISE (over-confident)? -> sweep speed_noise on AWT only.
  3. Does SURFACE-SPECIFIC form help? -> form from AWT-prior runs only (data-starved but a test).

Run: uv run --group bayes python studies/awt_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from quick_eval import load
from hkjc.backtesting import walk_forward_folds
from hkjc.backtesting.scoring import race_win_log_loss
from hkjc.model.fit import fit_predict_traffic

AWT = 1650


def _pct(p):
    gm = p.with_columns(
        (pl.col("p") / pl.col("p").sum().over("race_id")).alias("mp"),
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("kp"),
        (1.0 / pl.col("field_size")).alias("up"))
    m = race_win_log_loss(gm, prob_col="mp", winner_col="won", race_col="race_id")
    k = race_win_log_loss(gm, prob_col="kp", winner_col="won", race_col="race_id")
    u = race_win_log_loss(gm, prob_col="up", winner_col="won", race_col="race_id")
    return (u - m) / (u - k) if u > k else float("nan")


def collect(df, jky, **kw):
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    parts = []
    for fold in folds:
        tr = df.filter(pl.col("date").is_in(fold.train))
        te = df.filter(pl.col("date").is_in(fold.test))
        if tr.height and te.height:
            parts.append(fit_predict_traffic(tr, te, jky_mean=jky, sim="probit", n_samples=20000, **kw))
    return pl.concat(parts)


def main():
    df, jky = load("ST")
    p = collect(df, jky)
    awt = p.filter(pl.col("distance") == AWT)
    turf = p.filter(pl.col("distance") != AWT)
    print(f"AWT races={awt['race_id'].n_unique()}, turf={turf['race_id'].n_unique()}")
    print(f"  %market: AWT {_pct(awt):.1%} | turf {_pct(turf):.1%}\n")

    # 1) does model ability (v_c_eb) rank winners on AWT? (Spearman-ish: mean v_c_eb rank of winners)
    for name, q in (("AWT", awt), ("turf", turf)):
        qq = q.with_columns(pl.col("v_c_eb").rank().over("race_id").alias("vrank"),
                            pl.col("field_size").alias("fs"))
        wr = qq.filter(pl.col("won") == 1)
        # normalised rank of winner by model ability (1=model's top pick won); 0.5=random
        nr = ((wr["vrank"] - 1) / (wr["fs"] - 1)).mean()
        mr = (qq.filter(pl.col("won") == 1)["market_win_prob"].mean())
        print(f"  {name}: winner's model-ability percentile {nr:.2f} (1=top pick won, 0.5=random) | "
              f"mean market prob of winner {mr:.3f}")

    # 2) noise sensitivity on AWT (over-confidence?)
    print("\nAWT noise sweep (is it over-confident?):")
    for sn in (0.12, 0.18, 0.25, 0.35):
        pa = collect(df, jky, speed_noise=sn).filter(pl.col("distance") == AWT)
        print(f"  speed_noise={sn:.2f}   AWT %market {_pct(pa):.1%}")


if __name__ == "__main__":
    main()
