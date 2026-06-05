"""Recalibrate bend_curvature (ST) — residual-queue #1: draw OVER-weighted (t=-17.4).

The market-residual diagnosis at FORM_GAIN=4 left `draw` as the biggest miss: the ABM gives
inner draws MORE win probability than the market (residual slope t=-17.4). `bend_curvature`
is the one knob that sets the geometric draw bias (extra ground per metre off the rail on a
bend), so OVER-weighting draw ⇒ bend_curvature is too strong. This sweeps it against OUTCOMES
(% of market = goodness-of-fit), not against the market, so it's a legitimate mechanism
recalibration — the same move that fixed FORM_GAIN.

Speed: subsample TEST_CAP test races/fold (train stays full); the relative ordering across the
grid is robust to subsampling, and the winner is re-confirmed at full scale before baking.

Run: uv run --group bayes python studies/bend_curvature_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load
from hkjc.backtesting import walk_forward_folds
from hkjc.backtesting.scoring import race_win_log_loss
from hkjc.model.calibration import CALIBRATED_BEND_CURVATURE
from hkjc.model.fit import fit_predict_traffic

TEST_CAP = 180
SEED = 7
# around the current ST value (0.0037); over-weighting predicts the optimum is LOWER.
BENDS = [0.0, 0.0010, 0.0020, 0.0030, 0.0037, 0.0050]


def evaluate(df, folds, jky_mean, bend, rng):
    mrows, krows, urows = [], [], []
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if train.height == 0 or test.height == 0:
            continue
        rids = test["race_id"].unique().to_list()
        if len(rids) > TEST_CAP:
            keep = set(rng.choice(rids, size=TEST_CAP, replace=False).tolist())
            test = test.filter(pl.col("race_id").is_in(list(keep)))
        preds = fit_predict_traffic(train, test, jky_mean=jky_mean, bend=bend)
        mrows.append(preds.select("race_id", "won", pl.col("p").alias("q")))
        krows.append(test.with_columns(
            (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("q"))
            .select("race_id", "won", "q"))
        urows.append(test.with_columns((1.0 / pl.col("field_size")).alias("q"))
                     .select("race_id", "won", "q"))

    def ll(rows):
        return race_win_log_loss(pl.concat(rows), prob_col="q", winner_col="won", race_col="race_id")
    return ll(mrows), ll(krows), ll(urows)


def main():
    df, jky_mean = load("ST")
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"ST {df['race_id'].n_unique()} races; sweep on {TEST_CAP}/fold test subsample")
    print(f"baseline bend_curvature(ST) = {CALIBRATED_BEND_CURVATURE['ST']}\n")
    print(f"  {'bend':>8s} {'model_LL':>9s} {'%market':>8s}")
    best = None
    for bend in BENDS:
        rng = np.random.default_rng(SEED)               # same subsample across configs
        m, k, u = evaluate(df, folds, jky_mean, bend, rng)
        pct = (u - m) / (u - k)
        star = ""
        if best is None or pct > best[0]:
            best, star = (pct, bend), "  <-- best"
        print(f"  {bend:>8.4f} {m:>9.4f} {pct:>7.0%}{star}")
    print(f"\nbest: bend_curvature(ST)={best[1]} -> {best[0]:.0%} of market "
          f"(baseline 0.0037). Re-confirm at full scale before baking into hkjc.model.calibration.")


if __name__ == "__main__":
    main()
