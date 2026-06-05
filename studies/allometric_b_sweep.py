"""Calibrate ALLOMETRIC_B (b) — residual-queue #1: body mass UNDER-weighted (t=+8.8, confirmed real).

Mass enters via the EOM's STEADY-STATE law v_c ∝ m^(b−1) — the analytic solution of
v_c = p_c/(K_LOCO·m·g) with allometric aerobic power p_c = κ·m^b. (The full integrated EOM was
tried and underperforms the kinematic sim by ~1.4 nats/race; see journal — so production uses the
steady-state law, which is first-principles, not a tack-on.) The probe confirmed the bodywt
residual is genuine mass (survives rating+class; monotone). POSITIVE residual ⟹ b>1. This sweeps
the single physiological exponent b on the production (kinematic) path.

Subsampled (TEST_CAP races/fold); re-confirm at full scale + re-diagnose before baking.

Run: uv run --group bayes python studies/allometric_b_sweep.py
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
from hkjc.model.fit import fit_predict_traffic

TEST_CAP = 180
SEED = 7
BS = [1.0, 1.5, 2.0, 3.0, 4.0]   # b=1 -> no mass effect; b>1 -> heavier faster


def evaluate(df, folds, jky_mean, b, rng):
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
        preds = fit_predict_traffic(train, test, jky_mean=jky_mean, allometric_b=b)
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
    print(f"ST {df['race_id'].n_unique()} races; {TEST_CAP}/fold subsample")
    print("v_c *= (m/m_field)^(b-1)  (steady-state allometric law)\n")
    print(f"  {'ALLOMETRIC_B':>12s} {'model_LL':>9s} {'%market':>8s}")
    best = None
    for b in BS:
        rng = np.random.default_rng(SEED)
        m, k, u = evaluate(df, folds, jky_mean, b, rng)
        pct = (u - m) / (u - k)
        star = ""
        if best is None or pct > best[0]:
            best, star = (pct, b), "  <-- best"
        print(f"  {b:>12.1f} {m:>9.4f} {pct:>7.0%}{star}")
    print(f"\nbest: ALLOMETRIC_B={best[1]} -> {best[0]:.0%} of market (baseline b=1.0). "
          f"Re-confirm at full scale + re-diagnose before baking.")


if __name__ == "__main__":
    main()
