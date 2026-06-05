"""Calibrate MASS_EXP (β) — residual-queue #1: body mass UNDER-weighted (t=+8.8, confirmed real).

The coupling is DERIVED from physics, not a free gain. The EOM gives v_c = p_c/(K_LOCO·m·g);
aerobic power scales with muscle mass allometrically, p_c = κ·m^b, so v_c ∝ m^(b−1) = m^β. The
probe confirmed the bodywt residual is genuine mass (survives rating+class controls; monotone),
and POSITIVE (heavier = faster) ⟹ β>0 ⟹ b>1 (power scales super-linearly with mass). Applied as
a field-relative factor v_c *= (m/m̄_field)^β. This sweeps the single physiological exponent β.

Subsampled (TEST_CAP races/fold); re-confirm at full scale + re-diagnose before baking.

Run: uv run --group bayes python studies/mass_exp_sweep.py
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
EXPS = [0.0, 0.25, 0.5, 1.0, 2.0]


def evaluate(df, folds, jky_mean, exp, rng):
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
        preds = fit_predict_traffic(train, test, jky_mean=jky_mean, mass_exp=exp)
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
    cov = df["bodywt"].is_not_null().mean()
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"ST {df['race_id'].n_unique()} races; bodywt coverage {cov:.1%}; {TEST_CAP}/fold subsample")
    print("v_c *= (m/m_field)^MASS_EXP  (MASS_EXP = allometric exponent beta = b-1)\n")
    print(f"  {'MASS_EXP':>8s} {'model_LL':>9s} {'%market':>8s}")
    best = None
    for exp in EXPS:
        rng = np.random.default_rng(SEED)
        m, k, u = evaluate(df, folds, jky_mean, exp, rng)
        pct = (u - m) / (u - k)
        star = ""
        if best is None or pct > best[0]:
            best, star = (pct, exp), "  <-- best"
        print(f"  {exp:>8.2f} {m:>9.4f} {pct:>7.0%}{star}")
    print(f"\nbest: MASS_EXP={best[1]} -> {best[0]:.0%} of market (baseline 0.0). "
          f"Re-confirm at full scale + re-diagnose before baking.")


if __name__ == "__main__":
    main()
