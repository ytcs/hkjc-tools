"""Does coupling jockey into the ABM (as an effective-v_c measurement) help OOS? Paired bootstrap.

The conditional-logit scan said jockey is +7pp of market skill beyond form. Here we test it in
the real mechanistic model: same races/folds, only JOCKEY_GAIN changes. Reports per-race
race-win log-loss and the paired diff (off − on); positive = jockey helps. Also vs market.

Run: uv run python experiments/jockey_backtest.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.eval.walkforward import walk_forward_folds

import backtest_traffic_st as bt
from backtest_traffic_st import load_st, fit_predict_traffic

bt.N_SAMPLES = 4000


def per_race_loss(pred: pl.DataFrame) -> dict:
    out = {}
    for rid, g in pred.group_by("race_id"):
        p = np.clip(g["p"].to_numpy().astype(float), 1e-12, None)
        p = p / p.sum()
        won = g["won"].to_numpy().astype(bool)
        out[rid[0]] = -np.log(p[won][0]) if won.any() else np.nan
    return out


def run(df, folds, gain):
    bt.JOCKEY_GAIN = gain
    loss = {}
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if not test.is_empty():
            loss.update(per_race_loss(fit_predict_traffic(train, test)))
    return loss


def market_loss(df, folds):
    out = {}
    for fold in folds:
        for rid, g in df.filter(pl.col("date").is_in(fold.test)).group_by("race_id"):
            won = g["won"].to_numpy().astype(bool)
            if won.any():
                m = g["market_win_prob"].to_numpy().astype(float)
                m = np.clip(m / m.sum(), 1e-12, 1)
                out[rid[0]] = -np.log(m[won][0])
    return out


def main():
    df = load_st()
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    print(f"{df['race_id'].n_unique()} races, {df.height} runners; JKY_MEAN={bt.JKY_MEAN:.4f}\n")
    base = run(df, folds, 0.0)
    mk = market_loss(df, folds)
    k0 = sorted(set(base) & set(mk))
    bb = np.array([base[k] for k in k0]); mm = np.array([mk[k] for k in k0])
    print(f"  market mean LL  = {mm.mean():.4f}")
    print(f"  jockey-off  LL  = {bb.mean():.4f}  (gap to market {bb.mean()-mm.mean():+.4f})\n")
    rng = np.random.default_rng(0)
    print(f"{'JOCKEY_GAIN':>11s} {'mean LL':>9s} {'Δ vs off':>9s} {'95% CI':>20s} {'P(helps)':>9s}")
    print(f"{0.0:>11.1f} {np.nanmean(list(base.values())):>9.4f} {'—':>9s} {'—':>20s} {'—':>9s}")
    for gain in (3.0, 6.0, 10.0, 15.0):
        on = run(df, folds, gain)
        keys = sorted(set(base) & set(on))
        b = np.array([base[k] for k in keys]); o = np.array([on[k] for k in keys])
        m = np.isfinite(b) & np.isfinite(o)
        diff = b[m] - o[m]
        boot = np.array([rng.choice(diff, diff.size, replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{gain:>11.1f} {o[m].mean():>9.4f} {diff.mean():>+9.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {(boot>0).mean():>8.1%}")


if __name__ == "__main__":
    main()
