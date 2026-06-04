"""Does the running-style → pace-policy coupling help? Paired bootstrap vs pace-off.

The litmus for the emergent-pace hypothesis: a per-runner covariate HURT (every flat
pace feature lost), so if pace carries signal it must show up HERE — through the
field-coupled sim policy, not a feature. Same races/folds; only PACE_GAIN changes.

Run: uv run python experiments/pace_backtest.py
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
        p = g["p"].to_numpy().astype(float)
        p = np.clip(p / p.sum(), 1e-12, 1.0)
        won = g["won"].to_numpy().astype(bool)
        out[rid[0]] = -np.log(p[won][0]) if won.any() else np.nan
    return out


def run(df, folds, gain):
    bt.PACE_GAIN = gain
    loss = {}
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if not test.is_empty():
            loss.update(per_race_loss(fit_predict_traffic(train, test)))
    return loss


def main():
    df = load_st()
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    print(f"{df['race_id'].n_unique()} races, {df.height} runners\n")

    base = run(df, folds, 0.0)
    rng = np.random.default_rng(0)
    bk = sorted(base)
    print(f"{'PACE_GAIN':>10s} {'mean LL':>9s} {'Δ vs off':>9s} {'95% CI':>20s} {'P(helps)':>9s}")
    print(f"{0.0:>10.2f} {np.nanmean([base[k] for k in bk]):>9.4f} {'—':>9s} {'—':>20s} {'—':>9s}")
    for gain in (0.03, 0.06, 0.10, 0.15):
        on = run(df, folds, gain)
        keys = sorted(set(base) & set(on))
        b = np.array([base[k] for k in keys]); o = np.array([on[k] for k in keys])
        m = np.isfinite(b) & np.isfinite(o)
        diff = b[m] - o[m]
        boot = np.array([rng.choice(diff, diff.size, replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{gain:>10.2f} {o[m].mean():>9.4f} {diff.mean():>+9.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {(boot>0).mean():>8.1%}")


if __name__ == "__main__":
    main()
