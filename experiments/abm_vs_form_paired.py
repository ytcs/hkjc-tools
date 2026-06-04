"""Paired bootstrap: is the ABM's edge over 1-factor form-logit real or noise?

Produces per-race race-win log-loss for BOTH models on the IDENTICAL test races
(same walk-forward folds), then bootstraps the paired difference. A 1-factor logit is
fit per-fold (no peeking); the ABM uses the baked FORM_GAIN/SPEED_NOISE (tuned on the
aggregate — a mild advantage we acknowledge). N reduced for speed; ranking is robust.

Run: uv run python experiments/abm_vs_form_paired.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

import backtest_traffic_st as bt
from backtest_traffic_st import load_st, fit_predict_traffic

bt.N_SAMPLES = 3000  # plenty for ranking; per-race loss is stable


def per_race_loss(pred: pl.DataFrame) -> dict:
    """race_id -> -log p(winner), renormalising p within race."""
    out = {}
    for rid, g in pred.group_by("race_id"):
        p = g["p"].to_numpy().astype(float)
        p = np.clip(p / p.sum(), 1e-12, 1.0)
        won = g["won"].to_numpy().astype(bool)
        out[rid[0]] = -np.log(p[won][0]) if won.any() else np.nan
    return out


def main():
    df = load_st().with_columns(
        ((pl.col("form") - pl.col("form").mean()) / pl.col("form").std()).alias("z_form"))
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)

    logit_fp = conditional_logit_fit_predict(["z_form"], race_keys=["race_id"], prob_col="p")

    abm_loss, logit_loss = {}, {}
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if test.is_empty():
            continue
        abm_loss.update(per_race_loss(fit_predict_traffic(train, test)))
        logit_loss.update(per_race_loss(logit_fp(train, test)))

    keys = sorted(set(abm_loss) & set(logit_loss))
    a = np.array([abm_loss[k] for k in keys])
    l = np.array([logit_loss[k] for k in keys])
    m = np.isfinite(a) & np.isfinite(l)
    a, l = a[m], l[m]
    diff = l - a  # positive = ABM better (lower loss)

    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_abm_better = (boot > 0).mean()

    print(f"paired on {len(a)} test races  (N_SAMPLES={bt.N_SAMPLES})\n")
    print(f"  ABM        mean race-win log-loss : {a.mean():.4f}")
    print(f"  form-logit mean race-win log-loss : {l.mean():.4f}")
    print(f"  mean paired diff (logit - ABM)    : {diff.mean():+.4f} nats/race")
    print(f"  95% bootstrap CI                  : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(ABM better than form-logit)     : {p_abm_better:.1%}")
    wins = (diff > 0).mean()
    print(f"  ABM wins the race on              : {wins:.1%} of races")


if __name__ == "__main__":
    main()
