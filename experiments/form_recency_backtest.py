"""Does exp-weighted (recency) form beat the career-mean form INSIDE the mechanistic ABM?

conditional_form.py showed recency form is +8pp of market in a plain logit (28%->36%). Here we
confirm it in the real traffic+energetics model, where form perturbs effective v_c via FORM_GAIN.
A large FORM_HALFLIFE makes ewm_mean ~ a flat career mean, so the half-life sweep is a clean A/B:
HL=1000 ~ career-mean baseline, HL=2 = recency. Paired bootstrap on per-race win log-loss; positive
diff = recency helps. Same races/folds/energetics as the ST backtest, only the form table changes.

Run: uv run python experiments/form_recency_backtest.py
"""
from __future__ import annotations

import importlib

import numpy as np
import polars as pl

from hkjc.eval.walkforward import walk_forward_folds

import backtest_traffic_st as bt

bt.N_SAMPLES = 4000


def per_race_loss(pred: pl.DataFrame) -> dict:
    out = {}
    for rid, g in pred.group_by("race_id"):
        p = np.clip(g["p"].to_numpy().astype(float), 1e-12, None)
        p = p / p.sum()
        won = g["won"].to_numpy().astype(bool)
        out[rid[0]] = -np.log(p[won][0]) if won.any() else np.nan
    return out


def run_with_halflife(hl: float) -> dict:
    """Rebuild the dataset with FORM_HALFLIFE=hl (form table reads the module global), backtest."""
    bt.FORM_HALFLIFE = hl
    df = bt.load_st()
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    loss = {}
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if not test.is_empty():
            loss.update(per_race_loss(bt.fit_predict_traffic(train, test)))
    return loss


def main():
    base = run_with_halflife(1000.0)   # ~ career mean
    print(f"career-mean (HL=1000):  mean LL {np.nanmean(list(base.values())):.4f}\n")
    rng = np.random.default_rng(0)
    print(f"{'HALFLIFE':>9s} {'mean LL':>9s} {'Δ vs career':>12s} {'95% CI':>20s} {'P(helps)':>9s}")
    print(f"{1000.0:>9.1f} {np.nanmean(list(base.values())):>9.4f} {'—':>12s} {'—':>20s} {'—':>9s}")
    for hl in (4.0, 2.0, 1.5):
        on = run_with_halflife(hl)
        keys = sorted(set(base) & set(on))
        b = np.array([base[k] for k in keys]); o = np.array([on[k] for k in keys])
        m = np.isfinite(b) & np.isfinite(o)
        diff = b[m] - o[m]
        boot = np.array([rng.choice(diff, diff.size, replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{hl:>9.1f} {o[m].mean():>9.4f} {diff.mean():>+12.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {(boot>0).mean():>8.1%}")


if __name__ == "__main__":
    main()
