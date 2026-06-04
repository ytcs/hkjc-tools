"""Resolve the recency contradiction: re-tune FORM_GAIN per form type, compare best-vs-best.

conditional_form.py: EW (recency) form is +8pp of market vs career mean. But form_recency_backtest.py:
coupling EW form into the ABM at the FIXED career-tuned FORM_GAIN=2.0 slightly HURTS. Hypothesis:
EW form (last 1-2 races dominate) has HIGHER variance than the smooth career mean, so the same gain
over-perturbs v_c with noise; the logit re-standardises + learns its coefficient, the ABM can't.
Fair test: sweep FORM_GAIN for each form type, compare each at its own optimum. Dataset loaded once
per half-life (FORM_GAIN acts only inside fit_predict), so this is cheap.

Run: uv run python experiments/form_gain_recency_sweep.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.eval.walkforward import walk_forward_folds

import backtest_traffic_st as bt

bt.N_SAMPLES = 3000


def per_race_loss(pred: pl.DataFrame) -> dict:
    out = {}
    for rid, g in pred.group_by("race_id"):
        p = np.clip(g["p"].to_numpy().astype(float), 1e-12, None)
        p = p / p.sum()
        won = g["won"].to_numpy().astype(bool)
        out[rid[0]] = -np.log(p[won][0]) if won.any() else np.nan
    return out


def eval_gain(df, folds, gain) -> dict:
    bt.FORM_GAIN = gain
    loss = {}
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if not test.is_empty():
            loss.update(per_race_loss(bt.fit_predict_traffic(train, test)))
    return loss


def form_std(df):
    return float(df["form"].std())


def main():
    gains = [0.0, 1.0, 2.0, 3.0, 4.0]
    print(f"{'form':>12s} {'std':>6s} " + " ".join(f"g={g:<5.1f}" for g in gains))
    rows = {}
    for label, hl in [("career", 1000.0), ("EW hl=2", 2.0)]:
        bt.FORM_HALFLIFE = hl
        df = bt.load_st()
        folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
        lls = []
        per_gain = {}
        for g in gains:
            loss = eval_gain(df, folds, g)
            per_gain[g] = loss
            lls.append(np.nanmean(list(loss.values())))
        rows[label] = per_gain
        print(f"{label:>12s} {form_std(df):>6.3f} " + " ".join(f"{v:>7.4f}" for v in lls))

    # paired: best EW gain vs best career gain
    def best(label):
        pg = rows[label]
        means = {g: np.nanmean(list(l.values())) for g, l in pg.items()}
        bg = min(means, key=means.get)
        return bg, pg[bg]
    cg, cl = best("career"); eg, el = best("EW hl=2")
    keys = sorted(set(cl) & set(el))
    c = np.array([cl[k] for k in keys]); e = np.array([el[k] for k in keys])
    m = np.isfinite(c) & np.isfinite(e)
    diff = c[m] - e[m]   # career - EW; positive = EW better
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(diff, diff.size, replace=True).mean() for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nbest career g={cg} LL {c[m].mean():.4f}   best EW g={eg} LL {e[m].mean():.4f}")
    print(f"EW - career (at each own best gain): {-diff.mean():+.4f} "
          f"[{-hi:+.4f},{-lo:+.4f}]  P(EW better)={(boot>0).mean():.1%}")


if __name__ == "__main__":
    main()
