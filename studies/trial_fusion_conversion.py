"""CONVERSION test: additive TRIAL_GAIN vs the coherent precision-fusion — paired, full-scale.

Runs ONE walk-forward over the full venue panel, collecting per-runner predictions for BOTH
trial structures in the SAME process so they share the per-race MC seed (PYTHONHASHSEED=0 ⇒ the
delta is the mechanism, not seed noise):

  (a) additive   — TRIAL_GAIN on (the production baseline, trial_pi_ratio=0)
  (b) fusion     — the precision-weighted v_c posterior (trial_pi_ratio=ρ, additive off)

Then reports % of market + ECE OVERALL and sliced by CAREER STAGE of the race winner:
  debut (0 prior starts) / lightly-raced (1–5) / established (6+).
The structural win = match additive on debuts while ADDING on lightly-raced (where additive's
flat weight is wrong) and not hurting established.

  PYTHONHASHSEED=0 [VENUE=HV] [RHO=0.8] uv run --group bayes python studies/trial_fusion_conversion.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.backtesting import walk_forward_folds
from hkjc.model.fit import fit_predict_traffic

_spec = importlib.util.spec_from_file_location("backtest_st", Path(__file__).parent / "backtest_st.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

VENUE = os.environ.get("VENUE", "ST")
RHO = float(os.environ.get("RHO", "0.8"))
EPS = 1e-15

STAGES = [("debut", 0, 0), ("light(1-5)", 1, 5), ("estab(6+)", 6, 10**9)]


def _collect(df, folds, fp, jky_mean):
    """Run the model over all folds, return concatenated test predictions (one row/runner)."""
    parts = []
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if train.height == 0 or test.height == 0:
            continue
        parts.append(fp(train, test))
    return pl.concat(parts)


def _renorm_market(preds):
    return preds.with_columns(
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("mkt"))


def _pct_market(preds):
    """% of market + ECE on a runner-level predictions frame (model col 'p', market 'mkt')."""
    win = preds.filter(pl.col("won") == 1)
    if win.height == 0:
        return float("nan"), float("nan"), win.height
    mdl = float(-np.mean(np.log(np.clip(win["p"].to_numpy().astype(float), EPS, 1.0))))
    mkt = float(-np.mean(np.log(np.clip(win["mkt"].to_numpy().astype(float), EPS, 1.0))))
    uni = float(-np.mean(np.log(1.0 / win["field_size"].to_numpy().astype(float))))
    pct = (uni - mdl) / (uni - mkt) if uni > mkt else float("nan")
    from hkjc.backtesting import scoring
    ece = scoring.expected_calibration_error(preds["p"].to_numpy(), preds["won"].to_numpy())
    return pct, ece, win.height


def main():
    df, jky_mean = bt.load(VENUE)
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{VENUE}: {df['race_id'].n_unique()} races, {df.height} runners, "
          f"{len(folds)} folds, RHO={RHO}, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")

    def add_fp(train, test):
        return fit_predict_traffic(train, test, jky_mean=jky_mean, sim="wald", trial_pi_ratio=0.0)

    def fus_fp(train, test):
        return fit_predict_traffic(train, test, jky_mean=jky_mean, sim="wald", trial_pi_ratio=RHO)

    add = _renorm_market(_collect(df, folds, add_fp, jky_mean))
    fus = _renorm_market(_collect(df, folds, fus_fp, jky_mean))

    print(f"\n  {'slice':14s} {'n_win':>6s}  {'additive':>18s}   {'fusion':>18s}")
    print(f"  {'':14s} {'':>6s}  {'%mkt':>8s} {'ECE':>8s}   {'%mkt':>8s} {'ECE':>8s}   Δ%mkt")
    for name, lo, hi in [("OVERALL", 0, 10**9)] + STAGES:
        a = add.filter((pl.col("prior_starts") >= lo) & (pl.col("prior_starts") <= hi))
        f = fus.filter((pl.col("prior_starts") >= lo) & (pl.col("prior_starts") <= hi))
        # slice on the WINNER's career stage but keep whole races for the renormalised probs
        pa, ea, na = _pct_market(a)
        pf, ef, nf = _pct_market(f)
        print(f"  {name:14s} {na:6d}  {pa:8.1%} {ea:8.4f}   {pf:8.1%} {ef:8.4f}   {pf-pa:+.1%}")


if __name__ == "__main__":
    main()
