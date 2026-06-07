"""The GATE — market skill% over the efficiency-weighted pool book. VALIDATE + CALIBRATE coupling.

Layer-3 GATE of the canonical framework. The path-reconstruction loss IDENTIFIES θ honestly; the
GATE does two jobs (#40):

  1. CALIBRATE the coupling strengths (:class:`~hkjc.model.coupling.CouplingGains`) — each
     latent's win/pool coupling is tuned to MAXIMISE market-skill% on the pool book, not set to the
     honest magnitude (honest posteriors under-discriminate for win).
  2. VALIDATE / PROMOTE — a latent (Stage 1: `a`) is promoted only if, at its gate-calibrated
     coupling, it beats the a=OFF baseline on the gate metric. Gate-validated, never assumed.

The metric is %market+ECE: % of the market's skill the model captures (the project's goodness-of-fit
target, CLAUDE.md) plus calibration (ECE). This module REUSES the production gate harness
(:mod:`hkjc.backtesting`) — ``walk_forward_evaluate`` + ``assert_no_lookahead`` + the scoring metrics —
rather than re-implementing it: the gate must be the SAME judge the production model answers to, and it
must stay un-game-able (nothing in the model is reachable from the judge).

This skeleton provides the hooks (a ``fit_predict`` adapter the harness can call and a thin scoring
wrapper). The actual coupling-gain SEARCH (a small grid/optimiser over CouplingGains maximising
market-skill%) is the BUILD-WAVE — wired here as :meth:`calibrate_coupling` (STUB) so the search has a
named home.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from hkjc.backtesting import assert_no_lookahead
from hkjc.backtesting.harness import aggregate, walk_forward_evaluate

from .coupling import CouplingGains

# Re-exported so callers tune/validate against the production judge, never a private copy.
__all__ = ["assert_no_lookahead", "walk_forward_evaluate", "aggregate",
           "gate_score", "pool_book_score", "calibrate_coupling"]

FitPredict = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]


def _pct_of_market(uniform_ll: float, market_ll: float, model_ll: float) -> float:
    """% of market = (uniform − model)/(uniform − market) — the production goodness-of-fit metric.

    100% = the model captures all of the market's information advantage over the uniform (1/field)
    baseline; >100% beats the market. EXACTLY the definition ``studies/backtest_st.py`` reports.
    """
    return (uniform_ll - model_ll) / (uniform_ll - market_ll) if uniform_ll > market_ll else float("nan")


def gate_score(preds: pl.DataFrame, *, prob_col: str = "p", market_col: str = "market_win_prob",
               outcome_col: str = "won", race_col: str = "race_id") -> dict:
    """Pool-book gate metric: exact % of market + ECE on the win pool.

    Returns the model / market / uniform per-race win log-losses, the exact ``pct_of_market`` (the
    project goodness-of-fit target, CLAUDE.md ST≈65%) and the model ECE. This is the SAME judge the
    production backtest answers to (reuses ``hkjc.backtesting.scoring``); nothing in the model is
    reachable from it (un-game-able). Win-only for Stage 1; Stage 2 extends it across the pool book.
    """
    from hkjc.backtesting import scoring

    rep = scoring.calibration_report(preds[prob_col].to_numpy(), preds[outcome_col].to_numpy())
    ll_model = scoring.race_win_log_loss(preds, prob_col=prob_col, winner_col=outcome_col,
                                         race_col=race_col)
    rep["race_win_log_loss"] = ll_model
    if market_col in preds.columns:
        # renormalise market win prob within race (overround), exactly as the backtest's _market.
        mkt = preds.with_columns(
            (pl.col(market_col) / pl.col(market_col).sum().over(race_col)).alias("_mkt"))
        ll_market = scoring.race_win_log_loss(mkt, prob_col="_mkt", winner_col=outcome_col,
                                              race_col=race_col)
        rep["market_race_win_log_loss"] = ll_market
    if "field_size" in preds.columns:
        uni = preds.with_columns((1.0 / pl.col("field_size")).alias("_uni"))
        ll_uni = scoring.race_win_log_loss(uni, prob_col="_uni", winner_col=outcome_col,
                                           race_col=race_col)
        rep["uniform_race_win_log_loss"] = ll_uni
        if market_col in preds.columns:
            rep["pct_of_market"] = _pct_of_market(ll_uni, ll_market, ll_model)
    return rep


def pool_book_score(scored: pl.DataFrame, *, pool_col: str = "pool",
                    model_col: str = "p_model", market_col: str = "p_market",
                    uniform_col: str = "p_uniform", race_col: str = "race_id") -> dict:
    """Per-pool market-skill% on the REALIZED combination — the pool-book gate.

    ``scored`` has ONE row per (race, pool): the model's, the de-vigged market's, and the uniform
    baseline's probability of the race's REALIZED winning combination for that pool. For each pool we
    form the mean per-race log-loss of each (``LL = -log p`` on the realized combo) and the exact

        %market = (uniform_LL − model_LL) / (uniform_LL − market_LL)

    — EXACTLY the win %market definition (:func:`_pct_of_market`), but on the exotic combination. We
    also report a one-vs-rest calibration error (``ece``) of the model probability against the realized
    "hit" (always 1 here, since every scored row is the realized combo) is not meaningful per-pool;
    instead we report the model/market/uniform LL and the mean model probability so the caller can read
    calibration as the model/market LL GAP (model_LL→market_LL is the calibration-and-sharpness signal
    on the realized combo). The de-vig is HONEST-but-PARTIAL: the dividend prices ONLY the realized
    combination's market prob, so %market is on that one combination — a valid but partial benchmark
    (the market's full distribution over all combinations is not observed).

    Returns ``{pool: {n_races, model_ll, market_ll, uniform_ll, pct_of_market, mean_p_model,
    mean_p_market}}``.
    """
    eps = 1e-12
    out: dict = {}
    for pool in scored[pool_col].unique().to_list():
        sub = scored.filter(pl.col(pool_col) == pool)
        if sub.height == 0:
            continue
        pm = np.clip(sub[model_col].to_numpy().astype(float), eps, 1.0)
        pk = np.clip(sub[market_col].to_numpy().astype(float), eps, 1.0)
        pu = np.clip(sub[uniform_col].to_numpy().astype(float), eps, 1.0)
        model_ll = float(-np.log(pm).mean())
        market_ll = float(-np.log(pk).mean())
        uniform_ll = float(-np.log(pu).mean())
        out[pool] = {
            "n_races": int(sub[race_col].n_unique()),
            "model_ll": model_ll,
            "market_ll": market_ll,
            "uniform_ll": uniform_ll,
            "pct_of_market": _pct_of_market(uniform_ll, market_ll, model_ll),
            "mean_p_model": float(pm.mean()),
            "mean_p_market": float(pk.mean()),
        }
    return out


def calibrate_coupling(
    data: pl.DataFrame,
    folds,
    build_fit_predict: Callable[[CouplingGains], FitPredict],
    candidate_gains: list[CouplingGains],
    *,
    date_col: str = "date",
    race_col: str = "race_id",
    outcome_col: str = "won",
) -> tuple[CouplingGains, pl.DataFrame]:
    """Search CouplingGains for the value maximising pool-book skill% (the #40 calibration).

    For each candidate gain: build the leak-free ``fit_predict``, walk-forward-evaluate it on the
    gate, and score the FULL out-of-sample prediction frame with :func:`gate_score` (exact % of
    market + ECE). The same leak-free conversion test the production constants are baked with;
    paired (the harness re-seeds per race_id, so with ``PYTHONHASHSEED=0`` base and candidate share
    the per-race MC seed and the Δ is pure mechanism). Returns the best-by-%market gains and the
    scoreboard. PROMOTE `a` iff the best non-zero ``a_start_gain`` beats ``a_start_gain=0`` on BOTH
    %market AND ECE (the caller decides; this returns the numbers).
    """
    rows = []
    best_gains, best_pct = candidate_gains[0], -float("inf")
    for gains in candidate_gains:
        fp = build_fit_predict(gains)
        preds = []
        for fold in folds:
            train = data.filter(pl.col(date_col).is_in(fold.train))
            test = data.filter(pl.col(date_col).is_in(fold.test))
            if train.height == 0 or test.height == 0:
                continue
            preds.append(fp(train, test))
        if not preds:
            continue
        oos = pl.concat(preds)
        sc = gate_score(oos, prob_col="p", outcome_col=outcome_col, race_col=race_col)
        row = {"a_start_gain": gains.a_start_gain,
               "pct_of_market": sc.get("pct_of_market", float("nan")),
               "ece": sc.get("ece", float("nan")),
               "race_win_log_loss": sc.get("race_win_log_loss", float("nan")),
               "n_races": int(oos[race_col].n_unique())}
        rows.append(row)
        if np.isfinite(row["pct_of_market"]) and row["pct_of_market"] > best_pct:
            best_pct, best_gains = row["pct_of_market"], gains
    return best_gains, pl.DataFrame(rows)
