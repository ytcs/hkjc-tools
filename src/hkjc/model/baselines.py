"""Baseline predictors — the bars a real model must clear.

These produce a per-runner win probability column without any training, so they
slot into :func:`hkjc.eval.harness.walk_forward_evaluate` exactly like a learned
model and give an honest reference on the same folds:

- **market** — the de-vigged market win probability, renormalized per race. A
  results-trained model that can't beat this on calibration / CLV has found no edge
  (it's just reproducing the crowd).
- **uniform** — ``1 / field_size``. The floor; beating it is table stakes.
"""
from __future__ import annotations

from typing import Callable, Sequence

import polars as pl

from ..core.contracts import RACE_KEYS


def market_baseline_fit_predict(
    prob_col: str = "p",
    market_col: str = "market_win_prob",
    race_keys: Sequence[str] = tuple(RACE_KEYS),
) -> Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]:
    """Predict the market's de-vigged win probability (renormalized per race)."""
    race_keys = list(race_keys)

    def fit_predict(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
        return test.with_columns(
            (pl.col(market_col) / pl.col(market_col).sum().over(race_keys)).alias(prob_col))

    return fit_predict


def uniform_fit_predict(
    prob_col: str = "p",
    race_keys: Sequence[str] = tuple(RACE_KEYS),
) -> Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]:
    """Predict a uniform ``1 / field_size`` win probability."""
    race_keys = list(race_keys)

    def fit_predict(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
        return test.with_columns((1.0 / pl.len().over(race_keys)).alias(prob_col))

    return fit_predict
