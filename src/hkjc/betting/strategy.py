"""The ``Strategy`` seam — turn model probabilities + market state into stakes.

Separation of concerns: a :class:`Strategy` *consumes* model win-probabilities and the
current market (odds / pool state) and decides **how much to stake on what**. It does not
fit the model and it does not price pools (that is :mod:`hkjc.betting.pools`). The betting
strategist designs strategies against this interface; the trading agent executes the
chosen one live (paper for now). Backtests drive a Strategy through historical odds with
the no-lookahead machinery in :mod:`hkjc.backtesting`.

Two reference strategies are provided — a positive-EV value filter and fractional Kelly —
both honest about takeout: edge is measured on the de-vigged probability the *model*
holds versus the price actually offered, and Kelly fractions are capped to keep variance
and model-error risk bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Strategy(Protocol):
    """Maps a race's model probabilities + offered odds to a per-runner stake vector."""

    name: str

    def stakes(
        self,
        *,
        model_prob: np.ndarray,      # (n,) model win probability per runner, sums ~1
        market_odds: np.ndarray,     # (n,) decimal odds offered
        bankroll: float,
        pool_state: Optional[dict] = None,
    ) -> np.ndarray:
        """Return a (n,) vector of stakes (same currency as ``bankroll``); 0 = no bet."""
        ...


def _edge(model_prob: np.ndarray, market_odds: np.ndarray) -> np.ndarray:
    """Expected profit per unit staked: ``p·odds − 1`` (>0 ⇒ positive-EV at the offered price)."""
    o = np.asarray(market_odds, float)
    return np.asarray(model_prob, float) * o - 1.0


@dataclass
class ValueThreshold:
    """Flat stake on every runner whose model edge clears a threshold (positive-EV filter)."""

    edge_threshold: float = 0.05
    stake_frac: float = 0.01          # fraction of bankroll per qualifying bet
    name: str = "value_threshold"

    def stakes(self, *, model_prob, market_odds, bankroll, pool_state=None) -> np.ndarray:
        qualify = _edge(model_prob, market_odds) > self.edge_threshold
        return qualify.astype(float) * (self.stake_frac * bankroll)


@dataclass
class FractionalKelly:
    """Fractional-Kelly staking, capped. ``f* = (b·p − q)/b`` with ``b = odds − 1``.

    Only positive-edge runners are staked; the raw Kelly fraction is scaled by
    ``fraction`` (de-risks model error) and clipped to ``cap`` of bankroll.
    """

    fraction: float = 0.25
    cap: float = 0.05
    name: str = "fractional_kelly"

    def stakes(self, *, model_prob, market_odds, bankroll, pool_state=None) -> np.ndarray:
        p = np.asarray(model_prob, float)
        b = np.asarray(market_odds, float) - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(b > 0, (b * p - (1.0 - p)) / b, 0.0)
        f = np.clip(f * self.fraction, 0.0, self.cap)
        return f * bankroll
