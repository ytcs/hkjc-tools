"""Betting layer — price pools, value the closing line, and stake against a strategy.

Turns model probabilities and market state into betting decisions, in clearly-separated
concerns:

- :mod:`~hkjc.betting.pools` — name the HKJC pools (Win, Place, Quinella, …) and map a
  pool + HKJC combination string onto the simulator event, so a finishing-order
  distribution becomes a price for any pool. Computes no probabilities, embeds no strategy.
- :mod:`~hkjc.betting.carryover` / :mod:`~hkjc.betting.carryover_cover` — the carryover
  structural-edge detector and cover optimiser.
- :mod:`~hkjc.betting.clv` — closing-line value: de-vig, opening/closing snapshots, odds
  drift (steam), CLV of placed bets. The leading indicator.
- :mod:`~hkjc.betting.strategy` — the :class:`Strategy` seam: model probs + market →
  stakes. The strategist designs against it; the trading agent executes it.

Per-pool takeout is in :mod:`hkjc.core.hkjc_facts` (and ``docs/HKJC_BETTING_RULES.md``).
"""
from . import carryover, carryover_cover, clv, pools, strategy
from .carryover import (
    CarryoverFlag,
    aggregate_edge,
    effective_takeout,
    flag_carryover,
)
from .carryover_cover import Cover, evaluate_cover, optimize_cover, top3_coverage
from .pools import (
    POOL_PRICERS,
    parse_combination,
    pool_prob,
    price_dividends,
    quinella_place_prob,
    quinella_prob,
    trio_prob,
)
from .strategy import FractionalKelly, Strategy, ValueThreshold

__all__ = [
    "carryover", "CarryoverFlag", "aggregate_edge", "effective_takeout", "flag_carryover",
    "carryover_cover", "Cover", "evaluate_cover", "optimize_cover", "top3_coverage",
    "pools", "POOL_PRICERS", "parse_combination", "pool_prob", "price_dividends",
    "quinella_prob", "quinella_place_prob", "trio_prob",
    "clv", "strategy", "Strategy", "ValueThreshold", "FractionalKelly",
]
