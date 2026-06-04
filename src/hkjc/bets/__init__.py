"""Bets layer — price HKJC pools and turn probabilities into edge.

Sits above the simulator: where :mod:`hkjc.simulation` exposes generic event
predicates, this layer names the actual HKJC pools (Win, Place, Quinella, Quinella
Place, Forecast, Tierce, Trio, First 4, Quartet) and maps a pool + a HKJC
combination string (1-based runner numbers, e.g. ``"2,12"``) onto the right event,
so a model's finishing-order distribution can be turned into a price for any pool
and compared to the market / realized dividends.

It computes **no probabilities itself** and embeds **no strategy** — it composes
the simulator's events and applies arithmetic. Per-pool takeout lives in the
betting-rules doc, not here.
"""
from . import carryover, carryover_cover
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

__all__ = [
    "carryover", "CarryoverFlag", "aggregate_edge", "effective_takeout", "flag_carryover",
    "carryover_cover", "Cover", "evaluate_cover", "optimize_cover", "top3_coverage",
    "POOL_PRICERS", "parse_combination", "pool_prob", "price_dividends",
    "quinella_prob", "quinella_place_prob", "trio_prob",
]
