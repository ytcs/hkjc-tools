"""Jackpot carryover detector — the one structurally +EV HKJC window.

Double Trio, Triple Trio and Six Up are jackpot pools: when nobody wins, the net
pool is **carried forward** to the next meeting. On a carryover meeting the money
paid out can exceed the money bet that day, because last meeting's forfeited pool
(``carry_in``, ``C``) is added to this meeting's net pool. New money then faces a
*negative effective takeout on the aggregate pool* — a redistribution of forfeited
money, not a model-vs-crowd edge anyone can refute on calibration grounds.

If the pool is won this meeting, winners share ``net_fraction * G + C`` where ``G``
is today's gross pool and ``net_fraction`` (~0.85 for HKJC jackpot pools, after the
~7.5% jackpot-reserve + ~0.5% operator-reserve + take) is the share of gross paid
out. Per unit of new money, the aggregate expected return is therefore
``net_fraction + C/G``, so:

    effective_takeout = (1 - net_fraction) - C/G
    aggregate_edge    = -effective_takeout = net_fraction + C/G - 1

The window is structurally +EV in aggregate when ``C/G > 1 - net_fraction``.

Caveats this module does NOT resolve (see journal / build plan):
- This is the *aggregate* pool edge. Capturing it needs a Flexi cover with a high
  enough hit rate; a partial cover that misses flips +EV to -EV.
- ``G`` should be the *settled* (final) gross pool. Pricing against the pre-close
  pool overstates edge, since a publicized carryover draws other harvesters in.
- Pure arithmetic here; pool sizes and the cover layer live elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass

# Default share of a jackpot pool's gross that is paid to winning combinations.
DEFAULT_NET_FRACTION = 0.85


def effective_takeout(carry_in: float, gross_pool: float, net_fraction: float = DEFAULT_NET_FRACTION) -> float:
    """Effective takeout on new money if the pool is won this meeting.

    Negative means the aggregate pool pays out more than today's bettors put in.
    """
    if gross_pool <= 0:
        return float("nan")
    return (1.0 - net_fraction) - carry_in / gross_pool


def aggregate_edge(carry_in: float, gross_pool: float, net_fraction: float = DEFAULT_NET_FRACTION) -> float:
    """Aggregate expected edge per unit of new money (= ``-effective_takeout``)."""
    return -effective_takeout(carry_in, gross_pool, net_fraction)


def carryover_ratio(carry_in: float, gross_pool: float) -> float:
    """``C / G`` — the carried-in jackpot relative to today's new gross pool."""
    return carry_in / gross_pool if gross_pool > 0 else float("nan")


@dataclass(frozen=True)
class CarryoverFlag:
    is_positive: bool          # structurally +EV in aggregate (before cover/dilution)
    effective_takeout: float
    aggregate_edge: float
    carryover_ratio: float
    margin: float              # required edge margin applied


def flag_carryover(
    carry_in: float,
    gross_pool: float,
    net_fraction: float = DEFAULT_NET_FRACTION,
    margin: float = 0.0,
) -> CarryoverFlag:
    """Flag a meeting whose carryover makes the pool structurally +EV in aggregate.

    ``margin`` is an extra edge cushion required above break-even to absorb cover
    miss-rate and late-pool dilution — set it well above 0 before trusting a flag.
    """
    eff = effective_takeout(carry_in, gross_pool, net_fraction)
    edge = -eff
    return CarryoverFlag(
        is_positive=bool(edge > margin),
        effective_takeout=eff,
        aggregate_edge=edge,
        carryover_ratio=carryover_ratio(carry_in, gross_pool),
        margin=margin,
    )
