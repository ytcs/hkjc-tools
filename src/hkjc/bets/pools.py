"""Price HKJC pools from a :class:`~hkjc.simulation.RaceSimulator`.

Combinations use HKJC's 1-based runner numbers (``"2,12"``); the simulator is
0-based, so we convert. Each pricer returns the model probability of that pool
combination winning, which — multiplied by a decimal payout — gives EV.
"""
from __future__ import annotations

from typing import List

import polars as pl

from ..simulation import RaceSimulator


def parse_combination(combo) -> List[int]:
    """HKJC combination string/iterable of 1-based runners -> 0-based indices."""
    if isinstance(combo, (list, tuple)):
        parts = combo
    else:
        parts = str(combo).replace(" ", "").split(",")
    return [int(x) - 1 for x in parts if str(x) != ""]


# -- per-pool pricers: P(this combination is the winning one) -----------------
def win_prob(sim: RaceSimulator, combo) -> float:
    (h,) = parse_combination(combo)
    return sim.prob(sim.wins(h))


def place_prob(sim: RaceSimulator, combo, k: int = 3) -> float:
    (h,) = parse_combination(combo)
    return sim.prob(sim.places(h, k=k))


def quinella_prob(sim: RaceSimulator, combo) -> float:
    """Both runners finish 1-2 (any order)."""
    return sim.prob(sim.top_set(parse_combination(combo), k=2))


def quinella_place_prob(sim: RaceSimulator, combo, k: int = 3) -> float:
    """Both runners finish in the top ``k`` (the paying places)."""
    return sim.prob(sim.all_place(parse_combination(combo), k=k))


def forecast_prob(sim: RaceSimulator, combo) -> float:
    """The two runners finish 1st-2nd in the exact order given."""
    return sim.prob(sim.exact_order(parse_combination(combo)))


def trio_prob(sim: RaceSimulator, combo) -> float:
    """The three runners are the top 3 (any order)."""
    return sim.prob(sim.top_set(parse_combination(combo), k=3))


def tierce_prob(sim: RaceSimulator, combo) -> float:
    """The three runners finish 1-2-3 in the exact order given."""
    return sim.prob(sim.exact_order(parse_combination(combo)))


def first4_prob(sim: RaceSimulator, combo) -> float:
    """The four runners are the top 4 (any order)."""
    return sim.prob(sim.top_set(parse_combination(combo), k=4))


def quartet_prob(sim: RaceSimulator, combo) -> float:
    """The four runners finish 1-2-3-4 in the exact order given."""
    return sim.prob(sim.exact_order(parse_combination(combo)))


POOL_PRICERS = {
    "WIN": win_prob, "PLA": place_prob,
    "QIN": quinella_prob, "QPL": quinella_place_prob,
    "FCT": forecast_prob, "TCE": tierce_prob, "TRI": trio_prob,
    "FF": first4_prob, "QTT": quartet_prob,
}


def pool_prob(sim: RaceSimulator, pool: str, combo, place_k: int = 3) -> float:
    """Model probability that ``combo`` is the winning combination of ``pool``.

    ``pool`` is a code (``WIN``/``PLA``/``QIN``/``QPL``/``FCT``/``TCE``/``TRI``/
    ``FF``/``QTT``). Raises ``KeyError`` for an unsupported pool.
    """
    pricer = POOL_PRICERS[pool]
    if pool in ("PLA", "QPL"):
        return pricer(sim, combo, k=place_k)
    return pricer(sim, combo)


def price_dividends(
    sim: RaceSimulator,
    dividends: pl.DataFrame,
    place_k: int = 3,
    pool_col: str = "pool",
    combo_col: str = "combination",
    decimal_col: str = "decimal",
) -> pl.DataFrame:
    """Attach model probability and realized edge to a race's dividend rows.

    For each (pool, winning combination, decimal payout) row, computes the model's
    probability of that combination and ``edge = model_prob * decimal - 1`` — the
    EV you would have realized betting that combination at the declared dividend.
    Rows for pools the simulator can't price are dropped.
    """
    rows = []
    for r in dividends.iter_rows(named=True):
        pool = r[pool_col]
        if pool not in POOL_PRICERS:
            continue
        try:
            p = pool_prob(sim, pool, r[combo_col], place_k=place_k)
        except (ValueError, IndexError, KeyError):
            continue
        dec = r.get(decimal_col)
        edge = (p * dec - 1.0) if dec is not None else None
        rows.append({**r, "model_prob": p, "edge": edge})
    return pl.DataFrame(rows)
