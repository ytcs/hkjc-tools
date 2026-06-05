"""Tests for the bets pricing layer (exact fractions on an equal-strength field)."""
import numpy as np
import polars as pl

from hkjc.betting import pools
from hkjc.model import PlackettLuceModel
from hkjc.core.simulation import RaceSimulator


def _sim(n: int = 3, ns: int = 200_000, seed: int = 0) -> RaceSimulator:
    return RaceSimulator(PlackettLuceModel(np.ones(n)), n_samples=ns, seed=seed)


def test_parse_combination():
    assert pools.parse_combination("2,12") == [1, 11]
    assert pools.parse_combination([2, 12]) == [1, 11]
    assert pools.parse_combination("7") == [6]


def test_equal_field_exact_fractions():
    sim = _sim(3)
    # 3 equal horses: quinella {1,2} = top-2 set = 1/3; forecast 1->2 = 1/6; trio = 1
    assert abs(pools.quinella_prob(sim, "1,2") - 1 / 3) < 0.01
    assert abs(pools.pool_prob(sim, "FCT", "1,2") - 1 / 6) < 0.01
    assert abs(pools.trio_prob(sim, "1,2,3") - 1.0) < 1e-9


def test_win_and_quinella_place_dispatch():
    sim = _sim(4)
    assert abs(pools.pool_prob(sim, "WIN", "1") - 0.25) < 0.01
    # both of {1,2} in top 3 of 4 <=> the last-placed horse is one of the other two
    assert abs(pools.pool_prob(sim, "QPL", "1,2", place_k=3) - 0.5) < 0.01


def test_price_dividends_edge():
    sim = _sim(3)
    div = pl.DataFrame({"pool": ["WIN", "QIN"], "combination": ["1", "1,2"], "decimal": [3.0, 3.0]})
    out = pools.price_dividends(sim, div)
    assert {"model_prob", "edge"} <= set(out.columns)
    win = out.filter(pl.col("pool") == "WIN").row(0, named=True)
    assert abs(win["model_prob"] - 1 / 3) < 0.01
    assert abs(win["edge"] - (1 / 3 * 3.0 - 1.0)) < 0.02   # ~0 edge at fair odds


def test_unsupported_pool_dropped():
    sim = _sim(3)
    div = pl.DataFrame({"pool": ["TT"], "combination": ["1,2,3"], "decimal": [100.0]})
    assert pools.price_dividends(sim, div).height == 0
