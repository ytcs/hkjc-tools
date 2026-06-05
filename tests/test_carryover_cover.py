"""Tests for the jackpot Flexi-cover optimizer."""
from math import comb

import numpy as np

from hkjc.bets import carryover_cover as cc
from hkjc.model import PlackettLuceModel
from hkjc.core.simulation import RaceSimulator


def _sim(n: int = 8, ns: int = 120_000, seed: int = 0) -> RaceSimulator:
    return RaceSimulator(PlackettLuceModel(np.ones(n)), n_samples=ns, seed=seed)


def test_top3_coverage_matches_combinatorics():
    sim = _sim(8)
    for k in (3, 5, 7):
        expected = comb(k, 3) / comb(8, 3)         # equal field: uniform 3-subset
        assert abs(cc.top3_coverage(sim, range(k)) - expected) < 0.01


def test_evaluate_cover_formula():
    sims = [_sim(8, seed=i) for i in range(3)]
    cov = cc.evaluate_cover(sims, [list(range(5))] * 3, expected_dividend=100_000, stake=1.0)
    assert cov.n_combos == comb(5, 3) ** 3
    assert abs(cov.hit_prob - (comb(5, 3) / comb(8, 3)) ** 3) < 0.02
    assert abs(cov.ev - (cov.expected_return - cov.cost)) < 1e-9
    assert cov.runner_sets()[0] == [h + 1 for h in cov.leg_sets[0]]   # 1-based


def test_optimizer_responds_to_economics():
    sims = [_sim(8, seed=i) for i in range(3)]
    # tiny dividend: not worth expanding past the minimal top-3 box
    small = cc.optimize_cover(sims, expected_dividend=10.0, bankroll=1e12)
    assert small.set_sizes == [3, 3, 3]
    # huge dividend: expand the cover; EV strictly better than the minimal box
    big = cc.optimize_cover(sims, expected_dividend=5e6, bankroll=1e12)
    assert sum(big.set_sizes) > 9
    assert big.ev >= small.ev


def test_bankroll_cap_binds():
    sims = [_sim(8, seed=i) for i in range(3)]
    # huge dividend but a cost cap that only allows the minimal 1-combo-per-leg box
    capped = cc.optimize_cover(sims, expected_dividend=5e6, bankroll=1.0, stake=1.0)
    assert capped.n_combos == 1 and capped.set_sizes == [3, 3, 3]
