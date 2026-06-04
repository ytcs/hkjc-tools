"""Flexi-cover construction for jackpot pools (carryover strategy, B7).

Given a per-leg finishing-order model (a :class:`~hkjc.simulation.RaceSimulator`
per leg, e.g. from the closing win odds) and the pool economics on a carryover
meeting, pick the Trio "box" cover — a set of horses per leg — that maximizes EV.

A box cover picks a set ``S_i`` of horses in each leg and buys every trio within
it: ``n_combos = prod_i C(|S_i|, 3)``. It hits iff every leg's actual top-3 falls
inside that leg's set. Per unit stake (small-stake, own-impact ignored):

    hit_prob = prod_i P(top-3 of leg i  ⊆  S_i)
    EV       = hit_prob · expected_dividend · dilution  −  n_combos        (units of stake)

``expected_dividend`` is the dividend per unit stake you expect to collect if you
hit (anchor on HKJC's ``estimated_dividend``); ``dilution`` ∈ (0, 1] haircuts it
for late/chase money landing on your combo — **the crux knob, calibrated later
from the capture flywheel (B6).** This module is the optimizer; the dilution model
is separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import List, Sequence

import numpy as np

from ..simulation import RaceSimulator


def top3_coverage(sim: RaceSimulator, horses: Sequence[int]) -> float:
    """P(all three placed horses of the leg fall within ``horses``) (0-based indices)."""
    members = np.zeros(sim.n, dtype=bool)
    members[list(horses)] = True
    top3 = sim.orders[:, :3]
    return float(members[top3].all(axis=1).mean())


def leg_horses_by_place(sim: RaceSimulator, k: int) -> List[int]:
    """The ``k`` horses most likely to finish in the top 3 (descending)."""
    place_prob = (sim.positions < 3).mean(axis=0)
    return np.argsort(-place_prob)[:k].tolist()


@dataclass
class Cover:
    leg_sets: List[List[int]]          # 0-based horse indices per leg
    set_sizes: List[int]
    n_combos: int
    hit_prob: float
    cost: float
    expected_return: float
    ev: float
    leg_coverage: List[float] = field(default_factory=list)

    def runner_sets(self) -> List[List[int]]:
        """The cover as 1-based HKJC runner numbers per leg."""
        return [[h + 1 for h in s] for s in self.leg_sets]


def evaluate_cover(
    leg_sims: Sequence[RaceSimulator],
    leg_sets: Sequence[Sequence[int]],
    expected_dividend: float,
    dilution: float = 1.0,
    stake: float = 1.0,
) -> Cover:
    """EV of a specific box cover (per-leg horse sets) at the given pool economics."""
    n_combos = 1
    hit = 1.0
    coverage = []
    for sim, horse_set in zip(leg_sims, leg_sets):
        n_combos *= comb(len(horse_set), 3)
        c = top3_coverage(sim, horse_set)
        coverage.append(c)
        hit *= c
    cost = n_combos * stake
    exp_ret = hit * expected_dividend * dilution * stake
    return Cover(
        leg_sets=[list(s) for s in leg_sets], set_sizes=[len(s) for s in leg_sets],
        n_combos=n_combos, hit_prob=hit, cost=cost, expected_return=exp_ret,
        ev=exp_ret - cost, leg_coverage=coverage,
    )


def optimize_cover(
    leg_sims: Sequence[RaceSimulator],
    expected_dividend: float,
    bankroll: float,
    dilution: float = 1.0,
    stake: float = 1.0,
    max_set: int = 10,
) -> Cover:
    """Greedily grow the box cover to maximize EV under a ``bankroll`` cost cap.

    Starts from the minimal top-3-each-leg cover and repeatedly expands whichever
    leg's +1 horse buys the most EV, stopping when no expansion helps or the cost
    cap binds. Returns the best :class:`Cover` found (may have EV ≤ 0 — that itself
    is the answer: no profitable cover exists at these economics).
    """
    n_legs = len(leg_sims)
    sizes = [3] * n_legs
    sets = [leg_horses_by_place(s, 3) for s in leg_sims]
    best = evaluate_cover(leg_sims, sets, expected_dividend, dilution, stake)

    while True:
        best_gain, choice = 0.0, None
        for i in range(n_legs):
            if sizes[i] >= min(max_set, leg_sims[i].n):
                continue
            trial_sizes = sizes.copy()
            trial_sizes[i] += 1
            trial_sets = [leg_horses_by_place(leg_sims[j], trial_sizes[j]) for j in range(n_legs)]
            cand = evaluate_cover(leg_sims, trial_sets, expected_dividend, dilution, stake)
            if cand.cost <= bankroll and (cand.ev - best.ev) > best_gain:
                best_gain, choice = cand.ev - best.ev, (trial_sizes, trial_sets, cand)
        if choice is None:
            return best
        sizes, sets, best = choice
