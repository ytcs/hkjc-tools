"""Monte-Carlo race simulation.

The simulator draws many finishing orders from a :class:`~hkjc.model.RaceModel`
and lets the caller estimate the probability and expected value of *any* bet by
expressing it as a boolean event over those samples. Nothing about specific bet
types or staking is baked in: the agent composes events with ordinary array
logic (``&``, ``|``, ``~``) and supplies her own payouts.

Indices are 0-based throughout; horse index ``i`` corresponds to HKJC runner
number ``i + 1`` (consistent with the odds/strength arrays elsewhere in the
library).

Example::

    from hkjc.model import PlackettLuceModel
    from hkjc.simulation import RaceSimulator

    sim = RaceSimulator(PlackettLuceModel.from_odds(odds), n_samples=200_000)

    # Probability that runner 3 (index 2) wins:
    sim.prob(sim.wins(2))

    # EV of a $1 win bet on runner 3 at decimal odds 5.0:
    sim.expected_profit(sim.wins(2), 5.0)

    # A custom event: runner 3 wins AND runner 5 finishes in the top 3:
    sim.prob(sim.wins(2) & sim.places(4, k=3))
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .model import RaceModel


class RaceSimulator:
    """Estimate bet probabilities and EVs from sampled finishing orders.

    Args:
        model: Any :class:`~hkjc.model.RaceModel`.
        n_samples: Number of finishing orders to draw (more = lower variance).
        seed: Seed for the internal RNG, for reproducibility.

    Attributes:
        orders (np.ndarray): ``(n_samples, n)`` sampled finishing orders;
            ``orders[s, r]`` is the index of the horse finishing in position ``r``.
        positions (np.ndarray): ``(n_samples, n)`` finishing positions;
            ``positions[s, h]`` is the finishing position (0 = winner) of horse ``h``.
    """

    def __init__(self, model: RaceModel, n_samples: int = 100_000, seed: int = 0) -> None:
        self.model = model
        self.n = model.n
        self.n_samples = n_samples
        rng = np.random.default_rng(seed)
        self.orders = model.sample_orders(n_samples, rng)
        # positions[s, h] = rank of horse h, since orders[s] is a permutation.
        self.positions = self.orders.argsort(axis=1)

    # -- core estimators ---------------------------------------------------

    def prob(self, event: np.ndarray) -> float:
        """Estimated probability of a boolean ``event`` of shape ``(n_samples,)``."""
        return float(np.asarray(event, dtype=bool).mean())

    def expected_return(self, payout: np.ndarray) -> float:
        """Mean per-sample ``payout`` (shape ``(n_samples,)``); for arbitrary dividends."""
        return float(np.asarray(payout, dtype=float).mean())

    def expected_profit(self, event: np.ndarray, decimal_odds: float, stake: float = 1.0) -> float:
        """EV of a fixed-odds bet returning ``decimal_odds`` per unit staked on ``event``.

        Uses HKJC/parimutuel convention where ``decimal_odds`` includes the stake
        (odds of 5.0 returns $5 per $1 on a win), so EV per unit = ``p * odds - 1``.
        """
        return (self.prob(event) * decimal_odds - 1.0) * stake

    # -- generic outcome predicates ---------------------------------------
    # Each returns a boolean array of shape (n_samples,). They are building
    # blocks, not strategies: combine them freely to express any bet.

    def wins(self, horse: int) -> np.ndarray:
        """``horse`` finishes first."""
        return self.positions[:, horse] == 0

    def finishes(self, horse: int, position: int) -> np.ndarray:
        """``horse`` finishes exactly in ``position`` (0 = first)."""
        return self.positions[:, horse] == position

    def places(self, horse: int, k: int = 3) -> np.ndarray:
        """``horse`` finishes in the top ``k`` (default 3, the HKJC place pool)."""
        return self.positions[:, horse] < k

    def exact_order(self, sequence: Iterable[int]) -> np.ndarray:
        """The leading finishers are exactly ``sequence``, in that order.

        Covers straight forecast (2), trifecta (3), first-4 (4), etc.
        """
        seq = list(sequence)
        hit = np.ones(self.n_samples, dtype=bool)
        for pos, horse in enumerate(seq):
            hit &= self.positions[:, horse] == pos
        return hit

    def top_set(self, horses: Iterable[int], k: int | None = None) -> np.ndarray:
        """``horses`` occupy the top ``k`` positions, in any order.

        With ``k`` defaulting to ``len(horses)`` this covers quinella (2),
        trio (3) and quartet (4) box bets.
        """
        horses = list(horses)
        k = len(horses) if k is None else k
        in_top = self.positions[:, horses] < k
        return in_top.all(axis=1)

    def all_place(self, horses: Iterable[int], k: int = 3) -> np.ndarray:
        """Every horse in ``horses`` finishes in the top ``k``."""
        horses = list(horses)
        return (self.positions[:, horses] < k).all(axis=1)

    def any_place(self, horses: Iterable[int], k: int = 3) -> np.ndarray:
        """At least one horse in ``horses`` finishes in the top ``k``."""
        horses = list(horses)
        return (self.positions[:, horses] < k).any(axis=1)

    # -- convenience -------------------------------------------------------

    def place_matrix(self) -> np.ndarray:
        """Empirical position-probability matrix ``P[i, j] = P(horse i finishes j-th)``.

        The Monte-Carlo counterpart of the exact Harville ``P`` matrix; useful for
        sanity-checking a model or feeding position probabilities to the agent.
        """
        counts = np.zeros((self.n, self.n))
        for j in range(self.n):
            idx, c = np.unique(self.orders[:, j], return_counts=True)
            counts[idx, j] = c
        return counts / self.n_samples
