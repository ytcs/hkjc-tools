"""The ``RaceModel`` contract.

A :class:`RaceModel` is anything that can generate finishing orders for a race.
That single capability is all the Monte-Carlo simulator (:mod:`hkjc.simulation`)
needs, so the agent can swap in *any* model — Plackett-Luce fitted to market odds,
a results-trained model, a finish-time model — without touching the rest of the
stack. The only contract is :meth:`RaceModel.sample_orders`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RaceModel(ABC):
    """Abstract race outcome model.

    Implementations must expose the field size ``n`` and a vectorised
    :meth:`sample_orders`. Everything downstream (probability and EV estimation for
    arbitrary bet types) is derived from samples, so a custom model only has to know
    how to produce plausible finishing orders.
    """

    n: int

    @abstractmethod
    def sample_orders(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` finishing orders.

        Returns:
            np.ndarray: integer array of shape ``(size, n)``. Each row is a
            permutation of ``0..n-1`` giving a finishing order: row ``[0]`` is the
            (0-indexed) winner, row ``[1]`` the runner-up, and so on.
        """
        raise NotImplementedError
