"""Plackett-Luce model — the reference :class:`~hkjc.model.base.RaceModel`.

It is the generative law implied by the Harville model: each finishing order is
drawn by repeatedly picking the next finisher with probability proportional to the
remaining runners' strengths. Sampling is vectorised via the Gumbel-max trick, so
drawing hundreds of thousands of full race orders is a single argsort.
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import RaceModel


class PlackettLuceModel(RaceModel):
    """Plackett-Luce model parameterised by per-runner strengths.

    The probability of a finishing order is the product, down the order, of each
    runner's strength divided by the total strength still in contention. This is
    exactly the Harville finishing-order law, so a model built from a Harville fit
    reproduces that fit's win/place/quinella probabilities.

    Args:
        strengths: Positive strength per runner, shape ``(n,)``. Need not be
            normalised; only ratios matter.
    """

    def __init__(self, strengths: np.ndarray) -> None:
        strengths = np.asarray(strengths, dtype=np.float64)
        if strengths.ndim != 1:
            raise ValueError("strengths must be a 1D array")
        if np.any(strengths <= 0) or not np.all(np.isfinite(strengths)):
            raise ValueError("strengths must be finite and strictly positive")

        self.strengths = strengths / strengths.sum()
        self.n = strengths.shape[0]
        self._log_strengths = np.log(self.strengths)

    def sample_orders(self, size: int, rng: np.random.Generator) -> np.ndarray:
        # Gumbel-max trick: perturbing log-strengths with i.i.d. Gumbel noise and
        # ranking the result yields exact Plackett-Luce samples.
        gumbel = rng.gumbel(size=(size, self.n))
        scores = self._log_strengths[None, :] + gumbel
        return np.argsort(-scores, axis=1).astype(np.int64)

    @classmethod
    def from_harville_fit(cls, fit: dict) -> "PlackettLuceModel":
        """Build from the dict returned by :func:`hkjc.model.harville_model.fit_harville_to_odds`."""
        return cls(fit["prob_fit"]["theta"])

    @classmethod
    def from_odds(cls, odds: dict[str, np.ndarray], **fit_kwargs) -> "PlackettLuceModel":
        """Fit a Harville model to market ``odds`` and wrap the result.

        Args:
            odds: Dict of decimal-odds arrays keyed by pool type
                (``'WIN'``, ``'PLA'``, ``'QIN'``, ``'QPL'``).
            **fit_kwargs: Forwarded to :func:`hkjc.model.harville_model.fit_harville_to_odds`
                (e.g. ``lambdas``, ``takeout_rate``).

        Raises:
            RuntimeError: If the Harville fit fails to converge.
        """
        from .harville_model import fit_harville_to_odds

        fit = fit_harville_to_odds(odds, **fit_kwargs)
        if not fit["success"]:
            raise RuntimeError(f"Harville fit failed: {fit.get('message', '')}")
        return cls.from_harville_fit(fit)
