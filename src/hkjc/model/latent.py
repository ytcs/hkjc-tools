"""The per-horse latent attribute state θ and its hierarchical population prior.

THE STATE. A runner is summarised by a small vector of *mechanistic* latents — every
component a real physiological/control quantity, never an arbitrary regressor (the
canonical-framework discipline). The current axes (extensible — add a field, a prior, a
generative consequence):

    v_c      aerobic critical velocity (m/s)        — the settled sustainable level (THEORY §2.1)
    w_prime  anaerobic reserve W′_0 (dist-unit m)    — the finite tank spent above v_c (THEORY §2.2)
    rho      recovery rate (1/s, FIXED at pop level) — reserve refilled below v_c (#38 dynamic engine)
    a        acceleration / gate ramp (dimensionless)— opening-split ramp v_1/v_cruise (#41; THEORY §3e Phase 1)

`rho` is a POPULATION constant in Stage 1 (the #38 engine fixes it), carried on the state so
the generative model has a single place to read it; per-horse `rho` is a later-stage CV target
(control/ground-saved latents join the same vector then). `a` is the #41 latent the rich
profile likelihood identifies and win alone cannot (population-invisible, opening-split
fingerprint) — the Stage-1 promotion candidate.

LEAK-FREE BY CONSTRUCTION. A :class:`LatentState` is *just a value object*: it never knows an
outcome or another horse. The only way to obtain a per-horse state is through
:class:`PopulationPrior` from a horse's STRICTLY-PRIOR runs (see :meth:`PopulationPrior.posterior`),
mirroring ``calibration.horse_energetics_trailing`` — a race's own θ can never enter the fit
that predicts it. The population itself must be fit train-only per walk-forward fold; the
contract pushes that decision up to the pipeline (:mod:`hkjc.model.pipeline`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

# The canonical θ axes, in vector order. EXTENSION POINT: append a name here, give it a prior
# in PopulationPrior and a generative consequence in GenerativeRaceModel — nothing else couples
# to the literal order except `as_array` / `from_array`.
THETA_AXES: tuple[str, ...] = ("v_c", "w_prime", "rho", "a")


@dataclass(frozen=True)
class LatentState:
    """One runner's latent attribute vector θ = (v_c, W′_0, ρ, a, …).

    A pure value object: positive, finite, mechanism-named scalars. Carries NO outcome and
    NO other-horse information — it is the unit the generative model consumes. Construct it
    only via :meth:`PopulationPrior.posterior` (strictly-prior) or :meth:`from_array` in tests.

    ``extra`` holds future axes (control, ground-saved) added by later stages without
    breaking the frozen core; they round-trip through ``as_array``/``from_array`` in sorted
    key order after the canonical axes.
    """

    v_c: float
    w_prime: float
    rho: float = 0.0
    a: float = 1.0
    extra: Mapping[str, float] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        """θ as a flat float vector in ``THETA_AXES`` order, then sorted ``extra`` values."""
        core = [getattr(self, k) for k in THETA_AXES]
        ext = [self.extra[k] for k in sorted(self.extra)]
        return np.asarray(core + ext, dtype=float)

    @classmethod
    def from_array(cls, vec: np.ndarray, extra_keys: tuple[str, ...] = ()) -> "LatentState":
        """Inverse of :meth:`as_array`. ``extra_keys`` names the trailing components (sorted)."""
        vec = np.asarray(vec, dtype=float)
        n = len(THETA_AXES)
        core = dict(zip(THETA_AXES, vec[:n]))
        ext = dict(zip(sorted(extra_keys), vec[n:]))
        return cls(**core, extra=ext)

    def with_(self, **changes: float) -> "LatentState":
        """Copy with overridden core axes (frozen-safe; used by coupling calibration)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class PopulationPrior:
    """Hierarchical population prior over θ — the train-only Empirical-Bayes population.

    THIN ADAPTER over the production EB hierarchy (``calibration.EnergeticsPopulation``: the
    band×rating Normal–Normal random-effects fit) so the state-space package REUSES the
    validated population rather than re-deriving it. The adapter adds:

      * the canonical-framework framing (this IS the population prior p(θ) of Layer 1), and
      * the `a` / `rho` axes the EB hierarchy does not yet model — for Stage 1 these come from
        a per-horse strictly-prior trailing mean (the #41 pilot's leak-free recipe), shrunk to
        a population mean; later stages fold them into the random-effects fit proper.

    The instance is immutable and must be built from TRAIN ROWS ONLY (the pipeline enforces
    the fold boundary). ``a_pop``/``rho_pop`` are the population fallbacks used when a horse has
    no prior history (a debut runner) — exactly the rating-conditioned-mean fallback the
    production fit uses for v_c/W′.
    """

    energetics: object  # calibration.EnergeticsPopulation (typed loosely to avoid a hard import)
    a_pop: float = 1.0
    rho_pop: float = 0.0

    def prior_mean(self, band: str, rating: float | None = None) -> LatentState:
        """Population-mean θ for a runner's trip band and rating (the no-history fallback)."""
        return LatentState(
            v_c=self.energetics.v_c.fixed(band, rating),
            w_prime=self.energetics.w_prime.fixed(band, rating),
            rho=self.rho_pop,
            a=self.a_pop,
        )

    def posterior(
        self,
        band: str,
        rating: float | None,
        *,
        n_prior: float,
        resid_v_c: float,
        resid_w_prime: float,
        a_prior: float | None = None,
    ) -> LatentState:
        """Strictly-prior partially-pooled θ for one runner.

        ``n_prior`` is the count of that horse's EARLIER runs and ``resid_*`` the mean EB
        residual over them (the leak-free quantities ``calibration.horse_energetics_trailing``
        produces). The shrinkage λ = τ²/(τ² + σ²/n) is the production EB weight: a debut horse
        (n_prior=0) returns the population mean; an established horse trusts its own residual.

        `a` uses the horse's strictly-prior trailing mean (``a_prior``) when available, else the
        population `a_pop` — the #41 leak-free recipe. ρ is the fixed population value.
        """
        vc_re, wp_re = self.energetics.v_c, self.energetics.w_prime
        lam_vc = vc_re.tau2 / (vc_re.tau2 + vc_re.sigma2 / n_prior) if n_prior > 0 else 0.0
        lam_wp = wp_re.tau2 / (wp_re.tau2 + wp_re.sigma2 / n_prior) if n_prior > 0 else 0.0
        return LatentState(
            v_c=vc_re.fixed(band, rating) + lam_vc * resid_v_c,
            w_prime=wp_re.fixed(band, rating) + lam_wp * resid_w_prime,
            rho=self.rho_pop,
            a=a_prior if (a_prior is not None and np.isfinite(a_prior)) else self.a_pop,
        )
