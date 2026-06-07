"""PathLikelihood — p(observed measurement surface | θ, conditions). The IDENTIFY objective.

Layer-3 objective of the canonical framework: the PATH-RECONSTRUCTION loss. We fit θ (and the
generative params) to the observed *measurement surface* — not the win bit, which #40/#41 proved
is too coarse to identify rich mechanism. The surface is built from pluggable TERMS, each a
log-likelihood of one observable channel given θ:

    profile  : the K-section velocity profile (#41; the proven term — +22–30% over the scalar
               heuristic, and the channel that surfaced `a`). REUSES energetics_dynamic.profile_loglik.
    lateral  : (Stage 3, STUB) the across-track path y_lat(t) from CV — identifies control/geometry.
    margin   : (Stage 2, STUB) the beaten-margins surface — denser than win, sharpens the order law.

A term is a :class:`SurfaceTerm` (a callable + a name + a stage gate). :class:`PathLikelihood`
sums the active terms. This is the OBJECTIVE used to IDENTIFY θ honestly from the path — the GATE
(market skill% over the pool book) lives separately in :mod:`hkjc.model.gate`, per the #40
lesson that honest path-identified strength must be coupling-CALIBRATED for win, not used raw.

LEAK-FREE: a term scores a run's OWN measurement surface against θ — and the θ it is handed is
strictly-prior (the population fits θ from a horse's EARLIER runs; see :mod:`pipeline`). The
surface terms here are deterministic per-run transforms (no other horse, no outcome), exactly the
status of ``energetics_dynamic`` recoveries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .conditions import RaceConditions
from .latent import LatentState


class SurfaceTermFn(Protocol):
    """A single surface log-likelihood channel: log p(observation | θ, conditions)."""

    def __call__(self, theta: LatentState, conditions: RaceConditions,
                 observation: dict) -> float:
        ...


@dataclass(frozen=True)
class SurfaceTerm:
    """A named, stage-gated surface log-likelihood term.

    ``stage`` records which build stage owns the term (1 profile, 2 margins, 3 lateral) so the
    pipeline can switch terms on as stages land. ``weight`` lets the objective up/down-weight a
    channel (e.g. the margin term once calibrated). ``fn`` is the actual log-likelihood.
    """

    name: str
    stage: int
    fn: SurfaceTermFn
    weight: float = 1.0


def profile_term(*, sigma: float = 0.06, unit: str = "dist", rho: float = 0.0) -> SurfaceTerm:
    """The Stage-1 PROFILE term — REUSES ``energetics_dynamic.profile_loglik`` (#41).

    Scores log p(observed K-section velocity profile | v_c, W′_0, a) under the production
    log-velocity Gaussian noise model. The observation dict must carry ``section_times`` and
    ``section_dists`` (the run's own public sectionals). This is the *proven* identifying term.
    """
    from hkjc.model.energetics_dynamic import profile_loglik

    def _fn(theta: LatentState, conditions: RaceConditions, observation: dict) -> float:
        return profile_loglik(
            observation["section_times"], observation["section_dists"],
            theta.v_c, theta.w_prime, theta.a,
            sigma=sigma, unit=unit, rho=rho,
        )

    return SurfaceTerm(name="profile", stage=1, fn=_fn)


def crossed_term() -> SurfaceTerm:
    """The CAUSAL crossed-effects term (STUB) — log p(profile | physiology, control, prep).

    The causal-rebuild successor to :func:`profile_term` (``docs/CAUSAL_MODEL.md`` §3.2): instead of
    attributing the whole systematic path to one horse latent, it scores the path against the
    CROSS-CLASSIFIED decomposition — a horse physiology effect + a crossed jockey control effect + a
    crossed trainer×time prep effect — so cause (ability) is separated from cause (control/prep). The
    FIRST BUILD (``docs/CAUSAL_MODEL.md`` §6) does NOT need this (physiology-only via
    :func:`profile_term`); this is the Stage-2/3 bridge. Inference runs on #50's corrected generator.
    """

    def _fn(theta: LatentState, conditions: RaceConditions, observation: dict) -> float:
        raise NotImplementedError(
            "Causal crossed-effects surface term — see docs/CAUSAL_MODEL.md §3 (Stage 2/3). "
            "The first build uses profile_term (physiology only)."
        )

    return SurfaceTerm(name="crossed", stage=2, fn=_fn)


def margin_term() -> SurfaceTerm:
    """The Stage-2 MARGIN term (STUB) — log p(beaten margins | order law).

    Denser than the win bit: the finishing MARGINS pin the spacing of the joint order law, not
    just its argmax. BUILD-WAVE: derive the margin likelihood from the Wald first-passage times
    the generative model already samples (margin_ij ∝ ΔT_ij·v), and score the observed margins.
    """

    def _fn(theta: LatentState, conditions: RaceConditions, observation: dict) -> float:
        raise NotImplementedError("Stage-2 margin surface term — see BUILD-WAVE in MODEL_DESIGN.md")

    return SurfaceTerm(name="margin", stage=2, fn=_fn)


def lateral_term() -> SurfaceTerm:
    """The Stage-3 LATERAL term (STUB) — log p(across-track path y_lat(t) | θ, geometry).

    The CV-recovered lateral path identifies the control/ground-saved latents (THEORY §3c). BUILD-
    WAVE: define the lateral observation model once CV paths land, add the latents to ``LatentState``.
    """

    def _fn(theta: LatentState, conditions: RaceConditions, observation: dict) -> float:
        raise NotImplementedError("Stage-3 lateral surface term — see BUILD-WAVE in MODEL_DESIGN.md")

    return SurfaceTerm(name="lateral", stage=3, fn=_fn)


@dataclass
class PathLikelihood:
    """The path-reconstruction objective: a weighted sum of active surface terms.

    ``log_prob(theta, conditions, observation)`` returns Σ_term weight·log p(channel | θ). The
    IDENTIFY half of the two-objective loop: maximise this over θ (and shared generative params)
    to recover mechanism from the measurement surface. The GATE (pool-book skill%) validates and
    coupling-calibrates separately — they are deliberately different objects.
    """

    terms: list[SurfaceTerm]

    @classmethod
    def stage1(cls, **profile_kwargs) -> "PathLikelihood":
        """Stage-1 objective: the profile term only (the proven, runnable channel)."""
        return cls(terms=[profile_term(**profile_kwargs)])

    def active(self, max_stage: int) -> list[SurfaceTerm]:
        return [t for t in self.terms if t.stage <= max_stage]

    def log_prob(self, theta: LatentState, conditions: RaceConditions,
                 observation: dict, *, max_stage: int = 1) -> float:
        """Total surface log-likelihood at θ (only terms up to ``max_stage`` active)."""
        total = 0.0
        for term in self.active(max_stage):
            total += term.weight * term.fn(theta, conditions, observation)
        return total

    def identify(
        self,
        conditions: RaceConditions,
        observation: dict,
        *,
        recover: Callable[[dict], LatentState] | None = None,
        max_stage: int = 1,
    ) -> LatentState:
        """Recover the θ that best explains this run's surface (the per-run point identification).

        Stage 1 uses the CLOSED-FORM deterministic recovery (``energetics_dynamic.estimate_run_profile``
        via the default ``recover``) rather than an optimiser — the #41 recipe, which IS the
        argmax of the profile likelihood for the dynamic engine. Later stages may replace
        ``recover`` with a true argmax over :meth:`log_prob` once non-profile terms are active.
        Note: this recovers a run's OWN θ for population pooling; the per-RACE prediction uses the
        strictly-prior pooled θ from :class:`~hkjc.model.latent.PopulationPrior`.
        """
        if recover is None:
            recover = _default_profile_recover
        return recover(observation)


def _default_profile_recover(observation: dict) -> LatentState:
    """Closed-form Stage-1 recovery of (v_c, W′_0, a) from a run's sectionals (#41)."""
    from hkjc.model.energetics_dynamic import estimate_run_profile

    e = estimate_run_profile(
        np.asarray(observation["section_times"], float),
        np.asarray(observation["section_dists"], float),
    )
    return LatentState(v_c=e.v_c, w_prime=e.w_prime, rho=0.0, a=e.accel)
