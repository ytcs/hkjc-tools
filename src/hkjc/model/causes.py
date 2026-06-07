"""LatentCauses — the CAUSAL successor to :class:`~hkjc.model.latent.LatentState`.

DESIGN STUB ONLY (the causal rebuild — ``docs/CAUSAL_MODEL.md``). This is the interface surface
for the cause-not-effect architecture: the per-runner state is split into the three crossed CAUSES
that jointly generate a run's path — horse PHYSIOLOGY, jockey CONTROL, trainer PREP — plus the
exogenous ENVIRONMENT. The meta-signals (form, jockey-SR, collateral, peak-gap, speed-fig,
trainer-hot) are NOT here: they are downstream CONSEQUENCES of these causes, to be RE-DERIVED and
validated by re-emergence (``docs/CAUSAL_MODEL.md`` §4), never fed as inputs.

THE CONTRAST WITH ``LatentState``. ``LatentState`` is the per-horse latent θ as identified by the
single-horse profile recovery — it attributes the WHOLE systematic path to the horse.
``LatentCauses`` is the causal decomposition: it splits the path's systematic level across the
``(horse, jockey, trainer)`` triple via a CROSS-CLASSIFIED random-effects hierarchy over the path
likelihood (``docs/CAUSAL_MODEL.md`` §3). The horse effect is the physiology cause; the jockey
effect is the (coarse, longitudinal) control cause; the trainer×time effect is the prep cause.

IDENTIFIABILITY (honest — ``docs/CAUSAL_MODEL.md`` §3.3):
  * horse physiology  — identifiable NOW from the longitudinal profile (0040/0044).
  * trainer prep      — PARTIAL now (separable where crossings are dense).
  * jockey control    — WEAK now (control is fundamentally LATERAL); the clean split needs CV (#34,
                        Stage 3). The longitudinal jockey effect carries ability leakage.

THE FIRST BUILD uses the HORSE effect only (physiology from the corrected path; ``docs/CAUSAL_MODEL.md``
§6). The crossed jockey/trainer inference is STUBBED here (Stage 2/3) — every method that would
require it raises ``NotImplementedError`` with a pointer to the design.

LEAK-FREE: as with ``LatentState``, a ``LatentCauses`` is a pure value object carrying no outcome
and no other-horse information; every crossed effect is produced strictly-prior, train-only per fold
(the pipeline owns the boundary). All inference runs through #50's CORRECTED unbiased generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .latent import LatentState


@dataclass(frozen=True)
class PhysiologyCause:
    """The horse PHYSIOLOGY cause — intrinsic, slowly-varying (the per-horse random effect).

    The bulk of the systematic path and the only fully-identifiable cause now (0040/0044). ``D`` is
    the per-horse consistency (own finishing-time variance); ``m`` the allometric mass (``v_c ∝
    m^(b-1)``). ``rho`` is recovery (population-fixed now; per-horse is a CV/margins target).
    """

    v_c: float
    w_prime: float
    a: float = 1.0
    rho: float = 0.0
    d: float = 0.0
    mass: float = 0.0

    def as_latent_state(self) -> LatentState:
        """The physiology-only ``LatentState`` the generative law consumes in the FIRST BUILD."""
        return LatentState(v_c=self.v_c, w_prime=self.w_prime, rho=self.rho, a=self.a,
                           extra={"d": self.d, "mass": self.mass})


@dataclass(frozen=True)
class ControlCause:
    """The jockey CONTROL cause — path-efficiency / pace policy (the crossed rider random effect).

    Fundamentally LATERAL (``c_lat`` ground-saved) + intertemporal (``c_pol`` commit timing). Only
    WEAKLY identifiable from the longitudinal profile (confounded with horse ``v_c``); the clean
    split is CV-only (#34, Stage 3). STUBBED until then.
    """

    c_lat: float = 0.0   # lateral path-efficiency / ground-saved (CV-only; Stage 3)
    c_pol: float = 0.0   # pace/effort policy — when reserve is committed (intertemporal)


@dataclass(frozen=True)
class PrepCause:
    """The trainer CONDITIONING / day-fitness cause (the crossed trainer×time random effect).

    A slow level shift in how race-ready a yard brings its horses. PARTIALLY identifiable now where
    a trainer's horses crowd the crossings; STUBBED inference (Stage 2).
    """

    fitness: float = 0.0     # day-fitness level shift
    trajectory: float = 0.0  # conditioning trajectory (improving vs holding)


@dataclass(frozen=True)
class LatentCauses:
    """The crossed causal state for one runner: physiology + control + prep + environment.

    The causal successor to ``LatentState``. The FIRST BUILD populates ``physiology`` only (from the
    corrected path) and leaves ``control``/``prep`` at their population-neutral defaults; Stage 2/3
    fill them via the crossed-effects inference (:func:`infer_crossed_causes`, STUBBED).
    """

    physiology: PhysiologyCause
    control: ControlCause = field(default_factory=ControlCause)
    prep: PrepCause = field(default_factory=PrepCause)
    env: Mapping[str, float] = field(default_factory=dict)

    def as_latent_state(self) -> LatentState:
        """Collapse to the generative ``LatentState``.

        FIRST BUILD: physiology only (the strip — ``docs/CAUSAL_MODEL.md`` §6). When control/prep
        are inferred (Stage 2/3) they modulate the drift here; until then they are neutral.
        """
        return self.physiology.as_latent_state()


def infer_crossed_causes(*args, **kwargs) -> "LatentCauses":
    """Crossed-effects inference of (physiology, control, prep) from the path likelihood — STUB.

    The Stage-2/3 inference (``docs/CAUSAL_MODEL.md`` §3.2): a cross-classified random-effects
    hierarchy over the path likelihood splitting the systematic path across the (horse, jockey,
    trainer) crossing, train-only per fold, strictly-prior, on #50's corrected generator. The FIRST
    BUILD does NOT need this (physiology-only); it is the bridge to control/prep re-derivation.
    """
    raise NotImplementedError(
        "Crossed-effects causal inference (jockey/trainer) — see docs/CAUSAL_MODEL.md §3 "
        "(Stage 2 trainer-prep; Stage 3 jockey-control with CV lateral). The FIRST BUILD uses "
        "PhysiologyCause only (docs/CAUSAL_MODEL.md §6)."
    )
