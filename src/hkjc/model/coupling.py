"""CouplingCalibrator — map honestly-identified θ into the predictive race model with
GATE-CALIBRATED coupling strengths (the #40 lesson, operationalised).

THE #40 LESSON. θ identified honestly from the path (the IDENTIFY objective) is the *mechanistically
correct* state — but honest posteriors UNDER-discriminate for WIN (an easy winner eases down, so its
honest v_c under-reads; an honest `a` is population-invisible). So the strength with which each latent
COUPLES into the predictive race model is NOT its honest magnitude — it is a coupling calibrated on the
GATE (the pool-book market skill%). IDENTIFY honestly; CALIBRATE coupling on the pool book.

This object is the seam between Layer 1 (identified θ) and Layer 2 (the generative law): it takes the
strictly-prior θ per runner and produces the (μ, λ) the :class:`~hkjc.model.generative.GenerativeRaceModel`
samples, applying per-latent coupling gains. Stage 1 calibrates the SINGLE `a`→start-phase coupling
(``a_start_gain``); the production v_c/W′ couplings are already baked, so the skeleton folds them in
unchanged and exposes `a_start_gain` as the one knob the gate tunes.

The coupling gains live here (a small, named, mechanism-keyed set — never an arbitrary regressor stack)
and are fit by the GATE harness (:mod:`hkjc.model.gate`) via the leak-free conversion test, exactly
as the production constants are tuned. ``a_start_gain=0`` ⇒ `a` is OFF (exact production-equivalent μ),
the honest Stage-0 baseline the gate must beat to PROMOTE `a`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conditions import RaceConditions
from .generative import GenerativeRaceModel
from .latent import LatentState

# Start-phase reference length (THEORY §3e Phase 1): the start is a roughly fixed acceleration
# transient, so its share of the trip ∝ 1/L ⇒ the `a` correction bites in SPRINTS. Sourced from
# the production GATE_BREAK_LREF (the same start-phase reference the production gate-break term
# uses), so the `a`→start-phase coupling shares the production sprint reference (~1200m).
from hkjc.model.wald_constants import GATE_BREAK_LREF as _START_LREF


@dataclass(frozen=True)
class CouplingGains:
    """The gate-calibrated coupling strengths (Stage-1 set). Mechanism-keyed, gate-tuned.

    ``a_start_gain`` — how strongly the opening ramp `a` shortens/lengthens the effective trip via
    the start phase (the DRIFT channel). 0 ⇒ OFF (production-equivalent). The GATE tunes this on the
    pool book; a value that beats a=OFF on market-skill% PROMOTES `a`.

    ``a_d_gain`` — the parallel `a`→DIFFUSION-D variance channel (Stage-1.5): a slow/troubled opener
    (a<field, a_c<0) injects START-PHASE variance — D_i *= (1 − a_d_gain·start_frac·ā_centred), so a
    poor opener raises D (more upsets), a clean opener lowers it. Sprint-biased by the same start_frac.
    0 ⇒ OFF. Addresses the symptom the drift channel left (it moved the order but hurt calibration).

    ``rho_gain`` — the RECOVERY latent ρ → LATE-RESERVE deployment channel (RQ-rho / journal 0044).
    A high-ρ horse sits off a hot pace BELOW v_c, banks reserve, and deploys it in the closing kick — a
    CLOSER that gets up late for the minor placings. We give such a runner MORE effective usable reserve
    in the closing phase: the reserve boost (1+W′/L) is scaled by (1 + rho_gain·ρ_centred), so a
    high-ρ closer reaches a lower effective μ (a faster effective target velocity from the banked
    reserve) and a low-ρ front-burner reaches a higher one. ρ_centred is the within-race-centred
    strictly-prior recovery proxy (a horse can only be a *relative* closer in its field). 0 ⇒ OFF
    (production-equivalent μ). This is a BODY-mover: it reshuffles the 2nd/3rd/4th placings the
    PLA/QIN/FCT pools price — the OPPOSITE profile to `a` (a top-mover). The gate tunes ``rho_gain``;
    it must be tested INCREMENTAL over the baked FADE_W/POSFADE_W couplings (which already shadow late
    reserve from fade/early-position) — if it merely re-expresses them it is collinear, not new signal.
    """

    a_start_gain: float = 0.0
    a_d_gain: float = 0.0
    rho_gain: float = 0.0


class CouplingCalibrator:
    """Turn strictly-prior θ + conditions into the (μ, λ) of the predictive law, with calibrated gains.

    The gains are NOT the honest path-identified magnitudes — they are tuned on the GATE (#40). This
    object is where every latent's win/pool coupling is set, keeping the generative law honest about
    the *mechanism* (the μ/λ structure) while the *strength* is gate-calibrated.
    """

    def __init__(self, gains: CouplingGains | None = None) -> None:
        self.gains = gains or CouplingGains()

    def couple(self, thetas: list[LatentState], conditions: RaceConditions,
               model: GenerativeRaceModel) -> tuple[np.ndarray, np.ndarray]:
        """Compute the gate-calibrated (μ, λ) for the field.

        Starts from the model's bare drift/diffusion (the production-equivalent v_c/W′ law), then
        applies the Stage-1 `a`→START-PHASE coupling: a slower opening ramp (a<1, still accelerating
        off the gate) lengthens the effective trip; the correction is weighted by the start's share
        of the trip (∝ 1/L, sprint-biased) and the gate-calibrated ``a_start_gain``. Field-centred so
        a uniform `a` shift cancels (only relative ramp matters to the order statistic).
        """
        n = len(thetas)
        mu = model._drift_times(thetas, conditions)
        lam = model._diffusion_shape(conditions, n)
        g, gd, gr = self.gains.a_start_gain, self.gains.a_d_gain, self.gains.rho_gain
        if (g != 0.0 or gd != 0.0) and n > 1:
            a = np.array([t.a for t in thetas])
            a_c = a - np.nanmean(a)                      # field-relative ramp deficit
            start_frac = min(1.0, _START_LREF / max(float(conditions.distance), 1.0))
            if g != 0.0:
                # a<cruise (a_c<0) ⇒ longer effective trip ⇒ larger μ; gain folds onto the START share.
                mu = mu * (1.0 - g * start_frac * a_c)
            if gd != 0.0:
                # a<field (a_c<0) ⇒ troubled break ⇒ MORE start-phase variance ⇒ larger D ⇒ smaller λ.
                # λ = L²/(2D), so D *= f  ⇔  λ /= f, with f = clip(1 − gd·start_frac·a_c, 0.05, ∞).
                f = np.clip(1.0 - gd * start_frac * a_c, 0.05, None)
                lam = lam / f
        if gr != 0.0 and n > 1:
            # ρ → LATE-RESERVE deployment. A high-ρ closer (ρ_c>0) banks reserve off the pace and
            # deploys it late ⇒ MORE effective usable reserve ⇒ a faster effective v_target ⇒ smaller μ.
            # μ = l_eff/(v_c·(1+W′/L)); scaling the reserve boost by (1+gr·ρ_c) is μ /= (1+gr·ρ_c).
            # Within-race-centred (a runner is a *relative* closer in its field) so a uniform ρ shift
            # cancels at the order statistic. clip keeps the factor positive for large |gr·ρ_c|.
            rho = np.array([t.rho for t in thetas], dtype=float)
            rho_c = rho - np.nanmean(rho)
            mu = mu / np.clip(1.0 + gr * rho_c, 0.05, None)
        return mu, lam
