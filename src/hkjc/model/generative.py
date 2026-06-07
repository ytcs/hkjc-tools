"""GenerativeRaceModel — θ + conditions → first-passage law → JOINT finishing-order distribution.

Layer 2 of the canonical framework, the GENERATIVE core. Lifts the production first-passage law
(``hkjc.model.fit``: the validated Wald/Inverse-Gaussian drift-diffusion, ``T = first-passage of
dx = v·dt + √(2D)·dW``, Var(T)=2D·L/v³, λ=L²/2D — VALIDATED #33/#36/#37, CV∝1/√L, distance-reserve)
and re-expresses it on the (θ_i, conditions) state-space signature so the SAME Monte-Carlo sampler
that powers production now consumes a :class:`~hkjc.model.latent.LatentState` per runner.

The flow:

    θ_i, conditions  ──drift──▶  μ_i = L_eff / (v_c·(1 + W′/L) · ramp(a))   [per-runner mean time]
                     ──diff───▶  λ_i = L² / (2·D_i)                          [Wald shape]
    sample T_i ~ Wald(μ_i, λ_i)  (n_samples × n)
    argsort along runners        ──▶  sampled finishing ORDERS (the joint object)
    ──▶  :class:`OrderDistribution`: win = P(1st), place = P(top-k), and every higher-order
         marginal a pool needs (quinella/forecast/trifecta) as Monte-Carlo functionals.

STAGE 1 acceleration→start-phase: the `a` latent enters the drift here via a START-PHASE effective-L
correction (a slow ramp lengthens the effective trip; THEORY §3e Phase 1) — the gate-CALIBRATED
coupling strength is injected by :class:`~hkjc.model.coupling.CouplingCalibrator`, NOT read raw
from the honestly-identified `a` (#40 lesson). The skeleton wires a trivial-but-correct coupling so
the end-to-end path produces real win probabilities; the BUILD-WAVE replaces the coupling magnitude
with the calibrated one.

LEAK-FREE: the sampler is a pure function of (θ, conditions) + a race-id-seeded RNG. θ is strictly-
prior; conditions are pre-race. No outcome, no odds enter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conditions import RaceConditions
from .latent import LatentState

# Production wald-law parameters, sourced from hkjc.model.wald_constants so the bare law (a=OFF)
# is BYTE-FOR-BYTE the production first-passage drift/diffusion (Acceptance Gate A). The defaults
# below match the production `fit_predict_traffic` signature defaults (diff_coef, awt_diff_mult);
# the per-venue geometric gains are imported lazily inside the assembly to avoid a heavy import at
# module load and to track production exactly.
_DIFF_COEF = 170.0
_AWT_DIFF_MULT = 2.0

# FIELD/DEPLOYMENT cause (journal 0065): when a horse's effective reserve is reduced by the early
# pace-war burn it must keep a physical floor of reserve (a front-runner does not finish with NEGATIVE
# W′; the burn cannot push the reserve below this fraction of W′). Physical clamp, not a fit.
_WPRIME_FLOOR_FRAC = 0.0   # W′_eff ≥ _WPRIME_FLOOR_FRAC · W′ (floor at 0 ⇒ a fully-burnt front-runner
                           # carries no reserve to the finish, never a negative reserve).


@dataclass(frozen=True)
class OrderDistribution:
    """The JOINT finishing-order distribution as a Monte-Carlo sample of orders.

    ``orders`` has shape ``(n_samples, n)``: each row is a finishing order (column 0 = winner
    index, column 1 = runner-up, …), exactly the :class:`hkjc.core.contracts.RaceModel` output
    convention so the production simulator/pricer ecosystem is reusable. Every pool price is a
    functional of THIS object (see :mod:`hkjc.model.pricer`).
    """

    orders: np.ndarray  # (n_samples, n) int finishing orders
    n: int

    def win_probs(self) -> np.ndarray:
        """P(runner i finishes 1st) — the win pool. Laplace-smoothed like the production sampler."""
        winners = self.orders[:, 0]
        counts = np.bincount(winners, minlength=self.n).astype(float)
        return (counts + 1.0) / (len(self.orders) + self.n)

    def topk_probs(self, k: int) -> np.ndarray:
        """P(runner i finishes in the top k) — the place / show pool marginals."""
        topk = self.orders[:, :k]
        counts = np.bincount(topk.reshape(-1), minlength=self.n).astype(float)
        return counts / len(self.orders)


class GenerativeRaceModel:
    """θ + conditions → sampled paths → joint finishing-order distribution.

    Stage-1 surface: it predicts finishing TIMES via the Wald first-passage law and returns the
    sampled ORDERS. The full predicted PATH (x_long, y_lat, v, t) per runner is the Stage-2/3
    extension (the times are the along-track projection of that path); :meth:`sample_paths` is the
    stub seam for it. ``n_samples`` and ``diff_coef`` mirror the production sampler.
    """

    def __init__(self, *, n_samples: int = 20000, diff_coef: float = _DIFF_COEF,
                 awt_diff_mult: float = _AWT_DIFF_MULT,
                 field_pace_war: bool = False) -> None:
        self.n_samples = n_samples
        self.diff_coef = diff_coef
        self.awt_diff_mult = awt_diff_mult
        # GUARDED FIELD/DEPLOYMENT term (journal 0065). DEFAULT OFF ⇒ the drift is BYTE-IDENTICAL to
        # the physiology-only baseline (Acceptance Gate A). ON ⇒ each runner's effective reserve is
        # reduced by ``conditions.pace_war_burn`` (the front-group pace-war early W′-burn) before the
        # reserve boost enters the drift. The win delta is measured ON vs OFF.
        self.field_pace_war = field_pace_war

    # -------------------------------------------------------------- drift / diffusion
    def _drift_times(self, thetas: list[LatentState], conditions: RaceConditions) -> np.ndarray:
        """Per-runner mean first-passage time μ_i — the PRODUCTION wald drift, byte-for-byte.

        Reproduces ``hkjc.model.fit.fit_predict_traffic``'s μ assembly: the reserve-boosted target
        velocity v_c·(1+W′/L) over the bend-ground-loss + gate-break effective trip ``l_eff``.
        With the production-computed ``v_c_eb``/``w_prime_eb`` carried on θ and the geometry on
        ``conditions``, μ here equals production μ exactly — the a=OFF equivalence baseline. The
        Stage-1 `a`→start-phase overlay is applied ON TOP of this by :class:`CouplingCalibrator`.
        """
        L = float(conditions.distance)
        n = len(thetas)
        vc = np.array([t.v_c for t in thetas], dtype=float)
        wp = np.array([t.w_prime for t in thetas], dtype=float)
        wp_eff = self._effective_w_prime(wp, conditions)
        wp_boost = wp_eff / L
        l_eff = self._effective_length(conditions, n)
        v_target = vc * (1.0 + wp_boost)
        return l_eff / np.clip(v_target, 1e-6, None)

    def _effective_w_prime(self, wp: np.ndarray, conditions: RaceConditions) -> np.ndarray:
        """Effective anaerobic reserve after the GUARDED field pace-war early burn (journal 0065).

        When ``field_pace_war`` is OFF (the default) this returns ``wp`` UNCHANGED — so the drift, the
        sampled orders and every existing number are BYTE-IDENTICAL to the physiology-only baseline
        (Acceptance Gate A). When ON, each front-runner's reserve is reduced by
        ``conditions.pace_war_burn`` — the early reserve [m] the contested front pace depletes
        (``field.pace_war_burn_per_runner``, from the field's strictly-prior style/pace composition,
        KNOWN pre-race ⇒ leak-free) — clamped to a physical floor (no negative reserve). A horse that
        contests a hot, front-heavy field carries LESS reserve into the finish ⇒ its drift slows. The
        burn magnitude is the EOM physics; only the ON/OFF and the existing physical threshold are knobs.
        """
        if not self.field_pace_war:
            return wp
        burn = conditions.pace_war_burn
        if burn is None:
            return wp
        burn = np.asarray(burn, dtype=float)
        if burn.shape != wp.shape:
            return wp
        return np.maximum(wp - np.clip(burn, 0.0, None), _WPRIME_FLOOR_FRAC * wp)

    def _effective_length(self, conditions: RaceConditions, n: int) -> np.ndarray:
        """Production effective trip ``l_eff`` = L·(1 + bend-draw penalty)·(1 − gate-break shrink).

        Mirrors the ``wald`` branch of ``fit_predict_traffic`` exactly (per-venue DRAW_GAIN with the
        field-size crowding amplifier, zeroed on no-bend straights; the GATE_BREAK→effective-L
        start-phase term). Pure function of pre-race conditions ⇒ no leak.
        """
        from hkjc.core.track import track_for
        from hkjc.model.wald_constants import (DRAW_FS_GAIN, DRAW_FS_GAIN_BY_VENUE, DRAW_GAIN,
                                          DRAW_GAIN_AWT, DRAW_GAIN_BY_VENUE, GATE_BREAK_D_GAIN,
                                          GATE_BREAK_FS_GAIN, GATE_BREAK_GAIN,
                                          GATE_BREAK_GAIN_BY_VENUE, GATE_BREAK_LREF)
        L = float(conditions.distance)
        venue = conditions.venue
        surface = conditions.surface
        draw = np.asarray(conditions.draws, dtype=float)
        geom = track_for(venue, L, surface)
        if surface == "awt":
            draw_g = DRAW_GAIN_AWT
        else:
            draw_g = DRAW_GAIN_BY_VENUE.get(venue, DRAW_GAIN)
        if geom.bend_fraction(L) < 0.05:
            draw_g = 0.0
        dfs_g = DRAW_FS_GAIN_BY_VENUE.get(venue, DRAW_FS_GAIN)
        dg = draw_g * max(0.0, 1.0 + dfs_g * (n - 12.0))
        l_eff = L * (1.0 + dg * (draw - 1.0))
        gbg = GATE_BREAK_GAIN_BY_VENUE.get(venue, GATE_BREAK_GAIN)
        gbrk = conditions.gate_break_shrunk
        if gbg != 0.0 and gbrk is not None:
            start_frac = min(1.0, GATE_BREAK_LREF / max(L, 1.0))
            fs = max(0.0, 1.0 + GATE_BREAK_FS_GAIN * (n - 12.0))
            l_eff = l_eff * (1.0 - gbg * start_frac * fs * np.asarray(gbrk, dtype=float))
        # GATE_BREAK_D_GAIN is a D-channel knob (not l_eff); handled in _diffusion_shape.
        _ = GATE_BREAK_D_GAIN
        return l_eff

    def _diffusion_shape(self, conditions: RaceConditions, n: int) -> np.ndarray:
        """Per-runner Wald shape λ_i = L²/(2·D_i) — the PRODUCTION diffusion, byte-for-byte.

        Reproduces the production D_i assembly: the surface base D, the draw→D trip-variance knob
        (DRAW_VAR_GAIN, bend×lateral-extremity), the per-horse figure-spread redistribution
        (FIGVAR_GAIN), and the gate-break→D start-phase variance channel (GATE_BREAK_D_GAIN). All
        per-venue, pure functions of pre-race conditions ⇒ no leak.
        """
        from hkjc.core.track import track_for
        from hkjc.model.wald_constants import (DRAW_VAR_GAIN, DRAW_VAR_GAIN_BY_VENUE, FIGVAR_GAIN,
                                          FIGVAR_GAIN_BY_VENUE, GATE_BREAK_D_GAIN, GATE_BREAK_LREF)
        L = float(conditions.distance)
        venue = conditions.venue
        surface = conditions.surface
        d_eff = self.diff_coef * (self.awt_diff_mult if surface == "awt" else 1.0)
        geom = track_for(venue, L, surface)
        draw = np.asarray(conditions.draws, dtype=float)
        d_h = np.full(n, d_eff, dtype=float)
        dvg = DRAW_VAR_GAIN_BY_VENUE.get(venue, DRAW_VAR_GAIN)
        if dvg != 0.0 and n > 1:
            extremity = 2.0 * np.abs(draw - (n + 1.0) / 2.0) / (n - 1.0)
            d_h = d_h * (1.0 + dvg * geom.bend_fraction(L) * extremity)
        fvg = FIGVAR_GAIN_BY_VENUE.get(venue, FIGVAR_GAIN)
        fdisp = conditions.fig_disp
        if fvg != 0.0 and fdisp is not None and np.isfinite(fdisp).any():
            fdisp = np.asarray(fdisp, dtype=float)
            dbar = np.nanmean(fdisp)
            if dbar > 0:
                rel = np.where(np.isfinite(fdisp), fdisp / dbar - 1.0, 0.0)
                d_h = d_h * np.clip(1.0 + fvg * rel, 0.05, None)
        gbrk = conditions.gate_break_shrunk
        if GATE_BREAK_D_GAIN != 0.0 and gbrk is not None:
            start_frac = min(1.0, GATE_BREAK_LREF / max(L, 1.0))
            d_h = d_h * np.clip(1.0 - GATE_BREAK_D_GAIN * start_frac * np.asarray(gbrk, dtype=float),
                                0.05, None)
        return L * L / (2.0 * np.asarray(d_h))

    # ------------------------------------------------------------------- the law
    def order_distribution(
        self, thetas: list[LatentState], conditions: RaceConditions,
        *, mu: np.ndarray | None = None, lam: np.ndarray | None = None,
    ) -> OrderDistribution:
        """Sample the joint finishing-order distribution from the Wald first-passage law.

        ``mu`` / ``lam`` may be supplied PRE-COUPLED by :class:`CouplingCalibrator` (the calibrated
        `a`→start and other couplings already folded into μ/λ); otherwise the bare (θ,conditions)
        drift/diffusion is used. Deterministic given the race_id seed — the production discipline.
        """
        n = len(thetas)
        if mu is None:
            mu = self._drift_times(thetas, conditions)
        if lam is None:
            lam = self._diffusion_shape(conditions, n)
        seed = abs(hash(conditions.race_id)) % (2**31)
        rng = np.random.default_rng(seed)
        t = rng.wald(np.broadcast_to(mu, (self.n_samples, n)),
                     np.broadcast_to(lam, (self.n_samples, n)))
        orders = np.argsort(t, axis=1)
        return OrderDistribution(orders=orders, n=n)

    def sample_paths(self, thetas: list[LatentState], conditions: RaceConditions):
        """Full (x_long, y_lat, v, t) per-runner paths (Stage-2/3 STUB).

        Stage 1 only needs finishing times (the along-track first-passage). The full path — the
        across-track trajectory and the velocity profile through time — is the surface the lateral
        and margin likelihoods score and the geometry game lives on. BUILD-WAVE: emit per-section
        velocity/position paths so the margin term (Stage 2) and lateral term (Stage 3) can score them.
        """
        raise NotImplementedError("Full-path emission is Stage-2/3 — see BUILD-WAVE in MODEL_DESIGN.md")
