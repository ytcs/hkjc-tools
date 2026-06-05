"""Power-balance equation of motion — the mechanistic core (replaces the kinematic v_target).

Each horse is a body of mass ``m`` (body + carried weight) driven by metabolic power
``P(t)`` against real resistances, integrated by Newton's law each step:

    m·dv/dt = F_prop − F_resist
    F_prop  = P(t) / v
    P(t)    = P_c + p_anaer ,   p_anaer drawn from a finite reserve W′ (joules)
    dW_bal  = −(P − P_c)·dt    (deplete above the aerobic ceiling; when empty, P ≤ P_c)
    F_resist = K_LOCO·m·g            (locomotion: cost of carrying/moving the mass)
             + ½·ρ·cda·v²·(1−draft)  (aerodynamic drag; drafting behind a leader cuts it)
             + bend ground-loss      (geometric, via the track — longer path off the rail)

Why this and not the kinematic model: here **mass, drag and drafting are terms in the
equation**, not multipliers tacked on the output. Carrying more weight raises both the
inertia ``m·dv/dt`` and the locomotion cost ``K_LOCO·m·g``, so a horse's sustainable
velocity ``v_c = P_c/(K_LOCO·m·g)`` falls *mechanistically* with mass; sitting behind a
leader cuts drag; emptying ``W′`` forces the late-race fade. The control **policy** (the
even-pace target, where jockey/tactics will live) is the only other input.

Calibration mapping (``from_kinematic``): the existing per-horse estimates are a
critical-velocity ``v_c`` (m/s) and an anaerobic *distance* reserve ``w_prime`` (m) at a
known race-day mass ``m``. These map to the physical parameters by
``P_c = K_LOCO·m·g·v_c`` and ``W′(J) = K_LOCO·m·g·w_prime`` (the force ``K_LOCO·m·g``
times the reserve distance = energy). With drag off and an even-pace policy the model
then reproduces the kinematic finish time ``T = (D − w_prime)/v_c`` — the reduces-to test.
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import RaceModel
from ..core.track import HAPPY_VALLEY, TrackGeometry
from .eom import G, K_LOCO, RHO, integrate_jax  # noqa: F401  (G/K_LOCO/RHO re-exported for callers)


class DynamicsRaceModel(RaceModel):
    """Multi-agent power-balance race model (mechanistic successor to TrafficRaceModel).

    Parameters are *physical*: per-horse aerobic power ``p_c`` (W), anaerobic reserve
    ``w_reserve`` (J), mass ``mass`` (kg). Build from the existing kinematic estimates
    with :meth:`from_kinematic`. ``cda`` is the drag area (set 0 to recover the pure
    energetics model); ``draft`` is the fractional drag relief sitting behind a leader.
    """

    def __init__(
        self, p_c, w_reserve, mass, draw, distance: float,
        track: TrackGeometry = HAPPY_VALLEY, course: str = "A",
        dt: float = 0.5, cda: float = 0.5, draft: float = 0.4,
        lane_sep: float = 1.2, follow_gap: float = 6.0, lane_band: float = 1.5,
        lane_change_prob: float = 0.3, lane_step: float = 1.2,
        bend_curvature: float = 1.0 / 120.0, speed_noise: float = 0.02,
        pace_bias=None, pace_gain: float = 0.0,
        kick_frac=None, kick_overspeed: float = 0.35,
        tau: float = 1.5, max_steps: int = 600,
    ):
        self.p_c = np.asarray(p_c, float)
        self.w_reserve = np.asarray(w_reserve, float)
        self.mass = np.asarray(mass, float)
        draw = np.asarray(draw, float)
        self.n = self.p_c.shape[0]
        self.distance = float(distance)
        self.track, self.course = track, course
        self.dt, self.cda, self.draft, self.tau = dt, cda, draft, tau
        self.follow_gap, self.lane_band = follow_gap, lane_band
        self.lane_change_prob, self.lane_step = lane_change_prob, lane_step
        self.bend_curvature, self.speed_noise, self.max_steps = bend_curvature, speed_noise, max_steps
        # tactical pace policy on the target velocity: >0 spends speed (and thus W′) EARLY,
        # <0 conserves for a late run. Energy-neutral in shape — but here W′ is really
        # depleted, so over-fast early pace empties the reserve and forces a genuine fade.
        self.pace_gain = float(pace_gain)
        self.pace_bias = np.zeros(self.n) if pace_bias is None else np.asarray(pace_bias, float)
        # state-FEEDBACK "kick" policy (rung 2): hold at sustainable v_c (banking W′, staying
        # in touch) until within kick_frac·D of home, then deploy the reserve at v_c·(1+overspeed)
        # — capped by what W′ allows, so the kick naturally fades when the tank empties. A
        # front-runner ≈ kick_frac→1 (spend from the gun); a closer ≈ small kick_frac (pounce late).
        self.kick_frac = None if kick_frac is None else np.asarray(kick_frac, float)
        self.kick_overspeed = float(kick_overspeed)
        self._floco = K_LOCO * self.mass * G                       # locomotion force, N
        self.v_c = self.p_c / self._floco                          # sustainable velocity (mass-aware)
        wdist = np.clip(self.w_reserve / self._floco, 0.0, 0.6 * self.distance)  # reserve as distance
        self.v_target = self.v_c / (1.0 - wdist / self.distance)   # even-pace optimum
        self.off0 = np.clip((draw - 1.0) * lane_sep, 0.0, max(track.width(course) - 1.0, lane_sep))
        bends = track.bend_segments or ()
        self._bend_lo = tuple(float(a) for a, _ in bends)
        self._bend_hi = tuple(float(b) for _, b in bends)
        self._circ = float(track.circumference)

    @classmethod
    def from_kinematic(cls, v_c, w_prime, mass, draw, distance, **kw):
        """Build from critical velocity (m/s), anaerobic distance reserve (m), mass (kg)."""
        v_c = np.asarray(v_c, float)
        mass = np.asarray(mass, float)
        floco = K_LOCO * mass * G
        return cls(p_c=floco * v_c, w_reserve=floco * np.asarray(w_prime, float),
                   mass=mass, draw=draw, distance=distance, **kw)

    def sample_orders(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.argsort(self.finish_times(size, rng), axis=1).astype(np.int64)

    def finish_times(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Integrate the canonical EOM for ``size`` samples; return finishing times ``(size, n)`` (s).

        Delegates to :func:`hkjc.model.eom.integrate_jax` — the single source for the
        mass-aware power-balance dynamics. ``rng`` only supplies the JAX PRNG seed (the
        sampling itself runs under ``jax.random``), so results are reproducible per seed.
        """
        seed = int(rng.integers(0, 2**31 - 1))
        return integrate_jax(
            size, seed,
            v_c=self.v_c, v_target=self.v_target, w_reserve=self.w_reserve,
            p_c=self.p_c, floco=self._floco, mass=self.mass, off0=self.off0,
            kick=self.kick_frac is not None, kick_frac=self.kick_frac,
            kick_overspeed=self.kick_overspeed, pace_bias=self.pace_bias,
            pace_gain=self.pace_gain, speed_noise=self.speed_noise,
            cda=self.cda, draft=self.draft, tau=self.tau, dist=self.distance,
            follow_gap=self.follow_gap, lane_band=self.lane_band,
            lane_change_prob=self.lane_change_prob, lane_step=self.lane_step,
            bend_curvature=self.bend_curvature, bend_lo=self._bend_lo,
            bend_hi=self._bend_hi, circ=self._circ, dt=self.dt, max_steps=self.max_steps)


def win_probs(v_c, w_prime, mass, draw, distance, *, allometric_b: float = 1.0,
              track: TrackGeometry = HAPPY_VALLEY, course: str = "A", size: int = 4000,
              speed_noise: float = 0.05, seed: int = 0, dt: float = 1.0, max_steps: int = 260,
              cda: float = 0.5, draft: float = 0.4, **kw) -> np.ndarray:
    """P(each horse wins) from the power-balance EOM — mass enters every term via Newton's law.

    First-principles mass coupling (no tacked-on v_c gain): declared body mass sets the
    locomotion cost ``K_LOCO·m·g``, the inertia ``m·dv/dt`` (so heavier horses accelerate the
    late kick more slowly), the anaerobic reserve energy ``W'(J)=K_LOCO·m·g·w'``, and — through
    the one physiological constant, the allometric exponent ``b`` in ``p_c=κ·m^b`` — the aerobic
    power. Power is built as ``p_c = K_LOCO·m·g·v_c·(m/m̄_field)^(b−1)``: at ``b=1`` mass cancels in
    the steady state (pure dynamics), ``b>1`` gives heavier horses a power surplus (super-linear
    scaling, the market's "heavier = faster"). ``v_c`` is the ability estimate (m/s), ``w_prime``
    the anaerobic distance reserve (m); both at the field's masses. Laplace-smoothed, sums to 1.
    """
    v_c = np.asarray(v_c, float)
    mass = np.asarray(mass, float)
    w_prime = np.asarray(w_prime, float)
    ref = float(np.exp(np.nanmean(np.log(mass))))            # field geometric-mean mass
    floco = K_LOCO * mass * G
    p_c = floco * v_c * (mass / ref) ** (allometric_b - 1.0)
    model = DynamicsRaceModel(
        p_c=p_c, w_reserve=floco * w_prime, mass=mass, draw=draw, distance=distance,
        track=track, course=course, speed_noise=speed_noise, dt=dt, max_steps=max_steps,
        cda=cda, draft=draft, **kw)
    ft = model.finish_times(size, np.random.default_rng(seed))
    n = ft.shape[1]
    counts = np.bincount(np.argmin(ft, axis=1), minlength=n).astype(float)
    return (counts + 1.0) / (ft.shape[0] + n)
