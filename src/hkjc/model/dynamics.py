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

from ..track import HAPPY_VALLEY, TrackGeometry
from .base import RaceModel

G = 9.81            # m/s²
RHO = 1.2           # air density, kg/m³
K_LOCO = 0.10       # cost-of-transport coefficient (net propulsive force per unit weight)


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
        self._bends = np.array(track.bend_segments, float)
        self._circ = track.circumference

    @classmethod
    def from_kinematic(cls, v_c, w_prime, mass, draw, distance, **kw):
        """Build from critical velocity (m/s), anaerobic distance reserve (m), mass (kg)."""
        v_c = np.asarray(v_c, float)
        mass = np.asarray(mass, float)
        floco = K_LOCO * mass * G
        return cls(p_c=floco * v_c, w_reserve=floco * np.asarray(w_prime, float),
                   mass=mass, draw=draw, distance=distance, **kw)

    def _is_bend(self, d_from_finish):
        dm = np.mod(d_from_finish, self._circ)
        out = np.zeros(dm.shape, bool)
        for a, b in self._bends:
            out |= (dm >= a) & (dm < b)
        return out

    def sample_orders(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.argsort(self.finish_times(size, rng), axis=1).astype(np.int64)

    def finish_times(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Integrate the EOM for ``size`` samples; return finishing times ``(size, n)`` (s)."""
        n, dist, dt = self.n, self.distance, self.dt
        m = self.mass[None, :]
        floco = self._floco[None, :]
        p_c = self.p_c[None, :]
        p_an_max = p_c                                              # anaerobic ≤ aerobic ceiling
        v_tgt0 = np.tile(self.v_target, (size, 1)) * (1.0 + self.speed_noise * rng.standard_normal((size, n)))
        pb = (self.pace_gain * self.pace_bias)[None, :]            # per-horse early/late tilt
        if self.kick_frac is not None:                            # kick-policy hold/kick targets
            _kn = 1.0 + self.speed_noise * rng.standard_normal((size, n))
            v_hold = np.tile(self.v_c, (size, 1)) * _kn
            v_kick = v_hold * (1.0 + self.kick_overspeed)
            kf = self.kick_frac[None, :]
        v = np.tile(self.v_c, (size, 1)).astype(float)             # break at sustainable speed
        s = np.zeros((size, n))
        off = np.tile(self.off0, (size, 1)).astype(float)
        w_bal = np.tile(self.w_reserve, (size, 1)).astype(float)
        finish_t = np.full((size, n), np.inf)
        t = 0.0
        for _ in range(self.max_steps):
            t += dt
            live = finish_t == np.inf
            if not live.any():
                break
            on_bend = self._is_bend(dist - s)
            if self.kick_frac is None:
                # tactical tilt: front-load (>0) or conserve (<0) speed by race progress
                v_tgt = v_tgt0 * (1.0 + pb * (1.0 - 2.0 * np.clip(s / dist, 0.0, 1.0)))
            else:
                # kick feedback: hold at v_c, then dump W′ inside kick_frac·D of the finish
                v_tgt = np.where((dist - s) <= kf * dist, v_kick, v_hold)
            # --- traffic: car-following velocity cap + drafting (drag relief) ---
            si, sj = s[:, :, None], s[:, None, :]
            oi, oj = off[:, :, None], off[:, None, :]
            gap = sj - si
            same_lane = np.abs(oj - oi) < self.lane_band
            blocking = (gap > 0) & (gap < self.follow_gap) & same_lane
            vj = np.broadcast_to(v[:, None, :], blocking.shape)
            v_cap = np.where(blocking, vj, np.inf).min(axis=2)
            drafting = blocking.any(axis=2)                        # someone just ahead in lane
            v_goal = np.minimum(v_tgt, v_cap)
            # --- power balance ---
            draft_relief = np.where(drafting, self.draft, 0.0)
            f_drag = 0.5 * RHO * self.cda * v * v * (1.0 - draft_relief)
            a_des = (v_goal - v) / self.tau
            p_need = (floco + f_drag + m * a_des) * np.maximum(v, 1.0)
            p_avail = p_c + np.where(w_bal > 0, p_an_max, 0.0)
            p = np.clip(p_need, 0.0, p_avail)
            f_prop = p / np.maximum(v, 1.0)
            a = (f_prop - floco - f_drag) / m
            v = np.maximum(0.5, v + a * dt)
            w_bal = np.maximum(0.0, w_bal - np.maximum(p - p_c, 0.0) * dt)
            # --- lateral: blocked move out, else drift to rail ---
            blocked = v_cap < v_tgt - 1e-9
            move_out = blocked & (rng.random((size, n)) < self.lane_change_prob)
            off = off + move_out * self.lane_step
            drift_in = (~blocked) & (off > 0)
            off = np.maximum(0.0, off - drift_in * 0.3 * self.lane_step)
            # --- advance with bend ground-loss; interpolate finish ---
            ground = 1.0 + off * self.bend_curvature * on_bend
            ds = np.where(live, v * dt / ground, 0.0)
            s_new = s + ds
            crossed = live & (s_new >= dist)
            frac = np.where(crossed, (dist - s) / np.maximum(ds, 1e-9), 1.0)
            finish_t = np.where(crossed, t - dt + frac * dt, finish_t)
            s = np.minimum(s_new, dist)
        return np.where(finish_t == np.inf, t + (dist - s), finish_t)
