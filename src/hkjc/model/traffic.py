"""Tier-A interacting race model: a stochastic traffic-flow simulation (Happy Valley).

Horses are NOT independent here. Each runs its energetic even-pace target
(``v_target = v_c·(1 + W'/D)`` — so with empty track this reduces to the energetics
model), but on each step it is subject to traffic:

- **Car-following:** it cannot pass *through* a horse ahead in its lateral band —
  it is capped to that horse's speed (blocked).
- **Lane-change (gap-acceptance):** if blocked it may shift outward to find clear
  air (stochastic); otherwise it drifts back toward the rail (everyone wants the
  short path).
- **Bend ground-loss:** running ``off`` metres off the inner rail on a turn costs
  extra arc (``ground_factor = 1 + off·curvature``), so finish-progress per stride
  shrinks — this is the geometric draw bias. Draw → starting lateral offset
  (low draw = inner rail), and the usable width comes from the day's ``Course``.

Draw bias and trip-trouble therefore *emerge* from the dynamics + geometry rather
than being fitted. Vectorised over samples for speed. v1: even-pace targets, a
single ground-loss curvature for all bends; lateral lanes are modelled from the
draw (Tier-B will calibrate true lanes from aerial-video trajectories).
"""
from __future__ import annotations

import numpy as np

from .base import RaceModel
from .track import HAPPY_VALLEY, TrackGeometry


class TrafficRaceModel(RaceModel):
    def __init__(
        self,
        v_c,
        w_prime,
        draw,
        distance: float,
        track: TrackGeometry = HAPPY_VALLEY,
        course: str = "A",
        dt: float = 1.0,
        lane_sep: float = 1.2,        # m between adjacent starting draws
        follow_gap: float = 6.0,      # m: can't sit closer behind a same-lane horse
        lane_band: float = 1.5,       # m lateral half-width that counts as "same lane"
        lane_change_prob: float = 0.3,
        lane_step: float = 1.2,       # m shifted outward per lane-change
        bend_curvature: float = 1.0 / 120.0,  # extra ground per m off-rail on a bend
        max_steps: int = 220,
    ):
        v_c = np.asarray(v_c, float)
        w_prime = np.asarray(w_prime, float)
        draw = np.asarray(draw, float)
        self.n = v_c.shape[0]
        self.distance = float(distance)
        self.v_target = v_c * (1.0 + w_prime / self.distance)
        self.off0 = np.clip((draw - 1.0) * lane_sep, 0.0, max(track.width(course) - 1.0, lane_sep))
        self.track = track
        self.dt, self.follow_gap, self.lane_band = dt, follow_gap, lane_band
        self.lane_change_prob, self.lane_step = lane_change_prob, lane_step
        self.bend_curvature, self.max_steps = bend_curvature, max_steps
        self._bends = np.array(track.bend_segments, float)
        self._circ = track.circumference

    def _is_bend(self, d_from_finish: np.ndarray) -> np.ndarray:
        dm = np.mod(d_from_finish, self._circ)
        out = np.zeros(dm.shape, bool)
        for a, b in self._bends:
            out |= (dm >= a) & (dm < b)
        return out

    def sample_orders(self, size: int, rng: np.random.Generator) -> np.ndarray:
        n, dist, dt = self.n, self.distance, self.dt
        s = np.zeros((size, n))                              # finish-progress (m)
        off = np.tile(self.off0, (size, 1)).astype(float)    # lateral offset (m)
        vt = np.tile(self.v_target, (size, 1)) * (1.0 + 0.02 * rng.standard_normal((size, n)))
        finish_t = np.full((size, n), np.inf)
        t = 0.0
        for _ in range(self.max_steps):
            t += dt
            live = finish_t == np.inf
            if not live.any():
                break
            on_bend = self._is_bend(dist - s)
            # car-following: cap to the slowest horse ahead within the lane band
            si, sj = s[:, :, None], s[:, None, :]
            oi, oj = off[:, :, None], off[:, None, :]
            gap = sj - si
            blocking = (gap > 0) & (gap < self.follow_gap) & (np.abs(oj - oi) < self.lane_band)
            vj = np.broadcast_to(vt[:, None, :], blocking.shape)
            v_cap = np.where(blocking, vj, np.inf).min(axis=2)
            v = np.minimum(vt, v_cap)
            blocked = v_cap < vt - 1e-9
            # lane change: blocked horses may move out; clear horses drift toward rail
            move_out = blocked & (rng.random((size, n)) < self.lane_change_prob)
            off = off + move_out * self.lane_step
            drift_in = (~blocked) & (off > 0)
            off = np.maximum(0.0, off - drift_in * 0.3 * self.lane_step)
            # advance with bend ground-loss; deduct reached-finish, interpolate time
            ground_factor = 1.0 + off * self.bend_curvature * on_bend
            ds = np.where(live, v * dt / ground_factor, 0.0)
            s_new = s + ds
            crossed = live & (s_new >= dist)
            # interpolate crossing time within the step
            frac = np.where(crossed, (dist - s) / np.maximum(ds, 1e-9), 1.0)
            finish_t = np.where(crossed, t - dt + frac * dt, finish_t)
            s = np.minimum(s_new, dist)
        # any non-finishers: rank by progress (slower than finishers)
        finish_t = np.where(finish_t == np.inf, t + (dist - s), finish_t)
        return np.argsort(finish_t, axis=1).astype(np.int64)
