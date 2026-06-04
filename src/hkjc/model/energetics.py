"""First-principles energetics: critical velocity + anaerobic reserve from sectionals.

Dynamics, not statistics. A horse is modelled as a critical-power system: a
sustainable **critical velocity** ``v_c`` it can hold aerobically, plus a finite
**anaerobic reserve** ``W'`` it spends to run *above* ``v_c`` — which depletes and
forces the late-race fade we measure in the sectional velocity profile.

This module recovers ``(v_c, W')`` from one run's velocity profile (the inverse
problem); a generative race simulator (next) uses per-horse ``(v_c, W')`` to
*simulate* finishing times under a predicted pace. Section distances follow HKJC's
convention (verified): the first section is the variable start segment, the rest
are 400m.

Units: distances in metres, times in seconds, velocities in m/s; ``W'`` is the
anaerobic work proxy ``∫ (v − v_c)^+ dt`` (m²/s), monotone in true reserve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import RaceModel

SECTION_LEN = 400.0   # metres; HKJC times all-but-the-first section at 400m


def section_distances(total_distance: float, n_sections: int) -> np.ndarray:
    """Per-section distances (metres). First = ``D − 400·(n−1)``; rest = 400m."""
    if n_sections < 1:
        return np.array([])
    first = total_distance - SECTION_LEN * (n_sections - 1)
    return np.array([first] + [SECTION_LEN] * (n_sections - 1), dtype=float)


@dataclass(frozen=True)
class RunEnergetics:
    v_c: float          # critical (sustainable) velocity, m/s
    w_prime: float      # anaerobic work spent above v_c (m²/s proxy)
    v_peak: float       # fastest section velocity (anaerobic ceiling)
    v_mean: float       # average velocity over the run
    fade: float         # v_peak / v_end - 1 (how far it dropped from its peak)
    n_sections: int


def estimate_run_energetics(section_times, section_dists) -> RunEnergetics:
    """Recover ``(v_c, W')`` from one run's section times + distances.

    ``v_c`` is estimated as the settled (late-race) velocity the horse fades *to*
    (the slower of the final two sections — robust to a late sprint); ``W'`` is the
    work run above that critical velocity. Per-run estimates are noisy (easy winners
    ease down → ``v_c`` under-read); aggregate across a horse's runs for a stable
    pace-neutral parameter.
    """
    t = np.asarray(section_times, dtype=float)
    d = np.asarray(section_dists, dtype=float)
    v = d / t
    v_c = float(min(v[-2:])) if len(v) >= 2 else float(v[-1])
    w_prime = float(np.sum(np.clip(v - v_c, 0, None) * t))
    v_peak = float(v.max())
    v_end = float(v[-1])
    return RunEnergetics(
        v_c=v_c, w_prime=w_prime, v_peak=v_peak,
        v_mean=float(d.sum() / t.sum()),
        fade=float(v_peak / v_end - 1.0) if v_end > 0 else float("nan"),
        n_sections=len(v),
    )


class EnergeticsRaceModel(RaceModel):
    """Generative race model: simulate finishing TIMES from per-horse (v_c, W').

    Each horse's predicted time follows Hill's critical-velocity law
    ``T = (D − W') / v_c`` (run the sustainable pace, with the anaerobic distance
    reserve ``W'`` shaving time off). We sample times with race-to-race noise and
    rank them — the fastest finisher is the winner — so the simulator turns
    *physiology + distance* into win/place/exotic probabilities. This is dynamics,
    not a feature regression: change the distance or a horse's ``(v_c, W')`` and the
    whole finishing-order distribution moves mechanistically.

    Args:
        v_c: per-horse critical velocity (m/s), shape (n,).
        w_prime: per-horse anaerobic distance reserve (m), shape (n,).
        distance: race distance (m).
        time_noise: std of per-run finishing-time noise (s) — race-day variability.
    """

    def __init__(self, v_c, w_prime, distance: float, time_noise: float = 0.6):
        v_c = np.asarray(v_c, dtype=float)
        w_prime = np.asarray(w_prime, dtype=float)
        if v_c.ndim != 1 or v_c.shape != w_prime.shape:
            raise ValueError("v_c and w_prime must be 1D arrays of equal length")
        self.n = v_c.shape[0]
        self.distance = float(distance)
        self.time_noise = float(time_noise)
        self.pred_time = (self.distance - w_prime) / np.where(v_c > 0, v_c, np.nan)

    def sample_orders(self, size: int, rng: np.random.Generator) -> np.ndarray:
        times = self.pred_time[None, :] + rng.normal(0.0, self.time_noise, size=(size, self.n))
        return np.argsort(times, axis=1).astype(np.int64)   # ascending time -> winner first
