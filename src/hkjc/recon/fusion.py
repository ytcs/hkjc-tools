"""Path containers shared by the core (``core``) and the products.

``Obs``: observations of one horse from one feed (draw, official times, finish camera) on the race clock (s = metres
run along the racing rail, lat outward of it). ``Path``: a horse's posterior on the common grid (``DT`` = 0.1 s).
``Prior``: the physics prior's constants (bounded acceleration, lateral box). ``with_tail``: the prior-only tail on
written video-only paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

DT = 0.1


@dataclass
class Obs:
    """Observations of one horse from one feed."""

    t: np.ndarray  # race-clock seconds
    s: np.ndarray | None  # metres run (NaN = unobserved)
    lat: np.ndarray | None  # lateral metres (NaN = unobserved)
    sd_s: float
    sd_l: float
    feed: str
    horse: int = 0
    tid: int = -1


@dataclass
class Prior:
    a_sd_s: float = 1.2  # m/s^2 longitudinal accel sd once galloping (|a| < ~2 m/s^2)
    a_sd_start: float = 6.0  # m/s^2 during the standing start (0 -> ~16 m/s in ~3 s)
    t_start: float = 6.0  # s: duration of the start phase (race clock)
    a_sd_l: float = 0.25  # m/s^2 lateral accel sd
    l_min: float = 0.3  # m: a horse centre is at least this far outside the racing rail
    width: float = 30.0  # m: racing rail -> outer rail
    vl_sd: float = 2.0  # lateral-velocity prior (m/s): sideways drift is slow
    # no along-track velocity prior: the acceleration prior alone extrapolates at constant speed


@dataclass
class Path:
    horse: int
    t: np.ndarray
    s: np.ndarray
    lat: np.ndarray
    sd_s: np.ndarray
    sd_l: np.ndarray
    n_obs: np.ndarray  # observations within +-DT of each node (coverage)
    feeds: dict = field(default_factory=dict)


TAIL = 0.5  # s: prior-only tail on WRITTEN video-only paths (never fed back into the core)


def with_tail(p: Path, prior: Prior, tail: float = TAIL) -> Path:
    """``p`` plus nodes every DT for ``tail`` s after its end: the posterior mean of a prior-only tail
    under the acceleration prior (s at the end-node velocity, lat held), sd_s grown by the prior."""
    if len(p.t) < 2:
        return p
    tau = DT * np.arange(1, int(round(tail / DT)) + 1)
    v = (p.s[-1] - p.s[-2]) / (p.t[-1] - p.t[-2])
    k = len(tau)
    return replace(
        p,
        t=np.r_[p.t, p.t[-1] + tau],
        s=np.r_[p.s, p.s[-1] + v * tau],
        lat=np.r_[p.lat, np.full(k, p.lat[-1])],
        sd_s=np.r_[p.sd_s, np.hypot(p.sd_s[-1], 0.5 * prior.a_sd_s * tau**2)],
        sd_l=np.r_[p.sd_l, np.full(k, p.sd_l[-1])],
        n_obs=np.r_[p.n_obs, np.zeros(k, p.n_obs.dtype)],
    )
