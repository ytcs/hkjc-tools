"""The shared ``race_t`` clock — one race-time axis tying all four feeds together.

Every feed measures the same latent race on a different video timeline; the fusion needs them
on ONE axis. ``race_t`` is anchored on the official sections (cumulative section times sum to
the winner's time), and ``arclen`` (metres run from the gate) maps to ``race_t`` through the
official ``(cumulative time, cumulative arclen)`` curve. The per-feed video-time -> ``race_t``
maps (scattered across the studies) are unified here:

- **aerial** carries an on-render ODOMETER arclen per frame; ``arclen -> race_t`` via the
  section curve (the aerial has no independent clock offset, it reads distance directly).
- **patrol** ``race_t = patrol_vid - offset`` (kinematic finish-cross anchor).
- **RF** ``race_t = RF_vid - offset`` (the on-screen race clock).
- **drone** is an AFFINE map ``drone_vid -> race_t`` from two real-time anchors (the gate-jump
  and the photo-finish cut), since the drone has neither an odometer nor a clean on-screen clock.

All per-race scalars (the section table, the per-feed offsets, the drone affine anchors) are
INPUTS — gathered into :class:`ClockAnchors`, never hard-coded. The formulas are faithful to
``studies/cv_fusion_4feed.py`` (``arclen<->race_t`` interpolation, drone affine) and the per-feed
``vid_to_race_t`` helpers in ``hkjc.vision.feeds.{patrol,rf,drone}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def section_curve(sections: tuple | list, section_bounds: tuple | list) -> tuple[np.ndarray, np.ndarray]:
    """Build the official ``(cumulative time, cumulative arclen)`` lookup arrays.

    ``sections`` are the per-section times (s) in race order (length adapts 3/4/5);
    ``section_bounds`` are the cumulative arclen (m run) at each section boundary, with
    ``len(section_bounds) == len(sections) + 1`` (it starts at 0 = the gate). Returns
    ``(t_cum, arc_bounds)`` — both monotone-increasing, suitable for ``np.interp``.
    """
    sec = np.asarray(sections, dtype=float)
    bounds = np.asarray(section_bounds, dtype=float)
    if bounds.size != sec.size + 1:
        raise ValueError(
            f"section_bounds must have len(sections)+1 entries "
            f"(got {bounds.size} bounds for {sec.size} sections)"
        )
    t_cum = np.concatenate([[0.0], np.cumsum(sec)])
    return t_cum, bounds


@dataclass(frozen=True)
class ClockAnchors:
    """The per-race sync/calibration scalars that put every feed on the shared ``race_t`` axis.

    These are VALIDATION-class empirical inputs (frame-verified offsets / anchors), NOT model
    inputs and NOT in ``core.Race``. The section table comes from the official sectionals.
    """

    sections: tuple = ()              # per-section times (s), race order; sum == winner time
    section_bounds: tuple = ()        # cumulative arclen at section boundaries (m); len = len(sections)+1
    patrol_offset: float = 0.0        # race_t = patrol_vid - patrol_offset
    rf_offset: float = 0.0            # race_t = rf_vid - rf_offset
    drone_gate_vid_t: float = 0.0     # drone video time of the gate-jump (-> race_t = drone_gate_race_t)
    drone_gate_race_t: float = 0.0    # usually 0.0 (the gun)
    drone_finishcut_vid_t: float = 0.0  # drone video time of the photo-finish cut
    drone_finishcut_race_t: float = 0.0  # race_t of the winner's line-cross (== sum(sections))
    extra: dict = field(default_factory=dict)

    # ---- the official section curve ----
    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        """``(t_cum, arc_bounds)`` for ``arclen<->race_t`` interpolation."""
        return section_curve(self.sections, self.section_bounds)

    @property
    def winner_finish_race_t(self) -> float:
        """The winner's line-cross ``race_t`` (== the sum of the official section times)."""
        return float(np.sum(np.asarray(self.sections, dtype=float)))

    # ---- arclen <-> race_t (shared, via the official section curve) ----
    def arclen_to_race_t(self, arclen):
        """Arclen (m run from the gate) -> shared ``race_t`` (s)."""
        t_cum, bounds = self.curve()
        return np.interp(np.asarray(arclen, dtype=float), bounds, t_cum)

    def race_t_to_arclen(self, race_t):
        """Shared ``race_t`` (s) -> arclen (m run from the gate)."""
        t_cum, bounds = self.curve()
        return np.interp(np.asarray(race_t, dtype=float), t_cum, bounds)

    # ---- per-feed video time -> race_t ----
    def patrol_to_race_t(self, vid_t):
        """Patrol video time -> ``race_t`` (``race_t = vid_t - patrol_offset``)."""
        return np.asarray(vid_t, dtype=float) - self.patrol_offset

    def rf_to_race_t(self, vid_t):
        """RF video time -> ``race_t`` (``race_t = vid_t - rf_offset``, the on-screen clock)."""
        return np.asarray(vid_t, dtype=float) - self.rf_offset

    def drone_to_race_t(self, vid_t):
        """Drone video time -> ``race_t`` via the gate-jump + finish-cut affine map."""
        v0, v1 = self.drone_gate_vid_t, self.drone_finishcut_vid_t
        r0, r1 = self.drone_gate_race_t, self.drone_finishcut_race_t
        if v1 == v0:
            raise ValueError("drone gate and finish-cut video times coincide (no affine slope)")
        return r0 + (np.asarray(vid_t, dtype=float) - v0) * (r1 - r0) / (v1 - v0)

    @property
    def drone_rate(self) -> float:
        """The drone affine slope ``d(race_t)/d(vid_t)`` (for diagnostics)."""
        v0, v1 = self.drone_gate_vid_t, self.drone_finishcut_vid_t
        r0, r1 = self.drone_gate_race_t, self.drone_finishcut_race_t
        return (r1 - r0) / (v1 - v0) if v1 != v0 else float("nan")


def anchors_from_recipe(distance: float, sections, section_bounds=None, **kw) -> ClockAnchors:
    """Build :class:`ClockAnchors`, defaulting ``section_bounds`` to an EVEN arclen split.

    When the official per-section distance boundaries are not separately known, the cumulative
    arclen is spread evenly across the sections (``0, distance/n, ..., distance``) — a coarse but
    monotone curve. Prefer passing real ``section_bounds`` (the official call marks) when known.
    """
    sec = tuple(float(s) for s in sections)
    if section_bounds is None:
        n = len(sec)
        section_bounds = tuple(distance * i / n for i in range(n + 1))
    return ClockAnchors(sections=sec, section_bounds=tuple(float(b) for b in section_bounds), **kw)
