"""Stage 6/7 — PROJECT + TRAJECTORY: pixels -> (x,y) -> (d, off), then smooth.

Project the FOOT point (ground contact), apply H to world (x,y), then map onto the
venue/course centreline spline from ``track.py`` to get (d, off). Resample to a
uniform grid and RTS-smooth to clean speed/accel without killing the accel signal
(where W' lives). numpy/scipy imported lazily.
"""
from __future__ import annotations

from typing import Optional

from .schema import (
    Detection, Homography, RaceContext, RunnerTrajectory, TrajPoint, TrafficEvent,
)


class Centreline:
    """Venue/course centreline + rail spline built from ``TrackGeometry`` (read-only).

    Provides ``to_d_off(x, y) -> (d, off)``: nearest-point projection onto the
    centreline (arc-length => d = metres-from-finish; signed normal => off, outward
    from the current inner rail). Built once per (venue, course).
    """

    def __init__(self, ctx: RaceContext) -> None:
        self.ctx = ctx
        self._spline = None

    def _ensure(self):
        if self._spline is None:
            from hkjc.track import track_for  # lazy, read-only
            geo = track_for(self.ctx.venue, self.ctx.distance, self.ctx.surface)
            # STUB: build a centreline polyline/spline from geo.circumference,
            # geo.bend_segments, geo.home_straight(course), geo.width(course),
            # offsetting the inner rail by the movable-rail amount for ctx.course.
            self._spline = geo
        return self._spline

    def to_d_off(self, x: float, y: float) -> tuple[float, float]:
        """Map world (x, y) -> (d metres-from-finish, off metres outward from rail). STUB."""
        self._ensure()
        raise NotImplementedError(
            "Stub — nearest-point projection of (x,y) onto the course centreline spline."
        )


def project_point(h: Homography, foot_u: float, foot_v: float) -> tuple[float, float]:
    """Apply homography to a pixel foot point -> world (x, y). STUB."""
    raise NotImplementedError(
        "Stub. reshape h.matrix to 3x3; p = H @ [u,v,1]; return (p0/p2, p1/p2)."
    )


def build_trajectory(
    ctx: RaceContext,
    runner: int,
    detections: list[Detection],
    homographies: dict[int, Homography],
    centreline: Centreline,
    out_hz: float = 10.0,
) -> RunnerTrajectory:
    """Project + resample + smooth one runner's detections into a RunnerTrajectory.

    Steps (STUB): project each foot point through its frame H -> (x,y) -> (d,off);
    resample to ``out_hz``; Kalman+RTS smooth (d, off); derive speed/accel; mark gap
    points; compute total_distance / ground_lost / coverage.
    """
    raise NotImplementedError("Stub — see CV_PIPELINE Stage 6/7.")


def detect_traffic_events(traj: RunnerTrajectory, field: list[RunnerTrajectory]) -> list[TrafficEvent]:
    """Derive blocked/steadied/checked/switched/wide events from kinematics + neighbours. STUB."""
    raise NotImplementedError(
        "Stub — sharp negative accel with a runner just ahead in d (small off-band) => "
        "blocked/steadied; sustained high off on a bend => wide (ground lost)."
    )
