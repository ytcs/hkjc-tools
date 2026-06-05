"""Target output schema for the CV trajectory pipeline.

The deliverable of the whole pipeline is, per runner, a dense time series in the
coordinate system of ``src/hkjc/track.TrackGeometry``:

- ``d``   metres-from-finish (arc-length; horse runs from d=distance down to d=0)
- ``off`` lateral metres outward from the *current inner rail* (0 = on the rail)
- ``x, y`` optional world metric coords (intermediate; ``(d, off)`` is canonical)
- ``speed`` / ``accel`` along-track kinematics

These dataclasses are pure stdlib (no numpy/torch/cv2) so this module — and the
package — imports cleanly with nothing heavy installed. Heavy deps are lazily
imported inside the stage functions only.

Nothing here modifies ``src/hkjc``; the geometry there is consumed read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------- intermediates


@dataclass(frozen=True)
class RaceContext:
    """Identity needed to resolve geometry via ``hkjc.track.track_for(...)``."""
    venue: str                 # "HV" | "ST"
    date: str                  # ISO "YYYY-MM-DD" or HKJC "DD/MM/YYYY"
    race_no: int
    distance: float            # metres
    surface: str = "turf"      # "turf" | "awt"
    course: str = "A"          # movable-rail setting: A, A+2, B, ... (track.py)
    fps: Optional[float] = None
    mode: str = "B"            # "A" broadcast camera | "B" virtual render


@dataclass
class Detection:
    """One detected box in one frame (pixel space). Stage 2 output."""
    frame: int
    t: float                          # seconds from race start
    u: float                          # box centre x (px)
    v: float                          # box centre y (px)
    foot_u: float                     # ground-contact x (px) — project THIS
    foot_v: float                     # ground-contact y (px)
    w: float                          # box width (px)
    h: float                          # box height (px)
    score: float = 0.0
    cls: str = "horse"                # "horse" | "marker" | "rail" | "number"


@dataclass
class Tracklet:
    """A locally-consistent chain of detections. Stage 3 output."""
    track_id: int
    detections: list[Detection] = field(default_factory=list)


@dataclass
class IdentityVote:
    """A per-tracklet identity observation (OCR / silks). Stage 4 internal."""
    track_id: int
    saddlecloth: Optional[int] = None   # OCR'd number
    ocr_conf: float = 0.0
    silks_label: Optional[str] = None   # silks-classifier label
    silks_conf: float = 0.0


@dataclass(frozen=True)
class Homography:
    """Pixel->world planar map for a frame (or shot). Stage 5 output.

    ``matrix`` is a row-major 3x3 (list of 9 floats) so the schema stays
    numpy-free. ``valid`` flags frames where calibration failed / was extrapolated.
    """
    frame: int
    t: float
    matrix: tuple = ()           # 9 floats, row-major; empty => invalid
    valid: bool = False
    n_refs: int = 0              # number of world references used
    note: str = ""


# --------------------------------------------------------------------------- output


@dataclass
class TrajPoint:
    """One resampled sample on the output grid for one runner."""
    t: float                     # seconds from race start
    d: float                     # metres-from-finish (arc-length)
    off: float                   # lateral metres outward from current inner rail
    speed: float                 # m/s along track (d-rate)
    accel: float = 0.0           # m/s^2 along track
    x: Optional[float] = None    # optional world metric coords
    y: Optional[float] = None
    gap: bool = False            # True => motion-model-filled (occluded), not observed
    conf: float = 1.0


@dataclass
class TrafficEvent:
    """A derived traffic / interference event for one runner."""
    t: float
    kind: str                    # "blocked"|"steadied"|"checked"|"switched"|"wide"
    d: float                     # where (metres-from-finish)
    severity: float = 0.0        # 0..1
    note: str = ""


@dataclass
class RunnerTrajectory:
    """The per-runner deliverable."""
    runner: int                  # saddlecloth number (the canonical key component)
    horse_id: Optional[str] = None
    points: list[TrajPoint] = field(default_factory=list)
    events: list[TrafficEvent] = field(default_factory=list)
    # QA / provenance
    total_distance: Optional[float] = None     # integrated path length (m)
    ground_lost: Optional[float] = None        # extra over race distance (m)
    id_confidence: float = 0.0                 # identity binding confidence 0..1
    coverage: float = 0.0                      # fraction of race with observed (non-gap) points

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RaceTrajectory:
    """Full race output: context + per-runner trajectories + QA summary."""
    context: RaceContext
    runners: list[RunnerTrajectory] = field(default_factory=list)
    qa: dict = field(default_factory=dict)     # see qa() in export.py

    def as_dict(self) -> dict:
        return asdict(self)


# Documented dict form (for non-Python consumers / Parquet column reference):
#
# RaceTrajectory = {
#   "context": {venue, date, race_no, distance, surface, course, fps, mode},
#   "runners": [
#     {
#       "runner": int, "horse_id": str|None,
#       "points": [{t, d, off, speed, accel, x, y, gap, conf}, ...],
#       "events": [{t, kind, d, severity, note}, ...],
#       "total_distance": float, "ground_lost": float,
#       "id_confidence": float, "coverage": float,
#     }, ...
#   ],
#   "qa": {section_time_errors: {...}, finish_order_match: bool, ...},
# }

__all__ = [
    "RaceContext", "Detection", "Tracklet", "IdentityVote", "Homography",
    "TrajPoint", "TrafficEvent", "RunnerTrajectory", "RaceTrajectory",
]
