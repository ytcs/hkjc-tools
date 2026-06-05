"""CV trajectory pipeline — SKELETON / SCAFFOLD.

Extract per-horse dense track trajectories (d = metres-from-finish, off = lateral
metres from the inner rail, speed) from race video pixels, to feed the physics
energetics model (v_c, W'). Design doc: ``docs/HKJC_DATA.md (§ trajectory)``.

Stages: ingest -> detect -> track -> identify -> homography -> project -> export(+qa).

This package imports cleanly with NO heavy deps installed: cv2 / torch /
ultralytics / pyav / pandas are imported *lazily inside functions*. Only the schema
dataclasses (pure stdlib) are exported eagerly. Does NOT modify ``src/hkjc``.
"""
from __future__ import annotations

from .schema import (
    RaceContext, Detection, Tracklet, IdentityVote, Homography,
    TrajPoint, TrafficEvent, RunnerTrajectory, RaceTrajectory,
)

__all__ = [
    "RaceContext", "Detection", "Tracklet", "IdentityVote", "Homography",
    "TrajPoint", "TrafficEvent", "RunnerTrajectory", "RaceTrajectory",
]
