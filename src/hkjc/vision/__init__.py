"""CV perception for race reconstruction — per-feed pixel -> observation.

Pure perception: the per-feed observation models (:mod:`~hkjc.vision.feeds`), the silk
identity descriptor (:mod:`~hkjc.vision.silk`, :mod:`~hkjc.vision.identity`) and the
shared CV primitives (:mod:`~hkjc.vision.geometry_cv`, :mod:`~hkjc.vision.ground_contact`,
:mod:`~hkjc.vision.ocr`). The output coordinate schema (``d`` = metres-from-finish,
``off`` = lateral metres from the inner rail, ``speed``) lives in :mod:`~hkjc.vision.schema`.

Orchestration — the shared clock, the fusion smoother and the ``Reconstructor`` that turns
four feeds into per-horse trajectories — lives in :mod:`hkjc.reconstruction` (on top of this
package + ``core``/``data``). This package NEVER imports ``data``.

Imports cleanly with NO heavy CV deps: cv2 / av / rapidocr load *lazily inside functions*;
only the pure-stdlib schema dataclasses are exported eagerly.
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
