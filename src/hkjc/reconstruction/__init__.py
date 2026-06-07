"""Race reconstruction — fuse HKJC's replay feeds into per-horse trajectories.

The orchestration layer ON TOP of two packages:

- :mod:`hkjc.vision` — pure perception: per-feed pixel -> observation models, the silk
  identity descriptor, shared CV primitives. Never imports ``data``.
- :mod:`hkjc.core` / :mod:`hkjc.data` — the race itself: track geometry, the barrier
  draw, official sectionals.

So race facts (width, bend map, draw, sections) are read once from :class:`hkjc.core.Race`
and never re-encoded in the CV code. Pipeline: per-feed observations -> shared clock ->
identity consensus -> constrained physics-kernel fusion -> :class:`hkjc.vision.RaceTrajectory`.

Methods + the validated state live in ``docs/CV_RECON.md``; heavy CV deps load lazily.
"""
from __future__ import annotations

from .clock import ClockAnchors, anchors_from_recipe, section_curve
from .fusion import Obs, smooth_field
from .geometry import RaceGeom, draw_to_start_lateral
from .reconstruct import RaceRecipe, Reconstructor

__all__ = [
    "RaceGeom", "draw_to_start_lateral",
    "ClockAnchors", "anchors_from_recipe", "section_curve",
    "Obs", "smooth_field",
    "RaceRecipe", "Reconstructor",
]
