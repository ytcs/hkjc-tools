"""Stage 5 — CALIBRATE / HOMOGRAPHY: pixel <-> world planar map.

Mode A (broadcast): per-frame H from rail/marker landmarks + frame-to-frame
chaining + smoothing; escalate to learned field registration. The hard stage.
Mode B (render): a few homographies from EXACT on-screen markers — much lighter.

World references come from ``src/hkjc/track.TrackGeometry`` (rail offset for the
``course``, sectional/marker ``d`` values, ``home_straight``/``width`` for scale).
cv2 imported lazily.
"""
from __future__ import annotations

from typing import Optional

from .schema import Homography, RaceContext


def world_references(ctx: RaceContext) -> list[tuple[float, float, str]]:
    """Known (x, y, label) world landmarks for this race, from the geometry (read-only).

    e.g. finish line (d=0), sectional/distance markers at known d, rail endpoints at
    the current course offset, winning post. STUB — derive from track_for(ctx...).
    """
    raise NotImplementedError(
        "Stub. from hkjc.track import track_for; geo = track_for(ctx.venue, ctx.distance,\n"
        "ctx.surface); emit world landmarks (marker d-values, rail offset for ctx.course,\n"
        "home_straight/width for scale) as (x, y, label)."
    )


def solve_homography(
    pixel_refs: list[tuple[float, float]],
    world_refs: list[tuple[float, float]],
) -> Optional[Homography]:
    """Solve H from >=4 pixel<->world correspondences (Mode A landmark / Mode B marker)."""
    raise NotImplementedError(
        "Stub. import cv2 (lazy); H, _ = cv2.findHomography(src, dst, cv2.RANSAC). "
        "Wrap as Homography(matrix=H.flatten().tolist(), valid=True, n_refs=len(...))."
    )


def chain_homography(prev: Homography, frame_pts, next_pts) -> Homography:
    """Mode A: propagate H between landmark anchors via frame-to-frame matching.

    ORB/LoFTR/SuperGlue inter-frame homography, chained from the last absolute anchor;
    DRIFTS — must be re-anchored when a landmark reappears, then RTS-smoothed. STUB.
    """
    raise NotImplementedError("Stub — frame-to-frame chaining for landmark-sparse stretches.")


def calibrate_sequence(ctx: RaceContext, marker_dets_per_frame) -> list[Homography]:
    """Top-level: produce a per-frame (or per-shot) homography series.

    Mode B path: solve per shot from exact markers (often a handful covers a race).
    Mode A path: absolute solve where >=4 refs visible, chain + smooth between. STUB.
    """
    raise NotImplementedError("Stub — orchestrate solve/chain/smooth per CV_PIPELINE Stage 5.")
