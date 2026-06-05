"""Stage 3 — TRACK: link detections into locally-consistent tracklets.

Default: BoT-SORT (motion + camera-motion compensation + Re-ID). ByteTrack as the
fast baseline; OC-SORT when long full occlusions cause ID switches. Trackers give
*temporary* IDs only — identity is fixed in Stage 4 (identify.py).

NB: named ``track.py`` for the *pipeline stage*; unrelated to ``src/hkjc/track.py``
(the world geometry), which we never modify.
"""
from __future__ import annotations

from .schema import Detection, Tracklet


class Tracker:
    """Wraps a multi-object tracker. Heavy import is lazy."""

    def __init__(self, method: str = "botsort") -> None:
        # "botsort" (default, Re-ID) | "bytetrack" (fast) | "ocsort" (occlusion)
        self.method = method
        self._state = None

    def update(self, frame_idx: int, detections: list[Detection]) -> list[Detection]:
        """Feed one frame's detections, return them tagged with track ids. STUB."""
        raise NotImplementedError(
            "Stub. Use Ultralytics' built-in tracker (model.track(..., tracker='botsort.yaml'))\n"
            "or a standalone BoT-SORT/ByteTrack/OC-SORT. Attach a track_id per detection.\n"
            "Trackers are only locally consistent; Stage 4 re-anchors identity across switches."
        )

    def tracklets(self) -> list[Tracklet]:
        """Collapse the streamed updates into per-id Tracklets. STUB."""
        raise NotImplementedError("Stub — group tagged detections by track_id into Tracklets.")
