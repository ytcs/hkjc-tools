"""Stage 2 — DETECT: per-frame horse / marker boxes.

Default detector: Ultralytics YOLO (v11). Escalate to RT-DETR for dense,
occluding horses (Mode A). Heavy imports (ultralytics/torch) are lazy.
"""
from __future__ import annotations

from typing import Iterable

from .ingest import Frame
from .schema import Detection


class Detector:
    """Wraps a YOLO/RT-DETR model. Import + load happen lazily on first use."""

    def __init__(self, weights: str = "yolo11x.pt", classes=("horse", "marker")) -> None:
        self.weights = weights
        self.classes = tuple(classes)
        self._model = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy heavy import
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, frame: Frame) -> list[Detection]:
        """Return detections for one frame. STUB."""
        raise NotImplementedError(
            "Stub. model = self._ensure(); res = model(frame.image)[0]\n"
            "Map each box -> Detection; set foot_(u,v) = bottom-centre of the box\n"
            "(the ground-contact point — project THIS, not the box centre; see\n"
            "CV_PIPELINE Stage 6). Mode B: also detect 'marker'/'number' classes."
        )

    def detect_many(self, frames: Iterable[Frame]) -> list[list[Detection]]:
        return [self.detect(f) for f in frames]


def detect_markers(frame: Frame) -> list[Detection]:
    """Mode B helper: locate on-screen distance markers / rail glyphs for calibration.

    In the virtual render these are consistent and can be found by a small fine-tune
    or even classical template/colour matching. STUB.
    """
    raise NotImplementedError("Stub — Mode B marker detection for homography references.")
