"""Stage 1 — INGEST: video file -> timestamped frames + race context.

Heavy deps (pyav / cv2) are imported lazily *inside* functions so this module
imports with nothing installed. Prefer pyav for exact PTS timestamps; OpenCV is
the simpler fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from .schema import RaceContext


@dataclass
class Frame:
    """A decoded frame: index, timestamp (s from start), and pixels (ndarray)."""
    index: int
    t: float
    image: object  # HxWx3 ndarray (numpy) — typed loosely to stay numpy-free here


def open_video(path: str, target_fps: Optional[float] = None) -> "VideoReader":
    """Return a lazy frame reader. Does not decode until iterated."""
    return VideoReader(path, target_fps=target_fps)


class VideoReader:
    """Lazy frame source. Decodes native fps; downsamples in-pipeline (see CV_PIPELINE §4)."""

    def __init__(self, path: str, target_fps: Optional[float] = None) -> None:
        self.path = path
        self.target_fps = target_fps
        self.native_fps: Optional[float] = None

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError(
            "Stub. Implement with pyav (preferred, exact PTS) or cv2.VideoCapture.\n"
            "  import av  # lazy\n"
            "  container = av.open(self.path)\n"
            "  for i, frame in enumerate(container.decode(video=0)):\n"
            "      t = float(frame.pts * frame.time_base)\n"
            "      if self.target_fps and downsample-skip: continue\n"
            "      yield Frame(i, t, frame.to_ndarray(format='bgr24'))"
        )


def resolve_geometry(ctx: RaceContext):
    """Resolve the world model for this race from ``src/hkjc/track.py`` (read-only)."""
    from hkjc.track import track_for  # lazy: keeps package importable standalone
    return track_for(ctx.venue, ctx.distance, ctx.surface)
