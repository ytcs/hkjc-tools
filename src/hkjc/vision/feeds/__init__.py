"""Per-feed observation models — the PERCEPTION half of the 4-feed race reconstruction.

Each module is a faithful port of one feed's detector/extraction functions (rail fit, horse
detection, ground-contact, lateral/longitudinal/margin readout, per-feed clock anchors). The
per-race constants the studies hard-coded (video path, distance, track width, section times,
clock offsets, OCR windows, field size) are threaded as PARAMETERS — grouped into a frozen
:class:`FeedContext` where a function needs many. No module-global re-pointing.

Feeds are pure perception: they take geometry scalars from the caller and must NOT import
``hkjc.reconstruction`` or ``hkjc.data``. Heavy deps (cv2, av, scipy, rapidocr) are lazy.

See ``docs/CV_RECON.md`` for the validated method and the fusion that consumes these.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeedContext:
    """Per-race scalars one feed needs, threaded from the caller (no per-race module globals).

    Only the fields a given feed reads are required; the rest carry race-agnostic defaults.
    All distances are metres, all times seconds. ``sections`` are the official cumulative-sum
    section times (length adapts 3/4/5); ``clock_offset`` maps that feed's video time to the
    shared ``race_t`` (see each feed for its convention).
    """

    video: str = ""
    distance: float = 0.0            # race distance (m)
    track_width_m: float = 0.0       # usable track width for the venue x course (m)
    field_size: int = 12
    sections: tuple = ()             # official per-section times (s), in order
    section_bounds: tuple = ()       # cumulative arclen at section boundaries (m); len = len(sections)+1
    clock_offset: float = 0.0        # feed-specific video->race_t offset
    fps: float | None = None
    windows: dict = field(default_factory=dict)  # named (t0, t1, dt) OCR/extraction windows
