"""Aerial layout: split the Aerial Virtual View frame into its fixed ROIs.

The 1920x1080 frame is letter-boxed; content sits in a horizontal band. The LEFT
half is a real broadcast pan (occluded, motion-blurred -> we ignore it). The RIGHT
half is the synthetic top-down render we use. Within the right half:

* a top text/odometer strip (~top 90 px of content) -> the "920M" odometer lives there
* a bottom legend strip (silk icons + numbers in running order)
* the TRACK ROI in between -> where the runners + drawn rails are

The split (x = W/2) is fixed by the product. The content band and legend row drift a
little frame to frame, so we locate the legend band from the data each frame and derive
the track ROI from it. numpy/cv2 imported lazily by the caller; this module takes arrays.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    """Per-frame ROIs for the right (synthetic) panel, in absolute pixel coords."""

    panel_x0: int          # left edge of the right (synthetic) panel
    panel_x1: int          # right edge (content)
    track_y0: int          # top of the track ROI (below the text strip)
    track_y1: int          # bottom of the track ROI (above the legend)
    legend_y0: int         # top of the legend strip
    legend_y1: int         # bottom of the legend strip
    aerial_active: bool    # is the synthetic view present this frame? (start-gate frames are not)
    odo_x0: int            # odometer ("420M") box, top-right of the synthetic panel
    odo_x1: int
    odo_y0: int
    odo_y1: int
    clock_x0: int          # broadcast clock + published splits, top of the LEFT panel
    clock_x1: int
    clock_y0: int
    clock_y1: int

    @property
    def odo_roi(self) -> tuple[int, int, int, int]:
        """(y0, y1, x0, x1) slice of the distance-to-go odometer box."""
        return (self.odo_y0, self.odo_y1, self.odo_x0, self.odo_x1)

    @property
    def clock_roi(self) -> tuple[int, int, int, int]:
        """(y0, y1, x0, x1) slice of the left-panel clock + published splits region."""
        return (self.clock_y0, self.clock_y1, self.clock_x0, self.clock_x1)


def find_layout(img, *, min_legend_sat: int = 120) -> Layout:
    """Locate the right-panel ROIs for one BGR frame (HxWx3 ndarray).

    Strategy: the legend strip is the only place in the right half with a dense band of
    *vivid* (high-sat/high-val) pixels low in the frame — the rendered silk icons. We find
    the peak saturated-pixel row in the bottom half; if it clears ``min_legend_sat`` the
    aerial view is active and the legend sits in a ~50 px band around it; the track ROI is
    everything from below the top text strip down to just above the legend.
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    panel_x0 = w // 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    vivid = ((sat > 110) & (val > 110)).astype(np.uint8)
    vivid[:, :panel_x0] = 0
    rowsum = vivid.sum(axis=1)
    lo = int(0.55 * h)              # legend is low in the frame
    band = rowsum.copy()
    band[:lo] = 0
    legend_y = int(np.argmax(band))
    peak = int(band[legend_y])

    # right content x-extent: columns that are not pure letterbox
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colmax = gray.max(axis=0)
    rcols = np.where(colmax[panel_x0:] > 25)[0]
    panel_x1 = (panel_x0 + int(rcols.max())) if len(rcols) else (w - 1)

    aerial_active = peak >= min_legend_sat
    legend_y0 = max(lo, legend_y - 28)
    legend_y1 = min(h - 1, legend_y + 32)
    track_y0 = int(0.34 * h)        # below the top text/odometer strip
    track_y1 = max(track_y0 + 1, legend_y0 - 8)
    return Layout(
        panel_x0=panel_x0 + 10,
        panel_x1=panel_x1 - 6,
        track_y0=track_y0,
        track_y1=track_y1,
        legend_y0=legend_y0,
        legend_y1=legend_y1,
        aerial_active=aerial_active,
        # The distance-to-go odometer sits in a dark map box at the top-RIGHT of the
        # synthetic panel; the official clock + published splits live at the top of the
        # LEFT (broadcast) panel. Both are at fixed fractional positions in this product.
        # x0 starts at 0.83 (not 0.80): the left ~1/4 of the box is sponsor "headphones"
        # text — pure OCR noise; dropping it tightens precision without clipping the numerals.
        odo_x0=int(0.83 * w),
        odo_x1=int(0.92 * w),
        odo_y0=int(0.27 * h),
        odo_y1=int(0.34 * h),
        clock_x0=int(0.40 * w),
        clock_x1=int(0.50 * w),
        clock_y0=int(0.27 * h),
        clock_y1=int(0.45 * h),
    )


# Clean-render gate thresholds (AVV product constants). The two gates catch complementary signatures of
# the live->render cross-dissolve that the legend strip alone misses:
#   * TOP_GREEN_MAX  the render's top strip is the dark odometer/clock box (low green fraction); a
#                    blended/live frame leaks turf into it (high green fraction).
#   * TRACK_LAP_MIN  a clean top-down render is sharp; a blended frame is smooth/ghosted (low texture).
TOP_GREEN_MAX = 0.30
TRACK_LAP_MIN = 240.0


def _green_frac(hsv_band) -> float:
    """Fraction of a HSV band that is turf-green (hue 35-90, sat>60)."""
    h, s = hsv_band[:, :, 0], hsv_band[:, :, 1]
    return float(((h > 35) & (h < 90) & (s > 60)).mean())


def render_active(panel) -> bool:
    """Is a CLEAN AVV 3D render present in this RIGHT panel (vs the live broadcast / a dissolve frame)?

    The replay opens with the live broadcast before switching to the split-screen render, with a short
    cross-dissolve between. Three gates, all required:

      (1) the SILK-LEGEND strip is present (a row of saturated non-green boxes the live footage lacks);
      (2) the TOP strip is the dark odometer box, not turf (``_green_frac`` <= TOP_GREEN_MAX);
      (3) the TRACK band has render-level detail, not a smooth/ghosted blend (Laplacian var >=
          TRACK_LAP_MIN).

    The legend (1) flickers on during the dissolve, so (2)+(3) reject the blended frames it lets
    through. numpy/cv2 lazy-imported.
    """
    import cv2
    import numpy as np

    ph = panel.shape[0]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    strip = slice(int(0.70 * ph), int(0.86 * ph))
    nongreen = ((s[strip] > 110) & (v[strip] > 110) & ((h[strip] < 35) | (h[strip] > 90))).astype("uint8")
    nongreen = cv2.morphologyEx(nongreen, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
    n, _lbl, st, _c = cv2.connectedComponentsWithStats(nongreen, 8)
    boxes = sum(1 for k in range(1, n)
                if st[k, cv2.CC_STAT_AREA] > 120 and st[k, cv2.CC_STAT_WIDTH] > 8)
    if boxes < 1:
        return False
    if _green_frac(hsv[:int(0.30 * ph)]) > TOP_GREEN_MAX:    # live turf/horses leak into the top strip
        return False
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)[int(0.34 * ph):int(0.70 * ph)]
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= TRACK_LAP_MIN


def content_box(panel):
    """Tight (x0, x1, y0, y1) of the non-letterbox CONTENT in a render panel — the rows/cols that are
    not the black border. 'Spans the frame' then means spans the CONTENT, not the padded panel, so the
    rail-span gates are resolution/letterbox-invariant. cv2/numpy lazy-imported."""
    import cv2
    import numpy as np

    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    cols = np.where(g.max(0) > 25)[0]
    rows = np.where(g.max(1) > 25)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, panel.shape[1], 0, panel.shape[0]
    return int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
