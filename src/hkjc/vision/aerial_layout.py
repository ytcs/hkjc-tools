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
        odo_x0=int(0.80 * w),
        odo_x1=int(0.92 * w),
        odo_y0=int(0.27 * h),
        odo_y1=int(0.34 * h),
        clock_x0=int(0.40 * w),
        clock_x1=int(0.50 * w),
        clock_y0=int(0.27 * h),
        clock_y1=int(0.45 * h),
    )
