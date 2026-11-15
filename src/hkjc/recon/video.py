"""Video frame access — AVV render panel: render gating, 10 Hz de-dup, static-HUD masks.

The MP4 is 25 fps but the render updates at 10 Hz (pattern new,dup,new,dup,dup), so only ~2/5 of
frames carry new information; downstream works on the UNIQUE frames, each keeping its true video
frame index (its timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import numpy as np

# Static screen overlays inside the render panel (panel-relative px, 1080p product).
DISCLAIMER_ROWS = 110  # top band: CN/EN disclaimer text
ODO_COLS = (-320, -140)  # odometer box columns (from the right edge) ...
ODO_ROWS = 100  # ... within the top rows


@dataclass(frozen=True)
class PanelBox:
    """Absolute pixel box of the render panel's CONTENT (letterbox removed)."""

    x0: int
    x1: int
    y0: int
    y1: int

    def crop(self, img):
        return img[self.y0 : self.y1, self.x0 : self.x1]

    @property
    def shape(self):
        return (self.y1 - self.y0, self.x1 - self.x0)


def find_panel_box(img) -> PanelBox:
    """Render-panel content box from one full frame (right half, non-black rows/cols)."""
    import cv2

    w = img.shape[1]
    g = cv2.cvtColor(img[:, w // 2 :], cv2.COLOR_BGR2GRAY)
    cols = np.where(np.percentile(g, 90, axis=0) > 20)[0]
    rows = np.where(np.percentile(g, 90, axis=1) > 20)[0]
    # the panel abuts the left panel with a thin seam: start a few px in
    return PanelBox(
        w // 2 + int(cols[0]) + 2, w // 2 + int(cols[-1]) - 1, int(rows[0]) + 1, int(rows[-1])
    )


def legend_top(panel) -> int:
    """Top row of the bottom silk-legend strip (dense band of vivid boxes), panel-relative."""
    import cv2

    h = panel.shape[0]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    vivid = (
        (hsv[..., 1] > 110) & (hsv[..., 2] > 110) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 80))
    ).astype(np.uint8)
    band = vivid.sum(1).astype(float)
    band[: int(0.65 * h)] = 0
    if band.max() < 0.05 * panel.shape[1]:
        return int(0.80 * h)
    rows = np.where(band > 0.30 * band.max())[0]
    return int(rows.min())


def hud_mask(panel, legend_row: int | None = None) -> np.ndarray:
    """uint8 {0,1} mask of panel pixels that belong to the WORLD (not a screen overlay)."""
    h, w = panel.shape[:2]
    m = np.ones((h, w), np.uint8)
    m[:DISCLAIMER_ROWS] = 0
    m[:ODO_ROWS, w + ODO_COLS[0] : w + ODO_COLS[1]] = 0
    lt = legend_top(panel) if legend_row is None else legend_row
    m[max(0, lt - 8) :] = 0
    return m


# Render-active gates:
#   * TOP_GREEN_MAX  the render's top strip is the dark odometer/clock box (low green fraction); a
#                    blended/live frame leaks turf into it (high green fraction).
#   * TRACK_LAP_MIN  a clean top-down render is sharp; a blended frame is smooth/ghosted (low texture).
TOP_GREEN_MAX = 0.30
TRACK_LAP_MIN = 240.0


def _green_frac(hsv_band) -> float:
    """Fraction of a HSV band that is turf-green (hue 35-90, sat>60)."""
    h, s = hsv_band[:, :, 0], hsv_band[:, :, 1]
    return float(((h > 35) & (h < 90) & (s > 60)).mean())


def render_active(panel, *, relaxed: bool = False) -> bool:
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

    def legend_boxes(lo, hi):
        # colour conversions are per-pixel: convert ONLY the strips the gates read
        hs = cv2.cvtColor(panel[int(lo * ph) : int(hi * ph)], cv2.COLOR_BGR2HSV)
        h, s, v = hs[:, :, 0], hs[:, :, 1], hs[:, :, 2]
        nongreen = ((s > 110) & (v > 110) & ((h < 35) | (h > 90))).astype("uint8")
        nongreen = cv2.morphologyEx(nongreen, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
        n, _lbl, st, _c = cv2.connectedComponentsWithStats(nongreen, 8)
        return sum(
            1 for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 120 and st[k, cv2.CC_STAT_WIDTH] > 8
        )

    # The legend's height varies by race layout: on ST 0419 R4 / 0426 R9 it sits at ~0.67 of the
    # frame, so only the bottom 4-8 px of its boxes entered the 0.70-0.86 strip and the gate
    # flickered OFF for whole stretches (ST9: 336 race frames, ~13 s; ST4: 84 -> every tracklet broke
    # at 432 m out). Fallback (``relaxed``, only once the strict test has passed in this replay, so it
    # fills holes INSIDE the render segment): >= 5 boxes in a strip widened to 0.60 (MEASURED: legend
    # frames show 6-12 there, pre-race live frames <= 3). REJECTED: the widened strip with >= 1 box
    # (admitted 11-65 pre-race live frames per race); the fallback before the first strict frame
    # (admitted a dissolve-tail frame of HV 0527 R3, which flipped its result 5/5 -> 2/5; px/m unchanged).
    strict = legend_boxes(0.70, 0.86) >= 1
    if not strict and not (relaxed and legend_boxes(0.60, 0.86) >= 5):
        return False
    top = cv2.cvtColor(panel[: int(0.30 * ph)], cv2.COLOR_BGR2HSV)
    if _green_frac(top) > TOP_GREEN_MAX:  # live turf/horses leak into the top strip
        return False
    gray = cv2.cvtColor(panel[int(0.34 * ph) : int(0.70 * ph)], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= TRACK_LAP_MIN


def is_duplicate(a, b, tol: float = 3.0) -> bool:
    """Same render tick? (decoded duplicates differ only by codec noise, mean |diff| << 1)."""
    return float(np.abs(a[::4, ::4].astype(np.int16) - b[::4, ::4]).mean()) < tol


def grid_keep(idx) -> np.ndarray:
    """Boolean keep-mask over yielded frame indices: only frames ON the render's tick grid.

    The render updates at exactly 10 Hz in 25 fps video, so new ticks start at two fixed residues of
    ``frame % 5`` for a whole replay. MEASURED on all 26 races: one phase fits 99.9-100 % of the
    yielded frames, and the first 40 ticks already give the same residues as the whole race. The
    off-grid frames are re-encodes of the CURRENT tick that the fixed duplicate tolerance let through:
    HV 04-22 R1/R9 have a video keyframe every 100 frames (codec noise 3.8 > tolerance 3.0, so 14 zero-
    motion "ticks" per race), plus the first render frame and a few frames after render-gate dropouts
    at the finish on the other races. The residues come from a majority vote over the whole replay;
    the filter is skipped (all kept) if the top two residues do not hold >= 90 % of the frames or are
    not 2-3 frames apart (not a 10 Hz render).

    Filtering the de-dup stream AFTERWARDS is exact. An off-grid frame stays the duplicate
    reference for the next frame, so the next real tick is compared against the keyframe's codec
    noise and passes as it did before.
    REJECTED: ``skip_gap1`` (drop a frame 1 after the last one, keep the old reference). The frame
    after the keyframe then differed from the stale reference by the keyframe's noise and entered as
    a zero-motion tick, and the next real ticks were stamped 1 frame late: HV 04-22 index
    399,401,403,405 instead of 399,402,404; stalls 14 -> 14 per race (pod run W)."""
    r = np.asarray(idx, dtype=np.int64) % 5
    n = len(r)
    cnt = np.bincount(r, minlength=5)
    top = np.argsort(-cnt, kind="stable")[:2]
    if n == 0 or cnt[top].sum() < 0.9 * n or (int(top[0]) - int(top[1])) % 5 not in (2, 3):
        return np.ones(n, bool)
    return np.isin(r, top)


def iter_unique_render(
    path,
    *,
    box: PanelBox | None = None,
    start: int = 0,
    stop: int | None = None,
    keep_full: bool = False,
):
    """Yield ``(frame_idx, panel[, full])`` for render-active frames whose panel CHANGED vs the
    previous yielded one (the 10 Hz de-dup). ``box`` is auto-found on the first frame if None."""
    import av

    prev = None
    seen = False  # the strict render test has passed: the legend-strip fallback may fill holes
    with av.open(str(path)) as c:
        c.streams.video[0].thread_type = "AUTO"  # frame-threaded decode: same pixels, ~3x faster
        for i, fr in enumerate(c.decode(video=0)):
            if i < start:
                continue
            if stop is not None and i >= stop:
                break
            img = fr.to_ndarray(format="bgr24")
            active = render_active(img[:, img.shape[1] // 2 :], relaxed=seen)  # relaxed >= strict
            seen = seen or active
            if not active:
                prev = None
                continue
            if box is None:
                box = find_panel_box(img)
            p = box.crop(img)
            if prev is not None and is_duplicate(p, prev):
                continue
            prev = p
            yield (i, p, img) if keep_full else (i, p)


# ----------------------------------------------------------------------------- frame cache
class FrameCache:
    """Unique render panels decoded ONCE (JPEG q92 in one binary file + offsets) — every later
    stage reads this instead of re-decoding the 1080p MP4 (was 4 full decodes + render gating per
    race). Also stores the left-panel race-clock crop per frame."""

    def __init__(self, root):
        from pathlib import Path

        self.root = Path(root)
        z = np.load(self.root / "frames_idx.npz")
        self.idx, self.off, self.box = z["idx"], z["off"], tuple(z["box"])
        self.clock = np.load(self.root / "frames_clock.npy", mmap_mode="r")
        self._bin = np.memmap(self.root / "frames.bin", dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self.idx)

    def panel(self, k):
        import cv2

        return cv2.imdecode(np.asarray(self._bin[self.off[k] : self.off[k + 1]]), cv2.IMREAD_COLOR)

    def __iter__(self):
        for k in range(len(self.idx)):
            yield int(self.idx[k]), self.panel(k)

    @staticmethod
    def build(path, root, clock_box=(288, 342, 800, 945), quality=92):
        import cv2
        from pathlib import Path

        root = Path(root)
        idx, bufs, clocks, box = [], [], [], None
        for i, p, full in iter_unique_render(path, keep_full=True):
            box = box or find_panel_box(full)
            ok, buf = cv2.imencode(".jpg", p, [cv2.IMWRITE_JPEG_QUALITY, quality])
            idx.append(i)
            bufs.append(buf.tobytes())
            y0, y1, x0, x1 = clock_box
            clocks.append(full[y0:y1, x0:x1].copy())
        if os.environ.get("HKJC_FRAME_GRID", "0") != "0":  # see grid_keep
            keep = np.flatnonzero(grid_keep(idx))
            idx, bufs, clocks = [idx[k] for k in keep], [bufs[k] for k in keep], [clocks[k] for k in keep]
        off = [0]
        with open(root / "frames.bin", "wb") as fb:
            for b in bufs:
                fb.write(b)
                off.append(off[-1] + len(b))
        np.savez(
            root / "frames_idx.npz",
            idx=np.array(idx),
            off=np.array(off),
            box=np.array([box.x0, box.x1, box.y0, box.y1]),
        )
        np.save(root / "frames_clock.npy", np.array(clocks))
        return len(idx)
