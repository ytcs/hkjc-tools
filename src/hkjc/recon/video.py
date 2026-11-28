"""Video frame access — AVV render panel: render gating, 10 Hz de-dup, static-HUD masks.

The MP4 is 25 fps but the render updates at 10 Hz (pattern new,dup,new,dup,dup), so only ~2/5 of
frames carry new information; downstream works on the UNIQUE frames, each keeping its true video
frame index (its timestamp).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np


@contextmanager
def atomic(path):
    """Yield a temporary sibling of ``path`` (same suffix) to write; renamed onto ``path`` only when
    the block completes, so a killed or failed writer never leaves a partial artifact behind."""
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.part{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

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


BOX_JITTER = 3  # px: frame-to-frame jitter of the detected panel edges (antialiased borders)


def frame_box(img, layout_box: PanelBox | None = None) -> PanelBox:
    """``find_panel_box(img)``, or ``layout_box`` (the replay's sampled layout) when they differ by
    more than the edge jitter: a single frame can be a crossfade whose letterbox is not yet black."""
    b = find_panel_box(img)
    if layout_box is None:
        return b
    d = np.abs(np.subtract(astuple(b), astuple(layout_box))).max()
    return b if d <= BOX_JITTER else layout_box


LEGEND_TOL = 8  # px: a frame's legend row farther than this from the race's list row is replaced by it


def legend_scan(panel) -> tuple[int, bool]:
    """(top row of the bottom silk-legend strip, panel-relative: the first row whose vivid-pixel count exceeds 30 % of
    the strip's maximum; whether a strip was found: maximum >= 5 % of the panel width, else 0.8 x the panel height)."""
    import cv2

    h = panel.shape[0]
    top = int(0.65 * h)  # the strip is searched below this row only: convert only those rows
    hsv = cv2.cvtColor(panel[top:], cv2.COLOR_BGR2HSV)
    # s > 110, v > 110, h < 25 or h > 80, as 0/1
    vivid = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 111, 111), (24, 255, 255)), cv2.inRange(hsv, (81, 111, 111), (255, 255, 255))
    )
    vivid &= 1
    band = vivid.sum(1).astype(float)
    if band.max(initial=0.0) < 0.05 * panel.shape[1]:
        return int(0.80 * h), False
    rows = np.where(band > 0.30 * band.max())[0]
    return int(rows.min()) + top, True


def legend_top(panel) -> int:
    """Top row of the bottom silk-legend strip of one panel (``legend_scan``)."""
    return legend_scan(panel)[0]


def stable_legend_rows(rows, found) -> np.ndarray:
    """A race's per-frame legend rows (``legend_scan``): where the strip was found, a row more than LEGEND_TOL px from the
    race's list row (the median over those frames) is replaced by it (a vivid band below the list, e.g. the red apron
    in the final view, raises the strip maximum and pushes the row under the icons); frames without a strip keep theirs."""
    rows, found = np.asarray(rows, int), np.asarray(found, bool)
    if not found.any():
        return rows.copy()
    lrow = int(np.median(rows[found]))
    return np.where(found & (np.abs(rows - lrow) > LEGEND_TOL), lrow, rows)


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
ODO_YELLOW_MIN = 0.05  # circle render: yellow fraction of the odometer box


def _green_frac(hsv_band) -> float:
    """Fraction of a HSV band that is turf-green (hue 35-90, sat>60)."""
    import cv2

    n = cv2.countNonZero(cv2.inRange(hsv_band, (36, 61, 0), (89, 255, 255)))  # 35 < h < 90, s > 60
    return n / (hsv_band.shape[0] * hsv_band.shape[1])


def _texture_var_at_least(gray, thr: float) -> bool:
    """``cv2.Laplacian(gray, CV_64F).var() >= thr``. The 3x3 Laplacian of 8-bit pixels is an integer below 2^11, so
    the 16-bit Laplacian holds the same values and ``meanStdDev`` sums them exactly; its variance can differ from
    numpy's only by rounding (~1e-15 relative), so it decides unless it lies within 1e-6 of ``thr``, where numpy's
    variance of the float64 Laplacian decides as before."""
    import cv2

    _m, sd = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
    v = float(sd[0, 0]) ** 2
    if abs(v - thr) > 1e-6 * thr:
        return v >= thr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= thr


def render_active(panel, *, relaxed: bool = False, style: str = "silk") -> bool:
    """Is a CLEAN AVV 3D render present in this RIGHT panel (vs the live broadcast / a dissolve frame)?

    The replay opens with the live broadcast before switching to the split-screen render, with a short
    cross-dissolve between. Three gates, all required:

      (1) the SILK-LEGEND strip is present (a row of saturated non-green boxes the live footage lacks);
      (2) the TOP strip is the dark odometer box, not turf (``_green_frac`` <= TOP_GREEN_MAX);
      (3) the TRACK band has render-level detail, not a smooth/ghosted blend (Laplacian var >=
          TRACK_LAP_MIN).

    The legend (1) flickers on during the dissolve, so (2)+(3) reject the blended frames it lets
    through. ``relaxed`` (only after the strict test has passed in this replay) also accepts >= 5
    legend boxes in a strip widened to 0.60 of the height. ``style='circle'`` (the 2026/27 numbered-
    disc render, no legend) replaces (1) by the render's yellow odometer text plus >= 2 discs
    (``relaxed``: the discs alone) and drops (3).
    """
    import cv2
    import numpy as np

    ph = panel.shape[0]

    def legend_boxes(lo, hi):
        # colour conversions are per-pixel: convert ONLY the strips the gates read
        hs = cv2.cvtColor(panel[int(lo * ph) : int(hi * ph)], cv2.COLOR_BGR2HSV)
        # s > 110, v > 110, h < 35 or h > 90, as 0/1
        nongreen = cv2.bitwise_or(
            cv2.inRange(hs, (0, 111, 111), (34, 255, 255)), cv2.inRange(hs, (91, 111, 111), (255, 255, 255))
        )
        nongreen &= 1
        nongreen = cv2.morphologyEx(nongreen, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
        n, _lbl, st, _c = cv2.connectedComponentsWithStats(nongreen, 8)
        return sum(
            1 for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 120 and st[k, cv2.CC_STAT_WIDTH] > 8
        )

    # The legend height varies by race layout (~0.67 of the frame on some ST races), so the strict
    # 0.70-0.86 strip can miss it; the relaxed 0.60 strip needs >= 5 boxes (legend frames show 6-12
    # there, pre-race live frames <= 3) and only fills holes inside the render segment.
    if style == "circle":  # no legend strip: the render shows numbered discs instead
        from . import circles

        # the render's odometer box (yellow distance-to-go text, fixed screen position): live film and the
        # cross-dissolve lack it (yellow fraction <= 0.02 vs >= 0.10), and the start gate's blue banner can
        # pass as discs
        oy0, oy1, ox0, ox1 = 286, 320, 1606 - 960, 1700 - 960  # ocr.AVV_ODO in right-half coordinates
        hs = cv2.cvtColor(panel[oy0:oy1, ox0:ox1], cv2.COLOR_BGR2HSV)
        yellow = ((hs[..., 0] >= 18) & (hs[..., 0] <= 38) & (hs[..., 1] > 120) & (hs[..., 2] > 150)).mean()
        # ``relaxed`` (the render has started): the odometer text leaves ~20 m out while the render
        # runs on through the finish line to the end of the replay
        if (yellow < ODO_YELLOW_MIN and not relaxed) or circles.count(panel) < 2:
            return False
    else:
        strict = legend_boxes(0.70, 0.86) >= 1
        if not strict and not (relaxed and legend_boxes(0.60, 0.86) >= 5):
            return False
    top = cv2.cvtColor(panel[: int(0.30 * ph)], cv2.COLOR_BGR2HSV)
    if _green_frac(top) > TOP_GREEN_MAX:  # live turf/horses leak into the top strip
        return False
    if style == "circle":  # no texture floor: the smoother 2026/27 render sits below the silk
        return True  # render's (median Laplacian var ~230)
    gray = cv2.cvtColor(panel[int(0.34 * ph) : int(0.70 * ph)], cv2.COLOR_BGR2GRAY)
    return _texture_var_at_least(gray, TRACK_LAP_MIN)


def layout(path, n_runners: int, fracs=tuple(np.linspace(0.2, 0.85, 12))):
    """(style, box) of a replay, from frames sampled across it.

    style: 'silk' (avatar render with the top-8 legend strip), 'circle' (numbered discs, no legend)
    or 'unknown'. A circle frame shows >= max(3, 0.4 x runners) discs (tested first: discs crossing
    the legend strip would pass the silk gate), a silk frame passes the legend render gate; frames
    showing neither (live film, finish replay) abstain. Needs >= 2 votes and a strict plurality.

    box: the render panel's content box, a constant of the broadcast layout: the per-coordinate
    median of ``find_panel_box`` over the voting frames; None when no frame voted."""
    import av

    from . import circles

    votes, boxes = [], []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        dur = float(st.duration * st.time_base) if st.duration else float(c.duration) / 1e6
        for f in fracs:
            c.seek(int(f * dur / st.time_base), stream=st)
            fr = next(c.decode(st), None)
            if fr is None:
                continue
            img = fr.to_ndarray(format="bgr24")
            panel = img[:, img.shape[1] // 2 :]
            if circles.count(panel) >= max(3, 0.4 * n_runners):
                votes.append("circle")
            elif render_active(panel, relaxed=True):  # relaxed: legend height varies by layout
                votes.append("silk")
            else:
                continue
            boxes.append(astuple(find_panel_box(img)))
    ns, nc = votes.count("silk"), votes.count("circle")
    style = "silk" if ns >= 2 and ns > nc else "circle" if nc >= 2 and nc > ns else "unknown"
    box = PanelBox(*(int(v) for v in np.median(boxes, axis=0))) if boxes else None
    return style, box


def is_duplicate(a, b, tol: float = 3.0) -> bool:
    """Same render tick? (decoded duplicates differ only by codec noise, mean |diff| << 1)."""
    return float(np.abs(a[::4, ::4].astype(np.int16) - b[::4, ::4]).mean()) < tol


def _residue_vote(idx):
    """(a, b) with b = a + 2 (mod 5): the top two residues of ``idx % 5``; None if they hold < 90 %
    of ``idx`` or are not 2-3 frames apart."""
    if len(idx) == 0:
        return None
    cnt = np.bincount(idx % 5, minlength=5)
    top = np.argsort(-cnt, kind="stable")[:2]
    a, b = int(top[0]), int(top[1])
    if cnt[top].sum() < 0.9 * len(idx) or (a - b) % 5 not in (2, 3):
        return None
    return (a, b) if (b - a) % 5 == 2 else (b, a)


def tick_grid(idx):
    """(a, b): the two residues of ``frame % 5`` at which 10 Hz render ticks start, with
    b = a + 2 (mod 5), by majority vote over all frames; None if the top two hold < 90 % of the
    frames or are not 2-3 frames apart (not a 10 Hz render).

    When the vote over all frames fails, only frames that follow the previous one by 2-3 frames (the
    10 Hz cadence) vote. Live film that passes the render gate (``film_keep``; e.g. ST 2024-12-29 R8)
    changes on every frame, so a few seconds of it put the top two residues below 90 % of all frames;
    without a grid, ``grid_keep`` keeps every video keyframe re-encode and ``tick_frame`` leaves tick
    times unshifted."""
    idx = np.asarray(idx, dtype=np.int64)
    grid = _residue_vote(idx)
    if grid is None and len(idx) > 1:
        gap = np.diff(idx)
        grid = _residue_vote(idx[1:][(gap == 2) | (gap == 3)])
    return grid


def tick_frame(f, grid) -> np.ndarray:
    """Frame index -> render tick time in frames. Ticks start 2 and 3 frames apart while their
    content advances a uniform 2.5 frames, so on-grid frames move onto that grid (residue a -0.25,
    b +0.25; mean shift zero). Off-grid frames (a tick's only copy after a render dropout) and
    non-10 Hz renders (``grid`` None) are unchanged."""
    f = np.asarray(f, dtype=float)
    if grid is None:
        return f
    r = f.astype(np.int64) % 5
    return f + np.where(r == grid[0], -0.25, np.where(r == grid[1], 0.25, 0.0))


def grid_keep(idx) -> np.ndarray:
    """Boolean keep-mask over yielded frame indices: drop a frame only if it RE-ENCODES a tick
    that is already kept.

    The render updates at exactly 10 Hz in 25 fps video, so new ticks start at two fixed residues of
    ``frame % 5`` for a whole replay (``tick_grid``; nothing is dropped when it finds no grid, i.e.
    not a 10 Hz render).

    An off-grid frame is dropped only when the on-grid start of its tick (the latest on-grid frame
    at or before it) was yielded too: then it is a re-encode of that tick (e.g. a video keyframe
    whose codec noise exceeds the de-dup tolerance). Any other off-grid frame is kept: the first
    render frame and the first frame after a render-gate dropout are the only copy of a real tick
    (stamped 1-2 frames late). Filtering the de-dup stream afterwards is exact: a dropped keyframe
    stays the duplicate reference for the next frame."""
    idx = np.asarray(idx, dtype=np.int64)
    r = idx % 5
    n = len(r)
    top = tick_grid(idx)
    if top is None:
        return np.ones(n, bool)
    on = np.isin(r, top)
    have = set(idx[on].tolist())
    keep = np.ones(n, bool)
    for k in np.flatnonzero(~on):
        g = int(idx[k]) - 1
        while g % 5 not in top:  # the latest on-grid frame before this one = its tick's start
            g -= 1
        keep[k] = g not in have
    return keep


def _idr(packet, nal_len: int) -> bool:
    """Does an AVCC (length-prefixed) H.264 packet hold an IDR slice (NAL type 5)?"""
    b, p = bytes(packet), 0
    while p + nal_len < len(b):
        n = int.from_bytes(b[p : p + nal_len], "big")
        if b[p + nal_len] & 0x1F == 5:
            return True
        p += nal_len + n
    return False


def _seek_point(c, st, start: int):
    """(packets, step) to decode frames from ``start`` on without decoding the frames before it, or None.

    Frame index = pts / step (one frame per ``step`` time-base units at 25 fps). Only when the decode is the
    same as decoding from frame 0: H.264 in AVCC form, the packets before the seek keyframe have exactly the
    pts 0, step, ..., (k - 1) step (so the k-th decoded frame is frame k), and that keyframe is an IDR (no later
    frame references a frame before it). ``packets`` iterates the demuxed packets from the keyframe on."""
    from fractions import Fraction

    step = Fraction(1, 25) / st.time_base if st.time_base else None
    ex = bytes(st.codec_context.extradata or b"")
    if step is None or step.denominator != 1 or st.codec_context.name != "h264" or len(ex) < 6 or ex[0] != 1:
        return None
    step, nal_len = int(step), (ex[4] & 3) + 1
    key = None  # pts of the last IDR keyframe at or before start
    seen = []
    for pk in c.demux(st):
        if pk.pts is None:
            continue
        if pk.pts > start * step and key is not None:
            break
        if pk.is_keyframe and pk.pts <= start * step and _idr(pk, nal_len):
            key = pk.pts
        seen.append(pk.pts)
    if key is None or key % step:
        return None
    before = sorted(p for p in seen if p < key)
    if before != list(range(0, key, step)):
        return None
    c.seek(key, stream=st, backward=True, any_frame=False)

    def packets():
        first = True
        for pk in c.demux(st):
            if first and pk.pts is not None:
                if pk.pts != key or not pk.is_keyframe:
                    raise RuntimeError(f"seek landed on pts {pk.pts}, not the keyframe {key}")
                first = False
            yield pk

    return packets(), step


def decode_from(c, start: int = 0):
    """Yield ``(i, frame)`` for the frames ``i >= start`` of open container ``c``'s video stream, ``i`` the
    frame's position in the decode from frame 0 (``enumerate(c.decode(video=0))``), with the same pixels.
    Seeks to the IDR keyframe at or before ``start`` when ``_seek_point`` allows it; else decodes from 0."""
    st = c.streams.video[0]
    sp = _seek_point(c, st, start) if start > 0 else None
    if sp is None:
        if start > 0:
            c.seek(0, stream=st, backward=True, any_frame=False)
        for i, fr in enumerate(c.decode(st)):
            if i >= start:
                yield i, fr
        return
    pkts, step = sp
    nxt = None
    for pk in pkts:
        for fr in st.codec_context.decode(pk):
            i = fr.pts // step
            if fr.pts % step or (nxt is not None and i != nxt):
                raise RuntimeError(f"frame pts {fr.pts} breaks the {step}-unit frame grid")
            nxt = i + 1
            if i >= start:
                yield i, fr


FILM_RUN = 10  # consecutive panel changes 1 frame apart (0.4 s): live film, never the 10 Hz render
RENDER_LOCK = 10  # consecutive gaps alternating 2 and 3 frames (1 s of 10 Hz ticks): the render runs


def _render_lock(gap):
    """Index of the first frame of the first ``RENDER_LOCK`` consecutive inter-frame gaps that
    alternate 2 and 3 frames (``gap = diff(idx)``); None if there is none."""
    gap = np.asarray(gap)
    if len(gap) < RENDER_LOCK:
        return None
    ok = (gap == 2) | (gap == 3)
    alt = ok[:-1] & ok[1:] & (gap[:-1] + gap[1:] == 5)  # gaps k and k + 1 alternate
    run = 0
    for k, a in enumerate(alt):
        run = run + 1 if a else 0
        if run == RENDER_LOCK - 1:
            return k + 2 - RENDER_LOCK
    return None


def film_keep(idx) -> np.ndarray:
    """Boolean keep-mask over yielded frame indices: drop live film that passed the render gate
    before the render.

    On 9 of 329 replays (e.g. ST 2024-12-29 R8) 0.6-5 s of live start film in the right panel pass the
    silk render gate. The render changes every 2-3 frames (10 Hz in 25 fps video) and never on two
    consecutive frame pairs; film changes on every frame, except where a static gate shot falls under
    the de-dup tolerance. The render runs from its first ``RENDER_LOCK`` gaps alternating 2 and 3
    frames. Fires only if a run of ``FILM_RUN`` consecutive 1-frame gaps precedes that point; then
    every frame up to the end of the last run of >= 2 consecutive 1-frame gaps before it is dropped
    (the film's slow parts included). Nothing is dropped otherwise, or when the stream never shows
    the 10 Hz cadence. Filtering the de-dup stream afterwards is exact: the film ends at a render-gate
    dropout or differs from the first render frame by far more than the de-dup tolerance."""
    idx = np.asarray(idx, dtype=np.int64)
    keep = np.ones(len(idx), bool)
    gap = np.diff(idx)
    lock = _render_lock(gap)
    if lock is None:
        return keep
    run, longest, last = 0, 0, -1
    for k in range(lock):  # gaps before the render lock
        run = run + 1 if gap[k] == 1 else 0
        longest = max(longest, run)
        if run >= 2:
            last = k + 1  # frame position at the end of this run
    if longest >= FILM_RUN:
        keep[: last + 1] = False
    return keep


def iter_unique_render(
    path,
    *,
    box: PanelBox | None = None,
    start: int = 0,
    stop: int | None = None,
    keep_full: bool = False,
    style: str = "silk",
    layout_box: PanelBox | None = None,
):
    """Yield ``(frame_idx, panel[, full])`` for render-active frames whose panel CHANGED vs the
    previous yielded one (the 10 Hz de-dup). ``box`` is ``frame_box`` of the first frame if None."""
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
            active = render_active(img[:, img.shape[1] // 2 :], relaxed=seen, style=style)
            seen = seen or active
            if not active:
                prev = None
                continue
            if box is None:
                box = frame_box(img, layout_box)
            p = box.crop(img)
            if prev is not None and is_duplicate(p, prev):
                continue
            prev = p
            yield (i, p, img) if keep_full else (i, p)


# ----------------------------------------------------------------------------- frame cache
class FrameCache:
    """Unique render panels decoded once (JPEG q92 in frames.bin + offsets in frames_idx.npz) so later
    stages never re-decode the 1080p MP4. Also stores the left-panel race-clock crop per frame
    (frames_clock.npy)."""

    def __init__(self, root):

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
        return self.items(range(len(self.idx)))

    def items(self, ks):
        """Yield ``(frame_idx, panel)`` for cache positions ``ks`` in order (only those panels are decoded)."""
        for k in ks:
            yield int(self.idx[k]), self.panel(k)

    @staticmethod
    def build(path, root, clock_box=(288, 342, 800, 945), quality=92, style="silk", layout_box=None):
        import cv2

        root = Path(root)
        idx, bufs, clocks, box = [], [], [], None
        for i, p, full in iter_unique_render(
            path, keep_full=True, style=style, layout_box=layout_box
        ):
            box = box or frame_box(full, layout_box)
            ok, buf = cv2.imencode(".jpg", p, [cv2.IMWRITE_JPEG_QUALITY, quality])
            idx.append(i)
            bufs.append(buf.tobytes())
            y0, y1, x0, x1 = clock_box
            clocks.append(full[y0:y1, x0:x1].copy())
        film = film_keep(idx)
        if not film.all():  # live film leaked through the render gate before the render
            keep = np.flatnonzero(film)
            idx, bufs, clocks = [idx[k] for k in keep], [bufs[k] for k in keep], [clocks[k] for k in keep]
        keep = np.flatnonzero(grid_keep(idx))
        idx, bufs, clocks = [idx[k] for k in keep], [bufs[k] for k in keep], [clocks[k] for k in keep]
        off = [0]
        with atomic(root / "frames.bin") as tmp, open(tmp, "wb") as fb:
            for b in bufs:
                fb.write(b)
                off.append(off[-1] + len(b))
        extra = {} if style == "silk" else {"style": style}  # silk caches keep their exact layout
        with atomic(root / "frames_idx.npz") as tmp:
            np.savez(
                tmp,
                idx=np.array(idx),
                off=np.array(off),
                box=np.array([box.x0, box.x1, box.y0, box.y1]),
                **extra,
            )
        with atomic(root / "frames_clock.npy") as tmp:
            np.save(tmp, np.array(clocks))
        return len(idx)
