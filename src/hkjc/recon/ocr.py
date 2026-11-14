"""Burnt-in text on the replays, read with RECOGNITION-ONLY OCR (RapidOCR, no text detection) on
fixed-position crops: ~25x faster than det+rec and more accurate on these boxes (det split "850M"
into fragments).

AVV render (1080p): left-panel race clock, the odometer box ("850M" = leader's distance-to-go,
a round-up display: it ticks V+10 -> V as the leader REACHES V), the legend (top-8 running order,
8 fixed-pitch number boxes). RF: race clock top-right.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from . import video

AVV_CLOCK = (288, 342, 800, 945)  # y0, y1, x0, x1 (absolute, 1080p)
AVV_ODO = (286, 320, 1606, 1700)
RF_CLOCK = (55, 125, 1640, 1850)
_CLOCK = re.compile(r"(\d):(\d\d)[.:](\d\d)")
_ODO = re.compile(r"(\d{1,4})\s*M", re.IGNORECASE)


@lru_cache(maxsize=1)
def _engine():
    import os

    from rapidocr_onnxruntime import RapidOCR

    n = int(os.environ.get("HKJC_OCR_THREADS", "-1"))  # parallel races: cap threads per process
    return RapidOCR(intra_op_num_threads=n, inter_op_num_threads=1) if n > 0 else RapidOCR()


def rec_only(crop, scale: int = 3) -> tuple[str, float]:
    import cv2

    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res, _ = _engine()(up, use_det=False, use_cls=False, use_rec=True)
    return (str(res[0][0]), float(res[0][1])) if res else ("", 0.0)


def rec_batch(crops, scale: int = 3) -> list[tuple[str, float]]:
    """Recognition-only OCR of many crops in ONE batched call (the finish-camera reader OCR'd ~20
    cloth candidates per frame one by one: ~12 s/frame under load)."""
    import cv2

    if not crops:
        return []
    ups = [cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) for c in crops]
    res, _ = _engine().text_rec(ups)
    return [(str(t), float(c)) for t, c in res]


def read_clock(crop) -> float | None:
    txt, _c = rec_only(crop)
    m = _CLOCK.search(txt.replace(" ", ""))
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100 if m else None


def read_odometer(crop, max_m: int = 4000) -> float | None:
    txt, conf = rec_only(crop)
    m = _ODO.search(txt.replace(" ", ""))
    if not m or conf < 0.6:
        return None
    v = float(m.group(1))
    return v if v <= max_m else None


def read_legend(panel, n_field: int = 14) -> np.ndarray:
    """The 8 legend slots (left->right = running positions 1..8); 0 = unread."""
    lt = video.legend_top(panel)
    W = panel.shape[1]
    pitch = W / 8.8
    row = np.zeros(8, int)
    for k in range(8):
        cx = int(0.064 * W + k * pitch)
        txt, conf = rec_only(panel[max(0, lt - 4) : lt + 34, max(0, cx - 22) : cx + 16])
        dg = "".join(ch for ch in txt if ch.isdigit())
        if dg and conf > 0.6 and 1 <= int(dg) <= n_field:
            row[k] = int(dg)
    return row


def _crop(img, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def read_render_hud(path, *, n_field: int = 14, clock_every: int = 10, legend_every: int = 3):
    """Per unique render frame: [frame, clock|NaN, odometer] and [frame, legend slots x8].
    The clock is video-locked (read sparsely); the odometer is OCR'd only when its crop changes."""
    rows, leg = [], []
    prev_o, last_o, last_l = None, np.nan, np.zeros(8, int)
    for n, (i, p, full) in enumerate(video.iter_unique_render(path, keep_full=True)):
        c = np.nan
        if n % clock_every == 0:
            v = read_clock(_crop(full, AVV_CLOCK))
            c = np.nan if v is None else v
        oc = _crop(full, AVV_ODO)
        if prev_o is None or float(np.abs(oc.astype(np.int16) - prev_o).mean()) > 2.0:
            v = read_odometer(oc)
            last_o = np.nan if v is None else v
            prev_o = oc
        rows.append([i, c, last_o])
        if n % legend_every == 0:
            last_l = read_legend(p, n_field)
        leg.append([i, *last_l])
    return np.array(rows), np.array(leg)


def _legend_boxes(panel):
    """Pixels of the 8 legend number boxes (opaque; they change only on overtakes)."""
    lt = video.legend_top(panel)
    W = panel.shape[1]
    pitch = W / 8.8
    crops = [
        panel[
            max(0, lt - 4) : lt + 34,
            max(0, int(0.064 * W + k * pitch) - 22) : int(0.064 * W + k * pitch) + 16,
        ]
        for k in range(8)
    ]
    return np.concatenate([c.reshape(-1, 3) for c in crops])


def read_render_hud_cached(
    cache, *, n_field: int = 14, clock_every: int = 10, legend_every: int = 3
):
    """Same as ``read_render_hud`` from a ``video.FrameCache`` (no re-decode): the clock from the
    cached left-panel crops, the odometer + legend from the panel."""
    x0, _x1, y0, _y1 = cache.box
    oy0, oy1, ox0, ox1 = AVV_ODO
    rows, leg = [], []
    prev_o, last_o, last_l, prev_l = None, np.nan, np.zeros(8, int), None
    for n, (i, p) in enumerate(cache):
        c = np.nan
        if n % clock_every == 0:
            v = read_clock(np.asarray(cache.clock[n]))
            c = np.nan if v is None else v
        oc = p[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0]
        if prev_o is None or float(np.abs(oc.astype(np.int16) - prev_o).mean()) > 2.0:
            v = read_odometer(oc)
            last_o = np.nan if v is None else v
            prev_o = oc
        rows.append([i, c, last_o])
        if n % legend_every == 0:
            # the number boxes are opaque: re-OCR only when they change (overtakes), ~10x fewer calls
            lb = _legend_boxes(p)
            if (
                prev_l is None
                or lb.shape != prev_l.shape
                or float(np.abs(lb.astype(np.int16) - prev_l).mean()) > 8.0
            ):
                last_l = read_legend(p, n_field)
                prev_l = lb
        leg.append([i, *last_l])
    return np.array(rows), np.array(leg)


def rf_clock_offset(path, every: int = 50, fps: float = 25.0):
    """RF (and drone: same edit timeline) video_t - race clock (s), median over sparse reads."""
    import av

    d = []
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i % every:
                continue
            v = read_clock(_crop(fr.to_ndarray(format="bgr24"), RF_CLOCK))
            if v is not None:
                d.append(i / fps - v)
            if i > 3000:
                break
    return float(np.median(d)) if d else float("nan")
