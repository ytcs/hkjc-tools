"""Aerial-render OCR — read the saddlecloth numbers off the rendered legend strip.

The *Aerial Virtual View* legend strip is a fixed-pitch row of silk icons, each preceded
by a small coloured box holding the runner's saddlecloth number (1-2 digits) in the current
running order, left->right. Binding a silk-colour cluster to its saddlecloth number is the
gate to stable re-id (journal 0055 §3/§5).

This module exposes ONE deployable reader, :func:`read_legend`, that takes the legend strip
crop (BGR ndarray, as produced from ``aerial_layout.find_layout``) and returns the digit
tokens with their horizontal positions, spatially ordered. The engine is selectable but
defaults to the bake-off winner (RapidOCR / PP-OCRv5 on onnxruntime — see journal 0058 and
journal 0058). Detection-capable engines run on the WHOLE upscaled strip: the render's
multi-coloured boxes (incl. dark boxes) defeat structural per-cell segmentation, so we rely on
the engine's own text localisation rather than cropping cells first.

It also exposes :func:`read_odometer` (distance-to-go box) and :func:`read_clock_splits`
(left-panel clock + published sectionals), plus :func:`_read_legend_robust`, a drift-robust
legend reader used by the aerial feed.

cv2/numpy and the OCR engine are imported lazily so importing ``hkjc.vision`` stays cheap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

_DIGITS = "0123456789"
_ODO_RE = re.compile(r"(\d{1,4})\s*M", re.IGNORECASE)
_CLOCK_RE = re.compile(r"(\d):(\d{2})\.(\d{2})")
_SPLIT_RE = re.compile(r"^\d{1,2}\.\d{2}$")


@dataclass(frozen=True)
class LegendRead:
    """One recognised legend cell: a saddlecloth number at a horizontal position."""

    number: int          # parsed saddlecloth number (1-2 digits)
    x: float             # box-centre x in strip pixel coords (for left->right ordering)
    conf: float          # engine confidence in [0, 1]
    text: str            # raw token text


def _upscale(strip, scale: int = 3):
    import cv2

    return cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


@lru_cache(maxsize=4)
def _rapidocr():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def read_legend(
    strip,
    *,
    engine: str = "rapidocr",
    scale: int = 3,
    min_conf: float = 0.3,
    valid_set: frozenset[int] | None = None,
) -> list[LegendRead]:
    """Read saddlecloth numbers off one legend strip crop (BGR HxWx3 ndarray).

    Returns :class:`LegendRead` records sorted left->right (by box-centre x in *strip*
    coordinates). ``valid_set`` (e.g. ``frozenset(range(1, 13))`` for a 12-runner field)
    drops out-of-field reads. ``engine`` is one of ``rapidocr`` | ``easyocr`` | ``tesseract``;
    the default is the bake-off winner.
    """
    up = _upscale(strip, scale)
    if engine == "rapidocr":
        toks = _read_rapidocr(up)
    elif engine == "easyocr":
        toks = _read_easyocr(up)
    elif engine == "tesseract":
        toks = _read_tesseract(up)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown OCR engine: {engine!r}")

    out: list[LegendRead] = []
    for text, xc, conf in toks:
        digits = "".join(ch for ch in text if ch in _DIGITS)
        if not digits or conf < min_conf:
            continue
        num = int(digits)
        if valid_set is not None and num not in valid_set:
            continue
        out.append(LegendRead(number=num, x=xc / scale, conf=conf, text=text))
    out.sort(key=lambda r: r.x)
    return out


def _read_rapidocr(up) -> list[tuple[str, float, float]]:
    res, _ = _rapidocr()(up)
    if not res:
        return []
    return [(txt, sum(p[0] for p in box) / 4.0, float(conf)) for box, txt, conf in res]


def read_odometer(crop, *, scale: int = 4, max_m: int = 4000) -> float | None:
    """Read the distance-to-go odometer ("420M") off the top-right map box crop.

    Returns the remaining metres as a float, or None if no "<digits>M" token is found.
    The render's odometer ticks DOWN from the race distance to ~0 at the finish; this is the
    per-frame anchor for a true (non-proxy) per-runner arc-length axis. RapidOCR reads it at
    ~1.0 confidence (journal 0058 winner); we keep only a plausible value (< ``max_m``).
    """
    up = _upscale(crop, scale)
    res, _ = _rapidocr()(up)
    if not res:
        return None
    best: float | None = None
    best_conf = 0.0
    for box, txt, conf in res:
        m = _ODO_RE.search(txt.replace(" ", ""))
        if not m:
            continue
        v = float(m.group(1))
        if v > max_m:
            continue
        if float(conf) > best_conf:
            best, best_conf = v, float(conf)
    return best


@dataclass(frozen=True)
class ClockSplits:
    """The official running clock + published cumulative sectional splits (LEFT panel)."""

    clock_s: float | None        # elapsed race time, seconds (e.g. "1:23.23" -> 83.23)
    splits: tuple[float, ...]    # published section times in order (e.g. 13.74, 22.76, ...)


def read_clock_splits(crop, *, scale: int = 3) -> ClockSplits:
    """Read the broadcast clock + published per-section split times off the left-panel crop.

    The *left* (broadcast) half of the frame carries the official race clock ("1:23.23") and,
    as each sectional boundary is passed, the cumulative published split times
    ("START-1600M 13.74", "1600M-1200M 22.76", ...). These are HKJC's own ground-truth
    sectionals embedded in the video, used for the QA / kill-criterion cross-check.
    """
    up = _upscale(crop, scale)
    res, _ = _rapidocr()(up)
    clock = None
    splits: list[float] = []
    for _box, txt, _conf in (res or []):
        t = txt.replace(" ", "")
        cm = _CLOCK_RE.search(t)
        if cm and clock is None:
            clock = int(cm.group(1)) * 60 + int(cm.group(2)) + int(cm.group(3)) / 100.0
        if _SPLIT_RE.match(t):
            splits.append(float(t))
    return ClockSplits(clock_s=clock, splits=tuple(splits))


@lru_cache(maxsize=2)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _read_easyocr(up) -> list[tuple[str, float, float]]:
    res = _easyocr_reader().readtext(up, allowlist=_DIGITS)
    return [(txt, sum(p[0] for p in box) / 4.0, float(conf)) for box, txt, conf in res]


def _read_tesseract(up) -> list[tuple[str, float, float]]:
    """Tesseract on the full strip via image_to_data (psm 11 = sparse text)."""
    import cv2
    import pytesseract
    from pytesseract import Output

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    cfg = "--psm 11 -c tessedit_char_whitelist=" + _DIGITS
    d = pytesseract.image_to_data(gray, config=cfg, output_type=Output.DICT)
    out = []
    for i, txt in enumerate(d["text"]):
        txt = txt.strip()
        if not txt:
            continue
        xc = d["left"][i] + d["width"][i] / 2.0
        conf = max(0.0, float(d["conf"][i])) / 100.0
        out.append((txt, xc, conf))
    return out


def _read_legend_robust(img, layout, valid_set, *, band_cache=None) -> list:
    """Read the legend running-order, robust to the overlay row DRIFTING vertically.

    The *Aerial Virtual View* legend strip (silk icons + saddlecloth-number boxes in running
    order) sits at a fixed screen position MID-RACE but the number-box row drifts DOWN as the
    render tilts toward the run-in (journal 0069: the prior pipeline's single ``find_layout``
    band caught it mid-race but read EMPTY from ~t=72s on, dropping the finish). Strategy
    (cheap-first to keep OCR cost down): try the LAST-GOOD band centre (``band_cache``) first,
    then the layout's primary band; only if both come up short do we sweep a few candidate
    bands and keep the read with the most valid in-field numbers. The good band centre is
    cached for the next frame (the legend drifts slowly), so the steady state is ONE OCR call
    per frame, not six. RapidOCR reads the digits at ~1.0 confidence once the band is right.
    Returns the de-duplicated left->right running order (saddlecloth numbers).
    """
    h = img.shape[0]
    lx0, lx1 = layout.panel_x0, layout.panel_x1

    def read_at(yc: float) -> list:
        y0 = max(0, int(yc * h) - 30)
        y1 = min(h - 1, int(yc * h) + 30)
        return _dedup_order(
            [r.number for r in read_legend(img[y0:y1, lx0:lx1], valid_set=valid_set)]
        )

    # 1) cached good band (steady state: this hits, one OCR call)
    if band_cache and band_cache.get("yc") is not None:
        cached = read_at(band_cache["yc"])
        if len(cached) >= 6:
            return cached
    # 2) the layout's primary band
    primary = _dedup_order(
        [r.number for r in read_legend(
            img[layout.legend_y0:layout.legend_y1, lx0:lx1], valid_set=valid_set)]
    )
    best, best_yc = primary, None
    if len(primary) >= 6:
        best_yc = (layout.legend_y0 + layout.legend_y1) / 2.0 / h
    else:
        # 3) sweep candidate bands (mid-race ~0.62h through the run-in ~0.72h)
        for yc in (0.60, 0.63, 0.66, 0.69, 0.72):
            order = read_at(yc)
            if len(order) > len(best):
                best, best_yc = order, yc
    if best_yc is not None and band_cache is not None:
        band_cache["yc"] = best_yc
    return best


def _is_green_render(img, layout) -> bool:
    """True if the track ROI is GREEN turf (newer render) vs BROWN (older render).

    The two Aerial Virtual View styles differ in track colour, which changes the detector
    tuning and the px->m scale source (single drawn rail on green vs both rails on brown).
    """
    import cv2
    import numpy as np

    x0, x1 = layout.panel_x0, layout.panel_x1
    y0, y1 = layout.track_y0, layout.track_y1
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return False
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    turf = hue[(sat > 40) & (val > 40)]
    if turf.size < 100:
        return False
    return bool(28 <= int(np.median(turf)) <= 70)


def _best_align(cv: list, pub: list) -> tuple:
    """Best contiguous alignment of CV section times against published splits.

    Slides the shorter list over the longer and returns (mae, max_err, offset) at the offset
    minimising MAE. Handles the published list leading with a short opening run-up section
    that the CV 400 m odometer grid does not resolve.
    """
    import numpy as np

    if not cv or not pub:
        return float("nan"), float("nan"), 0
    best = None
    for off in range(0, max(1, len(pub) - len(cv) + 1)):
        k = min(len(cv), len(pub) - off)
        if k <= 0:
            continue
        errs = [abs(cv[j] - pub[off + j]) for j in range(k)]
        mae = float(np.mean(errs))
        if best is None or mae < best[0]:
            best = (mae, float(np.max(errs)), off)
    return best if best else (float("nan"), float("nan"), 0)


def _dedup_order(order: list[int]) -> list[int]:
    """Drop consecutive/duplicate OCR repeats from a legend running-order read.

    RapidOCR occasionally double-reads a cell (e.g. ``[9, 8, 8, 6, ...]``); the legend lists
    each runner ONCE in running order, so we keep the first occurrence of each number while
    preserving left->right order.
    """
    seen: set[int] = set()
    out: list[int] = []
    for n in order:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out
