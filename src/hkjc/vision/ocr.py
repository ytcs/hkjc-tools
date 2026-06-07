"""Saddlecloth-number OCR for the side-on / finish-cam feeds (RF + patrol finish-cam).

The real side-on and finish-tracking cameras show razor-sharp saddlecloth NUMBERS on the
horses themselves. This is the dual-scale RapidOCR wrapper that reads them — distinct from
``hkjc.vision.aerial_ocr``, which reads the synthetic *Aerial Virtual View* LEGEND strip
(a rendered data overlay). Reuse ``aerial_ocr`` for the AVV legend; use this for the camera feeds.

See ``docs/CV_RECON.md``. RapidOCR + cv2 are imported lazily.
"""
from __future__ import annotations

import re

import numpy as np

DEFAULT_SCALES = (1.6, 2.4)   # dual-scale: catches both large near and small/blurred back numbers
DEFAULT_CONF = 0.40


def _rapidocr():
    # Reuse the aerial reader's cached singleton so we don't spin a second OCR engine.
    from .aerial_ocr import _rapidocr as _engine
    return _engine()


def ocr_numbers(bgr: np.ndarray, ocr=None, *, band: tuple[float, float] | None = None,
                scales=DEFAULT_SCALES, conf_min: float = DEFAULT_CONF,
                num_range: tuple[int, int] = (1, 12)) -> list[dict]:
    """OCR saddlecloth numbers on the horses in a side-on frame.

    Crops to the track ``band`` (height fractions ``(y0, y1)``, or the whole frame if None) to
    skip scenery/crowd/overlay and speed the OCR; runs RapidOCR at each scale; keeps single/double
    digit tokens above ``conf_min`` inside ``num_range``. Returns dicts in FULL-FRAME coords:
    ``{num, cx, cy, conf, h}``.
    """
    import cv2

    if ocr is None:
        ocr = _rapidocr()
    h = bgr.shape[0]
    y0 = int(h * band[0]) if band else 0
    crop = bgr[y0: int(h * band[1]), :] if band else bgr
    out: list[dict] = []
    for scale in scales:
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(big)
        if not res:
            continue
        for box, txt, conf in res:
            txt = txt.strip()
            if not (re.fullmatch(r"\d{1,2}", txt) and conf > conf_min):
                continue
            num = int(txt)
            if not num_range[0] <= num <= num_range[1]:
                continue
            cx = float(np.mean([p[0] for p in box])) / scale
            cy = float(np.mean([p[1] for p in box])) / scale + y0
            bh = (max(p[1] for p in box) - min(p[1] for p in box)) / scale
            out.append({"num": num, "cx": cx, "cy": cy, "conf": float(conf), "h": bh})
    return out


def best_per_number(dets: list[dict]) -> dict[int, dict]:
    """Keep the highest-confidence detection per number within one frame."""
    best: dict[int, dict] = {}
    for d in dets:
        if d["num"] not in best or d["conf"] > best[d["num"]]["conf"]:
            best[d["num"]] = d
    return best
