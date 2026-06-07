"""RF / race-film feed — per-horse LONGITUDINAL order + finishing MARGINS.

The real side-on broadcast feed delivers the two things the lateral feeds lack: per-horse
finishing MARGINS (photo-finish geometry near the line — within-frame horizontal pixel
separation / body-length scale, pan-robust) and a per-horse running-ORDER backbone over the
whole race (the rendered bottom legend strip, read by NAME->number bridge).

Saddlecloth OCR comes from ``vision.ocr``; the legend-strip number boxes are too small to OCR
so the english NAMES are read instead and bridged to numbers via a caller-supplied LUT. cv2
lazy. See ``docs/CV_RECON.md`` §2 (feeds), §3 (clock).
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from ...core.track import THOROUGHBRED_LENGTH_M
from ..ocr import best_per_number, ocr_numbers

# OCR runs on the TRACK BAND crop only (height fractions) — skips scenery/crowd/overlay, faster.
BAND = (0.30, 0.82)


# --------------------------------------------------------------------------- clock
def vid_to_race_t(vid_t: float, offset: float) -> float:
    """RF video time -> shared ``race_t`` (``race_t = vid_t - offset``, the on-screen clock anchor)."""
    return vid_t - offset


def race_t_to_arclen(race_t: float, section_times: np.ndarray, section_bounds: np.ndarray) -> float:
    """Shared ``race_t`` -> arclen (m run) via the official cumulative-section curve."""
    return float(np.interp(race_t, section_times, section_bounds))


# --------------------------------------------------------------------------- detection / OCR
def detect_horses(fr: np.ndarray, band: tuple[float, float] = BAND) -> list[dict]:
    """Detect horse+jockey blobs (dark body below a saturated silk) on the turf, side-on.

    Returns per blob ``{cx, cy, x0, y0, x1, y1, bw, bh, area}``; ``bw`` is the body-length pixel
    scale used by the margin readout. Restricted to the track ``band`` (excludes scenery/overlay).
    """
    import cv2

    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (hh > 30) & (hh < 95) & (ss > 30) & (vv > 40)
    dark = vv < 80
    silk = (ss > 110) & (vv > 70) & (~turf)
    fg = ((dark | silk) & (~turf)).astype(np.uint8)
    bandmask = np.zeros_like(fg)
    bandmask[int(h * band[0]): int(h * band[1]), :] = 1
    fg = fg & bandmask
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _lbl, st, ct = cv2.connectedComponentsWithStats(fg, 8)
    out = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if area < 1200 or bw < 35 or bh < 25 or bw > w * 0.5:
            continue
        bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        out.append({"cx": float(ct[i][0]), "cy": float(ct[i][1]),
                    "x0": bx, "y0": by, "x1": bx + bw, "y1": by + bh,
                    "bw": bw, "bh": bh, "area": area})
    return out


def in_legend_strip(cy: float, frame_h: int) -> bool:
    """OCR hits in the bottom legend strip are the DATA OVERLAY, not the horse — exclude."""
    return cy > frame_h * 0.82


def ocr_saddlecloths(fr: np.ndarray, ocr=None, field_size: int = 12) -> list[dict]:
    """Saddlecloth-number OCR on the track band, dropping the legend-strip overlay reads."""
    h = fr.shape[0]
    dets = ocr_numbers(fr, ocr, band=BAND, num_range=(1, field_size))
    return [d for d in dets if not in_legend_strip(d["cy"], h)]


# --------------------------------------------------------------------------- finish margins
def measure_margins(frames, ocr=None, *, field_size: int = 12,
                    winner: int | None = None) -> list[tuple]:
    """Per finish-window frame, the numbered horses' x-positions + a body-length scale.

    ``frames`` is an iterable of ``(vid_t, BGR)`` over the clean side-on finish window. Returns
    ``[(vid_t, {num: cx}, body_px), ...]`` for frames with >=2 numbered horses, the within-frame
    relative separation being the pan-robust margin signal.
    """
    recs = []
    for vid_t, fr in frames:
        h = fr.shape[0]
        horses = detect_horses(fr)
        if len(horses) < 2:
            continue
        dets = [d for d in ocr_saddlecloths(fr, ocr, field_size) if not in_legend_strip(d["cy"], h)]
        if len(dets) < 2:
            continue
        best = best_per_number(dets)
        clean = [hb for hb in horses if 1.3 <= hb["bw"] / max(hb["bh"], 1) <= 3.2] or horses
        body_px = float(np.median([hb["bw"] for hb in clean]))
        numcx = {num: float(d["cx"]) for num, d in best.items()}
        if len(numcx) >= 2:
            recs.append((float(vid_t), numcx, body_px))
    _ = winner
    return recs


def margins_from_frames(frame_recs, winner: int) -> dict[int, dict]:
    """Aggregate per-frame within-frame gaps (relative to ``winner``) into a length margin.

    Horses run right->left at the line, so a horse behind the winner has larger ``cx``; its gap
    in lengths is ``(cx - cx_winner) / body_px``. The camera loses back-markers before the gap
    fully opens, so the OPEN-GAP (75th-pct) per-horse value best proxies the line margin. Returns
    ``{num: {margin_len, sd_len, n}}``; requires n>=2 (drop single-frame OCR flukes).
    """
    per = defaultdict(list)
    for _vid_t, numcx, body_px in frame_recs:
        if winner not in numcx or body_px <= 0:
            continue
        xw = numcx[winner]
        for num, x in numcx.items():
            per[num].append((x - xw) / body_px)
    res = {}
    for num, vals in per.items():
        v = np.array(vals)
        if num != winner and len(v) < 2:
            continue
        est = 0.0 if num == winner else float(np.percentile(v, 75))
        res[num] = {
            "margin_len": max(0.0, est),
            "sd_len": float(np.subtract(*np.percentile(v, [75, 25])) / 1.35)
            if len(v) > 2 else float("nan"),
            "n": int(len(v)),
        }
    return res


def margin_to_arclen(margin_len: float, leader_arclen: float,
                     length_m: float = THOROUGHBRED_LENGTH_M) -> float:
    """A length margin behind the leader -> absolute arclen (``leader_arclen - margin*length``)."""
    return leader_arclen - margin_len * length_m


# --------------------------------------------------------------------------- legend order backbone
def _norm_name(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def match_name(txt: str, name2num: dict[str, int], min_jaccard: float = 0.6) -> int | None:
    """Map an OCR'd legend english NAME to a runner number (exact, then best bigram Jaccard).

    ``name2num`` keys are normalised (uppercased, non-letters stripped). Robust to RapidOCR
    dropping/merging a letter.
    """
    q = _norm_name(txt)
    if len(q) < 5:
        return None
    if q in name2num:
        return name2num[q]

    def bigrams(s):
        return {s[i:i + 2] for i in range(len(s) - 1)}

    qb = bigrams(q)
    best, bj = None, 0.0
    for name, num in name2num.items():
        j = len(qb & bigrams(name)) / max(1, len(qb | bigrams(name)))
        if j > bj:
            bj, best = j, num
    return best if bj >= min_jaccard else None


def read_legend_order(frames, ocr, name2num: dict[str, int],
                      conf_min: float = 0.40) -> list[tuple]:
    """Read the bottom LEGEND STRIP running order over the race (lead->back left->right).

    ``frames`` is an iterable of ``(vid_t, BGR)``. Each entry's saddlecloth box is too small to
    OCR; the english NAME OCRs cleanly and bridges to a number via ``name2num``. Returns
    ``[(vid_t, [numbers lead->back]), ...]`` for frames with >=3 resolved names.
    """
    import cv2

    seq = []
    for vid_t, fr in frames:
        h, w = fr.shape[:2]
        strip = fr[int(h * 0.855): int(h * 0.93), : int(w * 0.92)]
        big = cv2.resize(strip, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(big)
        if not res:
            continue
        hits = []
        for box, txt, conf in res:
            num = match_name(txt, name2num) if conf > conf_min else None
            if num is not None:
                hits.append((float(np.mean([p[0] for p in box])), num))
        hits.sort()
        order = []
        for _, num in hits:
            if num not in order:
                order.append(num)
        if len(order) >= 3:
            seq.append((float(vid_t), order))
    return seq
