"""R&D study — LEGEND-TEMPLATED SILK DETECTION for the Aerial Virtual View (AVV) render.

Hypothesis (journal 0112): each runner avatar = a dark horse body with a small BRIGHT, SATURATED,
NON-GREEN SILK PATCH (the jockey torso) on top. The silk patch is the SHADOW-INVARIANT,
RAIL-INVARIANT, race-consistent anchor (cast shadows carry no silk colour). The AVV also paints a
LEGEND STRIP along the bottom of the panel: one swatch per runner (a saddlecloth number box + the
silk colours), in running order. If we read this race's actual silk colours off the legend and then
find the matching coloured patches in the track band, we get a race-ADAPTIVE detector with no
hand-tuned colours, identity for free, and immunity to the shadow false-positives that explode the
dark-body detector.

This is a STUDY (disposable scratch). It does NOT touch any production code: it imports the existing
``detect_avatars`` only for a side-by-side count comparison and reuses ``track_band`` /
``render_active``. Everything is derived ADAPTIVELY from the frame (legend located by its structure,
swatch count read from the data, colours read in LAB with tolerance, soft matching) so it degrades
gracefully if HKJC finetunes the render style. Falls back to a generic vivid-silk detector if the
legend cannot be parsed.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_legend_detect.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "studies")

# --------------------------------------------------------------------------- test set
# A handful of frames per race (decode once, break early). R02A is the shadow stress test.
RACES = {
    "R08": ("data_cache/video/aerial_20260603_08.mp4", 11, [336, 406, 600, 1080]),
    "R03": ("data_cache/video/aerial_20260603_03.mp4", 12, [700]),
    "R02A": ("data_cache/video/aerial_20260520_02.mp4", 12, [900]),
}
DIAG = "data_cache/diag"

# --------------------------------------------------------------------------- tunables (graphics, not per-race)
# A legend swatch is a saturated box ~40-80 px wide; a track silk patch is a tiny vivid blob ~4-40 px.
LEG_SAT_MIN = 70        # a legend box pixel must be at least this saturated OR dark (number box)
SILK_AREA_MIN = 4       # px^2 — a silk patch is small
SILK_AREA_MAX = 90      # px^2 — bigger = horse body / merged / rail furniture
SILK_SAT_MIN = 90       # a silk patch is vivid
SILK_VAL_MIN = 70       # ... and bright (rules out dark body + shadow)
MATCH_TOL = 34.0        # LAB ΔE tolerance for a silk→legend colour match (tolerant, soft)


# --------------------------------------------------------------------------- data records
@dataclass
class Swatch:
    """One parsed legend swatch: its number-box label region and silk colour signature."""

    idx: int                       # left-to-right order on the strip
    cx: int                        # swatch centre x (panel px)
    x0: int
    x1: int
    sigs: list = field(default_factory=list)   # up to 2 dominant LAB silk colours (np arrays)


@dataclass
class SilkDet:
    """One silk patch detected in the track band."""

    cx: float
    cy: float
    area: int
    lab: tuple                      # mean chromatic LAB of the patch
    match: int = -1                 # matched swatch idx (-1 = unmatched / generic)
    dist: float = 1e9               # LAB ΔE to the matched swatch


# --------------------------------------------------------------------------- legend parsing
def _legend_band(panel: np.ndarray) -> tuple[int, int] | None:
    """Locate the legend strip ADAPTIVELY: the lowest sustained row-band of saturated non-turf boxes.

    No hardcoded pixel coords — we scan the lower panel for rows whose saturated-non-green count
    stays high over a few rows (the swatch boxes), and return that band. ``None`` if not found.
    """
    import cv2

    h, w = panel.shape[:2]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # turf is green+saturated; a swatch is saturated non-green OR a bright/dark number box.
    nonturf = ((ss > LEG_SAT_MIN) & ((hh < 35) | (hh > 90)) & (vv > 60)).astype(np.uint8)
    row = nonturf.sum(axis=1).astype(float)
    lo, hi = int(0.58 * h), h
    thr = max(40, 0.06 * w)
    # find the first row (top-down, lower half) that begins a sustained run of swatch boxes
    y0 = None
    for y in range(lo, hi - 4):
        if row[y] > thr and row[y + 3] > thr:
            y0 = y
            break
    if y0 is None:
        return None
    y1 = y0
    while y1 < hi - 1 and row[y1] > 0.4 * thr:
        y1 += 1
    if y1 - y0 < 12:                       # too thin to be a swatch row
        return None
    return max(0, y0 - 4), min(h, y1 + 4)


def _chroma_profile(panel: np.ndarray, ly0: int, ly1: int) -> np.ndarray:
    """Per-column mean saturation of NON-GREEN pixels in the legend band.

    Robust across backgrounds: over turf the gaps are green (zeroed by the non-green mask), over the
    dark rail/wall (R02A) the gaps are low-saturation — either way the swatches stand out as peaks.
    """
    import cv2

    band = panel[ly0:ly1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hh, ss = hsv[:, :, 0], hsv[:, :, 1]
    nongreen = (hh < 35) | (hh > 90)
    sig = (ss.astype(float) * nongreen).mean(axis=0)
    return np.convolve(sig, np.ones(7) / 7.0, mode="same")


def _segment_swatches(panel: np.ndarray, ly0: int, ly1: int) -> list[tuple[int, int]]:
    """Segment the legend band into per-entry column spans via the REGULAR pitch.

    The swatch row is a periodic comb of saturated boxes. Raw column-run thresholding is fragile
    (dark silks drop out, neighbouring silks touch, the R02A wall floods the gaps), so instead we:
      1. read the non-green saturation profile (``_chroma_profile``);
      2. recover the entry pitch by AUTOCORRELATION, constrained to a plausible swatch-width range;
      3. phase-align a comb of that pitch to maximise the per-slot peak signal over the active span;
      4. keep slots whose silk window carries real chromatic content (median-relative threshold).
    Everything is derived from the data — no hardcoded coords/count/pitch — so it absorbs style drift.
    """
    sig = _chroma_profile(panel, ly0, ly1)
    w = len(sig)
    if sig.max() <= 0:
        return []
    sig0 = sig - sig.mean()
    ac = np.correlate(sig0, sig0, mode="full")[w - 1:]
    rng = np.arange(95, min(136, w))          # plausible per-entry width (px) in the AVV render
    if len(rng) == 0:
        return []
    pitch = int(rng[np.argmax(ac[rng])])
    on = np.where(sig > 0.15 * sig.max())[0]
    if len(on) < pitch:
        return []
    x_lo, x_hi = int(on[0]), int(on[-1])
    half, q = pitch // 2, pitch // 3
    best, best_cs = -1.0, None
    for ph in range(pitch):                    # phase the comb to the swatch peaks
        cs = np.arange(x_lo + ph % pitch, x_hi + 1, pitch)
        if len(cs) == 0:
            continue
        score = float(sum(sig[max(0, c - q):c + q].max() for c in cs))
        if score > best:
            best, best_cs = score, cs
    if best_cs is None:
        return []
    peaks = [float(sig[max(0, c - q):c + q].max()) for c in best_cs]
    thr = 0.30 * float(np.median(peaks))       # a real swatch is at least ~1/3 the typical swatch
    spans = [(max(0, int(c - half)), min(w, int(c + half)))
             for c, pk in zip(best_cs, peaks) if pk > thr]
    return spans


def _top2_lab(patch_bgr: np.ndarray) -> list:
    """Top-2 dominant CHROMATIC LAB colours of a swatch (k-means on the vivid pixels).

    The mean is useless for split silks (red+white → pink), so we cluster the chromatic pixels and
    return up to 2 cluster centres, ordered by population. Falls back to the single chromatic mean.
    """
    import cv2

    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    chroma = np.abs(lab[:, 1] - 128.0) + np.abs(lab[:, 2] - 128.0)
    vivid = lab[chroma > np.percentile(chroma, 60)]
    if len(vivid) < 6:
        if len(lab):
            return [lab.mean(axis=0)]
        return []
    kk = 2 if len(vivid) >= 20 else 1
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _c, labels, ctr = cv2.kmeans(vivid, kk, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    order = np.argsort([-(labels == j).sum() for j in range(kk)])
    return [ctr[j] for j in order]


def parse_legend(panel: np.ndarray) -> list:
    """Parse the legend strip into a list of :class:`Swatch` with robust silk colour signatures.

    Returns ``[]`` if the legend cannot be located (the caller then falls back to generic detection).
    Each swatch's silk colour is read from its RIGHT portion (the number box is on the left).
    """
    band = _legend_band(panel)
    if band is None:
        return []
    ly0, ly1 = band
    spans = _segment_swatches(panel, ly0, ly1)
    out = []
    for i, (a, b) in enumerate(spans):
        # the silk is the right ~60% of the entry (number box on the left); be generous.
        sx0 = a + int(0.40 * (b - a))
        patch = panel[ly0:ly1, sx0:b]
        if patch.size == 0:
            continue
        sigs = _top2_lab(patch)
        if not sigs:
            continue
        out.append(Swatch(idx=i, cx=(a + b) // 2, x0=a, x1=b, sigs=sigs))
    return out


# --------------------------------------------------------------------------- silk detection in the track band
def _turf_hue(band_hsv: np.ndarray) -> int:
    """Modal turf hue of the band (same adaptive idea as detect_avatars)."""
    h, s, v = band_hsv[:, :, 0].astype(int), band_hsv[:, :, 1], band_hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    if not turf.any():
        return 45
    return int(np.bincount(h[turf].ravel(), minlength=180).argmax())


def detect_silks(panel: np.ndarray, ty0: int, ty1: int) -> list:
    """Detect candidate SILK PATCHES in the track band: small, vivid, non-turf blobs.

    Turf is suppressed adaptively (modal hue ± tolerance). What remains that is saturated, bright and
    small is a silk patch — NOT the dark horse body and NOT a cast shadow (both are low-value /
    low-saturation). Returns the chromatic LAB of each blob for legend matching.
    """
    import cv2

    band = panel[ty0:ty1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    thue = _turf_hue(hsv)
    d_hue = np.minimum(np.abs(h - thue), 180 - np.abs(h - thue))
    is_turf = (d_hue < 16) & (s > 35) & (v > 45)
    vivid = (s > SILK_SAT_MIN) & (v > SILK_VAL_MIN) & (~is_turf)
    mask = vivid.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    lab_img = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (SILK_AREA_MIN <= area <= SILK_AREA_MAX):
            continue
        bx, by = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw > 22 or bh > 22:                 # elongated/large = not a silk patch
            continue
        sub = lab_img[by:by + bh, bx:bx + bw].reshape(-1, 3).astype(float)
        chroma = np.abs(sub[:, 1] - 128.0) + np.abs(sub[:, 2] - 128.0)
        kk = max(3, int(0.4 * len(sub)))
        top = sub[np.argsort(chroma)[-kk:]]
        mlab = top.mean(axis=0)
        out.append(SilkDet(cx=float(cent[i, 0]), cy=float(cent[i, 1] + ty0), area=area,
                           lab=(float(mlab[0]), float(mlab[1]), float(mlab[2]))))
    return out


# --------------------------------------------------------------------------- matching
def _de(a, b) -> float:
    """LAB ΔE (Euclidean) — perceptually-robust colour distance."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def match_silks(silks: list, swatches: list) -> list:
    """Tolerant soft assignment of silk blobs to legend swatches.

    Each silk is matched to the nearest legend signature (min over both top-2 colours) within
    ``MATCH_TOL``; greedy by best distance, one swatch per silk and at most one silk per swatch
    (uniqueness gives identity for free). Unmatched silks keep ``match = -1`` (still real detections,
    just not legend-identified). Tolerant, not exact — colour drift / partial silks are absorbed.
    """
    if not swatches:
        return silks
    pairs = []
    for si, s in enumerate(silks):
        for sw in swatches:
            d = min(_de(s.lab, sig) for sig in sw.sigs)
            pairs.append((d, si, sw.idx))
    pairs.sort()
    used_s, used_w = set(), set()
    for d, si, wi in pairs:
        if d > MATCH_TOL or si in used_s or wi in used_w:
            continue
        silks[si].match = wi
        silks[si].dist = d
        used_s.add(si)
        used_w.add(wi)
    return silks


# --------------------------------------------------------------------------- overlay
def draw_overlay(panel, swatches, silks, ty0, ty1, field_n, det_av_n, name):
    """Annotate the panel: legend swatch boxes, matched/unmatched silks, and a count summary."""
    import cv2

    img = panel.copy()
    cv2.rectangle(img, (0, ty0), (img.shape[1] - 1, ty1), (60, 60, 60), 1)
    for sw in swatches:
        cv2.rectangle(img, (sw.x0, _LEG_Y[0]), (sw.x1, _LEG_Y[1]), (255, 255, 0), 1)
        cv2.putText(img, str(sw.idx), (sw.x0 + 2, _LEG_Y[0] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    matched = 0
    for s in silks:
        if s.match >= 0:
            matched += 1
            cv2.circle(img, (int(s.cx), int(s.cy)), 7, (0, 255, 0), 2)
            cv2.putText(img, str(s.match), (int(s.cx) + 6, int(s.cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.circle(img, (int(s.cx), int(s.cy)), 5, (0, 165, 255), 1)
    txt = f"{name} field={field_n} legend={len(swatches)} silks={len(silks)} matched={matched} detect_avatars={det_av_n}"
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out = f"{DIAG}/legend_{name}.png"
    cv2.imwrite(out, img)
    return out, matched


_LEG_Y = (0, 0)   # filled per-frame so the overlay can draw swatch boxes


# --------------------------------------------------------------------------- driver
def run_race(name: str) -> None:
    import av
    import cv2  # noqa: F401  (ensure cv2 import errors surface early)

    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars

    vid, field_n, frames = RACES[name]
    want = set(frames)
    mx = max(want)
    panels = {}
    for i, fr in enumerate(av.open(vid).decode(video=0)):
        if i in want:
            panels[i] = fr.to_ndarray(format="bgr24")[:, 960:].copy()
        if i >= mx:
            break

    os.makedirs(DIAG, exist_ok=True)
    print(f"\n=== {name}  field={field_n}  {vid} ===")
    for i in sorted(panels):
        panel = panels[i]
        active = render_active(panel)
        ty0, ty1 = track_band(panel)
        swatches = parse_legend(panel)
        # remember the legend y for overlay drawing
        global _LEG_Y
        lb = _legend_band(panel)
        _LEG_Y = lb if lb else (ty1, ty1 + 10)
        silks = detect_silks(panel, ty0, ty1)
        if swatches:
            silks = match_silks(silks, swatches)
            matched = sum(1 for s in silks if s.match >= 0)
            mode = "legend"
        else:
            matched = len(silks)
            mode = "generic-fallback"
        try:
            dets, _dbg = detect_avatars(panel, ty0, ty1, field=field_n)
            det_av_n = len(dets)
        except Exception as exc:                       # noqa: BLE001 — study robustness
            det_av_n = -1
            print(f"  detect_avatars failed: {exc}")
        tag = f"{name}_f{i:05d}"
        out, drawn = draw_overlay(panel, swatches, silks, ty0, ty1, field_n, det_av_n, tag)
        print(f"  frame {i:5d} active={active!s:5} mode={mode:16} "
              f"legend={len(swatches):2d} silks={len(silks):2d} matched={matched:2d} "
              f"field={field_n} detect_avatars={det_av_n:2d}  -> {out}")


def main() -> None:
    race = os.environ.get("RACE", "ALL").upper()
    targets = list(RACES) if race == "ALL" else [race]
    for r in targets:
        if r not in RACES:
            print(f"unknown RACE={r}; choose from {list(RACES)}")
            continue
        run_race(r)


if __name__ == "__main__":
    main()
