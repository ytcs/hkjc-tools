"""Silk-colour signature — the per-runner identity descriptor shared by all feeds.

A jockey's silks are an ABSOLUTE colour fingerprint: a saturation-weighted hue
histogram separates runners robustly where a single mean/median colour collapses
overlapping silks. Distance is a circular EMD over the hue ring (silks are absolute
hues, so it is NOT rotation-invariant — that would map blue onto red).

Pure numpy: imports with no heavy CV deps. Pixels are supplied already silk-masked.
"""
from __future__ import annotations

import numpy as np

NBINS = 18  # hue bins, 10deg each over the OpenCV 0..180 hue range


def silk_signature(hsv_pixels: np.ndarray) -> np.ndarray | None:
    """Saturation-weighted hue histogram of an ``(N, 3)`` HSV silk-pixel array (L1-normed)."""
    if hsv_pixels is None or hsv_pixels.size < 24:
        return None
    p = np.asarray(hsv_pixels).reshape(-1, 3)
    return _hue_hist(p[:, 0], p[:, 1], p[:, 2])


def _hue_hist(hh: np.ndarray, ss: np.ndarray, vv: np.ndarray) -> np.ndarray | None:
    """Saturation-weighted hue histogram over silk pixels (channels already split)."""
    if hh.size < 8:
        return None
    sel = (ss > 80) & (vv > 60)
    if sel.sum() < 8:
        return None
    bins = (hh[sel].astype(int) * NBINS // 180) % NBINS
    hist = np.bincount(bins, weights=ss[sel].astype(float), minlength=NBINS).astype(float)
    s = hist.sum()
    return hist / s if s > 0 else None


def silk_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Circular EMD between two normalised hue histograms (the wrap-around minimum)."""
    ca, cb = np.cumsum(a), np.cumsum(b)
    fwd = float(np.abs(ca - cb).sum())
    bwd = float(np.abs((1 - ca) - (1 - cb)).sum())
    return min(fwd, bwd)


def silk_distinctness(lut: dict[int, np.ndarray]) -> dict[int, float]:
    """Per-runner silk DISTINCTNESS in [0, 1]: how far this silk's nearest neighbour is.

    A degenerate silk (close to many others) scores ~0; an unambiguous one ~1. Used to
    confidence-weight the own-silk identity vote. Derived from the LUT's own pairwise spread.
    """
    nums = sorted(lut)
    if len(nums) < 2:
        return {n: 1.0 for n in nums}
    nn = {}
    for n in nums:
        ds = [silk_distance(lut[n], lut[m]) for m in nums if m != n and lut[m] is not None]
        nn[n] = min(ds) if ds else 1.0
    vals = np.array(list(nn.values()))
    lo, hi = float(np.percentile(vals, 10)), float(np.percentile(vals, 90))
    rng = max(1e-9, hi - lo)
    return {n: float(np.clip((nn[n] - lo) / rng, 0.0, 1.0)) for n in nums}
