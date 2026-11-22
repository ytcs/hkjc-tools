"""Gate opening in the start film: the race-clock zero when the overlay clock lags it.

The aerial feed opens on the live start film. The closed stalls' doors are a band of dark pixels that drops sharply
when they swing open. Their colour depends on the light: dark green in sun, dark teal in shade (evening Happy
Valley), so ``DOOR_HUES`` holds two hue bands, tried in order; the teal band is consulted only when the green band
finds no opening, and it stops short of the turf's green, which the green band admits where the turf is shaded.
``onset`` finds that step on an NR x NC grid of the 1/4-resolution frames, within [clock0 - 100, clock0 + 2]
frames (the doors cannot open after the clock starts): the row strip with the largest relative drop whose drop
persists (doors stay open) and is shared across the strip's width (not a person crossing). ``lag_frames`` is the overlay clock's lag behind the opening, net of the detector's convention lag
(measured where the clock matches the official start); lags within the measured spread count as zero."""

from __future__ import annotations

import numpy as np

NR, NC = 8, 6
CONVENTION = {"ST": 1.4, "HV": 2.55}  # frames: clock zero - detected onset where the clock is synced
TRIGGER = 4.0  # frames: larger lags re-anchor the clock
SEARCH = 100  # frames before the clock zero searched for the opening
# OpenCV hue bands (0-180) of closed doors, tried in order: green (sunlit), then teal (shaded; turf is <= 55, the
# blue banner and night sky >= 105)
DOOR_HUES = ((35, 95), (60, 105))


def door_mask(hsv: np.ndarray, hue: tuple[int, int]) -> np.ndarray:
    """Dark, saturated pixels of an HSV image within the OpenCV hue band ``hue``."""
    return (hsv[..., 0] >= hue[0]) & (hsv[..., 0] <= hue[1]) & (hsv[..., 1] > 60) & (hsv[..., 2] < 110)


def door_grid(path, nmax: int) -> np.ndarray:
    """(len(DOOR_HUES), T, NR, NC) closed-door fraction per grid cell of the first ``nmax`` frames, per hue band."""
    import av
    import cv2

    out = []
    with av.open(str(path)) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, fr in enumerate(c.decode(video=0)):
            if i >= nmax:
                break
            hsv = cv2.cvtColor(fr.to_ndarray(format="bgr24")[::4, ::4], cv2.COLOR_BGR2HSV)
            row = []
            for hue in DOOR_HUES:
                m = door_mask(hsv, hue).astype(np.float32)
                row.append(
                    [[b.mean() for b in np.array_split(r, NC, axis=1)] for r in np.array_split(m, NR, axis=0)]
                )
            out.append(row)
    return np.array(out, float).reshape(-1, len(DOOR_HUES), NR, NC).transpose(1, 0, 2, 3)


def onset(G, lo, hi, *, w=8, rel_min=0.25, abs_min=0.01, persist=0.7, plen=40, share=0.6):
    """Sub-frame index of the doors opening in grid series ``G`` within [lo, hi), or None."""
    D = G.mean(2)
    best = (0.0, None, None)
    for s in range(G.shape[1]):
        d = D[:, s]
        for i in range(max(w, lo), min(len(d) - w - 2, hi)):
            pre, post = np.median(d[i - w : i]), np.median(d[i + 2 : i + 2 + w])
            if pre <= 0 or pre - post < abs_min:
                continue
            if pre - np.median(d[i + 2 : i + 2 + plen]) < persist * (pre - post):
                continue
            pb = np.median(G[i - w : i, s, :], axis=0)
            qb = np.median(G[i + 2 : i + 2 + w, s, :], axis=0)
            door = pb > 0.05
            if door.sum() < 2 or ((pb - qb)[door] / pb[door] >= 0.15).mean() < share:
                continue
            rel = (pre - post) / pre
            if rel > best[0]:
                best = (rel, i, s)
    rel, bi, s = best
    if bi is None or rel < rel_min:
        return None
    d = D[:, s]
    pre, post = np.median(d[bi - w : bi]), np.median(d[bi + 2 : bi + 2 + w])
    thr = pre - 0.25 * (pre - post)
    j = bi - w
    while j < len(d) - 1 and d[j] >= thr:
        j += 1
    return float(j - 1 + (d[j - 1] - thr) / max(d[j - 1] - d[j], 1e-9))


def lag(path, clock0: float, venue: str) -> dict:
    """{clock0, onset, lag_frames, status}: status 'late' (lag applied), 'synced' or 'none' (no opening found)."""
    f = None
    for G in door_grid(path, int(clock0) + 50):  # hue bands in order; the first that finds an opening
        f = onset(G, int(clock0) - SEARCH, int(clock0) + 2)
        if f is not None:
            break
    if f is None:
        return dict(clock0=clock0, onset=np.nan, lag_frames=0.0, status="none")
    net = clock0 - f - CONVENTION.get(venue, 2.0)
    late = net > TRIGGER
    return dict(clock0=clock0, onset=f, lag_frames=net if late else 0.0, status="late" if late else "synced")
