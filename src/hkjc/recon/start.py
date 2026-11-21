"""Gate opening in the start film: the race-clock zero when the overlay clock lags it.

The aerial feed opens on the live start film. The closed stalls' doors are a band of dark-green pixels that drops
sharply when they swing open. ``onset`` finds that step on an NR x NC grid of the 1/4-resolution frames, within
[clock0 - 100, clock0 + 2] frames (the doors cannot open after the clock starts): the row strip with the largest
relative drop whose drop persists (doors stay open) and is shared across the strip's width (not a person
crossing). ``lag_frames`` is the overlay clock's lag behind the opening, net of the detector's convention lag
(measured where the clock matches the official start); lags within the measured spread count as zero."""

from __future__ import annotations

import numpy as np

NR, NC = 8, 6
CONVENTION = {"ST": 1.4, "HV": 2.55}  # frames: clock zero - detected onset where the clock is synced
TRIGGER = 4.0  # frames: larger lags re-anchor the clock
SEARCH = 100  # frames before the clock zero searched for the opening


def door_grid(path, nmax: int) -> np.ndarray:
    """(T, NR, NC) dark-green (closed-door) fraction per grid cell of the first ``nmax`` frames."""
    import av
    import cv2

    out = []
    with av.open(str(path)) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, fr in enumerate(c.decode(video=0)):
            if i >= nmax:
                break
            hsv = cv2.cvtColor(fr.to_ndarray(format="bgr24")[::4, ::4], cv2.COLOR_BGR2HSV)
            m = (
                (hsv[..., 0] >= 35) & (hsv[..., 0] <= 95) & (hsv[..., 1] > 60) & (hsv[..., 2] < 110)
            ).astype(np.float32)
            out.append(
                [[b.mean() for b in np.array_split(r, NC, axis=1)] for r in np.array_split(m, NR, axis=0)]
            )
    return np.array(out, float).reshape(-1, NR, NC)


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
    G = door_grid(path, int(clock0) + 50)
    f = onset(G, int(clock0) - SEARCH, int(clock0) + 2)
    if f is None:
        return dict(clock0=clock0, onset=np.nan, lag_frames=0.0, status="none")
    net = clock0 - f - CONVENTION.get(venue, 2.0)
    late = net > TRIGGER
    return dict(clock0=clock0, onset=f, lag_frames=net if late else 0.0, status="late" if late else "synced")
