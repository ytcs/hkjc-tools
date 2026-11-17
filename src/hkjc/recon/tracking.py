"""Multi-horse tracking in the WORLD track frame (s, lat) in metres.

The camera is already factored out (poses), so association is pure kinematics: a horse moves
~1.7 m per 10 Hz tick along s at a slowly-varying speed and drifts laterally slowly, while any two
horses are >= ~1 m apart. Per tick: constant-velocity prediction + Hungarian assignment inside a
physical gate; tracks coast through detection gaps; tracklets are then STITCHED across gaps when the
kinematic extrapolation agrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    tid: int
    t: list = field(default_factory=list)  # times (s)
    s: list = field(default_factory=list)
    lat: list = field(default_factory=list)
    det: list = field(default_factory=list)  # detection row index
    miss: int = 0

    def predict(self, t):
        if len(self.t) >= 3:
            k = min(len(self.t), 8)
            tt, ss = np.array(self.t[-k:]), np.array(self.s[-k:])
            v = np.polyfit(tt - tt[-1], ss, 1)[0]
        else:
            v = (
                16.0
                if len(self.t) < 2
                else (self.s[-1] - self.s[-2]) / max(self.t[-1] - self.t[-2], 1e-3)
            )
        return self.s[-1] + v * (t - self.t[-1]), self.lat[-1], v


def track(times, S, L, frame_of, *, sig_s=0.9, sig_l=0.45, gate=3.0, max_miss=20):
    """times: per-tick times (sorted unique). S, L, frame_of: per detection (metres, tick index).
    Returns list of Track (all, including short)."""
    from scipy.optimize import linear_sum_assignment

    order = np.argsort(frame_of, kind="stable")
    bnd = np.searchsorted(frame_of[order], np.arange(len(times) + 1))
    live, done, nid = [], [], 0
    for k, t in enumerate(times):
        ids = order[bnd[k] : bnd[k + 1]]
        if live and len(ids):
            pred = np.array([tr.predict(t)[:2] for tr in live])
            ds = (S[ids][None, :] - pred[:, :1]) / sig_s
            dl = (L[ids][None, :] - pred[:, 1:2]) / sig_l
            C = ds**2 + dl**2
            gated = C > gate**2
            C[gated] = 1e6
            r, c = linear_sum_assignment(C)
            ok = C[r, c] < 1e6
            used = set()
            for a, b in zip(r[ok], c[ok]):
                tr = live[a]
                j = ids[b]
                tr.t.append(t)
                tr.s.append(S[j])
                tr.lat.append(L[j])
                tr.det.append(j)
                tr.miss = 0
                used.add(b)
            matched = set(r[ok])
            for a, tr in enumerate(live):
                if a not in matched:
                    tr.miss += 1
            new = [ids[b] for b in range(len(ids)) if b not in used]
        else:
            for tr in live:
                tr.miss += 1
            new = list(ids)
        for j in new:
            live.append(Track(nid, [t], [S[j]], [L[j]], [j]))
            nid += 1
        still = []
        for tr in live:
            (done if tr.miss > max_miss else still).append(tr)
        live = still
    return done + live


def stitch(tracks, *, max_gap=6.0, sig_s=1.5, sig_l=0.6, gate=3.0, min_len=3):
    """Greedy-optimal linking of tracklet ends to later starts by kinematic extrapolation
    (one successor / one predecessor). Returns merged tracks."""
    from scipy.optimize import linear_sum_assignment

    T = [tr for tr in tracks if len(tr.t) >= min_len]
    while True:
        n = len(T)
        C = np.full((n, n), 1e6)
        for a, A in enumerate(T):
            ta = A.t[-1]
            for b, B in enumerate(T):
                tb = B.t[0]
                if a == b or not (0 < tb - ta <= max_gap):
                    continue
                sp, lp, v = A.predict(tb)
                es = sig_s * (1 + 0.5 * (tb - ta))
                el = sig_l * (1 + 0.3 * (tb - ta))
                c = ((B.s[0] - sp) / es) ** 2 + ((B.lat[0] - lp) / el) ** 2
                if c < gate**2:
                    C[a, b] = c
        r, c = linear_sum_assignment(C)
        pairs = [(a, b) for a, b in zip(r, c) if C[a, b] < 1e6]
        if not pairs:
            return T
        # apply links without cycles: merge b into a (process by time)
        nxt = dict(pairs)
        has_pred = set(nxt.values())
        out = []
        for a in range(n):
            if a in has_pred:
                continue
            cur = T[a]
            m = Track(cur.tid, list(cur.t), list(cur.s), list(cur.lat), list(cur.det))
            while a in nxt:
                a = nxt[a]
                B = T[a]
                m.t += B.t
                m.s += B.s
                m.lat += B.lat
                m.det += B.det
            out.append(m)
        if len(out) == len(T):
            return out
        T = out


SPLIT_TV = 0.30  # max-split silk distance above which a tracklet holds two horses (see below)


def split_by_appearance(members, t, F, *, thr=SPLIT_TV, min_n=40):
    """Split tracker tracklets at APPEARANCE change points (identity switches the motion tracker
    cannot see, e.g. in the render's camera swings). Binary segmentation on the tracklet's per-
    detection collapsed v2 silk signatures ``F`` (rows = detections): the split maximising the
    total-variation distance between the left and right mean signatures (>= ``min_n`` silk-bearing
    detections per side) is taken while that distance exceeds ``thr``; recursive.
    MEASURED (26 turf races, legend votes evaluated on each side of the max split): tracklets whose two
    sides both vote for their label (pure, 85): max-split distance <= 0.28 (p95 0.24) -> thr 0.30
    splits none of them; it catches the switches between distinct silks (ST4 0419: trk 2 0.56, trk 6
    0.40 = #3 until ~50 s, then #9 at the line; ST3 #4/#10 -> #3 0.70; ST7 #14 -> #5 0.68, #12 -> #14
    0.68) but not those between similar silks (4 of 14 vote-confirmed switches). ADOPTED 2026-11-15
    (26 races x 6 seeds vs Z2, rule declared before reading): non-lateral gates 200.0 = 200.0,
    hand-truth majority 58 -> 62, unanimity 69.8 -> 74.1 %. HKJC_SPLIT_APP=0 disables.
    Known limit: the silk tie-break has no "none of these" option, so an out-of-view horse's number
    can still land on a lookalike (ST4 #3). Returns new members."""
    from .assoc import _collapse

    out = []
    stack = [np.asarray(m, int) for m in members]
    while stack:
        m = stack.pop()
        m = m[np.argsort(t[m], kind="stable")]
        f = F[m]
        ok = f.sum(1) > 0
        if ok.sum() < 2 * min_n:
            out.append(m)
            continue
        idx = np.flatnonzero(ok)
        X = _collapse(f[ok] / f[ok].sum(1, keepdims=True))
        c = np.cumsum(X, 0)
        n = len(X)
        best, bi = 0.0, None
        for i in range(min_n, n - min_n + 1):
            D = 0.5 * np.abs(c[i - 1] / i - (c[-1] - c[i - 1]) / (n - i)).sum()
            if D > best:
                best, bi = D, i
        if bi is None or best <= thr:
            out.append(m)
            continue
        cut = idx[bi]  # first silk-bearing detection of the right part
        stack += [m[:cut], m[cut:]]
    return out
