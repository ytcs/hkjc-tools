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


SPLIT_TV = 0.30  # max-split silk distance above which a tracklet holds two horses (pure ones <= 0.28)


NUMBER_WIN = 10  # confident number reads in the change window (~1 s at 10 Hz)
NUMBER_SHARE = 0.8  # share of the window a new number needs to count as a switch


def _same_cloth(a: int, b: int) -> bool:
    """A read of ``b`` on cloth ``a`` is a dropped digit ('1' on '11', '2' on '12'), not another horse."""
    return a == b or str(b) in str(a) or str(a) in str(b)


def split_by_number(members, t, num):
    """Split tracklets where the number read on the disc CHANGES and stays changed (circle render: the
    tracker swaps identities where discs touch). ``num`` per detection row: the confident read, 0 if none.
    Walking a tracklet's reads in time order, a switch is >= NUMBER_SHARE of the last NUMBER_WIN reads
    naming one other number that is not a dropped digit of the current one; the split falls just before
    the first of those reads. Returns new members (order kept, then appended pieces)."""
    out = []
    for m in members:
        m = np.asarray(m)
        if len(m) == 0:
            out.append(m)
            continue
        m = m[np.argsort(t[m], kind="stable")]
        rd = [(k, int(num[j])) for k, j in enumerate(m) if num[j] > 0]
        cuts, cur, k = [], None, 0
        while k < len(rd):
            if cur is None:
                win = [n for _, n in rd[k : k + NUMBER_WIN]]
                cur = max(set(win), key=win.count)
                k += 1
                continue
            win = rd[max(0, k - NUMBER_WIN + 1) : k + 1]
            if len(win) == NUMBER_WIN:
                c = {}
                for _, n in win:
                    if not _same_cloth(cur, n):
                        c[n] = c.get(n, 0) + 1
                if c:
                    n, cnt = max(c.items(), key=lambda x: x[1])
                    if cnt >= NUMBER_SHARE * NUMBER_WIN:
                        first = next(i for i, x in win if x == n)
                        if not cuts or first > cuts[-1]:
                            cuts.append(first)
                        cur = n
            k += 1
        edges = [0, *cuts, len(m)]
        out += [m[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return out


def split_by_appearance(members, t, F, *, thr=SPLIT_TV, min_n=40):
    """Split tracker tracklets at APPEARANCE change points (identity switches the motion tracker
    cannot see, e.g. in the render's camera swings). Binary segmentation on the tracklet's per-
    detection collapsed v2 silk signatures ``F`` (rows = detections): the split maximising the
    total-variation distance between the left and right mean signatures (>= ``min_n`` silk-bearing
    detections per side) is taken while that distance exceeds ``thr``; recursive. Catches switches
    between distinct silks, not between similar ones. Returns new members."""
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
