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


def common_shift(pred, S, L, sig_s, sig_l, span=6.0, step=0.1):
    """Along-track shift shared by ALL horses at one tick: the shift that lets the most predictions find
    a detection within 1 sigma (ties: smallest |shift|), refined by the median matched residual.
    MEASURED (pixels + per-tick median speed): the render updates horse positions on its own cadence,
    so some unique frames (camera motion only) carry STALE positions (whole-field speed ~0 for 1-3
    ticks, then a 23-29 m/s catch-up; ST 0419 R4 53.6 s, 77 such ticks on ST3). A constant-velocity
    prediction then runs ~1 horse spacing ahead and the Hungarian step ROTATES identities through a
    bunched field (ST4 54 s: '#7' -> '#13' -> '#12' -> '#5' a second later, silks unchanged).
    Hand-set: span 6 m, step 0.1 m. NOT ADOPTED (off by default, HKJC_TRACK_SHIFT=1): alone it cost 4
    gates on 26 races (ST3 rho 0.98 -> 0.88); a vote can alias by one horse spacing."""
    if len(pred) < 3 or len(S) < 3:
        return 0.0
    ds = S[None, :] - pred[:, :1]
    dl = np.abs(L[None, :] - pred[:, 1:2]) < sig_l
    grid = np.arange(-span, span + 1e-9, step)
    cnt = np.array([((np.abs(ds - g) < sig_s) & dl).any(1).sum() for g in grid])
    best = np.flatnonzero(cnt == cnt.max())
    g = grid[best[np.argmin(np.abs(grid[best]))]]
    m = (np.abs(ds - g) < sig_s) & dl
    r = [ds[i][m[i]][np.argmin(np.abs(ds[i][m[i]] - g))] for i in range(len(pred)) if m[i].any()]
    return float(np.median(r)) if r else 0.0


def _tv(a, b):
    return 0.5 * np.abs(a - b).sum(-1)


def calibrate_llr(tracks, F, t, frame_of, S, L, nb=40):
    """Per-race APPEARANCE log-likelihood ratio, learned from the race itself (no hand weight):
    SAME = a detection vs the mean of its own track's previous 5 valid detections, at UNAMBIGUOUS
    ticks (no other detection within 3 m along / 1.5 m lateral); DIFF = a detection vs every other
    detection in the same frame (different horses by construction). Returns (edges, cost) with
    cost = -2 log(p_same / p_diff) per TV-distance bin (add-one smoothed): chi^2 units, so it adds
    to the kinematic cost directly. MEASURED (6 races): AUC same-vs-diff 0.96 at 1 detection lag.
    Hand-set choices it still carries: the 3 m / 1.5 m unambiguity window, 40 bins, the 5-detection
    running mean (``track(nmean=5)``). NOT ADOPTED (off by default, HKJC_TRACK_APP=1): see RECON 3i
    "tracker" -- purer tracklets by eye, but cross-seed label unanimity 54.5 % -> 49.9 %."""
    valid = F.sum(1) > 0
    unamb = np.ones(len(t), bool)
    for f in np.unique(frame_of):
        j = np.flatnonzero(frame_of == f)
        if len(j) < 2:
            continue
        dd = (np.abs(S[j][:, None] - S[j][None]) < 3) & (np.abs(L[j][:, None] - L[j][None]) < 1.5)
        unamb[j] = dd.sum(1) == 1
    same, diff = [], []
    for tr in tracks:
        m = np.asarray(tr.det)
        m = m[np.argsort(t[m])]
        m = m[valid[m]]
        for q in range(5, len(m)):
            if unamb[m[q]]:
                same.append(_tv(F[m[q]], F[m[q - 5 : q]].mean(0)))
    for f in np.unique(frame_of):
        j = np.flatnonzero((frame_of == f) & valid)
        if len(j) >= 2:
            a, b = np.triu_indices(len(j), 1)
            diff.extend(_tv(F[j[a]], F[j[b]]).tolist())
    edges = np.linspace(0, 1, nb + 1)
    ps = np.histogram(same, edges)[0] + 1.0
    pd = np.histogram(diff, edges)[0] + 1.0
    return edges, -2.0 * np.log((ps / ps.sum()) / (pd / pd.sum()))


def track(
    times, S, L, frame_of, *, sig_s=0.9, sig_l=0.45, gate=3.0, max_miss=20, shift=False, F=None,
    llr=None, nmean=5,
):
    """times: per-tick times (sorted unique). S, L, frame_of: per detection (metres, tick index).
    ``shift``: estimate the per-tick common along-track shift first (``common_shift``).
    ``F``/``llr``: per-detection appearance (L1-normalised silk signature) and ``calibrate_llr``
    output -> the appearance cost vs the track's running mean is added inside the kinematic gate.
    Returns list of Track (all, including short)."""
    from scipy.optimize import linear_sum_assignment

    valid = F.sum(1) > 0 if F is not None else None

    def appearance(tr):
        v = [j for j in tr.det[-3 * nmean :] if valid[j]][-nmean:]
        return F[v].mean(0) if v else None

    order = np.argsort(frame_of, kind="stable")
    bnd = np.searchsorted(frame_of[order], np.arange(len(times) + 1))
    live, done, nid = [], [], 0
    for k, t in enumerate(times):
        ids = order[bnd[k] : bnd[k + 1]]
        if live and len(ids):
            pred = np.array([tr.predict(t)[:2] for tr in live])
            if shift:
                pred[:, 0] += common_shift(pred, S[ids], L[ids], sig_s, sig_l)
            ds = (S[ids][None, :] - pred[:, :1]) / sig_s
            dl = (L[ids][None, :] - pred[:, 1:2]) / sig_l
            C = ds**2 + dl**2
            gated = C > gate**2
            if llr is not None:
                edges, cost = llr
                for a_, tr in enumerate(live):
                    A = appearance(tr)
                    if A is None:
                        continue
                    for b_, j in enumerate(ids):
                        if valid[j] and not gated[a_, b_]:
                            k_ = min(np.searchsorted(edges, _tv(A, F[j])) - 1, len(cost) - 1)
                            C[a_, b_] += cost[max(k_, 0)]
                C = C - C[~gated].min() if (~gated).any() else C
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
