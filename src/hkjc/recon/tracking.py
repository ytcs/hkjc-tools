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
    MEASURED (pixels + per-tick median speed): on some ticks the whole field's world speed is ~0 for
    1-3 ticks, then 23-29 m/s (ST 0419 R4 53.6 s). Cause (later, pixels): NOT stale render positions
    (pixel boxes move 4-6 px on those ticks) but POSE STALLS -- the sequential step match locks onto
    unmasked horse avatars (static in the image: the camera follows them), so the camera "stops"
    (``register.world_mask`` misses brown-bodied / pastel-silk avatars). A constant-velocity
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


SAME_ALONG = 0.8  # m: two DIFFERENT horses' centres cannot be this close along the track ...
SAME_LAT = 0.5  # m: ... and laterally at once (side-by-side horses are ~1 m apart; shadow paths)


def resolve_seams(members, frame_of, d, lat):
    """Tracker HANDOVER duplicates: where two tracklets share frames and EVERY shared frame has their
    detections physically coincident (< SAME_ALONG along and < SAME_LAT lateral: one horse, two
    boxes), they are one horse. The shorter tracklet's detections inside the longer one's time span
    join the longer (the coincident duplicates are dropped); the rest of it stays its own tracklet.
    MEASURED (26 turf races, tag V): 54 tracklets with a clear legend-vote winner stayed unnumbered
    because ``identity.assign`` forbids a number on two time-overlapping tracklets; ~12 of them
    overlapped the number's holder by 1-15 frames at a seam with median |ds| 0.4-0.9 m, |dlat|
    <= 0.2 m (HV 06-03 R8 #7: 162 votes, 1 frame; ST 06-07 R6 #14: 136 votes, 11 frames). The others
    are real conflicts (metres apart) and are left alone. Returns new members (list of int arrays).
    NOT ADOPTED (opt-in HKJC_SEAMS=1; rule declared before reading): 26 races x 6 seeds, all gates
    206.00 -> 206.17 per seed (ST 05-24 +1), but hand-truth majority 57 -> 48 of 79 (ST 05-24 13 -> 8,
    ST 05-31 13 -> 9). Observed on ST 05-24 seed 0: the merge moved 9 detections and identify's
    labels were the same bar one small tracklet (16 -> 15 numbered), so the loss looks like the EM
    amplifying a small input change rather than the merge itself.
    Retest only once the EM is stable to such perturbations."""
    mem = [np.asarray(m, int) for m in members]
    order = np.argsort([-len(m) for m in mem], kind="stable")
    for ia in order:  # longest first absorbs
        for ib in order:
            a, b = mem[ia], mem[ib]
            if ia == ib or len(a) == 0 or len(b) == 0 or len(b) > len(a):
                continue
            a0, a1 = frame_of[a].min(), frame_of[a].max()
            inside = b[(frame_of[b] >= a0) & (frame_of[b] <= a1)]
            if len(inside) == 0:
                continue
            at = {int(f): j for f, j in zip(frame_of[a], a)}
            shared = [(j, at[int(frame_of[j])]) for j in inside if int(frame_of[j]) in at]
            if not shared:
                continue
            if not all(
                abs(d[j] - d[k]) < SAME_ALONG and abs(lat[j] - lat[k]) < SAME_LAT for j, k in shared
            ):
                continue
            dup = {j for j, _k in shared}
            rest = np.setdiff1d(b, inside)
            if len(rest) and frame_of[rest].min() < a0 and frame_of[rest].max() > a1:
                inside = b  # b straddles a: its pieces would still overlap a -> all of it joins a
            move = np.array([j for j in inside if j not in dup], int)
            mem[ia] = np.array(sorted(np.r_[a, move], key=lambda j: frame_of[j]), int)
            mem[ib] = np.setdiff1d(b, inside)
            mem[ib] = mem[ib][np.argsort(frame_of[mem[ib]], kind="stable")]
    return [m for m in mem if len(m)]


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
