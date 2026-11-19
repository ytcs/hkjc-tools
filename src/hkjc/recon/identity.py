"""Bind world tracklets to saddlecloth numbers.

Evidence per tracklet:
  1. LEGEND ORDER — the render's legend strip lists the top-8 runners left->right in running order;
     on frames where every runner is tracked, the tracklet at longitudinal rank r votes for the
     legend's slot-r number, and ranks beyond the legend vote for the unlisted set.
  2. SILK — an optional per-tracklet silk log-likelihood ratio.
``assign_global`` fuses them in one integer program under a no-time-overlap constraint (one horse
cannot be two tracklets at once).
"""

from __future__ import annotations

import numpy as np


def legend_votes(members, frame_of_det, d, legend_rows, numbers, max_rank=8, parts=False):
    """``numbers``: the runners' saddlecloth numbers (withdrawn numbers leave gaps).
    -> votes (n_tracklets, max(numbers)+1): votes[t, num] = #frames tracklet t sat at the legend slot
    holding saddlecloth ``num``; ranks beyond the legend add 1/|missing| to every unlisted number.
    ``parts=True`` -> (votes, rank, elim, n_elim): the same evidence kept SET-VALUED for
    ``assign_global`` -- rank = slot votes only, elim[t, num] = #elimination frames whose unlisted set
    contains num, n_elim[t] = #elimination frames."""
    runners = sorted({int(x) for x in numbers})
    is_runner = set(runners)
    n_field = runners[-1]
    T = len(members)
    votes = np.zeros((T, n_field + 1))
    rank = np.zeros((T, n_field + 1))
    elim = np.zeros((T, n_field + 1))
    n_elim = np.zeros(T)
    det_track = {}
    for t, mem in enumerate(members):
        for j in mem:
            det_track[int(j)] = t
    by_frame = {}
    for j, f in enumerate(frame_of_det):
        if j in det_track:
            by_frame.setdefault(int(f), []).append(j)
    leg = {int(r[0]): r[1:] for r in legend_rows}
    lf = np.array(sorted(leg))
    for f, js in by_frame.items():
        # nearest legend read within 2 unique ticks
        if not len(lf):
            break
        g = lf[np.argmin(np.abs(lf - f))]
        if abs(g - f) > 6:
            continue
        slots = leg[g]
        js = sorted(js, key=lambda j: d[j])  # leader first (smallest distance-to-go)
        tracks = [det_track[j] for j in js]
        if len(set(tracks)) != len(tracks):
            continue
        present = len(js)
        if present < len(runners):  # ranks are trusted only when the whole field is tracked
            continue
        shown = [int(x) for x in slots[:max_rank]]
        for r in range(max_rank):
            num = shown[r]
            if num in is_runner:
                votes[tracks[r], num] += 1
                rank[tracks[r], num] += 1
        # elimination: ranks beyond the legend belong to the numbers NOT shown this frame
        if all(x in is_runner for x in shown) and len(set(shown)) == max_rank:
            missing = [x for x in runners if x not in shown]
            for r in range(max_rank, present):
                for num in missing:
                    votes[tracks[r], num] += 1.0 / len(missing)
                elim[tracks[r], missing] += 1
                n_elim[tracks[r]] += 1
    return (votes, rank, elim, n_elim) if parts else votes


def time_overlap(a_frames, b_frames) -> bool:
    return not (a_frames.max() < b_frames.min() or b_frames.max() < a_frames.min())


def assign_global(
    members,
    frame_of_det,
    votes,
    n_field,
    *,
    min_len=3,
    iters=10,
    silk_llr=None,
    elim=None,
    n_elim=None,
    extra=None,
    valid=None,
    max_rank=8,
):
    """Number the tracklets by the global maximum-likelihood assignment, with the set-valued
    elimination evidence and the silk LLR.

    Each legend vote names the tracklet's horse correctly with probability rho, otherwise one of
    the other N-1 numbers uniformly. Labelling tracklet t with ``num`` instead of leaving it
    unlabelled gains the log-likelihood ratio against a uniform label:
        G[t, num] = v_num log(rho N) + (V_t - v_num) log((1 - rho) N / (N - 1)),
    which grows with the evidence (200 votes outweigh 2). An integer program maximises sum G subject
    to: at most one number per tracklet, and a number on at most one tracklet at any frame
    (interval-graph cliques). Labels with G <= 0 stay 0. rho is re-estimated from the labelled
    tracklets (votes agreeing with the label / all their votes) until the labels stop changing.

    SET-VALUED elimination evidence (``elim``/``n_elim`` from ``legend_votes(parts=True)``; then
    ``votes`` holds the slot votes only): a frame where tracklet t ranks beyond a fully read top-8
    says "t is one of the m = N - 8 unlisted numbers" (true with probability rho_e, re-estimated
    like rho): LLR log(rho_e N / m) for each number in the set, log((1 - rho_e) N / (N - m)) for
    each listed one. (As fractional votes the uniform-confusion model would read most of each
    elimination vote as evidence against every candidate, so back-markers could never be numbered.)
    ``extra``: finish-camera votes, added as slot evidence but kept out of the rho estimate (their
    many agreeing votes would pin rho at its bound). ``valid``: runner numbers (scratched numbers are
    not in the universe)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    T = len(members)
    N = n_field
    frames = [np.asarray(frame_of_det[np.asarray(m, int)]) for m in members]
    ok = np.array([len(m) >= min_len for m in members])
    lo = np.array([f.min() for f in frames])
    hi = np.array([f.max() for f in frames])
    allowed = np.ones(N, bool)
    if valid is not None:
        allowed[:] = False
        allowed[[x - 1 for x in valid if 1 <= x <= N]] = True
    Nu = int(allowed.sum())  # size of the label universe
    V = votes[:, 1 : N + 1] * allowed
    X = np.zeros_like(V) if extra is None else extra[:, 1 : N + 1] * allowed
    tot = V.sum(1, keepdims=True)
    totx = X.sum(1, keepdims=True)
    sets = elim is not None
    if sets:
        E = elim[:, 1 : N + 1] * allowed
        ne = np.asarray(n_elim, float)[:, None]
        m_set = Nu - max_rank
        sets = m_set >= 1 and ne.sum() > 0
    rho_e = 0.9
    # interval cliques: at each tracklet start, the set of tracklets spanning it
    cliques = []
    for x in sorted(set(lo[ok].tolist())):
        c = np.where(ok & (lo <= x) & (hi >= x))[0]
        if len(c) > 1:
            cliques.append(c)
    lab = np.zeros(T, int)
    rho = 0.5
    for _ in range(iters):
        lr_in, lr_out = np.log(rho * Nu), np.log((1 - rho) * Nu / (Nu - 1))
        G = (V + X) * lr_in + (tot + totx - V - X) * lr_out
        if sets:
            G = (
                G
                + E * np.log(rho_e * Nu / m_set)
                + (ne - E) * np.log((1 - rho_e) * Nu / (Nu - m_set))
            )
        # silk_llr: appearance evidence, only where the legend has spoken at all
        if silk_llr is not None:
            spoken = tot + totx + (ne if sets else 0.0)
            G = G + np.where(spoken > 0, silk_llr[:, :N], 0.0)
        G[~ok] = 0.0
        G[:, ~allowed] = 0.0
        G = np.where(G > 0, G, 0.0)
        nv = T * N
        rows, cols = [], []
        r = 0
        for t in range(T):  # one number per tracklet
            rows += [r] * N
            cols += list(range(t * N, t * N + N))
            r += 1
        for c in cliques:  # one tracklet per number at any frame
            for num in range(N):
                rows += [r] * len(c)
                cols += [t * N + num for t in c]
                r += 1
        A = np.zeros((r, nv))
        A[rows, cols] = 1.0
        res = milp(
            -G.ravel(),
            constraints=LinearConstraint(A, -np.inf, 1.0),
            integrality=np.ones(nv),
            bounds=Bounds(0, (G.ravel() > 0).astype(float)),
        )
        x = np.round(res.x).reshape(T, N)
        new = np.where(x.max(1) > 0.5, x.argmax(1) + 1, 0)
        lt = new > 0
        if lt.any():
            agree = V[lt, new[lt] - 1].sum()
            if tot[lt].sum() > 0:
                rho = float(np.clip(agree / tot[lt].sum(), 1.5 / Nu, 0.99))
            if sets and ne[lt].sum() > 0:
                agree_e = E[lt, new[lt] - 1].sum() / ne[lt].sum()
                rho_e = float(np.clip(agree_e, (m_set + 0.5) / Nu, 0.99))
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def legend_votes_completed(
    members,
    frame_of_det,
    t_of_det,
    d,
    legend_rows,
    field,
    *,
    nmax,
    max_rank=8,
    sd_max=2.0,
    eps=0.6,
):
    """Legend votes on frames where some horses are UNDETECTED, using a fused posterior to complete
    the field (EM-style second pass). ``field(t) -> (d array of all N horses, sd array)`` from the
    first fusion. A detection's true rank = number of horses whose posterior is ahead of it by > eps;
    only frames where every horse's posterior is tight (sd < sd_max) vote."""
    det_track = {}
    for t, mem in enumerate(members):
        for j in mem:
            det_track[int(j)] = t
    by_frame = {}
    for j, f in enumerate(frame_of_det):
        if j in det_track:
            by_frame.setdefault(int(f), []).append(j)
    leg = {int(r[0]): r[1:] for r in legend_rows}
    lf = np.array(sorted(leg))
    votes = None
    for f, js in by_frame.items():
        g = lf[np.argmin(np.abs(lf - f))] if len(lf) else None
        if g is None or abs(g - f) > 6:
            continue
        dd, sd = field(float(t_of_det[js[0]]))
        if dd is None or not np.all(sd < sd_max):
            continue
        if votes is None:
            votes = np.zeros((len(members), nmax + 1))
        shown = [int(x) for x in leg[g][:max_rank]]
        for j in js:
            r = int((dd < d[j] - eps).sum())
            if r < max_rank and 1 <= shown[r] <= nmax:
                votes[det_track[j], shown[r]] += 1
    return votes if votes is not None else np.zeros((len(members), nmax + 1))


