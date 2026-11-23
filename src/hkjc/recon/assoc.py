"""Detection-level association by EM against the continuous fused paths.

Every horse is one continuous trajectory, so every DETECTION (not tracker fragment) is re-assigned
each iteration:

  E-step  per frame, one joint assignment of all detections to all horses (Hungarian; at most one
          detection per horse, a "clutter" option per detection) with cost =
            chi^2 of the detection vs the horse's posterior at that instant (posterior sd + obs sd)
          + a one-sided RUNNING-ORDER constraint from the legend: if the legend shows horse h at
            rank r, a detection with >= r detections clearly ahead of it in the same frame cannot be
            h (undetected horses can only push true ranks DOWN, never up)
  M-step  re-fuse the paths from the assigned detections (``fusion``), plus the other feeds.

The fragments from the tracker are only the initialisation.
"""

from __future__ import annotations


import numpy as np

CLUTTER = 16.0  # chi^2 cost of leaving a detection unassigned (4 sigma)
RANK_EPS = 0.6  # m: a detection is "clearly ahead" if > RANK_EPS m closer to the finish
RANK_COST = 50.0
RANK_SOFT = 9.0  # 3 sigma: confident-rank mismatch with the legend
def _collapse(a):
    """Silk signature -> hue (pastel + vivid) + white + dark."""
    nh = (a.shape[-1] - 2) // 2
    return np.concatenate([a[..., :nh] + a[..., nh : 2 * nh], a[..., 2 * nh :]], -1)


RANK_ATTR = 10.0  # cost cap: final-stretch detection at a listed horse's legend rank
RANK_ATTR_M = 100.0  # m before the line where the rank attraction applies
FLOOR_EPS = 8.0  # m: 'ahead of every listed horse' margin


def _legend_at(legend_rows):
    leg = {int(r[0]): r[1:] for r in legend_rows}
    lf = np.array(sorted(leg))

    def at(f):
        if not len(lf):
            return None
        g = lf[np.argmin(np.abs(lf - f))]
        return leg[g] if abs(g - f) <= 6 else None

    return at


def assign_frames(
    frames,
    t,
    s,
    lat,
    paths,
    legend_rows,
    *,
    sd_s=0.8,
    sd_l=0.35,
    max_rank=8,
    near_line=None,
    n_field=None,
    forbid_out=None,
    s_line=None,
    margin_out=None,
):
    """E-step. frames/t/s/lat: per detection (s = metres run). paths: {horse: fusion.Path}.
    ``near_line`` = (s_line_from, {horse: extra cost}): detections at s >= s_line_from (the last metres
    before the line) pay the extra cost for those horses — the finish camera's NEGATIVE evidence: a
    horse it never read (while reading most of the field) is unlikely to be crossing with the field.
    ``n_field`` = runners in the race (for the negative legend evidence; default: horses with paths).
    ``s_line`` = race distance (metres run at the line): enables the final-stretch rank attraction.
    ``forbid_out`` (dict, optional) receives {detection index: set of horses} excluded by the HARD
    legend costs, so later stages (fusion's anonymous association) cannot hand them back.
    ``margin_out`` (float array over detections, optional) receives, per labelled detection, the frame's optimal cost
    with that detection forbidden from its horse minus the optimal cost (-2 log-likelihood units); read-only.
    Returns per-detection horse number (0 = clutter/unassigned) and per-detection chi^2."""
    from scipy.optimize import linear_sum_assignment

    hs = sorted(paths)
    at = _legend_at(legend_rows)
    lab = np.zeros(len(t), int)
    chi = np.full(len(t), np.inf)
    order = np.argsort(frames, kind="stable")
    fr_sorted = frames[order]
    bounds = np.r_[0, np.where(np.diff(fr_sorted))[0] + 1, len(order)]
    # path state at every frame's time, interpolated per horse in one vectorised call (np.interp is
    # pointwise, so the values equal the per-frame scalar calls exactly)
    tf = t[order[bounds[:-1]]]
    if hs:
        inr = np.array([(paths[h].t[0] <= tf) & (tf <= paths[h].t[-1]) for h in hs]).T
        PS = np.array([np.interp(tf, paths[h].t, paths[h].s) for h in hs]).T
        PL = np.array([np.interp(tf, paths[h].t, paths[h].lat) for h in hs]).T
        PVS = np.array([np.interp(tf, paths[h].t, paths[h].sd_s) for h in hs]).T
        PVL = np.array([np.interp(tf, paths[h].t, paths[h].sd_l) for h in hs]).T
    for fi, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        js = order[a:b]
        if not hs or not inr[fi].any():
            continue
        cols = np.flatnonzero(inr[fi])
        ok = [hs[c] for c in cols]
        ps = PS[fi, cols]
        pl = PL[fi, cols]
        vs = PVS[fi, cols] ** 2 + sd_s**2
        vl = PVL[fi, cols] ** 2 + sd_l**2
        # full Gaussian -2 log-likelihood: chi^2 + log det ratio, so a vague (extrapolated) path pays
        # for its width instead of stealing nearby detections
        C = (s[js][:, None] - ps) ** 2 / vs + (lat[js][:, None] - pl) ** 2 / vl
        C = C + np.log(vs * vl / (sd_s**2 * sd_l**2))[None, :]
        if near_line is not None:
            for s_from, pen in near_line if isinstance(near_line, list) else [near_line]:
                nl = s[js] >= s_from
                if nl.any():
                    for col, h in enumerate(ok):
                        if h in pen:
                            hit = nl
                            C[hit, col] += pen[h]
                            if forbid_out is not None and pen[h] >= CLUTTER:  # hard (blind horses)
                                for i in np.where(hit)[0]:
                                    forbid_out.setdefault(int(js[i]), set()).add(int(h))
        leg = at(int(frames[js[0]]))
        if leg is not None:
            sj = s[js]
            ahead = (sj[None, :] > sj[:, None] + RANK_EPS).sum(1)  # detections ahead (vectorised)
            # HARD exclusions (RANK_COST + forbid_out) need a certain ahead-count: ahead by more than
            # 4 sigma on a difference of two detections (sqrt(CLUTTER * 2) * sd_s = 4.5 m)
            ahead_h = (sj[None, :] > sj[:, None] + np.sqrt(CLUTTER * 2.0) * sd_s).sum(1)
            # horses whose TIGHT posterior is ahead of each detection (completes undetected ones)
            tight = np.sqrt(vs - sd_s**2) < 2.0
            post_ahead = ((ps[None, :] > sj[:, None] + RANK_EPS) & tight[None, :]).sum(1)
            for r, num in enumerate(leg[:max_rank]):
                if num not in ok:
                    continue
                col = ok.index(num)
                C[ahead_h > r, col] += RANK_COST  # >= r+1 detections ahead: cannot be rank r+1
                if forbid_out is not None:
                    for i in np.where(ahead_h > r)[0]:
                        forbid_out.setdefault(int(js[i]), set()).add(int(num))
                # TWO-SIDED where the rank is confident (detected ahead == posterior ahead): a
                # detection of known rank q must not take the legend's rank-r horse, r != q
                conf = (ahead == post_ahead) & (ahead != r)
                C[conf, col] += RANK_SOFT
            # NEGATIVE legend evidence: a fully read top-max_rank legend that does NOT list horse h
            # says h ranks below max_rank. A detection with at most ``ub`` horses possibly ahead
            # (every detection ahead + every undetected horse + 1 for a false positive) is surely
            # inside the top max_rank, so it cannot be h.
            listed = {int(x) for x in leg[:max_rank] if x > 0}
            if len(listed) == max_rank:
                nf = n_field or len(ok)
                ahead_c = (sj[None, :] > sj[:, None]).sum(1)
                ub = ahead_c + max(0, nf - len(js)) + 1
                inside = ub < max_rank
                # ... and h runs behind EVERY listed horse: a detection ahead of the rearmost TIGHT
                # posterior among the listed horses cannot be h (a vague listed horse could only be
                # further back, so the bound stays valid). Margin FLOOR_EPS: rail-projected s and the
                # legend's ranking disagree by a few metres.
                lt = [c for c, h in enumerate(ok) if h in listed and tight[c]]
                if lt:
                    inside |= s[js] > ps[lt].min() + FLOOR_EPS
                if inside.any():
                    for col, h in enumerate(ok):
                        if h not in listed:
                            C[inside, col] += RANK_COST
                    if forbid_out is not None:
                        out = {int(h) for h in ok if h not in listed}
                        for i in np.where(inside)[0]:
                            forbid_out.setdefault(int(js[i]), set()).update(out)
            # RANK ATTRACTION in the final stretch: a detection whose ahead-count equals a listed
            # horse's legend rank costs that horse at most RANK_ATTR: an unread horse's path is
            # extrapolated at constant speed while it slows, so its real detections sit 7-13 m behind
            # at chi^2 > CLUTTER.
            if s_line is not None:
                near = s[js] >= s_line - RANK_ATTR_M
                for r, num in enumerate(leg[:max_rank]):
                    if num in ok:
                        col = ok.index(num)
                        cells = (ahead == r) & near
                        C[cells, col] = np.minimum(C[cells, col], RANK_ATTR)
        n, m = C.shape
        M = np.full((n, m + n), 1e6)
        M[:, :m] = C
        M[np.arange(n), m + np.arange(n)] = CLUTTER
        r, c = linear_sum_assignment(M)
        for i, k in zip(r, c):
            if k < m and C[i, k] < CLUTTER:
                lab[js[i]] = ok[k]
                chi[js[i]] = C[i, k]
        if margin_out is not None:
            best = M[r, c].sum()
            for i, k in zip(r, c):
                if k < m and C[i, k] < CLUTTER:
                    M2 = M.copy()
                    M2[i, k] = 1e6
                    r2, c2 = linear_sum_assignment(M2)
                    margin_out[js[i]] = M2[r2, c2].sum() - best
    return lab, chi


def fragment_forbid(forbid, idx, frac=None):
    """Horses excluded for a tracklet fragment: those the HARD legend costs excluded on at least
    ``frac`` of its detections (guards against one frame's noise, e.g. a clutter detection ahead
    inflating the rank count)."""
    from collections import Counter

    frac = 0.5 if frac is None else frac
    c = Counter(h for i in idx for h in forbid.get(int(i), ()))
    return frozenset(h for h, k in c.items() if k >= frac * len(idx))


def release_post_finish(lab, t, s, s_line, forbid=None, *, sd_s=0.8):
    """INVARIANT: a horse that has crossed the line is not back on the course. A horse's detections
    LATER than its first own detection past the line (s > s_line) that lie more than sqrt(CLUTTER) x
    sd_s (4 sigma = 3.2 m: the E-step's own unassigned threshold) before the line are set anonymous,
    and the horse is forbidden for them (``forbid`` {det: set}, which travels with the fragment).
    -> (labels, number released)."""
    lab = lab.copy()
    margin = np.sqrt(CLUTTER) * sd_s
    n = 0
    for h in np.unique(lab[lab > 0]):
        k = np.flatnonzero(lab == h)
        past = k[s[k] > s_line]
        if not len(past):
            continue
        tf = t[past].min()
        bad = k[(t[k] > tf) & (s[k] < s_line - margin)]
        if len(bad):
            lab[bad] = 0
            n += len(bad)
            if forbid is not None:
                for j in bad:
                    forbid.setdefault(int(j), set()).add(int(h))
    return lab, n


RUN_MIN = 3  # detections; shorter runs are dropped (singletons near the line drag the camera clock)


def runs_to_obs(lab, t, s, lat, *, feed="render", sd_s=0.8, sd_l=0.35, gap=1.0, tid0=50_000):
    """Per-horse observation runs (split at gaps > ``gap`` s) as fusion.Obs; runs shorter than
    RUN_MIN detections are dropped."""
    from .fusion import Obs

    out = []
    tid = tid0
    for h in np.unique(lab[lab > 0]):
        k = np.where(lab == h)[0]
        k = k[np.argsort(t[k])]
        cut = np.r_[0, np.where(np.diff(t[k]) > gap)[0] + 1, len(k)]
        for a, b in zip(cut[:-1], cut[1:]):
            if b - a < RUN_MIN:
                continue
            kk = k[a:b]
            out.append(Obs(t[kk], s[kk], lat[kk], sd_s, sd_l, feed, horse=int(h), tid=tid))
            tid += 1
    return out
