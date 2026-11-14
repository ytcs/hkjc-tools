"""Detection-level association by EM against the continuous fused paths.

Frames are not independent: every horse is one continuous trajectory. Instead of fixing identity on
tracker fragments (a fragment carries one label for life; a swap or break poisons it), every
DETECTION is re-assigned each iteration:

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

import os

import numpy as np

CLUTTER = 16.0  # chi^2 cost of leaving a detection unassigned (4 sigma)
RANK_EPS = 0.6  # m: a detection is "clearly ahead" if > RANK_EPS m closer to the finish
RANK_COST = 50.0
RANK_SOFT = 9.0  # 3 sigma: confident-rank mismatch with the legend
FLOOR_EPS = 8.0  # m: 'ahead of every listed horse' margin (4/8/16 m equivalent; 0.6 m failed)


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
):
    """E-step. frames/t/s/lat: per detection (s = metres run). paths: {horse: fusion.Path}.
    ``near_line`` = (s_line_from, {horse: extra cost}): detections at s >= s_line_from (the last metres
    before the line) pay the extra cost for those horses — the finish camera's NEGATIVE evidence: a
    horse it never read (while reading most of the field) is unlikely to be crossing with the field.
    ``n_field`` = runners in the race (for the negative legend evidence; default: horses with paths).
    ``forbid_out`` (dict, optional) receives {detection index: set of horses} excluded by the HARD
    legend costs, so later stages (fusion's anonymous association) cannot hand them back.
    Returns per-detection horse number (0 = clutter/unassigned) and per-detection chi^2."""
    from scipy.optimize import linear_sum_assignment

    hs = sorted(paths)
    at = _legend_at(legend_rows)
    lab = np.zeros(len(t), int)
    chi = np.full(len(t), np.inf)
    order = np.argsort(frames, kind="stable")
    fr_sorted = frames[order]
    bounds = np.r_[0, np.where(np.diff(fr_sorted))[0] + 1, len(order)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        js = order[a:b]
        tj = t[js[0]]
        ok = [h for h in hs if paths[h].t[0] <= tj <= paths[h].t[-1]]
        if not ok:
            continue
        ps = np.array([np.interp(tj, paths[h].t, paths[h].s) for h in ok])
        pl = np.array([np.interp(tj, paths[h].t, paths[h].lat) for h in ok])
        vs = np.array([np.interp(tj, paths[h].t, paths[h].sd_s) for h in ok]) ** 2 + sd_s**2
        vl = np.array([np.interp(tj, paths[h].t, paths[h].sd_l) for h in ok]) ** 2 + sd_l**2
        # full Gaussian -2 log-likelihood relative to a tight prediction: chi^2 + log det ratio. A
        # horse whose path is VAGUE (out of view, extrapolated) must pay for it, or it steals any
        # nearby detection (HV 0527 #9, tailed off 3.4 s behind: path ended 2.8 s early).
        C = (s[js][:, None] - ps) ** 2 / vs + (lat[js][:, None] - pl) ** 2 / vl
        C = C + np.log(vs * vl / (sd_s**2 * sd_l**2))[None, :]
        if near_line is not None:
            for s_from, pen in near_line if isinstance(near_line, list) else [near_line]:
                nl = s[js] >= s_from
                if nl.any():
                    for col, h in enumerate(ok):
                        if h in pen:
                            C[nl, col] += pen[h]
                            if forbid_out is not None and pen[h] >= CLUTTER:  # hard (blind horses)
                                for i in np.where(nl)[0]:
                                    forbid_out.setdefault(int(js[i]), set()).add(int(h))
        leg = at(int(frames[js[0]]))
        if leg is not None:
            ahead = np.array([(s[js] > s[j] + RANK_EPS).sum() for j in js])  # detections ahead
            # horses whose TIGHT posterior is ahead of each detection (completes undetected ones)
            tight = np.sqrt(vs - sd_s**2) < 2.0
            post_ahead = np.array([((ps > s[j] + RANK_EPS) & tight).sum() for j in js])
            for r, num in enumerate(leg[:max_rank]):
                if num not in ok:
                    continue
                col = ok.index(num)
                C[ahead > r, col] += RANK_COST  # >= r+1 detections ahead: cannot be rank r+1
                if forbid_out is not None:
                    for i in np.where(ahead > r)[0]:
                        forbid_out.setdefault(int(js[i]), set()).add(int(num))
                # TWO-SIDED where the rank is confident (detected ahead == posterior ahead): a
                # detection of known rank q must not take the legend's rank-r horse, r != q. The
                # one-sided form never ATTRACTED, so tail labels drifted (HV 0527 '#9' in 5th).
                conf = (ahead == post_ahead) & (ahead != r)
                C[conf, col] += RANK_SOFT
            # NEGATIVE legend evidence: a fully read top-max_rank legend that does NOT list horse h
            # says h ranks below max_rank. A detection with at most ``ub`` horses possibly ahead
            # (every detection ahead + every undetected horse + 1 for a false positive) is surely
            # inside the top max_rank, so it cannot be h. HV 0520 #8 (outside the legend until 88 s)
            # held a detection ranked 2-3 of 8-9 detected for ~50 s -> 2 s early at 1200/800 m.
            listed = {int(x) for x in leg[:max_rank] if x > 0}
            if len(listed) == max_rank:
                nf = n_field or len(ok)
                ub = np.array([(s[js] > s[j]).sum() for j in js]) + max(0, nf - len(js)) + 1
                inside = ub < max_rank
                # ... and h runs behind EVERY listed horse: a detection ahead of the rearmost TIGHT
                # posterior among the listed horses cannot be h (a vague listed horse could only be
                # further back, so the bound stays valid). The count bound alone left #8 on stray
                # detections in frames with only ~7 of 12 horses detected. Margin FLOOR_EPS, not
                # RANK_EPS: rail-projected s and the legend's ranking disagree by a few metres (HV
                # 0603 R3: #10/#5, closing from 9th-10th, sat 2-5 m "ahead" of the legend's 8th and
                # lost every detection with 0.6 m; cause not established — lateral offset, rail
                # curvature or legend lag). Ensemble: 4, 8 and 16 m equivalent; rule off -> HV 0520
                # sectional 0.147 -> 0.187 s.
                lt = [c for c, h in enumerate(ok) if h in listed and tight[c]]
                feps = float(os.environ.get("HKJC_FLOOR_EPS", FLOOR_EPS))  # diagnostic ("inf": off)
                if lt and np.isfinite(feps):
                    inside |= s[js] > ps[lt].min() + feps
                if inside.any():
                    for col, h in enumerate(ok):
                        if h not in listed:
                            C[inside, col] += RANK_COST
                    if forbid_out is not None:
                        out = {int(h) for h in ok if h not in listed}
                        for i in np.where(inside)[0]:
                            forbid_out.setdefault(int(js[i]), set()).update(out)
        n, m = C.shape
        M = np.full((n, m + n), 1e6)
        M[:, :m] = C
        M[np.arange(n), m + np.arange(n)] = CLUTTER
        r, c = linear_sum_assignment(M)
        for i, k in zip(r, c):
            if k < m and C[i, k] < CLUTTER:
                lab[js[i]] = ok[k]
                chi[js[i]] = C[i, k]
    return lab, chi


def fragment_forbid(forbid, idx, frac=None):
    """Horses excluded for a tracklet fragment: those the HARD legend costs excluded on at least
    ``frac`` of its detections (a conservative default against one frame's noise, e.g. a clutter
    detection ahead inflating the rank count). MEASURED: indistinguishable from the union over its
    detections on a 6-race perturbation ensemble (HV 0603 R3's lost #10/#5 were the floor margin)."""
    from collections import Counter

    frac = float(os.environ.get("HKJC_FORBID_FRAC", 0.5)) if frac is None else frac  # diagnostic
    c = Counter(h for i in idx for h in forbid.get(int(i), ()))
    return frozenset(h for h, k in c.items() if k >= frac * len(idx))


def runs_to_obs(lab, t, s, lat, *, feed="render", sd_s=0.8, sd_l=0.35, gap=1.0, tid0=50_000):
    """Per-horse observation runs (split at gaps > ``gap`` s) as fusion.Obs."""
    from .fusion import Obs

    out = []
    tid = tid0
    for h in np.unique(lab[lab > 0]):
        k = np.where(lab == h)[0]
        k = k[np.argsort(t[k])]
        cut = np.r_[0, np.where(np.diff(t[k]) > gap)[0] + 1, len(k)]
        for a, b in zip(cut[:-1], cut[1:]):
            if b - a < 3:
                continue
            kk = k[a:b]
            out.append(Obs(t[kk], s[kk], lat[kk], sd_s, sd_l, feed, horse=int(h), tid=tid))
            tid += 1
    return out
