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
# OFF by default (HKJC_SILK_W > 0 enables): per-detection silk in the E-step is CHAOTIC. Hand-set
# (79 crops, 4 races) label accuracy: none 53/79; w=20 split bins 58; w=10 51; w=40 55; collapsed
# w=20 56, w=40 57; feasible-relative 50. Gates unchanged on 26 races (205 -> 205). ST4 9 -> 15/16 and
# ST 0524 13 -> 18/19 improve, ST 0531 12 -> 7-9 and HV 0527 R9 19 -> 16-18 get worse. Next: silk at
# the TRACKLET level (aggregated, official prototypes) in identify.
SILK_W = 0.0  # cost per unit of total-variation distance (relative to the frame's best candidate)
SILK_CAP = 6.0  # bounded: arbitrates between positional candidates, never overrides a tight path


def SILK_W_ENV() -> float:  # noqa: N802
    """SILK_W, or the diagnostic override HKJC_SILK_W (<= 0 disables the appearance term: NaN)."""
    v = float(os.environ.get("HKJC_SILK_W", SILK_W))
    return v if v > 0 else float("nan")


def _collapse(a):
    """v2 silk signature -> hue (pastel + vivid) + white + dark."""
    nh = (a.shape[-1] - 2) // 2
    return np.concatenate([a[..., :nh] + a[..., nh : 2 * nh], a[..., 2 * nh :]], -1)


RANK_ATTR = 10.0  # cost cap: final-stretch detection at a listed horse's legend rank
RANK_ATTR_M = 100.0  # m before the line where the rank attraction applies
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
    s_line=None,
    silk=None,
):
    """E-step. frames/t/s/lat: per detection (s = metres run). paths: {horse: fusion.Path}.
    ``near_line`` = (s_line_from, {horse: extra cost}): detections at s >= s_line_from (the last metres
    before the line) pay the extra cost for those horses — the finish camera's NEGATIVE evidence: a
    horse it never read (while reading most of the field) is unlikely to be crossing with the field.
    ``n_field`` = runners in the race (for the negative legend evidence; default: horses with paths).
    ``silk`` = (F, proto): per-detection v2 silk signatures (rows aligned with ``t``) and {horse:
    official-silk signature}: a bounded APPEARANCE cost, relative to the best candidate in the frame.
    ``s_line`` = race distance (metres run at the line): enables the final-stretch rank attraction.
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
        # full Gaussian -2 log-likelihood relative to a tight prediction: chi^2 + log det ratio. A
        # horse whose path is VAGUE (out of view, extrapolated) must pay for it, or it steals any
        # nearby detection (HV 0527 #9, tailed off 3.4 s behind: path ended 2.8 s early).
        C = (s[js][:, None] - ps) ** 2 / vs + (lat[js][:, None] - pl) ** 2 / vl
        C = C + np.log(vs * vl / (sd_s**2 * sd_l**2))[None, :]
        if silk is not None and np.isfinite(SILK_W_ENV()):
            F, proto = silk
            Fj = F[js]
            have = [c for c, h in enumerate(ok) if h in proto]
            valid = Fj.sum(1) > 0  # an empty signature (all pixels excluded) is no evidence
            if have and valid.any():
                P = np.array([proto[ok[c]] for c in have])
                # match on hue + white + dark with pastel/vivid COLLAPSED: the render desaturates and
                # lights silks (a yellow-green silk lands in the pastel bins, the official GIF in the
                # vivid ones). MEASURED (79 hand-labelled crops): correct group in the official-silk
                # top 3: 63/79 split -> 73/79 collapsed; the split broke ST 0531 in the pipeline.
                tvd = 0.5 * np.abs(_collapse(Fj)[:, None, :] - _collapse(P)[None, :, :]).sum(-1)
                rel = tvd - tvd.min(axis=1, keepdims=True)
                # REJECTED: relative to the best POSITIONALLY PLAUSIBLE candidate (hand set 50/79)
                add = np.minimum(SILK_W_ENV() * rel, SILK_CAP)
                add[~valid] = 0.0
                C[:, have] += add
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
            sj = s[js]
            ahead = (sj[None, :] > sj[:, None] + RANK_EPS).sum(1)  # detections ahead (vectorised)
            # horses whose TIGHT posterior is ahead of each detection (completes undetected ones)
            tight = np.sqrt(vs - sd_s**2) < 2.0
            post_ahead = ((ps[None, :] > sj[:, None] + RANK_EPS) & tight[None, :]).sum(1)
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
                ub = (sj[None, :] > sj[:, None]).sum(1) + max(0, nf - len(js)) + 1
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
            # RANK ATTRACTION in the final stretch: a detection whose ahead-count equals a listed
            # horse's legend rank costs that horse at most RANK_ATTR. MEASURED (26 turf races,
            # in-sample): unread horses lose their detections near the line because their path is
            # extrapolated at constant speed from the last own detection while the horse slows --
            # their real detections sit 7-13 m BEHIND (not duplicates; 3.5-10 m from any assigned
            # one) at chi^2 ~22 > CLUTTER, with sd ~1.3 m (overconfident). Final 100 m only: ST9
            # rho 0.89 -> 1.00 on 6/6 seeds, ST4 sectional 0.476 -> 0.441, gates 202 -> 205.
            # REJECTED: whole race (ST3 straight sectional 0.174 -> 0.266); only non-tight horses
            # (sd >= 2: never fired for ST9 #5). Known bad: HV 0527 R9 #7 placed 0.88 s early.
            attr = float(os.environ.get("HKJC_RANK_ATTR", RANK_ATTR))  # diagnostic ("inf": off)
            if np.isfinite(attr) and s_line is not None:
                near = s[js] >= s_line - float(os.environ.get("HKJC_RANK_ATTR_M", RANK_ATTR_M))
                for r, num in enumerate(leg[:max_rank]):
                    if num in ok:
                        col = ok.index(num)
                        cells = (ahead == r) & near
                        C[cells, col] = np.minimum(C[cells, col], attr)
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


def release_post_finish(lab, t, s, s_line, forbid=None, *, sd_s=0.8):
    """INVARIANT: a horse that has crossed the line is not back on the course. A horse's detections
    LATER than its first own detection past the line (s > s_line) that lie more than sqrt(CLUTTER) x
    sd_s (4 sigma = 3.2 m: the E-step's own unassigned threshold) before the line are set anonymous,
    and the horse is forbidden for them (``forbid`` {det: set}, which travels with the fragment).
    MEASURED (ST 05-24 R1, Z6): two back-marker detections of an unnumbered tracklet (t 96.7 / 97.4, d
    16.6 / 7.7) went to #7 after its own detections ended at d -7.3 (t 95.84); runs_to_obs merged them
    into #7's single render run, the Gaussian smoother bent the path end 0.36 s late against its
    finish-camera anchor (sd 0.19 m), the anchor check could not demote a horse's only render run,
    and the late crossing dragged the camera clock fit (95.11 vs official 95.04) so the unanchored
    #14 out-ran #1 (official short head). Census (Z6, 6 seeds): 4-6 such detections per seed in 2
    races. A general monotone-progress rule was REJECTED before building: field-wide pose slips
    (several horses reversing together, ST4 77.5 s, ST3 27 s) are not identity errors.
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
