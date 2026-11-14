"""Bind world tracklets to saddlecloth numbers.

Two independent evidence sources, fused per tracklet:
  1. LEGEND ORDER — the render's legend strip lists the top-8 runners left->right in running order.
     On frames where every runner is tracked, the tracklet at longitudinal rank r votes for the
     legend's slot-r number. (A rank is only trusted when all tracklets AHEAD of it are present.)
  2. SILK COLOUR — rotation-invariant hue histograms of the detections; once some tracklets are
     confidently numbered, the rest are matched by silk distance under a no-time-overlap constraint
     (one horse cannot be two tracklets at once).
"""

from __future__ import annotations

import numpy as np


def legend_votes(members, frame_of_det, d, legend_rows, n_field, max_rank=8):
    """-> votes (n_tracklets, n_field+1): votes[t, num] = #frames tracklet t sat at the legend slot
    holding saddlecloth ``num``."""
    T = len(members)
    votes = np.zeros((T, n_field + 1))
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
        if present < n_field:  # ranks are trusted only when the whole field is tracked
            continue
        shown = [int(x) for x in slots[:max_rank]]
        for r in range(max_rank):
            num = shown[r]
            if 1 <= num <= n_field:
                votes[tracks[r], num] += 1
        # elimination: ranks beyond the legend belong to the numbers NOT shown this frame
        if all(1 <= x <= n_field for x in shown) and len(set(shown)) == max_rank:
            missing = [x for x in range(1, n_field + 1) if x not in shown]
            for r in range(max_rank, present):
                for num in missing:
                    votes[tracks[r], num] += 1.0 / len(missing)
    return votes


def time_overlap(a_frames, b_frames) -> bool:
    return not (a_frames.max() < b_frames.min() or b_frames.max() < a_frames.min())


def assign(
    members, frame_of_det, votes, hists, n_field, *, w_silk=4.0, margin=np.log(2.0), min_len=3
):
    """Number each tracklet by likelihood: log P(num | votes) + w_silk * (-TV(silk, proto[num])).
    Longest tracklets first; a number may not be used by two time-overlapping tracklets; a tracklet
    is labelled only if its best number beats the runner-up by ``margin``. Prototypes are learned
    from labelled tracklets (an official-silk-image prior was tried and REJECTED: top-1 1/8 — the
    top-down avatar vs jersey-image domain gap).
    Returns (labels, tracklet signatures, learned prototypes)."""
    T = len(members)
    lab = np.zeros(T, int)
    frames = [np.asarray(frame_of_det[np.asarray(m, int)]) for m in members]
    sig = np.array([hists[np.asarray(m, int)].mean(0) for m in members])
    nums = np.arange(1, n_field + 1)
    for _round in range(3):
        proto = {}
        for num in nums:
            u = np.where(lab == num)[0]
            if len(u):
                w = np.array([len(members[k]) for k in u], float)
                proto[num] = (sig[u] * w[:, None]).sum(0) / w.sum()
        changed = False
        for t in np.argsort([-len(m) for m in members]):
            if lab[t] or len(members[t]) < min_len:
                continue
            tot = votes[t, 1:].sum()
            sc = {}
            for num in nums:
                if any(time_overlap(frames[t], frames[u]) for u in np.where(lab == num)[0]):
                    continue
                lp = np.log((votes[t, num] + 0.5) / (tot + 0.5 * n_field))
                p = proto.get(num)
                if p is not None:
                    lp += w_silk * -(0.5 * np.abs(sig[t] - p).sum())
                sc[num] = lp
            if not sc:
                continue
            ranked = sorted(sc.items(), key=lambda kv: -kv[1])
            if len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= margin:
                lab[t] = ranked[0][0]
                changed = True
        if not changed:
            break
    return lab, sig, proto


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
