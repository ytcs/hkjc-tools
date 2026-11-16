"""Consistency AUDIT: label-free invariants of a reconstruction (no official data).

Every failure found by eye in this project broke a one-line physical or evidential invariant. The
audit computes them per race and returns violations with locations, so failures are found by
measurement instead of by inspection, and a fix is verified on the whole vector:

RAIL (after ``rails``; BLOCKING -- an ensemble must not start on a rail that fails these)
  rail_off_map     rail stations inside the observed span where the mosaic has no pixels (a rail
                   cannot leave the image; ST 05-24's smoothed bulge ran ~150 px off the map)
  markers          painted-marker grid found (>= 3 markers)
  marker_spacing   max |gap between consecutive matched markers / (50 m x ppm) - 1| (arclength must
                   match the paint; the ST 05-24 bulge added ~14 m between the 100 m and 50 m marks)
  kinks            4 m chords turning > 8 deg (a rail is smooth; switch diagonals and jagged snaps)
FIELD (after ``observe``)
  field_gap        100 m segments where the innermost horses (5th pct lateral) run > 4 m outside the
                   rail (the rail followed the wrong line: fixed rail, apron edge, band edge)
  lateral_slide    20 m bins where the whole field's median lateral velocity exceeds 0.8 m/s (a field
                   cannot slide sideways together: rail switches, scaffold folds)
IDENTITY / FUSION (after ``fuse``)
  legend_negative  horse-seconds where a horse absent from a fully read top-8 legend runs > 8 m
                   ahead of the rearmost tight listed horse (ST4 #3 on a lookalike tracklet)
  legend_rank      horse-seconds where a listed horse's path rank differs from its legend rank by > 2
  ghost_crossing   horses whose path crosses the line TIGHTLY (sd < 1.5 m) although their own
                   detections end > 3 s earlier and the finish camera did not read them (HV 05-27
                   R3 #9 crossing on an anonymous fragment)
  finish_order     clear finish-camera order pairs (gap > 150 px) the fused crossings invert
Thresholds are first measured values; they are to be replaced by null-derived ones.
"""

from __future__ import annotations

import numpy as np

KINK_DEG = 8.0
KINK_CHORD = 40  # stations (2 px each)
FIELD_GAP_M = 4.0
SLIDE_MS = 0.8
SD_TIGHT = 1.5
FLOOR_M = 8.0  # = assoc.FLOOR_EPS
# contiguous off-map stations that block: accepted rails reach 44 (straight-course rails along the
# coverage edge, ST3); the ST 05-24 smoother bulge was 699 -> 100 (~10 m, > 2x the clean maximum)
OFF_MAP_RUN = 100


def _kinks(R, ch=KINK_CHORD, thr=KINK_DEG):
    A = R[ch::ch] - R[:-ch:ch]
    h = np.degrees(np.arctan2(A[:, 1], A[:, 0]))
    dh = np.abs((np.diff(h) + 180) % 360 - 180)
    return [int((i + 1) * ch) for i in np.flatnonzero(dh > thr)]


def rail_checks(run) -> dict:
    """Rail-level invariants (rail.npz, metric.npz, map_cnt.npy, map_meta.npz)."""
    z = np.load(run.p("rail.npz"))
    m = np.load(run.p("metric.npz"))
    R = z["rail"]
    out = {}
    st = z["st_of_frame"]
    lo, hi = int(st.min()), int(st.max())
    if run.p("map_cnt.npy").exists() and run.p("map_meta.npz").exists():
        cnt = np.load(run.p("map_cnt.npy"))
        mm = np.load(run.p("map_meta.npz"))
        P = ((R[lo : hi + 1] - mm["bounds"]) * float(mm["sc"])).astype(int)
        inside = (P[:, 0] >= 0) & (P[:, 1] >= 0) & (P[:, 0] < cnt.shape[1]) & (P[:, 1] < cnt.shape[0])
        on = np.zeros(len(P), bool)
        on[inside] = cnt[P[inside, 1], P[inside, 0]] > 0.5
        off = np.flatnonzero(~on) + lo
        out["rail_off_map"] = int(len(off))
        out["rail_off_map_at"] = _runs(off)
    ks, ps, ppm = m["ks"], m["ps"], float(m["ppm"])
    out["markers"] = int(len(ks))
    if len(ks) >= 2:
        o = np.argsort(ks)
        dk, dp = np.diff(ks[o]), np.abs(np.diff(ps[o]))
        out["marker_spacing"] = float(np.max(np.abs(dp / (dk * 50.0 * ppm) - 1.0)))
    else:
        out["marker_spacing"] = float("nan")
    out["kinks_at"] = _kinks(R)
    out["kinks"] = len(out["kinks_at"])
    return out


def _runs(idx, min_len=1):
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(idx, cut) if len(r) >= min_len]


def field_checks(run) -> dict:
    """Field-level invariants from tracks.npz (lateral is relative to the extracted rail)."""
    z = np.load(run.p("tracks.npz"), allow_pickle=True)
    d, lat, t = z["d"], z["lat"], z["t"]
    gap = []
    for a in np.arange(0, d.max(), 100.0):
        q = (d >= a) & (d < a + 100)
        if q.sum() >= 50 and np.percentile(lat[q], 5) > FIELD_GAP_M:
            gap.append(int(a))
    S = []
    for m in z["members"]:
        m = np.asarray(m, int)
        if len(m) < 10:
            continue
        m = m[np.argsort(t[m])]
        tt, ll, dd = t[m], lat[m], d[m]
        dt = tt[5:] - tt[:-5]
        ok = (dt > 0.3) & (dt < 0.8)
        S += list(zip(((dd[5:] + dd[:-5]) / 2)[ok], ((ll[5:] - ll[:-5]) / np.maximum(dt, 1e-9))[ok]))
    slide = []
    if S:
        S = np.array(S)
        for a in np.arange(-20, d.max() + 20, 20.0):
            q = (S[:, 0] >= a) & (S[:, 0] < a + 20)
            if q.sum() >= 15 and abs(np.median(S[q, 1])) > SLIDE_MS:
                slide.append((int(a), round(float(np.median(S[q, 1])), 2)))
    return {"field_gap": len(gap), "field_gap_at": gap, "lateral_slide": len(slide), "slide_at": slide}


def identity_checks(run) -> dict:
    """Legend / finish-camera reprojection residuals and ghost crossings from the fused paths."""
    import polars as pl

    from . import finishcam

    P = pl.read_parquet(run.p("paths_video.parquet"))
    paths = {}
    for h in P["horse_no"].unique().to_list():
        g = P.filter(pl.col("horse_no") == h).sort("t")
        paths[int(h)] = (g["t"].to_numpy(), g["d"].to_numpy(), g["sd_s"].to_numpy())
    leg = np.load(run.p("legend.npz"))["rows"]
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    f2t = dict(zip(T["rows"][:, 0].astype(int).tolist(), T["t"].tolist()))
    neg, rank, seen = {}, {}, 0
    last_t = -1e9
    for r in leg:
        tt = f2t.get(int(r[0]))
        if tt is None or tt - last_t < 1.0:  # ~1 Hz
            continue
        last_t = tt
        listed = [int(x) for x in r[1:9] if x > 0]
        dd = {}
        for h, (pt, pd, ps) in paths.items():
            if pt[0] <= tt <= pt[-1] and np.interp(tt, pt, ps) < SD_TIGHT:
                dd[h] = float(np.interp(tt, pt, pd))
        if len(dd) < 4:
            continue
        seen += 1
        order = sorted(dd, key=dd.get)
        rk = {h: i for i, h in enumerate(order)}
        if len(listed) == 8:
            # an unlisted horse clearly AHEAD of the rearmost tight listed horse (the E-step's floor
            # rule, FLOOR_EPS): ranks among tight horses alone over-count when listed horses are vague
            tight_listed = [dd[h] for h in listed if h in dd]
            if tight_listed:
                rear = max(tight_listed)
                for h in order:
                    if h not in listed and dd[h] < rear - FLOOR_M:
                        neg[h] = neg.get(h, 0) + 1
        for i, h in enumerate(r[1:9]):
            h = int(h)
            if h > 0 and h in rk and abs(rk[h] - i) > 2:
                rank[h] = rank.get(h, 0) + 1
    out = {
        "legend_seconds": seen,
        "legend_negative": int(sum(neg.values())),
        "legend_negative_by": neg,
        "legend_rank": int(sum(rank.values())),
        "legend_rank_by": rank,
    }
    # ghost crossings
    read, gaps = set(), {}
    if run.p("finish.npz").exists():
        zf = np.load(run.p("finish.npz"))
        sc = dict(line_x=float(zf["line_x"]), reads=zf["reads"])
        read = finishcam.read_numbers(sc)
        gaps = {int(k): v for k, v in finishcam.finish_gaps(sc)[1].items()}
    DL = np.load(run.p("det_labels.npz"))["labels"]
    ghosts, cross = [], {}
    for h, (pt, pd, ps) in paths.items():
        k = np.flatnonzero(pd <= 0)
        if not len(k) or ps[k[0]] >= SD_TIGHT:
            continue
        cross[h] = float(pt[k[0]])
        own = T["t"][DL == h]
        if h not in read and (not len(own) or own.max() < cross[h] - 3.0):
            ghosts.append(h)
    out["ghost_crossing"] = len(ghosts)
    out["ghost_by"] = ghosts
    inv = []
    hs = [h for h in gaps if h in cross]
    for i, a in enumerate(hs):
        for b in hs[i + 1 :]:
            if abs(gaps[a] - gaps[b]) > 150 and (gaps[a] < gaps[b]) != (cross[a] < cross[b]):
                inv.append((a, b))
    out["finish_order"] = len(inv)
    out["finish_order_pairs"] = inv
    return out


def audit(run, stages=("rail", "field", "identity")) -> dict:
    out = {}
    if "rail" in stages:
        out.update(rail_checks(run))
    if "field" in stages and run.p("tracks.npz").exists():
        out.update(field_checks(run))
    if "identity" in stages and run.p("paths_video.parquet").exists():
        out.update(identity_checks(run))
    return out


def rail_blocking(res) -> list:
    """Rail violations that must block an ensemble (BLOCKING list; empty = pass)."""
    bad = []
    runs = [b - a + 1 for a, b in res.get("rail_off_map_at", [])]
    if runs and max(runs) > OFF_MAP_RUN:
        bad.append(f"rail off map ({max(runs)} contiguous stations)")
    if res.get("markers", 0) < 3:
        bad.append("marker grid not found")
    if np.isfinite(res.get("marker_spacing", np.nan)) and res["marker_spacing"] > 0.10:
        bad.append(f"marker spacing off by {res['marker_spacing']:.0%}")
    return bad
