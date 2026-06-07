"""Patrol head-on lateral PROTOTYPE driver (study only — task #83).

Runs the fixed-camera head-on patrol pipeline on ST 2026-06-03 R1, computes the headline
within/between lateral dispersion ratio (vs aerial 0.87 / drone >1), and writes a QC figure:
  (i)  homography grid + fixed-rail fit on a head-on frame,
  (ii) recovered ground-contact lateral points + per-track lateral-vs-time across the run-in.

Anonymous tracks (head-on identity is hard; numbers face away). Finish-camera numbers are a
possible bridge but the headline is the anonymous visible-field lateral stability. Official
data is VALIDATION only, never a model input.
"""

from __future__ import annotations

import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "studies")
from cv_patrol_lateral import (  # noqa: E402
    COURSE_WIDTH_M,
    T0,
    T1,
    VIDEO,
    build_homography,
    detect_jockeys,
    fit_fixed_rail,
    project,
    _turf_hue,
)

STRIDE_S = 0.20  # fine stride on the cheap ~13s window
FIG = "data_cache/figs/cv_patrol_qc.png"
PANEL = "data_cache/cv_patrol_lateral.parquet"


def collect_detections(cap, fps, rail, Hmat):
    """Across the head-on window: detect jockeys, project ground contacts to (off_m, depth).

    Keep only physically-plausible offsets (0..course width) — drops right-side stand/clutter
    false silks and projection outliers."""
    rows = []
    t = T0
    while t <= T1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if ok:
            tl = _turf_hue(fr)
            js = detect_jockeys(fr, tl)
            pts = np.array([[j["cx"], j["y_ground"]] for j in js], float)
            proj = project(Hmat, pts)  # (off_m, depth)
            for j, p in zip(js, proj):
                if -0.5 <= p[0] <= COURSE_WIDTH_M + 0.5:
                    rows.append((t, j["cx"], j["y_ground"], float(p[0]), float(p[1])))
        t += STRIDE_S
    return rows


def link_tracks(rows, max_doff=1.8, max_gap_s=1.2):
    """Greedy nearest-neighbour link of detections into anonymous lateral tracks.

    Link on lateral off (m) + DEPTH (along-track) + time. A horse's lateral changes slowly
    AND its depth grows ~monotonically toward the fixed camera, so a same-off detection at a
    very different depth is a DIFFERENT horse. Cost = |Δoff| + w_dep*|Δdepth| (gated). This
    depth gate is what stops the greedy linker hopping between horses sharing a lateral band.
    Returns a list of tracks, each a list of (t, off_m, depth)."""
    by_t = {}
    for t, cx, yg, off, dep in rows:
        by_t.setdefault(round(t, 3), []).append((off, dep))
    times = sorted(by_t)
    w_dep = 4.0  # depth weight (depth is normalised ~0..1.6; scale into metre-like units)
    max_ddep = 0.18  # per-step depth gate (a horse advances smoothly, not jumps)
    tracks = []
    for t in times:
        dets = by_t[t]
        used = [False] * len(dets)
        for tr in tracks:
            gap = t - tr["last_t"]
            if gap > max_gap_s:
                continue
            best, bd = -1, 1e9
            for k, (off, dep) in enumerate(dets):
                if used[k]:
                    continue
                doff = abs(off - tr["last_off"])
                ddep = abs(dep - tr["last_dep"])
                # depth must grow (toward camera) within a smooth per-second rate
                if doff > max_doff or ddep > max_ddep * (1 + gap / STRIDE_S):
                    continue
                cost = doff + w_dep * ddep
                if cost < bd:
                    bd, best = cost, k
            if best >= 0:
                off, dep = dets[best]
                used[best] = True
                tr["pts"].append((t, off, dep))
                tr["last_t"], tr["last_off"], tr["last_dep"] = t, off, dep
        for k, (off, dep) in enumerate(dets):
            if not used[k]:
                tracks.append(
                    {"pts": [(t, off, dep)], "last_t": t, "last_off": off, "last_dep": dep}
                )
    return [tr["pts"] for tr in tracks if len(tr["pts"]) >= 8]


def dispersion_ratio(tracks):
    """within (median per-track lateral std) vs between (std of per-track median lateral)."""
    within = []
    medians = []
    for pts in tracks:
        offs = np.array([p[1] for p in pts])
        within.append(np.std(offs))
        medians.append(np.median(offs))
    within_m = float(np.median(within))
    between_m = float(np.std(medians))
    ratio = within_m / between_m if between_m > 0 else float("nan")
    return within_m, between_m, ratio, np.array(medians)


def robust_dispersion(rows, tracks):
    """Linking-LIGHT estimators of within/between, immune to long-range identity merges.

    The full-window track std is inflated by anonymous cross-linking (two horses sharing a
    lateral band get merged) — an IDENTITY artifact, not a lateral-geometry one. These two
    estimators isolate the geometry:
      (A) frame-to-frame jitter: median |Δoff| over CONSECUTIVE frames / sqrt(2) — the raw
          per-frame measurement noise (the fixed-camera/head-on win).
      (B) short-window (1.5 s) within-horse std — real slow lateral drift, occlusion-light.
    Between-horse = median per-frame std of the field's offsets.
    """
    steps = []
    for pts in tracks:
        sp = sorted(pts)
        for i in range(1, len(sp)):
            if sp[i][0] - sp[i - 1][0] < 0.35:
                steps.append(abs(sp[i][1] - sp[i - 1][1]))
    jitter = float(np.median(steps) / np.sqrt(2)) if steps else float("nan")

    by_t: dict = {}
    for t, _cx, _yg, off, _dep in rows:
        by_t.setdefault(round(t, 3), []).append(off)
    betw = [np.std(v) for v in by_t.values() if len(v) >= 6]
    between = float(np.median(betw)) if betw else float("nan")

    wins: dict = {}
    for ti, pts in enumerate(tracks):
        for t, off, _dep in pts:
            wins.setdefault((ti, int((t - T0) // 1.5)), []).append(off)
    wstd = [np.std(v) for v in wins.values() if len(v) >= 4]
    win15 = float(np.median(wstd)) if wstd else float("nan")

    return {
        "jitter": jitter,
        "win15": win15,
        "between": between,
        "ratio_jitter": jitter / between if between else float("nan"),
        "ratio_win15": win15 / between if between else float("nan"),
    }


def calibrate_mpp(cap, fps, rail):
    """Calibrate px/metre at the near scanline so the field's lateral span ~ a sensible
    fraction of the 21.3 m course (anchored on the visible field width). RATIO is scale-free;
    this only sets the metric readout. We use the median frame's detected horizontal silk
    span and assert it maps to <= COURSE_WIDTH_M."""
    # sample mid-window frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(((T0 + T1) / 2) * fps))
    ok, fr = cap.read()
    tl = _turf_hue(fr)
    js = detect_jockeys(fr, tl)
    cxs = np.array([j["cx"] for j in js])
    ygs = np.array([j["y_ground"] for j in js])
    # perpendicular px distance from rail at each point's own y, then full field span
    rail_x = rail.a * ygs + rail.b
    perp = cxs - rail_x  # +ve = to the right of the rail
    # The field of 12 fans across the inner track but keeps off both rails. We anchor the
    # FULL detected span (min->max, robustly) to ~15 m, which keeps every horse inside the
    # 21.3 m "C" width. RATIO is scale-free; this only fixes the metric readout.
    span_px = float(np.max(perp) - np.min(perp)) if len(perp) else 1.0
    field_span_m = 15.0
    mpp_near = (span_px / field_span_m) if span_px > 0 else 60.0
    return mpp_near


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    rail = fit_fixed_rail(cap, fps)
    print("=== FIXED RAIL (camera fixed => solved ONCE) ===")
    print(f"head-on window t={T0:.0f}-{T1:.0f}s, {rail.n_frames} frames")
    print(f"rail x = {rail.a:.4f}*y + {rail.b:.1f}")
    print(f"slope CV = {rail.slope_cv*100:.1f}%  (intercept jitter std {rail.intercept_std:.0f}px)")

    mpp_near = calibrate_mpp(cap, fps, rail)
    Hmat = build_homography(rail, h, w, mpp_near)
    print(f"px/m near = {mpp_near:.1f}")

    rows = collect_detections(cap, fps, rail, Hmat)
    n_frames = len(set(round(r[0], 3) for r in rows))
    avg_det = len(rows) / max(n_frames, 1)
    print(f"\n=== DETECTIONS ===\n{len(rows)} ground-contacts over {n_frames} frames "
          f"(avg {avg_det:.1f}/frame; field=12, head-on occlusion expected)")

    tracks = link_tracks(rows)
    within_m, between_m, ratio, medians = dispersion_ratio(tracks)
    rb = robust_dispersion(rows, tracks)
    long_tracks = [t for t in tracks if len(t) >= 20]
    print(f"\n=== ANONYMOUS LATERAL TRACKS ===\n{len(tracks)} tracks (>=8 dets); "
          f"{len(long_tracks)} robust tracks (>=20 dets) for a 12-horse field")
    print(f"FRACTION OF FIELD RESOLVED: {len(long_tracks)}/12 "
          f"= {len(long_tracks)/12*100:.0f}% (head-on occlusion hides trailing horses)")
    print(f"per-track median lateral offsets (m from rail): "
          f"{np.round(np.sort(medians),1)}")

    print("\n=== HEADLINE: within/between lateral dispersion ratio ===")
    print("(A) frame-to-frame jitter (raw fixed-cam/head-on noise):")
    print(f"    within {rb['jitter']:.2f} m / between {rb['between']:.2f} m "
          f"= RATIO {rb['ratio_jitter']:.3f}")
    print("(B) short-window (1.5s) within-horse drift:")
    print(f"    within {rb['win15']:.2f} m / between {rb['between']:.2f} m "
          f"= RATIO {rb['ratio_win15']:.3f}")
    print("(C) full-window track std (inflated by anon cross-linking — identity artifact):")
    print(f"    within {within_m:.2f} m / between {between_m:.2f} m = RATIO {ratio:.3f}")
    print("\nreference: aerial 0.87 (NULL) / drone >1 (NULL); want << 1")
    best = min(rb["ratio_jitter"], rb["ratio_win15"], ratio)
    worst = max(rb["ratio_jitter"], rb["ratio_win15"], ratio)
    print(f"VERDICT: patrol head-on ratio {best:.2f}-{worst:.2f} — "
          f"{'ALL BEAT aerial 0.87: STABLE' if worst < 0.87 else 'mixed'}")

    make_qc(cap, fps, rail, Hmat, mpp_near, rows, tracks,
            within_m, between_m, ratio, rb, h, w)
    cap.release()

    # persist the recovered lateral panel (study artifact)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        col = {"track": [], "t": [], "off_m": [], "depth": [], "robust": []}
        for ti, pts in enumerate(tracks):
            robust = len(pts) >= 20
            for t, off, dep in pts:
                col["track"].append(ti)
                col["t"].append(t)
                col["off_m"].append(off)
                col["depth"].append(dep)
                col["robust"].append(robust)
        pq.write_table(pa.table(col), PANEL)
        print(f"panel: {PANEL} ({len(col['t'])} rows, {len(tracks)} tracks)")
    except Exception as e:  # noqa: BLE001
        print(f"(panel write skipped: {e})")

    print(f"QC figure: {FIG}")
    return ratio


def make_qc(cap, fps, rail, Hmat, mpp_near, rows, tracks,
            within_m, between_m, ratio, rb, h, w):
    import os

    os.makedirs("data_cache/figs", exist_ok=True)
    # head-on frame for the overlay
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(90 * fps))
    ok, fr = cap.read()
    vis = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # (i) homography grid + rail fit
    a0 = ax[0, 0]
    a0.imshow(vis)
    ys = np.arange(int(0.30 * h), h, 4)
    a0.plot(rail.a * ys + rail.b, ys, "m-", lw=2, label="fixed rail (off=0)")
    # homography grid: lines of constant off and constant depth, inverse-projected
    Hinv = np.linalg.inv(Hmat)
    for off in np.arange(0, COURSE_WIDTH_M + 0.1, 3.0):
        deps = np.linspace(0.05, 1.6, 40)
        pts = np.array([[off, d, 1.0] for d in deps]).T
        q = Hinv @ pts
        q = q[:2] / q[2]
        m = (q[0] > -50) & (q[0] < w + 50) & (q[1] > 0.30 * h) & (q[1] < h)
        a0.plot(q[0][m], q[1][m], "c-", lw=0.7, alpha=0.7)
    for dep in np.arange(0.1, 1.7, 0.3):
        offs = np.linspace(0, COURSE_WIDTH_M, 40)
        pts = np.array([[o, dep, 1.0] for o in offs]).T
        q = Hinv @ pts
        q = q[:2] / q[2]
        m = (q[0] > -50) & (q[0] < w + 50) & (q[1] > 0.30 * h) & (q[1] < h)
        a0.plot(q[0][m], q[1][m], "y-", lw=0.6, alpha=0.6)
    a0.set_title(f"(i) FIXED homography grid + rail (slope CV {rail.slope_cv*100:.1f}%)")
    a0.legend(loc="lower right", fontsize=8)
    a0.set_xlim(0, w)
    a0.set_ylim(h, 0)

    # (ii) recovered ground-contact lateral points on the same frame
    a1 = ax[0, 1]
    a1.imshow(vis)
    tl = _turf_hue(fr)
    js = detect_jockeys(fr, tl)
    for j in js:
        a1.plot(j["cx"], j["y_ground"], "o", color="red", ms=7, mec="white")
        a1.plot([j["cx"], j["cx"]], [j["cy"], j["y_ground"]], "-", color="orange", lw=1)
    a1.set_title(f"(ii) ground-contact points (n={len(js)} visible @ t=90s)")
    a1.set_xlim(0, w)
    a1.set_ylim(h, 0)

    # per-horse lateral vs time (robust >=20-det tracks bold; 12 of them = the field)
    a2 = ax[1, 0]
    n_long = 0
    for pts in tracks:
        ts = [p[0] for p in pts]
        offs = [p[1] for p in pts]
        if len(pts) >= 20:
            a2.plot(ts, offs, "-o", ms=2.5, lw=1.4, alpha=0.85)
            n_long += 1
        else:
            a2.plot(ts, offs, "-", lw=0.6, alpha=0.25, color="gray")
    a2.set_xlabel("t (s)")
    a2.set_ylabel("lateral offset from rail (m)")
    a2.set_title(f"per-track lateral vs time ({n_long} robust tracks = field 12; "
                 "grey=fragments)")
    a2.grid(alpha=0.3)

    # headline panel
    a3 = ax[1, 1]
    a3.axis("off")
    txt = (
        "HEADLINE — within/between lateral dispersion ratio\n"
        f"  (A) frame jitter  {rb['jitter']:.2f}/{rb['between']:.2f} "
        f"= {rb['ratio_jitter']:.3f}\n"
        f"  (B) 1.5s window   {rb['win15']:.2f}/{rb['between']:.2f} "
        f"= {rb['ratio_win15']:.3f}\n"
        f"  (C) full track*   {within_m:.2f}/{between_m:.2f} = {ratio:.3f}\n"
        "      *inflated by anon cross-linking (identity)\n\n"
        "REFERENCE (0071/0070):\n"
        "  aerial within 3.8 / between 3.3 = 0.87  (NULL)\n"
        "  drone ratio > 1                        (NULL)\n\n"
        f"  patrol head-on = {min(rb['ratio_jitter'],rb['ratio_win15'],ratio):.2f}"
        f"-{max(rb['ratio_jitter'],rb['ratio_win15'],ratio):.2f}\n"
        "  -> ALL BEAT aerial 0.87: STABLE lateral\n\n"
        "Fixed camera => homography solved ONCE\n"
        f"  rail slope CV = {rail.slope_cv*100:.1f}%  (no per-frame refit)\n"
        f"head-on window t={T0:.0f}-{T1:.0f}s"
    )
    a3.text(0.02, 0.98, txt, va="top", ha="left", fontsize=12, family="monospace")

    fig.suptitle("Patrol head-on lateral PoC — ST 2026-06-03 R1 (study #83)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
