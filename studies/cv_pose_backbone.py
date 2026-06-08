"""Pose backbone v2 — FIXED-anisotropy lateral scale (the fundamental metric fix; journal 0105).

The AVV is a tilted top-down render whose virtual camera only PANS / ZOOMS to follow the field, so
the lateral pixels-per-metre ``s_lat(f) = (inner->outer pixel separation) / 22.5`` is a SMOOTH,
slowly-varying function of frame. The residual lateral-recon noise (journal 0104) was NOT a smoothing
problem: it came from per-frame OUTER-rail mis-detections (the rail's shadow / a parallel marker, not
the true 22.5 m outer) injecting a wrong ``s_lat`` for whole stretches (f750-1000, f1700-2150). Batch
smoothing could not reject it because the error is regionally SYSTEMATIC, not zero-mean jitter.

The fix exploits the slow zoom directly: measure ``s_lat`` on every two-rail frame, then build a
ROBUST s_lat(f) track that REJECTS the wrong-outer outliers (a fake-outer separation is far smaller
than the true 22.5 m one, so it sits well BELOW the slowly-varying zoom curve) and INTERPOLATES the
scale through the bad stretches from the good neighbours. Lateral distance is then

    y_lat(m) = |perp-to-inner-rail (px)| / s_lat(f)

measured against the inner rail CURVE (so the bend is handled by the perpendicular itself) and never
depending on a per-frame outer detection. This is strictly simpler than the affine/EKF backbone and
attacks the exact root cause from journal 0104.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_pose_backbone.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
OBS = "data_cache/cv_joint_aerial_R08_obs.npz"
PASS1 = "data_cache/cv_pose_pass1.npz"     # cached per-frame detection (skip the 12-min decode)
DIAG = "data_cache/diag"
HALF = 960
DIST = 1800.0
FIELD = 11
RAIL_W = 22.5
SEP_MIN_PX = 30.0     # an inner->outer separation below this is too small to be the true 22.5 m rail
FW = 90               # robust-median half-window (frames, ~3.6 s) for the slow zoom track
MAD_K = 2.5           # reject an s_lat measurement this many robust-sigmas off the zoom level


def _sep_px(inner, outer):
    """Median perpendicular separation (px) of the outer rail polyline from the inner rail curve."""
    from hkjc.vision.geometry_cv import project_to_polyline
    perps = [abs(project_to_polyline(inner, float(x), float(y))[1]) for x, y in outer]
    return float(np.median(perps)) if perps else float("nan")


def _robust_track(frames, vals, fw, mad_k):
    """Robust slow-zoom s_lat track. Returns ``(gated_frames, smoothed_vals)`` for np.interp.

    The virtual camera holds a near-CONSTANT zoom, so a GLOBAL robust gate kills the gross wrong-outer
    spikes in BOTH directions (a shadow/marker outer reads too NEAR -> s_lat too small; a far
    background line reads too FAR -> s_lat too large) — crucially this survives stretches where MOST
    measurements are wrong, which a local-window median cannot (it chases the regional bias). Among the
    survivors a gentle local median then tracks the slow drift, and the returned values are smoothed so
    interpolation through the rejected gaps is clean.
    """
    f = np.asarray(frames, float)
    v = np.asarray(vals, float)
    m0 = np.median(v)
    mad0 = np.median(np.abs(v - m0)) + 1e-6
    glob = np.abs(v - m0) <= mad_k * 1.4826 * mad0     # global gate: zoom is near-constant
    keep = glob.copy()
    smooth = np.full(len(f), m0)
    for _ in range(2):
        for j in range(len(f)):
            w = keep & (np.abs(f - f[j]) <= fw)
            smooth[j] = np.median(v[w]) if w.sum() >= 5 else m0
        resid = v - smooth
        mad = np.median(np.abs(resid[keep])) + 1e-6
        keep = glob & (np.abs(resid) <= mad_k * 1.4826 * mad)   # gross gate AND local-consistency gate
        if keep.sum() < 5:
            break
    return f[keep], smooth[keep]


def main():
    import av
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cv_rigid_rail import build_rail_geometry, local_radius
    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars
    from hkjc.vision.feeds.aerial_rail import line_candidates, select_running_rail
    from hkjc.vision.geometry_cv import project_to_polyline
    os.makedirs(DIAG, exist_ok=True)
    z = np.load(OBS, allow_pickle=True)
    fr_all, odo = z["frames"], z["odo"]
    fin = np.isfinite(odo)
    geom = build_rail_geometry("HV", "C")

    # PASS 1 — per render frame: inner rail (always), field pixels, and the raw two-rail s_lat where
    # an outer is detected. Inner selection runs temporally (carry the previous pick/tangent). Cached
    # to PASS1 (the 12-min decode) so the gate/recon (PASS 2/3) can be iterated instantly.
    if os.path.exists(PASS1):
        c = np.load(PASS1, allow_pickle=True)
        keys = list(c["keys"])
        frame_inner = dict(zip(keys, c["inner"], strict=True))
        frame_fpts = dict(zip(keys, c["fpts"], strict=True))
        meas_f, meas_slat = list(c["meas_f"]), list(c["meas_slat"])
    else:
        frame_inner, frame_fpts = {}, {}
        meas_f, meas_slat = [], []
        prev_tang = prev_curve = None
        for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
            img = fr.to_ndarray(format="bgr24")
            panel = img[:, HALF:]
            if not render_active(panel):
                continue
            ph = panel.shape[0]
            ty0, ty1 = track_band(panel)
            dets, _d = detect_avatars(panel, ty0, ty1, field=FIELD)
            if len(dets) < 3:
                continue
            fpts = np.array([[d.cx, d.cy] for d in dets], float)
            d_ff = DIST - float(np.interp(i, fr_all[fin], odo[fin]))
            radius = local_radius(geom, d_ff)
            rb0, rb1 = int(0.28 * ph), int(0.82 * ph)
            cands = line_candidates(panel, rb0, rb1)
            inner, tang, info = select_running_rail(
                cands, deg=(2 if np.isfinite(radius) else 1), field_pts=fpts,
                prev_tang=prev_tang, prev_curve=prev_curve)
            if inner is None:
                continue
            prev_tang, prev_curve = tang, inner
            frame_inner[i] = inner
            frame_fpts[i] = fpts
            outer = (info or {}).get("outer")
            if outer is not None:
                sep = _sep_px(inner, outer)
                if np.isfinite(sep) and sep >= SEP_MIN_PX:
                    meas_f.append(i)
                    meas_slat.append(sep / RAIL_W)
        keys = sorted(frame_inner)
        np.savez(PASS1, keys=np.array(keys),
                 inner=np.array([frame_inner[k] for k in keys], dtype=object),
                 fpts=np.array([frame_fpts[k] for k in keys], dtype=object),
                 meas_f=np.array(meas_f), meas_slat=np.array(meas_slat))

    # PASS 2 — robust slow-zoom s_lat track: gate the wrong-outer outliers, interpolate to any frame.
    gf, gv = _robust_track(meas_f, meas_slat, FW, MAD_K)

    def s_lat_at(f):
        return float(np.interp(f, gf, gv))

    # PASS 3 — lateral reconstruction: |perp to inner curve| / s_lat(f), perspective/zoom corrected.
    # ON-TRACK gate: with a validated rail + scale, any detection beyond the 22.5 m running surface is
    # background (the detector occasionally grabs apron/grandstand blobs at 30-40 m), so drop it.
    rows = []
    for f in sorted(frame_inner):
        inner, fpts = np.asarray(frame_inner[f], float), np.asarray(frame_fpts[f], float)
        s_lat = s_lat_at(f)
        b = np.array([abs(project_to_polyline(inner, float(x), float(y))[1]) for x, y in fpts]) / s_lat
        b = b[np.isfinite(b) & (b <= RAIL_W + 2.5)]
        if len(b) < 3:
            continue
        rows.append((f, float(np.median(b)), float(np.percentile(b, 10)), float(np.percentile(b, 90))))
    rows = np.array(rows)
    drop = len(meas_f) - len(gf)
    print(f"two-rail s_lat measured={len(meas_f)}; wrong-outer rejected={drop}; "
          f"frames reconstructed={len(rows)} of inner-rail {len(frame_inner)}")
    print(f"lateral median over race = {np.median(rows[:, 1]):.1f} m  (10-90 spread "
          f"{np.median(rows[:, 3] - rows[:, 2]):.1f} m)")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax0.scatter(meas_f, meas_slat, s=6, c="0.7", label="raw two-rail s_lat")
    ax0.plot(gf, gv, ".", ms=3, c="C1", label="gated (wrong-outer rejected)")
    fg = np.array(sorted(frame_inner))
    ax0.plot(fg, [s_lat_at(f) for f in fg], lw=1.2, c="C3", label="s_lat(f) track (interp)")
    ax0.set_ylabel("s_lat (px per metre)")
    ax0.set_title("Slow-zoom lateral scale — wrong-outer outliers gated, interpolated through gaps")
    ax0.legend(loc="upper left")
    ax1.fill_between(rows[:, 0], rows[:, 2], rows[:, 3], alpha=0.3, label="field 10-90 pct")
    ax1.plot(rows[:, 0], rows[:, 1], lw=1.0, label="field median y_lat")
    ax1.set_xlabel("aerial frame")
    ax1.set_ylabel("y_lat from running rail (m)")
    ax1.set_title("Pose-backbone v2 LATERAL reconstruction — fixed-anisotropy s_lat, perp-to-inner")
    ax1.set_ylim(0, 30)
    ax1.legend(loc="upper left")
    out = f"{DIAG}/cv_pose_lateral.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
