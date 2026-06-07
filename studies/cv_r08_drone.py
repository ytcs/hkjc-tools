"""R08 DRONE feed — REUSE cv_drone_revisit (#87/0080) + the R03 race-agnostic bridge on R08 (#95).

HELD-OUT route generalization. Imports the R1 drone GEOMETRY unchanged (_detect_field /
_fit_stable_rail / _rail_perp / _rail_points / _corridor_px) + the R03 driver machinery
(find_finish_cut / build_silk_anchors_from_rf / _extract / _window_rows) and re-points ONLY
the clock anchors + the route distance curve. NO algorithm tuned for R08.

The drone's R1/R03 role is ordering-only / metric-null (a supplementary lateral feed); if the
RF-finish-cam identity bridge is thin it falls back to anonymous, as expected.

Writes data_cache/cv_r08_drone_numbered.parquet. Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_r08_drone.py
"""
from __future__ import annotations

import re
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "studies")
import cv_drone_revisit as D1  # noqa: E402,N812  (R1 drone geometry, reused)
from cv_drone_revisit import (  # noqa: E402
    _corridor_px,
    _detect_field,
    _fit_stable_rail,
    _rail_perp,
    _rail_points,
)
from cv_r08_common import (  # noqa: E402
    TRACK_WIDTH_M,
    WINNER_FINISH_RACE_T,
    lateral_truth,
    parse_official,
    race_t_to_arclen,
)

DRONE_VIDEO = "data_cache/video/drone_20260603_08.mp4"
RF_VIDEO = "data_cache/video/full_20260603_08.mp4"
OUT_PANEL = "data_cache/cv_r08_drone_numbered.parquet"
FIG = "data_cache/figs/cv_r08_drone_qc.png"
FPS = 25.0

D1.DRONE_VIDEO = DRONE_VIDEO
D1.TRACK_WIDTH_M = TRACK_WIDTH_M

# Shared-clock anchors (R08 route). Gate jump + finish-cut frame-anchored below; the route winner
# crosses much later (108.6s vs the sprint's 69.3s).
DRONE_START_VID_T = 3.0
DRONE_FINISH_VID_T = 120.5   # winner #6 line-cross on the drone feed (route; verified: field at
# the 50m marker ~vid 118, #6 surging clear, crosses ~120.5 — matches the patrol 120.7 cross)

# RF finish-cam legible-approach window for the identity bridge (route timing; the winner crosses
# RF ~vid 109.7, legible approach ~108-109.5).
RF_FIN_T0, RF_FIN_T1, RF_FIN_DT = 107.5, 109.6, 0.25


def drone_vid_to_race_t(vid_t):
    v0, v1 = DRONE_START_VID_T, DRONE_FINISH_VID_T
    return (np.asarray(vid_t, float) - v0) * (WINNER_FINISH_RACE_T - 0.0) / (v1 - v0)


def find_finish_cut(cap):
    """Re-derive the finish/run-in vid time by field-centroid jump (REUSE R03 logic, route range)."""
    import cv2
    prev = None
    jumps = []
    turf_lock = None
    t = 90.0
    while t < 140.0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
        ok, fr = cap.read()
        if not ok:
            break
        blobs, turf_lock, _turf = _detect_field(fr, turf_lock)
        if blobs:
            cx = float(np.median([b[0] for b in blobs]))
            cy = float(np.median([b[1] for b in blobs]))
            if prev is not None:
                jump = np.hypot(cx - prev[0], cy - prev[1])
                jumps.append((t, jump))
            prev = (cx, cy)
        t += 0.5
    late = [(t, j) for t, j in jumps if t > 108]
    if late:
        return max(late, key=lambda x: x[1])[0]
    return DRONE_FINISH_VID_T + 5.0


def build_silk_anchors_from_rf() -> dict:
    """Per-number silk-LAB anchors from the RF finish-cam OCR (REUSE R03 race-agnostic bridge)."""
    import cv2
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return {}
    ocr = RapidOCR()
    cap = cv2.VideoCapture(RF_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    acc: dict[int, list] = {}
    for t in np.arange(RF_FIN_T0, RF_FIN_T1, RF_FIN_DT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        h = fr.shape[0]
        crop = fr[int(h * 0.30): int(h * 0.82), :]
        lab_img = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
        for scale in (1.6, 2.4):
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(big)
            if not res:
                continue
            for box, txt, conf in res:
                txt = txt.strip()
                if not (re.fullmatch(r"\d{1,2}", txt) and conf > 0.45 and 1 <= int(txt) <= 11):
                    continue
                num = int(txt)
                cx = float(np.mean([p[0] for p in box])) / scale
                cy = float(np.mean([p[1] for p in box])) / scale + int(h * 0.30)
                y0 = max(0, int(cy - 36))
                y1 = max(0, int(cy - 6))
                x0 = max(0, int(cx - 16))
                x1 = min(fr.shape[1], int(cx + 16))
                patch = lab_img[y0:y1, x0:x1].reshape(-1, 3).astype(float)
                if len(patch) >= 8:
                    chroma = np.abs(patch[:, 1] - 128) + np.abs(patch[:, 2] - 128)
                    k = max(3, int(0.3 * len(patch)))
                    acc.setdefault(num, []).append(patch[np.argsort(chroma)[-k:]].mean(0))
    cap.release()
    return {n: np.median(np.vstack(v), 0) for n, v in acc.items() if len(v) >= 2}


def main():
    import av
    import cv2

    off = parse_official()
    rail_t, wide_t = lateral_truth(off)
    print("=" * 80)
    print("R08 DRONE #95 — 3rd lateral feed on HV 1800m ROUTE (reusing cv_drone toolkit, no tuning)")
    print("=" * 80)

    cap = cv2.VideoCapture(DRONE_VIDEO)
    cut = find_finish_cut(cap)
    cap.release()
    print(f"[GENERALIZE] drone finish/cut auto-scan -> vid {cut:.1f}s; "
          f"using winner-cross vid {DRONE_FINISH_VID_T:.1f}s")
    print(f"[clock] race_t = (vid - {DRONE_START_VID_T}) * "
          f"{WINNER_FINISH_RACE_T / (DRONE_FINISH_VID_T - DRONE_START_VID_T):.4f}")

    anchors = build_silk_anchors_from_rf()
    print(f"[identity] RF-finish-cam silk anchors: {sorted(anchors)} ({len(anchors)}/11)")
    if len(anchors) < 3:
        print("[GENERALIZE] drone identity DEGRADED — falls back to anonymous lateral "
              "(drone's role was ordering-only/metric-null anyway).")

    c = av.open(DRONE_VIDEO)
    frames = []
    for i, fr in enumerate(c.decode(video=0)):
        t = i / FPS
        if t > DRONE_FINISH_VID_T + 1:
            break
        if i % 5 == 0:
            frames.append((t, fr.to_ndarray(format="bgr24")))
    c.close()
    print(f"[decode] {len(frames)} frames (stride 5) over vid 0-{DRONE_FINISH_VID_T:.0f}s")

    rows = _extract(frames, anchors)
    if not rows:
        print("[GENERALIZE-FAIL] no drone rows extracted")
        return
    df = pl.DataFrame(rows)
    df.write_parquet(OUT_PANEL)
    nnum = df.filter(pl.col("number") > 0)["number"].n_unique() if "number" in df.columns else 0
    print(f"[panel] {OUT_PANEL} ({df.height} rows, {nnum} numbered runners)")

    _validate(df, off, rail_t, wide_t)
    _qc(df, off)


def _extract(frames, anchors):
    rows = []
    win = []
    win_t0 = frames[0][0] if frames else 0.0
    turf_lock = None
    for (t, img) in frames:
        if t - win_t0 > 3.0 and win:
            rows += _window_rows(win, anchors, turf_lock)
            win = []
            win_t0 = t
        win.append((t, img))
    if win:
        rows += _window_rows(win, anchors, turf_lock)
    return rows


def _window_rows(win, anchors, turf_lock):
    rail_pool, per_frame = [], []
    for vid_t, img in win:
        blobs, turf_lock, turf = _detect_field(img, turf_lock)
        if not blobs or turf is None:
            continue
        field_xy = [(b[0], b[1]) for b in blobs]
        rp = _rail_points(turf, field_xy)
        cpx = _corridor_px(turf)
        if rp is not None:
            rail_pool.append(rp)
        per_frame.append((vid_t, blobs, cpx))
    if len(rail_pool) < 2 or not per_frame:
        return []
    rail = _fit_stable_rail(rail_pool)
    out = []
    for vid_t, blobs, cpx in per_frame:
        if cpx is None or cpx < 50:
            continue
        mpp = TRACK_WIDTH_M / cpx
        race_t = float(drone_vid_to_race_t(vid_t))
        arclen = float(race_t_to_arclen(race_t))
        for (cx, cy, gcx, gcy, _area, lab) in blobs:
            num = _assign(lab, anchors)
            perp = _rail_perp(rail, gcx, gcy)
            y_lat = float(min(max(abs(perp) * mpp, 0.0), TRACK_WIDTH_M))
            out.append(dict(vid_t=round(vid_t, 3), race_t=round(race_t, 3),
                            arclen=round(arclen, 2), number=num, y_lat=round(y_lat, 3),
                            straight=round(rail["straight"], 3)))
    return out


def _assign(lab, anchors, max_d=30.0):
    if not anchors:
        return 0
    best, bd = 0, 1e9
    for num, al in anchors.items():
        d = float(np.linalg.norm(np.array(lab) - al))
        if d < bd:
            bd, best = d, num
    return best if bd <= max_d else 0


def _validate(df, off, rail_t, wide_t):
    print("\n[VALIDATE drone lateral vs stewards rail/wide]")
    numbered = df.filter(pl.col("number") > 0)
    if numbered.height == 0:
        print("  (anonymous only — no numbered validation)")
        return
    band = numbered.filter(pl.col("straight") > 0.7)
    use = band if band.height > 20 else numbered
    med = {int(n): float(np.median(use.filter(pl.col("number") == n)["y_lat"].to_numpy()))
           for n in use["number"].unique().to_list()}
    rv = [med[n] for n in rail_t if n in med]
    wv = [med[n] for n in wide_t if n in med]
    if rv and wv:
        print(f"  rail {sorted(rail_t & set(med))} {np.mean(rv):.1f}m | "
              f"wide {sorted(wide_t & set(med))} {np.mean(wv):.1f}m | "
              f"GAP {np.mean(wv) - np.mean(rv):+.1f}m  ({len(med)} numbered)")


def _qc(df, off):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("data_cache/figs", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax[0].scatter(df["arclen"], df["race_t"], c=df["straight"], cmap="RdYlGn",
                       vmin=0.5, vmax=1.0, s=10)
    ax[0].set_xlabel("arclen (m)")
    ax[0].set_ylabel("race_t (s)")
    ax[0].set_title("drone coverage census (green=straight)")
    plt.colorbar(sc, ax=ax[0])
    numbered = df.filter(pl.col("number") > 0)
    for n in numbered["number"].unique().to_list():
        s = numbered.filter(pl.col("number") == n).sort("arclen")
        ax[1].plot(s["arclen"], s["y_lat"], "-o", ms=2, lw=1, alpha=0.7, label=f"#{n}")
    ax[1].set_xlabel("arclen (m)")
    ax[1].set_ylabel("y_lat (m)")
    ax[1].set_title("drone numbered y_lat")
    ax[1].legend(fontsize=6, ncol=2)
    fig.suptitle("R08 DRONE (#95, HV 2026-06-03 R8 1800m route)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=90)
    plt.close(fig)
    print(f"[fig] {FIG}")


if __name__ == "__main__":
    main()
