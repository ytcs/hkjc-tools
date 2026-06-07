"""Re-run harness for the UNIFIED IDENTITY module (#92).

Loads the solve-once identity (cv_identity.Identity, cached npz), PROPAGATES it to the patrol
head-on anonymous tracks and the drone sprites by silk + the shared-clock ORDER constraint, writes
NEW *_identity-suffixed parquets (no existing parquet mutated), and re-runs the R03 4-feed fusion
with the propagated identity -> a *_v2 trajectory. Reports per-feed coverage before->after, the
patrol numbered rail-vs-wide gap (sign), and the fused lateral gap.

DISCIPLINE: study-only; imports cv_identity / cv_r03_* READ-ONLY; src/hkjc untouched; writes only
new identity/_v2 outputs; official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group vision python studies/cv_identity_rerun.py
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
import polars as pl

sys.path.insert(0, "studies")

import cv_r03_patrol as RP  # noqa: E402,N812
from cv_identity import Identity, silk_hist  # noqa: E402
from cv_r03_common import (  # noqa: E402
    lateral_truth,
    parse_official,
    race_t_to_arclen,
)
from cv_r03_patrol import (  # noqa: E402
    autodetect_headon,
    build_homography,
    calibrate_mpp,
    collect_detections,
    detect_jockeys,
    fit_fixed_rail_window,
    link_tracks,
    project,
)
from cv_patrol_lateral import _turf_hue  # noqa: E402

IDENT_CACHE = "data_cache/cv_identity_R03.npz"
PATROL_VIDEO = "data_cache/video/patrol_20260603_03.mp4"
PATROL_NUMBERED_ID = "data_cache/cv_r03_patrol_numbered_identity.parquet"


# ============================================================================= patrol propagation
def propagate_patrol(ident: Identity, lam_order: float = 0.6):
    """Rebuild the patrol head-on tracks, sample each track's silk signature, and assign runner
    numbers by the UNIFIED identity (silk-EMD + RF-clock ORDER constraint).

    The order proxy for a head-on track is its along-track DEPTH (further from the head-on camera
    == further advanced == lower rank-number). The RF backbone supplies the true running rank at
    the patrol race_t; the constraint penalises silk assignments whose depth-order disagrees with
    the RF rank, breaking the red/pink silk degeneracy that flipped the R03 sign (0088).
    """
    cap = cv2.VideoCapture(PATROL_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    t0, t1 = autodetect_headon(cap, fps)
    rail = fit_fixed_rail_window(cap, fps, t0, t1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mpp = calibrate_mpp(cap, fps, rail, t0, t1)
    Hmat = build_homography(rail, h, w, mpp)
    rows = collect_detections(cap, fps, Hmat, t0, t1)
    tracks = link_tracks(rows)
    print(f"[patrol] head-on t={t0:.1f}-{t1:.1f}s, rail CV {rail.slope_cv * 100:.1f}%; "
          f"{len(tracks)} tracks ({sum(len(t) >= 20 for t in tracks)} robust)")

    # sample each robust track's silk hue-hist signature + its (off, depth)
    robust = [(ti, pts) for ti, pts in enumerate(tracks) if len(pts) >= 20]
    track_px: dict[int, list] = {}
    for t in np.arange(t0, t1, 0.4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        tl = _turf_hue(fr)
        js = detect_jockeys(fr, tl)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        proj = (project(build_homography(rail, fr.shape[0], fr.shape[1], mpp),
                        np.array([[j["cx"], j["y_ground"]] for j in js], float)) if js else [])
        for j, p in zip(js, proj):
            off_m, dep = float(p[0]), float(p[1])
            best_ti, bd = None, 1e9
            for ti, pts in robust:
                near = [pp for pp in pts if abs(pp[0] - t) < 0.6]
                if not near:
                    continue
                d = min(abs(off_m - pp[1]) + 4.0 * abs(dep - pp[2]) for pp in near)
                if d < bd:
                    bd, best_ti = d, ti
            if best_ti is not None and bd < 2.5:
                y0, y1 = max(0, int(j["cy"] - 12)), min(fr.shape[0], int(j["cy"] + 12))
                x0, x1 = max(0, int(j["cx"] - 14)), min(fr.shape[1], int(j["cx"] + 14))
                patch = hsv[y0:y1, x0:x1]
                if patch.size:
                    track_px.setdefault(best_ti, []).append(patch.reshape(-1, 3))
    cap.release()

    track_sig, track_off, track_depth, track_rt = {}, {}, {}, {}
    for ti, pts in robust:
        if ti not in track_px:
            continue
        s = silk_hist(np.vstack(track_px[ti]))
        if s is None:
            continue
        track_sig[ti] = s
        track_off[ti] = float(np.median([p[1] for p in pts]))
        track_depth[ti] = float(np.median([p[2] for p in pts]))
        mt = float(np.median([p[0] for p in pts]))
        track_rt[ti] = mt - RP.PATROL_OFFSET

    # ORDER proxy: greater depth == further from the head-on cam == further advanced (lower rank).
    assign = ident.propagate(track_sig, track_order=track_depth, track_race_t=track_rt,
                             lam_order=lam_order)
    print(f"[patrol] propagate (lam_order={lam_order}): assigned "
          f"{sorted(n for n, _ in assign.values())} ({len(assign)} tracks)")

    out = []
    for ti, (num, conf) in assign.items():
        for t, off_m, _dep in tracks[ti]:
            race_t = t - RP.PATROL_OFFSET
            out.append(dict(runner=int(num), race_t=round(race_t, 3),
                            arclen=round(float(race_t_to_arclen(race_t)), 2),
                            y_lat_m=round(off_m, 3), confidence=round(conf, 3)))
    df = pl.DataFrame(out).sort(["runner", "race_t"]) if out else None
    if df is not None:
        df.write_parquet(PATROL_NUMBERED_ID)
        print(f"[panel] {PATROL_NUMBERED_ID} ({df.height} rows, {df['runner'].n_unique()} runners)")
    return df, assign, track_sig


def validate_patrol(df, off):
    rail_t, wide_t = lateral_truth(off)
    print("\n[VALIDATE patrol numbered vs stewards rail/wide]")
    med = {n: float(np.median(df.filter(pl.col("runner") == n)["y_lat_m"].to_numpy()))
           for n in df["runner"].unique().to_list()}
    for n in sorted(med):
        tag = "rail/tight" if n in rail_t else ("wide" if n in wide_t else "mid")
        print(f"    #{n:2d} [{tag:10s}] {med[n]:5.1f}m")
    rv = [med[n] for n in rail_t if n in med]
    wv = [med[n] for n in wide_t if n in med]
    gap = float("nan")
    if rv and wv:
        gap = float(np.mean(wv) - np.mean(rv))
        print(f"  GAP wide-rail = {gap:+.1f}m  (rail {sorted(rail_t & set(med))} {np.mean(rv):.1f}m, "
              f"wide {sorted(wide_t & set(med))} {np.mean(wv):.1f}m)  [R03 baseline was -5.7m]")
        print(f"  SIGN: {'CORRECT (wide>rail)' if gap > 0 else 'WRONG (rail>wide)'}")
    return gap, med


def r01_regression(lam_order: float = 0.6):
    """REGRESSION: build the unified identity on R01 + propagate to R01 patrol tracks.

    R01 is the known-good case (banked patrol-numbered gap +4.1m, fused +4.26m, journal 0082/0087).
    We confirm the unified module does NOT regress it. Reuses the R01 cv_patrol_identity Stage-B
    track-silk sampler and the R01 RF/patrol constants; identity cached to cv_identity_R01.npz.
    """
    import os

    import cv_patrol_identity as PI  # noqa: N812
    import pyarrow.parquet as pq

    from cv_identity import build_rf_anchor

    print("\n" + "=" * 80)
    print("R01 REGRESSION — unified identity on the known-good race (banked +4.1m patrol gap)")
    print("=" * 80)
    # R01 field name->number (from the R01 patrol module's FIELD table)
    name2num = {v: k for k, v in PI.FIELD.items()}
    cache = "data_cache/cv_identity_R01.npz"
    if os.path.exists(cache):
        ident = Identity.load(cache)
        print(f"[identity-R01] loaded cache LUT {ident.known_numbers()}")
    else:
        ident = build_rf_anchor(
            "data_cache/video/full_20260603_01.mp4", name2num, rf_offset=1.34,
            fin_window=(96.5, 100.5, 0.08), legend_window=(4.0, 90.0, 2.0), field_size=12,
        )
        from cv_identity import augment_silk_lut_from_patrol
        ident = augment_silk_lut_from_patrol(
            ident, "data_cache/video/patrol_20260603_01.mp4",
            fin_window=(102.5, 123.0, 0.2), field_size=12)
        ident.save(cache)
        print(f"[identity-R01] cached -> {cache}")

    # R01 patrol track silks (reuse the R01 Stage-B sampler) + per-track depth order proxy
    cap = cv2.VideoCapture(PI.VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    panel = pq.read_table(PI.PANEL_IN).to_pydict()
    track_sig_raw = PI.headon_track_silks(cap, fps, panel)
    cap.release()
    track_sig = {ti: d["hist"] for ti, d in track_sig_raw.items()}

    # per-track depth (order proxy) + race_t from the R01 panel
    depth, rt = {}, {}
    for ti in track_sig:
        idx = [k for k in range(len(panel["track"])) if panel["track"][k] == ti]
        depth[ti] = float(np.median([panel["depth"][k] for k in idx]))
        rt[ti] = float(np.median([panel["t"][k] for k in idx])) - PI.PATROL_OFFSET

    for lam in (0.0, lam_order):
        assign = ident.propagate(track_sig, track_order=depth, track_race_t=rt, lam_order=lam)
        # off per track from the panel
        med_off = {}
        for ti, (num, _c) in assign.items():
            idx = [k for k in range(len(panel["track"])) if panel["track"][k] == ti]
            med_off[num] = float(np.median([panel["off_m"][k] for k in idx]))
        tight = {n: med_off[n] for n in PI.STEW_TIGHT if n in med_off}
        wide = {n: med_off[n] for n in PI.STEW_WIDE if n in med_off}
        gap = (np.mean(list(wide.values())) - np.mean(list(tight.values()))) \
            if (tight and wide) else float("nan")
        print(f"  lam={lam:.1f}: assigned {sorted(n for n, _ in assign.values())}; "
              f"tight {sorted(tight)}={np.mean(list(tight.values())):.1f}m "
              f"wide {sorted(wide)}={np.mean(list(wide.values())):.1f}m  "
              f"GAP {gap:+.1f}m (R01 banked +4.1m)")


def rerun_fusion_with_identity():
    """Re-run the R03 4-feed fusion with the IDENTITY-propagated patrol panel -> *_v2 trajectory."""
    import cv_r03_fusion as FUS  # noqa: N812

    FUS.PATROL_PANEL = PATROL_NUMBERED_ID
    FUS.F1.PATROL_PANEL = PATROL_NUMBERED_ID
    FUS.OUT_TRAJ = "data_cache/cv_fusion_trajectories_R03_v2.parquet"
    FUS.FIG_RECON = "data_cache/figs/cv_fusion_recon_R03_v2.png"
    print("\n" + "=" * 80)
    print("RE-RUN R03 4-FEED FUSION WITH UNIFIED IDENTITY (patrol panel -> identity-propagated)")
    print("=" * 80)
    FUS.run()


def make_qc(ident: Identity, off, assign, track_sig):
    """QC figure: silk swatch per LUT number; per-feed propagated identity; corrected lateral gap."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("data_cache/figs", exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # (a) silk hue-hist swatch per LUT number
    ax = fig.add_subplot(gs[0, 0])
    nums = ident.known_numbers()
    for i, n in enumerate(nums):
        hist = ident.silk_lut[n]
        # render the dominant-hue swatch from the histogram modes
        for b in range(len(hist)):
            hue = int((b + 0.5) * 180 / len(hist))
            bgr = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
            ax.barh(i, hist[b], left=hist[:b].sum(), height=0.8,
                    color=np.array([bgr[2], bgr[1], bgr[0]]) / 255.0)
    ax.set_yticks(range(len(nums)))
    ax.set_yticklabels([f"#{n} (v{ident.silk_votes.get(n, 0)})" for n in nums], fontsize=8)
    ax.set_xlabel("sat-weighted hue-hist fraction")
    ax.set_title(f"(a) silk LUT — {len(nums)}/12 (RF + patrol finish-cam, multi-frame voted)")
    ax.invert_yaxis()

    # (b) RF rank backbone (the order constraint)
    ax = fig.add_subplot(gs[0, 1])
    for n in range(1, ident.field_size + 1):
        ts, rks = [], []
        for t, order in ident.rank_seq:
            if n in order:
                ts.append(t)
                rks.append(order.index(n) + 1)
        if ts:
            ax.plot(ts, rks, "-o", ms=3, lw=1, label=f"#{n}")
    ax.set_xlabel("race_t (s)")
    ax.set_ylabel("RF legend rank (1=lead)")
    ax.set_title("(b) RF rank backbone — the shared-clock ORDER constraint")
    ax.invert_yaxis()
    ax.legend(ncol=4, fontsize=6)

    # (c) propagated patrol identity: lateral per number
    ax = fig.add_subplot(gs[1, 0])
    rail_t, wide_t = lateral_truth(off)
    df = pl.read_parquet(PATROL_NUMBERED_ID)
    med = {}
    for n in sorted(df["runner"].unique().to_list()):
        v = df.filter(pl.col("runner") == n)["y_lat_m"].to_numpy()
        med[n] = float(np.median(v))
        col = "tab:blue" if n in rail_t else ("tab:red" if n in wide_t else "0.5")
        ax.barh(n, med[n], color=col)
    ax.set_yticks(sorted(med))
    ax.set_xlabel("median y_lat (m from rail)")
    ax.set_ylabel("runner #")
    ax.set_title("(c) patrol propagated identity (blue=rail-truth, red=wide-truth)")
    ax.invert_yaxis()

    # (d) corrected lateral gap text
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    rv = [med[n] for n in rail_t if n in med]
    wv = [med[n] for n in wide_t if n in med]
    gap = (np.mean(wv) - np.mean(rv)) if (rv and wv) else float("nan")
    txt = ["UNIFIED IDENTITY (#92) — R03",
           f"silk LUT: {len(nums)}/12 {nums}",
           f"patrol assigned: {sorted(n for n, _ in assign.values())}",
           "",
           "patrol numbered rail-vs-wide GAP:",
           "   baseline (per-feed OCR): -5.7m (WRONG sign, 0088)",
           f"   identity (silk+order):   {gap:+.1f}m "
           f"({'CORRECT' if gap > 0 else 'still wrong'})",
           "",
           f"rail {sorted(rail_t & set(med))}: {np.mean(rv):.1f}m" if rv else "",
           f"wide {sorted(wide_t & set(med))}: {np.mean(wv):.1f}m" if wv else ""]
    ax.text(0.02, 0.98, "\n".join(t for t in txt if t), va="top", fontsize=11,
            family="monospace")
    fig.suptitle("cv_identity QC — solve-once silk LUT + RF order constraint, propagated", fontsize=13)
    fig.tight_layout()
    fig.savefig("data_cache/figs/cv_identity_qc.png", dpi=100)
    plt.close(fig)
    print("[fig] data_cache/figs/cv_identity_qc.png")


if __name__ == "__main__":
    _off = parse_official()
    _ident = Identity.load(IDENT_CACHE)
    print(f"[identity] loaded LUT {_ident.known_numbers()} ({len(_ident.silk_lut)}/12), "
          f"legend timepoints {len(_ident.rank_seq)}")
    # honesty sweep: pure silk (lam=0) vs silk+order constraint (lam=0.6).
    print("\n--- lam_order=0.0 (PURE SILK, no clock constraint) ---")
    df0, _, _ = propagate_patrol(_ident, lam_order=0.0)
    if df0 is not None:
        validate_patrol(df0, _off)
    print("\n--- lam_order=0.6 (SILK + RF clock-ORDER constraint) ---")
    _df, _assign, _sig = propagate_patrol(_ident, lam_order=0.6)
    if _df is not None:
        validate_patrol(_df, _off)
        make_qc(_ident, _off, _assign, _sig)
        rerun_fusion_with_identity()
    r01_regression()
