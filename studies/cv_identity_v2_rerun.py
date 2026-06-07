"""Identity v2 (#94) — the SINGLE FIXED CONSENSUS, re-run on BOTH R01 and R03.

Refines the #92 module (journal 0089). #92 corrected R03's patrol sign (-7.2 -> +4.4m) but
REGRESSED the known-good R01 (banked +4.1m -> -2.6/-9.5m): it REPLACED each feed's OWN finish-cam
silk LUT with an RF-only LUT (lead-group-biased, lower fidelity) AND the head-on DEPTH order-proxy
is weak/race-dependent.

THE FIX — identity as a CONSENSUS of 4 independent votes with FIXED weights (cv_identity.
consensus_assign; the SAME weights on R01 AND R03):
  1. the feed's OWN finish-cam silk LUT (restore per-feed strength; confidence-weighted by silk
     distinctness),
  2. the RF anchor LUT (cross-feed backbone),
  3. the shared-clock ORDER constraint (gated by rank reliability — only where unbunched),
  4. the DRAW ORDER (race-independent START lateral order, a strong order proxy).

PLUS the DRAW -> y_lat(t=0) FUSION ANCHOR (#93): a low-sd lateral observation at the gate-jump
(race_t~0) from draw_to_metres(), giving a SECOND metric anchor (start=draw, mid=patrol) to restore
the lateral magnitude diluted in #92.

VALIDATE — the bar is NO REGRESSION: same fixed scheme on BOTH races; R01 gap back to ~+4m (correct
sign, winner #3 preserved); R03 keeps the fix (gap >= +2.7m, winner #7 preserved). R08 HELD OUT.

DISCIPLINE: study-only; imports cv_identity / cv_r03_* / cv_patrol_identity READ-ONLY; src/hkjc
untouched; writes only NEW *_v2-suffixed parquets (banked R01/R03 parquets unmutated); official data
VALIDATION-only (draw is a pre-race INITIAL CONDITION, allowed as input — NOT an outcome).

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_identity_v2_rerun.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import polars as pl

sys.path.insert(0, "studies")

from cv_identity import (  # noqa: E402
    Identity,
    augment_silk_lut_from_patrol,
    draw_to_metres,
    silk_distinctness,
    silk_hist,
)

PATROL_NUMBERED_V2_R03 = "data_cache/cv_r03_patrol_numbered_v2.parquet"
PATROL_NUMBERED_V2_R01 = "data_cache/cv_patrol_numbered_v2.parquet"


# ============================================================================= own finish-cam LUTs
def _load_own_cache(path):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {int(n): h for n, h in zip(z["nums"], z["hists"], strict=True)}
    return None


def _save_own_cache(path, lut):
    nums = sorted(lut)
    np.savez(path, nums=np.array(nums, dtype=int),
             hists=np.array([lut[n] for n in nums]))


def own_lut_r03(ocr) -> dict[int, np.ndarray]:
    """R03 patrol's OWN finish-cam silk LUT (number->hue-hist). The patrol finish-cam is the
    feed's dedicated crisp side-on number source; build a patrol-ONLY LUT (NOT the RF-merged one)."""
    cache = "data_cache/cv_own_lut_R03_patrol.npz"
    c = _load_own_cache(cache)
    if c is not None:
        return c
    blank = Identity(field_size=12)
    blank = augment_silk_lut_from_patrol(
        blank, "data_cache/video/patrol_20260603_03.mp4",
        fin_window=(72.0, 100.0, 0.2), field_size=12, ocr=ocr)
    lut = dict(blank.silk_lut)
    _save_own_cache(cache, lut)
    return lut


def own_lut_r01(ocr) -> dict[int, np.ndarray]:
    """R01 patrol's OWN dedicated 102.5-123s finish-cam LUT (the one that EARNED +4.1m, 0082)."""
    import cv_patrol_identity as PI  # noqa: N812

    cache = "data_cache/cv_own_lut_R01_patrol.npz"
    c = _load_own_cache(cache)
    if c is not None:
        return c
    cap = cv2.VideoCapture(PI.VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    lut_raw = PI.build_lut(cap, fps, ocr)
    cap.release()
    lut = {n: e["hist"] for n, e in lut_raw.items() if e.get("hist") is not None}
    _save_own_cache(cache, lut)
    return lut


# ============================================================================= R03 patrol consensus
def patrol_tracks_r03():
    """Rebuild R03 patrol head-on tracks + per-track silk sig, lateral, depth, race_t."""
    import cv_r03_patrol as RP  # noqa: N812
    from cv_patrol_lateral import _turf_hue
    from cv_r03_patrol import (
        autodetect_headon,
        build_homography,
        calibrate_mpp,
        collect_detections,
        detect_jockeys,
        fit_fixed_rail_window,
        link_tracks,
        project,
    )

    cap = cv2.VideoCapture(RP.VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    t0, t1 = autodetect_headon(cap, fps)
    rail = fit_fixed_rail_window(cap, fps, t0, t1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mpp = calibrate_mpp(cap, fps, rail, t0, t1)
    hmat = build_homography(rail, h, w, mpp)
    rows = collect_detections(cap, fps, hmat, t0, t1)
    tracks = link_tracks(rows)
    print(f"[R03 patrol] head-on t={t0:.1f}-{t1:.1f}s, rail CV {rail.slope_cv * 100:.1f}%; "
          f"{len(tracks)} tracks")

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
        for j, p in zip(js, proj, strict=False):
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

    sig, lat, depth, rt = {}, {}, {}, {}
    for ti, pts in robust:
        if ti not in track_px:
            continue
        s = silk_hist(np.vstack(track_px[ti]))
        if s is None:
            continue
        sig[ti] = s
        lat[ti] = float(np.median([p[1] for p in pts]))
        depth[ti] = float(np.median([p[2] for p in pts]))
        rt[ti] = float(np.median([p[0] for p in pts])) - RP.PATROL_OFFSET
    return tracks, sig, lat, depth, rt, RP.PATROL_OFFSET


def patrol_tracks_r01():
    """R01 patrol head-on track silks/lateral/depth/race_t from the banked R01 panel + sampler."""
    import cv_patrol_identity as PI  # noqa: N812
    import pyarrow.parquet as pq

    cap = cv2.VideoCapture(PI.VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    panel = pq.read_table(PI.PANEL_IN).to_pydict()
    track_sig_raw = PI.headon_track_silks(cap, fps, panel)
    cap.release()
    sig = {ti: d["hist"] for ti, d in track_sig_raw.items()}
    lat, depth, rt = {}, {}, {}
    for ti in sig:
        idx = [k for k in range(len(panel["track"])) if panel["track"][k] == ti]
        lat[ti] = float(np.median([panel["off_m"][k] for k in idx]))
        depth[ti] = float(np.median([panel["depth"][k] for k in idx]))
        rt[ti] = float(np.median([panel["t"][k] for k in idx])) - PI.PATROL_OFFSET
    # track -> full (t, off) series for the numbered-panel write
    series: dict[int, list] = {}
    for k in range(len(panel["track"])):
        if panel["robust"][k]:
            series.setdefault(int(panel["track"][k]), []).append(
                (float(panel["t"][k]), float(panel["off_m"][k])))
    return series, sig, lat, depth, rt, PI.PATROL_OFFSET


# ============================================================================= one race
def run_race(race: str):
    if race == "R03":
        from cv_r03_common import (
            TRACK_WIDTH_M,
            lateral_truth,
            parse_official,
            race_t_to_arclen,
        )
        off = parse_official()
        rail_t, wide_t = lateral_truth(off)
        draw = dict(off.draw)
        cache = "data_cache/cv_identity_R03.npz"
        out_panel = PATROL_NUMBERED_V2_R03
        twidth = TRACK_WIDTH_M
        winner = off.finish_order[0]

        def arclen_fn(rt):
            return float(race_t_to_arclen(rt))
    else:
        import cv_patrol_identity as PI  # noqa: N812
        import re

        off = None
        # R01 stewards rail/wide + draw (validation only; draw is a pre-race input)
        rail_t = set(PI.STEW_TIGHT)
        wide_t = set(PI.STEW_WIDE)
        html = open("data_cache/race_official/20260603R1_results.html",
                    encoding="utf-8", errors="ignore").read()
        draw = {}
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            cells = [re.sub(r"<[^>]+>", " ", c).replace("&nbsp;", " ").strip()
                     for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
            if len(cells) >= 9 and cells[1].isdigit() and cells[7].isdigit():
                draw[int(cells[1])] = int(cells[7])
        cache = "data_cache/cv_identity_R01.npz"
        out_panel = PATROL_NUMBERED_V2_R01
        twidth = PI.TRACK_WIDTH if hasattr(PI, "TRACK_WIDTH") else 21.3
        winner = 3

        def arclen_fn(rt):
            return PI.race_t_to_arclen(rt)

    print("\n" + "=" * 80)
    print(f"IDENTITY v2 CONSENSUS (#94) — {race}")
    print("=" * 80)
    print(f"draw (1=rail): {dict(sorted(draw.items()))}")
    print(f"rail/tight truth {sorted(rail_t)} | wide truth {sorted(wide_t)} | winner #{winner}")

    ident = Identity.load(cache)
    ident.draw = draw
    print(f"[identity] RF anchor LUT {ident.known_numbers()} ({len(ident.silk_lut)}/12); "
          f"legend tp {len(ident.rank_seq)}")

    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except Exception:
        ocr = None

    own = own_lut_r03(ocr) if race == "R03" else own_lut_r01(ocr)
    distinct = silk_distinctness(own) if own else {}
    print(f"[own LUT] {race} patrol finish-cam own LUT {sorted(own)} ({len(own)}/12); "
          f"distinctness {{{', '.join(f'{n}:{distinct[n]:.2f}' for n in sorted(own))}}}")

    if race == "R03":
        tracks, sig, lat, depth, rt, poff = patrol_tracks_r03()
    else:
        series, sig, lat, depth, rt, poff = patrol_tracks_r01()

    assign = ident.consensus_assign(
        track_sig=sig, track_order=depth, track_lat=lat, track_race_t=rt,
        own_lut=own, own_distinct=distinct)
    print(f"[consensus] assigned runners {sorted(n for n, _ in assign.values())} "
          f"({len(assign)} tracks)")

    # write numbered patrol panel (v2)
    rows = []
    for ti, (num, conf) in assign.items():
        if race == "R03":
            for t, off_m, _dep in tracks[ti]:
                race_t = t - poff
                rows.append(dict(runner=int(num), race_t=round(race_t, 3),
                                 arclen=round(arclen_fn(race_t), 2),
                                 y_lat_m=round(off_m, 3), confidence=round(conf, 3)))
        else:
            for t, off_m in series.get(ti, []):
                race_t = t - poff
                rows.append(dict(runner=int(num), race_t=round(race_t, 3),
                                 arclen=round(float(arclen_fn(race_t)), 2),
                                 y_lat_m=round(off_m, 3), confidence=round(conf, 3)))
    df = pl.DataFrame(rows).sort(["runner", "race_t"]) if rows else None
    if df is not None:
        df.write_parquet(out_panel)
        print(f"[panel] {out_panel} ({df.height} rows, {df['runner'].n_unique()} runners)")

    # validate the patrol numbered rail-vs-wide gap (sign + magnitude)
    gap = float("nan")
    if df is not None:
        med = {n: float(np.median(df.filter(pl.col("runner") == n)["y_lat_m"].to_numpy()))
               for n in df["runner"].unique().to_list()}
        print("\n[VALIDATE patrol numbered vs stewards rail/wide]")
        for n in sorted(med):
            tag = "rail" if n in rail_t else ("wide" if n in wide_t else "mid")
            print(f"    #{n:2d} [{tag:4s}] {med[n]:6.2f}m")
        rv = [med[n] for n in rail_t if n in med]
        wv = [med[n] for n in wide_t if n in med]
        if rv and wv:
            gap = float(np.mean(wv) - np.mean(rv))
            banked = "+4.1m" if race == "R01" else "-5.7m(0088)/+4.4m(0089)"
            print(f"  GAP wide-rail = {gap:+.2f}m  (rail {sorted(rail_t & set(med))} "
                  f"{np.mean(rv):.1f}m, wide {sorted(wide_t & set(med))} {np.mean(wv):.1f}m)  "
                  f"[{race} banked {banked}]  SIGN "
                  f"{'CORRECT' if gap > 0 else 'WRONG'}")

    # draw -> metres calibration + cross-check vs patrol metric span
    d2m = draw_to_metres(draw, twidth)
    print(f"\n[draw->metres] stall_width {twidth / 14:.2f}m (={twidth}m/14 stalls); "
          f"y_lat(0) span {min(d2m.values()):.1f}-{max(d2m.values()):.1f}m")
    if df is not None and rows:
        pat_span = (df["y_lat_m"].min(), df["y_lat_m"].max())
        print(f"   cross-check: patrol metric field span {pat_span[0]:.1f}-{pat_span[1]:.1f}m "
              f"(draw map span {max(d2m.values()) - min(d2m.values()):.1f}m vs patrol "
              f"{pat_span[1] - pat_span[0]:.1f}m)")

    return dict(race=race, gap=gap, assign=assign, ident=ident, own=own, distinct=distinct,
                draw=draw, d2m=d2m, out_panel=out_panel, rail_t=rail_t, wide_t=wide_t,
                winner=winner)


# ============================================================================= fusion w/ draw anchor
def fusion_r03(res):
    """Re-run the R03 4-feed fusion with the v2 patrol panel + the DRAW -> y_lat(0) anchor."""
    import cv_r03_fusion as FUS  # noqa: N812
    from cv_fusion_4feed import Obs

    d2m = res["d2m"]
    # monkey-patch: inject draw anchor obs at race_t ~ 0 into the fusion's observation set by
    # wrapping load_patrol_r03 to append the gate-jump lateral anchors.
    FUS.PATROL_PANEL = res["out_panel"]
    FUS.F1.PATROL_PANEL = res["out_panel"]
    FUS.OUT_TRAJ = "data_cache/cv_fusion_trajectories_R03_v2c.parquet"
    FUS.FIG_RECON = "data_cache/figs/cv_fusion_recon_R03_v2c.png"

    orig_load_patrol = FUS.load_patrol_r03
    SD_DRAW = 1.5  # low-sd START lateral anchor (~half a stall); the 2nd metric anchor (#93)

    def load_patrol_with_draw():
        obs = orig_load_patrol()
        for n, y in d2m.items():
            obs.append(Obs(runner=int(n), race_t=0.0, arclen=0.0, ylat=float(y),
                           feed="draw_anchor", sd_arclen=999.0, sd_ylat=SD_DRAW))
        print(f"[draw anchor] added {len(d2m)} gate-jump y_lat(0) obs (sd {SD_DRAW}m)")
        return obs

    FUS.load_patrol_r03 = load_patrol_with_draw
    try:
        FUS.run()
    finally:
        FUS.load_patrol_r03 = orig_load_patrol


def fusion_r01(res):
    """Re-run the R01 4-feed fusion with the v2 patrol panel + the DRAW -> y_lat(0) anchor."""
    import cv_fusion_4feed as F1  # noqa: N812
    from cv_fusion_4feed import Obs

    d2m = res["d2m"]
    F1.PATROL_PANEL = res["out_panel"]
    F1.OUT_TRAJ = "data_cache/cv_fusion_trajectories_v2c.parquet"
    F1.FIG_RECON = "data_cache/figs/cv_fusion_recon_4feed_v2c.png"

    orig_load_patrol = F1.load_patrol
    SD_DRAW = 1.5

    def load_patrol_with_draw():
        obs = orig_load_patrol()
        for n, y in d2m.items():
            obs.append(Obs(runner=int(n), race_t=0.0, arclen=0.0, ylat=float(y),
                           feed="draw_anchor", sd_arclen=999.0, sd_ylat=SD_DRAW))
        print(f"[draw anchor] added {len(d2m)} gate-jump y_lat(0) obs (sd {SD_DRAW}m)")
        return obs

    F1.load_patrol = load_patrol_with_draw
    try:
        F1.run()
    finally:
        F1.load_patrol = orig_load_patrol


# ============================================================================= QC figure
def make_qc(res03, res01):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("data_cache/figs", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for col, res in enumerate((res01, res03)):
        race = res["race"]
        # (top) patrol numbered lateral per runner, colour-coded by truth
        ax = axes[0, col]
        df = pl.read_parquet(res["out_panel"])
        med = {}
        for n in sorted(df["runner"].unique().to_list()):
            v = df.filter(pl.col("runner") == n)["y_lat_m"].to_numpy()
            med[n] = float(np.median(v))
            c = "tab:blue" if n in res["rail_t"] else ("tab:red" if n in res["wide_t"] else "0.5")
            ax.barh(n, med[n], color=c)
        ax.set_yticks(sorted(med))
        ax.invert_yaxis()
        ax.set_xlabel("median y_lat (m from rail)")
        ax.set_ylabel("runner #")
        ax.set_title(f"({'a' if col == 0 else 'b'}) {race} patrol CONSENSUS identity "
                     f"(blue=rail, red=wide)  gap {res['gap']:+.2f}m")

        # (bottom) draw -> y_lat(0) map vs patrol metric
        ax = axes[1, col]
        d2m = res["d2m"]
        ns = sorted(d2m)
        ax.bar([str(n) for n in ns], [d2m[n] for n in ns], color="0.6", label="draw->metres (start)")
        ax.plot([str(n) for n in ns if n in med],
                [med[n] for n in ns if n in med], "rs", ms=7, label="patrol mid metric")
        ax.set_xlabel("runner #")
        ax.set_ylabel("y_lat (m from rail)")
        ax.set_title(f"({'c' if col == 0 else 'd'}) {race} draw->y_lat(0) anchor (#93) "
                     f"vs patrol mid")
        ax.legend(fontsize=8)

    fig.suptitle("cv_identity v2 (#94) — FIXED consensus identity + draw->y_lat(0) anchor; "
                 "R01 (no regression) | R03 (fix kept)", fontsize=13)
    fig.tight_layout()
    fig.savefig("data_cache/figs/cv_identity_v2_qc.png", dpi=100)
    plt.close(fig)
    print("[fig] data_cache/figs/cv_identity_v2_qc.png")


def qc_res(race: str) -> dict:
    """Rebuild the minimal res dict for the QC figure from the WRITTEN v2 panel (no re-sampling)."""
    if race == "R03":
        from cv_r03_common import TRACK_WIDTH_M, lateral_truth, parse_official
        off = parse_official()
        rail_t, wide_t = lateral_truth(off)
        draw = dict(off.draw)
        panel = PATROL_NUMBERED_V2_R03
        tw = TRACK_WIDTH_M
    else:
        import re
        import cv_patrol_identity as PI  # noqa: N812
        rail_t, wide_t = set(PI.STEW_TIGHT), set(PI.STEW_WIDE)
        html = open("data_cache/race_official/20260603R1_results.html",
                    encoding="utf-8", errors="ignore").read()
        draw = {}
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            cells = [re.sub(r"<[^>]+>", " ", c).replace("&nbsp;", " ").strip()
                     for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
            if len(cells) >= 9 and cells[1].isdigit() and cells[7].isdigit():
                draw[int(cells[1])] = int(cells[7])
        panel = PATROL_NUMBERED_V2_R01
        tw = 21.3
    df = pl.read_parquet(panel)
    med = {n: float(np.median(df.filter(pl.col("runner") == n)["y_lat_m"].to_numpy()))
           for n in df["runner"].unique().to_list()}
    rv = [med[n] for n in rail_t if n in med]
    wv = [med[n] for n in wide_t if n in med]
    gap = float(np.mean(wv) - np.mean(rv)) if rv and wv else float("nan")
    return dict(race=race, gap=gap, out_panel=panel, rail_t=rail_t, wide_t=wide_t,
                d2m=draw_to_metres(draw, tw))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["consensus", "fusion", "all", "qc"])
    ap.add_argument("--race", default="both", choices=["R01", "R03", "both"])
    args = ap.parse_args()

    if args.stage == "qc":
        make_qc(qc_res("R03"), qc_res("R01"))
        sys.exit(0)

    races = ["R01", "R03"] if args.race == "both" else [args.race]
    out = {}
    for r in races:
        out[r] = run_race(r)

    if args.stage in ("fusion", "all"):
        if "R01" in out:
            fusion_r01(out["R01"])
        if "R03" in out:
            fusion_r03(out["R03"])

    if "R01" in out and "R03" in out:
        make_qc(out["R03"], out["R01"])

    print("\n" + "=" * 80)
    print("IDENTITY v2 (#94) SUMMARY — same FIXED consensus on both races")
    print("=" * 80)
    for r in races:
        print(f"  {r}: patrol consensus gap {out[r]['gap']:+.2f}m  "
              f"({'CORRECT' if out[r]['gap'] > 0 else 'WRONG'} sign)  "
              f"assigned {len(out[r]['assign'])} tracks")
