"""S0d AERIAL #78 — extend the genuine recon THROUGH the true finish + bind 12/12 +
validate the lateral (y_lat) trip QUANTITATIVELY (journal 0069's reframing of 0068).

0068 banked the render-adaptive geometry (green-render detect, scale_from_sprites, ocr_every,
"C"-width) but its recon DROPPED the field ~140m before the line (~t=91s), so the winner #3's
7th->1st surge was never tracked, only READ off the legend; and it bound only 7/12. Worse, the
legend OCR went EMPTY from ~t=72s (the number-box overlay row drifts down toward the run-in and
the single ``find_layout`` band lost it). The library fix (``_read_legend_robust``) sweeps
fallback bands so the legend running-order is recovered through the FINISH (t=100, "10M").

This driver:
  1. runs the FULL video (no early stop) so the recon reaches the true finish (~t=100s),
  2. binds runners to saddlecloth numbers via the (now-robust) legend anchor,
  3. cross-checks the binding against the START-GATE silk anchor (stalls 1-13 @ t~1s) +
     the official draw->saddle map (re-id ANCHOR, never a model input),
  4. validates the y_lat TRIP quantitatively against the official running-position trajectory
     (rail-saved vs wide), de-confounded per PHILOSOPHY 4b, across ALL bound runners,
  5. reads section times off the LEFT-panel broadcast clock (finer than the 400m odometer grid).

Writes data_cache/cv_s0d_aerial.parquet (overwrites 0068's). Disposable scratch; reads the
video + official HTML + ``hkjc.core.track`` READ-ONLY; ``src/hkjc/model|core`` UNTOUCHED.

    PYTHONHASHSEED=0 uv run --group vision python studies/cv_s0d_aerial_finish.py
"""
from __future__ import annotations

import html
import os
import re

import numpy as np
import polars as pl

AERIAL_VIDEO = "data_cache/video/aerial_20260603_01.mp4"
RESULTS_HTML = "data_cache/race_official/20260603R1_results.html"
OUT_AERIAL = "data_cache/cv_s0d_aerial.parquet"
DIST = 1650.0
OFFICIAL_SECTIONS = [27.65, 23.30, 23.75, 24.15]

CTX = dict(
    venue="ST", date="2026-06-03", race_no=1, distance_m=DIST,
    surface="turf", course="C", field_size=12,
)


def parse_official(path: str) -> dict:
    t = open(path, encoding="utf-8", errors="ignore").read()
    rest = t[t.find("RACE 1 (745)"):]
    out: dict = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", rest, re.S):
        cc = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(x))).strip()
              for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cc) == 12 and re.match(r"^\d+$", cc[0]) and re.match(r"^\d+$", cc[1]):
            out[int(cc[1])] = dict(pla=int(cc[0]), horse=cc[2], dr=int(cc[7]),
                                   rp=[int(x) for x in cc[9].split() if x.isdigit()])
    return out


# ----------------------------------------------------------------- start-gate silk anchor
def gate_silk_anchor(video_path: str, official: dict, *, field_size: int = 13) -> dict:
    """Independent silk->number anchor from the REAL start-gate camera (~t=1s).

    The opening frame is a real broadcast gate shot: a row of numbered stalls (1-13 = DRAW
    positions) with the silks visible. We locate the stall band, split it into ``field_size``
    equal cells, take each cell's dominant chromatic silk LAB, and map stall(=draw) ->
    saddlecloth via the official draw map. This is a RE-ID ANCHOR (an external silk<->number
    binding to cross-check the legend re-id), never a model input. Returns
    {saddle#: silk_lab} for the runners whose stall we could read.
    """
    import av
    import cv2

    draw2no = {official[n]["dr"]: n for n in official}
    c = av.open(video_path)
    fps = float(c.streams.video[0].average_rate)
    gate = None
    for i, fr in enumerate(c.decode(video=0)):
        if i / fps >= 1.0:
            gate = fr.to_ndarray(format="bgr24")
            break
    c.close()
    if gate is None:
        return {}
    h, w = gate.shape[:2]
    # the stall row sits in the mid-band; the silks are the saturated patches just below the
    # numbered gate header. Use a horizontal strip across the gate and split into cells.
    y0, y1 = int(0.40 * h), int(0.58 * h)
    x0, x1 = int(0.10 * w), int(0.95 * w)
    strip = gate[y0:y1, x0:x1]
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    cw = strip.shape[1] / field_size
    out: dict = {}
    for stall in range(1, field_size + 1):
        cx0 = int((stall - 1) * cw)
        cx1 = int(stall * cw)
        cell_lab = lab[:, cx0:cx1].reshape(-1, 3).astype(float)
        cell_sat = sat[:, cx0:cx1].reshape(-1)
        chroma = np.abs(cell_lab[:, 1] - 128) + np.abs(cell_lab[:, 2] - 128)
        sel = (cell_sat > 60) & (chroma > 18)
        if sel.sum() < 20:
            continue
        no = draw2no.get(stall)
        if no is None:
            continue
        out[no] = tuple(float(v) for v in cell_lab[sel].mean(axis=0))
    return out


# --------------------------------------------------------------- y_lat trip validation 4b
def validate_ylat(df: pl.DataFrame, official: dict) -> dict:
    """Quantitatively validate the rail-relative y_lat TRIP against the official RP trajectory.

    Mechanism (the de-confound, PHILOSOPHY 4b): a rail-hugging trip saves ground; a wide trip
    loses it. The cleanest external proxy for ground-position is each runner's MEAN running
    position (1=lead) over the 4 official poles: front-runners ride the rail (low y_lat); horses
    that race wide/midfield carry more lateral metres. We correlate, ACROSS ALL BOUND RUNNERS,
    the CV median y_lat against (a) the mean official RP and (b) the early-RP (the trip is set
    early; the finish position confounds with late closing). We report Pearson + Spearman + n,
    and a leave-one-out robustness sweep (is it a 3-horse anecdote or a field-wide signal?).
    Honest 4b read: ceiling here is the structural ground-position content of y_lat; if it does
    NOT track RP robustly we say so with the de-confounded reason (synthetic-sprite collapse).
    """
    runners = sorted(df["runner"].unique().to_list())
    rows = []
    for r in runners:
        if r not in official:
            continue
        s = df.filter(pl.col("runner") == r)
        rp = official[r]["rp"]
        rows.append(dict(
            no=r,
            ylat_med=float(s["y_lat"].median()),
            ylat_early=float(s.filter(pl.col("x_arclen") < 0.6 * DIST)["y_lat"].median())
            if s.filter(pl.col("x_arclen") < 0.6 * DIST).height else float("nan"),
            rp_mean=float(np.mean(rp)),
            rp_early=float(np.mean(rp[:2])) if len(rp) >= 2 else float(rp[0]),
            place=official[r]["pla"],
            n=s.height,
        ))
    if len(rows) < 3:
        return dict(n=len(rows), note="too few bound runners")

    def corr(xs, ys):
        xs, ys = np.array(xs, float), np.array(ys, float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() < 3 or xs[m].std() < 1e-9 or ys[m].std() < 1e-9:
            return float("nan"), float("nan")
        pear = float(np.corrcoef(xs[m], ys[m])[0, 1])

        def rank(a):
            o = np.argsort(a)
            rr = np.empty_like(o, float)
            rr[o] = np.arange(len(a))
            return rr
        spear = float(np.corrcoef(rank(xs[m]), rank(ys[m]))[0, 1])
        return pear, spear

    yl = [r["ylat_med"] for r in rows]
    yle = [r["ylat_early"] for r in rows]
    rpm = [r["rp_mean"] for r in rows]
    rpe = [r["rp_early"] for r in rows]
    # leave-one-out on the headline (median y_lat vs mean RP)
    loo = []
    for k in range(len(rows)):
        idx = [j for j in range(len(rows)) if j != k]
        p, _ = corr([yl[j] for j in idx], [rpm[j] for j in idx])
        loo.append(p)
    pear_med, spear_med = corr(yl, rpm)
    pear_early, spear_early = corr(yle, rpe)
    return dict(
        n=len(rows), rows=rows,
        pearson_med_vs_rpmean=pear_med, spearman_med_vs_rpmean=spear_med,
        pearson_early_vs_rpearly=pear_early, spearman_early_vs_rpearly=spear_early,
        loo_pearson=[round(x, 2) for x in loo],
        loo_min=float(np.nanmin(loo)), loo_max=float(np.nanmax(loo)),
    )


def main() -> None:
    from hkjc.vision.mode_b import extract_lateral_paths
    from hkjc.vision.mode_b.pipeline_b import rows_to_dicts

    official = parse_official(RESULTS_HTML)
    off_winner = next(n for n in official if official[n]["pla"] == 1)
    print("=" * 78)
    print("S0d AERIAL #78 — recon THROUGH the finish + 12/12 bind + y_lat validation")
    print("=" * 78)
    print(f"OFFICIAL winner #{off_winner} {official[off_winner]['horse']} (rp {official[off_winner]['rp']})")
    print("OFFICIAL finish: " + ", ".join(
        f"{official[n]['pla']}:#{n}" for n in sorted(official, key=lambda n: official[n]['pla'])))

    # ---- start-gate silk anchor -------------------------------------------------------
    anchor = gate_silk_anchor(AERIAL_VIDEO, official)
    print(f"\n[gate anchor] silk LAB read for {len(anchor)} runners (stall=draw -> saddle):")
    print("  " + ", ".join(f"#{n}" for n in sorted(anchor)))

    # ---- FULL-video recon (no early stop) ---------------------------------------------
    # NB: section times come from the recon itself (it reads the LEFT-panel clock + the
    # odometer in the same single pass via ``read_clock_splits`` on ``clock_roi``), so we do
    # NOT run a separate clock pass — that just triples the decode/OCR cost.
    stride = int(os.environ.get("S0D_AERIAL_STRIDE", "5"))
    ocr_every = int(os.environ.get("S0D_OCR_EVERY", "2"))
    print(f"\n[recon] FULL video, stride={stride}, ocr_every={ocr_every} ...")
    res = extract_lateral_paths(AERIAL_VIDEO, stride=stride, ocr_every=ocr_every, **CTX)
    d = res.diagnostics
    for k in ("frames_aerial_active", "scale_source", "scale_m_per_px", "reid_mode",
              "reid_purity_median", "legend_frames", "runners_bound", "arclen_source",
              "section_time_mae_s", "cv_finish_order"):
        print(f"  {k:22s}: {d.get(k)}")
    # section times off the LEFT-panel clock (read in the same pass)
    cvsec = d.get("cv_section_times") or []
    pub = d.get("published_splits") or []
    print(f"\n[left-panel clock] CV section times : {cvsec}")
    print(f"  video published splits            : {pub}")
    print(f"  official sections                 : {OFFICIAL_SECTIONS}")
    if cvsec:
        k = min(len(cvsec), len(OFFICIAL_SECTIONS))
        mae = float(np.mean([abs(cvsec[-k + j] - OFFICIAL_SECTIONS[-k + j]) for j in range(k)]))
        print(f"  MAE vs official on last {k} sections: {mae:.2f}s")

    df = pl.DataFrame(rows_to_dicts(res.rows)) if res.rows else pl.DataFrame()
    if not df.height:
        print("  NO ROWS — abort")
        return
    os.makedirs(os.path.dirname(OUT_AERIAL), exist_ok=True)
    df.write_parquet(OUT_AERIAL)
    print(f"  wrote {OUT_AERIAL} ({df.height} rows)")

    bound = sorted(df["runner"].unique().to_list())
    print(f"\n[binding] {len(bound)}/12 bound: {bound}")
    # winner #3 tracked through the finish?
    s3 = df.filter(pl.col("runner") == 3).sort("t")
    if s3.height:
        print(f"  winner #3 tracked to x_arclen={s3['x_arclen'].max():.0f}m "
              f"(t={s3['t'].max():.1f}s)  -> {'REACHES finish' if s3['x_arclen'].max() > DIST - 60 else 'STOPS short'}")

    # ---- y_lat validation 4b ----------------------------------------------------------
    v = validate_ylat(df, official)
    print("\n[y_lat TRIP validation — 4b de-confounded, ALL bound runners]")
    if v.get("rows"):
        print("  per-runner (sorted by official RP-mean, rail-saved -> wide expected):")
        for r in sorted(v["rows"], key=lambda r: r["rp_mean"]):
            print(f"    #{r['no']:>2}  rp_mean {r['rp_mean']:4.1f}  place {r['place']:>2}  "
                  f"y_lat_med {r['ylat_med']:5.1f}m  early {r['ylat_early']:5.1f}m  n={r['n']}")
        print(f"  median y_lat vs mean RP:  Pearson {v['pearson_med_vs_rpmean']:+.2f}  "
              f"Spearman {v['spearman_med_vs_rpmean']:+.2f}  (n={v['n']})")
        print(f"  early y_lat vs early RP:  Pearson {v['pearson_early_vs_rpearly']:+.2f}  "
              f"Spearman {v['spearman_early_vs_rpearly']:+.2f}")
        print(f"  leave-one-out Pearson range: [{v['loo_min']:+.2f}, {v['loo_max']:+.2f}]  "
              f"{v['loo_pearson']}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
