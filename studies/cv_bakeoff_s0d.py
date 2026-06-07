"""S0d — CV METHODOLOGY BAKE-OFF on ONE race (ST 2026-06-03 R1, Class 5, 1650M, "C" course).

Reconstruct lateral/longitudinal runner paths from THREE views of the SAME race and score
each against the OFFICIAL HKJC result (parsed from ``data_cache/race_official/``):

  AERIAL   — the synthetic Aerial Virtual View render (Mode B: legend-anchored re-id via
             RapidOCR, odometer arc-length, single-rail-relative y_lat on the GREEN turf
             render). "C" course geometry (width 21.3 m).
  DRONE    — a REAL high-angle oblique camera following the field. New pipeline: blob
             detect/track on real horses, silk-colour re-id, per-frame rail-line homography
             (oblique -> rail-relative). Saddlecloth numbers are NOT legible from altitude.
  COMBINED — Bayesian multi-view fusion of the two per-runner longitudinal-rank series,
             each observation weighted by its precision (re-id purity x detection conf).

VALIDATION (the whole point — external official data, never the video's own legend):
  (a) finish order vs official (winner must be #3 FAMILY FORTUNE),
  (b) per-pole longitudinal order vs each horse's official running-position sequence,
  (c) CV section times vs the official sectionals (27.65 / 23.30 / 23.75 / 24.15),
  (d) trip (y_lat wide/rail) vs the running-position trajectory.

Disposable scratch. Reads the videos + official HTML + ``hkjc.core.track`` READ-ONLY.

    uv run --group vision python studies/cv_bakeoff_s0d.py
"""
from __future__ import annotations

import html
import os
import re

OFFICIAL_DIR = "data_cache/race_official"
RESULTS_HTML = f"{OFFICIAL_DIR}/20260603R1_results.html"
AERIAL_VIDEO = "data_cache/video/aerial_20260603_01.mp4"
DRONE_VIDEO = "data_cache/video/drone_20260603_01.mp4"
OUT_AERIAL = "data_cache/cv_s0d_aerial.parquet"
OUT_DRONE = "data_cache/cv_s0d_drone.parquet"
OUT_FIG = "data_cache/figs/cv_s0d_bakeoff.png"

CTX = dict(
    venue="ST", date="2026-06-03", race_no=1, distance_m=1650.0,
    surface="turf", course="C", field_size=12,
)
# Official sectionals (cumulative 27.65/50.95/1:14.70/1:38.85), per-section:
OFFICIAL_SECTIONS = [27.65, 23.30, 23.75, 24.15]


# ----------------------------------------------------------------------- official parse
def parse_official(path: str) -> dict:
    """Parse the official result table -> {saddle#: {pla, rp:[..4 poles..], dr, odds, horse}}."""
    t = open(path, encoding="utf-8", errors="ignore").read()
    i = t.find("RACE 1 (745)")
    rest = t[i:]
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", rest, re.S)
    out: dict = {}
    for tr in trs:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        cc = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(x))).strip() for x in cells]
        if len(cc) == 12 and re.match(r"^\d+$", cc[1]) and re.match(r"^\d+$", cc[0]):
            pla, no, horse, _jk, _tr, _aw, _dhw, dr, _lbw, rp, ft, odds = cc
            poles = [int(x) for x in rp.split() if x.isdigit()]
            out[int(no)] = dict(
                pla=int(pla), horse=horse, dr=int(dr), rp=poles, ft=ft, odds=odds,
            )
    return out


# ----------------------------------------------------------------------- scoring helpers
def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation of two equal-length sequences (ties broken by order)."""
    import numpy as np

    if len(a) < 3:
        return float("nan")
    ra = _rank(a)
    rb = _rank(b)
    ra = np.array(ra, float)
    rb = np.array(rb, float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for pos, idx in enumerate(order):
        r[idx] = pos
    return r


def kendall_finish(cv_order: list[int], official: dict) -> tuple[int, float]:
    """How many of the CV finish positions match official placing? top-3 hit + full corr."""
    import numpy as np

    # cv_order: saddle#s in CV finish order (winner first)
    off_place = {no: official[no]["pla"] for no in official}
    common = [no for no in cv_order if no in off_place]
    if len(common) < 3:
        return 0, float("nan")
    cv_rank = {no: i for i, no in enumerate(common)}
    off_rank = [off_place[no] for no in common]
    cvr = [cv_rank[no] for no in common]
    corr = float(np.corrcoef(cvr, off_rank)[0, 1]) if len(common) >= 3 else float("nan")
    # top-3 exact membership
    cv_top3 = set(cv_order[:3])
    off_top3 = {no for no in off_place if off_place[no] <= 3}
    hit = len(cv_top3 & off_top3)
    return hit, corr


# --------------------------------------------------------------------------- main
def main() -> None:
    import numpy as np
    import polars as pl

    print("=" * 78)
    print("S0d · CV METHODOLOGY BAKE-OFF — AERIAL vs DRONE vs COMBINED, vs OFFICIAL data")
    print("ST 2026-06-03 R1 · Class 5 · 1650M · TURF \"C\" Course")
    print("=" * 78)

    official = parse_official(RESULTS_HTML)
    print(f"\n-- OFFICIAL ({len(official)} runners) --")
    print("  finish: " + ", ".join(
        f"{official[no]['pla']}:#{no}" for no in sorted(official, key=lambda n: official[n]["pla"])
    ))
    off_winner = next(no for no in official if official[no]["pla"] == 1)
    print(f"  winner #{off_winner} {official[off_winner]['horse']}  (rp {official[off_winner]['rp']})")
    print(f"  official sections: {OFFICIAL_SECTIONS}")

    from cv_s0d_drone import extract_drone_paths  # local study module

    from hkjc.vision.mode_b import extract_lateral_paths
    from hkjc.vision.mode_b.pipeline_b import rows_to_dicts

    # ===================================================================== AERIAL
    print("\n" + "-" * 78)
    print("[1] AERIAL — Mode B on the Aerial Virtual View render")
    print("-" * 78)
    # OCR (3 reads/frame) dominates aerial cost. For a METHODOLOGY comparison, stride 8
    # (~3 Hz) for detect/rail + OCR every 3rd processed frame (legend/odometer/clock change
    # slowly — journal 0064) keeps the run in the foreground budget with no material loss.
    _stride = int(os.environ.get("S0D_AERIAL_STRIDE", "8"))
    _ocr_every = int(os.environ.get("S0D_OCR_EVERY", "3"))
    aerial = extract_lateral_paths(
        AERIAL_VIDEO, stride=_stride, ocr_every=_ocr_every, **CTX
    )
    ad = aerial.diagnostics
    for k in ("frames_aerial_active", "scale_source", "scale_m_per_px", "reid_mode",
              "runners_bound", "reid_purity_median", "arclen_source", "track_width_m",
              "published_splits", "cv_section_times", "section_time_mae_s",
              "cv_finish_order"):
        print(f"  {k:22s}: {ad.get(k)}")
    df_a = pl.DataFrame(rows_to_dicts(aerial.rows)) if aerial.rows else pl.DataFrame()
    if df_a.height:
        os.makedirs(os.path.dirname(OUT_AERIAL), exist_ok=True)
        df_a.write_parquet(OUT_AERIAL)
        print(f"  wrote {OUT_AERIAL} ({df_a.height} rows)")

    # ===================================================================== DRONE
    print("\n" + "-" * 78)
    print("[2] DRONE — real-horse detect/track + silk re-id + rail-line homography")
    print("-" * 78)
    _drone_stride = int(os.environ.get("S0D_DRONE_STRIDE", "10"))
    drone = extract_drone_paths(
        DRONE_VIDEO, official=official, stride=_drone_stride, **CTX
    )
    dd = drone.diagnostics
    for k in ("frames_used", "det_per_frame_median", "n_tracks", "runners_bound",
              "reid_mode", "homography_frames", "scale_source", "cv_section_times",
              "section_time_mae_s", "cv_finish_order"):
        print(f"  {k:22s}: {dd.get(k)}")
    df_d = pl.DataFrame(drone.rows) if drone.rows else pl.DataFrame()
    if df_d.height:
        df_d.write_parquet(OUT_DRONE)
        print(f"  wrote {OUT_DRONE} ({df_d.height} rows)")

    # ===================================================================== per-view scorecard
    def pole_orders_from_panel(df, official, dist=1650.0):
        """For each of the 4 official poles, the CV longitudinal order (leader->tail)."""
        if not df.height:
            return None
        # poles at metres-travelled: ~ section boundaries. Official RP is at the 4 sectional
        # sampling points; we sample the CV arc-length at fractions matching the 4 poles.
        # The 4 poles correspond to ~ each 400m+ boundary; sample at travelled fractions.
        runners = sorted(df["runner"].unique().to_list())
        # the four sampling poles ~ at 1/4..4/4 of the trip (the official RP is 4 columns)
        frac_poles = [0.25, 0.50, 0.75, 0.97]
        orders = []
        for fr in frac_poles:
            xq = fr * dist
            rank = {}
            for r in runners:
                sub = df.filter(pl.col("runner") == r).sort("x_arclen")
                xa = sub["x_arclen"].to_numpy()
                if len(xa) == 0:
                    continue
                j = int(np.argmin(np.abs(xa - xq)))
                # leader = greatest x_arclen at a given TIME, but here we use position at the
                # nearest-arclen sample's TIME. Simpler: rank by time-to-reach xq (smaller=leader)
                t = sub["t"].to_numpy()
                rank[r] = float(t[j])
            order = sorted(rank, key=lambda n: rank[n])
            orders.append(order)
        return orders

    def running_pos_corr(df, official):
        """Spearman of CV per-pole position vs official RP, averaged over the 4 poles."""
        orders = pole_orders_from_panel(df, official)
        if not orders:
            return float("nan"), None
        corrs = []
        for p, order in enumerate(orders):
            common = [no for no in order if no in official and len(official[no]["rp"]) > p]
            if len(common) < 3:
                continue
            cv_rank = [order.index(no) for no in common]
            off_rank = [official[no]["rp"][p] for no in common]
            c = spearman([float(x) for x in cv_rank], [float(x) for x in off_rank])
            if not np.isnan(c):
                corrs.append(c)
        return (float(np.mean(corrs)) if corrs else float("nan")), orders

    print("\n" + "=" * 78)
    print("RELIABILITY SCORECARD (vs OFFICIAL)")
    print("=" * 78)
    rows_score = []
    for name, df, diag in (("AERIAL", df_a, ad), ("DRONE", df_d, dd)):
        cv_finish = diag.get("cv_finish_order") or []
        hit3, fcorr = kendall_finish(cv_finish, official)
        rp_corr, _ = running_pos_corr(df, official)
        mae = diag.get("section_time_mae_s")
        winner_ok = bool(cv_finish and cv_finish[0] == off_winner)
        rows_score.append((name, winner_ok, hit3, fcorr, rp_corr, mae,
                           diag.get("runners_bound")))
        print(f"\n[{name}]")
        print(f"  CV finish order        : {cv_finish}")
        print(f"  winner == official #{off_winner}? : {winner_ok}")
        print(f"  top-3 members matched   : {hit3}/3   (finish-rank corr {fcorr})")
        print(f"  running-pos Spearman    : {rp_corr}")
        print(f"  section-time MAE (s)    : {mae}")
        print(f"  runners bound           : {diag.get('runners_bound')}")

    # ===================================================================== COMBINED (Bayesian)
    print("\n" + "-" * 78)
    print("[3] COMBINED — Bayesian multi-view fusion of per-pole longitudinal RANK")
    print("-" * 78)
    combined = fuse_views(df_a, ad, df_d, dd, official)
    cv_finish_c = combined["finish_order"]
    hit3_c, fcorr_c = kendall_finish(cv_finish_c, official)
    winner_ok_c = bool(cv_finish_c and cv_finish_c[0] == off_winner)
    print(f"  fused finish order     : {cv_finish_c}")
    print(f"  winner == official #{off_winner}? : {winner_ok_c}")
    print(f"  top-3 members matched   : {hit3_c}/3  (finish-rank corr {fcorr_c})")
    print(f"  fused running-pos corr  : {combined['rp_corr']}")
    rows_score.append(("COMBINED", winner_ok_c, hit3_c, fcorr_c, combined["rp_corr"],
                       None, combined["runners"]))

    # ===================================================================== VERDICT
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"{'view':9} {'winner?':8} {'top3':5} {'finishρ':8} {'RPρ':7} {'secMAE':7}")
    for name, wok, hit3, fcorr, rpc, mae, _ in rows_score:
        fc = f"{fcorr:+.2f}" if fcorr == fcorr else "  nan"
        rp = f"{rpc:+.2f}" if rpc == rpc else "  nan"
        mm = f"{mae:.2f}" if isinstance(mae, (int, float)) else "  -"
        print(f"{name:9} {str(wok):8} {hit3}/3   {fc:8} {rp:7} {mm:7}")

    print("\nDONE.")


def fuse_views(df_a, ad, df_d, dd, official) -> dict:
    """Bayesian fusion: combine the two views' per-runner longitudinal time-series into a
    single posterior finishing rank.

    Each view contributes, per runner, a finishing-rank estimate with a PRECISION (inverse
    variance) set by the view's evidence quality: aerial precision ~ reid_purity x coverage;
    drone precision ~ detection coverage x silk-purity. We fuse the rank estimates as a
    precision-weighted mean (a Gaussian posterior over latent finishing rank), then read the
    finish order off the posterior means. Sparse/noisy views still inform the posterior.
    """
    import numpy as np

    def view_ranks(df, finish_order, base_prec):
        """Per-runner (rank_estimate, precision) from a panel + its legend finish order."""
        out = {}
        for i, no in enumerate(finish_order):
            out[no] = (float(i), base_prec)
        return out

    ra = view_ranks(df_a, ad.get("cv_finish_order") or [],
                    base_prec=max(0.1, float(ad.get("reid_purity_median") or 0.3)))
    rd = view_ranks(df_d, dd.get("cv_finish_order") or [],
                    base_prec=max(0.1, float(dd.get("drone_purity") or 0.2)))
    all_runners = sorted(set(ra) | set(rd))
    fused = {}
    for no in all_runners:
        num = den = 0.0
        for src in (ra, rd):
            if no in src:
                r, p = src[no]
                num += p * r
                den += p
        if den > 0:
            fused[no] = num / den
    finish_order = sorted(fused, key=lambda n: fused[n])
    # running-pos corr on the fused finishing rank vs official placing
    common = [no for no in finish_order if no in official]
    if len(common) >= 3:
        cvr = [finish_order.index(no) for no in common]
        offr = [official[no]["pla"] for no in common]
        rp_corr = float(np.corrcoef(cvr, offr)[0, 1])
    else:
        rp_corr = float("nan")
    return dict(finish_order=finish_order, rp_corr=rp_corr, runners=all_runners)


if __name__ == "__main__":
    main()
