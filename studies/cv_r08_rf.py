"""R08 RF / race-film feed — REUSE cv_rf_longitudinal (#89/0086) on HV R08 1800m route (#95).

HELD-OUT generalization. Imports the R1 RF functions unchanged (measure_margins /
read_legend_order / make_qc) and re-points ONLY race PARAMETERS at R08:
  * name->number LUT REBUILT from the R08 official parse (race-agnostic).
  * sections 4->5 (1800m route) -> SEC_T/SEC_S re-derived (5-call boundaries 0/200/600/1000/1400/1800).
  * RF_OFFSET re-anchored (winner crosses later — the route is ~108.6s vs the sprint's 69.3s).
  * finish window re-found for the route.

NO algorithm tuned for R08; the finish-window time is a race PARAMETER (clock anchor), auto/
frame-derived like R03's was. Writes data_cache/cv_r08_rf_longitudinal.parquet. Official data
VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_r08_rf.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "studies")
import cv_rf_longitudinal as RF1  # noqa: E402,N812  (R1 RF toolkit, reused)
from cv_r08_common import (  # noqa: E402
    DIST,
    OFFICIAL_SECTIONS,
    WINNER_FINISH_RACE_T,
    _D_BOUNDS,
    parse_official,
)

VIDEO = "data_cache/video/full_20260603_08.mp4"
PANEL_OUT = "data_cache/cv_r08_rf_longitudinal.parquet"
FIG = "data_cache/figs/cv_r08_rf_qc.png"

# Clock anchor (R08): the broadcast lead-in. R01 RF_OFFSET 1.34s, R03 2.71s; the route's winner
# crosses near vid ~ WINNER_FINISH_RACE_T + offset. We frame-anchor the winner-cross below
# (the offset is a clock PARAMETER, not an algorithm change).
WINNER_CROSS_VID = 109.7   # frame-anchored winner #6 line-cross on the 150.7s RF feed (route);
# verified visually: #6 surges to the line ~vid 109.5, field past the post by 110.0.


def _setup_r08(off) -> None:
    """[GENERALIZE] Re-point every R1-hard-coded RF constant at R08 (race parameters only)."""
    RF1.VIDEO = VIDEO
    RF1.PANEL_OUT = PANEL_OUT
    RF1.FIG = FIG
    RF1.FIELD = dict(off.horse)
    RF1.NAME2NUM = {v.replace(" ", "").upper(): k for k, v in off.horse.items()}
    RF1.OFFICIAL_ORDER = list(off.finish_order)
    RF1.OFFICIAL_LBW = dict(off.lbw)
    # sections 4 -> 5 (1800m route). Calls at 0/200/600/1000/1400/1800m.
    RF1.SEC_T = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])
    RF1.SEC_S = _D_BOUNDS
    RF1.RF_OFFSET = WINNER_CROSS_VID - WINNER_FINISH_RACE_T
    RF1.WINNER_CROSS_VID = WINNER_CROSS_VID
    # finish window: the legible APPROACH to the line (last ~2.7s before the cross), the R03-hardened
    # window strategy (the finish cam blurs the instant the field crosses).
    RF1.FIN_T0, RF1.FIN_T1, RF1.FIN_DT = WINNER_CROSS_VID - 2.7, WINNER_CROSS_VID + 0.3, 0.04
    RF1.OFFICIAL_TIME = {off.finish_order[0]: WINNER_FINISH_RACE_T}


def main():
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    off = parse_official()
    _setup_r08(off)
    print("=" * 80)
    print("R08 RF #95 — race-film longitudinal+margins on HV 1800m ROUTE (held out)")
    print("=" * 80)
    print(f"[GENERALIZE] RF_OFFSET={RF1.RF_OFFSET:.2f}s; finish window "
          f"{RF1.FIN_T0:.1f}-{RF1.FIN_T1:.1f}s; winner #{off.finish_order[0]}; sections "
          f"{len(OFFICIAL_SECTIONS)} (route)")

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ocr = RapidOCR()

    print("\n=== STEP 1/2 — finish-window detection + saddlecloth OCR + margins ===")
    frame_recs = RF1.measure_margins(cap, fps, ocr)
    seen = set()
    for _, numcx, _ in frame_recs:
        seen |= set(numcx)
    print(f"finish-window frames used: {len(frame_recs)}; numbers OCR'd: {sorted(seen)} "
          f"({len(seen)}/{len(off.finish_order)})")

    winner = off.finish_order[0]
    margins = _margins_rel_winner(frame_recs, winner)
    ordered = sorted(margins, key=lambda n: margins[n]["margin_len"])
    print(f"\n  RF finishing order + margins (lengths behind winner #{winner}):")
    rf_rank = {n: i + 1 for i, n in enumerate(ordered)}
    for i, num in enumerate(ordered):
        m = margins[num]
        off_lbw = m["margin_len"] - off.lbw.get(num, float("nan"))
        print(f"  {i+1:>2} #{num:>2} {off.horse.get(num, '?'):<18} {m['margin_len']:>6.2f}len "
              f"sd {m['sd_len']:>4.2f} n {m['n']:>2}  off_LBW {off_lbw:>+5.2f}")

    common = [n for n in ordered if n in off.lbw]
    rf_m = np.array([margins[n]["margin_len"] for n in common])
    of_m = np.array([off.lbw[n] for n in common])
    mae = a = b = rho = rho_rank = float("nan")
    if len(common) >= 3:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(rf_m, of_m)
        mae = float(np.mean(np.abs(rf_m - of_m)))
        a, b = np.polyfit(of_m, rf_m, 1)
        off_rank = {n: off.finish_order.index(n) + 1 for n in common}
        rho_rank, _ = spearmanr([rf_rank[n] for n in common], [off_rank[n] for n in common])
        print(f"\n=== VALIDATION (margins) ===\n  MAE {mae:.2f}len  fit {a:.2f}*off{b:+.2f}  "
              f"margin rho {rho:+.2f}  ORDER rho {rho_rank:+.2f}  (R1: MAE 0.41, R03 1.70)")

    print("\n=== STEP 3 — legend-strip running ORDER backbone ===")
    seq = RF1.read_legend_order(cap, fps, ocr, t0=4.0, t1=WINNER_CROSS_VID - 2.0, dt=2.0)
    print(f"  legend reads: {len(seq)} timepoints"
          + (f" (vid {seq[0][0]:.0f}-{seq[-1][0]:.0f}s)" if seq else " — NONE"))
    if not seq:
        print("[GENERALIZE-WARN] legend strip not read on R08 — backbone degrades to margins only.")
    for vid_t, order in (seq[::max(1, len(seq) // 6)] if seq else []):
        print(f"    race_t {vid_t - RF1.RF_OFFSET:5.1f}s  lead->back: {order}")

    _validate_legend(seq, off)
    _write_panel(margins, ordered, seq)
    RF1.make_qc(ordered, margins, rf_rank, seq, mae, rho, rho_rank, a, b)
    cap.release()
    return {"mae": mae, "rho": rho, "n_id": len(seen)}


def _margins_rel_winner(frame_recs, winner):
    from collections import defaultdict
    per = defaultdict(list)
    for _vt, numcx, body_px in frame_recs:
        if winner not in numcx or body_px <= 0:
            continue
        xw = numcx[winner]
        for num, x in numcx.items():
            per[num].append((x - xw) / body_px)
    res = {}
    for num, vals in per.items():
        v = np.array(vals)
        if num != winner and len(v) < 2:
            continue
        est = 0.0 if num == winner else float(np.percentile(v, 75))
        res[num] = {
            "margin_len": max(0.0, est),
            "sd_len": float(np.subtract(*np.percentile(v, [75, 25])) / 1.35)
            if len(v) > 2 else float("nan"),
            "n": int(len(v)),
        }
    return res


def _validate_legend(seq, off):
    if not seq:
        return
    from scipy.stats import spearmanr
    print("\n  [legend ORDER vs official RP at the route calls]")
    sec_t = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])
    sec_s = _D_BOUNDS
    # 5 RP calls on the route: 1600/1200/800/400/finish-to-go -> run 200/600/1000/1400/1800.
    calls = {"c1 (200m)": 200.0, "c2 (600m)": 600.0, "c3 (1000m)": 1000.0,
             "c4 (1400m)": 1400.0, "late (1700m)": 1700.0}
    for name, arc in calls.items():
        target_rt = float(np.interp(arc, sec_s, sec_t))
        best = min(seq, key=lambda s: abs((s[0] - RF1.RF_OFFSET) - target_rt))
        order = best[1]
        legend_rank = {n: i + 1 for i, n in enumerate(order)}
        ci = min(list(calls).index(name), 4)
        common = [n for n in order if n in off.rp and len(off.rp[n]) > ci]
        if len(common) >= 3:
            lr = [legend_rank[n] for n in common]
            rp = [off.rp[n][ci] for n in common]
            rho, _ = spearmanr(lr, rp)
            print(f"    {name:14s} race_t {target_rt:5.1f}s n={len(common):2d}  "
                  f"rho(legend, RP) = {rho:+.2f}")


def _write_panel(margins, ordered, seq):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {"runner": [], "race_t": [], "arclen_or_gap": [], "margin_len": [], "sd": [],
            "kind": []}
    cross_rt = WINNER_FINISH_RACE_T
    for num in ordered:
        m = margins[num]
        cols["runner"].append(num)
        cols["race_t"].append(cross_rt)
        cols["arclen_or_gap"].append(DIST - m["margin_len"] * 2.4)
        cols["margin_len"].append(m["margin_len"])
        cols["sd"].append(m["sd_len"] if not np.isnan(m["sd_len"]) else 0.5)
        cols["kind"].append("finish_margin")
    for vid_t, order in seq:
        rt = vid_t - RF1.RF_OFFSET
        for rank0, num in enumerate(order):
            cols["runner"].append(num)
            cols["race_t"].append(rt)
            cols["arclen_or_gap"].append(float(-(rank0 + 1)))
            cols["margin_len"].append(float("nan"))
            cols["sd"].append(float("nan"))
            cols["kind"].append("legend_order")
    pq.write_table(pa.table(cols), PANEL_OUT)
    print(f"\n[panel] {PANEL_OUT} ({len(cols['runner'])} rows; "
          f"{sum(1 for k in cols['kind'] if k == 'finish_margin')} margin, "
          f"{sum(1 for k in cols['kind'] if k == 'legend_order')} legend)")


if __name__ == "__main__":
    main()
