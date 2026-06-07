"""R03 RF / race-film feed — GENERALIZE cv_rf_longitudinal (#89/0086) to HV R03 1200m (#91).

REUSE strategy: import the R1 RF functions unchanged (detect_horses / ocr_numbers /
measure_margins / margins_from_frames / read_legend_order / _match_name) and re-point the R1
module constants at R03:
  * the name->number LUT (FIELD/NAME2NUM) is REBUILT from the R03 official parse (race-agnostic).
  * sections 4->3 -> SEC_T/SEC_S re-derived (1200m / 3 calls).
  * RF_OFFSET re-anchored (winner crosses ~vid 72s; R1 was 1.34s — the broadcast lead-in differs).
  * the finish window FIN_T0/T1 re-found for the sprint (winner crosses earlier).

Writes data_cache/cv_r03_rf_longitudinal.parquet. Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_r03_rf.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "studies")
import cv_rf_longitudinal as RF1  # noqa: E402,N812  (R1 RF toolkit, reused)
from cv_r03_common import (  # noqa: E402
    DIST,
    OFFICIAL_SECTIONS,
    WINNER_FINISH_RACE_T,
    parse_official,
)

VIDEO = "data_cache/video/full_20260603_03.mp4"
PANEL_OUT = "data_cache/cv_r03_rf_longitudinal.parquet"
FIG = "data_cache/figs/cv_r03_rf_qc.png"


def _setup_r03(off) -> None:
    """[GENERALIZE] Re-point every R1-hard-coded RF constant at R03 (the refinement backlog)."""
    RF1.VIDEO = VIDEO
    RF1.PANEL_OUT = PANEL_OUT
    RF1.FIG = FIG
    RF1.FIELD = dict(off.horse)
    RF1.NAME2NUM = {v.replace(" ", "").upper(): k for k, v in off.horse.items()}
    RF1.OFFICIAL_ORDER = list(off.finish_order)
    RF1.OFFICIAL_LBW = dict(off.lbw)
    # sections 4 -> 3 (1200m). HK 1200m calls at 0/400/800/1200m.
    RF1.SEC_T = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])  # 0,23.54,46.19,69.29
    RF1.SEC_S = np.array([0.0, 400.0, 800.0, 1200.0])
    # clock anchor: winner crosses ~vid 72s (frame-verified); race winner time 69.29s.
    RF1.RF_OFFSET = 72.0 - WINNER_FINISH_RACE_T  # ~2.71 (R1 1.34)
    RF1.WINNER_CROSS_VID = WINNER_FINISH_RACE_T + RF1.RF_OFFSET
    # [GENERALIZE friction] the R03 finish cam zooms LESS than R1: saddlecloths blur+cluster the
    # instant the field crosses (~t>=71s OCR yield -> 0). The legible window is the APPROACH
    # (t~68.5-71s), not the line itself. Window it there + finer step to pool more co-framed reads.
    RF1.FIN_T0, RF1.FIN_T1, RF1.FIN_DT = 68.5, 71.2, 0.04
    RF1.OFFICIAL_TIME = {off.finish_order[0]: WINNER_FINISH_RACE_T}


def main():
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    off = parse_official()
    _setup_r03(off)
    print("=" * 80)
    print("R03 RF #91 — race-film longitudinal+margins GENERALIZED to HV 1200m")
    print("=" * 80)
    print(f"[GENERALIZE] RF_OFFSET={RF1.RF_OFFSET:.2f}s (R1 1.34); finish window "
          f"{RF1.FIN_T0}-{RF1.FIN_T1}s (R1 96.5-100.5); winner #{off.finish_order[0]}")

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ocr = RapidOCR()

    print("\n=== STEP 1/2 — finish-window detection + saddlecloth OCR + margins ===")
    frame_recs = RF1.measure_margins(cap, fps, ocr)
    seen = set()
    for _, numcx, _ in frame_recs:
        seen |= set(numcx)
    print(f"finish-window frames used: {len(frame_recs)}; numbers OCR'd: {sorted(seen)} "
          f"({len(seen)}/12)")

    winner = off.finish_order[0]
    # [GENERALIZE 4] margins_from_frames HARD-CODES winner==#3 as the reference. For R03 the
    # winner is #7 -> recompute gaps relative to #7.
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
              f"margin rho {rho:+.2f}  ORDER rho {rho_rank:+.2f}  (R1: MAE 0.41, rho +0.90)")

    print("\n=== STEP 3 — legend-strip running ORDER backbone ===")
    seq = RF1.read_legend_order(cap, fps, ocr, t0=4.0, t1=68.0, dt=2.0)
    print(f"  legend reads: {len(seq)} timepoints"
          + (f" (vid {seq[0][0]:.0f}-{seq[-1][0]:.0f}s)" if seq else " — NONE"))
    if not seq:
        print("[GENERALIZE-WARN] legend strip not read on R03 — name OCR / strip rows may differ; "
              "the longitudinal backbone degrades to the finish margins only.")
    for vid_t, order in (seq[::max(1, len(seq) // 6)] if seq else []):
        print(f"    race_t {vid_t - RF1.RF_OFFSET:5.1f}s  lead->back: {order}")

    _validate_legend(seq, off)
    _write_panel(margins, ordered, seq)
    RF1.make_qc(ordered, margins, rf_rank, seq, mae, rho, rho_rank, a, b)
    cap.release()
    return {"mae": mae, "rho": rho, "n_id": len(seen)}


def _margins_rel_winner(frame_recs, winner):
    """Re-derive open-gap margins relative to the actual winner (R1 hard-coded #3)."""
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
    print("\n  [legend ORDER vs official RP at the 3 calls]")
    sec_t = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])
    sec_s = np.array([0.0, 400.0, 800.0, 1200.0])
    calls = {"1/4 (400m)": 400.0, "1/2 (800m)": 800.0, "late (1100m)": 1100.0}
    for name, arc in calls.items():
        target_rt = float(np.interp(arc, sec_s, sec_t))
        best = min(seq, key=lambda s: abs((s[0] - RF1.RF_OFFSET) - target_rt))
        order = best[1]
        legend_rank = {n: i + 1 for i, n in enumerate(order)}
        ci = list(calls).index(name)
        common = [n for n in order if n in off.rp and len(off.rp[n]) > ci]
        if len(common) >= 3:
            lr = [legend_rank[n] for n in common]
            rp = [off.rp[n][ci] for n in common]
            rho, _ = spearmanr(lr, rp)
            print(f"    {name:14s} race_t {target_rt:4.1f}s n={len(common):2d}  "
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
