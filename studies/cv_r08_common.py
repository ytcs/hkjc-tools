"""Shared R08 race constants + official parse + shared-clock geometry (task #95).

HELD-OUT generalization of the HARDENED 4-feed pipeline to the LONGEST race in the
distance-spread corpus: HV 2026-06-03 R08, Class 2 1800M turf "C" (Happy Valley), winner
#6 SILVERY BREEZE. R08 was DELIBERATELY held out while the pipeline was hardened on R01
(1650m) + R03 (1200m). NO per-race tuning of the identity consensus weights or any feed
algorithm/threshold — only race-PARAMETERS (distance, the 5-section clock, field size,
video paths, auto-detected windows) are re-pointed.

THE ROUTE STRESS CASE: 1800m has FIVE sectionals and the MOST/sharpest bends of the corpus
(sprint R03 1200m had 3, mile R01 1650m had 4) -> the foreshortening metric gap (the aerial
render lateral degradation on bends) bites hardest here. This module factors out everything
the R03 driver baked as R03 (1200m / 3 sections) so the per-feed R08 drivers REUSE the R1/R03
geometry functions unchanged.

This module REUSES cv_r03_common's parser/LBW machinery (same JSON-wrapped rendered-text
official format) verbatim — only the race FACTS differ. Field is 11 (not 12).

Official data is VALIDATION-ONLY (PHILOSOPHY); the draw "Dr." column is a pre-race INITIAL
CONDITION (allowed input, NOT an outcome).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "studies")

# REUSE the R03 parser machinery READ-ONLY (same official JSON-text format, same LBW/RP regex).
from cv_r03_common import (  # noqa: E402
    Official,
    _lbw_to_lengths,  # noqa: F401  (re-exported for completeness)
)
from cv_r03_common import parse_official as _parse_r03  # noqa: E402

# --------------------------------------------------------------------------- race facts
DIST = 1800.0
TRACK_WIDTH_M = 21.3              # "C" course width (same C rail as R01/R03)
RACE_NO = 8
VENUE = "HV"                     # Happy Valley
FIELD_SIZE = 11                  # 11 runners (R01/R03 were 12)

BODY_LEN_M = 2.4                 # 1 length == 2.4 m (0086)

RESULTS_JSON = "data_cache/race_official/20260603R8_results.json"

# 1800M HK route: FIVE sectionals (R01 1650m had four; R03 1200m had three). From the chart:
#   13.78 / 23.56 / 24.56 / 23.62 / 23.10  (sum 108.62 == winner 1:48.62).
OFFICIAL_SECTIONS = [13.78, 23.56, 24.56, 23.62, 23.10]
WINNER_FINISH_RACE_T = sum(OFFICIAL_SECTIONS)   # 108.62 s

# distance-run boundaries. An HK 1800m route has a SHORT first sectional (the run to the first
# 400m-call marker) then four 400m sections: 1800 - 4*400 = 200m first section. So the running
# -position CALLS sit at 0/200/600/1000/1400/1800m (5 RP columns, matching the 5-call RP strings
# parsed below). (R01 1650m used 0/450/850/1250/1650; R03 1200m used 0/400/800/1200.)
_D_BOUNDS = np.array([0.0, 200.0, 600.0, 1000.0, 1400.0, 1800.0])
_T_CUM = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])  # 0,13.78,37.34,61.90,85.52,108.62


def arclen_to_race_t(arclen_m) -> np.ndarray:
    return np.interp(np.asarray(arclen_m, float), _D_BOUNDS, _T_CUM)


def race_t_to_arclen(race_t) -> np.ndarray:
    return np.interp(np.asarray(race_t, float), _T_CUM, _D_BOUNDS)


# --------------------------------------------------------------------------- official parse
def parse_official(path: str = RESULTS_JSON) -> Official:
    """Parse the R08 rendered-text results. REUSES the R03 parser (identical format) but:
      * field size 11 (the R03 parser asserts >= its FIELD_SIZE; we temporarily relax it),
      * fixes the bare 'N' (neck) LBW token the R03 _lbw_to_lengths missed (#11 reads 'N').
    Re-points the section list to the 5-section route clock.
    """
    import cv_r03_common as R3C  # noqa: N812

    saved_fs, saved_json, saved_sec = R3C.FIELD_SIZE, R3C.RESULTS_JSON, R3C.OFFICIAL_SECTIONS
    R3C.FIELD_SIZE = FIELD_SIZE
    R3C.RESULTS_JSON = path
    R3C.OFFICIAL_SECTIONS = OFFICIAL_SECTIONS
    try:
        off = _parse_r03(path)
    finally:
        R3C.FIELD_SIZE, R3C.RESULTS_JSON, R3C.OFFICIAL_SECTIONS = saved_fs, saved_json, saved_sec

    # R08-specific token fix: #11 finished a 'N' (neck) behind; the R03 LBW parser handles 'NK'/
    # 'NECK' but not the bare 'N'. Patch any NaN LBW from the raw text's neck/short-head token.
    for n, v in list(off.lbw.items()):
        if v != v:  # NaN
            off.lbw[n] = 0.3  # a neck (the only NaN here is #11 'N')
    return Official(
        finish_order=off.finish_order, place_of=off.place_of, lbw=off.lbw, rp=off.rp,
        draw=off.draw, horse=off.horse, sections=OFFICIAL_SECTIONS, comments=off.comments,
    )


# --------------------------------------------------------------------------- lateral truth
def lateral_truth(off: Official) -> tuple[set[int], set[int]]:
    """Classify rail/tight vs wide horses from the stewards' Comments-on-Running + draw.

    REUSES the R03 race-agnostic phrase set + barrier-draw fallback (cv_r03_common.lateral_truth)
    verbatim. R08's stewards comments DO carry genuine trip-width calls (#8 "obliged to race wide
    and without cover", #5 "caught wide ... shifted behind runners", #3/#1 "from a wide barrier
    ... shifted across behind runners") on top of the start-bump incidents, so the phrase scan is
    informative here, with the low/high-draw fallback for the unmentioned.
    """
    import cv_r03_common as R3C  # noqa: N812

    saved = R3C.FIELD_SIZE
    R3C.FIELD_SIZE = FIELD_SIZE
    try:
        return R3C.lateral_truth(off)
    finally:
        R3C.FIELD_SIZE = saved


if __name__ == "__main__":
    o = parse_official()
    print(f"R08 — HV 2026-06-03, 1800M C route, {len(o.finish_order)} runners")
    print("finish:", o.finish_order)
    print("winner:", o.finish_order[0], o.horse[o.finish_order[0]], "rp", o.rp[o.finish_order[0]])
    print("sections:", o.sections, "-> winner_t", WINNER_FINISH_RACE_T)
    print("LBW:", {n: round(o.lbw[n], 2) for n in o.finish_order})
    print("draw:", o.draw)
    print("D_BOUNDS:", _D_BOUNDS, "T_CUM:", np.round(_T_CUM, 2))
    rail, wide = lateral_truth(o)
    print("derived rail/tight:", sorted(rail))
    print("derived wide      :", sorted(wide))
    print(f"comments parsed: {len(o.comments)}/{FIELD_SIZE}")
