"""Shared R03 race constants + official parser + shared-clock geometry (task #91).

GENERALIZATION of the R1 4-feed pipeline (journals 0075/0076/0080/0086/0087) to a SECOND
race: HV 2026-06-03 R03, Class 4 1200M turf "C" (Happy Valley, tight 1450m circuit). This
module factors out everything the per-feed R1 studies HARD-CODED as R1 (1650m / 4 sections /
ST) constants, so the per-feed R03 drivers can REUSE the R1 geometry functions unchanged.

Every R1->R03 friction this surfaced is journalled (#91 refinement backlog). The big ones
captured here:
  * DISTANCE 1650 -> 1200 (sprint).
  * SECTIONS 4 -> 3 (a 1200m HK race has THREE published sectionals, not four) -> the
    arclen<->race_t curve and the distance-run call boundaries change shape.
  * winner finish race_t 98.85 -> 69.29.
  * official format HTML (R1) -> JSON-wrapped rendered TEXT (R03): a NEW parser is required.
  * the venue is HAPPY VALLEY (tight circuit), not ST — CV reads the rail directly so the
    geometry pipeline is venue-agnostic, but the track-geometry physics (draw/ground-loss)
    would need HV constants (out of scope for this CV recon).

Official data is VALIDATION-ONLY (PHILOSOPHY), never a model input.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- race facts
DIST = 1200.0
TRACK_WIDTH_M = 21.3              # "C" course width (same as R1 HV "C" rail)
RACE_NO = 3
VENUE = "HV"                     # Happy Valley (NOT "ST" — the 0067-0087 labels were wrong)
FIELD_SIZE = 12
BODY_LEN_M = 2.4                 # 1 length == 2.4 m (0086)

RESULTS_JSON = "data_cache/race_official/20260603R3_results.json"

# 1200M HK race: THREE sectionals (R1's 1650m had four). From the official chart:
#   23.54 / 22.65 / 23.10  (sum 69.29 == winner 1:09.29).
OFFICIAL_SECTIONS = [23.54, 22.65, 23.10]
WINNER_FINISH_RACE_T = sum(OFFICIAL_SECTIONS)   # 69.29 s

# distance-run boundaries. HK sectionals for a 1200m race are the FIRST (to the 800m mark =
# 400m run), the MIDDLE 400m, and the LAST 400m. So the call boundaries are 0/400/800/1200.
# (R1's 1650m used 0/450/850/1250/1650.) The running-position columns are 3 calls here, not 4.
_D_BOUNDS = np.array([0.0, 400.0, 800.0, 1200.0])
_T_CUM = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])   # 0, 23.54, 46.19, 69.29


def arclen_to_race_t(arclen_m) -> np.ndarray:
    return np.interp(np.asarray(arclen_m, float), _D_BOUNDS, _T_CUM)


def race_t_to_arclen(race_t) -> np.ndarray:
    return np.interp(np.asarray(race_t, float), _T_CUM, _D_BOUNDS)


# --------------------------------------------------------------------------- official parse
@dataclass
class Official:
    finish_order: list[int]                 # by place
    place_of: dict[int, int]                # runner -> place
    lbw: dict[int, float]                   # runner -> lengths behind winner
    rp: dict[int, list[int]]                # runner -> running positions (3 calls)
    draw: dict[int, int]                    # runner -> barrier draw
    horse: dict[int, str]                   # runner -> name
    sections: list[float]
    comments: dict[int, str]                # runner -> stewards comment text


def _lbw_to_lengths(s: str) -> float:
    """Parse an HK LBW string ('1/2', '3-1/4', '5-1/2', '---', 'NK', 'SH') to lengths."""
    s = s.strip().upper()
    if s in ("---", "", "DH"):
        return 0.0
    if s in ("NK", "NECK"):
        return 0.3
    if s in ("SH", "SHD", "SHORT HEAD"):
        return 0.1
    if s in ("HD", "HEAD"):
        return 0.2
    if s in ("DIST",):
        return 30.0
    m = re.match(r"^(\d+)-(\d+)/(\d+)$", s)        # '3-1/4'
    if m:
        return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
    m = re.match(r"^(\d+)/(\d+)$", s)              # '1/2'
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.match(r"^(\d+(?:\.\d+)?)$", s)
    if m:
        return float(m.group(1))
    return float("nan")


def parse_official(path: str = RESULTS_JSON) -> Official:
    """Parse the R03 rendered-text results (JSON-wrapped). NEW parser vs R1's HTML <td> walk.

    The JSON file is a single rendered string. The results block is a TAB/newline table:
        Pla. Horse No. Horse Jockey Trainer Act.Wt DeclHWt Dr. LBW Running\nPosition FinishTime WinOdds
    with the running-position column itself containing newlines ('8 7 1'). We walk the lines,
    locate the 12 finisher rows by 'Pla. \t HorseNo. \t HORSE (Gxxx)', and stitch the
    multi-line RP / LBW back together.
    """
    raw = open(path, encoding="utf-8", errors="ignore").read()
    try:
        text = json.loads(raw) if raw.lstrip().startswith(("\"", "{", "[")) else raw
    except json.JSONDecodeError:
        text = raw
    if isinstance(text, dict):
        text = next((v for v in text.values() if isinstance(v, str)), raw)

    # split into logical rows on the place-number boundary. A finisher row looks like:
    #   "<pla>\t<no>\t<HORSE> (<id>)\t<jockey>\t<trainer>\t<wt>\t<hwt>\t<dr>\t<lbw>\t\n<rp>\n\t<time>\t<odds>"
    rows: dict[int, dict] = {}
    # tokenise: find each finisher by the 'NAME (ID)' anchor with preceding pla/no.
    pat = re.compile(
        r"(?m)^\s*(\d{1,2})\s*\t\s*(\d{1,2})\s*\t\s*([A-Z][A-Z '.&-]+?)\s*\(([A-Z]\d+)\)\s*\t"
        r"(.*?)\t(.*?)\t(\d+)\t(\d+)\t(\d+)\t([^\n]*)\n([\d\s]+)\n\s*\t([\d:.]+)\t([\d.]+)"
    )
    for m in pat.finditer(text):
        pla = int(m.group(1))
        no = int(m.group(2))
        rps = [int(x) for x in m.group(11).split() if x.isdigit()]
        rows[no] = dict(
            pla=pla, horse=m.group(3).strip(), draw=int(m.group(9)),
            lbw=_lbw_to_lengths(m.group(10)), rp=rps,
        )

    if len(rows) < FIELD_SIZE:
        raise RuntimeError(f"R03 parse found only {len(rows)} rows (expected {FIELD_SIZE})")

    finish_order = [no for no, _ in sorted(rows.items(), key=lambda kv: kv[1]["pla"])]
    place_of = {no: r["pla"] for no, r in rows.items()}
    lbw = {no: r["lbw"] for no, r in rows.items()}
    rp = {no: r["rp"] for no, r in rows.items()}
    draw = {no: r["draw"] for no, r in rows.items()}
    horse = {no: r["horse"] for no, r in rows.items()}

    # ---- comments-on-running (the lateral ground-truth) ----
    comments: dict[int, str] = {}
    cstart = text.find("Racing Incident Report")
    if cstart >= 0:
        cblock = text[cstart:]
        cpat = re.compile(
            r"(?m)^\s*(\d{1,2})\s*\t\s*(\d{1,2})\s*\t\s*[A-Z][A-Z '.&-]+?\s*\([A-Z]\d+\)\s*\t(.*?)"
            r"(?=\n\s*\d{1,2}\s*\t\s*\d{1,2}\s*\t|\nBreed|\Z)",
            re.S,
        )
        for m in cpat.finditer(cblock):
            comments[int(m.group(2))] = re.sub(r"\s+", " ", m.group(3)).strip()

    return Official(
        finish_order=finish_order, place_of=place_of, lbw=lbw, rp=rp,
        draw=draw, horse=horse, sections=OFFICIAL_SECTIONS, comments=comments,
    )


# --------------------------------------------------------------------------- lateral truth
def lateral_truth(off: Official) -> tuple[set[int], set[int]]:
    """Classify rail/tight vs wide horses from the stewards' Comments-on-Running + draw.

    R1 hand-curated RAIL_HORSES/WIDE_HORSES (0087). [GENERALIZE friction] Deriving them from
    free text is HARDER on R03: most R03 comments are INTERFERENCE incidents ("shifted out and
    made contact"), NOT sustained trip-width, so a naive keyword scan over-reads WIDE. We use a
    PRECISE phrase set for the genuine trip-position calls (e.g. "raced wide and without cover"
    => WIDE; "closer to the rail"/"obtain cover" => RAIL) and fall back to the BARRIER DRAW (the
    objective lateral proxy: low draw == inside/rail, high draw == wide) for the unmentioned
    horses. This is the race-agnostic lateral ground-truth.
    """
    rail, wide = set(), set()
    wide_ph = ("raced wide", "wide and without cover", "without cover", "caught wide",
               "taken wider", "raced keenly and", "from a wide")
    rail_ph = ("closer to the rail", "obtain cover", "to the inside", "on the rail",
               "behind runners", "obtain a position closer")
    for no, c in off.comments.items():
        cl = c.lower()
        if any(k in cl for k in wide_ph):
            wide.add(no)
        elif any(k in cl for k in rail_ph):
            rail.add(no)
    # draw fallback for the unclassified (objective lateral proxy): bottom third draw -> rail,
    # top third draw -> wide. (HV "C": low gate hugs the inside.)
    unclassified = [n for n in off.draw if n not in rail and n not in wide]
    if unclassified:
        draws = sorted(unclassified, key=lambda n: off.draw[n])
        k = max(1, len(draws) // 3)
        rail |= set(draws[:k])
        wide |= set(draws[-k:])
    return rail, wide


if __name__ == "__main__":
    o = parse_official()
    print(f"R03 — HV 2026-06-03, 1200M C, {len(o.finish_order)} runners")
    print("finish:", o.finish_order)
    print("winner:", o.finish_order[0], o.horse[o.finish_order[0]],
          "rp", o.rp[o.finish_order[0]])
    print("sections:", o.sections, "-> winner_t", WINNER_FINISH_RACE_T)
    print("LBW:", {n: round(o.lbw[n], 2) for n in o.finish_order})
    print("draw:", o.draw)
    rail, wide = lateral_truth(o)
    print("derived rail/tight:", sorted(rail))
    print("derived wide      :", sorted(wide))
    print(f"comments parsed: {len(o.comments)}/{FIELD_SIZE}")
