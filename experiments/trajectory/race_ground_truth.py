"""Ground-truth reference for the ONE PoC race: HV 2026-06-03 Race 1.

Scraped from the HKJC sectional-time + results pages (server-rendered HTML).
Used by the CV pipeline's VALIDATION stage (Stage 9) to cross-check recovered
section times / running positions against the official record.

Race: Happy Valley, 2026-06-03, Race 1.
  Distance 1650 m, "C" course (turf), going Good To Firm, 12 runners.
  Winning time 1:38.85 = 98.85 s.

The sectional page gives FOUR splits per horse. HKJC's standard sectional scheme
fixes the FINAL section at the last 400 m and divides the remainder; for a 1650 m
HV race the four sections are (cross-checked: splits average ~24.7 s, ~16-17 m/s
=> ~400 m each, with a longer opening run from the gate):
  S1 = first 450 m   (d: 1650 -> 1200)
  S2 = next  400 m   (d: 1200 ->  800)
  S3 = next  400 m   (d:  800 ->  400)
  S4 = final 400 m   (d:  400 ->    0)
(These four sum to the winner's 98.85 s.) Running positions are called at the
section boundaries 1650 / 1200 / 800 / 400 and at the finish.

CAVEAT: HKJC does not publish the exact marker metres on the public sectional page;
the [450,400,400,400] split is the standard HKJC scheme inferred from the split
durations and is used only to convert recovered times<->distance for validation.
Treat the section *lengths* as ~±25 m uncertain. The split TIMES are exact.

NOTE on (d, off) coords used by src/hkjc/track.py (d = metres-from-finish, race
runs d: 1650 -> 0): marker d-values = 1650, 1200, 800, 400, 0.
"""
from __future__ import annotations

RACE = dict(venue="HV", date="2026-06-03", race_no=1, distance=1650,
            course="C", surface="turf", going="Good To Firm",
            win_time_s=98.85, n_runners=12)

# section boundaries (metres-from-finish), high -> low along the run
SECTION_MARKERS_D = (1650, 1200, 800, 400, 0)
SECTION_LENGTHS_M = (450, 400, 400, 400)  # S1..S4 (HKJC standard scheme, +-25m)

# saddle -> (horse, finish_pos, [S1,S2,S3,S4] split seconds)
SPLITS = {
    3:  ("FAMILY FORTUNE",      1,  [28.17, 23.34, 23.71, 23.63]),
    12: ("THE WAY WE WIN",      2,  [27.65, 23.30, 23.75, 24.20]),
    6:  ("TELECOM POWER",       3,  [28.89, 23.38, 23.35, 23.47]),
    2:  ("SETANTA",             4,  [28.65, 23.26, 23.59, 23.69]),
    9:  ("WAH MAY WAI WAI",     5,  [28.01, 23.22, 23.75, 24.22]),
    8:  ("MANYTHANKS FOREVER",  6,  [28.09, 22.86, 23.95, 24.63]),
    11: ("CASA LEGEND",         7,  [28.37, 23.34, 23.71, 24.14]),
    4:  ("PRINCE ALEX",         8,  [28.09, 23.38, 23.75, 24.38]),
    7:  ("ZETTA FORCE",         9,  [29.05, 23.26, 23.31, 24.12]),
    10: ("SMART TRIO",         10,  [27.89, 23.10, 24.07, 24.78]),
    5:  ("HAPPY BUDDIES",      11,  [27.85, 23.42, 23.99, 24.73]),
    1:  ("MASTER OF HUMOR",    12,  [28.69, 23.38, 23.79, 24.67]),
}

# saddle -> running position at the 4 section calls [1650, 1200, 800, finish]
# (the 4 in-running calls from the sectional page, in race order start->finish)
RUNNING_POS = {
    3:  [7, 3, 6, 1],
    12: [1, 1, 1, 2],
    6:  [11, 11, 10, 3],
    2:  [9, 9, 9, 4],
    9:  [4, 4, 3, 5],
    8:  [6, 2, 2, 6],
    11: [8, 8, 8, 7],
    4:  [5, 6, 5, 8],
    7:  [12, 12, 11, 9],
    10: [3, 3, 4, 10],
    5:  [2, 5, 7, 11],
    1:  [10, 10, 12, 12],
}


def cumulative_times():
    """saddle -> cumulative time (s) at each marker [1650,1200,1050,600,0]."""
    out = {}
    for s, (_, _, sp) in SPLITS.items():
        c, acc = [0.0], 0.0
        for x in sp:
            acc += x
            c.append(acc)
        out[s] = c
    return out


if __name__ == "__main__":
    print(f"{RACE['venue']} {RACE['date']} R{RACE['race_no']} "
          f"{RACE['distance']}m {RACE['course']} course, {RACE['n_runners']} runners")
    print("winner check: sum of splits for #3 =",
          round(sum(SPLITS[3][2]), 2), "vs win_time", RACE["win_time_s"])
    ct = cumulative_times()
    print("cumulative @ markers (s) for #3:", [round(x, 2) for x in ct[3]])
