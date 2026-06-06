"""Offline tests for the localtrackwork parser (mirrors the real page layout)."""
from hkjc.data.trackwork import (
    _gallop_col_index,
    _intensity,
    _parse_horse_html,
    _parse_html,
    _session_iso,
    _split_rc_track,
    _time_to_sec,
    _to_iso,
)

# A trackwork table: a header row naming the work-type columns, then one row per
# runner whose "Gallop" cell holds the dated timed sessions (multi-furlong + total).
HTML = b"""
<table>
  <tr><th>Horse No.</th><th>Name of Horse/ Trainer/ Last 6 Runs</th>
      <th>Barrier Trial</th><th>Gallop</th><th>Trotting</th><th>All</th></tr>
  <tr>
    <td>1</td>
    <td><a href="/en-us/local/information/horse?horseid=HK_2023_J183">FIREFOOT W Y So 2/2/4/8/3/3</a></td>
    <td></td>
    <td>02/06: CONGHUA TurG 25.9 24.6 (50.5) (R.B.)
        29/05: CONGHUA AWT 27.7 26.1 (53.8) (R.B.)</td>
    <td>05/06: SHA TIN TroR TroR Canter (R.B.)</td>
    <td>Details</td>
  </tr>
  <tr>
    <td>2</td>
    <td><a href="/en-us/local/information/horse?horseid=HK_2024_K502">I EXCELLE P C Ng 10/13/11/11/9/6</a></td>
    <td></td>
    <td>04/06: SHA TIN AWT 33.0 30.0 25.9 (1.28.9) (H Y Yuen)</td>
    <td></td>
    <td>Details</td>
  </tr>
</table>
"""


def test_date_helpers():
    assert _to_iso("2026/06/07") == "2026-06-07"
    assert _to_iso("2026-06-07") == "2026-06-07"


def test_time_to_sec():
    assert abs(_time_to_sec("1.28.9") - 88.9) < 1e-9    # m.ss.f
    assert _time_to_sec("50.5") == 50.5                 # ss.f
    assert _time_to_sec("") is None


def test_session_iso_resolves_year():
    # race in June: a June session is same year
    assert _session_iso("02/06", "2026-06-07") == "2026-06-02"
    # race in January: a December session belongs to the prior year
    assert _session_iso("28/12", "2026-01-04") == "2025-12-28"


def test_intensity():
    dist, vel = _intensity([25.9, 24.6], 50.5)
    assert abs(dist - 2 * 201.168) < 1e-6
    assert abs(vel - (2 * 201.168 / 50.5)) < 1e-3
    assert _intensity([], None) == (None, None)


def test_gallop_col_index():
    assert _gallop_col_index(["No.", "Name", "Barrier Trial", "Gallop", "All"]) == 3
    assert _gallop_col_index(["No.", "Name", "Swimming"]) is None


def test_parse_html_extracts_dated_gallops():
    rows = _parse_html(HTML, "2026-06-07", "ST", 1)
    # 2 gallops for runner 1 + 1 for runner 2 (Trotting/Barrier cols ignored)
    assert len(rows) == 3
    by_brand = {}
    for r in rows:
        by_brand.setdefault(r["horse_brand"], []).append(r)
    assert set(by_brand) == {"J183", "K502"}

    g0 = next(r for r in rows if r["session_date"] == "2026-06-02")
    assert g0["horse_brand"] == "J183"
    assert g0["horse_id"] == "HK_2023_J183"
    assert g0["location"] == "CONGHUA"
    assert g0["surface"] == "TurG"
    assert g0["n_splits"] == 2
    assert g0["total_time"] == 50.5
    assert g0["days_before_race"] == 5          # strictly prior (> 0)
    assert g0["rider"] == "R.B."
    assert g0["gallop_velocity"] is not None

    # the 3-furlong Sha Tin gallop with a m.ss.f total
    g1 = next(r for r in rows if r["horse_brand"] == "K502")
    assert g1["n_splits"] == 3
    assert abs(g1["total_time"] - 88.9) < 1e-9
    assert g1["location"] == "SHA TIN"


def test_parse_html_all_sessions_strictly_prior():
    rows = _parse_html(HTML, "2026-06-07", "ST", 1)
    assert all(r["days_before_race"] is not None and r["days_before_race"] > 0 for r in rows)


# The per-horse historical-archive page (trackworkresult?horseid=…): a Date/Type/Track/
# Workouts/Gear table mixing timed gallops with non-timed work types.
HORSE_HTML = b"""
<table>
  <tr><th>Date</th><th>Type</th><th>Racecourse/Track</th><th>Workouts</th><th>Gear</th></tr>
  <tr><td>05/06/2026</td><td>Trotting</td><td>Sha Tin TroR</td>
      <td>TroR TroR Canter (R.B.)</td><td>H/P</td></tr>
  <tr><td>02/06/2026</td><td>Gallop</td><td>Conghua TurG</td>
      <td>25.9 24.6 (50.5) (R.B.)</td><td>B</td></tr>
  <tr><td>13/04/2026</td><td>Gallop</td><td>Sha Tin AWT</td>
      <td>33.1 29.6 25.8 (1.28.5) (R.B.)</td><td>H/P</td></tr>
  <tr><td>03/06/2026</td><td>Swimming</td><td>Conghua</td><td></td><td></td></tr>
</table>
"""


def test_split_rc_track():
    assert _split_rc_track("Sha Tin AWT") == ("SHA TIN", "AWT")
    assert _split_rc_track("Conghua TurG") == ("CONGHUA", "TurG")
    assert _split_rc_track("Sha Tin Stall Test Turf") == ("SHA TIN", "Stall Test Turf")


def test_parse_horse_html_only_timed_gallops():
    rows = _parse_horse_html(HORSE_HTML, "HK_2023_J183", "J183")
    # only the 2 Gallop rows (Trotting/Swimming skipped — no timed impulse)
    assert len(rows) == 2
    assert all(r["horse_brand"] == "J183" for r in rows)
    g = next(r for r in rows if r["session_date"] == "2026-06-02")
    assert g["location"] == "CONGHUA"
    assert g["surface"] == "TurG"
    assert g["n_splits"] == 2
    assert g["total_time"] == 50.5
    assert g["gallop_velocity"] is not None
    # race-relative fields are null on the horse-archive route
    assert g["race_date"] is None and g["days_before_race"] is None
    g3 = next(r for r in rows if r["n_splits"] == 3)
    assert abs(g3["total_time"] - 88.5) < 1e-9
