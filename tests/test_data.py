from pathlib import Path

import pytest

from hkjc.data import race_info, replay_url, track_for


def test_replay_url_pattern():
    u = replay_url("2026-06-03", 8, "drone")
    assert u.endswith(
        "/replay-drone/2026/20260603/08/eng/replay-drone_20260603_08_eng_2500kbps.mp4"
    )


def test_track_widths_by_course():
    hv = track_for("HV", 1800)
    assert hv.width("C") == 22.5 and hv.width("A") == 30.5
    assert track_for("ST", 1000).bend_segments == ()
    assert track_for("ST", 1650, "awt").width("AWT") == 22.8
    # official course diagram (racing.hkjc.com .../racing-course), every listed setting
    st = track_for("ST", 1400)
    official_st = {"A": 30.5, "A+2": 28.5, "A+3": 27.5, "B": 26.0, "B+2": 24.0, "C": 21.3, "C+3": 18.3}
    assert {c: st.width(c) for c in official_st} == official_st
    official_hv = {"A": 30.5, "A+2": 28.5, "B": 26.5, "B+2": 24.5, "B+3": 23.5, "C": 22.5, "C+3": 19.5}
    assert {c: hv.width(c) for c in official_hv} == official_hv


def test_bend_segments_measured():
    # measured from rendered rail curvature (> 1 mrad/m), journal 0249
    hv, st, st_back = track_for("HV", 1650), track_for("ST", 2000), track_for("ST", 1600)
    assert hv.bend_segments == ((315.0, 590.0), (920.0, 1480.0)) and hv.circumference == 1480.0
    assert st.bend_segments == ((430.0, 900.0), (1385.0, 1900.0)) and st.circumference == 1900.0
    assert st_back.bend_segments == ((430.0, 900.0),)
    assert track_for("ST", 1200) is st_back and track_for("ST", 1800) is st_back
    assert track_for("ST", 2200) is st and track_for("ST") is st
    assert not hv.is_bend(750.0) and hv.is_bend(450.0)  # HV back straight, home bend
    assert not st_back.is_bend(1700.0) and st.is_bend(1700.0)  # 1,800 m start straight vs 2,000 m far bend
    # sanity: the measured home straight (start of the first turn) matches the official course-A figure
    # within the threshold sensitivity of the measurement (10 m)
    for g in (hv, st):
        assert abs(g.bend_segments[0][0] - g.home_straight("A")) <= 10.0
    # turns are inside one lap and do not overlap
    for g in (hv, st, st_back):
        ends = [x for seg in g.bend_segments for x in seg]
        assert ends == sorted(ends) and 0 < ends[0] and ends[-1] <= g.circumference


@pytest.mark.skipif(
    not Path("data_cache/race_official/20260603HV8_results.parquet").exists(), reason="no cache"
)
def test_race_info_from_cache():
    r = race_info("20260603", "HV", 8)
    assert r.key == "20260603_HV_8" and r.course == "C" and r.n == 11 and r.width == 22.5
