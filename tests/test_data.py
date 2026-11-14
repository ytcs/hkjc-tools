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


@pytest.mark.skipif(
    not Path("data_cache/race_official/20260603HV8_results.parquet").exists(), reason="no cache"
)
def test_race_info_from_cache():
    r = race_info("20260603", "HV", 8)
    assert r.key == "20260603_HV_8" and r.course == "C" and r.n == 11 and r.width == 22.5
