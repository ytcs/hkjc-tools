"""Tests for the canonical Race domain object and its model builders."""
import numpy as np
import polars as pl

from hkjc.core.race import DEFAULT_V_C, Race, Runner
from hkjc.core.simulation import RaceSimulator
from hkjc.core.track import SHA_TIN_STRAIGHT, SHA_TIN_TURF


def _equal_field(n, venue="HV", distance=1200, **kw):
    runners = [Runner(no=i + 1, draw=i + 1) for i in range(n)]
    return Race.from_runners(runners, date="2026/01/01", venue=venue,
                             race_no=1, distance=distance, **kw)


def test_race_resolves_geometry_from_venue_distance_surface():
    assert _equal_field(12, "ST", 1400).geometry is SHA_TIN_TURF
    assert _equal_field(12, "ST", 1000).geometry is SHA_TIN_STRAIGHT
    assert _equal_field(12, "HV", 1200).width == 30.5            # A course default
    assert _equal_field(12, "HV", 1200, course="C+3").width == 19.5


def test_runner_arrays_are_in_starting_order():
    race = _equal_field(6)
    assert race.n == 6
    assert list(race.draws) == [1, 2, 3, 4, 5, 6]


def test_ability_defaults_when_runners_lack_energetics():
    race = _equal_field(8)
    m = race.traffic_model()
    # equal-ability fallback -> all v_target identical
    assert np.allclose(m.v_target, m.v_target[0])
    assert m.v_target[0] > DEFAULT_V_C                         # v_c * (1 + W'/D)


def test_traffic_model_draw_bias_emerges_at_hv():
    race = _equal_field(12, "HV", 1200)
    sim = RaceSimulator(race.traffic_model(), n_samples=6000)
    pw = np.array([sim.prob(sim.wins(i)) for i in range(race.n)])
    assert pw.argmax() == 0                                    # inner draw favoured
    assert pw[:3].mean() > pw[-3:].mean()


def test_straight_chute_has_no_draw_bias():
    # 1000m at Sha Tin is the straight course: no bends -> draws are symmetric
    race = _equal_field(12, "ST", 1000)
    assert race.bend_fraction == 0.0
    sim = RaceSimulator(race.traffic_model(), n_samples=6000)
    pw = np.array([sim.prob(sim.wins(i)) for i in range(race.n)])
    assert abs(pw[:3].mean() - pw[-3:].mean()) < 0.02          # ~uniform


def test_from_results_maps_schema_and_infers_surface():
    df = pl.DataFrame({
        "Date": ["2026/01/01"] * 3, "Venue": ["ST"] * 3, "RaceNumber": [4] * 3,
        "RaceDistance": ["1200"] * 3, "Going": ["GOOD"] * 3, "Course": ["A"] * 3,
        "RaceClass": ["Class 3"] * 3,
        "HorseNo": [1, 2, 3], "Dr": [5, 1, 9], "WinOdds": [3.2, 8.5, 12.0],
        "Pla": [2, 1, 3], "HorseID": ["A001", "B002", "C003"], "Horse": ["X", "Y", "Z"],
    })
    race = Race.from_results(df)
    assert race.venue == "ST" and race.distance == 1200.0 and race.surface == "turf"
    assert list(race.draws) == [5, 1, 9]
    assert race.runners[1].win_odds == 8.5 and race.runners[1].finish_pos == 1
    assert race.geometry is SHA_TIN_TURF


def test_from_results_detects_all_weather():
    df = pl.DataFrame({
        "Date": ["2026/01/01"] * 2, "Venue": ["ST"] * 2, "RaceNumber": [1] * 2,
        "RaceDistance": ["1650"] * 2, "Going": ["GOOD"] * 2,
        "Course": ["ALL WEATHER TRACK"] * 2,
        "HorseNo": [1, 2], "Dr": [3, 7], "WinOdds": [2.0, 5.0],
        "Pla": [1, 2], "HorseID": ["A", "B"], "Horse": ["X", "Y"],
    })
    race = Race.from_results(df)
    assert race.surface == "awt" and race.geometry.surface == "awt"
