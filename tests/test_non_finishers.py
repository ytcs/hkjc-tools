"""Dead heats and horses that started but did not finish: results, sectionals, the official accessors,
the finish re-timing and validation keep every runner."""

import json

import numpy as np
import polars as pl
import pytest
from bs4 import BeautifulSoup

from hkjc.data import Mask, MaskedRace, RaceInfo, Runner
from hkjc.data.historical import _clean_race_data
from hkjc.data.race import official_boundaries, official_finish, official_order, official_placing
from hkjc.data.sectional import _parse_sectional_table
from hkjc.recon.fusion import Path
from hkjc.recon.pipeline import RaceRun
from hkjc.recon.validate import crossing

# The sectional page's table, abridged from ST 2024-02-18 R4 (dead heat for 5th), HV 2025-06-25 R5 (#2 fell,
# #4 unseated: the last cell is a sub-split only) and ST 2025-02-12 R8 (WV-A, DISQ).
SECTIONAL_HTML = """<table>
<tr><td>Finishing Order</td><td>Horse No.</td><td>Horse</td><td>Running Position &amp; Margin Behind Leader</td>
<td>Time</td></tr>
<tr><td>Sectional Time</td></tr>
<tr><td>1st Sec.</td><td>2nd Sec.</td><td>3rd Sec.</td><td>4th Sec.</td><td>5th Sec.</td><td>6th Sec.</td></tr>
<tr><td>1</td><td>1</td><td>JAMES TAK&nbsp;(J007)</td><td>4 1-1/2 24.26</td><td>4 1-1/2 22.93 11.45 11.48</td>
<td>1 3/4 22.85 11.23 11.62</td><td></td><td></td><td></td><td>1:10.04</td></tr>
<tr><td>5 DH</td><td>3</td><td>GIMME FIVE&nbsp;(J099)</td><td>5 1-3/4 24.30</td><td>5 2-3/4 23.09 11.45 11.64</td>
<td>5 3 23.12 11.35 11.77</td><td></td><td></td><td></td><td>1:10.51</td></tr>
<tr><td>5 DH</td><td>10</td><td>NEW POWER&nbsp;(H325)</td><td>1 1/2 24.02</td><td>1 N 22.93 11.41 11.52</td>
<td>5 3 23.56 11.39 12.17</td><td></td><td></td><td></td><td>1:10.51</td></tr>
<tr><td>FE</td><td>2</td><td>SEASONS WIT&nbsp;(H223)</td><td>6 2-1/2 12.95</td><td>5 3-1/2 21.16 10.32 10.84</td>
<td>10.97</td><td></td><td></td><td></td><td>---</td></tr>
<tr><td>WV-A</td><td>6</td><td>EIGHTY LIGHT YEARS&nbsp;(H170)</td><td></td><td></td><td></td><td></td><td></td>
<td></td><td>---</td></tr>
<tr><td>DISQ</td><td>12</td><td>DRAGON AIR FORCE&nbsp;(AJ279)</td><td>5 3-1/4 24.12</td>
<td>4 4-1/2 22.65 11.03 11.62</td><td>2 HD 23.41 11.42 11.99</td><td></td><td></td><td></td><td>1:10.18</td></tr>
</table>"""


def _sectionals() -> pl.DataFrame:
    return pl.DataFrame(_parse_sectional_table(BeautifulSoup(SECTIONAL_HTML, "html.parser").find("table")))


def test_sectionals_keep_dead_heats_and_non_finishers():
    sec = _sectionals()
    per = {r: g for (r,), g in sec.group_by("runner")}
    assert sorted(per) == [1, 2, 3, 10, 12]  # the withdrawn horse (#6) has no rows
    assert per[3]["finish_order"].unique().to_list() == per[10]["finish_order"].unique().to_list() == [5]
    assert per[3].sort("section")["section_time"].to_list() == [24.30, 23.09, 23.12]
    # the faller: its two complete sections only, the sub-split in the last cell is not a section time
    assert per[2].sort("section")["section_time"].to_list() == [12.95, 21.16]
    assert per[2]["finish_order"].is_null().all() and per[2]["total_time"].is_null().all()
    assert per[12]["finish_order"].is_null().all() and per[12]["total_time"].unique().to_list() == [70.18]
    assert per[12]["horse_id"].unique().to_list() == ["AJ279"]  # two-letter brand


def test_official_boundaries_end_a_non_finisher_at_its_last_complete_section():
    info = type("I", (), {"sectionals": _sectionals()})()
    bnd, cum = official_boundaries(info)
    assert bnd == [800.0, 400.0, 0.0]
    assert np.allclose(cum[2][:2], [12.95, 34.11]) and np.isnan(cum[2][2])
    assert cum[3][2] == pytest.approx(70.51) and cum[10][2] == pytest.approx(70.51)


def _raw_results() -> pl.DataFrame:
    """Results-page rows as printed (HV 2025-06-25 R5 style: a faller and an unseated rider, plus a dead heat,
    a disqualified horse and withdrawn horses)."""
    return pl.DataFrame(
        {
            "Pla": ["1", "2 DH", "2 DH", "4", "DISQ", "FE", "UR", "WV", "WXNR"],
            "HorseNo": ["3", "1", "11", "6", "7", "2", "4", "5", "8"],
            "ActWt": ["120"] * 9,
            "DeclarHorseWt": ["1100"] * 9,
            "Dr": ["1", "2", "3", "4", "5", "6", "7", "---", "---"],
            "WinOdds": ["4.1"] * 7 + ["---", "---"],
            "FinishTime": ["0:56.44", "0:56.51", "0:56.51", "0:56.61", "0:56.70", "---", "---", "---", "---"],
        }
    )


def test_results_keep_every_starter_and_drop_withdrawn_horses():
    out = _clean_race_data(_raw_results())
    assert out["HorseNo"].to_list() == [3, 1, 11, 6, 7, 2, 4]  # WV and WXNR are not runners
    assert out["Pla"].to_list() == [1, 2, 2, 4, None, None, None]
    assert out["PlaCode"].to_list() == ["1", "2 DH", "2 DH", "4", "DISQ", "FE", "UR"]
    assert out["FinishTime"].to_list() == [56.44, 56.51, 56.51, 56.61, 56.70, None, None]  # DISQ keeps its time
    assert out["Dr"].to_list() == [1, 2, 3, 4, 5, 6, 7]


def test_results_reject_an_unknown_placing_code():
    raw = _raw_results().with_columns(pl.when(pl.col("HorseNo") == "6").then(pl.lit("XYZ")).otherwise("Pla").alias("Pla"))
    with pytest.raises(ValueError, match="XYZ"):
        _clean_race_data(raw)


def _info() -> RaceInfo:
    """Four runners over 1200 m: #3 wins, #1 and #2 dead-heat for second, #4 falls after the second section."""
    res = _clean_race_data(
        pl.DataFrame(
            {
                "Pla": ["1", "2 DH", "2 DH", "FE"],
                "HorseNo": ["3", "1", "2", "4"],
                "ActWt": ["120"] * 4,
                "DeclarHorseWt": ["1100"] * 4,
                "Dr": ["1", "2", "3", "4"],
                "WinOdds": ["4.1"] * 4,
                "FinishTime": ["1:09.80", "1:10.00", "1:10.00", "---"],
            }
        )
    )
    sections = {3: [23.9, 22.4, 23.5], 1: [24.0, 22.5, 23.5], 2: [24.2, 22.6, 23.2], 4: [24.1, 22.7]}
    rows = [(h, k + 1, t) for h, s in sections.items() for k, t in enumerate(s)]
    sec = pl.DataFrame(rows, schema=["runner", "section", "section_time"], orient="row")
    runners = tuple(Runner(h, f"HORSE {h}", f"X{h:03d}", h) for h in (1, 2, 3, 4))
    return RaceInfo("20250625", "HV", 5, 1200.0, "C", "turf", "GOOD", "Class 4", runners, res, sec)


def test_official_accessors_with_a_dead_heat_and_a_faller():
    info = _info()
    assert info.n == 4
    assert official_order(info) == [3, 1, 2]  # placed horses only
    assert official_placing(info) == {3: 1, 1: 2, 2: 2}
    assert official_finish(info) == {3: 69.8, 1: 70.0, 2: 70.0}
    v = MaskedRace(info)
    assert v.official_placing() == official_placing(info) and v.official_order() == [3, 1, 2]
    for m in (Mask(outcome=True), Mask(boundary=2)):
        assert MaskedRace(info, m).official_placing() == {}


def _path(h, v):
    t = np.linspace(0, 80, 1601)
    return Path(h, t, v * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))


def test_a_dead_heat_is_retimed_onto_one_instant():
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(_info())
    out, _w = run._official_retime({h: _path(h, v) for h, v in ((1, 17.1), (2, 16.9), (3, 17.3), (4, 17.0))})
    c = {h: crossing(q.t, 1200.0 - q.s, None, 0.0) for h, q in out.items()}
    assert abs(c[1] - 70.0) < 5e-3 and abs(c[2] - 70.0) < 5e-3 and abs(c[1] - c[2]) < 1e-3
    assert abs(c[3] - 69.8) < 5e-3


def test_validate_does_not_count_a_non_finisher_against_the_finish_gates(tmp_path):
    info = _info()
    run = RaceRun(info, root=tmp_path, log=lambda *a: None, mask=Mask(outcome=True))
    _b, cum = official_boundaries(info)
    t = np.linspace(0, 80, 1601)
    recs = []
    for h, c in cum.items():  # paths cross each boundary on the official time; the faller's runs on to the line
        c = np.where(np.isfinite(c), c, c[1] + 23.0)
        d = np.interp(t, np.r_[0.0, c], [1200.0, 800.0, 400.0, 0.0], right=-50.0)
        recs.append(pl.DataFrame({"horse_no": h, "t": t, "d": d, "sd_s": np.full_like(t, 0.2)}))
    P = pl.concat(recs)
    P.write_parquet(run.p("paths_video.parquet"))
    P.write_parquet(run.p("paths.parquet"))
    np.savez(run.p("tracks.npz"), t=P["t"].to_numpy(), d=P["d"].to_numpy())
    np.savez(run.p("metric.npz"))
    run.validate()
    res = json.loads(run.p("validation.json").read_text())
    assert res["winner"] == {"official": 3, "reconstructed": 3}
    assert res["sectional_coverage"] == 1.0 and res["sectional_mae"] < 0.02
    assert res["finish_n"] == 3 and res["finish_n_short"] == 0 and res["horses_with_path_short"] == 0
    assert np.isfinite(res["finish_rho"]) and np.isfinite(res["behind_winner_mae"])


def test_only_a_horse_coded_as_stopping_is_barred_and_only_with_the_finish_visible():
    """The faller (#4, FE, no time at the finish) is barred from 50 m before the line on; with the finish withheld its
    code is not visible; an unseated rider's horse (UR) is not barred."""
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(_info())
    assert run.info.official_nonfinish() == {4: "FE"}
    assert run._stopped(run.info.official_nonfinish()) == {4: 1150.0}
    assert MaskedRace(_info(), Mask(boundary=2)).official_nonfinish() == {}
    assert run._stopped({4: "UR"}) == {}
    assert run._stopped({4: "PU"}) == {4: 1150.0}
    run.info = MaskedRace(_info(), Mask(boundary=1))  # a withheld interior boundary is no evidence of stopping
    assert run._stopped(run.info.official_nonfinish()) == {4: 1150.0}


def test_an_excluded_horse_takes_no_segment_even_where_its_evidence_is_strongest():
    """A stopped horse's bar (``core.Evidence.forbid``, set by ``RaceRun._core_product`` from ``_stopped``) is hard in
    the assignment: the segment goes to another horse or to clutter, never to the barred one."""
    from types import SimpleNamespace

    from hkjc.recon import core

    ev = SimpleNamespace(links=np.zeros((0, 3)), seg_lo=np.array([0, 0]), seg_hi=np.array([9, 9]),
                         forbid=np.array([[False, True], [False, False]]))
    U = np.array([[-5.0, -50.0], [-40.0, -1.0]])  # segment 0 looks most like horse 1 (barred there)
    lab = core.milp(U, ev)
    assert lab[0] != 1 and lab.tolist() == [-1, 0]


def test_a_non_finishers_path_ends_at_its_last_labelled_detection(tmp_path):
    run = RaceRun.__new__(RaceRun)
    run.dir = tmp_path
    run._nonfin = {4: 1150.0, 2: -np.inf}
    paths = {h: _path(h, 15.0) for h in (1, 2, 4)}
    np.savez(tmp_path / "tracks.npz", t=np.array([10.0, 50.0, 60.0, 70.0]))
    np.savez(tmp_path / "det_labels_final.npz", labels=np.array([4, 4, 4, 1]))
    out = run._end_at_last_label(paths)
    assert 2 not in out  # no labelled detection: no path
    assert out[4].t[-1] == 60.0 and len(out[4].s) == len(out[4].t)
    assert len(out[1].t) == 1601
