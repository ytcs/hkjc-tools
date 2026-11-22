"""Race-facts mask: the reconstruction reads the race through ``MaskedRace``; validation scores the full record."""

import json
import re
from pathlib import Path as FsPath

import numpy as np
import polars as pl
import pytest

from hkjc.data import Mask, MaskedRace, RaceInfo, Runner
from hkjc.data.race import official_boundaries, official_finish, official_order
from hkjc.recon.fusion import Path
from hkjc.recon.pipeline import RaceRun
from hkjc.recon.validate import _cum_official, crossing

D = 1200.0
ORDER = [3, 1, 2]  # placing order (results rows), deliberately not HorseNo order
SECTIONS = {1: [24.0, 22.5, 23.5], 2: [24.2, 22.6, 23.8], 3: [23.9, 22.4, 23.5]}  # runner -> section times
DRAW = {1: 2, 2: 3, 3: 1}


def _info() -> RaceInfo:
    fin = {h: round(sum(s), 2) for h, s in SECTIONS.items()}
    res = pl.DataFrame(
        {
            "Pla": [1, 2, 3],
            "HorseNo": ORDER,
            "Horse": [f"HORSE {h} (X{h:03d})" for h in ORDER],
            "Dr": [DRAW[h] for h in ORDER],
            "LBW": ["-", "1", "2-1/2"],
            "RunningPosition": ["1 1 1", "2 2 2", "3 3 3"],
            "FinishTime": [fin[h] for h in ORDER],
            "HorseID": [f"X{h:03d}" for h in ORDER],
        }
    )
    rows = [
        (o + 1, h, k + 1, k + 1, t, round(sum(SECTIONS[h]), 2))
        for o, h in enumerate(ORDER)
        for k, t in enumerate(SECTIONS[h])
    ]
    sec = pl.DataFrame(rows, schema=["finish_order", "runner", "section", "position", "section_time", "total_time"],
                       orient="row")
    runners = tuple(Runner(h, f"HORSE {h}", f"X{h:03d}", DRAW[h]) for h in sorted(SECTIONS))
    return RaceInfo("20260603", "HV", 8, D, "C", "turf", "GOOD", "Class 4", runners, res, sec)


def _bare(info, mask) -> RaceRun:
    run = RaceRun.__new__(RaceRun)
    run.truth, run.mask, run.info, run.log = info, mask, MaskedRace(info, mask), (lambda *a: None)
    return run


def test_mask_spec():
    assert not Mask() and str(Mask()) == "none"
    assert Mask.parse("none") == Mask.parse("") == Mask.parse(None) == Mask()
    assert Mask.parse("boundary=2") == Mask(boundary=2) and str(Mask(boundary=2)) == "boundary=2"
    assert Mask.parse("outcome") == Mask(outcome=True) and Mask(outcome=True)
    for bad in ("boundary", "results", "boundary=x"):
        with pytest.raises(ValueError):
            Mask.parse(bad)
    with pytest.raises(ValueError):
        Mask(boundary=-1)
    with pytest.raises(ValueError):
        Mask(boundary=1, outcome=True)


def test_mask_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HKJC_MASK", raising=False)
    monkeypatch.delenv("HKJC_FINAL_HOLDOUT", raising=False)
    assert Mask.from_env() == Mask()
    monkeypatch.setenv("HKJC_FINAL_HOLDOUT", "1")  # alias
    assert Mask.from_env() == Mask(boundary=1)
    run = RaceRun(_info(), root=tmp_path, log=lambda *a: None)  # no mask given: the environment's
    assert run.mask == Mask(boundary=1) and run.info.mask == Mask(boundary=1)
    monkeypatch.setenv("HKJC_MASK", "outcome")
    with pytest.raises(ValueError):
        Mask.from_env()
    monkeypatch.delenv("HKJC_FINAL_HOLDOUT")
    assert Mask.from_env() == Mask(outcome=True)


def test_masked_view_exposes_pre_race_facts_only():
    info = _info()
    for m in (Mask(), Mask(boundary=1), Mask(outcome=True)):
        v = MaskedRace(info, m)
        for name in MaskedRace.PRE_RACE:
            assert getattr(v, name) == getattr(info, name)
        for name in ("results", "sectionals"):
            with pytest.raises(AttributeError):
                getattr(v, name)


def test_no_mask_is_the_full_record():
    info = _info()
    v = MaskedRace(info)
    bnd, cum = v.official_boundaries()
    bnd0, cum0 = _cum_official(info)
    assert bnd == bnd0 == [800.0, 400.0, 0.0]
    assert list(cum) == list(cum0) == [1, 2, 3]
    assert all(np.array_equal(cum[h], cum0[h]) for h in cum)
    assert v.official_finish() == official_finish(info) and list(v.official_finish()) == ORDER
    assert v.official_order() == official_order(info) == ORDER


@pytest.mark.parametrize("j", [0, 1, 2])
def test_withheld_boundary_is_hidden_exactly(j):
    info = _info()
    v = MaskedRace(info, Mask(boundary=j))
    bnd, cum = v.official_boundaries()
    bnd0, cum0 = official_boundaries(info)
    assert bnd == bnd0
    for h in cum0:
        assert np.isnan(cum[h][j])
        keep = np.arange(len(bnd)) != j
        assert np.array_equal(cum[h][keep], cum0[h][keep])
        assert np.isfinite(cum0[h][j])  # the full record is untouched
    last = j == len(bnd) - 1
    assert v.official_finish() == ({} if last else official_finish(info))
    assert v.official_order() == ([] if last else ORDER)


def test_withheld_outcome_hides_everything():
    v = MaskedRace(_info(), Mask(outcome=True))
    assert v.official_boundaries() == ([], {})
    assert v.official_finish() == {} and v.official_order() == []


def _official_points(obs):
    return {(o.horse, float(d)) for o in obs if o.feed == "official" for d in D - o.s}


def test_official_observations_follow_the_mask():
    info = _info()
    full = _bare(info, Mask()).official_observations()
    assert _official_points(full) == {(h, d) for h in SECTIONS for d in (800.0, 400.0, 0.0)}  # d to go
    assert sorted(o.horse for o in full if o.feed == "draw") == [1, 2, 3]
    held = _bare(info, Mask(boundary=1)).official_observations()  # boundary 1 = 400 m to go
    assert _official_points(held) == {(h, d) for h in SECTIONS for d in (800.0, 0.0)}
    out = _bare(info, Mask(outcome=True)).official_observations()
    assert [o.feed for o in out] == ["draw"] * 3
    assert [(o.horse, float(o.lat[0])) for o in out] == [(r.no, f.lat[0]) for r, f in zip(info.runners, full)]


def _paths():
    t = np.linspace(0, 80, 1601)
    return {
        h: Path(h, t, v * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
        for h, v in ((1, 17.2), (2, 17.0), (3, 17.4))
    }


def test_retime_follows_the_mask():
    info = _info()
    p = _paths()
    q, warp = _bare(info, Mask(outcome=True))._official_retime(p)
    assert warp == {} and all(q[h] is p[h] for h in p)  # no official record: a no-op
    q, _w = _bare(info, Mask(boundary=1))._official_retime(p)
    _b, cum = official_boundaries(info)
    for h, c in cum.items():
        x = q[h]
        assert abs(crossing(x.t, D - x.s, None, 400.0) - c[1]) > 0.05  # withheld: not snapped
        assert abs(crossing(x.t, D - x.s, None, 800.0) - c[0]) < 5e-3
        assert abs(crossing(x.t, D - x.s, None, 0.0) - c[2]) < 5e-3
    assert _bare(info, Mask(outcome=True))._metric_bias(p) is None


def test_validate_scores_against_the_full_record(tmp_path):
    info = _info()
    run = RaceRun(info, root=tmp_path, log=lambda *a: None, mask=Mask(outcome=True))
    assert run.truth is info and run.info.official_boundaries() == ([], {})
    _b, cum = official_boundaries(info)
    t = np.linspace(0, 80, 1601)
    recs = []
    for h, c in cum.items():  # paths crossing every boundary exactly on the official time
        d = np.interp(t, np.r_[0.0, c], [D, 800.0, 400.0, 0.0], right=-50.0)
        recs.append(pl.DataFrame({"horse_no": h, "t": t, "d": d, "sd_s": np.full_like(t, 0.2)}))
    P = pl.concat(recs)
    P.write_parquet(run.p("paths_video.parquet"))
    P.write_parquet(run.p("paths.parquet"))
    np.savez(run.p("tracks.npz"), t=P["t"].to_numpy(), d=P["d"].to_numpy())
    np.savez(run.p("metric.npz"))
    run.validate()
    res = json.loads(run.p("validation.json").read_text())
    assert res["winner"] == {"official": ORDER[0], "reconstructed": ORDER[0]}
    assert res["sectional_coverage"] == 1.0 and res["sectional_mae"] < 0.02  # grid interpolation
    assert res["finish_n"] == 3 and res["finish_rho"] == pytest.approx(1.0)


def test_recon_reads_outcome_only_through_the_masked_view():
    root = FsPath(__file__).parents[1] / "src/hkjc/recon"
    pat = re.compile(
        r"\.results\b|\.sectionals\b|_cum_official\(|FinishTime|RunningPosition|environ\.get\(\"HKJC_FINAL_HOLDOUT"
        r"|\._info\b|\.truth\b"
    )
    hits = [
        line.strip()
        for p in sorted(root.glob("*.py"))
        if p.name != "validate.py"
        for line in p.read_text().splitlines()
        if pat.search(line)
    ]
    assert hits == ["self.truth = info", "res = _v.validate(self, self.truth)"]


def test_run_dir_refuses_artifacts_of_another_mask(tmp_path):
    info = _info()
    plain = RaceRun(info, root=tmp_path, log=lambda *a: None, mask=Mask())
    assert plain.outcome_stages() == {"fuse_final", "validate"}
    plain.p("paths.parquet").write_bytes(b"x")  # unmasked product
    masked = RaceRun(info, root=tmp_path, log=lambda *a: None, mask=Mask(outcome=True))
    with pytest.raises(ValueError, match="own root"):
        masked.mask_guard(["fuse_final"])
    plain.mask_guard(["fuse_final"])  # same mask: fine, and no record for the empty mask
    assert not plain.p("mask.txt").exists()
    other = tmp_path / "masked"
    m2 = RaceRun(info, root=other, log=lambda *a: None, mask=Mask(boundary=1))
    m2.mask_guard(["fuse_final", "validate"])
    assert m2.p("mask.txt").read_text().strip() == "boundary=1"
    m2.p("paths.parquet").write_bytes(b"x")
    with pytest.raises(ValueError):
        RaceRun(info, root=other, log=lambda *a: None, mask=Mask()).mask_guard(["validate"])
