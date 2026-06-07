"""Behavioural lock for the promoted ``hkjc.reconstruction`` fusion half.

(a) the three new modules import with NO heavy CV deps (pure numpy; scipy is lazy in the box step
    only — but ``smooth_field`` does call it, so the synthetic test exercises scipy intentionally).
(b) a SYNTHETIC unit test of ``fusion.smooth_field``: a tiny fabricated Race (3 runners, 1200m HV C)
    + hand-built ``Obs`` sequences on the 5 Hz grid -> run the smoother -> assert sane output
    (monotone-ish arclen, y_lat within [0, W], finite sds, correct shapes). No video.

This locks the fusion math behaviourally — import via FULL module paths (never package-level
re-exports, which may be mid-restructure).
"""
from __future__ import annotations

import numpy as np

from hkjc.core.race import Race, Runner
from hkjc.reconstruction.clock import ClockAnchors, anchors_from_recipe
from hkjc.reconstruction.fusion import DT, Obs, smooth_field
from hkjc.reconstruction.geometry import RaceGeom


# ----------------------------------------------------------------------------- imports / no-CV
def test_reconstruction_modules_import_without_cv_deps():
    import hkjc.reconstruction.clock as clock
    import hkjc.reconstruction.fusion as fusion
    import hkjc.reconstruction.reconstruct as reconstruct

    assert clock.ClockAnchors and clock.anchors_from_recipe and clock.section_curve
    assert fusion.Obs and fusion.smooth_field and fusion.apply_constraints and fusion.fit_levels
    assert reconstruct.RaceRecipe and reconstruct.Reconstructor


# ----------------------------------------------------------------------------- clock
def test_clock_section_curve_and_per_feed_maps():
    # 3 sections (a 1200m sprint), even arclen split.
    anc = anchors_from_recipe(
        1200.0, sections=(23.5, 22.6, 23.2), section_bounds=(0.0, 400.0, 800.0, 1200.0),
        patrol_offset=18.0, rf_offset=1.3,
        drone_gate_vid_t=3.0, drone_gate_race_t=0.0,
        drone_finishcut_vid_t=73.0, drone_finishcut_race_t=69.3,
    )
    assert isinstance(anc, ClockAnchors)
    # winner time == sum of sections; arclen<->race_t round-trips on the boundaries.
    assert abs(anc.winner_finish_race_t - 69.3) < 1e-6
    assert abs(float(anc.race_t_to_arclen(0.0))) < 1e-9
    assert abs(float(anc.arclen_to_race_t(1200.0)) - 69.3) < 1e-6
    # per-feed video -> race_t conventions.
    assert abs(float(anc.patrol_to_race_t(20.0)) - 2.0) < 1e-9
    assert abs(float(anc.rf_to_race_t(1.3)) - 0.0) < 1e-9
    assert abs(float(anc.drone_to_race_t(3.0)) - 0.0) < 1e-9
    assert abs(float(anc.drone_to_race_t(73.0)) - 69.3) < 1e-6


# ----------------------------------------------------------------------------- helpers
def _tiny_race() -> Race:
    """3 runners, 1200m, HV "C" course — width 22.5 m, draws 1/4/8."""
    runners = (
        Runner(no=1, draw=1, horse_id="A"),
        Runner(no=2, draw=4, horse_id="B"),
        Runner(no=3, draw=8, horse_id="C"),
    )
    return Race.from_runners(runners, date="2026-06-03", venue="HV", race_no=3,
                             distance=1200.0, course="C")


def _arclen_obs(rn, grid, vc, *, feed="aerial", sd=7.0):
    """A clean constant-velocity arclen track for one runner over the whole grid."""
    return [Obs(rn, float(t), float(vc * t), None, feed, sd, 0.0) for t in grid]


def _lateral_obs(rn, grid, y, *, feed="patrol", sd=1.2):
    """A constant lateral (metric anchor) read for one runner over the whole grid."""
    return [Obs(rn, float(t), None, float(y), feed, 6.0, sd) for t in grid]


# ----------------------------------------------------------------------------- fusion (synthetic)
def test_smooth_field_synthetic_sane_output():
    race = _tiny_race()
    geom = RaceGeom(race)
    width = geom.width
    assert abs(width - 22.5) < 1e-9   # sourced from core (HV C)

    grid = np.arange(0.0, 60.0 + DT, DT)   # ~60 s of a 1200m sprint at 5 Hz
    vc = 16.5

    # runner 1 (rail), runner 2 (mid), runner 3 (wide) — lateral anchors inside [0, W];
    # all three roughly co-located longitudinally (constant cruise).
    obs_by_runner = {
        1: _arclen_obs(1, grid, vc) + _lateral_obs(1, grid, 2.0),
        2: _arclen_obs(2, grid, vc) + _lateral_obs(2, grid, 9.0),
        3: _arclen_obs(3, grid, vc) + _lateral_obs(3, grid, 16.0),
    }
    anc = anchors_from_recipe(1200.0, sections=(23.5, 22.6, 23.2),
                              section_bounds=(0.0, 400.0, 800.0, 1200.0))
    out = smooth_field(obs_by_runner, grid, geom,
                       race_t_to_arclen=anc.race_t_to_arclen)

    assert set(out) == {1, 2, 3}
    for rn, r in out.items():
        n = len(grid)
        # shapes
        for key in ("race_t", "arclen", "y_lat", "v", "sd_arclen", "sd_ylat"):
            assert r[key].shape == (n,), f"{key} shape for runner {rn}"
        # finite everything
        assert np.all(np.isfinite(r["arclen"])) and np.all(np.isfinite(r["y_lat"]))
        assert np.all(np.isfinite(r["sd_arclen"])) and np.all(np.isfinite(r["sd_ylat"]))
        assert np.all(r["sd_arclen"] >= 0) and np.all(r["sd_ylat"] >= 0)
        # arclen monotone-ish (smooth cruise -> non-decreasing within numerical slack).
        d = np.diff(r["arclen"])
        assert d.min() > -1.0, f"runner {rn} arclen went backward by {d.min():.2f} m"
        # y_lat within the hard box [0, W] (the #96 truncated-Gaussian constraint).
        assert r["y_lat"].min() >= -1e-6 and r["y_lat"].max() <= width + 1e-6
        # fitted cruise level is in the plausible equine band.
        assert 13.0 <= r["vc_fit"] <= 19.0

    # the lateral ORDER is preserved (rail < mid < wide) on the body of the track.
    body = slice(int(len(grid) * 0.3), int(len(grid) * 0.7))
    y1 = np.median(out[1]["y_lat"][body])
    y2 = np.median(out[2]["y_lat"][body])
    y3 = np.median(out[3]["y_lat"][body])
    assert y1 < y2 < y3, f"lateral order broke: {y1:.1f} {y2:.1f} {y3:.1f}"
    # and the rail/wide gap survives (the S8 control-classification signal).
    assert (y3 - y1) > 5.0


def test_smooth_field_box_caps_unanchored_lateral():
    """A runner with NO lateral obs (only arclen) still respects the hard box [0, W]."""
    race = _tiny_race()
    geom = RaceGeom(race)
    grid = np.arange(0.0, 40.0 + DT, DT)
    obs_by_runner = {
        1: _arclen_obs(1, grid, 16.0),                              # arclen only — unanchored lat
        2: _arclen_obs(2, grid, 16.0) + _lateral_obs(2, grid, 11.0),
    }
    out = smooth_field(obs_by_runner, grid, geom)
    for r in out.values():
        assert r["y_lat"].min() >= -1e-6 and r["y_lat"].max() <= geom.width + 1e-6
