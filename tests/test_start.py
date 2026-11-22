import numpy as np

from hkjc.recon import start
from hkjc.recon.pipeline import RaceRun


def _grid(T=120, open_at=50, closed=0.30, opened=0.05, cols=None, recover_at=None):
    """Synthetic (T, NR, NC) door-fraction grid: row 3 holds the doors, which open at ``open_at``."""
    G = np.full((T, start.NR, start.NC), 0.02)
    G[:, 3, :] = closed
    cols = range(start.NC) if cols is None else cols
    for c in cols:
        G[open_at:, 3, c] = opened
        if recover_at is not None:
            G[recover_at:, 3, c] = closed
    return G


def test_onset_finds_a_gate_opening_across_the_whole_width():
    f = start.onset(_grid(open_at=50), 0, 110)
    assert f is not None and 48.5 <= f <= 50.5


def test_onset_ignores_a_drop_confined_to_one_column():  # a person walking past the stalls
    assert start.onset(_grid(cols=[2]), 0, 110) is None


def test_onset_ignores_a_temporary_dip():  # doors stay open; a passing shadow does not
    assert start.onset(_grid(open_at=50, recover_at=56), 0, 110) is None


def test_onset_respects_the_search_window():
    assert start.onset(_grid(open_at=50), 60, 110) is None


class _Info:
    key = "20240101_ST_1"


def test_clock_f0_moves_back_by_the_measured_lag_only(tmp_path):
    run = RaceRun(_Info(), root=tmp_path)
    clock = np.arange(0.0, 40.0, 0.5)
    rows = np.c_[69.2 + 25.0 * clock, clock, np.full(len(clock), np.nan)]
    np.savez(run.p("hud.npz"), rows=rows)
    assert run.clock_f0() == run.overlay_f0()  # no start stage output: the overlay clock
    np.savez(run.p("start.npz"), clock0=69.2, onset=57.2, lag_frames=0.0, status="synced")
    assert run.clock_f0() == run.overlay_f0()  # synced: unchanged, exactly
    np.savez(run.p("start.npz"), clock0=69.2, onset=57.2, lag_frames=10.6, status="late")
    assert abs(run.clock_f0() - (run.overlay_f0() - 10.6)) < 1e-9


def test_door_hue_bands_separate_doors_from_turf_banner_and_sky():
    # (H, S, V) measured on start films: sunlit doors (ST 2025-07-13 R8), shaded teal doors (HV 2024-05-01 R5),
    # shaded turf (ST 2025-07-13 R8), sunlit turf, the blue banner and night trees (HV 2026-11-04 R3)
    px = np.array([[[83, 141, 85], [101, 131, 77], [48, 109, 108], [44, 89, 179], [113, 223, 68], [112, 210, 51]]])
    green, teal = (start.door_mask(px.astype(np.uint8), hue).ravel().tolist() for hue in start.DOOR_HUES)
    assert green == [True, False, True, False, False, False]
    assert teal == [True, True, False, False, False, False]


def test_lag_tries_the_teal_band_only_when_the_green_band_finds_no_opening(monkeypatch):
    flat = np.full((120, start.NR, start.NC), 0.3)
    monkeypatch.setattr(start, "door_grid", lambda path, nmax: np.stack([flat, _grid(open_at=50)]))
    r = start.lag("x.mp4", 69.0, "ST")
    assert r["status"] == "late" and 48.5 <= r["onset"] <= 50.5
    monkeypatch.setattr(start, "door_grid", lambda path, nmax: np.stack([_grid(open_at=60), _grid(open_at=50)]))
    assert 58.5 <= start.lag("x.mp4", 69.0, "ST")["onset"] <= 60.5  # the green band's opening wins
