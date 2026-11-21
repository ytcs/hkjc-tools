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
