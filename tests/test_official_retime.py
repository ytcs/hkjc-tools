"""Final product: paths are re-timed onto the official boundary times."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from hkjc.recon.fusion import Path
from hkjc.recon.pipeline import RaceRun
from hkjc.recon.validate import crossing


def test_retime_puts_every_official_boundary_exactly_on_time():
    secs = pl.DataFrame({"runner": [1, 1, 1], "section": [1, 2, 3], "section_time": [24.0, 23.0, 23.5]})
    res = pl.DataFrame({"HorseNo": [1], "Pla": [1], "FinishTime": [70.5]})
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.linspace(0, 80, 1601)
    p = Path(1, t, 15.5 * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
    run = RaceRun.__new__(RaceRun)
    run.info = info
    out, warp = run._official_retime({1: p})
    q = out[1]
    for b, tc in zip((800.0, 400.0, 0.0), np.cumsum([24.0, 23.0, 23.5])):
        assert abs(crossing(q.t, 1200.0 - q.s, None, b) - tc) < 5e-3  # grid resampling (0.05 s nodes)
    assert np.all(np.diff(q.s) >= 0) and warp[1] > 0


def test_equal_official_times_keep_the_official_placing():
    secs = pl.DataFrame({"runner": [1, 2], "section": [1, 1], "section_time": [70.5, 70.5]})
    res = pl.DataFrame({"HorseNo": [2, 1], "Pla": [1, 2], "FinishTime": [70.5, 70.5]})  # #2 won on the photo
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.linspace(0, 80, 1601)

    def mk(h, v):
        return Path(h, t, v * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))

    run = RaceRun.__new__(RaceRun)
    run.info = info
    out, _w = run._official_retime({1: mk(1, 17.1), 2: mk(2, 16.9)})
    c = {h: crossing(q.t, 1200.0 - q.s, None, 0.0) for h, q in out.items()}
    assert c[2] < c[1] and c[1] - c[2] < 0.01
