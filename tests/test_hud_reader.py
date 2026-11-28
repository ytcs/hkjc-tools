"""ocr.read_render_hud_cached: every frame's odometer and legend read equals that frame's own OCR (no read skipped by a
pixel-change test), on a synthetic frame cache with a stub recogniser."""

import numpy as np

from hkjc.recon import ocr, video

BOX = (960, 1905, 270, 806)  # x0, x1, y0, y1 (a real silk-render panel box)
LT = 450  # legend top row (panel-relative)


class _Cache:
    def __init__(self, panels, clocks):
        self.box, self.clock, self._p = BOX, clocks, panels
        self.idx = np.arange(100, 100 + 2 * len(panels), 2)

    def __iter__(self):
        for i, p in zip(self.idx, self._p):
            yield int(i), p


def _stub(crops, scale=3):
    """Deterministic recogniser: the crop's text is the value of its pixel (5, 5, 0)."""
    out = []
    for c in crops:
        v = int(c[5, 5, 0])
        if c.shape[:2] == (54, 145):  # clock crop
            out.append((f"0:{v % 60:02d}.00", 0.99))
        elif c.shape[:2] == (34, 94):  # odometer crop
            out.append((f"{10 * v}M", 0.99))
        else:  # legend slot crop
            out.append((str(v), 0.99))
    return out


def _frames(n=9):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 200, (536, 945, 3), dtype=np.uint8)  # moving background: refreshed per frame below
    oy, ox = ocr.AVV_ODO[0] - BOX[2], ocr.AVV_ODO[2] - BOX[0]
    xs = ocr._legend_slot_x(945, 8)
    panels = []
    for k in range(n):
        p = base.copy()
        for s, cx in enumerate(xs):  # slot s reads s + 1; slot 3 changes to 9 at frame 4 (ONE pixel changes)
            p[LT - 4 + 5, cx - 22 + 5, 0] = 9 if (s == 3 and k >= 4) else s + 1
        p[oy + 5, ox + 5, 0] = 80 if k < 5 else 79  # odometer 800M -> 790M at frame 5 (one pixel)
        panels.append(p)
    clocks = np.zeros((n, 54, 145, 3), np.uint8)
    clocks[:, 5, 5, 0] = np.arange(n)
    return panels, clocks


def _own_reads(cache, n_field=14, n_slots=8):
    """Each frame read on its own (one OCR call per frame, nothing carried between frames)."""
    x0, _x1, y0, _y1 = cache.box
    oy0, oy1, ox0, ox1 = ocr.AVV_ODO
    odo, leg = [], []
    for i, p in cache:
        (t, c), = ocr.rec_batch([p[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0].copy()])
        v = ocr._parse_odometer(t, c)
        odo.append(np.nan if v is None else v)
        leg.append([i, *ocr._parse_legend(ocr.rec_batch(ocr._legend_slot_crops(p, n_slots)), n_field)])
    return np.array(odo), np.array(leg)


def test_hud_reads_equal_per_frame_reads(monkeypatch):
    monkeypatch.setattr(ocr, "rec_batch", _stub)
    monkeypatch.setattr(video, "legend_top", lambda panel: LT)
    panels, clocks = _frames()
    cache = _Cache(panels, clocks)
    rows, leg = ocr.read_render_hud_cached(cache)
    odo, own = _own_reads(cache)
    assert np.array_equal(leg, own)
    assert np.array_equal(rows[:, 2], odo)
    # the one-pixel changes are read on the frame they appear (a pooled change test would hold the previous frame's reads)
    assert leg[:, 4].tolist() == [4] * 4 + [9] * 5
    assert rows[:, 2].tolist() == [800.0] * 5 + [790.0] * 4
    # the clock: video-locked, every CLOCK_EVERY-th frame, NaN between
    assert rows[0, 1] == 0.0 and np.isnan(rows[1:, 1]).all()
    assert rows[:, 0].tolist() == cache.idx.tolist()


def test_parse_legend_normalises_circled_and_fullwidth_digits():
    from hkjc.recon.ocr import _parse_legend

    row = _parse_legend([("④", 0.99), ("⑪", 0.99), ("１２", 0.99), ("7", 0.99)], n_field=14)
    assert list(row[:4]) == [4, 11, 12, 7]
