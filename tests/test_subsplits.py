"""Official 200 m sub-splits: parsed from the sectional page's cells, read through a separate accessor, masked
with the boundaries they border."""

import numpy as np
import polars as pl
import pytest

from hkjc.data import Mask, MaskedRace, RaceInfo, Runner
from hkjc.data.race import official_boundaries, official_subsplits
from hkjc.data.sectional import parse_sectional_page

D = 1200.0  # sections: 400 (plain cell) + 400 + 400 (cells with sub-splits); points at 600 and 200 m to go


def _plain(pos, time):
    return f'<td><p class="f_clear"><span class="f_fl">{pos}</span><i>1</i></p><p>{time}</p></td>'


def _split(pos, time, a, b):
    return (f'<td><p class="f_clear"><span class="f_fl">{pos}</span><i>1</i></p><p class="sectional_200">{time}'
            f'<span class="color_blue2"><span style="width: 28px;">{a}</span>&nbsp;&nbsp;'
            f'<span style="width: 28px;">{b}</span></span></p></td>')


def _partial(a):  # a non-finisher's last cell: the first 200 m only, no position, no section time
    return (f'<td><p class="f_clear"><span class="f_fl"></span><i></i></p><p class="sectional_200"><br/>'
            f'<span class="color_blue2"><span style="width: 28px;">{a}</span>&nbsp;&nbsp;'
            f'<span style="width: 28px;"></span></span></p></td>')


BLANK = '<td><img src="blank.gif"/></td>'
ROWS = [  # placing, horse, cells, total
    ("1", 2, [_plain(1, "24.00"), _split(1, "22.50", "11.10", "11.40"), _split(1, "23.50", "11.50", "12.00")], "1:10.00"),
    ("2", 1, [_plain(2, "24.20"), _split(2, "22.60", "11.20", "11.40"), _split(2, "23.80", "11.70", "12.10")], "1:10.60"),
    ("UR", 3, [_plain(3, "24.40"), _split(3, "22.80", "11.30", "11.50"), _partial("11.90")], "---"),
]


def _html():
    head = ('<table><tr><td>Finishing Order</td><td>Horse No.</td><td>Horse</td><td>Running Position</td><td>Time</td></tr>'
            '<tr><td>1st Sec.</td><td>2nd Sec.</td><td>3rd Sec.</td></tr>')
    body = "".join(
        f"<tr><td>{p}</td><td>{h}</td><td>HORSE {h}&nbsp;(X{h:03d})</td>{''.join(c)}{BLANK}<td>{tot}</td></tr>"
        for p, h, c, tot in ROWS
    )
    return f"<html><body>{head}{body}</table></body></html>"


def _info(sec) -> RaceInfo:
    res = pl.DataFrame({"Pla": [1, 2, None], "HorseNo": [2, 1, 3], "FinishTime": [70.0, 70.6, None]})
    runners = tuple(Runner(h, f"HORSE {h}", f"X{h:03d}", h) for h in (1, 2, 3))
    return RaceInfo("20260603", "HV", 8, D, "C", "turf", "GOOD", "Class 4", runners, res, sec)


def test_page_parses_sub_splits_on_section_rows():
    sec = parse_sectional_page(_html(), "03/06/2026", 8)
    assert sec.height == 8  # the partial cell gives no row
    assert sec.filter(pl.col("section") == 1)[["sub1", "sub2"]].null_count().row(0) == (3, 3)
    r = sec.filter((pl.col("runner") == 2) & (pl.col("section") == 3)).row(0, named=True)
    assert (r["section_time"], r["sub1"], r["sub2"]) == (23.5, 11.5, 12.0)
    full = sec.drop_nulls(["sub1", "sub2"])
    assert np.allclose(full["sub1"] + full["sub2"], full["section_time"])
    assert sec.filter(pl.col("runner") == 3)["section"].to_list() == [1, 2]


def test_subsplit_points_and_times():
    info = _info(parse_sectional_page(_html(), "03/06/2026", 8))
    pts, cum = official_subsplits(info)
    assert pts == [600.0, 200.0]
    assert np.allclose(cum[2], [24.0 + 11.1, 24.0 + 22.5 + 11.5])
    assert np.allclose(cum[1], [24.2 + 11.2, 24.2 + 22.6 + 11.7])
    assert np.isclose(cum[3][0], 24.4 + 11.3) and np.isnan(cum[3][1])  # non-finisher: no section-3 row


def test_boundaries_unchanged_by_the_sub_split_columns():
    sec = parse_sectional_page(_html(), "03/06/2026", 8)
    bnd, cum = official_boundaries(_info(sec))
    bnd0, cum0 = official_boundaries(_info(sec.drop(["sub1", "sub2"])))
    assert bnd == bnd0 == [800.0, 400.0, 0.0]
    assert all(np.array_equal(cum[h], cum0[h], equal_nan=True) for h in cum0)


def test_sectionals_without_sub_split_columns_give_nothing():
    info = _info(parse_sectional_page(_html(), "03/06/2026", 8).drop(["sub1", "sub2"]))
    assert official_subsplits(info) == ([], {})
    assert MaskedRace(info).official_subsplits() == ([], {})


@pytest.mark.parametrize("mask, hidden", [
    (Mask(), []),
    (Mask(boundary=0), [0]),       # 800 m: the 600 m point borders it
    (Mask(boundary=1), [0, 1]),    # 400 m: both points border it
    (Mask(boundary=2), [1]),       # the finish: the 200 m point borders it
    (Mask(boundary=(0, 2)), [0, 1]),  # two boundaries: each hides its own points
])
def test_subsplits_follow_the_mask(mask, hidden):
    info = _info(parse_sectional_page(_html(), "03/06/2026", 8))
    pts0, cum0 = official_subsplits(info)
    pts, cum = MaskedRace(info, mask).official_subsplits()
    assert pts == pts0
    keep = [k for k in range(len(pts)) if k not in hidden]
    for h in cum0:
        assert np.isnan(cum[h][hidden]).all()
        assert np.array_equal(cum[h][keep], cum0[h][keep], equal_nan=True)


def test_withheld_outcome_hides_the_sub_splits():
    info = _info(parse_sectional_page(_html(), "03/06/2026", 8))
    assert MaskedRace(info, Mask(outcome=True)).official_subsplits() == ([], {})
