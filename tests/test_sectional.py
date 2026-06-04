"""Offline test for the sectional-time parser (mirrors the real table layout)."""
import polars as pl
from bs4 import BeautifulSoup

from hkjc.data.sectional import (
    _parse_sectional_table,
    _time_to_sec,
    _to_ddmmyyyy,
    sectional_run_features,
)

HTML = """
<table>
  <tr><th>Finishing Order</th><th>Horse No.</th><th>Horse</th>
      <th>Running Position & Margin Behind Leader</th><th>Time</th></tr>
  <tr><td>Sectional Time</td></tr>
  <tr><td>1st Sec.</td><td>2nd Sec.</td><td>3rd Sec.</td></tr>
  <tr><td>1</td><td>9</td><td>YODA'S CHOICE&nbsp;(J146)</td>
      <td>1 SH 13.74</td><td>1 3/4 22.76</td><td>1 1-1/4 24.01 11.78 12.23</td>
      <td></td><td>1:49.43</td></tr>
  <tr><td>2</td><td>3</td><td>SOME HORSE&nbsp;(K021)</td>
      <td>5 2 14.10</td><td>4 1-1/2 22.90</td><td>2 3/4 24.05</td>
      <td></td><td>1:49.55</td></tr>
</table>
"""


def test_date_and_time_helpers():
    assert _to_ddmmyyyy("2026-05-06") == "06/05/2026"
    assert _to_ddmmyyyy("2026/05/06") == "06/05/2026"
    assert _to_ddmmyyyy("06/05/2026") == "06/05/2026"
    assert abs(_time_to_sec("1:49.43") - 109.43) < 1e-9
    assert _time_to_sec("13.74") == 13.74


def test_parse_per_horse_sections():
    table = BeautifulSoup(HTML, "html.parser").find("table")
    rows = _parse_sectional_table(table)
    # 2 horses x 3 non-empty sections each
    assert len(rows) == 6
    winner = [r for r in rows if r["runner"] == 9]
    assert winner[0]["horse_id"] == "J146"
    assert winner[0]["finish_order"] == 1
    assert abs(winner[0]["total_time"] - 109.43) < 1e-9
    assert winner[0]["section"] == 1 and winner[0]["position"] == 1
    assert winner[0]["section_time"] == 13.74
    # the closer started further back (position 5 in section 1)
    closer = [r for r in rows if r["runner"] == 3 and r["section"] == 1][0]
    assert closer["position"] == 5 and closer["section_time"] == 14.10


def test_sectional_run_features_fade_and_relatives():
    rows = []

    def add(runner, hid, fo, total, secs):
        for i, t in enumerate(secs, start=1):
            rows.append(dict(date="d", race_no=1, runner=runner, horse_id=hid,
                             finish_order=fo, total_time=total, section=i,
                             section_time=t, position=fo))

    add(1, "A", 1, 72.0, [24.0, 24.0, 24.0])   # steady
    add(2, "B", 2, 74.0, [23.0, 23.0, 28.0])   # fast early, big fade

    out = sectional_run_features(pl.DataFrame(rows))
    a = out.filter(pl.col("horse_id") == "A").row(0, named=True)
    b = out.filter(pl.col("horse_id") == "B").row(0, named=True)
    # fade = last/best: A steady ~1.0, B fades (28/23 ~ 1.22)
    assert abs(a["fade"] - 1.0) < 1e-9 and b["fade"] > 1.2
    # last-section field mean = 26 -> A faster (-2), B slower (+2)
    assert a["rel_finish"] < 0 and b["rel_finish"] > 0
    # first-section field mean = 23.5 -> A slower early (+0.5), B faster early (-0.5)
    assert a["rel_early"] > 0 and b["rel_early"] < 0
    assert a["n_sections"] == 3
