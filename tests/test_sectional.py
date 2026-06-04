"""Offline test for the sectional-time parser (mirrors the real table layout)."""
from bs4 import BeautifulSoup

from hkjc.data.sectional import _parse_sectional_table, _to_ddmmyyyy, _time_to_sec

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
