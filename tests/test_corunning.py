"""Offline tests for the running-comments parser and trouble classifier."""
from bs4 import BeautifulSoup

from hkjc.data.corunning import _parse_corunning_table, has_trouble

HTML = """
<table>
  <tr><th>Placing</th><th>Horse No</th><th>Horse Name</th><th>Jockey</th><th>Gear</th><th>Comment</th></tr>
  <tr><td>1</td><td>5</td><td>BRIGHT DAY (J530)</td><td>K C Leung</td><td>V/TT</td>
      <td>Raced in touch, clear run up inside rail, comfortably on top closing stages.</td></tr>
  <tr><td>4</td><td>8</td><td>GRACEFUL HEART (L207)</td><td>L Hewitson</td><td>CP</td>
      <td>Held up in last pair, blocked for run early straight, no clear run final 200m.</td></tr>
  <tr><td>7</td><td>2</td><td>SOME HORSE (K252)</td><td>C L Chau</td><td>TT</td>
      <td>Began awkwardly, short of room near 300m, kept on once clear.</td></tr>
</table>
"""


def test_trouble_classifier():
    assert has_trouble("Blocked for run, no clear run final 200m.") == 1
    assert has_trouble("Short of room near the 300m.") == 1
    assert has_trouble("Checked and steadied 400m out.") == 1
    assert has_trouble("Raced in touch, clear run, won well.") == 0
    assert has_trouble("Led throughout, kept on strongly.") == 0
    assert has_trouble("") == 0


def test_parse_corunning_table_flags_trouble():
    table = BeautifulSoup(HTML, "html.parser").find("table")
    rows = _parse_corunning_table(table)
    assert len(rows) == 3
    by_runner = {r["runner"]: r for r in rows}
    assert by_runner[5]["horse_id"] == "J530" and by_runner[5]["trouble"] == 0   # clear run
    assert by_runner[8]["trouble"] == 1   # blocked / no clear run
    assert by_runner[2]["trouble"] == 1   # short of room
