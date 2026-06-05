"""Offline tests for the barrier-trial parser (mirrors the real page layout)."""
from hkjc.data.barrier_trials import (
    _lbw_to_lengths,
    _parse_html,
    _time_to_sec,
    _to_ddmmyyyy,
    _to_iso,
)

# A batch-header table followed by a horse-rows table, like the live page.
HTML = b"""
<table><tr><td>Batch 1 - SHA TIN ALL WEATHER TRACK - 1050m
  Going: GOOD Time: 1.10.80 Sectional Time: 25.3 23.1 22.4</td></tr></table>
<table>
  <tr><th>Horse</th><th>Jockey</th><th>Trainer</th><th>Draw</th><th>Gear</th>
      <th>LBW</th><th>Running Position</th><th>Time</th><th>Result</th><th>Comment</th></tr>
  <tr>
    <td><a href="/horse/x?horseid=HK_2024_K400">AEROVOLANIC (K400)</a></td>
    <td>J Doe</td><td>T Smith</td><td>1</td><td>--</td>
    <td></td><td>1 1 1</td><td>1.10.80</td><td>Passed</td><td>Led</td>
  </tr>
  <tr>
    <td><a href="/horse/x?horseid=HK_2024_K136">WITH A SMILE (K136)</a></td>
    <td>J Roe</td><td>T Jones</td><td>2</td><td>B</td>
    <td>8-1/2L</td><td>2 2 2</td><td>1.12.93</td><td>Passed</td><td>Chased</td>
  </tr>
</table>
"""


def test_date_helpers():
    assert _to_ddmmyyyy("2026-06-05") == "05/06/2026"
    assert _to_ddmmyyyy("05/06/2026") == "05/06/2026"
    assert _to_iso("05/06/2026") == "2026-06-05"
    assert _to_iso("2026/06/05") == "2026-06-05"


def test_time_to_sec():
    assert abs(_time_to_sec("1.10.80") - 70.80) < 1e-9   # m.ss.ff
    assert _time_to_sec("23.10") == 23.10                # ss.ff
    assert _time_to_sec("") is None


def test_lbw_to_lengths():
    assert _lbw_to_lengths("") == 0.0          # the leader
    assert _lbw_to_lengths("8L") == 8.0
    assert _lbw_to_lengths("3/4L") == 0.75
    assert abs(_lbw_to_lengths("8-1/2L") - 8.5) < 1e-9
    assert abs(_lbw_to_lengths("8 1/2L") - 8.5) < 1e-9
    assert _lbw_to_lengths("N") == 0.3         # named short margin


def test_parse_html_batch_context_and_rows():
    rows = _parse_html(HTML, "2026-06-05")
    assert len(rows) == 2
    lead = rows[0]
    assert lead["horse_brand"] == "K400" and lead["horse_id"] == "HK_2024_K400"
    assert lead["horse"] == "AEROVOLANIC"
    assert lead["distance"] == 1050 and lead["going"] == "GOOD"
    assert lead["batch"] == 1 and abs(lead["batch_time"] - 70.80) < 1e-9
    assert lead["date"] == "2026-06-05"
    assert lead["lbw_lengths"] == 0.0          # leader, blank LBW
    assert abs(lead["trial_time"] - 70.80) < 1e-9
    second = rows[1]
    assert abs(second["lbw_lengths"] - 8.5) < 1e-9
    assert abs(second["trial_time"] - 72.93) < 1e-9
