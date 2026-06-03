"""Offline test for the dividend-table parser (mirrors the real page layout)."""
from bs4 import BeautifulSoup

from hkjc.data.historical import _parse_dividend_table

# PLACE uses a row-spanned pool cell across its three combinations, like the site.
HTML = """
<table>
  <tr><td>Dividend</td></tr>
  <tr><th>Pool</th><th>Winning Combination</th><th>Dividend (HK$)</th></tr>
  <tr><td>WIN</td><td>12</td><td>50.00</td></tr>
  <tr><td rowspan="3">PLACE</td><td>12</td><td>22.50</td></tr>
  <tr><td>2</td><td>46.50</td></tr>
  <tr><td>9</td><td>16.00</td></tr>
  <tr><td>QUINELLA</td><td>2,12</td><td>413.00</td></tr>
  <tr><td>TRIO</td><td>2,9,12</td><td>583.00</td></tr>
  <tr><td>QUARTET</td><td>12,2,9,11</td><td>39,100.00</td></tr>
</table>
"""


def _df():
    table = BeautifulSoup(HTML, "html.parser").find("table")
    return _parse_dividend_table(table)


def test_pool_codes_and_forward_fill():
    df = _df()
    assert df.filter(df["pool"] == "WIN").height == 1
    # the row-spanned PLACE label is forward-filled to all three combinations
    assert df.filter(df["pool"] == "PLA").height == 3
    assert set(df["pool"].to_list()) == {"WIN", "PLA", "QIN", "TRI", "QTT"}


def test_dividend_and_decimal_values():
    df = _df()
    win = df.filter(df["pool"] == "WIN").row(0, named=True)
    assert win["dividend"] == 50.0 and win["decimal"] == 5.0      # per HK$10
    qin = df.filter(df["pool"] == "QIN").row(0, named=True)
    assert qin["combination"] == "2,12" and qin["dividend"] == 413.0
    assert abs(qin["decimal"] - 41.3) < 1e-9
    qtt = df.filter(df["pool"] == "QTT").row(0, named=True)
    assert qtt["dividend"] == 39100.0   # comma-stripped
