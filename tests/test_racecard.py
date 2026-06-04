"""Offline test for the racecard parser."""
from bs4 import BeautifulSoup

from hkjc.data.racecard import (
    _best_time_sec,
    _parse_racecard_table,
    _records_to_frame,
    _split_jockey,
)

HEADER = ("Horse No.|Last 6 Runs|Colour|Horse|Brand No.|Wt.|Jockey|Over Wt.|Draw|"
          "Trainer|Int'l Rtg.|Rtg.|Rtg.+/-|Horse Wt. (Declaration)|Wt.+/- (vs Declaration)|"
          "Best Time|Age|WFA|Sex|Season Stakes|Priority|Days since Last Run|Gear|Owner|"
          "Sire|Dam|Import Cat.").split("|")
ROW = ("1|2/2/4/8/3/3||FIREFOOT|J183|130|K Teetan||1|W Y So|-|40|0||-|1.48.01|5|-|g|"
       "1,353,750|1|32|V-/B2/TT|Timon Tai|Shooting To Win|High Class|PPG").split("|")
ROW2 = ("2|10/13|.|I EXCELLE|K502|130|H Y Yuen (-10)||5|P C Ng|-|40|-3||-||4|-|g|"
        "23,400|1|14|B-/H/TT|Owner|Sire|Dam|PPG").split("|")


def _table(*rows):
    html = "<table><tr>" + "".join(f"<th>{h}</th>" for h in HEADER) + "</tr>"
    for r in rows:
        html += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return BeautifulSoup(html + "</table>", "html.parser").find("table")


def test_helpers():
    assert _best_time_sec("1.48.01") == 108.01
    assert _split_jockey("H Y Yuen (-10)") == ("H Y Yuen", -10)
    assert _split_jockey("K Teetan") == ("K Teetan", 0)


def test_parse_and_type_racecard():
    df = _records_to_frame(_parse_racecard_table(_table(ROW, ROW2)))
    assert df.height == 2
    r1 = df.filter(df["runner"] == 1).row(0, named=True)
    assert r1["horse_id"] == "J183" and r1["horse"] == "FIREFOOT"
    assert r1["draw"] == 1 and r1["rtg"] == 40 and r1["carried_wt"] == 130
    assert r1["jockey"] == "K Teetan" and r1["jockey_allowance"] == 0
    assert r1["season_stakes"] == 1_353_750 and r1["best_time"] == 108.01
    assert r1["gear"] == "V-/B2/TT"
    # apprentice allowance split out; negative rating move typed
    r2 = df.filter(df["runner"] == 2).row(0, named=True)
    assert r2["jockey"] == "H Y Yuen" and r2["jockey_allowance"] == -10
    assert r2["rtg_change"] == -3
