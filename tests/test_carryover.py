"""Tests for the jackpot carryover detector and live jackpot parsing."""
import math

from hkjc.bets import carryover as co
from hkjc.data.live import _parse_jackpots


def test_effective_takeout_and_edge():
    # net_fraction 0.85 -> break-even carry ratio = 0.15
    assert abs(co.effective_takeout(0.0, 1_000_000) - 0.15) < 1e-12          # no carry: 15% take
    assert abs(co.effective_takeout(150_000, 1_000_000) - 0.0) < 1e-12        # C/G = 0.15: break-even
    assert co.effective_takeout(300_000, 1_000_000) < 0                       # big carry: negative take
    # aggregate_edge is the negative of effective takeout
    assert abs(co.aggregate_edge(300_000, 1_000_000) + co.effective_takeout(300_000, 1_000_000)) < 1e-12


def test_effective_takeout_guards_zero_pool():
    assert math.isnan(co.effective_takeout(100.0, 0.0))


def test_flag_carryover_requires_margin():
    # C/G = 0.20 -> aggregate edge = 0.85 + 0.20 - 1 = 0.05
    f = co.flag_carryover(200_000, 1_000_000, margin=0.0)
    assert f.is_positive and abs(f.aggregate_edge - 0.05) < 1e-12
    # demand a 10% cushion for dilution/cover-miss -> no longer flagged
    assert not co.flag_carryover(200_000, 1_000_000, margin=0.10).is_positive


def test_parse_jackpots_joins_jackpot_and_investment():
    meeting = {
        "jpEsts": [
            {"oddsType": "TT", "jackpot": 8_000_000.0, "estimatedDividend": 12_000_000.0,
             "leg": {"number": 1, "races": [4, 5, 6]}},
        ],
        "poolInvs": [
            {"oddsType": "TT", "investment": 5_000_000.0, "mergedPoolId": None,
             "leg": {"number": 1, "races": [4, 5, 6]}},
            {"oddsType": "WIN", "investment": 9_000_000.0, "leg": {"races": [1]}},
        ],
    }
    df = _parse_jackpots(meeting)
    tt = df.filter(df["pool"] == "TT").row(0, named=True)
    assert tt["jackpot"] == 8_000_000.0          # carry-in C
    assert tt["investment"] == 5_000_000.0        # today's gross G
    assert tt["races"] == "4,5,6"
    # C/G = 1.6 -> wildly +EV window
    assert co.flag_carryover(tt["jackpot"], tt["investment"]).is_positive
    # WIN row has no jpEsts match -> missing jackpot (NaN)
    assert math.isnan(df.filter(df["pool"] == "WIN").row(0, named=True)["jackpot"])
