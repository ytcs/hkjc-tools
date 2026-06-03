"""Network-free tests for the odds tick store."""
import datetime as dt

from hkjc.data.odds_ticks import OddsTickStore


def _rows(win: dict) -> list:
    return [{"HorseID": s, "Type": "WIN", "Odds": o, "LastUpdate": "t"}
            for s, o in win.items()]


def _t(minute: int) -> dt.datetime:
    return dt.datetime(2026, 6, 3, 12, minute, tzinfo=dt.timezone.utc)


def test_record_read_roundtrip(tmp_path):
    s = OddsTickStore(tmp_path)
    df = s.record(_t(0), "2026-06-03", "HV", 1, _rows({"1": 5.0, "2": 3.0}))
    assert df.height == 2
    back = s.read(date="2026-06-03", venue="HV", race=1)
    assert back.height == 2
    assert set(back["selection"].to_list()) == {"1", "2"}
    assert back["pool"].to_list() == ["WIN", "WIN"]


def test_dedup_skips_unchanged_market(tmp_path):
    s = OddsTickStore(tmp_path)
    s.record(_t(0), "2026-06-03", "HV", 1, _rows({"1": 5.0, "2": 3.0}))
    s.record(_t(1), "2026-06-03", "HV", 1, _rows({"1": 5.0, "2": 3.0}))  # unchanged
    one = s.read(date="2026-06-03", venue="HV", race=1)
    assert one["captured_at"].n_unique() == 1

    s.record(_t(2), "2026-06-03", "HV", 1, _rows({"1": 4.5, "2": 3.0}))  # moved
    two = s.read(date="2026-06-03", venue="HV", race=1)
    assert two["captured_at"].n_unique() == 2


def test_read_filters_and_empty(tmp_path):
    s = OddsTickStore(tmp_path)
    assert s.read().is_empty()
    s.record(_t(0), "2026-06-03", "HV", 1, _rows({"1": 5.0}))
    s.record(_t(0), "2026-06-03", "HV", 2, _rows({"1": 2.0}))
    assert s.read(race=1).height == 1
    assert s.read(race=2).height == 1
    assert s.read().height == 2


def test_null_odds_dropped(tmp_path):
    s = OddsTickStore(tmp_path)
    rows = [{"HorseID": "1", "Type": "WIN", "Odds": 5.0, "LastUpdate": "t"},
            {"HorseID": "2", "Type": "WIN", "Odds": None, "LastUpdate": "t"}]
    df = s.record(_t(0), "2026-06-03", "HV", 1, rows)
    assert df.height == 1
