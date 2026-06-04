"""Tests for the jackpot capture store and carryover ledger (M1)."""
import datetime as dt

import polars as pl

from hkjc.data.carryover_ledger import (
    CarryoverLedger,
    build_carryover_rows,
    settled_jackpot_result,
)
from hkjc.data.jackpot_ticks import JackpotCaptureStore


def _snapshot(jackpot, investment, est):
    """A live_jackpots-shaped snapshot for one TT leg."""
    return pl.DataFrame({
        "pool": ["TT"], "leg": [1], "races": ["4,5,6"],
        "investment": [float(investment)], "jackpot": [float(jackpot)],
        "estimated_dividend": [float(est)], "merged_pool_id": [None],
        "date": ["2026-05-31"], "venue": ["ST"],
    })


def _t(minute):
    return dt.datetime(2026, 5, 31, 12, minute, tzinfo=dt.timezone.utc)


def test_capture_store_records_timeseries(tmp_path):
    s = JackpotCaptureStore(tmp_path)
    s.record(_t(0), _snapshot(8_000_000, 3_000_000, 11_000_000))
    s.record(_t(30), _snapshot(8_000_000, 5_000_000, 13_000_000))   # pool grew
    df = s.read(date="2026-05-31", venue="ST", pool="TT")
    assert df.height == 2
    assert df["captured_at"].n_unique() == 2
    # latest() picks the close-proxy snapshot
    last = s.latest("2026-05-31", "ST")
    assert last.filter(pl.col("pool") == "TT")["investment"].item() == 5_000_000


def test_settled_jackpot_result():
    divs = pl.DataFrame({
        "pool": ["WIN", "TT"], "combination": ["3", "F/M"], "decimal": [4.0, 1_200_000.0],
    })
    won = settled_jackpot_result("TT", divs)
    assert won["won"] == 1 and won["settled_dividend"] == 1_200_000.0
    notwon = settled_jackpot_result("DT", divs)   # absent -> carried forward
    assert notwon["won"] == 0 and notwon["settled_dividend"] is None


def test_build_rows_computes_edge_and_joins_settlement(tmp_path):
    s = JackpotCaptureStore(tmp_path)
    s.record(_t(0), _snapshot(8_000_000, 3_000_000, 11_000_000))
    s.record(_t(30), _snapshot(8_000_000, 4_000_000, 12_000_000))
    cap = s.read(date="2026-05-31", venue="ST")
    settled = {"TT": settled_jackpot_result("TT", pl.DataFrame(
        {"pool": ["TT"], "combination": ["F/M"], "decimal": [900_000.0]}))}

    rows = build_carryover_rows(cap, settled)
    r = rows.row(0, named=True)
    assert r["n_snapshots"] == 2
    assert r["carry_in"] == 8_000_000 and r["gross_pool"] == 4_000_000   # last snapshot
    # C/G = 2.0 -> eff_takeout = 0.15 - 2.0 < 0 -> structurally +EV
    assert r["eff_takeout"] < 0 and r["agg_edge"] > 0
    assert r["won"] == 1 and r["settled_dividend"] == 900_000.0


def test_ledger_append_upsert_roundtrip(tmp_path):
    led = CarryoverLedger(tmp_path / "ledger.parquet")
    s = JackpotCaptureStore(tmp_path / "caps")
    s.record(_t(0), _snapshot(8e6, 3e6, 11e6))
    rows = build_carryover_rows(s.read(date="2026-05-31", venue="ST"))
    led.append(rows)
    assert led.read().height == 1
    # re-appending the same event upserts, not duplicates
    led.append(rows)
    assert led.read().height == 1
