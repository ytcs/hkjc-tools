"""Backtest a Triple Trio Flexi-cover at realized dividends (carryover Gate-2).

Non-look-ahead test: for each historically WON Triple Trio, build a win-odds leg
model per leg, apply a FIXED cover rule (top-k box per leg — decided from pre-race
odds only), and settle at the ACTUAL declared dividend. PnL = (winning combo
covered ? settled_dividend : 0)·stake − n_combos·stake. Aggregated over events this
asks: would a simple win-odds cover have been profitable at real TT dividends?

Caveats: only WON TTs are observable from result pages (carried meetings are
invisible here), and small-stake full-dividend collection is assumed (dilution
ignored — optimistic). So this is an upper-ish bound / hit-rate study, not the
final word; the forward flywheel gives the pre-close economics. Network-heavy.

Run: uv run python experiments/backtest_carryover_cover.py [start] [end] [k]
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

from hkjc.bets.carryover_cover import leg_horses_by_place, top3_coverage
from hkjc.data.historical import get_race_data
from hkjc.data.jackpot_history import scan_jackpot_history
from hkjc.model import PlackettLuceModel
from hkjc.simulation import RaceSimulator


def _parse_legs(combo: str) -> list[set[int]]:
    legs = []
    for leg in combo.split("/"):
        nums = [int(x) for x in leg.replace(">", ",").split(",") if x.strip().isdigit()]
        legs.append(set(nums))
    return legs


def _meeting_top3(date: str, venue: str, max_races: int = 11) -> dict[int, set[int]]:
    out = {}
    for r in range(1, max_races + 1):
        try:
            rd = get_race_data(date, venue, r)
        except Exception:
            break
        top = rd.filter(pl.col("Pla").is_in([1, 2, 3]))["HorseNo"].to_list()
        out[r] = set(int(x) for x in top)
    return out


def _leg_sim(date: str, venue: str, race: int, ns: int = 40_000):
    rd = get_race_data(date, venue, race)
    odds = rd["WinOdds"].to_numpy().astype(float)
    inv = np.where((odds > 0) & np.isfinite(odds), 1.0 / odds, np.nan)
    floor = (np.nanmin(inv[inv > 0]) * 0.5) if np.any(inv > 0) else 1.0
    strengths = np.clip(np.nan_to_num(inv, nan=floor), 1e-6, None)
    sim = RaceSimulator(PlackettLuceModel(strengths), n_samples=ns)
    return sim, [int(x) for x in rd["HorseNo"].to_list()]


def backtest(start: str, end: str, k: int = 5, stake: float = 1.0):
    tt = scan_jackpot_history(start, end).filter(pl.col("pool") == "TT")
    print(f"Triple Trio won events {start}..{end}: {tt.height}")
    rows = []
    for ev in tt.iter_rows(named=True):
        date_slash = ev["date"]                       # YYYY/MM/DD
        venue, combo, dividend = ev["venue"], ev["combination"], ev["decimal"]
        legs = _parse_legs(combo)
        if len(legs) != 3:
            continue
        top3 = _meeting_top3(date_slash, venue)
        leg_races = [next((r for r, s in top3.items() if s == leg), None) for leg in legs]
        if any(r is None for r in leg_races):
            continue   # couldn't match legs to races (dead-heat / scratching quirks)

        n_combos, hit, cost = 1, True, 0
        from math import comb
        for race, leg in zip(leg_races, legs):
            sim, runners = _leg_sim(date_slash, venue, race)
            topk_idx = leg_horses_by_place(sim, min(k, sim.n))
            topk_runners = {runners[i] for i in topk_idx}
            n_combos *= comb(min(k, sim.n), 3)
            hit = hit and leg.issubset(topk_runners)
        cost = n_combos * stake
        payout = dividend * stake if hit else 0.0
        rows.append({"date": date_slash, "venue": venue, "dividend": dividend,
                     "n_combos": n_combos, "cost": cost, "hit": int(hit),
                     "pnl": payout - cost})

    if not rows:
        print("No settleable events.")
        return
    df = pl.DataFrame(rows)
    print(df.select(["date", "venue", "dividend", "n_combos", "cost", "hit", "pnl"]))
    print(f"\ntop-{k} box cover over {df.height} TT events:")
    print(f"  hit rate : {df['hit'].mean():.2f}  ({df['hit'].sum()}/{df.height})")
    print(f"  total PnL: {df['pnl'].sum():+,.0f}  (staked {df['cost'].sum():,.0f})")
    print(f"  ROI      : {df['pnl'].sum()/df['cost'].sum():+.3f}")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-03-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-05-31"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    backtest(start, end, k)
