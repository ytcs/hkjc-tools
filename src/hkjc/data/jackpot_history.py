"""Historical jackpot-pool outcomes from result pages (B13, outcome side).

Result pages show a jackpot pool's dividend on its final leg **only when it was
won** — so scanning meetings for DT/TT/6UP dividend rows yields the *won-event*
record (which meetings paid, what dividend, the winning combination). Carryover
(not-won) meetings are inferred by absence over cards that offered the pool.

This is the reliable, immediately-scrapeable half of the carryover frequency
dataset. The carried-in amount ``C`` (and "offered but not won") needs the Racing
News pre-meeting announcements — a separate, fuzzier scrape (TODO, informed by the
site catalog). Reuses the tested :func:`hkjc.data.historical.get_race_dividends`.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Sequence

import polars as pl

from .historical import get_race_dividends
from .live import JACKPOT_POOLS


def scrape_meeting_jackpots(date: str, venue: str, max_races: int = 11) -> pl.DataFrame:
    """Won jackpot-pool dividends for one meeting (``date`` as ``YYYY/MM/DD``).

    Returns rows ``date/venue/race_no/pool/combination/dividend/decimal`` for the
    DT/TT/6UP pools that paid out; empty if the meeting didn't run or none were won.
    """
    frames = []
    for r in range(1, max_races + 1):
        try:
            divs = get_race_dividends(date, venue, r)
        except Exception:
            if r == 1:
                return pl.DataFrame()   # not a race meeting
            break
        if divs.height:
            jp = divs.filter(pl.col("pool").is_in(JACKPOT_POOLS))
            if jp.height:
                frames.append(jp)
    return pl.concat(frames) if frames else pl.DataFrame()


def scan_jackpot_history(
    start_date: str,
    end_date: str,
    venues: Sequence[str] = ("ST", "HV"),
    delay: float = 0.1,
) -> pl.DataFrame:
    """Scan a date range (``YYYY-MM-DD``) for won jackpot-pool events.

    Polite (``delay`` between requests); skips non-meeting dates. Returns the
    concatenated won-event dataset — the empirical base for jackpot win/dividend
    frequency.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    frames = []
    d = start
    while d <= end:
        ds = d.strftime("%Y/%m/%d")
        for v in venues:
            f = scrape_meeting_jackpots(ds, v)
            if f.height:
                frames.append(f)
            time.sleep(delay)
        d += timedelta(days=1)
    return pl.concat(frames) if frames else pl.DataFrame()
