"""Sectional split times + running positions per horse (trajectory Tier-1).

The "Sectional Time & Position" page is **server-rendered** — but only with the
right params: ``racedate`` as **DD/MM/YYYY** and ``RaceNo`` (no ``Racecourse``;
the date selects the meeting). The wrong format yields an empty "released shortly"
page. This turns each horse's run from one number (finish time) into a dense
per-section velocity profile — the foundation for the energetics model (see
docs/docs/HKJC_DATA.md).

Output is tidy and keyed to join our canonical (date, race_no, runner) keys.
Per-section parsing extracts the running **position** and **section time** from
each cell; margins/sub-splits are left for a later refinement.
"""
from __future__ import annotations

import re

import polars as pl
import requests
from bs4 import BeautifulSoup

SECTIONAL_URL = "https://racing.hkjc.com/en-us/local/information/displaysectionaltime?racedate={date}&RaceNo={race}"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://racing.hkjc.com"}
_BRAND = re.compile(r"\(([A-Z]\d+)\)")
_TIME = re.compile(r"(\d+\.\d\d)")
_LEAD_INT = re.compile(r"^\s*(\d+)")


def _to_ddmmyyyy(date: str) -> str:
    """Accept YYYY-MM-DD / YYYY/MM/DD / DD/MM/YYYY → DD/MM/YYYY (the working format)."""
    s = str(date).replace("-", "/")
    parts = s.split("/")
    if len(parts) == 3 and len(parts[0]) == 4:      # YYYY/MM/DD
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return s


def _time_to_sec(t: str):
    t = t.strip()
    if ":" in t:                                     # m:ss.ff
        m, s = t.split(":", 1)
        try:
            return round(int(m) * 60 + float(s), 2)
        except ValueError:
            return None
    try:
        return float(t)
    except ValueError:
        return None


def _parse_sectional_table(table) -> list[dict]:
    out = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 4 or not cells[0].isdigit():
            continue                                 # skip headers / non-data rows
        finish_order = int(cells[0])
        runner = int(cells[1]) if cells[1].isdigit() else None
        name_brand = cells[2]
        brand = _BRAND.search(name_brand)
        horse_id = brand.group(1) if brand else None
        horse = _BRAND.sub("", name_brand).strip()
        total_time = _time_to_sec(cells[-1])
        for i, cell in enumerate(cells[3:-1], start=1):
            if not cell:
                continue
            pos = _LEAD_INT.match(cell)
            tm = _TIME.search(cell)
            out.append({
                "finish_order": finish_order, "runner": runner, "horse_id": horse_id,
                "horse": horse, "total_time": total_time, "section": i,
                "position": int(pos.group(1)) if pos else None,
                "section_time": float(tm.group(1)) if tm else None,
            })
    return out


def get_sectional_times(date: str, race_no: int) -> pl.DataFrame:
    """Per-horse, per-section running position + section time for one race.

    ``date`` may be ``YYYY-MM-DD`` / ``YYYY/MM/DD`` / ``DD/MM/YYYY``. Returns a tidy
    long frame (one row per horse per section) with ``date/race_no/runner`` keys,
    ``horse_id``, ``finish_order``, ``total_time``, ``section``, ``position`` and
    ``section_time``. Empty if sectionals aren't published for that race.
    """
    dd = _to_ddmmyyyy(date)
    resp = requests.get(SECTIONAL_URL.format(date=dd, race=race_no), headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    table = next((t for t in soup.find_all("table") if "Finishing Order" in t.get_text()), None)
    if table is None:
        return pl.DataFrame()
    rows = _parse_sectional_table(table)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns([
        pl.lit(dd).alias("date"),
        pl.lit(int(race_no)).alias("race_no"),
    ])
