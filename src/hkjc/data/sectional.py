"""Official sectional times + running positions per horse (one row per horse per section).

The "Sectional Time & Position" page is server-rendered only with ``racedate`` as DD/MM/YYYY and
``RaceNo`` (no ``Racecourse``; the date selects the meeting); any other date format yields an empty
"released shortly" page. A complete section cell reads ``position margin time [sub-splits]``; the
position and section time are parsed, margins and sub-splits are not.

Rows: every horse that started (as in :mod:`~hkjc.data.historical`). ``finish_order`` is the placing's
leading integer (both horses of a dead heat, "5 DH", get 5) and null for an unplaced horse (FE, UR, PU,
DNF, TNP, DISQ). A non-finisher's last cell holds only the sub-split it ran before falling or pulling
up, no position: such a partial cell gives no row, so the horse has no time at that section's end.
Withdrawn horses (WV, WV-A, WX, WX-A, WXNR, WR) give no rows.
"""
from __future__ import annotations

import re

import polars as pl
from ._util import get
from bs4 import BeautifulSoup

from .historical import NOT_RUNNER

SECTIONAL_URL = "https://racing.hkjc.com/en-us/local/information/displaysectionaltime?racedate={date}&RaceNo={race}"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://racing.hkjc.com"}
_BRAND = re.compile(r"\(([A-Z]{1,2}\d+)\)")
_LEAD_INT = re.compile(r"^\s*(\d+)")
# a complete section cell: position, an optional margin token, then the section time
_SECTION = re.compile(r"^\s*(\d+)\s+(?:\S+\s+)??(\d+\.\d\d)\b")


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
        if len(cells) < 4 or not cells[1].isdigit():
            continue                                 # skip headers / non-data rows
        code = cells[0].split()[0] if cells[0].split() else ""
        if code in NOT_RUNNER:
            continue
        lead = _LEAD_INT.match(cells[0])
        finish_order = int(lead.group(1)) if lead else None
        runner = int(cells[1])
        name_brand = cells[2]
        brand = _BRAND.search(name_brand)
        horse_id = brand.group(1) if brand else None
        horse = _BRAND.sub("", name_brand).strip()
        total_time = _time_to_sec(cells[-1])
        for i, cell in enumerate(cells[3:-1], start=1):
            m = _SECTION.match(cell)
            if m is None:
                continue                             # empty, or a partial cell (sub-split only)
            out.append({
                "finish_order": finish_order, "runner": runner, "horse_id": horse_id,
                "horse": horse, "total_time": total_time, "section": i,
                "position": int(m.group(1)), "section_time": float(m.group(2)),
            })
    return out


def get_sectional_times(date: str, race_no: int) -> pl.DataFrame:
    """Per-horse, per-section running position + section time for one race.

    ``date`` may be ``YYYY-MM-DD`` / ``YYYY/MM/DD`` / ``DD/MM/YYYY``. Columns: ``date/race_no/runner``,
    ``horse_id``, ``finish_order``, ``total_time``, ``section``, ``position``, ``section_time`` (s).
    Empty if sectionals aren't published for that race.
    """
    dd = _to_ddmmyyyy(date)
    resp = get(SECTIONAL_URL.format(date=dd, race=race_no), headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    return parse_sectional_page(resp.content, dd, race_no)


_SCHEMA = {
    "finish_order": pl.Int64, "runner": pl.Int64, "horse_id": pl.String, "horse": pl.String,
    "total_time": pl.Float64, "section": pl.Int64, "position": pl.Int64, "section_time": pl.Float64,
}


def parse_sectional_page(html, date: str, race_no: int) -> pl.DataFrame:
    """The sectional page's table as :func:`get_sectional_times` returns it (``date`` DD/MM/YYYY)."""
    soup = BeautifulSoup(html, "html.parser")
    table = next((t for t in soup.find_all("table") if "Finishing Order" in t.get_text()), None)
    if table is None:
        return pl.DataFrame()
    rows = _parse_sectional_table(table)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, schema=_SCHEMA).with_columns([
        pl.lit(date).alias("date"),
        pl.lit(int(race_no)).alias("race_no"),
    ])
