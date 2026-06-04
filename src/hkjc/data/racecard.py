"""Pre-race racecard — the richest single pre-race source (27 columns).

Server-rendered: ``racecard?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N``. Far
richer than the live GraphQL card — ratings and **rating moves**, gear, declared
**body weight and its delta**, days-since-run, best time, season stakes, pedigree.
These are the standard handicapping features, all known pre-race, keyed to our
canonical (date, venue, race_no, runner) keys for the feature store.
"""
from __future__ import annotations

import re

import polars as pl
import requests
from bs4 import BeautifulSoup

RACECARD_URL = "https://racing.hkjc.com/en-us/local/information/racecard?racedate={date}&Racecourse={venue}&RaceNo={race}"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://racing.hkjc.com"}

# Exact racecard header text -> canonical snake_case field.
_HEADER_MAP = {
    "Horse No.": "runner", "Last 6 Runs": "last_6_runs", "Colour": "colour",
    "Horse": "horse", "Brand No.": "horse_id", "Wt.": "carried_wt", "Jockey": "jockey",
    "Over Wt.": "over_wt", "Draw": "draw", "Trainer": "trainer",
    "Int'l Rtg.": "intl_rtg", "Rtg.": "rtg", "Rtg.+/-": "rtg_change",
    "Horse Wt. (Declaration)": "decl_horse_wt", "Wt.+/- (vs Declaration)": "wt_change",
    "Best Time": "best_time", "Age": "age", "WFA": "wfa", "Sex": "sex",
    "Season Stakes": "season_stakes", "Priority": "priority",
    "Days since Last Run": "days_since_last_run", "Gear": "gear",
    "Owner": "owner", "Sire": "sire", "Dam": "dam", "Import Cat.": "import_cat",
}
_INT_COLS = ("runner", "carried_wt", "draw", "intl_rtg", "rtg", "rtg_change",
             "decl_horse_wt", "wt_change", "age", "days_since_last_run")
_ALLOWANCE = re.compile(r"\(([-+]?\d+)\)")


def _fmt_date(date: str) -> str:
    s = str(date).replace("-", "/")
    parts = s.split("/")
    if len(parts) == 3 and len(parts[0]) != 4:   # DD/MM/YYYY -> YYYY/MM/DD
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return s


def _to_int(s):
    s = (s or "").strip().replace(",", "").replace("+", "")
    if s in ("", "-", "--"):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _best_time_sec(s):
    s = (s or "").strip()
    if not s or s == "-":
        return None
    parts = s.split(".")
    try:
        if len(parts) == 3:                       # m.ss.ff
            return round(int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 100, 2)
        if len(parts) == 2:                       # ss.ff
            return round(int(parts[0]) + int(parts[1]) / 100, 2)
    except ValueError:
        return None
    return None


def _split_jockey(s):
    s = (s or "").strip()
    m = _ALLOWANCE.search(s)
    allowance = int(m.group(1)) if m else 0
    name = _ALLOWANCE.sub("", s).strip()
    return name, allowance


def _parse_racecard_table(table) -> list[dict]:
    rows = table.find_all("tr")
    if not rows:
        return []
    # The header is the first row whose cells map to our known fields (the real
    # starters table; the page nests several wrapper tables).
    header, body_start = None, 0
    for i, tr in enumerate(rows):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if "Brand No." in cells and "Horse No." in cells:
            header, body_start = cells, i + 1
            break
    if header is None:
        return []
    cols = [_HEADER_MAP.get(h) for h in header]
    out = []
    for tr in rows[body_start:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) != len(header):
            continue
        rec = {name: val for name, val in zip(cols, cells) if name}
        if "runner" not in rec or not str(rec["runner"]).strip().isdigit():
            continue   # skip reserves / non-runner rows
        out.append(rec)
    return out


def _records_to_frame(records: list[dict]) -> pl.DataFrame:
    clean = []
    for r in records:
        jockey, allowance = _split_jockey(r.get("jockey"))
        row = dict(r)
        row["jockey"] = jockey
        row["jockey_allowance"] = allowance
        for c in _INT_COLS:
            if c in row:
                row[c] = _to_int(row[c])
        if "season_stakes" in row:
            row["season_stakes"] = _to_int(row["season_stakes"])
        if "best_time" in row:
            row["best_time"] = _best_time_sec(row["best_time"])
        clean.append(row)
    return pl.DataFrame(clean)


def get_racecard(date: str, venue: str, race_no: int) -> pl.DataFrame:
    """Fetch and parse the pre-race racecard for one race.

    ``date`` may be ``YYYY-MM-DD`` / ``YYYY/MM/DD`` / ``DD/MM/YYYY``. Returns a tidy
    per-runner frame (numeric columns typed; jockey allowance split out) plus
    ``date``/``venue``/``race_no`` keys. Empty if the card isn't published yet.
    """
    d = _fmt_date(date)
    resp = requests.get(RACECARD_URL.format(date=d, venue=venue, race=race_no), headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    # Parse every candidate table and keep the one yielding the most runners
    # (the page nests wrapper tables around the real starters table).
    records: list[dict] = []
    for t in soup.find_all("table"):
        if "Brand No." not in t.get_text():
            continue
        recs = _parse_racecard_table(t)
        if len(recs) > len(records):
            records = recs
    if not records:
        return pl.DataFrame()
    return _records_to_frame(records).with_columns([
        pl.lit(d).alias("date"), pl.lit(venue).alias("venue"), pl.lit(int(race_no)).alias("race_no"),
    ])
