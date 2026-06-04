"""Comments on Running — the stewards' per-horse trip notes (trouble-in-running).

Server-rendered: ``corunning?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N``. Each
runner gets a free-text running line written by the stipendiary stewards, who watch
every camera angle — so this is the **trouble signal** (blocked / no clear run /
checked / short of room) that section *times* can't see (the lateral dimension).

A horse whose result was compromised by traffic has a finishing position that
*understates* its ability — the classic, plausibly-underpriced "trouble" angle. We
flag that with a keyword classifier; the experiment then tests whether a horse's
**prior-run** trouble predicts it out-performing the market.
"""
from __future__ import annotations

import re

import polars as pl
import requests
from bs4 import BeautifulSoup

CORUNNING_URL = "https://racing.hkjc.com/en-us/local/information/corunning?racedate={date}&Racecourse={venue}&RaceNo={race}"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://racing.hkjc.com"}
_BRAND = re.compile(r"\(([A-Z]\d+)\)")

# Phrases denoting interference / lack of a clear run (the lateral-blocking signal),
# as opposed to purely tactical lines ("held up", "ridden patiently").
TROUBLE_PATTERN = re.compile(
    r"no clear run|no clear (?:racing )?room|no clear passage|short of room|no room|"
    r"denied (?:a )?(?:clear )?(?:run|passage)|blocked|checked|hampered|tightened|"
    r"squeezed|crowded|bumped|interfered|carried (?:wide|out)|forced (?:wide|out)|"
    r"trapped|in(?:to)? a pocket|behind a wall|had to steady|steadied(?! away| from the start)|"
    r"had to switch|shuffled|awkwardly placed|lost ground|wide no cover|no luck",
    re.IGNORECASE,
)


def has_trouble(comment: str) -> int:
    """1 if the running comment describes interference / no clear run, else 0."""
    return int(bool(TROUBLE_PATTERN.search(comment or "")))


def _fmt_date(date: str) -> str:
    s = str(date).replace("-", "/")
    parts = s.split("/")
    if len(parts) == 3 and len(parts[0]) != 4:   # DD/MM/YYYY -> YYYY/MM/DD
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return s


def _parse_corunning_table(table) -> list[dict]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 6 or not cells[1].strip().isdigit():
            continue   # skip header / non-data rows
        name_brand = cells[2]
        brand = _BRAND.search(name_brand)
        comment = cells[5]
        rows.append({
            "runner": int(cells[1]),
            "horse_id": brand.group(1) if brand else None,
            "horse": _BRAND.sub("", name_brand).strip(),
            "gear": cells[4],
            "comment": comment,
            "trouble": has_trouble(comment),
        })
    return rows


def get_running_comments(date: str, venue: str, race_no: int) -> pl.DataFrame:
    """Per-horse running comments + a ``trouble`` flag for one race.

    ``date`` may be ``YYYY-MM-DD`` / ``YYYY/MM/DD`` / ``DD/MM/YYYY``. Returns a tidy
    frame keyed to ``(date, venue, race_no, runner)`` with ``horse_id``, ``comment``
    and ``trouble`` (0/1). Empty if not published.
    """
    d = _fmt_date(date)
    resp = requests.get(CORUNNING_URL.format(date=d, venue=venue, race=race_no), headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    table = next((t for t in soup.find_all("table")
                  if "Comment" in t.get_text() and "Horse" in t.get_text()), None)
    if table is None:
        return pl.DataFrame()
    rows = _parse_corunning_table(table)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns([
        pl.lit(d).alias("date"), pl.lit(venue).alias("venue"), pl.lit(int(race_no)).alias("race_no"),
    ])
