"""Barrier-trial results — the only on-track form prior for debut/returning horses.

The "Barrier Trial Result" page is **server-rendered** with a day-first slashed
date param (``?Date=DD/MM/YYYY``). Each trial morning has one or more *batches*
(a small field over a short distance); each batch has a header (course / distance
/ going + the batch winner's Time and 3 sectional splits) followed by the per-horse
rows (Draw / Gear / LBW / Running-Position / Time / Result / Comment). Trials run
only on certain mornings (mostly Mon/Tue/Thu/Fri, occasionally Wed/Sat).

This module mirrors the house scrape style of :mod:`~hkjc.data.sectional`
(``requests`` + BeautifulSoup, ``_to_ddmmyyyy`` date handling, ``_time_to_sec``).
Output is keyed to join our panels: ``horse_brand`` is the brand code (e.g. ``K400``)
that matches the energetics panel's ``horse_id``, and ``horse_id`` is the long
``HK_YYYY_XXXX`` form.

Two public entry points:
  - :func:`get_barrier_trials` — one trial date.
  - :func:`backfill` — a polite, rate-limited sweep over a date range, using the
    page's own date dropdown where available and probing trial weekdays before it.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta

import polars as pl
import requests
from bs4 import BeautifulSoup

_LOG = logging.getLogger(__name__)

BT_URL = "https://racing.hkjc.com/en-us/local/information/btresult?Date={date}"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://racing.hkjc.com"}

_BRAND = re.compile(r"\(([A-Z]\d+)\)")
_BT_TIME = re.compile(r"Time:\s*([\d.:]+)")
_BT_SECT = re.compile(r"Sectional Time:\s*([\d.\s]+)")
_BT_BATCH = re.compile(r"Batch\s*(\d+)\s*-\s*(.+?)\s*-\s*(\d+)m", re.S)
_BT_GOING = re.compile(r"Going:\s*([A-Za-z /]+?)\s+Time:")

# LBW margin codes: a fraction/int + optional "L", or a named short margin.
_NAMED_MARGINS = {"N": 0.3, "SH": 0.1, "HD": 0.2, "NK": 0.3, "DH": 0.0}
_LBW = re.compile(r"^\s*(\d+)?[\s-]*(?:(\d+)/(\d+))?\s*L?\s*$")


def _to_ddmmyyyy(d: str) -> str:
    """Accept YYYY-MM-DD / YYYY/MM/DD / DD/MM/YYYY → DD/MM/YYYY (the page's format)."""
    s = str(d).replace("-", "/")
    parts = s.split("/")
    if len(parts) == 3 and len(parts[0]) == 4:      # YYYY/MM/DD
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return s


def _to_iso(d: str) -> str:
    """DD/MM/YYYY (or already-ISO) → YYYY-MM-DD."""
    s = str(d).replace("-", "/")
    parts = s.split("/")
    if len(parts) == 3 and len(parts[0]) == 4:      # already YYYY/MM/DD
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return f"{parts[2]}-{parts[1]}-{parts[0]}"       # DD/MM/YYYY


def _time_to_sec(t: str):
    """HKJC dot-time → seconds: ``m.ss.ff`` or ``ss.ff`` (also tolerates m:ss.ff)."""
    t = (t or "").strip()
    if not t:
        return None
    if ":" in t:
        m, s = t.split(":", 1)
        try:
            return round(int(m) * 60 + float(s), 2)
        except ValueError:
            return None
    parts = t.split(".")
    try:
        if len(parts) == 3:                          # m.ss.ff  e.g. 1.10.80
            return round(int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 100, 2)
        if len(parts) == 2:                          # ss.ff
            return round(int(parts[0]) + int(parts[1]) / 100, 2)
    except ValueError:
        return None
    return None


def _lbw_to_lengths(lbw: str):
    """Parse a beaten-margin string to lengths (float). ``""`` → 0.0 (the leader).

    Handles ``8-1/2L`` / ``8 1/2L`` / ``3/4L`` / ``8L`` / ``N`` (neck) / ``SH`` /
    ``HD`` / ``-`` and similar. Returns None if unparseable.
    """
    s = (lbw or "").strip().upper().replace("\xa0", " ")
    if not s or s == "-":
        return 0.0
    if s in _NAMED_MARGINS:
        return _NAMED_MARGINS[s]
    m = _LBW.match(s)
    if not m:
        return None
    whole = int(m.group(1)) if m.group(1) else 0
    frac = (int(m.group(2)) / int(m.group(3))) if (m.group(2) and m.group(3)) else 0.0
    val = whole + frac
    return val if (whole or frac) else None


def _parse_html(html: bytes, iso_date: str) -> list[dict]:
    """Parse one btresult page into per-horse rows (no network)."""
    soup = BeautifulSoup(html, "html.parser")
    ctx: dict = {}
    rows: list[dict] = []
    for tb in soup.find_all("table"):
        has_horse = tb.find("a", href=re.compile("horseid")) is not None
        if not has_horse:
            txt = tb.get_text(" ", strip=True)
            mb = _BT_BATCH.search(txt)
            if mb:
                mg = _BT_GOING.search(txt)
                mt = _BT_TIME.search(txt)
                ms = _BT_SECT.search(txt)
                ctx = {
                    "batch": int(mb.group(1)),
                    "course": mb.group(2).strip(),
                    "distance": int(mb.group(3)),
                    "going": mg.group(1).strip() if mg else None,
                    "batch_time": _time_to_sec(mt.group(1)) if mt else None,
                    "batch_sectionals": ([float(x) for x in ms.group(1).split()] if ms else None),
                }
            continue
        for tr in tb.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            link = tr.find("a", href=re.compile("horseid"))
            if not link or len(cells) < 10:
                continue
            vals = [c.get_text(" ", strip=True) for c in cells]
            name_cell = vals[0]
            brand = _BRAND.search(name_cell)
            hid = re.search(r"horseid=(HK_\d{4}_[A-Z]\d+)", str(link.get("href", "")))
            rows.append({
                "date": iso_date,
                "batch": ctx.get("batch"),
                "course": ctx.get("course"),
                "distance": ctx.get("distance"),
                "going": ctx.get("going"),
                "batch_time": ctx.get("batch_time"),
                "batch_sect": json.dumps(ctx.get("batch_sectionals")),
                "horse": _BRAND.sub("", name_cell).strip(),
                "horse_brand": brand.group(1) if brand else None,
                "horse_id": hid.group(1) if hid else None,
                "jockey": vals[1],
                "trainer": vals[2],
                "draw": vals[3],
                "gear": vals[4],
                "lbw": vals[5],
                "lbw_lengths": _lbw_to_lengths(vals[5]),
                "running_position": vals[6],
                "trial_time": _time_to_sec(vals[7]),
                "result": vals[8],
                "comment": vals[9],
            })
    return rows


def get_barrier_trials(d: str) -> pl.DataFrame:
    """Per-horse barrier-trial results for one trial date.

    ``d`` may be ``YYYY-MM-DD`` / ``YYYY/MM/DD`` / ``DD/MM/YYYY``. Returns one row
    per horse per batch with the batch context (course/distance/going, batch winner
    Time + the 3 batch sectional splits) and the per-horse Draw/Gear/LBW/
    RunningPosition/Time/Result/Comment. ``date`` is normalised to ``YYYY-MM-DD``;
    ``horse_brand`` (e.g. ``K400``) joins our panels and ``horse_id`` is the long
    ``HK_YYYY_XXXX`` form. Times are parsed to seconds; ``lbw_lengths`` is the parsed
    numeric margin. Empty frame if no trials ran that day.
    """
    dd = _to_ddmmyyyy(d)
    iso = _to_iso(dd)
    r = requests.get(BT_URL.format(date=dd), headers=_HEADERS, timeout=20)
    r.raise_for_status()
    rows = _parse_html(r.content, iso)
    return pl.DataFrame(rows, schema=_SCHEMA) if not rows else pl.DataFrame(rows)


def list_trial_dates() -> list[str]:
    """Trial dates the btresult page advertises in its date dropdown (ISO, sorted).

    The dropdown holds roughly the trailing ~13 months — authoritative for that
    window; older dates must be probed weekday-by-weekday (see :func:`backfill`).
    """
    r = requests.get(BT_URL.format(date=datetime.now().strftime("%d/%m/%Y")),
                     headers=_HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    sel = soup.find("select")
    if sel is None:
        return []
    out = []
    for o in sel.find_all("option"):
        v = o.get("value")
        if v and re.match(r"^\d{2}/\d{2}/\d{4}$", v):
            out.append(_to_iso(v))
    return sorted(set(out))


# Empty-frame schema so concat/round-trips stay typed even on no-trial days.
_SCHEMA = {
    "date": pl.Utf8, "batch": pl.Int64, "course": pl.Utf8, "distance": pl.Int64,
    "going": pl.Utf8, "batch_time": pl.Float64, "batch_sect": pl.Utf8,
    "horse": pl.Utf8, "horse_brand": pl.Utf8, "horse_id": pl.Utf8,
    "jockey": pl.Utf8, "trainer": pl.Utf8, "draw": pl.Utf8, "gear": pl.Utf8,
    "lbw": pl.Utf8, "lbw_lengths": pl.Float64, "running_position": pl.Utf8,
    "trial_time": pl.Float64, "result": pl.Utf8, "comment": pl.Utf8,
}

# Trial weekdays to probe before the dropdown window (Mon=0 … Sat=5; never Sun).
# Mostly Mon/Tue/Thu/Fri; Wed/Sat are rare but real, so we probe them too.
_PROBE_WEEKDAYS = {0, 1, 2, 3, 4, 5}


def backfill(start: str, end: str, *, sleep: float = 1.0,
             session: requests.Session | None = None) -> pl.DataFrame:
    """Polite, rate-limited sweep of barrier trials over ``[start, end]`` (inclusive).

    Dates are ISO/slashed in any order. Strategy: take the union of the page's date
    dropdown (authoritative recent window) and every Mon–Sat in the range, then fetch
    each candidate with a ~1 req/sec sleep, one retry on network error, skipping days
    with no trials. Returns the combined, de-duplicated panel sorted by date.
    """
    sess = session or requests.Session()
    sess.headers.update(_HEADERS)
    d0 = datetime.strptime(_to_iso(start), "%Y-%m-%d").date()
    d1 = datetime.strptime(_to_iso(end), "%Y-%m-%d").date()

    try:
        advertised = {datetime.strptime(x, "%Y-%m-%d").date() for x in list_trial_dates()}
    except requests.RequestException:
        advertised = set()
    candidates: set[date] = {d for d in advertised if d0 <= d <= d1}
    d = d0
    while d <= d1:
        if d.weekday() in _PROBE_WEEKDAYS:
            candidates.add(d)
        d += timedelta(days=1)

    frames: list[pl.DataFrame] = []
    cand_sorted = sorted(candidates)
    n_trial_days = 0
    for i, day in enumerate(cand_sorted):
        dd = day.strftime("%d/%m/%Y")
        df = None
        for attempt in (1, 2):
            try:
                r = sess.get(BT_URL.format(date=dd), timeout=20)
                r.raise_for_status()
                df = pl.DataFrame(_parse_html(r.content, day.isoformat()))
                break
            except requests.RequestException as e:
                if attempt == 2:
                    _LOG.warning("giving up on %s: %s", dd, e)
                else:
                    time.sleep(2.0)
        if df is not None and df.height:
            frames.append(df)
            n_trial_days += 1
            _LOG.info("%s: %d rows (%d/%d)", day.isoformat(), df.height, i + 1, len(cand_sorted))
        if sleep:
            time.sleep(sleep)

    _LOG.info("backfill done: %d trial days, %d candidates probed", n_trial_days, len(cand_sorted))
    if not frames:
        return pl.DataFrame(schema=_SCHEMA)
    out = pl.concat(frames, how="vertical_relaxed")
    return out.unique(subset=["date", "batch", "horse_brand"], keep="first").sort(["date", "batch"])
