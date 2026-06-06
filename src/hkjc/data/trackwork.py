"""Daily training-load (``localtrackwork``) — the densest leak-free fitness prior.

The "Trackwork" page is **server-rendered** (modern en-us route) and keyed to a race
card: ``?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N``. For every runner in that race
it lists ~3 weeks of dated training sessions across columns *Barrier Trial / Gallop /
Trotting / Swimming / Equine Treadmill / Aqua Walker / Spelling*. The **Gallop** column is
the signal: each entry is a DATED timed gallop —

    ``DD/MM: LOCATION SURFACE split split [split] (total) (rider) [companion]``

e.g. ``04/06: SHA TIN AWT 33.0 30.0 25.9 (1.28.9) (H Y Yuen)`` — a dated, located,
surfaced gallop with per-furlong split times, a bracketed total, and the work rider. Each
gallop is a real **training impulse**: a dated timed exertion whose intensity we read from
the split velocities and worked distance. This is the daily training load race history
cannot give (race-derived fitness is fully collinear with the ability channel — refuted).

This module mirrors the house scrape style of :mod:`~hkjc.data.barrier_trials` /
:mod:`~hkjc.data.sectional` (``requests`` + BeautifulSoup, ``_time_to_sec`` parsing).
Output is tidy **per-(horse, session)** rows so it joins our panels: ``horse_brand`` is the
short stable code (e.g. ``J183``) that matches the energetics panel's ``horse_id``, taken
from the page's ``HK_YYYY_<brand>`` link suffix; ``horse_id`` is the long registration.

**Leak-free guarantee.** The page is published *for an upcoming meeting* and each session
is dated (``DD/MM``). We stamp the as-of racecard date (``race_date``) on every row and emit
``days_before_race = race_date − session_date`` (> 0 for every real pre-race gallop). A
consumer keeps only sessions strictly before the race day, so nothing from race day or later
ever enters a feature — the same strictly-prior discipline as the rest of ``hkjc.model``.

Two scrape ROUTES, both fully server-rendered & leak-free:
  - :func:`get_trackwork` (``localtrackwork``, racecard-keyed) — the upcoming-meeting card. The
    real-time live capture seam: rich (per-race grouping) but **only populated for the IMMINENT
    upcoming meeting** (a past racedate serves "available shortly"), so it has NO history.
  - :func:`get_horse_trackwork` (``trackworkresult?horseid=…``, horse-keyed) — the per-horse
    **full historical archive** (years of dated sessions). This is the route that makes a
    HISTORICAL backfill possible: iterate the horses in your panel, each fetch returns that
    horse's entire dated work log. Same gallop split-time format; ``backfill`` rides this route.

Public entry points:
  - :func:`get_trackwork` — one (racedate, racecourse, race_no) live card.
  - :func:`get_horse_trackwork` — one horse's full dated work history.
  - :func:`backfill` — a polite, rate-limited sweep over a list of horse ids (the history route).
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta

import polars as pl
import requests
from bs4 import BeautifulSoup

_LOG = logging.getLogger(__name__)

TW_URL = ("https://racing.hkjc.com/en-us/local/information/localtrackwork"
          "?racedate={date}&Racecourse={venue}&RaceNo={race}")
# Per-horse FULL historical archive (years of dated sessions) — the backfill route.
TWR_URL = "https://racing.hkjc.com/en-us/local/information/trackworkresult?horseid={horseid}"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://racing.hkjc.com"}

# A trackwork header row carries these column labels; the "Gallop" column holds the
# timed sessions. We locate the Gallop column by header text (robust to column shuffles).
_GALLOP_HDR = "Gallop"
_HORSEID = re.compile(r"horseid=(HK_\d{4}_([A-Z]\d+))")

# One dated gallop entry inside the Gallop cell:
#   DD/MM: LOCATION SURFACE  s1 s2 [s3 ...]  (total)  (rider)  [companion name...]
# LOCATION is upper-case words (SHA TIN / CONGHUA); SURFACE is a single token (AWT, TurG,
# AWTG, RGal, SmT, ...); splits are space-separated furlong section seconds; total is
# bracketed (``m.ss.f`` or ``ss.f``); rider is the second bracket. A trailing companion
# horse name (mixed case) may follow — ignored. The leading lookahead stops a match at the
# next ``DD/MM:`` so consecutive entries split cleanly.
_GALLOP = re.compile(
    r"(\d{2}/\d{2}):\s+"            # session date DD/MM
    r"([A-Z][A-Z ]+?)\s+"          # location (upper words)
    r"([A-Za-z]+)\s+"              # surface token
    r"([\d.][\d.\s]*?)\s*"         # split times (>=1 space-separated)
    r"\(([\d.]+)\)\s*"             # (total)
    r"\(([^)]*)\)"                 # (rider)
)

# Furlong ≈ 201.168 m; HK gallop splits are per-furlong section times. Worked distance =
# n_splits furlongs; intensity reads off the per-furlong velocity.
_FURLONG_M = 201.168


def _to_slashes(d: str) -> str:
    """Accept YYYY-MM-DD / YYYY/MM/DD → YYYY/MM/DD (the page's racedate format)."""
    return str(d).replace("-", "/")


def _to_iso(d: str) -> str:
    """YYYY/MM/DD or YYYY-MM-DD → YYYY-MM-DD."""
    return str(d).replace("/", "-")


def _time_to_sec(t: str):
    """HKJC dot-time → seconds: ``m.ss.f`` / ``ss.f`` (also tolerates ``m:ss.f``)."""
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
        if len(parts) == 3:                       # m.ss.f  e.g. 1.28.9
            return round(int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10, 2)
        if len(parts) == 2:                       # ss.f
            return round(int(parts[0]) + int(parts[1]) / 10, 2)
    except ValueError:
        return None
    return None


def _session_iso(ddmm: str, race_iso: str) -> str | None:
    """Resolve a ``DD/MM`` session stamp to an ISO date relative to the race date.

    Sessions are within ~3 weeks before the race, so the session's year is the race year
    unless its month is *after* the race month (a December gallop before a January race),
    in which case it belongs to the prior year. Returns ISO ``YYYY-MM-DD`` or None.
    """
    try:
        dd, mm = (int(x) for x in ddmm.split("/"))
        ry, rm, _ = (int(x) for x in race_iso.split("-"))
    except (ValueError, AttributeError):
        return None
    year = ry - 1 if mm > rm else ry
    try:
        return datetime(year, mm, dd).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _intensity(splits: list[float], total: float | None) -> tuple[float | None, float | None]:
    """(worked_distance_m, mean_velocity m/s) for a gallop from its furlong splits.

    Distance = n_splits furlongs. Velocity prefers the bracketed total (the authoritative
    full-gallop clock) and falls back to the split sum. Higher velocity over more furlongs =
    a more intense training impulse — the weight the Banister trace integrates.
    """
    if not splits:
        return None, None
    dist = len(splits) * _FURLONG_M
    sec = total if (total and total > 0) else sum(splits)
    vel = dist / sec if sec and sec > 0 else None
    return dist, vel


def _gallop_col_index(header_cells: list[str]) -> int | None:
    for i, h in enumerate(header_cells):
        if h.strip().lower() == _GALLOP_HDR.lower():
            return i
    return None


def _parse_html(html: bytes, race_iso: str, venue: str, race_no: int) -> list[dict]:
    """Parse one localtrackwork page into per-(horse, gallop-session) rows (no network)."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    for tb in soup.find_all("table"):
        if tb.find("a", href=_HORSEID) is None:
            continue
        trs = tb.find_all("tr")
        if not trs:
            continue
        header = [c.get_text(" ", strip=True) for c in trs[0].find_all(["td", "th"])]
        gcol = _gallop_col_index(header)
        if gcol is None:
            continue
        for tr in trs[1:]:
            link = tr.find("a", href=_HORSEID)
            cells = tr.find_all(["td", "th"])
            if not link or len(cells) <= gcol:
                continue
            m = _HORSEID.search(str(link.get("href", "")))
            horse_id = m.group(1) if m else None
            brand = m.group(2) if m else None
            name_cell = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
            horse = re.split(r"\s+[A-Z]\s|\s+\d", name_cell, maxsplit=1)[0].strip()
            gallop_txt = cells[gcol].get_text(" ", strip=True)
            for gm in _GALLOP.finditer(gallop_txt):
                ddmm, loc, surface, splits_txt, total_txt, rider = gm.groups()
                splits = [float(x) for x in splits_txt.split()]
                total = _time_to_sec(total_txt)
                dist_m, vel = _intensity(splits, total)
                sess_iso = _session_iso(ddmm, race_iso)
                dbr = None
                if sess_iso is not None:
                    dbr = (datetime.strptime(race_iso, "%Y-%m-%d")
                           - datetime.strptime(sess_iso, "%Y-%m-%d")).days
                rows.append({
                    "race_date": race_iso,
                    "venue": venue,
                    "race_no": race_no,
                    "horse": horse,
                    "horse_brand": brand,
                    "horse_id": horse_id,
                    "session_date": sess_iso,
                    "days_before_race": dbr,
                    "location": loc.strip(),
                    "surface": surface,
                    "splits": " ".join(str(s) for s in splits),
                    "n_splits": len(splits),
                    "total_time": total,
                    "worked_distance_m": dist_m,
                    "gallop_velocity": round(vel, 3) if vel else None,
                    "rider": rider.strip(),
                })
    return rows


_SCHEMA = {
    "race_date": pl.Utf8, "venue": pl.Utf8, "race_no": pl.Int64,
    "horse": pl.Utf8, "horse_brand": pl.Utf8, "horse_id": pl.Utf8,
    "session_date": pl.Utf8, "days_before_race": pl.Int64,
    "location": pl.Utf8, "surface": pl.Utf8, "splits": pl.Utf8, "n_splits": pl.Int64,
    "total_time": pl.Float64, "worked_distance_m": pl.Float64,
    "gallop_velocity": pl.Float64, "rider": pl.Utf8,
}


# Per-horse archive: Racecourse/Track is "<location words> <surface...>"; the leading
# location is "Sha Tin" / "Conghua" (the two training bases). Workouts is the same
# "<splits> (total) (rider)" form, with NO leading DD/MM: (the date is its own column).
_TWR_WORK = re.compile(r"([\d.][\d.\s]*?)\s*\(([\d.]+)\)\s*\(([^)]*)\)")
_LOCATIONS = ("Sha Tin", "Conghua")


def _split_rc_track(rc_track: str) -> tuple[str, str]:
    """"Sha Tin AWT" → ("SHA TIN", "AWT");  "Conghua TurG" → ("CONGHUA", "TurG")."""
    rc = (rc_track or "").strip()
    for loc in _LOCATIONS:
        if rc.startswith(loc):
            return loc.upper(), rc[len(loc):].strip()
    parts = rc.split(None, 1)
    return (parts[0].upper() if parts else ""), (parts[1] if len(parts) > 1 else "")


def _parse_horse_html(html: bytes, horse_id: str, brand: str | None) -> list[dict]:
    """Parse one trackworkresult (per-horse archive) page into per-gallop-session rows.

    Only GALLOP rows carry timed workouts; the non-timed work types (Trotting / Swimming /
    Aqua Walker / Spelling / Barrier Trial) are skipped — the Banister impulses are the timed
    gallops. ``race_date`` / ``days_before_race`` are left null here (this is a horse archive,
    not a racecard); the consumer derives ``days_before_race`` per race when it as-of-joins.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    for tb in soup.find_all("table"):
        trs = tb.find_all("tr")
        if not trs:
            continue
        header = [c.get_text(" ", strip=True) for c in trs[0].find_all(["td", "th"])]
        if header[:2] != ["Date", "Type"]:
            continue
        for tr in trs[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 4 or cells[1] != "Gallop":
                continue
            ddmmyyyy, _typ, rc_track, workouts = cells[0], cells[1], cells[2], cells[3]
            wm = _TWR_WORK.search(workouts)
            if not wm:
                continue
            splits = [float(x) for x in wm.group(1).split()]
            total = _time_to_sec(wm.group(2))
            dist_m, vel = _intensity(splits, total)
            location, surface = _split_rc_track(rc_track)
            try:
                sess_iso = datetime.strptime(ddmmyyyy, "%d/%m/%Y").strftime("%Y-%m-%d")
            except ValueError:
                sess_iso = None
            rows.append({
                "race_date": None, "venue": None, "race_no": None,
                "horse": None, "horse_brand": brand, "horse_id": horse_id,
                "session_date": sess_iso, "days_before_race": None,
                "location": location, "surface": surface,
                "splits": " ".join(str(s) for s in splits), "n_splits": len(splits),
                "total_time": total, "worked_distance_m": dist_m,
                "gallop_velocity": round(vel, 3) if vel else None,
                "rider": wm.group(3).strip(),
            })
    return rows


def get_horse_trackwork(horse_id: str) -> pl.DataFrame:
    """One horse's FULL dated gallop history (the historical-backfill route).

    ``horse_id`` is the long ``HK_YYYY_XXXX`` registration. Returns one row per timed gallop
    session over the horse's whole archive (years), each with session date, location, surface,
    furlong splits, worked distance + mean gallop velocity (the training-impulse intensity), and
    rider. ``horse_brand`` is the brand suffix of ``horse_id`` (joins our panels). ``race_date`` /
    ``days_before_race`` are null on this route — they are race-relative and the consumer derives
    them when it as-of-joins these sessions to a race date (keeping only strictly-prior ones).
    """
    brand = horse_id.split("_")[-1] if "_" in horse_id else None
    r = requests.get(TWR_URL.format(horseid=horse_id), headers=_HEADERS, timeout=20)
    r.raise_for_status()
    rows = _parse_horse_html(r.content, horse_id, brand)
    return pl.DataFrame(rows, schema=_SCHEMA) if not rows else pl.DataFrame(rows, schema=_SCHEMA)


def get_trackwork(racedate: str, racecourse: str, race_no: int) -> pl.DataFrame:
    """Per-(horse, gallop-session) training-load rows for one race's card.

    ``racedate`` may be ``YYYY-MM-DD`` or ``YYYY/MM/DD``; ``racecourse`` is ``ST``/``HV``.
    Returns one row per timed gallop session for each runner: the session date,
    ``days_before_race`` (> 0, strictly prior), location (SHA TIN / CONGHUA), surface, the
    raw furlong split times, the worked distance + mean gallop velocity (the training-impulse
    intensity), and the work rider. ``horse_brand`` (e.g. ``J183``) joins our panels;
    ``horse_id`` is the long ``HK_YYYY_XXXX`` form. ``race_date`` is normalised to ISO.

    NOTE: the page is populated only for the IMMINENT upcoming meeting (SSR, no JS); for a
    past/future meeting it serves "available shortly" and yields an empty frame. Capture it
    in the pre-race window — that is also exactly what makes it leak-free.
    """
    iso = _to_iso(racedate)
    url = TW_URL.format(date=_to_slashes(racedate), venue=racecourse, race=race_no)
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    rows = _parse_html(r.content, iso, racecourse, int(race_no))
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema=_SCHEMA)


def backfill(horse_ids: list[str], *, sleep: float = 1.0,
             session: requests.Session | None = None) -> pl.DataFrame:
    """Polite, rate-limited sweep of the per-horse history route over a list of horse ids.

    ``horse_ids`` is the set of long ``HK_YYYY_XXXX`` registrations to fetch (build it from your
    race panel — one fetch returns that horse's ENTIRE dated gallop log, so a horse's full history
    arrives in a single request). Fetches each with a ~1 req/sec sleep, one retry on network error,
    skipping horses with no timed gallops. Returns the combined, de-duplicated panel
    (one row per horse × session_date). Idempotent — re-running merges cleanly.
    """
    sess = session or requests.Session()
    sess.headers.update(_HEADERS)
    frames: list[pl.DataFrame] = []
    n_ok = 0
    for i, horse_id in enumerate(horse_ids):
        brand = horse_id.split("_")[-1] if "_" in horse_id else None
        df = None
        for attempt in (1, 2):
            try:
                r = sess.get(TWR_URL.format(horseid=horse_id), timeout=20)
                r.raise_for_status()
                df = pl.DataFrame(_parse_horse_html(r.content, horse_id, brand), schema=_SCHEMA)
                break
            except requests.RequestException as e:
                if attempt == 2:
                    _LOG.warning("giving up on %s: %s", horse_id, e)
                else:
                    time.sleep(2.0)
        if df is not None and df.height:
            frames.append(df)
            n_ok += 1
            _LOG.info("%s: %d gallops (%d/%d)", horse_id, df.height, i + 1, len(horse_ids))
        if sleep:
            time.sleep(sleep)
    _LOG.info("backfill done: %d/%d horses had gallops", n_ok, len(horse_ids))
    if not frames:
        return pl.DataFrame(schema=_SCHEMA)
    out = pl.concat(frames, how="vertical_relaxed")
    return out.unique(subset=["horse_brand", "session_date"], keep="first").sort(
        ["horse_brand", "session_date"])


def upcoming_meeting_dates() -> list[str]:
    """ISO racedates the trackwork search dropdown advertises (the imminent window)."""
    r = requests.get(TW_URL.format(date=datetime.now().strftime("%Y/%m/%d"),
                                   venue="ST", race=1), headers=_HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    out: set[str] = set()
    for o in soup.find_all("option"):
        v = str(o.get("value", ""))
        if re.match(r"^\d{4}/\d{2}/\d{2}$", v):
            out.add(_to_iso(v))
    return sorted(out)


# Convenience: a session-date window helper for building backfill meeting lists.
def _date_range(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(_to_iso(start), "%Y-%m-%d")
    d1 = datetime.strptime(_to_iso(end), "%Y-%m-%d")
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out
