"""SCRATCH prototype scrapers for three overlooked HKJC data sources.

Driven by a live-browser recon (network capture). Each function pulls a small
SAMPLE so coverage/shape claims are concrete. NOT wired into the model — this is
a studies/ scratch probe per the data-mining brief. Matches house style of
hkjc.data.{sectional,racecard,live} (requests + BeautifulSoup for SSR HTML;
requests.post for the GraphQL ops).

Sources:
  1. btresult   — barrier trials (debut/return fitness read).  SSR HTML, ?Date=DD/MM/YYYY
  2. formline   — opponent-linkage / collateral form.           SSR HTML, racedate/Racecourse/RaceNo
  3. weather    — wind vectors + soil moisture (going proxy).   GraphQL op `Weather`, info.cld.hkjc.com

Run:  uv run python studies/newdata_probe.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).parent / "newdata_samples"
OUT.mkdir(exist_ok=True)

_HTML_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://racing.hkjc.com"}
_GQL_ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"
_GQL_HEADERS = {
    "Origin": "https://racing.hkjc.com",
    "Referer": "https://racing.hkjc.com/",
    "Content-Type": "application/json",
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/141.0.0.0 Safari/537.36",
}
_BRAND = re.compile(r"\(([A-Z]\d+)\)")
_DDMMYYYY = re.compile(r"^\d{2}/\d{2}/\d{4}$")


# ---------------------------------------------------------------------------
# 1. BARRIER TRIALS  —  https://racing.hkjc.com/en-us/local/information/btresult?Date=DD/MM/YYYY
# ---------------------------------------------------------------------------
BT_URL = "https://racing.hkjc.com/en-us/local/information/btresult?Date={date}"
_BT_TIME = re.compile(r"Time:\s*([\d.:]+)")
_BT_SECT = re.compile(r"Sectional Time:\s*([\d.\s]+)")
_BT_BATCH = re.compile(r"Batch\s*(\d+)\s*-\s*(.+?)\s*-\s*(\d+)m", re.S)
_BT_GOING = re.compile(r"Going:\s*([A-Za-z /]+?)\s+Time:")


def _time_to_sec(t: str):
    t = t.strip()
    if not t:
        return None
    parts = t.split(".")
    try:
        if len(parts) == 3:                  # m.ss.ff  e.g. 1.10.80
            return round(int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 100, 2)
        if len(parts) == 2:                  # ss.ff
            return round(int(parts[0]) + int(parts[1]) / 100, 2)
    except ValueError:
        return None
    return None


def get_barrier_trials(date: str) -> pl.DataFrame:
    """Per-horse barrier-trial results for one trial date.

    ``date`` is DD/MM/YYYY (the page's native param). Returns one row per horse per
    batch with the batch context (course/distance/going, batch race Time + the
    3 batch sectional splits) and the per-horse Draw/Gear/LBW/RunningPosition/Time/
    Result/Comment — plus ``horse_id`` (HK_YYYY_XXXX) so it joins our panels.
    """
    r = requests.get(BT_URL.format(date=date), headers=_HTML_HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    tables = soup.find_all("table")

    # Tables come in (batch-header, horse-rows) order. We walk all tables, tracking
    # the most recent batch header context, and parse any table that has horse links.
    ctx = {}
    rows: list[dict] = []
    for tb in tables:
        txt = tb.get_text(" ", strip=True)
        has_horse = tb.find("a", href=re.compile("horseid")) is not None
        if not has_horse:
            mb = _BT_BATCH.search(txt)
            if mb:
                mg = _BT_GOING.search(txt)
                mt = _BT_TIME.search(txt)
                ms = _BT_SECT.search(txt)
                ctx = {
                    "batch": int(mb.group(1)),
                    "course": mb.group(2).strip(),
                    "distance": int(mb.group(3)),
                    "going": (mg.group(1).strip() if mg else None),
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
                "date": date,
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
                "running_position": vals[6],
                "trial_time": _time_to_sec(vals[7]),
                "result": vals[8],
                "comment": vals[9],
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. FORMLINE  —  .../formline?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=N
# ---------------------------------------------------------------------------
FORMLINE_URL = ("https://racing.hkjc.com/en-us/local/information/formline"
                "?racedate={date}&Racecourse={venue}&RaceNo={race}")
_PAST = re.compile(r"(\d{2}/\d{2}/\d{4})\s*\((\d+)\)")    # "06/05/2026 (668)" -> date, race id


def get_formline(date: str, venue: str, race_no: int) -> pl.DataFrame:
    """Opponent-linkage / collateral-form for one UPCOMING race.

    ``date`` is YYYY/MM/DD. For each *shared past race* that ≥2 of today's runners
    contested, returns one row per rival in that past race: the shared-race context
    (past date/venue/dist/class/going) + the rival's Wt/Rtg/Draw/Jockey/LastJockey/
    Placing/Time/Margin/WinOdds/WeightCarried+/-/Rating+/-. This is the pre-assembled
    collateral-form graph: rows sharing (past_date, past_race_id) ran against each other.
    """
    r = requests.get(FORMLINE_URL.format(date=date, venue=venue, race=race_no),
                     headers=_HTML_HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")

    rows: list[dict] = []
    ctx = {}
    # All shared-race blocks live inside one table: a "DD/MM/YYYY (id)" header row,
    # then a column-header row, then the rival rows — repeated. Walk rows, switching
    # context on each past-race header.
    for tb in soup.find_all("table"):
        if not tb.find("a", href=re.compile("localresults")):
            continue
        for tr in tb.find_all("tr"):
            link = tr.find("a", href=re.compile("localresults"))
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if link:                                  # shared-race header row
                mp = _PAST.search(link.get_text(" ", strip=True))
                href = link.get("href", "")
                mv = re.search(r"Racecourse=([A-Z]{2})", href)
                ctx = {
                    "past_date": mp.group(1) if mp else None,
                    "past_race_id": int(mp.group(2)) if mp else None,
                    "past_venue": mv.group(1) if mv else None,
                    "past_course": cells[1] if len(cells) > 1 else None,
                    "past_dist": next((c for c in cells if c.endswith("m") and c[:-1].isdigit()), None),
                    "past_class": next((c for c in cells if c.startswith("Class")), None),
                    "past_going": cells[-1] if cells else None,
                }
                continue
            if len(cells) < 11 or cells[0] in ("Horse", "") or not cells[0][:1].isupper() \
                    or cells[0] == cells[0].title():
                continue                              # skip column-header / non-data rows
            rows.append({
                "today_date": date, "today_venue": venue, "today_race": race_no,
                **ctx,
                "rival_horse": cells[0],
                "wt": cells[1], "rtg": cells[2], "draw": cells[3],
                "jockey": cells[4], "last_jockey": cells[5],
                "placing": cells[6], "time": cells[7], "margin": cells[8],
                "win_odds": cells[9], "weight_carried_delta": cells[10],
                "rating_delta": cells[11] if len(cells) > 11 else None,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. WEATHER  —  GraphQL op `Weather` (info.cld.hkjc.com), variables: null
#    Server WHITELISTS by exact query body; this is the captured body verbatim.
# ---------------------------------------------------------------------------
_WEATHER_QUERY = """
query Weather {
  weather {
    racecourse
    sectional {
      location
      date
      time
      avgCorrectedWindDirection
      avgCorrectedWindSpeed
      correctedGustDirection
      correctedGustSpeed
      gpsPosition
      supplyVoltage
      status
      isValid
    }
    weatherStation {
      location
      date
      time
      avgCorrectedWindDirection
      avgCorrectedWindSpeed
      correctedGustDirection
      correctedGustSpeed
      gpsPosition
      supplyVoltage
      relativeHumidity
      temperature
      solarRadiation
      sunshineHour
      pressure
      correctedGustSpeedDailyMax
      relativeHumidityDailyMax
      relativeHumidityDailyMin
      temperatureDailyMax
      temperatureDailyMin
      rainFallPrecipitation1Min
      rainFallPrecipitation10Min
      rainPrecipitationDailyTotal
      soilVolumeticWaterContent
      soilMoistureLoss1Min
      soilMoistureLossDailyTotal
      soilMoistureLoss24H
      status
      isValid
    }
  }
}

  """


def get_weather() -> dict:
    """Latest wind + soil-moisture readings for BOTH HV & ST (one live snapshot).

    The op takes NO date param (variables:null) and is cached ~3s — it is a *latest*
    reading, so to build history you poll-and-log. Returns the raw `weather` list.
    Per racecourse: 4 anemometers A–D (sectional[]) + a full weather station E with
    temperature/humidity/pressure/rain and the soil-moisture going proxy.
    """
    r = requests.post(_GQL_ENDPOINT, json={"variables": None, "query": _WEATHER_QUERY},
                      headers=_GQL_HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()["data"]["weather"]


def weather_flat(weather: list) -> pl.DataFrame:
    """Flatten the going-relevant station fields to one row per racecourse."""
    rows = []
    for rc in weather:
        ws = rc.get("weatherStation", {}) or {}
        rows.append({
            "racecourse": rc.get("racecourse"),
            "date": ws.get("date"), "time": ws.get("time"),
            "wind_dir": ws.get("avgCorrectedWindDirection"),
            "wind_speed": ws.get("avgCorrectedWindSpeed"),
            "gust_speed_daily_max": ws.get("correctedGustSpeedDailyMax"),
            "temperature": ws.get("temperature"),
            "rel_humidity": ws.get("relativeHumidity"),
            "rain_daily_total": ws.get("rainPrecipitationDailyTotal"),
            "soil_water_content": ws.get("soilVolumeticWaterContent"),
            "soil_moisture_loss_24h": ws.get("soilMoistureLoss24H"),
            "n_anemometers": len(rc.get("sectional", []) or []),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    print("=== 1. BARRIER TRIALS (05/06/2026) ===")
    bt = get_barrier_trials("05/06/2026")
    print(bt.select("batch", "distance", "going", "batch_time", "horse_id",
                    "draw", "lbw", "running_position", "trial_time", "result").head(12))
    print(f"rows={bt.height}  unique horses={bt['horse_id'].n_unique()}  batches={bt['batch'].n_unique()}")
    bt.write_parquet(OUT / "barrier_trials_20260605.parquet")

    print("\n=== 2. FORMLINE (2026/06/07 ST R1) ===")
    fl = get_formline("2026/06/07", "ST", 1)
    print(fl.select("past_date", "past_race_id", "past_venue", "past_dist",
                    "rival_horse", "placing", "time", "margin", "win_odds").head(12))
    print(f"rows={fl.height}  shared past races={fl['past_race_id'].n_unique()}")
    fl.write_parquet(OUT / "formline_20260607_ST_R1.parquet")

    print("\n=== 3. WEATHER (live snapshot) ===")
    w = get_weather()
    wf = weather_flat(w)
    print(wf)
    (OUT / "weather_snapshot.json").write_text(json.dumps(w, indent=2))
    print(f"saved raw -> {OUT/'weather_snapshot.json'}")


if __name__ == "__main__":
    main()
