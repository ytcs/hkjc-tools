"""Functions to scrape HKJC SpeedPro data.

SpeedPro figures (energy grid + speed maps) used to be published as a per-meeting
``.xls`` energy grid. HKJC has since retired that endpoint; the data now lives in
per-race "speed guide" JSON files which bundle both the speed-map image and the
energy grid for each runner.

Note: the energy / fitness figures are only populated once HKJC publishes SpeedPro
for an upcoming meeting (typically the day before racing). Before publication the
JSON still exists but the numeric fields are blank, so callers must tolerate nulls.
"""
from __future__ import annotations

import json

import polars as pl
import requests
from cachetools.func import ttl_cache

SPEEDGUIDE_URL_TEMPLATE = (
    "https://racing.hkjc.com/racing/speedpro/assets/json/speedguide/race_{race_num}.json"
)

REQUEST_TIMEOUT = 15

# Upper bound on races in a single HK meeting (usually 8-11).
_MAX_RACES = 14


@ttl_cache(maxsize=32, ttl=300)
def _fetch_speedguide(race_num: int, lang: str = "en-us") -> dict | None:
    """Fetch and parse the SpeedPro speed-guide JSON for a single race.

    Returns the language sub-document (default English), or ``None`` if the race
    file does not exist (e.g. race number beyond the meeting's field).
    """
    r = requests.get(SPEEDGUIDE_URL_TEMPLATE.format(race_num=race_num), timeout=REQUEST_TIMEOUT)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise RuntimeError(f"Request failed: {r.status_code} - {r.text}")
    # Decode with 'utf-8-sig' to strip a possible UTF-8 BOM before JSON parsing.
    return json.loads(r.content.decode("utf-8-sig"))[lang]


def _parse_int(value) -> int | None:
    """Best-effort parse of the whitespace-padded numeric strings HKJC emits."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


_ENERGY_SCHEMA = {
    "RaceNo": pl.Int64,
    "RunnerNumber": pl.Int64,
    "HorseName": pl.Utf8,
    "Draw": pl.Int64,
    "BrandNo": pl.Utf8,
    "FitnessRatings": pl.Int64,
    "SpeedPRO_Energy": pl.Int64,
    "SpeedPRO_Energy_Difference": pl.Int64,
    "EnergyRequired": pl.Int64,
}


def speedpro_energy(race_num: int | None = None) -> pl.DataFrame:
    """Fetch SpeedPro energy/fitness figures for the current meeting.

    Args:
        race_num: If given, fetch only that race. Otherwise fetch every race in
            the current meeting (races are probed until the first missing file).

    Returns:
        pl.DataFrame: One row per runner with columns ``RaceNo``, ``RunnerNumber``,
        ``HorseName``, ``Draw``, ``BrandNo``, ``FitnessRatings``,
        ``SpeedPRO_Energy``, ``SpeedPRO_Energy_Difference`` and ``EnergyRequired``.
        Numeric columns are null where SpeedPro has not yet been published.

    Note:
        The underlying endpoint is keyed by race number for the *current* meeting
        only; it cannot be queried for arbitrary historical dates.
    """
    race_nums = [race_num] if race_num is not None else range(1, _MAX_RACES + 1)

    rows = []
    for n in race_nums:
        guide = _fetch_speedguide(n)
        if guide is None:
            if race_num is not None:
                raise RuntimeError(f"No SpeedPro speed guide available for race {n}")
            break  # first gap marks the end of the meeting
        energy_required = _parse_int(_first_energy_required(guide))
        for runner in guide.get("SpeedPRO", []):
            if runner.get("scratched"):
                continue
            rows.append(
                {
                    "RaceNo": _parse_int(guide.get("RaceNo")),
                    "RunnerNumber": _parse_int(runner.get("runnernumber")),
                    "HorseName": runner.get("name_eng") or runner.get("name"),
                    "Draw": _parse_int(runner.get("draw")),
                    "BrandNo": runner.get("brandno"),
                    "FitnessRatings": _parse_int(runner.get("fitnessrating")),
                    "SpeedPRO_Energy": _parse_int(runner.get("speedproenergy")),
                    "SpeedPRO_Energy_Difference": _parse_int(
                        runner.get("speedproenergydifference")
                    ),
                    "EnergyRequired": _parse_int(runner.get("energyrequired")),
                }
            )

    return pl.DataFrame(rows, schema=_ENERGY_SCHEMA, orient="row")


def _first_energy_required(guide: dict):
    """``energyrequired`` is a race-level figure repeated on each runner."""
    for runner in guide.get("SpeedPRO", []):
        val = runner.get("energyrequired")
        if str(val).strip():
            return val
    return None


def speedmap(race_num: int) -> str:
    """Fetch the SpeedMap for a race as a base64-encoded data URI.

    Args:
        race_num (int): Race number in the current meeting.

    Returns:
        str: Base64-encoded ``data:image/...`` string of the SpeedMap image.
    """
    guide = _fetch_speedguide(race_num)
    if guide is None:
        raise RuntimeError(f"No SpeedPro speed guide available for race {race_num}")
    return guide["RaceMap"]
