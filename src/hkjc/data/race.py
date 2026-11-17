"""One race's facts, bundled: meeting/course geometry, runners (number, brand id, draw) and the
official outcome. Pre-race facts (distance, course, draw, silks) may feed any product; the official
outcome (placings, times, sectionals, running positions) must not feed the video-only product (the
final product may use it)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from .tracks import TrackGeometry, track_for

CACHE = Path("data_cache/race_official")


@dataclass(frozen=True)
class Runner:
    no: int
    horse: str
    horse_id: str
    draw: int


@dataclass(frozen=True)
class RaceInfo:
    date: str  # YYYYMMDD
    venue: str  # "HV" | "ST"
    race_no: int
    distance: float
    course: str  # "A", "B", "C", "C+3", ... or "AWT"
    surface: str  # "turf" | "awt"
    going: str
    race_class: str
    runners: tuple
    results: pl.DataFrame = field(repr=False, compare=False)  # official outcome: never an input to the video-only product
    sectionals: pl.DataFrame = field(repr=False, compare=False)  # official outcome (same contract)

    @property
    def key(self) -> str:
        return f"{self.date}_{self.venue}_{self.race_no}"

    @property
    def geometry(self) -> TrackGeometry:
        return track_for(self.venue, self.distance, self.surface)

    @property
    def width(self) -> float:
        """Racing-rail -> outer-rail width (m) for this race's course setting."""
        return self.geometry.width(self.course)

    @property
    def n(self) -> int:
        return len(self.runners)


def _cached(path: Path, fetch):
    if path.exists():
        return pl.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def race_info(date: str, venue: str, race_no: int) -> RaceInfo:
    """Fetch (cached under data_cache/race_official) the results + sectionals of one race."""
    from .historical import get_race_data
    from .sectional import get_sectional_times

    d = str(date).replace("-", "").replace("/", "")
    slash = f"{d[:4]}/{d[4:6]}/{d[6:]}"
    res = _cached(
        CACHE / f"{d}{venue}{race_no}_results.parquet", lambda: get_race_data(slash, venue, race_no)
    )
    sec = _cached(
        CACHE / f"{d}{venue}{race_no}_sectional.parquet",
        lambda: get_sectional_times(slash, race_no),
    )
    m = res.row(0, named=True)
    label = m["Course"]
    surface = "awt" if "ALL WEATHER" in label.upper() else "turf"
    course = "AWT" if surface == "awt" else (label.split('"')[1] if '"' in label else "A")
    runners = tuple(
        sorted(
            (
                Runner(int(r["HorseNo"]), r["Horse"].split(" (")[0], r["HorseID"], int(r["Dr"]))
                for r in res.iter_rows(named=True)
            ),
            key=lambda r: r.no,
        )
    )
    return RaceInfo(
        d,
        venue,
        int(race_no),
        float(m["RaceDistance"]),
        course,
        surface,
        m["Going"],
        m["RaceClass"],
        runners,
        res,
        sec,
    )
