"""Official results page (LocalResults.aspx): race header, placings, finish times, running positions.
Incident rows (DISQ, DNF, PU, ...) are dropped; FinishTime is converted from m:ss.ff to seconds."""
from __future__ import annotations

import polars as pl
from bs4 import BeautifulSoup
from cachetools.func import ttl_cache

from ._util import _parse_html_table, get

HKJC_RACE_URL_TEMPLATE = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={date}&Racecourse={venue_code}&RaceNo={race_number}"

incidents = ['DISQ', 'DNF', 'FE', 'ML', 'PU', 'TNP', 'TO',
             'UR', 'VOID', 'WR', 'WV', 'WV-A', 'WX', 'WX-A', 'WXNR']

REQUEST_TIMEOUT = 10

HTML_HEADERS = {
    "Origin": "https://racing.hkjc.com",
    "Referer": "https://racing.hkjc.com",
    "Content-Type": "text/plain",
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
}


@ttl_cache(maxsize=100, ttl=3600)
def _soupify(url: str) -> BeautifulSoup:
    """GET + parse a page (1 h in-process cache)."""
    response = get(url, timeout=REQUEST_TIMEOUT, headers=HTML_HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')


def _soupify_race_page(date: str, venue_code: str, race_number: int) -> BeautifulSoup:
    """Parsed results page; ``date`` is YYYY/MM/DD."""
    url = HKJC_RACE_URL_TEMPLATE.format(
        date=date, venue_code=venue_code, race_number=race_number)
    return _soupify(url)


def _clean_race_data(df: pl.DataFrame) -> pl.DataFrame:
    """Drop incident rows, cast numeric columns, FinishTime m:ss.ff -> s."""
    if df.height == 0:
        return df

    df = df.with_columns(
        pl.col('Pla').str.split(' ').list.first().alias('Pla')
    ).filter(~pl.col('Pla').is_in(incidents))

    df = df.with_columns([
        pl.col('Pla').cast(pl.Int64, strict=False),
        pl.col('HorseNo').cast(pl.Int64, strict=False),
        pl.col('ActWt').cast(pl.Int64, strict=False),
        pl.col('DeclarHorseWt').cast(pl.Int64, strict=False),
        pl.col('Dr').cast(pl.Int64, strict=False),
        pl.col('WinOdds').cast(pl.Float64, strict=False)
    ])

    df = df.with_columns(
        (
            pl.col("FinishTime").str.splitn(":", 2).struct.field("field_0").cast(pl.Int64) * 60 +
            pl.col("FinishTime").str.splitn(
                ":", 2).struct.field("field_1").cast(pl.Float64)
        ).cast(pl.Float64).round(2).alias("FinishTime")
    )

    return df


def _extract_race_data(date: str, venue_code: str, race_number: int) -> pl.DataFrame:
    soup = _soupify_race_page(date, venue_code, race_number)
    table = soup.find('div', class_='race_tab').find('table')
    race_data = _parse_html_table(table)

    race_id = race_data.columns[0].replace(f'RACE{race_number}', '')
    race_class = race_data.item(1, 0).split('-')[0].strip()
    race_dist = race_data.item(1, 0).split('-')[1].strip().rstrip('M')
    race_name = race_data.item(2, 0).strip()
    going = race_data.item(1, 2).strip()
    course = race_data.item(2, 2).strip()

    race_info = {'Date': date,
                 'Venue': venue_code,
                 'RaceIndex': int(race_id),
                 'RaceNumber': race_number,
                 'RaceClass': race_class,
                 'RaceDistance': race_dist,
                 'RaceName': race_name,
                 'Going': going,
                 'Course': course}

    table = soup.find('div', class_='performance').find('table')
    race_data = (_parse_html_table(table)
                 .with_columns([
                     pl.lit(value).alias(key) for key, value in race_info.items()
                 ])
                 .with_columns(
                     pl.col("Horse").str.extract(r"\((.*?)\)")
                     .alias("HorseID")
    )
    )

    return race_data


def get_race_data(date: str, venue_code: str, race_number: int) -> pl.DataFrame:
    df = _extract_race_data(date, venue_code, race_number)
    return _clean_race_data(df)


