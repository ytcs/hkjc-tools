"""Batch processing helpers for building historical datasets."""
from __future__ import annotations
from typing import List

import polars as pl
from tqdm import tqdm
from datetime import datetime as dt

from .historical import _extract_horse_data, _extract_race_data, _clean_horse_data
from ..core.utils import _validate_date


def _historical_process_single_date_venue(date, venue_code: str) -> List[pl.DataFrame]:
    dfs = []
    iter_date = tqdm(
        range(1, 12), desc=f"Processing {date} {venue_code} ...", leave=False)
    for race_number in iter_date:
        try:
            dfs.append(_extract_race_data(date.strftime('%Y/%m/%d'),
                                          venue_code, race_number))
        except Exception:
            if race_number == 1:
                iter_date.close()
                return []
    return dfs


def generate_historical_data(start_date: str, end_date: str) -> pl.DataFrame:
    """Generate historical race dataset from start_date to end_date (inclusive).

    Args:
        start_date (str): Date in 'YYYY-MM-DD' format.
        end_date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        pl.DataFrame: DataFrame with all records.
    """
    _validate_date(start_date)
    _validate_date(end_date)
    start_dt = dt.strptime(start_date, '%Y-%m-%d')
    end_dt = dt.strptime(end_date, '%Y-%m-%d')

    dfs = []

    for date in tqdm(pl.date_range(start_dt, end_dt, interval='1d', eager=True, closed='both'), leave=False, desc='Scanning for horse IDs ...'):
        for venue_code in ['ST', 'HV']:
            dfs += _historical_process_single_date_venue(date, venue_code)

    if dfs == []:
        raise ValueError(
            "Failed to obtain any race data. This could be due to invalid date range, or server requests limit. Please try again later.")

    horse_ids = pl.concat(dfs)['HorseID'].unique()

    # Use horse track records
    dfs = [_extract_horse_data(horse_id) for horse_id in tqdm(horse_ids, desc='Processing horses ...', leave=False)]
    df = pl.concat(dfs)

    try:
        return _clean_horse_data(df).filter(pl.col('Date').is_between(start_dt, end_dt))
    except Exception:
        print('Failed to clean data. Returning raw data for debug.')
    return df
