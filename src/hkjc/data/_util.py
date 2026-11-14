"""Small parsing helpers shared by the HKJC readers (HTML tables, safe casts, date checks)."""
import polars as pl
from datetime import datetime as dt
import bs4
import re

def _try_int(value: str) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def _try_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def _validate_date(date_str: str) -> bool:
    # validate date format
    try:
        dt.strptime(date_str, "%Y-%m-%d")
    except Exception:
        raise ValueError("Date must be in 'YYYY-MM-DD' format")
    return True


def _validate_venue_code(venue_code: str) -> bool:
    if venue_code not in ['HV', 'ST']:
        raise ValueError(
            "Venue code must be 'HV' (Happy Valley) or 'ST' (Sha Tin)")
    return True


def _parse_html_table(table: bs4.element.Tag, skip_header=False) -> pl.DataFrame:
    """Parse an HTML table (HKJC format) into a Polars DataFrame
    """
    if table is None:
        raise ValueError("No table found in HTML tag")

    # Extract headers
    headers = []
    if not skip_header:
        header_row = table.find('thead')
        if header_row:
            headers = [td.get_text(strip=True) for td in header_row.find_all('td')]
        else:
            # If no thead, try first tr
            first_row = table.find('tr')
            if first_row:
                headers = [th.get_text(strip=True)
                        for th in first_row.find_all('th')]
                if not headers:
                    # If no th tags, use td tags from first row as headers
                    headers = [td.get_text(strip=True)
                            for td in first_row.find_all('td')]

    # Extract data rows
    data = []
    tbody = table.find('tbody')
    rows = tbody.find_all('tr') if tbody else table.find_all('tr')

    # Skip first row if it was used for headers
    start_idx = 1 if not tbody and headers else 0

    for row in rows[start_idx:]:
        cells = row.find_all(['td', 'th'])
        row_data = [cell.get_text(separator=' ',strip=True) for cell in cells]
        if row_data:  # Skip empty rows
            data.append(row_data)

    # Create DataFrame
    if not headers:
        # Generate default column names if no headers found
        headers = [f"column_{i}" for i in range(len(data[0]))] if data else []

    # Ensure all rows have the same number of columns
    if data:
        max_cols = len(headers)
        data = [row + [''] * (max_cols - len(row)) if len(row)
                < max_cols else row[:max_cols] for row in data]

    df = pl.DataFrame(data, schema=headers, orient='row')
    # Clean column names by removing special characters
    df.columns = [re.sub(r'[^\w]', '', col)
                         for col in df.columns]
    return df


