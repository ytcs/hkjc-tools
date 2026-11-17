"""Small parsing helpers shared by the HKJC readers (HTML tables, safe casts, date checks)."""
import polars as pl
import bs4
import re


def _parse_html_table(table: bs4.element.Tag, skip_header=False) -> pl.DataFrame:
    """Parse an HKJC HTML table into a DataFrame; non-word characters are stripped from column names."""
    if table is None:
        raise ValueError("No table found in HTML tag")

    headers = []
    if not skip_header:
        header_row = table.find('thead')
        if header_row:
            headers = [td.get_text(strip=True) for td in header_row.find_all('td')]
        else:
            first_row = table.find('tr')
            if first_row:
                headers = [th.get_text(strip=True)
                        for th in first_row.find_all('th')]
                if not headers:
                    headers = [td.get_text(strip=True)
                            for td in first_row.find_all('td')]

    data = []
    tbody = table.find('tbody')
    rows = tbody.find_all('tr') if tbody else table.find_all('tr')

    start_idx = 1 if not tbody and headers else 0

    for row in rows[start_idx:]:
        cells = row.find_all(['td', 'th'])
        row_data = [cell.get_text(separator=' ',strip=True) for cell in cells]
        if row_data:
            data.append(row_data)

    if not headers:
        headers = [f"column_{i}" for i in range(len(data[0]))] if data else []

    # pad/truncate rows to the header width
    if data:
        max_cols = len(headers)
        data = [row + [''] * (max_cols - len(row)) if len(row)
                < max_cols else row[:max_cols] for row in data]

    df = pl.DataFrame(data, schema=headers, orient='row')
    df.columns = [re.sub(r'[^\w]', '', col)
                         for col in df.columns]
    return df


