"""Stage 8/9 — EXPORT + QA: write the typed schema and validate vs published sectionals.

Export to Parquet (plays with the existing pandas/feature-store stack) + a JSONL
RaceTrajectory summary. QA's primary check is recovered section times vs HKJC
published sectionals (already scraped; see docs/HKJC_DATA.md § trajectory). pandas/pyarrow
imported lazily; JSON export is stdlib-only.
"""
from __future__ import annotations

import json
from typing import Optional

from .schema import RaceTrajectory


def to_jsonl(race: RaceTrajectory, path: str) -> None:
    """Write one RaceTrajectory as JSON (stdlib-only, no heavy deps)."""
    with open(path, "w") as fh:
        json.dump(race.as_dict(), fh, indent=2)


def to_parquet(race: RaceTrajectory, path: str) -> None:
    """Flatten per-runner points to a tidy Parquet table. Lazy pandas/pyarrow. STUB."""
    raise NotImplementedError(
        "Stub. import pandas (lazy); explode runners->points into rows keyed by\n"
        "(date, race_no, runner, t) with columns d, off, speed, accel, x, y, gap, conf;\n"
        "df.to_parquet(path). Mirrors the documented dict schema in schema.py."
    )


def qa(
    race: RaceTrajectory,
    published_sectionals: Optional[dict] = None,
    official_order: Optional[list[int]] = None,
) -> dict:
    """Validate the recovered trajectory. Returns a QA dict (also stored on race.qa).

    Checks (STUB):
      - section-time error vs published sectionals (PRIMARY / kill criterion):
        integrate recovered speed to each sectional d-boundary, diff vs
        get_sectional_times(date, race_no); target within ~0.1-0.3 s (Mode B).
      - finish-order match vs official result.
      - per-runner total_distance ~ race distance (wide trips legitimately exceed).
      - speed-band sanity (~14-19 m/s gallop).
    """
    raise NotImplementedError("Stub — see CV_PIPELINE Stage 9 (validation).")
