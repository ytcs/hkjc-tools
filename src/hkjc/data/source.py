"""The ``DataSource`` seam — a uniform contract for every input the agent adds.

Onboarding a new data source (odds ticks, results, SpeedPro, a Chinese tipster
feed, video-derived trip notes) should be the *same* shape of engineering job. A
``DataSource`` produces **runner-keyed records** for a meeting; the
:class:`~hkjc.data.feature_store.FeatureStore` joins them into one modelling frame.

Every source declares its ``languages`` so the harness can report coverage and
lift **broken down by source language** — the enforcement mechanism for the
language-parity requirement (``docs/AGENT_ARCHITECTURE.md`` §0.3): a Chinese-source
blind spot shows up as a measured number, not a silent gap.

Canonical keys (human-readable, walk-forward friendly — ``date`` is a column, not
hidden in an id):

    date     str   meeting date (the string the upstream source uses)
    venue    str   ST | HV
    race_no  int   race number within the meeting
    runner   int   saddle-cloth / runner number (HKJC runner i == index i-1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Protocol, Sequence, runtime_checkable

import polars as pl

KEY_COLUMNS: List[str] = ["date", "venue", "race_no", "runner"]
RACE_KEYS: List[str] = ["date", "venue", "race_no"]


@runtime_checkable
class DataSource(Protocol):
    """A source of runner-keyed records for a meeting.

    Attributes
    ----------
    name:       short unique identifier.
    languages:  source languages, e.g. ``["en-us"]``, ``["zh-hk"]`` or both.
    role:       ``"feature"`` (model inputs) or ``"outcome"`` (labels to predict).
    provides:   the value columns this source contributes (beyond the keys).
    """

    name: str
    languages: Sequence[str]
    role: str
    provides: Sequence[str]

    def fetch(self, date: str, venue: str) -> pl.DataFrame:
        """Return a frame with ``KEY_COLUMNS`` + ``provides`` for one meeting."""
        ...


@dataclass
class CallableSource:
    """A :class:`DataSource` backed by a plain function.

    Convenient for adapters over existing fetchers and for in-memory test sources.
    """

    name: str
    languages: List[str]
    provides: List[str]
    fetch_fn: Callable[[str, str], pl.DataFrame]
    role: str = "feature"

    def fetch(self, date: str, venue: str) -> pl.DataFrame:
        return self.fetch_fn(date, venue)


def make_race_id(date: str, venue: str, race_no: int) -> str:
    """Stable race id ``{YYYYMMDD}{VENUE}{race_no}`` from any date format."""
    digits = "".join(ch for ch in str(date) if ch.isdigit())
    return f"{digits}{venue}{int(race_no)}"
