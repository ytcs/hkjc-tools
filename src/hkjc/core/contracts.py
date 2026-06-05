"""Core contracts — the two protocols every layer agrees on.

Both seams live here so the dependency graph has a single sink: ``model`` depends on
:class:`RaceModel`, ``data`` depends on :class:`DataSource`, and neither has to import
the other. Nothing in ``core`` imports any other hkjc package.

- :class:`RaceModel` — anything that can generate finishing orders for a race. The one
  capability the Monte-Carlo simulator needs, so any model (Plackett-Luce on odds, a
  results-trained model, a finish-time model) is swappable behind it.
- :class:`DataSource` — a uniform contract for every input the agent adds. A source
  produces **runner-keyed records** for a meeting; the feature store joins them into one
  modelling frame. Each source declares its ``languages`` so coverage can be reported
  per source language (the language-parity enforcement mechanism).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Protocol, Sequence, runtime_checkable

import numpy as np
import polars as pl

# ---------------------------------------------------------------- RaceModel seam


class RaceModel(ABC):
    """Abstract race outcome model.

    Implementations must expose the field size ``n`` and a vectorised
    :meth:`sample_orders`. Everything downstream (probability and EV estimation for
    arbitrary bet types) is derived from samples, so a custom model only has to know
    how to produce plausible finishing orders.
    """

    n: int

    @abstractmethod
    def sample_orders(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` finishing orders.

        Returns:
            np.ndarray: integer array of shape ``(size, n)``. Each row is a
            permutation of ``0..n-1`` giving a finishing order: row ``[0]`` is the
            (0-indexed) winner, row ``[1]`` the runner-up, and so on.
        """
        raise NotImplementedError


# ---------------------------------------------------------------- DataSource seam

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
