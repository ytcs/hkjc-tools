"""Data-access layer: acquisition and persistence only.

Modules here fetch and parse data from HKJC (and persist time series); they do
**not** compute edge, fit models, or evaluate bets. One direction of dependency:
everything else depends on ``data``; ``data`` depends on nothing in the package
except shared parsing helpers in :mod:`hkjc.utils`.

- :mod:`~hkjc.data.live` — live race cards and live odds via the GraphQL endpoint.
- :mod:`~hkjc.data.historical` — past results and per-horse form (HTML scrape).
- :mod:`~hkjc.data.racecard` — pre-race racecard (ratings, gear, weights, …).
- :mod:`~hkjc.data.sectional` — per-horse sectional split times.
- :mod:`~hkjc.data.processing` — build historical datasets across a date range.
- :mod:`~hkjc.data.ticks` — capture/persist the live-odds and jackpot-pool time series.
- :mod:`~hkjc.data.carryover_ledger` — the carryover event ledger (flywheel).
- :mod:`~hkjc.data.feature_store` — assemble sources into a modelling frame.

The ``DataSource`` protocol (uniform input seam) lives in :mod:`hkjc.core.contracts`.
"""
from . import (live, historical, processing, ticks,
               feature_store, carryover_ledger, capture,
               jackpot_history, sectional, racecard, corunning)
from .ticks import OddsTickStore, JackpotCaptureStore
from .carryover_ledger import CarryoverLedger
from .capture import capture_once, settle_meeting, settle_recent
from ..core.contracts import DataSource, CallableSource, KEY_COLUMNS
from .feature_store import FeatureStore

__all__ = [
    "live", "historical", "racecard", "sectional", "corunning", "processing",
    "ticks", "feature_store", "carryover_ledger",
    "capture", "jackpot_history",
    "OddsTickStore", "JackpotCaptureStore", "CarryoverLedger",
    "capture_once", "settle_meeting", "settle_recent",
    "DataSource", "CallableSource", "FeatureStore", "KEY_COLUMNS",
]
