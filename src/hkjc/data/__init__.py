"""Data-access layer: acquisition and persistence only.

Modules here fetch and parse data from HKJC (and persist time series); they do
**not** compute edge, fit models, or evaluate bets. One direction of dependency:
everything else depends on ``data``; ``data`` depends on nothing in the package
except shared parsing helpers in :mod:`hkjc.utils`.

- :mod:`~hkjc.data.live` — live race cards and live odds via the GraphQL endpoint.
- :mod:`~hkjc.data.historical` — past results and per-horse form (HTML scrape).
- :mod:`~hkjc.data.speedpro` — SpeedPro energy/fitness figures and speed maps.
- :mod:`~hkjc.data.processing` — build historical datasets across a date range.
- :mod:`~hkjc.data.odds_ticks` — capture and persist the live-odds time series.
- :mod:`~hkjc.data.source` — the ``DataSource`` protocol (uniform input seam).
- :mod:`~hkjc.data.feature_store` — assemble sources into a modelling frame.
"""
from . import (live, historical, speedpro, processing, odds_ticks, source,
               feature_store, jackpot_ticks, carryover_ledger)
from .odds_ticks import OddsTickStore
from .jackpot_ticks import JackpotCaptureStore
from .carryover_ledger import CarryoverLedger
from .source import DataSource, CallableSource, KEY_COLUMNS
from .feature_store import FeatureStore

__all__ = [
    "live", "historical", "speedpro", "processing", "odds_ticks",
    "source", "feature_store", "jackpot_ticks", "carryover_ledger",
    "OddsTickStore", "JackpotCaptureStore", "CarryoverLedger",
    "DataSource", "CallableSource", "FeatureStore", "KEY_COLUMNS",
]
