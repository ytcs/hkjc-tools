"""Top-level package for hkjc tools.

The library is organised by concern, with a single direction of dependency
(``data`` ← ``model`` ← ``sim``; ``eval`` and ``ledger`` sit to the side):

- **Data access** (`data`: `live`, `historical`, `racecard`, `sectional`,
  `odds_ticks`, the carryover flywheel): fetch, parse and persist race cards, odds,
  results, form, sectionals, and the live-odds/jackpot time series.
- **Geometry & domain** (`track`, `race`): :class:`~hkjc.track.TrackGeometry` is the
  per-venue/surface/course physical substrate (resolved by ``track_for``); the
  canonical :class:`~hkjc.race.Race` bundles runners + geometry + conditions and
  builds any model from itself — the agent-based-modelling seam.
- **Model** (`model`, `harville_model`): turn inputs into a swappable
  :class:`~hkjc.model.RaceModel` that generates finishing orders. Harville /
  Plackett-Luce is the reference implementation; bring your own by subclassing.
- **Simulation** (`simulation`): draw finishing orders from a model and estimate
  the probability and EV of arbitrary bets you compose yourself.
- **Bets** (`bets`): price HKJC pools from a model and the carryover detector/cover.
- **Evaluation** (`eval`): the harness that judges models/strategies (calibration,
  CLV, walk-forward, selection).
- **Memory** (`ledger`): durable, indexed journal of findings the research loop
  accumulates (see :mod:`hkjc.ledger`).

Canonical import paths live under their concern, e.g. ``hkjc.data.live``,
``hkjc.data.racecard``, ``hkjc.betting.carryover``.
"""
from importlib.metadata import version as _version

__all__ = [
    "core", "data", "model", "betting", "backtesting", "ledger",
    "RaceModel", "PlackettLuceModel", "ConditionalLogit", "RaceSimulator",
    "Race", "Runner", "TrackGeometry", "track_for", "TRACKS",
    "OddsTickStore", "FeatureStore", "DataSource", "Ledger", "pareto_filter",
    "generate_historical_data",
]

try:
    __version__ = _version(__name__)
except Exception:  # pragma: no cover - best-effort version resolution
    __version__ = "0.0.0"

from . import core, data, model, betting, backtesting, ledger
from .core import (Race, Runner, RaceModel, RaceSimulator, TrackGeometry,
                   track_for, TRACKS, DataSource, pareto_filter)
from .data import OddsTickStore, FeatureStore
from .data.processing import generate_historical_data
from .model import PlackettLuceModel, ConditionalLogit
from .ledger import Ledger
