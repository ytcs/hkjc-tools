"""Top-level package for hkjc tools.

The library is organised by concern, with a single direction of dependency
(``data`` ← ``model`` ← ``sim``; ``eval`` and ``ledger`` sit to the side):

- **Data access** (`data`: `live`, `historical`, `racecard`, `sectional`,
  `odds_ticks`, the carryover flywheel): fetch, parse and persist race cards, odds,
  results, form, sectionals, and the live-odds/jackpot time series.
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
``hkjc.data.racecard``, ``hkjc.bets.carryover``.
"""
from importlib.metadata import version as _version

__all__ = [
    "data", "harville_model", "model", "simulation", "bets", "eval", "ledger",
    "RaceModel", "PlackettLuceModel", "ConditionalLogit", "RaceSimulator",
    "OddsTickStore", "FeatureStore", "DataSource", "Ledger", "pareto_filter",
    "generate_historical_data",
]

try:
    __version__ = _version(__name__)
except Exception:  # pragma: no cover - best-effort version resolution
    __version__ = "0.0.0"

from . import data, harville_model, model, simulation, bets, eval, ledger
from .data import OddsTickStore, FeatureStore, DataSource
from .data.processing import generate_historical_data
from .model import RaceModel, PlackettLuceModel, ConditionalLogit
from .simulation import RaceSimulator
from .ledger import Ledger
from .utils import pareto_filter
