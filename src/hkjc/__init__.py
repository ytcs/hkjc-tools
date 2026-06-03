"""Top-level package for hkjc tools.

The library is organised by concern, with a single direction of dependency
(``data`` ← ``model`` ← ``sim``; ``eval`` and ``ledger`` sit to the side):

- **Data access** (`data`: `live`, `historical`, `speedpro`, `odds_ticks`,
  `processing`): fetch, parse and persist race cards, odds, results, form, SpeedPro
  figures, and the live-odds time series. Acquisition/persistence only.
- **Model** (`model`, `harville_model`): turn inputs into a swappable
  :class:`~hkjc.model.RaceModel` that generates finishing orders. Harville /
  Plackett-Luce is the reference implementation; bring your own by subclassing.
- **Simulation** (`simulation`): draw finishing orders from a model and estimate
  the probability and EV of arbitrary bets you compose yourself.
- **Evaluation** (`eval`): the harness that judges models/strategies — currently
  closing-line value (`eval.clv`).
- **Memory** (`ledger`): durable, indexed journal of findings the research loop
  accumulates (see :mod:`hkjc.ledger`).

Submodules keep their old import names re-exported here (``hkjc.live`` etc.); the
canonical paths now live under their concern (``hkjc.data.live``).
"""
from importlib.metadata import version as _version

__all__ = [
    "data", "live", "historical", "speedpro", "features",
    "harville_model", "model", "simulation", "eval", "ledger",
    "RaceModel", "PlackettLuceModel", "ConditionalLogit", "RaceSimulator",
    "OddsTickStore", "FeatureStore", "DataSource", "Ledger", "pareto_filter",
    "speedpro_energy", "speedmap", "generate_historical_data",
]

try:
    __version__ = _version(__name__)
except Exception:  # pragma: no cover - best-effort version resolution
    __version__ = "0.0.0"

from . import data, features, harville_model, model, simulation, eval, ledger
from .data import live, historical, speedpro, OddsTickStore, FeatureStore, DataSource
from .data.processing import generate_historical_data
from .data.speedpro import speedmap, speedpro_energy
from .model import RaceModel, PlackettLuceModel, ConditionalLogit
from .simulation import RaceSimulator
from .ledger import Ledger
from .utils import pareto_filter
