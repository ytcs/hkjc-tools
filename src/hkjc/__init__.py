"""Top-level package for hkjc tools.

The library is organised in three layers:

- **Data access** (`live`, `historical`, `speedpro`): fetch and parse race cards,
  odds, results, form and SpeedPro figures from HKJC.
- **Model** (`model`, `harville_model`): turn market odds into a swappable
  :class:`~hkjc.model.RaceModel` that can generate finishing orders. Harville /
  Plackett-Luce is the reference implementation; bring your own model by
  subclassing ``RaceModel``.
- **Simulation** (`simulation`): draw finishing orders from a model and estimate
  the probability and EV of arbitrary bets you compose yourself.
"""
from importlib.metadata import version as _version

__all__ = [
    "live", "historical", "speedpro", "features",
    "harville_model", "model", "simulation",
    "RaceModel", "PlackettLuceModel", "RaceSimulator",
    "pareto_filter", "speedpro_energy", "speedmap",
    "generate_historical_data",
]

try:
    __version__ = _version(__name__)
except Exception:  # pragma: no cover - best-effort version resolution
    __version__ = "0.0.0"

from . import live, historical, speedpro, features, harville_model, model, simulation
from .model import RaceModel, PlackettLuceModel
from .simulation import RaceSimulator
from .processing import generate_historical_data
from .utils import pareto_filter
from .speedpro import speedmap, speedpro_energy
