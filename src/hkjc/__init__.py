"""Top-level package for hkjc tools.

Organised into seven packages with a one-way dependency graph (a package imports only
from those below it); ``backtesting`` and ``agents`` sit on top and nothing imports them:

- **core** — shared contracts (:class:`RaceModel`, :class:`DataSource`), the canonical
  :class:`Race` / :class:`TrackGeometry`, the Monte-Carlo simulator, and ``hkjc_facts``
  (the single source for HKJC site quirks + betting rules).
- **data** — connectors/scrapers (live + historical) and the persistence stores; acquires
  and persists only, computes no edge.
- **model** — the production win model: the leak-free fit→predict pipeline
  (:mod:`hkjc.model.fit`, the ``wald`` first-passage diffusion law + per-venue mean-field
  terms), the ability features, the empirical-Bayes calibration, and the baked constants.
- **betting** — price pools, value the closing line, and stake against a ``Strategy``.
- **backtesting** — replay history, enforce no-lookahead (``AsOfView`` + the leakage check),
  and score models/strategies (% of market, ECE, selection control).
- **vision** — the (opt-in, heavy) CV trajectory pipeline.
- **agents** — orchestration: the three agent prompts (modelling / strategist / trading)
  and the cross-session :class:`Journal` they communicate through.

Canonical import paths live under their concern, e.g. ``hkjc.core.track``,
``hkjc.data.live``, ``hkjc.model.fit``, ``hkjc.betting.carryover``.
"""
from importlib.metadata import version as _version

__all__ = [
    "core", "data", "model", "betting", "backtesting", "agents",
    "RaceModel", "PlackettLuceModel", "ConditionalLogit", "RaceSimulator",
    "Race", "Runner", "TrackGeometry", "track_for", "TRACKS",
    "OddsTickStore", "FeatureStore", "DataSource", "Journal", "pareto_filter",
    "generate_historical_data",
]

try:
    __version__ = _version(__name__)
except Exception:  # pragma: no cover - best-effort version resolution
    __version__ = "0.0.0"

from . import agents, backtesting, betting, core, data, model
from .core import (Race, Runner, RaceModel, RaceSimulator, TrackGeometry,
                   track_for, TRACKS, DataSource, pareto_filter)
from .data import OddsTickStore, FeatureStore
from .data.processing import generate_historical_data
from .model import PlackettLuceModel, ConditionalLogit
from .agents import Journal
