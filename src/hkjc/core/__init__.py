"""Core layer — shared contracts, domain objects, geometry and facts.

Nothing in ``core`` imports any other hkjc package; everything else depends on it.

- :mod:`~hkjc.core.contracts` — the :class:`RaceModel` and :class:`DataSource` seams.
- :mod:`~hkjc.core.race` — the canonical :class:`Race` / :class:`Runner` bundle.
- :mod:`~hkjc.core.track` — :class:`TrackGeometry`, the per-venue physical substrate.
- :mod:`~hkjc.core.simulation` — Monte-Carlo composer over any :class:`RaceModel`.
- :mod:`~hkjc.core.hkjc_facts` — site quirks + betting rules (single source of truth).
- :mod:`~hkjc.core.utils` — HTML/table parsing and shared helpers.
"""
from . import contracts, hkjc_facts, race, simulation, track, utils
from .contracts import (CallableSource, DataSource, KEY_COLUMNS, RACE_KEYS,
                        RaceModel, make_race_id)
from .race import Race, Runner
from .simulation import RaceSimulator
from .track import TRACKS, TrackGeometry, track_for
from .utils import pareto_filter

__all__ = [
    "contracts", "hkjc_facts", "race", "simulation", "track", "utils",
    "RaceModel", "DataSource", "CallableSource", "KEY_COLUMNS", "RACE_KEYS", "make_race_id",
    "Race", "Runner", "RaceSimulator", "TrackGeometry", "track_for", "TRACKS",
    "pareto_filter",
]
