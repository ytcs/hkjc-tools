"""Model layer — turn inputs into a swappable :class:`RaceModel`.

- :class:`~hkjc.model.base.RaceModel` — the contract (``sample_orders``).
- :class:`~hkjc.model.plackett_luce.PlackettLuceModel` — reference impl; the
  Harville-on-odds *baseline to beat*.
- :class:`~hkjc.model.conditional_logit.ConditionalLogit` — results-trained win
  model (breaks the odds-fitting circularity); ``conditional_logit_fit_predict``
  plugs it into the eval harness.
- :mod:`~hkjc.model.baselines` — market / uniform predictors as honest references.
- :mod:`~hkjc.model.energetics` — first-principles critical-velocity engine +
  generative `EnergeticsRaceModel`.
- :mod:`~hkjc.model.traffic` — Tier-A interacting traffic-flow model (draw geometry
  + car-following) over the `track` geometry.
"""
from .base import RaceModel
from .plackett_luce import PlackettLuceModel
from .conditional_logit import ConditionalLogit, conditional_logit_fit_predict
from .baselines import market_baseline_fit_predict, uniform_fit_predict
from .energetics import EnergeticsRaceModel, estimate_run_energetics, section_distances
from .traffic import TrafficRaceModel

__all__ = [
    "RaceModel", "PlackettLuceModel",
    "ConditionalLogit", "conditional_logit_fit_predict",
    "market_baseline_fit_predict", "uniform_fit_predict",
    "EnergeticsRaceModel", "estimate_run_energetics", "section_distances",
    "TrafficRaceModel",
]
