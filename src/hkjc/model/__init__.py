"""Model layer — turn inputs into a swappable :class:`RaceModel`.

- The :class:`RaceModel` contract (``sample_orders``) lives in :mod:`hkjc.core.contracts`.
- :class:`~hkjc.model.plackett_luce.PlackettLuceModel` — reference impl; the
  Harville-on-odds *baseline to beat*.
- :class:`~hkjc.model.conditional_logit.ConditionalLogit` — results-trained win
  model (breaks the odds-fitting circularity); ``conditional_logit_fit_predict``
  plugs it into the eval harness.
- :mod:`~hkjc.model.baselines` — market / uniform predictors as honest references.
- :mod:`~hkjc.model.energetics` — first-principles critical-velocity engine +
  generative `EnergeticsRaceModel`.
- :mod:`~hkjc.model.eom` — the single canonical power-balance equation of motion (JAX);
  mass, metabolic power, drag/drafting, reserve depletion as real terms.
- :mod:`~hkjc.model.dynamics` — the :class:`DynamicsRaceModel` wrapper over the EOM.
- :mod:`~hkjc.model.traffic` / :mod:`~hkjc.model.traffic_jax` — the calibrated kinematic
  interacting traffic sim (draw geometry + car-following); the production win-prob sim.
- :mod:`~hkjc.model.features` — leak-free ability features (form, jockey, style).
- :mod:`~hkjc.model.fit` — the leak-free fit→predict pipeline (the production model).
- :mod:`~hkjc.model.constants` — baked calibration gains (FORM_GAIN, SPEED_NOISE, …).
- :mod:`~hkjc.model.calibration` — empirical-Bayes population fit + bend-curvature grid.
"""
from ..core.contracts import RaceModel
from . import constants, features, fit
from .plackett_luce import PlackettLuceModel
from .conditional_logit import ConditionalLogit, conditional_logit_fit_predict
from .baselines import market_baseline_fit_predict, uniform_fit_predict
from .energetics import EnergeticsRaceModel, estimate_run_energetics, section_distances
from .traffic import TrafficRaceModel
from .dynamics import DynamicsRaceModel

__all__ = [
    "RaceModel", "PlackettLuceModel",
    "ConditionalLogit", "conditional_logit_fit_predict",
    "market_baseline_fit_predict", "uniform_fit_predict",
    "EnergeticsRaceModel", "estimate_run_energetics", "section_distances",
    "TrafficRaceModel", "DynamicsRaceModel",
    "constants", "features", "fit",
]
