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
- :mod:`~hkjc.model.features` — leak-free ability features (form, jockey, style).
- :mod:`~hkjc.model.fit` — the leak-free fit→predict pipeline (the production win model:
  the ``wald`` drift-diffusion / first-passage law, plus the ``pl`` closed-form probe).
- :mod:`~hkjc.model.constants` — baked calibration gains (FORM_GAIN, DRAW_GAIN, …).
- :mod:`~hkjc.model.calibration` — empirical-Bayes (band×rating) ability population fit.

The interacting agent-based sims (kinematic traffic + power-balance EOM) were deprecated
in 2026-06 — ``wald`` + per-venue mean-field terms beat them for WIN at ~35× the speed. The
physics they encoded is preserved in ``docs/THEORY.md`` (Appendix).
"""
from ..core.contracts import RaceModel
from . import constants, features, fit
from .plackett_luce import PlackettLuceModel
from .conditional_logit import ConditionalLogit, conditional_logit_fit_predict
from .baselines import market_baseline_fit_predict, uniform_fit_predict
from .energetics import EnergeticsRaceModel, estimate_run_energetics, section_distances

__all__ = [
    "RaceModel", "PlackettLuceModel",
    "ConditionalLogit", "conditional_logit_fit_predict",
    "market_baseline_fit_predict", "uniform_fit_predict",
    "EnergeticsRaceModel", "estimate_run_energetics", "section_distances",
    "constants", "features", "fit",
]
