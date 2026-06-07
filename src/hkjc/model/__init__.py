"""Model layer — the causal-physics generative win model.

``hkjc.model`` is the canonical (and only) win model: latent physical CAUSES → generative
race dynamics → finishing ORDER → every pool price. The acausal additive-regression model
(the meta-signal feature tables + fitted-gain assembly) was purged in 2026-06; the causal
state-space that succeeded it was folded in here from the former ``hkjc.statespace`` package.
See ``docs/PHILOSOPHY.md`` (causal-not-correlational; first-principles-not-regression) and
``docs/CAUSAL_MODEL.md``.

The three generative layers:

  1. LATENT ATTRIBUTE STATE — :mod:`~hkjc.model.latent` (``LatentState`` θ + ``PopulationPrior``),
     inferred from the PATH by :mod:`~hkjc.model.identify` / :mod:`~hkjc.model.field` and the
     first-principles EOM in :mod:`~hkjc.model.dynamics`.
  2. GENERATIVE RACE MODEL — :mod:`~hkjc.model.generative` (θ+conditions → first-passage law →
     sampled paths → ``OrderDistribution``), priced by :mod:`~hkjc.model.pricer` (``PoolPricer``).
     IDENTIFIED via :mod:`~hkjc.model.likelihood` (the path-reconstruction loss).
  3. TWO-OBJECTIVE LOOP — IDENTIFY honestly from the path; CALIBRATE coupling + VALIDATE on the
     pool book (:mod:`~hkjc.model.coupling`, :mod:`~hkjc.model.gate`). Coupling is gate-calibrated,
     not honest-strength (#40).

:func:`~hkjc.model.pure_mechanism.pure_mechanism_theta_builder` is the meta-signal-free build
(journal 0050): θ from Hill critical-velocity → de-confounded cruise → population prior, every
meta-signal STRIPPED so form/jockey must RE-EMERGE as consequences, never be consumed as inputs.

Retained substrate (physics + honest references, NOT the purged regression model):
  * :mod:`~hkjc.model.energetics` / :mod:`~hkjc.model.energetics_dynamic` — critical-velocity /
    dynamic W'-balance engines.
  * :mod:`~hkjc.model.calibration` / :mod:`~hkjc.model.calibration_bayes` — empirical-Bayes
    (band×rating) ability population prior.
  * :mod:`~hkjc.model.plackett_luce` / :mod:`~hkjc.model.harville_model` /
    :mod:`~hkjc.model.conditional_logit` / :mod:`~hkjc.model.baselines` — the Harville-on-odds,
    results-logit, and market/uniform references the causal model is scored against.
"""
from ..core.contracts import RaceModel

# --- retained substrate: physics engines + honest odds/market references -------------------
from .plackett_luce import PlackettLuceModel
from .conditional_logit import ConditionalLogit, conditional_logit_fit_predict
from .baselines import market_baseline_fit_predict, uniform_fit_predict
from .energetics import EnergeticsRaceModel, estimate_run_energetics, section_distances

# --- the causal-physics generative win model (formerly hkjc.statespace) --------------------
from .latent import THETA_AXES, LatentState, PopulationPrior
from .causes import (ControlCause, LatentCauses, PhysiologyCause, PrepCause,
                     infer_crossed_causes)
from .conditions import RaceConditions
from .coupling import CouplingCalibrator, CouplingGains
from .likelihood import PathLikelihood, SurfaceTerm, profile_term
from .generative import GenerativeRaceModel, OrderDistribution
from .pricer import PoolPricer
from .gate import calibrate_coupling, gate_score, pool_book_score
from .pipeline import build_fit_predict, predict_race, smoke_race
from .pure_mechanism import (attach_pure_mechanism, build_pure_mechanism_panel,
                             make_pure_mechanism_pre_fit, pure_mechanism_theta_builder)

__all__ = [
    "RaceModel",
    # substrate / references
    "PlackettLuceModel",
    "ConditionalLogit", "conditional_logit_fit_predict",
    "market_baseline_fit_predict", "uniform_fit_predict",
    "EnergeticsRaceModel", "estimate_run_energetics", "section_distances",
    # causal-physics generative model
    "LatentState", "PopulationPrior", "THETA_AXES",
    "LatentCauses", "PhysiologyCause", "ControlCause", "PrepCause", "infer_crossed_causes",
    "RaceConditions",
    "PathLikelihood", "SurfaceTerm", "profile_term",
    "GenerativeRaceModel", "OrderDistribution",
    "PoolPricer",
    "CouplingCalibrator", "CouplingGains",
    "predict_race", "build_fit_predict", "smoke_race",
    "gate_score", "pool_book_score", "calibrate_coupling",
    "build_pure_mechanism_panel", "attach_pure_mechanism",
    "make_pure_mechanism_pre_fit", "pure_mechanism_theta_builder",
]
