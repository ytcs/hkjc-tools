"""The pipeline — tie the layers into one leak-free fit→predict, with the contract made explicit.

The end-to-end spine of the canonical framework:

    surface ─▶ IDENTIFY θ (PathLikelihood / population prior, strictly-prior)
            ─▶ COUPLE (CouplingCalibrator, gate-calibrated gains)
            ─▶ GENERATE paths → JOINT order dist (GenerativeRaceModel)
            ─▶ PRICE the pool book (PoolPricer)
            ─▶ GATE: market skill% on the pool book (validate + calibrate coupling)

THE LEAK-FREE CONTRACT (made hard by construction):

  * θ per runner is STRICTLY-PRIOR — built from a horse's EARLIER runs only, via the train-only
    :class:`~hkjc.model.latent.PopulationPrior`. A race's own surface never enters the θ that
    predicts it. (Identifying a run's OWN θ for the population pool is a separate, pooling-only step.)
  * The population prior is fit on TRAIN ONLY, per walk-forward fold — passed in, never refit on test.
  * The generative sampler is a pure function of (θ, conditions) + a race-id seed — no outcome, no odds.
  * :func:`build_fit_predict` returns a ``(train, test) -> preds`` callable shaped for the production
    gate harness, so ``assert_no_lookahead`` and ``walk_forward_evaluate`` apply UNCHANGED.

This module provides the wiring and a RUNNABLE skeleton (:func:`predict_race`, :func:`smoke_race`)
that constructs every object and produces real win probabilities on a tiny in-memory slice. The
data-loading and per-horse strictly-prior θ assembly from real parquet panels is the BUILD-WAVE
(it mirrors ``studies/backtest_st.py`` ``load()`` + ``calibration.horse_energetics_trailing``).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .conditions import RaceConditions
from .coupling import CouplingCalibrator, CouplingGains
from .generative import GenerativeRaceModel, OrderDistribution
from .latent import LatentState
from .pricer import PoolPricer


def predict_race(
    thetas: list[LatentState],
    conditions: RaceConditions,
    *,
    calibrator: CouplingCalibrator | None = None,
    model: GenerativeRaceModel | None = None,
) -> tuple[OrderDistribution, np.ndarray]:
    """One race: strictly-prior θ + conditions → (joint order distribution, win probs).

    The core forward pass. ``calibrator`` carries the gate-calibrated coupling gains; ``model`` the
    sampler. Returns the joint order distribution (for the full pool book) and the win marginal (the
    headline). Deterministic given ``conditions.race_id``.
    """
    calibrator = calibrator or CouplingCalibrator()
    model = model or GenerativeRaceModel()
    mu, lam = calibrator.couple(thetas, conditions, model)
    od = model.order_distribution(thetas, conditions, mu=mu, lam=lam)
    return od, od.win_probs()


def build_fit_predict(
    *,
    n_samples: int = 4000,
    gains: CouplingGains | None = None,
    theta_builder=None,
    pre_fit=None,
    diff_coef: float = 170.0,
    awt_diff_mult: float = 2.0,
):
    """Return a leak-free ``fit_predict(train, test) -> preds`` for the production gate harness.

    The returned callable is the seam the gate (:mod:`hkjc.model.gate`) and ``assert_no_lookahead``
    exercise. ``theta_builder(train, race_group)`` produces the per-runner STRICTLY-PRIOR θ list and a
    :class:`RaceConditions` for one race; the default builder reads pre-computed strictly-prior θ columns
    already on the frame (``v_c_prior``/``w_prime_prior``/``a_prior``), so the skeleton runs and
    ``assert_no_lookahead`` passes (outcomes never read).

    ``pre_fit(train, test) -> test'`` is an optional, train-only frame preprocessor run ONCE per fold
    (not per race) — the real Stage-1 builder uses it to fit the train-only population and stamp each
    runner's production ``v_c_eb``/``w_prime_eb`` + strictly-prior ``a`` onto the test frame, so the
    per-race θ assembly is a cheap column read. It is part of the leak-free seam: it may train on
    ``train`` and read pre-race columns of ``test`` but never a ``test`` outcome (``assert_no_lookahead``
    enforces this — perturbing test outcomes must not change the stamped columns).
    """
    builder = theta_builder or _default_theta_builder
    calibrator = CouplingCalibrator(gains)
    model = GenerativeRaceModel(n_samples=n_samples, diff_coef=diff_coef, awt_diff_mult=awt_diff_mult)

    def _fit_predict(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
        if pre_fit is not None:
            test = pre_fit(train, test)
        out = []
        for _key, g in test.group_by("race_id", maintain_order=True):
            g = g.sort("runner")
            thetas, conditions = builder(train, g)
            _od, p = predict_race(thetas, conditions, calibrator=calibrator, model=model)
            out.append(g.with_columns(pl.Series("p", p)))
        return pl.concat(out)

    return _fit_predict


def _default_theta_builder(train: pl.DataFrame, g: pl.DataFrame):
    """Build θ + conditions for one race from PRE-COMPUTED strictly-prior columns (skeleton default).

    Reads ``v_c_prior`` / ``w_prime_prior`` / ``a_prior`` (the strictly-prior pooled latents a real
    loader produces) straight off the frame — never any outcome column. The BUILD-WAVE replaces this
    with the population-prior posterior over each horse's earlier runs.
    """
    vc = g["v_c_prior"].to_numpy().astype(float)
    wp = g["w_prime_prior"].to_numpy().astype(float)
    a = (g["a_prior"].to_numpy().astype(float) if "a_prior" in g.columns
         else np.ones(g.height))
    thetas = [LatentState(v_c=float(vc[i]), w_prime=float(wp[i]), rho=0.0, a=float(a[i]))
              for i in range(g.height)]
    from hkjc.core.hkjc_facts import AWT_DISTANCE
    distance = int(g["distance"][0])
    venue = str(g["venue"][0]) if "venue" in g.columns else "ST"
    surface = "awt" if distance == AWT_DISTANCE else "turf"
    conditions = RaceConditions(
        race_id=str(g["race_id"][0]), distance=distance, venue=venue, surface=surface,
        draws=g["draw"].to_numpy().astype(float) if "draw" in g.columns else np.arange(1, g.height + 1),
        field_size=g.height,
    )
    return thetas, conditions


def smoke_race(seed: int = 0, n: int = 8) -> dict:
    """Construct the whole stack on a tiny synthetic race and return win + pool prices.

    The minimal end-to-end SMOKE PATH: builds θ, conditions, the calibrator, the generative model,
    samples the joint order distribution, and prices win/place/forecast — proving the frame RUNS and
    produces a valid win distribution. Used by ``tests/test_model_skeleton.py``.
    """
    rng = np.random.default_rng(seed)
    thetas = [
        LatentState(v_c=float(16.0 + rng.normal(0, 0.4)),
                    w_prime=float(60.0 + rng.normal(0, 8.0)),
                    rho=0.0,
                    a=float(0.95 + rng.normal(0, 0.05)))
        for _ in range(n)
    ]
    conditions = RaceConditions(
        race_id=f"SMOKE{seed}", distance=1200, venue="ST", surface="turf",
        draws=np.arange(1, n + 1, dtype=float), field_size=n, band="sprint",
    )
    calibrator = CouplingCalibrator(CouplingGains(a_start_gain=0.5))
    model = GenerativeRaceModel(n_samples=5000)
    od, win = predict_race(thetas, conditions, calibrator=calibrator, model=model)
    pricer = PoolPricer(od)
    return {
        "win": win,
        "place": pricer.place(k=3),
        "forecast": pricer.forecast(),
        "order_dist": od,
    }
