"""Fit + predict for the production ABM — leak-free per-horse ability → win probabilities.

This is the model, not a study. Inside a walk-forward fold the energetics **population**
is fit on train only; each horse's ability is its empirical-Bayes estimate from its
strictly-prior runs, adapted to the test race's distance band and rating, then nudged by
the leak-free form and jockey measurements (the dominant signals). A debut / no-sectional
horse falls back to the rating-conditioned population mean. The model never sees odds.

The effective critical velocity each runner carries into the sim is::

    v_c_eff = pop.v_c.fixed(band, rtg)            # rating-conditioned population mean
            + λ_vc(n) · u_vc                      # shrunken per-horse sectional residual
            + FORM_GAIN · (form − 0.5)            # form ability coupling (the big one)
            + JOCKEY_GAIN · (jky − jky_mean)      # jockey trip/ability coupling

and the anaerobic reserve W′ is the rating-conditioned mean plus its shrunken residual.
These feed the interacting traffic sim (draw geometry + car-following) for win probs.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..core.hkjc_facts import AWT_DISTANCE
from ..core.track import track_for
from .calibration import (CALIBRATED_BEND_CURVATURE, EnergeticsPopulation,
                          distance_band, fit_energetics_population)
from .constants import FORM_GAIN, JOCKEY_GAIN, N_SAMPLES, PACE_GAIN, SPEED_NOISE, W_PRIME_GAIN
from .traffic_jax import win_probs


def _surface(distance: int) -> str:
    return "awt" if int(distance) == AWT_DISTANCE else "turf"


def horse_residuals(train: pl.DataFrame, pop: EnergeticsPopulation) -> dict:
    """Per-horse mean empirical-Bayes residual (v_c, w′) over the horse's train runs + count."""
    d = train.with_columns(
        pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("_b"))
    rtg = d["rtg"].to_numpy().astype(float)
    bands = d["_b"].to_list()
    horses = d["horse_id"].to_list()
    u_vc = d["v_c"].to_numpy() - np.array([pop.v_c.fixed(b, rtg[i]) for i, b in enumerate(bands)])
    u_w = d["w_prime"].to_numpy() - np.array([pop.w_prime.fixed(b, rtg[i]) for i, b in enumerate(bands)])
    agg: dict = {}
    for h, a, b in zip(horses, u_vc, u_w):
        s = agg.setdefault(h, [0, 0.0, 0.0])
        s[0] += 1
        s[1] += a
        s[2] += b
    return {h: (n, sa / n, sb / n) for h, (n, sa, sb) in agg.items()}


def fit_predict_traffic(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    jky_mean: float = 0.0,
    n_samples: int = N_SAMPLES,
    form_gain: float = FORM_GAIN,
    jockey_gain: float = JOCKEY_GAIN,
    w_prime_gain: float = W_PRIME_GAIN,
    speed_noise: float = SPEED_NOISE,
    pace_gain: float = PACE_GAIN,
    bend: float | None = None,
) -> pl.DataFrame:
    """Fit ability on ``train``, predict win probabilities for each race in ``test``.

    Leak-free: the population fit and per-horse residuals use train only; the test frame
    must already carry the leak-free ``form`` / ``jky_sr`` / ``style`` features and a
    backward-as-of ``rtg``. ``bend`` defaults to the per-race venue's calibrated value.
    Returns the test frame with ``p`` (win prob), ``v_c_eb``, ``w_prime_eb`` columns.
    """
    pop = fit_energetics_population(train, rating_col="rtg")
    res = horse_residuals(train, pop)
    lam_vc = lambda n: pop.v_c.tau2 / (pop.v_c.tau2 + pop.v_c.sigma2 / n) if n > 0 else 0.0       # noqa: E731
    lam_w = lambda n: pop.w_prime.tau2 / (pop.w_prime.tau2 + pop.w_prime.sigma2 / n) if n > 0 else 0.0  # noqa: E731
    preds = []
    for race_id, g in test.group_by("race_id"):
        g = g.sort("runner")
        dist = int(g["distance"][0])
        venue = g["venue"][0]
        rtg = g["rtg"].to_numpy().astype(float)
        form = g["form"].to_numpy().astype(float)
        jky = g["jky_sr"].to_numpy().astype(float)
        band = distance_band(dist)
        vc, wp = [], []
        for i, h in enumerate(g["horse_id"].to_list()):
            n, u_vc, u_w = res.get(h, (0, 0.0, 0.0))
            vc.append(pop.v_c.fixed(band, rtg[i]) + lam_vc(n) * u_vc
                      + form_gain * (form[i] - 0.5) + jockey_gain * (jky[i] - jky_mean))
            # w_prime_gain amplifies the per-horse reserve spread (the EB residual) — the
            # stamina analogue of form_gain (market under-weights W' at gain 1.0).
            wp.append(pop.w_prime.fixed(band, rtg[i]) + w_prime_gain * lam_w(n) * u_w)
        geom = track_for(venue, dist, _surface(dist))
        course = g["course"][0] if "course" in g.columns else "A"
        # within-race style z-score → pace policy (front-runners take position, closers wait)
        style = g["style"].to_numpy().astype(float)
        sd = style.std()
        pace_bias = (style - style.mean()) / sd if sd > 1e-9 else np.zeros_like(style)
        bend_curv = bend if bend is not None else CALIBRATED_BEND_CURVATURE.get(venue, 1.0 / 120.0)
        p = win_probs(np.array(vc), np.array(wp), g["draw"].to_numpy().astype(float), dist,
                      track=geom, course=course, size=n_samples, bend_curvature=bend_curv,
                      speed_noise=speed_noise, pace_bias=pace_bias, pace_gain=pace_gain,
                      seed=abs(hash(race_id[0])) % (2**31))
        preds.append(g.with_columns(pl.Series("p", p), pl.Series("v_c_eb", vc),
                                    pl.Series("w_prime_eb", wp)))
    return pl.concat(preds)
