"""Full hierarchical-Bayes energetics fit via NumPyro (the rigorous Layer-1 sibling).

:mod:`hkjc.calibration` fits the energetics hierarchy in closed form (empirical Bayes:
the variance components are point-estimated, then plugged in). This module fits the
*same* model with NumPyro/NUTS — a proper hierarchical Bayesian regression that puts
priors on the hyperparameters and returns full **posteriors** (so the rating slope,
the between/within variance components and every horse effect carry calibrated
uncertainty, not just a point). It is the "fitting" half the user asked to put on a
real probabilistic-programming stack.

Model (per parameter, e.g. ``v_c``), fit on standardised ``y`` so generic priors work:

    y[i] = α[band[i]] + β·(rating[i] − r̄) + u[horse[i]] + ε[i]
    u[h] ~ Normal(0, τ)        ε[i] ~ Normal(0, σ)
    α[b] ~ Normal(μ_α, σ_α)    β ~ Normal(0, 1)   τ, σ ~ HalfNormal(1)

The horse random intercept ``u`` is partial pooling; the ``β·rating`` term **segments
horses by class** (the population mean is rating-conditioned, not one mono-pool). JAX
JITs the log-density, so ~3k runs / ~1k horses sample in seconds on CPU.

Heavy deps (``jax``, ``numpyro``) are imported lazily — install with
``uv sync --group bayes``; the rest of the package never needs them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .calibration import distance_band


@dataclass(frozen=True)
class HierarchicalFit:
    """Posterior summaries for one parameter's hierarchical fit (original units)."""
    param: str
    band_intercept: dict          # band -> posterior-mean intercept
    rating_slope: float           # posterior-mean class slope (per rating point)
    rating_slope_sd: float        # posterior sd of the slope
    tau: float                    # between-horse sd
    sigma: float                  # within-horse sd
    horse_effect: dict            # horse_id -> posterior-mean random effect
    horse_effect_sd: dict         # horse_id -> posterior sd
    rating_mean: float

    def estimate(self, horse_id, band, rating=None) -> float:
        a = self.band_intercept.get(band, float(np.mean(list(self.band_intercept.values()))))
        val = a + self.horse_effect.get(horse_id, 0.0)
        if rating is not None:
            val += self.rating_slope * (rating - self.rating_mean)
        return val


def _model(band_idx, rating_c, horse_idx, n_bands, n_horses, y=None):
    import numpyro
    import numpyro.distributions as dist

    mu_a = numpyro.sample("mu_a", dist.Normal(0.0, 2.0))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(2.0))
    with numpyro.plate("bands", n_bands):
        alpha = numpyro.sample("alpha", dist.Normal(mu_a, sigma_a))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    with numpyro.plate("horses", n_horses):
        u = numpyro.sample("u", dist.Normal(0.0, tau))
    mean = alpha[band_idx] + beta * rating_c + u[horse_idx]
    numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


def _fit_one(df: pl.DataFrame, value_col: str, rating_col: str | None,
             num_warmup: int, num_samples: int, seed: int) -> HierarchicalFit:
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    bands = sorted(df["_band"].unique().to_list())
    horses = sorted(df["horse_id"].unique().to_list())
    bmap = {b: i for i, b in enumerate(bands)}
    hmap = {h: i for i, h in enumerate(horses)}
    band_idx = jnp.array(df["_band"].replace_strict(bmap, return_dtype=pl.Int32).to_numpy())
    horse_idx = jnp.array(df["horse_id"].replace_strict(hmap, return_dtype=pl.Int32).to_numpy())

    y = df[value_col].to_numpy().astype(float)
    y_mean, y_sd = float(y.mean()), float(y.std() + 1e-9)
    y_std = jnp.array((y - y_mean) / y_sd)

    if rating_col and rating_col in df.columns:
        r = df[rating_col].to_numpy().astype(float)
        r_mean, r_sd = float(r.mean()), float(r.std() + 1e-9)
        rating_c = jnp.array((r - r_mean) / r_sd)        # unit-scale predictor (good priors)
    else:
        r_mean, r_sd = 0.0, 1.0
        rating_c = jnp.zeros(len(y))

    mcmc = MCMC(NUTS(_model), num_warmup=num_warmup, num_samples=num_samples,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), band_idx, rating_c, horse_idx,
             len(bands), len(horses), y=y_std)
    p = mcmc.get_samples()

    # de-standardise: y = y_mean + y_sd * y_std
    alpha = np.asarray(p["alpha"]).mean(0) * y_sd + y_mean
    band_intercept = {b: float(alpha[bmap[b]]) for b in bands}
    # y_std = α + β·(r−r̄)/r_sd + u  ⇒  raw slope dy/dr = β·y_sd/r_sd
    slope_raw = np.asarray(p["beta"]) * (y_sd / r_sd)
    u = np.asarray(p["u"]) * y_sd
    return HierarchicalFit(
        param=value_col,
        band_intercept=band_intercept,
        rating_slope=float(slope_raw.mean()),
        rating_slope_sd=float(slope_raw.std()),
        tau=float(np.asarray(p["tau"]).mean() * y_sd),
        sigma=float(np.asarray(p["sigma"]).mean() * y_sd),
        horse_effect={h: float(u[:, hmap[h]].mean()) for h in horses},
        horse_effect_sd={h: float(u[:, hmap[h]].std()) for h in horses},
        rating_mean=r_mean,
    )


def fit_energetics_bayes(runs: pl.DataFrame, rating_col: str | None = "rtg",
                         num_warmup: int = 500, num_samples: int = 500,
                         seed: int = 0) -> dict:
    """Hierarchical-Bayes fit of ``v_c`` and ``w_prime`` (NumPyro/NUTS).

    Returns ``{"v_c": HierarchicalFit, "w_prime": HierarchicalFit}``. The fixed effect
    is rating-conditioned when ``rating_col`` is present (segments horses by class).
    """
    d = runs.with_columns(
        pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("_band"))
    rc = rating_col if (rating_col and rating_col in d.columns) else None
    return {
        "v_c": _fit_one(d, "v_c", rc, num_warmup, num_samples, seed),
        "w_prime": _fit_one(d, "w_prime", rc, num_warmup, num_samples, seed + 1),
    }


def horse_table_bayes(runs: pl.DataFrame, fits: dict, rating_col: str | None = "rtg") -> pl.DataFrame:
    """Per-run posterior-mean ``(v_c_bayes, w_prime_bayes)`` from the hierarchical fits."""
    d = runs.with_columns(
        pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("_band"))
    rt = d[rating_col].to_numpy().astype(float) if rating_col and rating_col in d.columns else None
    out = {}
    for col, fit in (("v_c", fits["v_c"]), ("w_prime", fits["w_prime"])):
        hids, bands = d["horse_id"].to_list(), d["_band"].to_list()
        out[f"{col}_bayes"] = np.array([
            fit.estimate(h, b, rt[i] if rt is not None else None)
            for i, (h, b) in enumerate(zip(hids, bands))])
    return d.with_columns([pl.Series(k, v) for k, v in out.items()]).drop("_band")
