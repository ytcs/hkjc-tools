"""Calibrate the production win model's ability population — empirical & hierarchical Bayes.

**Horse-level energetics (hierarchical / empirical Bayes).** Per-run
``(v_c, W')`` recovered from sectionals are *noisy* (an easy winner eases down → its
``v_c`` under-reads). We model each parameter with a one-way Normal–Normal
random-effects structure: a per-run observation is a horse effect plus within-horse
noise, and the horse effects are drawn from a population — with a **distance-band
fixed effect**, because ``v_c`` scales with trip. The variance components
``(σ²_within, τ²_between)`` are estimated from the data (empirical Bayes, ANOVA
method-of-moments), giving each horse a **partially-pooled** estimate: a horse with
one run shrinks almost fully to the population mean for its trip; an established
horse trusts its own runs. The *horse-independent* parameters are the band means and
variance components; the *horse-specific* parameters are the shrunken random
effects. Leave-one-out estimates (``loo=True``) drop a run from its own horse's
pool, so a run's energetics never leak into a fit that predicts that run.

Everything here is pure (numpy/polars) and offline. The per-venue draw bias is now a
calibrated mean-field term on the production ``wald`` path (``DRAW_GAIN_BY_VENUE``), not a
simulated bend_curvature — the interacting traffic sim that the old bend-curvature grid
calibrated was deprecated in 2026-06 (see ``hkjc.model.fit`` and docs/THEORY.md Appendix).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


# --------------------------------------------------------------- distance bands
def distance_band(d: float) -> str:
    """Coarse trip class for the ``v_c`` fixed effect (``v_c`` falls with distance)."""
    d = float(d)
    if d <= 1200:
        return "sprint"
    if d <= 1650:
        return "mile"
    if d <= 2000:
        return "middle"
    return "staying"


# ------------------------------------------------------ one-way random effects EB
@dataclass(frozen=True)
class RandomEffects:
    """Empirical-Bayes one-way random-effects fit of one parameter.

    The population (fixed-effect) mean is **rating-conditioned**, not a single
    mono-population: ``fixed = band_mean[band] + rating_slope·(rating − rating_mean)``,
    so a horse's prior centres on the expected value for *its class and trip*, not the
    global mean. ``sigma2`` is the within-horse variance and ``tau2`` the between-horse
    variance of the fixed-effect residual. The shrinkage weight for a horse with ``n``
    runs and residual mean ``u`` is ``λ = τ²/(τ² + σ²/n)`` and its estimate is
    ``fixed + λ·u`` — full pooling to the rating-predicted mean at ``n→0``, its own
    runs at large ``n``.
    """
    band_mean: dict
    grand_mean: float
    sigma2: float
    tau2: float
    rating_slope: float = 0.0
    rating_mean: float = 0.0

    def fixed(self, band: str, rating: float | None = None) -> float:
        bm = self.band_mean.get(band, self.grand_mean)
        if rating is None or self.rating_slope == 0.0:
            return bm
        return bm + self.rating_slope * (rating - self.rating_mean)

    def shrink(self, band: str, n: float, resid_mean: float, rating: float | None = None) -> float:
        f = self.fixed(band, rating)
        if n <= 0:
            return f
        lam = self.tau2 / (self.tau2 + self.sigma2 / n)
        return f + lam * resid_mean

    def posterior_sd(self, n: float) -> float:
        if n <= 0:
            return float(np.sqrt(self.tau2 + self.sigma2))
        return float(np.sqrt(1.0 / (n / self.sigma2 + 1.0 / self.tau2)))


def fit_random_effects(df: pl.DataFrame, value_col: str, group_col: str,
                       band_col: str, rating_col: str | None = None) -> RandomEffects:
    """Fit a Normal–Normal random-effects model with a (rating-conditioned) fixed effect.

    The fixed effect is a per-band intercept plus, if ``rating_col`` is given, a single
    global rating slope (estimated from the within-band, rating-vs-value relationship —
    so horses are *segmented by class*). ``σ²`` is the pooled within-group variance
    (groups with ≥2 obs); ``τ²`` the ANOVA method-of-moments between-group component.
    """
    cols = [value_col, group_col, band_col] + ([rating_col] if rating_col else [])
    d = df.select(cols).drop_nulls()
    band = d.group_by(band_col).agg(pl.col(value_col).mean().alias("_bm"))
    band_mean = dict(zip(band[band_col].to_list(), band["_bm"].to_list()))
    d = d.join(band, on=band_col).with_columns((pl.col(value_col) - pl.col("_bm")).alias("_r"))

    rating_slope, rating_mean = 0.0, 0.0
    if rating_col:
        rt = d[rating_col].to_numpy().astype(float)
        rating_mean = float(rt.mean())
        rc = rt - rating_mean
        resid = d["_r"].to_numpy()
        denom = float((rc * rc).sum())
        rating_slope = float((rc * resid).sum() / denom) if denom > 0 else 0.0
        d = d.with_columns((pl.col("_r") - rating_slope * (pl.Series(rc))).alias("_u"))
    else:
        d = d.with_columns(pl.col("_r").alias("_u"))

    g = d.group_by(group_col).agg(
        pl.col("_u").mean().alias("_um"),
        pl.len().alias("_n"),
        pl.col("_u").var(ddof=1).alias("_v"),     # null when n == 1
    )
    n = g["_n"].to_numpy().astype(float)
    um = g["_um"].to_numpy()
    v = g["_v"].fill_null(0.0).to_numpy()
    multi = n >= 2
    sigma2 = float((v[multi] * (n[multi] - 1)).sum() / max((n[multi] - 1).sum(), 1.0))
    sigma2 = max(sigma2, 1e-9)
    # ANOVA MoM for tau2
    h, total = len(n), float(n.sum())
    ssb = float((n * (um - 0.0) ** 2).sum())          # residuals are fixed-effect-centred
    msb = ssb / max(h - 1, 1)
    n0 = (total - float((n ** 2).sum()) / total) / max(h - 1, 1)
    tau2 = max((msb - sigma2) / max(n0, 1e-9), 1e-6)
    grand = float(np.mean(list(band_mean.values())))
    return RandomEffects(band_mean=band_mean, grand_mean=grand, sigma2=sigma2, tau2=tau2,
                         rating_slope=rating_slope, rating_mean=rating_mean)


@dataclass(frozen=True)
class EnergeticsPopulation:
    v_c: RandomEffects
    w_prime: RandomEffects


def fit_energetics_population(runs: pl.DataFrame, group_col: str = "horse_id",
                             distance_col: str = "distance",
                             rating_col: str | None = None) -> EnergeticsPopulation:
    """Empirical-Bayes population fit for ``v_c`` and ``w_prime`` from per-run rows.

    If ``rating_col`` is present, the population is **segmented by rating** (a class
    slope in the fixed effect) rather than a single mono-population.
    """
    d = runs.with_columns(
        pl.col(distance_col).map_elements(distance_band, return_dtype=pl.Utf8).alias("_band")
    )
    rc = rating_col if (rating_col and rating_col in d.columns) else None
    return EnergeticsPopulation(
        v_c=fit_random_effects(d, "v_c", group_col, "_band", rating_col=rc),
        w_prime=fit_random_effects(d, "w_prime", group_col, "_band", rating_col=rc),
    )


def horse_energetics(runs: pl.DataFrame, pop: EnergeticsPopulation,
                     group_col: str = "horse_id", distance_col: str = "distance",
                     rating_col: str | None = None, loo: bool = True) -> pl.DataFrame:
    """Per-run partially-pooled ``(v_c_eb, w_prime_eb)`` estimates.

    With ``loo=True`` each run's estimate excludes that run from its horse's pool
    (leave-one-out), so the energetics fed to a race model never contain that race's
    own outcome — the leakage-safe quantity for calibration and back-testing. The
    fixed effect is rating-conditioned when ``rating_col`` is supplied.
    """
    d = runs.with_columns(
        pl.col(distance_col).map_elements(distance_band, return_dtype=pl.Utf8).alias("_band"))
    rt = (d[rating_col].to_numpy().astype(float)
          if rating_col and rating_col in d.columns else None)
    out = {}
    for col, re in (("v_c", pop.v_c), ("w_prime", pop.w_prime)):
        bands = d["_band"].to_list()
        fixed = np.array([re.fixed(b, rt[i] if rt is not None else None)
                          for i, b in enumerate(bands)])
        u = d[col].to_numpy() - fixed
        groups = d[group_col].to_numpy()
        # per-group sum and count (vectorised leave-one-out)
        gidx = {g: i for i, g in enumerate(np.unique(groups))}
        gi = np.array([gidx[g] for g in groups])
        gsum = np.bincount(gi, weights=u)
        gcnt = np.bincount(gi).astype(float)
        if loo:
            n_eff = gcnt[gi] - 1.0
            u_eff = np.where(n_eff > 0, (gsum[gi] - u) / np.where(n_eff > 0, n_eff, 1.0), 0.0)
        else:
            n_eff = gcnt[gi]
            u_eff = gsum[gi] / n_eff
        lam = np.where(n_eff > 0, re.tau2 / (re.tau2 + re.sigma2 / np.maximum(n_eff, 1e-9)), 0.0)
        out[f"{col}_eb"] = fixed + lam * u_eff
    return d.with_columns([pl.Series(k, v) for k, v in out.items()]).drop("_band")


def horse_energetics_trailing(runs: pl.DataFrame, pop: EnergeticsPopulation,
                              date_col: str = "date", group_col: str = "horse_id",
                              distance_col: str = "distance",
                              rating_col: str | None = None) -> pl.DataFrame:
    """Per-run ``(v_c_eb, w_prime_eb)`` from each horse's **strictly-prior** runs only.

    The forecasting-safe sibling of :func:`horse_energetics`: a run's estimate uses an
    expanding window over that horse's *earlier* runs (ordered by ``date_col``), so it
    contains no information from the run itself or any later run — the quantity a
    walk-forward backtest must use. A run with no prior history falls back to the
    (rating-conditioned) population mean. The population ``pop`` should itself be fit on
    a training window to keep the whole pipeline leak-free.
    """
    d = (runs.with_columns(
            pl.col(distance_col).map_elements(distance_band, return_dtype=pl.Utf8).alias("_band"))
         .sort([group_col, date_col]))
    rt = (d[rating_col].to_numpy().astype(float)
          if rating_col and rating_col in d.columns else None)
    out = {}
    for col, re in (("v_c", pop.v_c), ("w_prime", pop.w_prime)):
        bands = d["_band"].to_list()
        fixed = np.array([re.fixed(b, rt[i] if rt is not None else None)
                          for i, b in enumerate(bands)])
        u = d[col].to_numpy() - fixed
        dd = d.with_columns(pl.Series("_u", u))
        prior_sum = dd.select(pl.col("_u").cum_sum().shift(1).over(group_col))["_u"].fill_null(0.0).to_numpy()
        prior_n = dd.select((pl.col("_u").cum_count().over(group_col) - 1))["_u"].to_numpy().astype(float)
        mean_prior = np.where(prior_n > 0, prior_sum / np.maximum(prior_n, 1.0), 0.0)
        lam = np.where(prior_n > 0, re.tau2 / (re.tau2 + re.sigma2 / np.maximum(prior_n, 1e-9)), 0.0)
        out[f"{col}_eb"] = fixed + lam * mean_prior
        out[f"{col}_prior_n"] = prior_n
    return d.with_columns([pl.Series(k, v) for k, v in out.items()]).drop("_band")


# ------------------------------------------------------- empirical draw-bias curve
def empirical_draw_curve(df: pl.DataFrame, draw_col: str = "draw",
                         won_col: str = "won", finish_col: str = "finish_pos",
                         field_col: str = "field_size") -> pl.DataFrame:
    """Win rate, top-3 rate and mean finishing position by draw (the calibration target).

    Draw is randomly assigned, so deviations by draw are the *pure* positional effect
    (ability averages out). ``rel_finish`` re-expresses finishing position on a 0–1
    scale within each race so draws are comparable across field sizes.
    """
    d = df.with_columns(
        ((pl.col(finish_col) - 1) / (pl.col(field_col) - 1)).alias("_relfin"))
    return (d.group_by(draw_col)
            .agg(pl.len().alias("n"),
                 pl.col(won_col).mean().alias("p_win"),
                 (pl.col(finish_col) <= 3).mean().alias("p_top3"),
                 pl.col(finish_col).mean().alias("mean_finish"),
                 pl.col("_relfin").mean().alias("rel_finish"))
            .sort(draw_col))
