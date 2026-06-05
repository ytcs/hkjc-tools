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
from . import dynamics
from .constants import (ALLOMETRIC_B, ALLOMETRIC_B_BY_VENUE, DRAW_FS_GAIN,
                        DRAW_FS_GAIN_BY_VENUE, DRAW_GAIN, DRAW_GAIN_AWT,
                        DRAW_GAIN_BY_VENUE, FORM_GAIN, FORM_GAIN_BY_VENUE, JOCKEY_GAIN,
                        N_SAMPLES, PACE_GAIN, POS_FADE_GAIN, SHRINK_MULT, SIM,
                        SPEED_NOISE, W_PRIME_GAIN)
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
    form_gain: float | None = None,
    jockey_gain: float = JOCKEY_GAIN,
    w_prime_gain: float = W_PRIME_GAIN,
    trainer_gain: float = 0.0,
    speedfig_gain: float = 0.0,
    pos_fade_gain: float = POS_FADE_GAIN,
    shrink_mult: float = SHRINK_MULT,
    draw_gain: float | None = None,
    draw_fs_gain: float | None = None,
    speed_noise: float = SPEED_NOISE,
    sigma_t: float = 2.5,
    diff_coef: float = 170.0,
    awt_diff_mult: float = 2.0,
    pace_gain: float = PACE_GAIN,
    bend: float | None = None,
    sim: str = SIM,
    allometric_b: float | None = None,
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
        # form coupling is PER-VENUE (HV wants a lower gain than ST); an explicit form_gain
        # arg overrides (for sweeps), else the venue's calibrated value, else the ST fallback.
        fg = form_gain if form_gain is not None else FORM_GAIN_BY_VENUE.get(venue, FORM_GAIN)
        rtg = g["rtg"].to_numpy().astype(float)
        form = g["form"].to_numpy().astype(float)
        jky = g["jky_sr"].to_numpy().astype(float)
        # speed figure is already a velocity (m/s vs condition par) -> kinematic coefficient ≈1
        sfig = g["speedfig"].to_numpy().astype(float) if "speedfig" in g.columns else np.zeros(g.height)
        # trainer strike rate, within-race centred (a constant field shift cancels in win prob)
        trn = g["trn_sr"].to_numpy().astype(float) if "trn_sr" in g.columns else np.zeros(g.height)
        trn = trn - np.nanmean(trn)
        # positional fade: habitual early position, within-race centred (front-runners over-rated by
        # the non-interacting physics; pos_fade_gain is NEGATIVE so front -> lower v_c, closer -> higher)
        epos = g["early_pos"].to_numpy().astype(float) if "early_pos" in g.columns else np.full(g.height, 0.764)
        epos = epos - np.nanmean(epos)
        band = distance_band(dist)
        vc, wp = [], []
        for i, h in enumerate(g["horse_id"].to_list()):
            n, u_vc, u_w = res.get(h, (0, 0.0, 0.0))
            vc.append(pop.v_c.fixed(band, rtg[i]) + shrink_mult * lam_vc(n) * u_vc
                      + fg * (form[i] - 0.5) + jockey_gain * (jky[i] - jky_mean)
                      + speedfig_gain * sfig[i] + trainer_gain * trn[i]
                      + pos_fade_gain * epos[i])
            # w_prime_gain amplifies the per-horse reserve spread (the EB residual) — the
            # stamina analogue of form_gain (market under-weights W' at gain 1.0).
            wp.append(pop.w_prime.fixed(band, rtg[i]) + w_prime_gain * lam_w(n) * u_w)
        vc = np.array(vc)
        geom = track_for(venue, dist, _surface(dist))
        course = g["course"][0] if "course" in g.columns else "A"
        draw = g["draw"].to_numpy().astype(float)
        # bend ground-loss gain is PER-VENUE and GEOMETRIC: HV (bend-dominated) wants ~2.7x ST. An
        # explicit draw_gain arg overrides (sweeps), else the venue value. On a no-bend course (the ST
        # straight 1000m chute) there is no arc to lose, so zero it — the residual draw effect there is a
        # reverse watering/going gradient, not ground-loss, and applying the penalty has the wrong sign.
        if draw_gain is not None:
            draw_g = draw_gain
        elif _surface(dist) == "awt":
            draw_g = DRAW_GAIN_AWT          # dirt has a stronger rail bias than turf (surface overrides venue)
        else:
            draw_g = DRAW_GAIN_BY_VENUE.get(venue, DRAW_GAIN)
        if geom.bend_fraction(dist) < 0.05:
            draw_g = 0.0
        bend_curv = bend if bend is not None else CALIBRATED_BEND_CURVATURE.get(venue, 1.0 / 120.0)
        seed = abs(hash(race_id[0])) % (2**31)
        mass = g["bodywt"].to_numpy().astype(float) if "bodywt" in g.columns else np.full(g.height, 500.0)
        mass = np.where(np.isnan(mass), np.nanmean(mass), mass)
        # ALLOMETRIC mass coupling (first-principles): steady-state v_c ∝ m^(b−1) — the analytic solution
        # of v_c = p_c/(K_LOCO·m·g) with allometric aerobic power p_c = κ·m^b. b>1 ⇒ heavier horses sustain
        # more speed (power scales super-linearly with muscle mass), b=1 = isometric (off). Field-relative
        # (centred on the race's geometric-mean mass). Applies to the statistical sims (pl/wald/kinematic);
        # the eom path instead lets mass act through the full Newton's-law dynamics, so skip it there.
        # PER-VENUE: an explicit allometric_b arg overrides (sweeps), else the venue value (mass converts
        # only at ST). The eom path uses the global ALLOMETRIC_B for its full dynamics.
        ab = allometric_b if allometric_b is not None else ALLOMETRIC_B_BY_VENUE.get(venue, ALLOMETRIC_B)
        if sim != "eom" and ab != 1.0:
            ref = float(np.exp(np.nanmean(np.log(mass))))
            vc = vc * (mass / ref) ** (ab - 1.0)
        if sim == "eom":
            # power-balance EOM: declared body mass enters every term (locomotion, inertia,
            # reserve energy, drag); aerobic power p_c = κ·m^b via the one allometric constant.
            # (Recorded future path — currently underperforms the kinematic sim; see journal.)
            p = dynamics.win_probs(vc, np.array(wp), mass, draw, dist,
                                   allometric_b=(allometric_b if allometric_b is not None else ALLOMETRIC_B),
                                   track=geom, course=course, size=n_samples,
                                   bend_curvature=bend_curv, speed_noise=speed_noise, seed=seed)
        elif sim == "pl":
            # FAST PROBE (closed form, no sampling). Gumbel finishing-time noise -> Plackett-Luce /
            # multinomial logit: P(i wins) = softmax(-mu_i/sigma_t), mu_i = D/v_target_i. Also gives the
            # full finishing-order distribution for place/exotics. Not the physical model (the diffusion
            # law is — see "wald"); kept as the instant probe for wide parameter sweeps.
            l_eff = dist * (1.0 + draw_g * (draw - 1.0))
            mu = l_eff / (vc * (1.0 + np.array(wp) / dist))
            s = np.exp(-(mu - mu.min()) / sigma_t)
            p = s / s.sum()
        elif sim == "wald":
            # DIFFUSION LAW (the correct standard theory): finishing time = first-passage time of a
            # drift-diffusion dx = v·dt + √(2D)·dW, which is the INVERSE GAUSSIAN (Wald) distribution —
            # strictly positive (time can't be negative), with Var(T)=2D·L/v³ (CV ∝ 1/√L) baked in.
            # One physical parameter: the diffusion coefficient D (m²/s); IG shape λ = L²/(2D).
            # AWT is a more diffusive surface (kickback, less reliable turf-trained signal) => larger D.
            d_eff = diff_coef * (awt_diff_mult if _surface(dist) == "awt" else 1.0)
            # bend ground-loss: a horse drawn k lanes off the rail runs a longer arc round the
            # turns => larger effective distance => longer first-passage time. Geometry fixes the
            # sign (outside = slower); the magnitude is calibrated and absorbs jockeys steering in.
            # draw_fs_gain modulates the penalty by field size: crowding (more rivals to be boxed by)
            # amplifies the draw effect — the mean-field shadow of the collision physics (horses are
            # not collisionless), centred at a 12-runner field. PER-VENUE (HV-only; converts only where
            # boxed-in physics dominates) unless passed explicitly (for sweeps). The max(0,·) clamp keeps
            # the penalty from REVERSING into a wide-draw bonus in small fields (unphysical).
            dfs_g = draw_fs_gain if draw_fs_gain is not None else DRAW_FS_GAIN_BY_VENUE.get(venue, DRAW_FS_GAIN)
            dg = draw_g * max(0.0, 1.0 + dfs_g * (g.height - 12.0))
            l_eff = dist * (1.0 + dg * (draw - 1.0))
            mu = l_eff / (vc * (1.0 + np.array(wp) / dist))
            lam = dist * dist / (2.0 * d_eff)
            rng = np.random.default_rng(seed)
            t = rng.wald(np.broadcast_to(mu, (n_samples, g.height)), lam)
            counts = np.bincount(t.argmin(axis=1), minlength=g.height).astype(float)
            p = (counts + 1.0) / (n_samples + g.height)
        else:
            # (mass already applied above via the allometric law for this path)
            # within-race style z-score → pace policy (front-runners take position, closers wait)
            style = g["style"].to_numpy().astype(float)
            sd = style.std()
            pace_bias = (style - style.mean()) / sd if sd > 1e-9 else np.zeros_like(style)
            p = win_probs(vc, np.array(wp), draw, dist,
                          track=geom, course=course, size=n_samples, bend_curvature=bend_curv,
                          speed_noise=speed_noise, pace_bias=pace_bias, pace_gain=pace_gain,
                          seed=seed)
        preds.append(g.with_columns(pl.Series("p", p), pl.Series("v_c_eb", vc),
                                    pl.Series("w_prime_eb", wp)))
    return pl.concat(preds)
