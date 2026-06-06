"""Fit + predict for the production win model — leak-free per-horse ability → win probabilities.

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
            + ability channels (trial fusion, collateral-form, peakgap, trainer-hot)
            × (m/m̄)^(b−1)                         # allometric mass (wald path)

and the anaerobic reserve W′ is the rating-conditioned mean + shrunken residual + the late-race
reserve couplings (fade-resistance, positional-fade — THEORY §3d T1).
These feed the ``wald`` first-passage diffusion law (with per-venue bend ground-loss +
crowding draw) for win probs — see the ``sim`` branches. The interacting ABM sims are
deprecated (physics preserved in docs/THEORY.md Appendix).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..core.hkjc_facts import AWT_DISTANCE
from ..core.track import track_for
from .calibration import (EnergeticsPopulation, distance_band,
                          fit_energetics_population)
from .constants import (ALLOMETRIC_B, ALLOMETRIC_B_BY_VENUE, COLLAT_GAIN,
                        COLLAT_GAIN_BY_VENUE, DRAW_FS_GAIN,
                        DRAW_FS_GAIN_BY_VENUE, DRAW_GAIN, DRAW_GAIN_AWT,
                        DRAW_GAIN_BY_VENUE, DRAW_VAR_GAIN, DRAW_VAR_GAIN_BY_VENUE,
                        FADE_GAIN, FADE_W_GAIN, FORM_GAIN, FORM_GAIN_BY_VENUE,
                        JOCKEY_GAIN, N_SAMPLES, PEAKGAP_GAIN, PEAKGAP_GAIN_BY_VENUE,
                        POS_FADE_GAIN, POSFADE_W_GAIN, SHRINK_MULT, SIM, SUSTAIN_GAIN, TRIAL_GAIN,
                        TRIAL_GAIN_BY_VENUE, TRIAL_PI_RATIO, TRIAL_PI_RATIO_BY_VENUE,
                        TRN_HOT_GAIN, TRN_HOT_GAIN_BY_VENUE, W_PRIME_GAIN)


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


def fit_trial_vc_map(train: pl.DataFrame, pop: EnergeticsPopulation,
                     signal: str = "trial_fig") -> tuple[float, float]:
    """Train-only affine map putting a trial signal onto the race v_c **residual** scale.

    Regresses the race v_c residual (own ``v_c`` minus the rating-band fixed effect) on the trial
    signal (``trial_fig`` speed-figure, or race-scale ``trial_vc``) over train rows with a prior
    trial, giving ``(a, b)`` with ``E[race_resid | trial] = a + b·trial``. Leak-free (train only).
    Too few trials returns ``(0.0, 0.0)`` so the fusion is a no-op.
    """
    col = signal if signal in train.columns else None
    if col is None:
        return 0.0, 0.0
    d = train.filter(pl.col(col) != 0.0) if signal == "trial_fig" else \
        train.filter(pl.col(col).is_not_null() & (pl.col(col) > 0))
    if d.height < 50:
        return 0.0, 0.0
    bands = d["distance"].map_elements(distance_band, return_dtype=pl.Utf8).to_list()
    rtg = d["rtg"].to_numpy().astype(float) if "rtg" in d.columns else None
    fixed = np.array([pop.v_c.fixed(b, rtg[i] if rtg is not None else None)
                      for i, b in enumerate(bands)])
    x = d[col].to_numpy().astype(float)
    if signal != "trial_fig":          # critical-velocity signal is on the race v_c scale -> residual
        x = x - fixed
    y = d["v_c"].to_numpy().astype(float) - fixed
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50:
        return 0.0, 0.0
    b, a = np.polyfit(x[mask], y[mask], 1)
    return float(a), float(b)


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
    trial_gain: float | None = None,
    trial_pi_ratio: float | None = None,
    collat_gain: float | None = None,
    sustain_gain: float = SUSTAIN_GAIN,
    fade_gain: float = FADE_GAIN,
    rtg_peakgap_gain: float | None = None,
    trn_hot_gain: float | None = None,
    pos_fade_gain: float = POS_FADE_GAIN,
    fade_w_gain: float = FADE_W_GAIN,
    posfade_w_gain: float = POSFADE_W_GAIN,
    shrink_mult: float = SHRINK_MULT,
    draw_gain: float | None = None,
    draw_fs_gain: float | None = None,
    draw_var_gain: float | None = None,
    sigma_t: float = 2.5,
    diff_coef: float = 170.0,
    awt_diff_mult: float = 2.0,
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
    # Coherent trial fusion: v_c residual posterior precisions — population prior (π_0=1/τ²), n race
    # observations (π_race=1/σ²), m trial observations (π_trial=ρ·π_race). ρ set per-race below.
    pi0 = 1.0 / pop.v_c.tau2
    pi_race = 1.0 / pop.v_c.sigma2
    rho = trial_pi_ratio  # set per-race below from the venue value when None; 0 ⇒ fusion is a no-op
    preds = []
    for race_id, g in test.group_by("race_id"):
        g = g.sort("runner")
        dist = int(g["distance"][0])
        venue = g["venue"][0]
        # form coupling is PER-VENUE (HV wants a lower gain than ST); an explicit form_gain
        # arg overrides (for sweeps), else the venue's calibrated value, else the ST fallback.
        fg = form_gain if form_gain is not None else FORM_GAIN_BY_VENUE.get(venue, FORM_GAIN)
        # barrier-trial debut signal is PER-VENUE (HV benefits more); explicit arg overrides (sweeps)
        tg_v = trial_gain if trial_gain is not None else TRIAL_GAIN_BY_VENUE.get(venue, TRIAL_GAIN)
        # coherent trial-fusion precision ratio ρ (π_trial/π_race), PER-VENUE; explicit arg overrides (sweeps)
        rho_v = rho if rho is not None else TRIAL_PI_RATIO_BY_VENUE.get(venue, TRIAL_PI_RATIO)
        # collateral-form (opponent-adjusted) coupling is PER-VENUE (HV wants ~2.5x ST); arg overrides
        cg = collat_gain if collat_gain is not None else COLLAT_GAIN_BY_VENUE.get(venue, COLLAT_GAIN)
        rtg = g["rtg"].to_numpy().astype(float)
        form = g["form"].to_numpy().astype(float)
        jky = g["jky_sr"].to_numpy().astype(float)
        # speed figure is already a velocity (m/s vs condition par) -> kinematic coefficient ≈1
        sfig = g["speedfig"].to_numpy().astype(float) if "speedfig" in g.columns else np.zeros(g.height)
        # barrier-trial speed figure (m/s vs trial par) — the only prior on-track form for debutants
        tfig = g["trial_fig"].to_numpy().astype(float) if "trial_fig" in g.columns else np.zeros(g.height)
        # prior trial COUNT m (number of trial pseudo-observations) for the coherent precision-fusion;
        # the fused trial SIGNAL is the speed-figure tfig above, the count comes from trial_vc_table.
        tm = g["trial_m"].to_numpy().astype(float) if "trial_m" in g.columns else np.zeros(g.height)
        # collateral-form opponent-adjusted rating (m/s), within-race centred (field-mean shift cancels)
        cfm = g["collat_form"].to_numpy().astype(float) if "collat_form" in g.columns else np.zeros(g.height)
        cfm = cfm - np.nanmean(cfm)
        # trajectory-shape z-traits (sustained-speed, fade-resistance), within-race centred
        sustain = g["traj_sustain"].to_numpy().astype(float) if "traj_sustain" in g.columns else np.zeros(g.height)
        sustain = sustain - np.nanmean(sustain)
        fade = g["traj_fade"].to_numpy().astype(float) if "traj_fade" in g.columns else np.zeros(g.height)
        fade = fade - np.nanmean(fade)
        # rating-trajectory PROCESS feature: own-peak deficit (≤0), within-race centred. PER-VENUE.
        pgap = g["rtg_peakgap"].to_numpy().astype(float) if "rtg_peakgap" in g.columns else np.zeros(g.height)
        pgap = pgap - np.nanmean(pgap)
        pgg = rtg_peakgap_gain if rtg_peakgap_gain is not None else PEAKGAP_GAIN_BY_VENUE.get(venue, PEAKGAP_GAIN)
        # trainer hot-streak (recency-weighted SR) — intent/stable-confidence overlay, within-race centred
        thot = g["trn_hot"].to_numpy().astype(float) if "trn_hot" in g.columns else np.zeros(g.height)
        thot = thot - np.nanmean(thot)
        thg = trn_hot_gain if trn_hot_gain is not None else TRN_HOT_GAIN_BY_VENUE.get(venue, TRN_HOT_GAIN)
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
            # Coherent trial fusion: v_c-residual term carrying m trial observations at precision
            # π_trial=ρ·π_race; weight w_trial decays as race evidence n grows. No-op at ρ=0 or m=0.
            # When ρ>0 the additive tg_v term is OFF (alternative trial structures).
            trial_term = 0.0
            if rho_v > 0.0 and tm[i] > 0.0 and tfig[i] != 0.0:
                # w_trial = m·π_trial/(π_0 + n·π_race + m·π_trial), normalised to 1 at debut (n=0, m=1)
                # so the fusion equals the additive for a debutant and fades smoothly as the horse races.
                m_pi = tm[i] * rho_v * pi_race
                w_trial = m_pi / (pi0 + n * pi_race + m_pi)
                w0 = (rho_v * pi_race) / (pi0 + rho_v * pi_race)   # w_trial at n=0, m=1
                w_norm = w_trial / w0 if w0 > 0 else 0.0
                trial_term = w_norm * tg_v * tfig[i]
            tg_eff = 0.0 if rho_v > 0.0 else tg_v
            vc.append(pop.v_c.fixed(band, rtg[i]) + shrink_mult * lam_vc(n) * u_vc + trial_term
                      + fg * (form[i] - 0.5) + jockey_gain * (jky[i] - jky_mean)
                      + speedfig_gain * sfig[i] + trainer_gain * trn[i]
                      + tg_eff * tfig[i] + cg * cfm[i] + pos_fade_gain * epos[i]
                      + sustain_gain * sustain[i] + fade_gain * fade[i] + pgg * pgap[i]
                      + thg * thot[i])
            # w_prime_gain amplifies the per-horse reserve spread (the EB residual), stamina analogue
            # of form_gain. Late-race reserve couplings on W′ (THEORY §3d): fade_w_gain>0 (holds speed
            # late => more reserve), posfade_w_gain<0 (front-runner burns W′ setting the pace).
            wp.append(pop.w_prime.fixed(band, rtg[i]) + w_prime_gain * lam_w(n) * u_w
                      + fade_w_gain * fade[i] + posfade_w_gain * epos[i])
        vc = np.array(vc)
        geom = track_for(venue, dist, _surface(dist))
        draw = g["draw"].to_numpy().astype(float)
        # bend ground-loss gain is PER-VENUE and GEOMETRIC; explicit draw_gain arg overrides (sweeps).
        # On a no-bend course (ST straight 1000m) there is no arc to lose, so it is zeroed below.
        if draw_gain is not None:
            draw_g = draw_gain
        elif _surface(dist) == "awt":
            draw_g = DRAW_GAIN_AWT          # dirt rail bias stronger than turf (surface overrides venue)
        else:
            draw_g = DRAW_GAIN_BY_VENUE.get(venue, DRAW_GAIN)
        if geom.bend_fraction(dist) < 0.05:
            draw_g = 0.0
        seed = abs(hash(race_id[0])) % (2**31)
        mass = g["bodywt"].to_numpy().astype(float) if "bodywt" in g.columns else np.full(g.height, 500.0)
        mass = np.where(np.isnan(mass), np.nanmean(mass), mass)
        # ALLOMETRIC mass coupling: steady-state v_c ∝ m^(b−1) (from v_c = p_c/(K_LOCO·m·g),
        # aerobic power p_c = κ·m^b). b>1 ⇒ heavier horses sustain more speed; b=1 = off.
        # Field-relative (geometric-mean mass). PER-VENUE; explicit allometric_b arg overrides (sweeps).
        ab = allometric_b if allometric_b is not None else ALLOMETRIC_B_BY_VENUE.get(venue, ALLOMETRIC_B)
        if ab != 1.0:
            ref = float(np.exp(np.nanmean(np.log(mass))))
            vc = vc * (mass / ref) ** (ab - 1.0)
        if sim == "pl":
            # FAST PROBE (closed form, no sampling). Gumbel finishing-time noise -> Plackett-Luce /
            # multinomial logit: P(i wins) = softmax(-mu_i/sigma_t), mu_i = D/v_target_i. Also gives the
            # full finishing-order distribution for place/exotics. Not the physical model (the diffusion
            # law is — see "wald"); kept as the instant probe for wide parameter sweeps.
            l_eff = dist * (1.0 + draw_g * (draw - 1.0))
            mu = l_eff / (vc * (1.0 + np.array(wp) / dist))
            s = np.exp(-(mu - mu.min()) / sigma_t)
            p = s / s.sum()
        else:
            # DIFFUSION LAW (the production win model): finishing time = first-passage of drift-diffusion
            # dx = v·dt + √(2D)·dW → INVERSE GAUSSIAN (Wald), strictly positive, Var(T)=2D·L/v³, shape
            # λ = L²/(2D). One parameter D (m²/s); AWT is more diffusive => larger D.
            d_eff = diff_coef * (awt_diff_mult if _surface(dist) == "awt" else 1.0)
            # bend ground-loss: a wider draw runs a longer arc => longer first-passage time (outside =
            # slower). draw_fs_gain amplifies the penalty by field-size crowding (centred at 12 runners);
            # PER-VENUE unless passed explicitly. max(0,·) clamps the penalty from reversing in small fields.
            dfs_g = draw_fs_gain if draw_fs_gain is not None else DRAW_FS_GAIN_BY_VENUE.get(venue, DRAW_FS_GAIN)
            dg = draw_g * max(0.0, 1.0 + dfs_g * (g.height - 12.0))
            l_eff = dist * (1.0 + dg * (draw - 1.0))
            mu = l_eff / (vc * (1.0 + np.array(wp) / dist))
            # draw → DIFFUSION D (trip variance), not drift: D_i = d_eff·(1 + dvg·risk_i), risk = bend
            # exposure × field-normalised lateral extremity. Larger D_i ⇒ larger Wald variance ⇒ more
            # upsets. PER-VENUE (THEORY §3d per-horse-D knob) unless passed explicitly.
            dvg = draw_var_gain if draw_var_gain is not None else DRAW_VAR_GAIN_BY_VENUE.get(venue, DRAW_VAR_GAIN)
            d_h = d_eff
            if dvg != 0.0 and g.height > 1:
                extremity = 2.0 * np.abs(draw - (g.height + 1.0) / 2.0) / (g.height - 1.0)
                d_h = d_eff * (1.0 + dvg * geom.bend_fraction(dist) * extremity)
            lam = dist * dist / (2.0 * np.asarray(d_h))
            rng = np.random.default_rng(seed)
            t = rng.wald(np.broadcast_to(mu, (n_samples, g.height)),
                         np.broadcast_to(lam, (n_samples, g.height)))
            counts = np.bincount(t.argmin(axis=1), minlength=g.height).astype(float)
            p = (counts + 1.0) / (n_samples + g.height)
        preds.append(g.with_columns(pl.Series("p", p), pl.Series("v_c_eb", vc),
                                    pl.Series("w_prime_eb", wp)))
    return pl.concat(preds)
