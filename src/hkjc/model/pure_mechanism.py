"""PureMechanismThetaBuilder — the CAUSAL PAYOFF-TEST θ assembly (journal 0050).

The contrast to the now-deleted production θ-stamp (which stamped the
production ``v_c_eb`` — a meta-signal-fused ability) is the WHOLE POINT of the causal rebuild
(``docs/CAUSAL_MODEL.md`` §6): assemble each runner's drift from PATH-INFERRED PHYSIOLOGY ONLY,
with EVERY meta-signal stripped, and test whether the meta-signals RE-EMERGE as consequences.

WHAT GOES IN (per runner, strictly-prior, leak-free — ``docs/CAUSAL_MODEL.md`` §6 INPUT):
  * ``v_c``     = the horse's **Hill critical velocity** (``identify.hill_fit`` — the de-confounded
                  aerobic CP anchor, NOT the pace-inflated per-race cruise; journal 0048/0049),
                  pooled strictly-prior over the horse's PRIOR races.
  * ``w_prime`` = the **Hill intercept** D' [m] (anaerobic distance capacity) — the velocity-space
                  reserve, on the SAME distance-unit axis the generative drift boost ``w'/L`` reads.
  * the field/draft **deployment** (``field.py``) is the per-race pace coupling — applied as a
    de-confounding correction to the cruise anchor where a horse has no Hill line (the field pace
    is contemporaneous public boundary info, ``docs/CAUSAL_MODEL.md`` §6 — fine).
  * ``mass``    = declared weight (physical; via ``dynamics.body_mass_kg``) → drives the going /
                  surface resistances the Hill CV is read against.
  * ENV: draw, going→C_rr/surface, geometry, field size — kept (genuine exogenous causes, §2.4).
  * RATING — ONLY as the population-prior MEAN for debut / no-path horses (the flagged concession,
    ``docs/CAUSAL_MODEL.md`` T1): a rating-conditioned population mean, NEVER a per-runner ability
    channel fed beside the path.

WHAT IS STRIPPED (must NOT appear — ``docs/CAUSAL_MODEL.md`` §6 STRIPPED):
  form, recency-form, speed-figure, collateral, peak-gap, finishing-sectional, fade, barrier-trial,
  gallop, jockey, trainer, trainer-hot, early-pos, gate-break. **This module imports NONE of
  ``hkjc.model.features`` and reads NONE of those columns.** The drift is Hill-CP + field coupling
  + mass, full stop (the discipline).

LEAK-FREEDOM (the contract — the same leak-free seam the production bridge used):
  * Per-horse physiology is STRICTLY-PRIOR: each run's Hill v_c / W' is fit over that horse's
    EARLIER runs only (expanding window, the run's own path never enters its anchor).
  * The population prior (the debut fallback) is fit TRAIN-ONLY per fold (``pre_fit`` runs once per
    fold on ``train``), rating-conditioned, and stamped onto the test frame as a cheap column read.
  * The per-race θ builder (:func:`pure_mechanism_theta_builder`) reads only pre-race columns.
  * ``assert_no_lookahead`` on the resulting ``fit_predict`` is the proof.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.model.conditions import RaceConditions
from hkjc.model.dynamics import body_mass_kg, cp_for_cruise, make_resistances
from hkjc.model.field import cover_fraction, deconfounded_cruise_velocity
from hkjc.model.identify import (
    Run,
    hill_fit,
    pmax_for_ceiling,
    pmax_opening_run,
    settled_cruise_velocity,
    sprint_ceiling_velocity,
    wprime_spent_run,
)
from hkjc.model.latent import LatentState

# Physical bounds on the inferred aerobic critical velocity (m/s) — a Hill line outside this is a
# degenerate fit (too-thin distance span), routed to the population prior. Matches the field test.
_VC_LO, _VC_HI = 8.0, 22.0
# Drift v_c SAFETY RAIL (widened from the old 12-19, which compressed between-horse spread — the
# #55/0050 finding). The fused read sits naturally in ~13-19; this rail only catches pathological
# fits while letting the genuine ability spread through (the Wald law stays sane to ~10-21 m/s).
_VC_DRIFT_LO, _VC_DRIFT_HI = 10.0, 21.0

# Instrument-fusion variances (RQ-vc1). _SE_FLOOR caps runaway precision; _DEC_SD0 is the default
# per-reading de-confounded-cruise sd (m/s) when a horse has <2 prior readings; _POP_SD is the
# population-prior dispersion (m/s) — large, so the prior only dominates a debut/no-path runner.
_SE_FLOOR, _DEC_SD0, _POP_SD = 0.15, 0.8, 1.5

# Physical bounds for the per-run W'_0 [J] (anaerobic reserve, equine ~100-400 kJ; 0048: Hill 220,
# spent 130 kJ) and P_max [W] (peak power; 0048: opening 66 kW, ceiling 33 kW). Estimates outside
# these are degenerate inversions (gate-timing noise / a too-short path) and are dropped to the
# career/population prior. P_max is read as the MEAN of the opening-transient and sprint-ceiling
# surfaces (0048: opening reads ~2× ceiling — peak-instantaneous vs sustained — so their mean is the
# least-aliased single number for the forward EOM, which wants the deployment ceiling).
_WPRIME_LO, _WPRIME_HI = 10_000.0, 600_000.0
_PMAX_LO, _PMAX_HI = 15_000.0, 120_000.0

# EB-shrink strength K (in "prior-run" units): the career mean of a constant is shrunk toward the
# train-only population mean by n/(n+K) — more prior runs ⇒ less shrink, the physics analog of the
# recency-form EB. K≈3 means a horse needs ~3 prior reads before its own mean outweighs the prior.
_EB_K = 3.0


def eb_shrink(career_mean, n_prior, pop_mean, *, k: float = _EB_K):
    """Empirical-Bayes shrink a career mean toward a population mean: more prior runs ⇒ less shrink.

    ``shrunk = w·career_mean + (1−w)·pop_mean`` with ``w = n/(n+k)``. Where the career mean is
    missing (no prior read) the result is the population mean (full shrink). Vectorised; leak-free
    given a strictly-prior ``career_mean``/``n_prior`` and a TRAIN-ONLY ``pop_mean``.
    """
    cm = np.asarray(career_mean, float)
    n = np.asarray(n_prior, float)
    pm = np.asarray(pop_mean, float)
    have = np.isfinite(cm) & (n > 0)
    w = np.where(have, n / (n + k), 0.0)
    cm = np.where(have, cm, 0.0)
    return w * cm + (1.0 - w) * pm


def fuse_vc_read(vc_hill, vc_hill_se, vc_decon, vc_decon_n, vc_decon_sd, vc_pop):
    """Precision-weighted fusion of the ability instruments (RQ-vc1) — the SINGLE source of truth
    for the v_c read, shared by the θ builder and the eval. Returns the fused critical velocity
    (m/s), UNCLAMPED (the caller applies the physical safety rail).

    Each instrument contributes its estimate at precision 1/variance:
      * Hill slope:           var = cv_se²            (noisy when the across-distance lever arm is short)
      * de-confounded level:  var = (sd/√n)²          (the 0049/0050 read that re-emerges form)
      * population prior:      var = _POP_SD²           (dominates only a debut / no-path runner)
    The noisy Hill slope is thus DERIVED-down-weighted toward the de-confounded level — the #55/0050
    fix, obtained by mechanism, not by hand-prioritising.
    """
    vc_hill, vc_hill_se = np.asarray(vc_hill, float), np.asarray(vc_hill_se, float)
    vc_decon = np.asarray(vc_decon, float)
    vc_decon_n, vc_decon_sd = np.asarray(vc_decon_n, float), np.asarray(vc_decon_sd, float)
    vc_pop = np.asarray(vc_pop, float)

    def _prec(estimate, se):
        ok = np.isfinite(estimate) & np.isfinite(se)
        var = np.maximum(np.where(ok, se, 1.0), _SE_FLOOR) ** 2
        return np.where(ok, 1.0 / var, 0.0)

    dec_sd = np.where(np.isfinite(vc_decon_sd) & (vc_decon_sd > 0), vc_decon_sd, _DEC_SD0)
    dec_n = np.where(np.isfinite(vc_decon_n) & (vc_decon_n >= 1), vc_decon_n, 1.0)
    p_hill = _prec(vc_hill, vc_hill_se)
    p_decon = _prec(vc_decon, dec_sd / np.sqrt(dec_n))
    p_pop = _prec(vc_pop, np.full(np.shape(vc_pop), _POP_SD))
    m_hill = np.where(np.isfinite(vc_hill), vc_hill, 0.0)
    m_decon = np.where(np.isfinite(vc_decon), vc_decon, 0.0)
    m_pop = np.where(np.isfinite(vc_pop), vc_pop, 0.0)
    prec = p_hill + p_decon + p_pop
    return np.where(prec > 0,
                    (p_hill * m_hill + p_decon * m_decon + p_pop * m_pop) / np.where(prec > 0, prec, 1.0),
                    15.0)


# ============================================================ per-run KINEMATICS + field assembly
def _runs_frame(traj: pl.DataFrame, going_mass: pl.DataFrame) -> pl.DataFrame:
    """One row per run: section kinematics + running position + KNOWN mass/going (for the Run)."""
    keys = ["date", "venue", "race_no", "runner"]
    agg = (traj.sort(keys + ["section"])
           .group_by(keys)
           .agg(pl.col("horse_id").first(),
                pl.col("distance").first().alias("distance"),
                pl.col("section_dist"), pl.col("section_time"),
                pl.col("velocity"), pl.col("position")))
    return agg.join(going_mass, on=keys, how="left")


def _going_mass_frame(results_rich: pl.DataFrame) -> pl.DataFrame:
    """Declared weight + carried weight + going per run (physical mass + surface — no meta-signal)."""
    return (results_rich.with_columns(
                pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
                pl.col("Venue").cast(pl.Utf8).alias("venue"),
                pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
                pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                pl.col("Going").cast(pl.Utf8).alias("going"),
                pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt_lb"),
                pl.col("ActWt").cast(pl.Float64, strict=False).alias("carriedwt_lb"))
            .select("date", "venue", "race_no", "runner", "going", "bodywt_lb", "carriedwt_lb")
            .unique(["date", "venue", "race_no", "runner"]))


def _run_from_row(r: dict) -> Run:
    m = body_mass_kg(r.get("bodywt_lb") or 1100.0, r.get("carriedwt_lb") or 0.0)
    return Run(horse_id=r["horse_id"], distance=float(r["distance"]),
               section_dist=np.asarray(r["section_dist"], float),
               section_time=np.asarray(r["section_time"], float),
               velocity=np.asarray(r["velocity"], float),
               mass=m, going=r.get("going") or "GOOD")


# ============================================================ STRICTLY-PRIOR Hill physiology table
def build_pure_mechanism_panel(traj: pl.DataFrame, results_rich: pl.DataFrame) -> pl.DataFrame:
    """Per-run STRICTLY-PRIOR Hill physiology (v_c, W') + the de-confounded fallback cruise level.

    For every run we form, from the horse's EARLIER runs ONLY (the run's own path excluded):
      * ``vc_hill``  — the Hill critical velocity (``identify.hill_fit`` slope) over the prior
        distance-time points (needs ≥3 distinct prior distances; null otherwise — debut/thin).
      * ``wprime_hill`` — the Hill intercept D' [m] (anaerobic distance capacity).
      * ``vc_decon`` — the mean of the horse's prior runs' DE-CONFOUNDED settled cruise (field/draft
        removed, ``field.deconfounded_cruise_velocity``) — the coarser per-run ability proxy that
        carries a horse with too few distinct distances for a Hill line. This is the field-coupling
        deployment read (0049), NOT a meta-signal.
      * ``hill_n_dist`` — prior distinct distances (the Hill identifiability width).

    Pure path physiology + KNOWN mass/going + the contemporaneous-boundary field pace. NO form,
    NO collateral, NO jockey/trainer — none of ``hkjc.model.features`` is imported here.
    """
    keys = ["date", "venue", "race_no", "runner"]
    gm = _going_mass_frame(results_rich)
    traj = traj.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("race_no").cast(pl.Int64, strict=False))
    # per (race, section) field pace (mean velocity) + field size — the OBSERVED boundary condition.
    fld = (traj.group_by(["date", "venue", "race_no", "section"])
           .agg(pl.col("velocity").mean().alias("v_field"), pl.len().alias("fsize")))
    traj = traj.join(fld, on=["date", "venue", "race_no", "section"], how="left")
    runs = (traj.sort(keys + ["section"]).group_by(keys)
            .agg(pl.col("horse_id").first(),
                 pl.col("distance").first().alias("distance"),
                 pl.col("section_dist"), pl.col("section_time"), pl.col("velocity"),
                 pl.col("position"), pl.col("v_field"), pl.col("fsize"))
            .join(gm, on=keys, how="left"))
    runs = runs.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_d"))
    runs = runs.sort(["horse_id", "_d", "race_no", "runner"])

    # accumulate per-horse prior (distance, total_time) for the strictly-prior Hill fit + the
    # prior de-confounded cruise levels + the prior P_max / W'_0 per-run reads (career profile).
    rows = runs.to_dicts()
    hist_dt: dict[str, list[tuple[float, float]]] = {}
    hist_decon: dict[str, list[float]] = {}
    hist_pmax: dict[str, list[float]] = {}      # prior per-run P_max reads [W]
    hist_wp0: dict[str, list[float]] = {}        # prior per-run W'_0 (spent-above-CP) reads [J]
    hist_vc: dict[str, list[float]] = {}         # prior per-run cruise-CP v_c reads [m/s]
    out_vc, out_wp, out_dec, out_n = [], [], [], []
    out_se, out_dn, out_dsd = [], [], []
    # career strictly-prior profile (mean of the horse's prior per-run reads) + the prior count.
    out_vc_car, out_vc_car_n = [], []
    out_pmax_car, out_pmax_car_n = [], []
    out_wp0_car, out_wp0_car_n = [], []
    for r in rows:
        h = r["horse_id"]
        prior = hist_dt.get(h, [])
        # ---- strictly-prior Hill from earlier runs ----
        vc_hill, wp_hill, n_dist, vc_hill_se = None, None, 0, None
        if len(prior) >= 3:
            d = np.array([p[0] for p in prior])
            t = np.array([p[1] for p in prior])
            fit = hill_fit(d, t, min_dist=3)
            if fit is not None and _VC_LO < fit.cv < _VC_HI and fit.d_prime > 0:
                vc_hill, wp_hill, n_dist = float(fit.cv), float(fit.d_prime), int(fit.n_dist)
                vc_hill_se = float(fit.cv_se)  # slope SE — the Hill instrument's precision width
        # ---- strictly-prior de-confounded cruise (the field-coupling read, 0049) ----
        prior_dec = hist_decon.get(h, [])
        vc_dec = float(np.mean(prior_dec)) if prior_dec else None
        vc_dec_n = len(prior_dec)
        vc_dec_sd = float(np.std(prior_dec)) if len(prior_dec) >= 2 else None
        # ---- strictly-prior CAREER PROFILE of the full θ (means of prior per-run reads) ----
        pv, pp, pw = hist_vc.get(h, []), hist_pmax.get(h, []), hist_wp0.get(h, [])
        out_vc_car.append(float(np.mean(pv)) if pv else None)
        out_vc_car_n.append(len(pv))
        out_pmax_car.append(float(np.mean(pp)) if pp else None)
        out_pmax_car_n.append(len(pp))
        out_wp0_car.append(float(np.mean(pw)) if pw else None)
        out_wp0_car_n.append(len(pw))
        out_vc.append(vc_hill)
        out_wp.append(wp_hill)
        out_dec.append(vc_dec)
        out_n.append(n_dist)
        out_se.append(vc_hill_se)
        out_dn.append(vc_dec_n)
        out_dsd.append(vc_dec_sd)
        # ---- update history with THIS run (so it becomes prior for the NEXT run) ----
        run = _run_from_row(r)
        hist_dt.setdefault(h, []).append((run.distance, run.total_time))
        v_settled = settled_cruise_velocity(run)
        cp_run = None
        if v_settled is not None and len(run.velocity) > 2:
            pos = np.asarray(r["position"], float)
            fs = np.asarray(r["fsize"], float)
            cover = np.array([cover_fraction(p, f) for p, f in zip(pos, fs)])
            cov = float(np.mean(cover[1:-1])) if cover[1:-1].size else 0.0
            try:
                dec = deconfounded_cruise_velocity(v_settled, cov, run.res)
            except Exception:
                dec = v_settled
            if _VC_LO < dec < _VC_HI:
                hist_decon.setdefault(h, []).append(dec)
                hist_vc.setdefault(h, []).append(dec)
                # the de-confounded aerobic CP [W] for this run — the W'-spent threshold + the
                # forward EOM's CP. Use the de-confounded v_c so the threshold is not pace-inflated.
                cp_run = cp_for_cruise(dec, run.res)
        # ---- per-run P_max [W]: mean of the opening transient + the sprint-ceiling surfaces ----
        # (0048: opening ≈ 2× ceiling — peak-instantaneous vs sustained — so the mean is the
        #  least-aliased single deployment ceiling for the forward EOM).
        p_open = pmax_opening_run(run)
        p_ceil = pmax_for_ceiling(sprint_ceiling_velocity(run), run.res)
        pmaxes = [p for p in (p_open, p_ceil) if p is not None and np.isfinite(p) and p > 0]
        if pmaxes:
            pmax_run = float(np.mean(pmaxes))
            if _PMAX_LO < pmax_run < _PMAX_HI:
                hist_pmax.setdefault(h, []).append(pmax_run)
        # ---- per-run W'_0 [J]: reserve spent above the de-confounded CP (0048 surface) ----
        if cp_run is not None and cp_run > 0:
            wp0_run = wprime_spent_run(run, cp_run)
            if wp0_run is not None and _WPRIME_LO < wp0_run < _WPRIME_HI:
                hist_wp0.setdefault(h, []).append(wp0_run)
    return runs.select(keys + ["horse_id", "distance"]).with_columns(
        pl.Series("vc_hill", out_vc, dtype=pl.Float64),
        pl.Series("wprime_hill", out_wp, dtype=pl.Float64),
        pl.Series("vc_decon", out_dec, dtype=pl.Float64),
        pl.Series("hill_n_dist", out_n, dtype=pl.Int64),
        pl.Series("vc_hill_se", out_se, dtype=pl.Float64),
        pl.Series("vc_decon_n", out_dn, dtype=pl.Int64),
        pl.Series("vc_decon_sd", out_dsd, dtype=pl.Float64),
        # career strictly-prior full-θ profile (mean of prior per-run reads) + the prior counts —
        # the substrate for the EB-shrunk forward-EOM ability index (journal 0056).
        pl.Series("vc_career", out_vc_car, dtype=pl.Float64),
        pl.Series("vc_career_n", out_vc_car_n, dtype=pl.Int64),
        pl.Series("pmax_career", out_pmax_car, dtype=pl.Float64),
        pl.Series("pmax_career_n", out_pmax_car_n, dtype=pl.Int64),
        pl.Series("wprime0_career", out_wp0_car, dtype=pl.Float64),
        pl.Series("wprime0_career_n", out_wp0_car_n, dtype=pl.Int64),
    )


def attach_pure_mechanism(df: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Join the strictly-prior Hill physiology panel onto the modelling frame by the run key."""
    keys = ["date", "venue", "race_no", "runner"]
    p = panel.select(keys + ["vc_hill", "wprime_hill", "vc_decon", "hill_n_dist",
                             "vc_hill_se", "vc_decon_n", "vc_decon_sd",
                             "vc_career", "vc_career_n", "pmax_career", "pmax_career_n",
                             "wprime0_career", "wprime0_career_n"]).with_columns(
        pl.col("race_no").cast(pl.Int64, strict=False))
    out = df.with_columns(pl.col("race_no").cast(pl.Int64, strict=False))
    return out.join(p, on=keys, how="left")


# ============================================================ train-only population prior (debut)
def make_pure_mechanism_pre_fit():
    """Return ``pre_fit(train, test) -> test'`` that stamps the train-only debut population mean.

    Fits the rating-conditioned population mean of the path-inferred ``vc_hill`` / ``wprime_hill``
    on TRAIN ONLY (per fold), so a debut / thin-prior runner (no Hill line, no de-confounded prior)
    falls back to the population mean for its trip-band and rating — the flagged concession
    (``docs/CAUSAL_MODEL.md`` §6, T1). RATING enters ONLY here (the debut anchor), never as a
    per-runner ability channel. Train-only ⇒ leak-free.
    """
    from hkjc.model.calibration import distance_band

    def _pre_fit(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
        tr = train.filter(pl.col("vc_hill").is_finite())
        # rating-conditioned band means (train-only). Fall back to grand mean where a band is empty.
        bands = ["sprint", "mile", "middle", "staying"]
        tr = tr.with_columns(
            pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("_band"))
        vc_grand = float(tr["vc_hill"].mean()) if tr.height else 15.0
        wp_grand = float(tr["wprime_hill"].mean()) if tr.height else 20.0
        # a single global rating slope on vc_hill (class compression), estimated train-only.
        slope, rmean = 0.0, 0.0
        if "rtg" in tr.columns and tr.height > 10:
            rt = tr["rtg"].to_numpy().astype(float)
            vc = tr["vc_hill"].to_numpy().astype(float)
            ok = np.isfinite(rt) & np.isfinite(vc)
            if ok.sum() > 10:
                rmean = float(rt[ok].mean())
                rc = rt[ok] - rmean
                denom = float((rc * rc).sum())
                slope = float((rc * (vc[ok] - vc[ok].mean())).sum() / denom) if denom > 0 else 0.0
        band_vc = {b: vc_grand for b in bands}
        band_wp = {b: wp_grand for b in bands}
        for b in bands:
            sub = tr.filter(pl.col("_band") == b)
            if sub.height:
                band_vc[b] = float(sub["vc_hill"].mean())
                band_wp[b] = float(sub["wprime_hill"].mean())

        te = test.with_columns(
            pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("_band"))
        b_arr = te["_band"].to_list()
        rt = (te["rtg"].to_numpy().astype(float) if "rtg" in te.columns
              else np.full(te.height, rmean))
        vc_pop = np.array([band_vc.get(b, vc_grand) for b in b_arr])
        vc_pop = vc_pop + slope * np.where(np.isfinite(rt), rt - rmean, 0.0)
        wp_pop = np.array([band_wp.get(b, wp_grand) for b in b_arr])
        return te.with_columns(
            pl.Series("vc_pop", vc_pop, dtype=pl.Float64),
            pl.Series("wprime_pop", wp_pop, dtype=pl.Float64),
        ).drop("_band")

    return _pre_fit


# ============================================================ the PURE-MECHANISM θ builder
def pure_mechanism_theta_builder(train: pl.DataFrame, g: pl.DataFrame):
    """Per-race θ + conditions from PATH-INFERRED PHYSIOLOGY ONLY (strictly-prior, leak-free).

    The drift v_c is, in priority order (each strictly-prior / pre-race):
      1. the horse's **Hill critical velocity** (``vc_hill``) — the de-confounded anchor;
      2. else the horse's prior **de-confounded cruise level** (``vc_decon``) — the field-coupling
         deployment read for a horse with too few distinct distances for a Hill line;
      3. else the **rating-conditioned population mean** (``vc_pop``) — the debut concession.
    W' is the Hill intercept (``wprime_hill``) where available, else the population mean. Every
    source is physics or the population prior; NO meta-signal column is read. The drift v_c is
    clamped to the generative cruise window so the Wald law stays in its physical regime.

    The CONTRAST with the now-deleted production θ-stamp is exact: that one
    stamps ``v_c_eb`` (the meta-signal-fused production ability); this one reads ONLY the Hill
    anchor / field-coupled cruise / population mean.
    """
    del train
    n = g.height

    def _col(name):
        return (g[name].to_numpy().astype(float) if name in g.columns
                else np.full(n, np.nan))

    vc_hill = _col("vc_hill")
    vc_decon = _col("vc_decon")
    vc_pop = _col("vc_pop")
    vc_hill_se = _col("vc_hill_se")
    vc_decon_n = _col("vc_decon_n")
    vc_decon_sd = _col("vc_decon_sd")
    wp_hill = _col("wprime_hill")
    wp_pop = _col("wprime_pop")

    # ---- RQ-vc1: precision-weighted FUSION of the instruments (single source of truth =
    # ``fuse_vc_read``), then the widened safety rail (NOT a spread compressor).
    vc = np.clip(fuse_vc_read(vc_hill, vc_hill_se, vc_decon, vc_decon_n, vc_decon_sd, vc_pop),
                 _VC_DRIFT_LO, _VC_DRIFT_HI)
    wp = np.where(np.isfinite(wp_hill), wp_hill, np.where(np.isfinite(wp_pop), wp_pop, 20.0))
    wp = np.clip(wp, 0.0, 200.0)

    thetas = [LatentState(v_c=float(vc[i]), w_prime=float(wp[i]), rho=0.0, a=1.0)
              for i in range(n)]

    from hkjc.core.hkjc_facts import AWT_DISTANCE
    distance = int(g["distance"][0])
    venue = str(g["venue"][0]) if "venue" in g.columns else "ST"
    surface = "awt" if distance == AWT_DISTANCE else "turf"
    draws = (g["draw"].to_numpy().astype(float) if "draw" in g.columns
             else np.arange(1, n + 1, dtype=float))
    course = str(g["course"][0]) if "course" in g.columns and g["course"][0] is not None else "A"
    conditions = RaceConditions(
        race_id=str(g["race_id"][0]), distance=distance, venue=venue, surface=surface,
        draws=draws, field_size=n, course=course,
    )
    # NB: gate_break_shrunk / fig_disp DELIBERATELY left None — those are stripped meta-signals.
    return thetas, conditions


# convenience for studies / the gate: the surface-resistance check (mass is physical, not a feature)
def physical_resistances(bodywt_lb: float, carried_lb: float, going: str):
    """The KNOWN resistances for a runner (mass + going) — physics only, exposed for diagnostics."""
    return make_resistances(body_mass_kg(bodywt_lb, carried_lb), going)
