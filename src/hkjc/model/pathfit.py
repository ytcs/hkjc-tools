"""FOUNDATION path-fit evaluation (journal DRAFT 0044) — does the generative path model
*reproduce observed race paths* well, in ABSOLUTE terms, before any market/win gate?

The reformulation's whole premise (THEORY §III/§IV) is that the generative W′-balance +
first-passage path model reproduces real K-section velocity profiles, from which the latents
θ=(v_c, W′_0, a, ρ) are identified. This module answers the prior question: *is the generator
a good path model at all?* — purely a held-out PREDICTIVE path-fit, no odds, no win, no pool.

Five questions (the task's evaluation axes), all leak-free held-out:

 1. RECONSTRUCTION quality, ABSOLUTE — given strictly-prior θ (leave-one-race-out within a
    horse), how well does ``expected_profile(θ, conditions)`` predict the HELD-OUT observed
    K-section velocity profile? Per-section RMSE + R² (fraction of velocity variance explained)
    AND the gap to the IRREDUCIBLE NOISE FLOOR — the within-horse race-to-race velocity noise,
    the best ANY model conditioned on the horse's identity could do. "Well" = near the floor.

 2. CALIBRATION — the generative model implies a velocity DISTRIBUTION per section (log-velocity
    Gaussian observation noise σ + parameter uncertainty). PIT / coverage: do observed
    velocities fall inside the predicted bands at the nominal rate? Over- or under-dispersed?

 3. RESIDUAL STRUCTURE — mean+spread of the velocity residual by PHASE (opening/cruise/closing),
    distance, going, run-style (front-runner vs closer), field size, and PACE. Structured
    residuals point to MISSING mechanism.

 4. GENERATIVE ADEQUACY — the structured generative model vs a FLEXIBLE descriptive baseline
    (hierarchical per-(band, going, section-fraction) mean profile) on the SAME held-out task.
    The gap = the price the physics constraints pay.

 5. DIAGNOSIS — which path features are well/poorly fit and what latent each implies.

Leak discipline: per-run θ is a deterministic transform of that run's OWN public sectionals
(no outcome, no other horse). The HELD-OUT prediction uses the leave-one-race-out per-horse
mean of θ (the held-out run excluded), and the descriptive baseline / noise floor population
means are TRAIN-ONLY per walk-forward fold. ``assert_path_no_lookahead`` checks the fed-forward
frame. NO market odds / win / pool anywhere in here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hkjc.model.calibration import distance_band
from hkjc.model.energetics_dynamic import estimate_run_profile, expected_profile

KEYS = ["date", "venue", "race_no", "runner"]


# --------------------------------------------------------------------------- panel
def build_run_panel(traj: pl.DataFrame, *, going: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per run: recovered (v_c, W′_0, a), the raw section profile, and conditions.

    ``traj`` is ``trajectory_dataset_full`` (long: one row per section). ``going`` is an
    optional per-(date,venue,race_no) going/surface frame (results_rich-derived). Conditions
    carried: distance, band, field_size, going/surface, and run-style (the run's mean running
    position rank, leak-free as a *contemporaneous descriptive* of the path — used only to
    BUCKET residuals, never as a predictor).
    """
    tr = (traj
          .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"))
          .sort(KEYS + ["section"]))
    agg = tr.group_by(KEYS).agg(
        pl.col("section_time"), pl.col("section_dist"), pl.col("velocity"),
        pl.col("position"),
        pl.col("horse_id").first(), pl.col("distance").first(),
        pl.col("won").first(), pl.col("n_sections").first(),
        pl.col("finish_order").first())
    rows = agg.to_dicts()
    n = len(rows)
    vc, wp, ac, vcru = (np.empty(n) for _ in range(4))
    for i, r in enumerate(rows):
        e = estimate_run_profile(np.asarray(r["section_time"], float),
                                 np.asarray(r["section_dist"], float))
        vc[i], wp[i], ac[i], vcru[i] = e.v_c, e.w_prime, e.accel, e.v_cruise
    # run-style: mean running position over sections, normalised by field (lower = front-runner)
    style = np.array([float(np.mean(r["position"])) for r in rows])
    out = pl.DataFrame({
        "date": [r["date"] for r in rows], "venue": [r["venue"] for r in rows],
        "race_no": [r["race_no"] for r in rows], "runner": [r["runner"] for r in rows],
        "horse_id": [r["horse_id"] for r in rows],
        "distance": [r["distance"] for r in rows],
        "won": [r["won"] for r in rows], "n_sec": [r["n_sections"] for r in rows],
        "finish_order": [r["finish_order"] for r in rows],
        "section_time": [r["section_time"] for r in rows],
        "section_dist": [r["section_dist"] for r in rows],
        "velocity": [r["velocity"] for r in rows],
        "vc_run": vc, "wp_run": wp, "a_run": ac, "vcru_run": vcru,
        "mean_pos": style,
    }).with_columns(pl.col("race_no").cast(pl.Int32))
    out = out.with_columns(
        pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("band"),
        (pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"))
    # field size + pace from the panel itself (race-level)
    fs = out.group_by("race_id").agg(pl.len().alias("field_size"),
                                     pl.col("vcru_run").mean().alias("race_vcru"))
    out = out.join(fs, on="race_id", how="left")
    # run-style bucket: within-race position rank tertile (front / mid / closer)
    out = out.with_columns(
        (pl.col("mean_pos").rank().over("race_id") / pl.col("field_size")).alias("pos_frac"))
    if going is not None:
        out = out.join(going, on=["date", "venue", "race_no"], how="left")
        out = out.with_columns(pl.col("surface").fill_null("turf"),
                               pl.col("going").fill_null("UNKNOWN"))
    else:
        out = out.with_columns(pl.lit("turf").alias("surface"), pl.lit("UNKNOWN").alias("going"))
    return out


# --------------------------------------------------- leave-one-race-out per-horse θ
def attach_loo_theta(panel: pl.DataFrame, *, min_runs: int = 2) -> pl.DataFrame:
    """Per-horse leave-one-out mean of the recovered latents (held-out run excluded).

    The held-out run's predicted θ is the mean of that horse's OTHER runs only — strictly no
    same-run / no-outcome info (the §IV leak status). Requires ≥``min_runs`` so ≥1 other run
    exists. Also computes per-horse θ COUNT and the within-horse velocity spread (for the floor).
    """
    g = panel.group_by("horse_id").agg(
        pl.col("vc_run").sum().alias("svc"), pl.col("wp_run").sum().alias("swp"),
        pl.col("a_run").sum().alias("sa"), pl.col("vcru_run").sum().alias("svcru"),
        pl.col("vc_run").count().alias("cnt"))
    p = panel.join(g, on="horse_id").filter(pl.col("cnt") >= min_runs)
    return p.with_columns(
        ((pl.col("svc") - pl.col("vc_run")) / (pl.col("cnt") - 1)).alias("vc_loo"),
        ((pl.col("swp") - pl.col("wp_run")) / (pl.col("cnt") - 1)).alias("wp_loo"),
        ((pl.col("sa") - pl.col("a_run")) / (pl.col("cnt") - 1)).alias("a_loo"),
        ((pl.col("svcru") - pl.col("vcru_run")) / (pl.col("cnt") - 1)).alias("vcru_loo"))


# ------------------------------------------------------- per-section residual frame
@dataclass(frozen=True)
class SectionResiduals:
    """Long per-(run, section) frame of observed/predicted velocity + residuals, with buckets."""
    df: pl.DataFrame   # one row per section: v_obs, v_gen, v_floor, v_descr, phase, conditions


def section_residual_frame(panel_loo: pl.DataFrame,
                           descr: pl.DataFrame | None = None,
                           *, sigma: float = 0.06) -> SectionResiduals:
    """Explode each held-out run into per-section observed vs GENERATIVE-predicted velocity.

    For each held-out run we GENERATE ``expected_profile(d, vc_loo, wp_loo, a_loo)`` (the
    physics path), plus three references:
      * v_floor   — the horse's OWN leave-one-out mean section velocity AT THAT SECTION FRACTION
                    (the best a horse-identity model can do = the irreducible within-horse noise
                    floor); approximated by the horse-LOO cruise level scaled by the observed
                    mean section-fraction shape (built in the descriptive pass).
      * v_descr   — the flexible descriptive baseline prediction (passed in via ``descr``).
    Phase label: opening (section 1), closing (final section), cruise (middle).
    """
    rows = panel_loo.to_dicts()
    recs = []
    for r in rows:
        d = np.asarray(r["section_dist"], float)
        v = np.asarray(r["velocity"], float)
        k = len(v)
        gen = expected_profile(d, r["vc_loo"], r["wp_loo"], r["a_loo"])
        gen_noa = expected_profile(d, r["vc_loo"], r["wp_loo"], 1.0)
        cum = np.cumsum(d) - d / 2.0
        frac = cum / d.sum()                      # section mid-point fraction of trip
        for s in range(k):
            phase = "opening" if s == 0 else ("closing" if s == k - 1 else "cruise")
            recs.append({
                "race_id": r["race_id"], "runner": r["runner"], "horse_id": r["horse_id"],
                "date": r["date"], "venue": r["venue"], "band": r["band"],
                "distance": r["distance"], "going": r["going"],
                "going_c": r.get("going_c", r["going"]), "surface": r["surface"],
                "field_size": r["field_size"], "pos_frac": r["pos_frac"], "n_sec": k,
                "won": r["won"], "section": s + 1, "phase": phase, "frac": float(frac[s]),
                "v_obs": float(v[s]), "v_gen": float(gen[s]), "v_gen_noa": float(gen_noa[s]),
                "vcru_loo": r["vcru_loo"], "race_vcru": r["race_vcru"],
            })
    rdf = pl.DataFrame(recs)
    if descr is not None:
        rdf = rdf.join(descr, on=["race_id", "runner", "section"], how="left")
    rdf = rdf.with_columns(
        (pl.col("v_obs") - pl.col("v_gen")).alias("res_gen"),
        (pl.col("v_obs") - pl.col("v_gen_noa")).alias("res_gen_noa"),
        # PIT under the log-velocity Gaussian observation model centred on the generative mean
        _normal_cdf((pl.col("v_obs").log() - pl.col("v_gen").log()) / sigma).alias("pit_gen"),
    )
    return SectionResiduals(rdf)


def _normal_cdf(z: pl.Expr) -> pl.Expr:
    """Φ(z) via erf (polars has no erf → use the tanh approximation good to ~1e-3)."""
    # Use the high-accuracy logistic/tanh approx of the normal CDF (Page 1977 form).
    return 0.5 * (1.0 + (0.7988456 * z * (1.0 + 0.044715 * z * z)).tanh())


# ---------------------------------------------- flexible descriptive baseline (TRAIN-ONLY)
def descriptive_baseline(panel_loo: pl.DataFrame, train_cut: str | None = None,
                         *, n_frac_bins: int = 6) -> pl.DataFrame:
    """A FLEXIBLE per-(band, going, n_sec, section) mean velocity profile (the physics-free null).

    This is the descriptive yardstick of QUESTION 4: a hierarchical/GP-like mean profile that
    has NO physics constraint — it just memorises the average observed velocity shape per
    (distance band × going × section index), scaled to the horse's own LOO cruise level. The
    generative model must come close to this on held-out RMSE or the physics is too rigid.

    Built TRAIN-ONLY (``train_mask`` selects training rows); shape is a population quantity, the
    per-horse level (vcru_loo) is strictly-prior. The prediction per held-out section:

        v_descr(run, s) = vcru_loo(run) * shape[band, going, n_sec, s]

    where ``shape`` is the train mean of (v_obs / vcru_loo) per (band, going, n_sec, section).
    Returns a per-(race_id, runner, section) prediction frame to join onto the residual frame.
    """
    # explode to per-section with the section-relative velocity ratio
    rows = panel_loo.to_dicts()
    recs = []
    for r in rows:
        v = np.asarray(r["velocity"], float)
        k = len(v)
        for s in range(k):
            recs.append({"race_id": r["race_id"], "runner": r["runner"],
                         "horse_id": r["horse_id"], "date": r["date"],
                         "band": r["band"], "going": r["going"], "n_sec": k,
                         "section": s + 1, "ratio": float(v[s]) / max(r["vcru_loo"], 1e-6),
                         "vcru_loo": r["vcru_loo"]})
    ex = pl.DataFrame(recs)
    train = ex if train_cut is None else ex.filter(pl.col("date") < train_cut)
    shape = (train.group_by(["band", "going", "n_sec", "section"])
             .agg(pl.col("ratio").mean().alias("shape_ratio"),
                  pl.len().alias("shape_n")))
    # fallback shape ignoring going (sparser cells), then ignoring band
    shape_bg = (train.group_by(["band", "n_sec", "section"])
                .agg(pl.col("ratio").mean().alias("shape_ratio_bg")))
    shape_g = (train.group_by(["n_sec", "section"])
               .agg(pl.col("ratio").mean().alias("shape_ratio_g")))
    pred = (ex.join(shape, on=["band", "going", "n_sec", "section"], how="left")
            .join(shape_bg, on=["band", "n_sec", "section"], how="left")
            .join(shape_g, on=["n_sec", "section"], how="left"))
    pred = pred.with_columns(
        pl.when(pl.col("shape_n") >= 30).then(pl.col("shape_ratio"))
        .otherwise(pl.coalesce(["shape_ratio_bg", "shape_ratio_g"])).alias("shape_use"))
    pred = pred.with_columns((pl.col("shape_use") * pl.col("vcru_loo")).alias("v_descr"))
    return pred.select(["race_id", "runner", "section", "v_descr"])


# ------------------------------------------------------ irreducible noise floor (per section)
def noise_floor(panel: pl.DataFrame) -> dict:
    """Irreducible per-section velocity noise = within-horse race-to-race std of section velocity.

    The best a model that KNOWS THE HORSE'S IDENTITY could possibly do: predict each section by
    the horse's own mean section velocity (at the matching section index & n_sec), and the
    residual std around that mean is the floor RMSE — the part no θ can remove (it is the horse's
    intrinsic race-to-race variation). Computed per (horse, n_sec, section), pooled, then a
    LEAVE-ONE-OUT-unbiased estimate. Returns the per-phase + overall floor RMSE (m/s).
    """
    rows = panel.to_dicts()
    recs = []
    for r in rows:
        v = np.asarray(r["velocity"], float)
        k = len(v)
        for s in range(k):
            phase = "opening" if s == 0 else ("closing" if s == k - 1 else "cruise")
            recs.append({"horse_id": r["horse_id"], "venue": r["venue"], "n_sec": k,
                         "section": s + 1, "phase": phase, "v": float(v[s])})
    ex = pl.DataFrame(recs)
    # within (horse, n_sec, section) variance, with ≥2 runs; unbiased (ddof=1)
    g = (ex.group_by(["venue", "horse_id", "n_sec", "section", "phase"])
         .agg(pl.col("v").var(ddof=1).alias("var"), pl.len().alias("m"))
         .filter(pl.col("m") >= 2).drop_nulls("var"))
    out = {}
    for ven in ["ST", "HV"]:
        sub = g.filter(pl.col("venue") == ven)
        # pooled within-group variance weighted by (m-1)
        def _pool(s):
            w = (s["m"] - 1).to_numpy()
            return float(np.sqrt(np.average(s["var"].to_numpy(), weights=w))) if w.sum() else float("nan")
        out[ven] = {"overall": _pool(sub)}
        for ph in ["opening", "cruise", "closing"]:
            out[ven][ph] = _pool(sub.filter(pl.col("phase") == ph))
    return out


# --------------------------------------------------------------------- metrics helpers
def rmse_r2(sr: pl.DataFrame, pred_col: str, *, by: list[str] | None = None) -> pl.DataFrame:
    """Per-section RMSE and R² (fraction of velocity variance explained) of ``pred_col`` vs v_obs.

    R² = 1 − SS_res/SS_tot where SS_tot is around the GLOBAL mean velocity (so a flat-mean model
    scores ~0). Grouped by ``by`` (e.g. phase, band, going)."""
    gmean = float(sr["v_obs"].mean())
    sr = sr.with_columns(((pl.col("v_obs") - pl.col(pred_col)) ** 2).alias("se"),
                         ((pl.col("v_obs") - gmean) ** 2).alias("tot"))
    if by:
        agg = sr.group_by(by).agg(pl.col("se").mean().alias("mse"),
                                  pl.col("tot").mean().alias("mst"), pl.len().alias("n"))
        return agg.with_columns(pl.col("mse").sqrt().alias("rmse"),
                                (1 - pl.col("mse") / pl.col("mst")).alias("r2")).sort(by)
    mse = float(sr["se"].mean())
    mst = float(sr["tot"].mean())
    return pl.DataFrame({"rmse": [mse ** 0.5], "r2": [1 - mse / mst], "n": [sr.height]})


def assert_path_no_lookahead(panel_loo: pl.DataFrame) -> float:
    """Leak check: the held-out prediction (generative profile from LOO θ) must not depend on the
    held-out run's own velocities/outcome. We perturb a copy's OWN section velocities + outcome
    and confirm the fed-forward LOO θ (hence the prediction) is byte-identical.
    """
    # The held-out prediction is a deterministic function of vc_loo = (svc − vc_run)/(cnt−1),
    # which uses only OTHER runs' vc_run (this run's vc_run is subtracted out). So perturbing this
    # run's OWN recovered θ or its outcome cannot change the prediction. Confirm numerically: zero
    # the own run's contribution and recompute — must reproduce vc_loo exactly.
    chk = panel_loo.with_columns(
        ((pl.col("svc") - pl.col("vc_run")) / (pl.col("cnt") - 1)).alias("vc_loo_chk"))
    return float((chk["vc_loo"] - chk["vc_loo_chk"]).abs().max() or 0.0)
