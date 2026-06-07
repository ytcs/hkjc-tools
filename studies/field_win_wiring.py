"""S7b WIN-WIRING — does the FIELD/DEPLOYMENT dimension MOVE the win diagnostic? (journal 0065)

The gated next step of the field/deployment cause (journals 0061→0063). 0063 BUILT the front-group
pace-WAR W′-burn gradient mechanism (``field.pace_war_gradient`` / ``pace_war_burn_per_runner``) and
validated it RECOVERS the de-confounded early-pace-gradient composition signal at HV (|ρ| 0.21) while
ST is a ceiling-checked TRUE NULL. The mechanism was NOT yet wired into the generative win drift —
the deliberate, separately-gated step, because pace is a PROXY and WIN is the parked arbiter
(PHILOSOPHY §6: WIN is the DIAGNOSTIC of cumulative causal coverage, not the fitting target).

THIS study wires it and measures the WIN-diagnostic DELTA. The mechanism: a front-runner in a hot,
front-heavy field contests the pace WAR and spends anaerobic reserve EARLY, so it carries LESS reserve
to the finish. We reduce each runner's effective W′ by the per-runner pace-war early burn
(``field.pace_war_burn_per_runner``, from the field's STRICTLY-PRIOR style/pace composition — KNOWN
pre-race ⇒ leak-free) and let the existing generative drift slow the burnt runners:

    W′_eff_i = max(W′_i − pace_war_burn_i, 0)        (the reserve boost (1+W′_eff/L) in the drift)

The term is GUARDED behind ``GenerativeRaceModel(field_pace_war=...)`` which DEFAULTS OFF, so OFF
reproduces the physiology-only model BYTE-IDENTICALLY (the existing 181 tests stay green). The win
delta is physiology-only (OFF) vs +FIELD (ON), both venues, PAIRED fixed-seed (per-race MC seed shared
so Δ is pure mechanism), leak-free (``assert_no_lookahead`` drift = 0). EXPECT: HV a small lift, ST
~flat (the 0062 true-null). Be HONEST: a real-but-small causal dimension that barely moves win is a
legitimate finding — WIN is the diagnostic, not the target.

No coefficient is fitted on the outcome: the burn magnitude is the EOM physics (the force-balance
over-spend at the contested pace), only the ON/OFF and the existing physical front-count threshold are
knobs; ``base``/``style``/v_c/W′ are all strictly-prior.

    PYTHONHASHSEED=0 uv run --group bayes python studies/field_win_wiring.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.backtesting import assert_no_lookahead
from hkjc.core.hkjc_facts import AWT_DISTANCE
from hkjc.model.calibration import fit_energetics_population, horse_energetics_trailing
from hkjc.model.conditions import RaceConditions
from hkjc.model.coupling import CouplingCalibrator
from hkjc.model.field import FRONT_STYLE_CUTOFF, FRONT_WAR_THRESHOLD, pace_war_burn_per_runner
from hkjc.model.gate import gate_score
from hkjc.model.generative import GenerativeRaceModel
from hkjc.model.latent import LatentState
from hkjc.model.pipeline import build_fit_predict

CACHE = Path("data_cache")
N_SAMPLES = 8000
SEED_NS_FAST = 512
TRAIN_FRAC = 0.7        # the earliest 70% of dates train the EB population; the rest is the win test.


# ============================================================ leak-free frame assembly
def _norm_date(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"))


def _race_id_expr() -> pl.Expr:
    return (pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id")


def load_frames(venue: str):
    """WPQ market/outcome panel + the energetics history (for horse_id + path θ), one venue."""
    wpq = _norm_date(pl.read_parquet(CACHE / "wpq_dataset_full.parquet").filter(pl.col("venue") == venue))
    en = _norm_date(pl.read_parquet(CACHE / "energetics_dataset_full.parquet").filter(pl.col("venue") == venue))
    return wpq, en


def prior_base_style(venue: str) -> pl.DataFrame:
    """Per-(date,venue,race_no,horse_id) STRICTLY-PRIOR opening-pace ``base`` + forwardness ``style``.

    ``base`` = expanding prior-mean (shift-by-one) of the horse's own section-1 velocity
    (``trajectory_dataset_full``); ``style`` = forwardness = ``1 − trail_early_pos_norm`` (the journal
    0063/0065 ORIENTATION FIX — ``trail_early_pos_norm`` is the trailing position RANK, high = closer,
    already strictly-prior by construction). Both are earlier-runs-only ⇒ leak-free.
    """
    tj = _norm_date(pl.read_parquet(CACHE / "trajectory_dataset_full.parquet")
                    .filter(pl.col("venue") == venue))
    op = (tj.filter(pl.col("section") == 1)
          .select(["date", "venue", "race_no", "horse_id", "velocity"])
          .rename({"velocity": "v_open"})
          .sort(["horse_id", "date", "race_no"]))
    op = op.with_columns([
        (pl.col("v_open").cum_sum().over("horse_id") - pl.col("v_open")).alias("_cs"),
        (pl.col("v_open").cum_count().over("horse_id") - 1).alias("_cc"),
    ]).with_columns(
        (pl.col("_cs") / pl.when(pl.col("_cc") > 0).then(pl.col("_cc")).otherwise(None)).alias("base")
    ).select(["date", "venue", "race_no", "horse_id", "base"])

    rp = _norm_date(pl.read_parquet(CACHE / "run_policy_dataset.parquet")
                    .filter(pl.col("venue") == venue)).select(
        ["date", "venue", "race_no", "horse_id", "trail_early_pos_norm"])
    rp = rp.with_columns(
        (1.0 - pl.col("trail_early_pos_norm")).clip(0.0, 1.0).alias("style")).drop(
        "trail_early_pos_norm")
    return op.join(rp, on=["date", "venue", "race_no", "horse_id"], how="full", coalesce=True)


def stamp(wpq: pl.DataFrame, en: pl.DataFrame, bs: pl.DataFrame,
          train_dates: list[str]) -> pl.DataFrame:
    """Stamp strictly-prior v_c/W′ θ + horse_id + base/style onto the WPQ frame, population TRAIN-only."""
    # distance per (date, race_no) — a pre-race boundary condition.
    dist = en.select(["date", "venue", "race_no", "distance"]).unique()
    wpq = wpq.join(dist, on=["date", "venue", "race_no"], how="left")
    # horse_id onto WPQ (the energetics panel carries it per runner).
    hid = en.select(["date", "venue", "race_no", "runner", "horse_id"]).unique()
    wpq = wpq.join(hid, on=["date", "venue", "race_no", "runner"], how="left")

    # strictly-prior path energetics (train-only EB population, expanding per-horse pool).
    en_train = en.filter(pl.col("date").is_in(train_dates))
    pop = fit_energetics_population(en_train, group_col="horse_id", distance_col="distance")
    pooled = horse_energetics_trailing(en, pop, date_col="date", group_col="horse_id",
                                       distance_col="distance").select(
        ["date", "venue", "race_no", "runner",
         pl.col("v_c_eb").alias("v_c_prior"), pl.col("w_prime_eb").alias("w_prime_prior")])
    out = wpq.join(pooled, on=["date", "venue", "race_no", "runner"], how="left")
    vc_pop, wp_pop = float(pop.v_c.fixed("mile")), float(pop.w_prime.fixed("mile"))
    out = out.with_columns(pl.col("v_c_prior").fill_null(vc_pop),
                           pl.col("w_prime_prior").fill_null(wp_pop))

    # base/style keyed by (date,venue,race_no,horse_id).
    out = out.join(bs, on=["date", "venue", "race_no", "horse_id"], how="left")
    out = out.with_columns(
        _race_id_expr(),
        pl.lit("turf").alias("surface") if "surface" not in out.columns else pl.col("surface"))
    return out


# ============================================================ per-race pace-war burn
def _conditions_with_burn(g: pl.DataFrame) -> tuple[list[LatentState], RaceConditions]:
    """θ list + RaceConditions for one race, with the per-runner pace-war reserve burn stamped on.

    The burn comes from the field's STRICTLY-PRIOR composition: ``base`` (own prior opening pace) and
    ``style`` (forwardness = 1 − trail_early_pos_norm). Missing base → field prior-mean; missing style
    → field-neutral mean. ``field.pace_war_burn_per_runner`` returns the per-runner reserve depleted
    early (0 unless a contested front war fires). KNOWN pre-race ⇒ leak-free.
    """
    g = g.sort("runner")
    n = g.height
    vc = g["v_c_prior"].to_numpy().astype(float)
    wp = g["w_prime_prior"].to_numpy().astype(float)
    thetas = [LatentState(v_c=float(vc[i]), w_prime=float(wp[i]), rho=0.0, a=1.0) for i in range(n)]

    base = g["base"].to_numpy().astype(float)
    style = g["style"].to_numpy().astype(float)
    ok = np.isfinite(base)
    bm = float(np.nanmean(base[ok])) if ok.any() else float("nan")
    b = np.where(ok, base, bm) if np.isfinite(bm) else base
    sm = float(np.nanmean(style)) if np.isfinite(style).any() else 0.5
    s = np.clip(np.where(np.isfinite(style), style, sm), 0.0, 1.0)
    burn = (pace_war_burn_per_runner(b, s) if np.isfinite(b).all()
            else np.zeros(n))

    distance = int(g["distance"][0])
    surface = "awt" if distance == AWT_DISTANCE else "turf"
    conditions = RaceConditions(
        race_id=str(g["race_id"][0]), distance=distance, venue=str(g["venue"][0]), surface=surface,
        draws=g["draw"].to_numpy().astype(float) if "draw" in g.columns else np.arange(1, n + 1),
        field_size=n, pace_war_burn=burn)
    return thetas, conditions


def win_preds(frame: pl.DataFrame, *, field_pace_war: bool) -> pl.DataFrame:
    """Per-runner win probabilities over the frame, field term ON or OFF, paired fixed-seed.

    OFF = physiology-only (byte-identical baseline); ON = +FIELD pace-war reserve burn. Both share the
    per-race MC seed (the model seeds on race_id) so the Δ is pure mechanism. Returns the frame with a
    ``p`` win-prob column (+ ``field_size`` for the uniform baseline) for ``gate_score``.
    """
    model = GenerativeRaceModel(n_samples=N_SAMPLES, field_pace_war=field_pace_war)
    calibrator = CouplingCalibrator()
    out = []
    for _key, g in frame.group_by("race_id", maintain_order=True):
        thetas, conditions = _conditions_with_burn(g)
        mu, lam = calibrator.couple(thetas, conditions, model)
        od = model.order_distribution(thetas, conditions, mu=mu, lam=lam)
        g = g.sort("runner").with_columns(
            pl.Series("p", od.win_probs()), pl.lit(g.height).alias("field_size"))
        out.append(g)
    return pl.concat(out)


# ============================================================ leak-free proof
def leak_check(framed: pl.DataFrame, cut: str) -> float:
    """Perturbing a test outcome must not move any win prob (assert_no_lookahead drift = 0).

    Uses the production build_fit_predict seam (the field term is a pre-race composition input, so the
    standard leak check on the strictly-prior θ pipeline covers the wiring: outcomes never enter base/
    style/θ or the burn).
    """
    tr = framed.filter(pl.col("date") < cut)
    te = framed.filter(pl.col("date") >= cut).head(2000)
    fp = build_fit_predict(n_samples=SEED_NS_FAST)
    return assert_no_lookahead(fp, tr, te, key_cols=("race_id", "runner"),
                               outcome_cols=["won", "placed", "finish_pos"])


# ============================================================ run
def run_venue(venue: str) -> dict:
    wpq, en = load_frames(venue)
    bs = prior_base_style(venue)
    dates = sorted(wpq["date"].unique().to_list())
    cut = dates[int(TRAIN_FRAC * len(dates))]
    train_dates = [d for d in dates if d < cut]
    framed = stamp(wpq, en, bs, train_dates)

    drift = leak_check(framed, cut)
    test = framed.filter(pl.col("date") >= cut)
    n_races = test.select("race_id").unique().height

    off = win_preds(test, field_pace_war=False)
    on = win_preds(test, field_pace_war=True)
    sc_off = gate_score(off, prob_col="p", outcome_col="won", race_col="race_id")
    sc_on = gate_score(on, prob_col="p", outcome_col="won", race_col="race_id")

    # how often / how much the field term actually fires (a sanity read on the threshold)
    fired = 0
    for _k, g in test.group_by("race_id", maintain_order=True):
        _t, c = _conditions_with_burn(g)
        if c.pace_war_burn is not None and float(np.max(c.pace_war_burn)) > 0.0:
            fired += 1

    print(f"\n{'='*86}\n[{venue}]  WIN-diagnostic delta: physiology-only (field OFF) vs +FIELD (ON)\n{'='*86}")
    print(f"  test window from {cut}   races={n_races}   runners={test.height}")
    print(f"  assert_no_lookahead drift = {drift:.2e}  (0 ⇒ strictly-prior, leak-free)")
    print(f"  pace war fires in {fired}/{n_races} test races "
          f"({100*fired/max(n_races,1):.1f}%)  [threshold n*={FRONT_WAR_THRESHOLD}, "
          f"cutoff={FRONT_STYLE_CUTOFF}]")
    print(f"  {'model':14s} {'%market':>9s} {'ECE':>8s} {'win_LL':>9s}")
    print(f"  {'physiology':14s} {sc_off.get('pct_of_market', float('nan')):>8.2%} "
          f"{sc_off.get('ece', float('nan')):>8.4f} {sc_off.get('race_win_log_loss', float('nan')):>9.4f}")
    print(f"  {'+FIELD':14s} {sc_on.get('pct_of_market', float('nan')):>8.2%} "
          f"{sc_on.get('ece', float('nan')):>8.4f} {sc_on.get('race_win_log_loss', float('nan')):>9.4f}")
    d_pct = sc_on.get("pct_of_market", float("nan")) - sc_off.get("pct_of_market", float("nan"))
    d_ece = sc_on.get("ece", float("nan")) - sc_off.get("ece", float("nan"))
    print(f"  {'Δ (FIELD−phys)':14s} {d_pct:>+8.2%} {d_ece:>+8.4f}  "
          f"(Δ%market >0 ⇒ the field dimension MOVES the win diagnostic)")
    return {"venue": venue, "n_races": n_races, "fired": fired,
            "pct_off": sc_off.get("pct_of_market"), "pct_on": sc_on.get("pct_of_market"),
            "ece_off": sc_off.get("ece"), "ece_on": sc_on.get("ece"), "drift": drift}


def main() -> None:
    venues = [os.environ["FWW_VENUE"]] if os.environ.get("FWW_VENUE") else ["ST", "HV"]
    print(f"S7b FIELD WIN-WIRING — physiology-only vs +FIELD pace-war reserve burn, "
          f"n_samples={N_SAMPLES}, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED','unset')}")
    print("WIN is the parked DIAGNOSTIC of causal coverage (PHILOSOPHY §6); the field term is GUARDED "
          "(default OFF ⇒ byte-identical baseline).")
    rows = [run_venue(v) for v in venues]
    print(f"\n{'='*86}\nSUMMARY  (Δ%market = +FIELD − physiology-only; both leak-free, paired seed)\n{'='*86}")
    print(f"  {'venue':6s} {'N':>5s} {'fired':>6s} {'phys %mkt':>10s} {'+FIELD %mkt':>12s} "
          f"{'Δ%mkt':>8s} {'Δece':>9s}")
    for r in rows:
        print(f"  {r['venue']:6s} {r['n_races']:>5d} {r['fired']:>6d} "
              f"{(r['pct_off'] or float('nan')):>10.2%} {(r['pct_on'] or float('nan')):>12.2%} "
              f"{((r['pct_on'] or float('nan')) - (r['pct_off'] or float('nan'))):>+8.2%} "
              f"{((r['ece_on'] or float('nan')) - (r['ece_off'] or float('nan'))):>+9.4f}")


if __name__ == "__main__":
    main()
