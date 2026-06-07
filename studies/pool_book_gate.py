"""S4 — the MULTI-POOL gate: the model's skill% across the efficiency-weighted pool BOOK.

The win gate scores only the order distribution's LEFT TAIL (P(1st)). The pool BOOK scores the
WHOLE joint object — every HKJC pool is a functional of the ONE :class:`OrderDistribution`
(``hkjc.model.pricer``), so scoring win/place/quinella/quinella-place/forecast/tierce/trio at once
measures whether the joint *structure* (rank correlations, separations) carries skill win cannot
(``docs/RESEARCH_PROGRAM.md`` §4, RQ-pool-*). This is the LAST yardstick (PHILOSOPHY §6) — a
MEASUREMENT of the mid-rebuild model, not a fitting target.

WHAT THE MODEL IS HERE. The per-runner drift θ = (v_c, W') is the horse's **path-inferred
energetics**, pooled STRICTLY-PRIOR (each horse's EARLIER runs only) by the train-only EB population
(``calibration.horse_energetics_trailing``), with NO meta-signal column read. The order distribution
is then the existing generative Wald first-passage law via ``build_fit_predict`` + the DEFAULT
θ-builder (it does NOT depend on the in-flight ability-READ fusion, journal 0052). This is the honest
physics generator; we score what it currently captures.

HONEST BASELINES (per pool, de-vigged, paired fixed-seed):
  * MARKET  — the de-vigged dividend: market prob of the realized combination = (1−takeout)/decimal,
    takeout SOURCED from ``hkjc.core.hkjc_facts.takeout_for``. The dividend prices ONLY the realized
    (winning) combination, so %market is on that one combination — a valid but PARTIAL benchmark
    (journal 0043 caveat).
  * UNIFORM — the no-skill order law: every runner equally fast (independent-uniform argmin), priced
    through the SAME pricer, so the baseline's joint structure is the honest no-information one.

The metric is the EXACT %-of-market the win gate uses, per pool (``gate.pool_book_score``):
    %market = (uniform_LL − model_LL) / (uniform_LL − market_LL)
on the realized-combination NLL. >0 ⇒ the model beats uniform; →100% ⇒ it captures the market's edge.

LEAK-FREE. θ is strictly-prior (``assert_no_lookahead`` drift = 0, proven below); the population is
fit train-only; the sampler is a pure function of (θ, conditions) + a race-id seed. The dividend
window never enters θ.

    PYTHONHASHSEED=0 uv run --group bayes python studies/pool_book_gate.py

Dividends on disk are ST only (``data_cache/dividends_st.parquet``); the study runs every venue with a
dividend file (HV would slot in automatically). Both venues' WIN/PLACE are also scored from the WPQ
panel directly (no dividend needed) so the sharp anchors are reported for ST AND HV.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.backtesting import assert_no_lookahead
from hkjc.core.hkjc_facts import places_paid, takeout_for
from hkjc.model.calibration import fit_energetics_population, horse_energetics_trailing
from hkjc.model.gate import pool_book_score
from hkjc.model.generative import GenerativeRaceModel, OrderDistribution
from hkjc.model.pipeline import build_fit_predict, predict_race
from hkjc.model.pricer import PoolPricer

CACHE = Path("data_cache")
N_SAMPLES = 8000          # MC sample size for the order distribution (exotics need the tail mass)
SEED_NS_FAST = 512        # smaller sample for the no-lookahead adversarial check

# The pools we score, in efficiency order (sharp anchors first; exotics down-weighted, §4).
POOL_ORDER = ["WIN", "PLA", "QIN", "QPL", "FCT", "TCE", "TRI"]


# ============================================================ leak-free frame assembly
def _norm_date(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"))


def load_frame(venue: str) -> pl.DataFrame:
    """The WPQ market/outcome panel + the STRICTLY-PRIOR path-energetics θ columns, one venue.

    Joins ``wpq_dataset_full`` (market_win_prob / draw / finish_pos / won / field_size — the env +
    outcome) to the per-run path energetics (``energetics_dataset_full``: v_c, W' inferred from the
    path) pooled strictly-prior. The result carries ``v_c_prior`` / ``w_prime_prior`` (the columns
    the DEFAULT θ-builder reads) and never a meta-signal. Population fit is deferred to per-fold so
    this returns the raw join; the strictly-prior pooling is leak-free regardless of the fold (it is
    an expanding window over a horse's OWN earlier runs).
    """
    wpq = _norm_date(pl.read_parquet(CACHE / "wpq_dataset_full.parquet").filter(pl.col("venue") == venue))
    en = _norm_date(pl.read_parquet(CACHE / "energetics_dataset_full.parquet").filter(pl.col("venue") == venue))
    return wpq, en


def stamp_priors(wpq: pl.DataFrame, en: pl.DataFrame, train_dates: list[str]) -> pl.DataFrame:
    """Stamp ``v_c_prior`` / ``w_prime_prior`` onto WPQ, strictly-prior, population fit TRAIN-ONLY.

    The EB population is fit on the TRAIN rows of the energetics panel only; the per-run pooled
    estimate is the strictly-prior expanding mean over each horse's EARLIER runs
    (``horse_energetics_trailing`` — a run never sees itself or any later run). The pooled θ is then
    joined onto the full WPQ frame by (date, venue, race_no, runner). A runner with no path history
    falls back to the population mean. NO outcome and NO meta-signal enters θ.
    """
    en_train = en.filter(pl.col("date").is_in(train_dates))
    pop = fit_energetics_population(en_train, group_col="horse_id", distance_col="distance")
    pooled = horse_energetics_trailing(en, pop, date_col="date", group_col="horse_id",
                                       distance_col="distance")
    pooled = pooled.select(["date", "venue", "race_no", "runner",
                            pl.col("v_c_eb").alias("v_c_prior"),
                            pl.col("w_prime_eb").alias("w_prime_prior")])
    out = wpq.join(pooled, on=["date", "venue", "race_no", "runner"], how="left")
    # debut / no-path-history fallback: the band population mean (rating-free here).
    vc_pop = float(pop.v_c.fixed("mile"))
    wp_pop = float(pop.w_prime.fixed("mile"))
    out = out.with_columns(
        pl.col("v_c_prior").fill_null(vc_pop), pl.col("w_prime_prior").fill_null(wp_pop))
    # the generative frame keys the default builder expects.
    return out.with_columns(
        (pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"),
        pl.lit(1200).alias("distance") if "distance" not in out.columns else pl.col("distance"),
    )


def _attach_distance(wpq: pl.DataFrame, en: pl.DataFrame) -> pl.DataFrame:
    """Race distance per (date, race_no) from the energetics panel (a pre-race boundary condition)."""
    dist = (en.select(["date", "venue", "race_no", "distance"]).unique())
    return wpq.join(dist, on=["date", "venue", "race_no"], how="left")


# ============================================================ order-distribution prediction
def predict_order_dists(frame: pl.DataFrame) -> dict[str, tuple[OrderDistribution, dict[int, int]]]:
    """One OrderDistribution per race via the DEFAULT θ-builder (the model), keyed by race_id.

    Returns ``{race_id: (order_dist, saddle->pos)}`` where ``saddle->pos`` maps the HKJC saddle
    (runner number) to the 0-based field position the order indices use (sorted by runner, the
    builder's convention). Deterministic given the race_id seed (paired fixed-seed).
    """
    from hkjc.model.coupling import CouplingCalibrator
    from hkjc.model.pipeline import _default_theta_builder
    model = GenerativeRaceModel(n_samples=N_SAMPLES)
    calibrator = CouplingCalibrator()
    out = {}
    for _key, g in frame.group_by("race_id", maintain_order=True):
        g = g.sort("runner")
        thetas, conditions = _default_theta_builder(frame, g)
        od, _p = predict_race(thetas, conditions, calibrator=calibrator, model=model)
        saddle_to_pos = {int(r): i for i, r in enumerate(g["runner"].to_list())}
        out[str(g["race_id"][0])] = (od, saddle_to_pos)
    return out


def _uniform_order_dist(n: int, race_id: str) -> OrderDistribution:
    """The no-skill order law: every runner equally fast (independent-uniform argmin), SAME pricer.

    A pure function of (n, race_id seed) — the honest baseline whose joint structure carries NO
    information, scored through the identical PoolPricer so the comparison is structure-for-structure.
    """
    seed = abs(hash(("UNI", race_id))) % (2**31)
    rng = np.random.default_rng(seed)
    t = rng.random((N_SAMPLES, n))
    return OrderDistribution(orders=np.argsort(t, axis=1), n=n)


# ============================================================ pricing the realized combinations
def _parse(combo: str) -> list[int]:
    """HKJC saddle combination string → list of 1-based saddle ints (handles ',' and '>' separators)."""
    s = str(combo).replace(">", ",").replace("/", ",").replace(" ", "")
    return [int(x) for x in s.split(",") if x.isdigit()]


def _model_prob(pricer: PoolPricer, pool: str, saddles: list[int], pos: dict[int, int],
                k: int) -> float | None:
    """Model probability of one realized combination via the pricer (a functional of the order dist).

    Maps the dividend's saddle numbers to 0-based field positions, then dispatches to the matching
    realized-combination functional. Returns None if a saddle is not in the field (a scratching /
    coupling artefact) so the row is skipped, not mis-scored.
    """
    try:
        idx = [pos[s] for s in saddles]
    except KeyError:
        return None
    if pool == "WIN" and len(idx) == 1:
        return pricer.prob_win(idx[0])
    if pool == "PLA" and len(idx) == 1:
        return float(pricer.place(k)[idx[0]]) if k > 0 else None
    if pool == "QIN" and len(idx) == 2:
        return pricer.prob_quinella(idx[0], idx[1])
    if pool == "QPL" and len(idx) == 2:
        return pricer.prob_quinella_place(idx[0], idx[1], k)
    if pool == "FCT" and len(idx) == 2:
        return pricer.prob_forecast(idx[0], idx[1])
    if pool == "TCE" and len(idx) == 3:
        return pricer.prob_tierce(idx[0], idx[1], idx[2])
    if pool == "TRI" and len(idx) == 3:
        return pricer.prob_trio(idx)
    return None


def score_pool_book(frame: pl.DataFrame, dividends: pl.DataFrame, venue: str) -> pl.DataFrame:
    """Build the scored frame (one row per realized combo) for ``pool_book_score``.

    For every dividend row of a pool we score: the model prob (the order-dist functional), the
    de-vigged MARKET prob = (1−takeout)/decimal, and the UNIFORM baseline prob (same functional, the
    no-skill order law). WIN is the winner (one row); PLA is scored PER LEG (each placed-runner
    dividend one row — exactly what the place pool pays and de-vigs cleanly); the order/set pools
    (QIN/QPL/FCT/TCE/TRI) are scored on the realized winning combination. Every pool's market
    baseline is the de-vigged DIVIDEND (a real observed price), so PLA's market is the place pool
    itself — not a synthetic win→place proxy.
    """
    div = dividends.filter(pl.col("pool").is_in(POOL_ORDER))
    # restrict to races present in the frame (have an order distribution)
    races = predict_order_dists(frame)
    uniform_cache: dict[str, OrderDistribution] = {}

    rows = []
    for r in div.iter_rows(named=True):
        rid = f"{r['date']}_{r['race_no']}"
        if rid not in races:
            continue
        od, pos = races[rid]
        n = od.n
        k = places_paid(n)
        saddles = _parse(r["combination"])
        pricer = PoolPricer(od)
        p_model = _model_prob(pricer, r["pool"], saddles, pos, k)
        if p_model is None:
            continue
        if rid not in uniform_cache:
            uniform_cache[rid] = _uniform_order_dist(n, rid)
        p_uniform = _model_prob(PoolPricer(uniform_cache[rid]), r["pool"], saddles, pos, k)
        if p_uniform is None:
            continue
        dec = r.get("decimal")
        if dec is None or dec <= 0:
            continue
        p_market = (1.0 - takeout_for(r["pool"])) / float(dec)
        rows.append({"race_id": rid, "pool": r["pool"], "p_model": p_model,
                     "p_market": p_market, "p_uniform": p_uniform})
    return pl.DataFrame(rows)


# ============================================================ win anchor (no-dividend fallback)
def win_anchor(frame: pl.DataFrame) -> pl.DataFrame:
    """WIN %market from the WPQ panel directly (no dividend needed) — the no-dividend fallback.

    The WPQ frame carries the per-runner market_win_prob and the realized winner, so the WIN sharp
    anchor is scorable for any venue (used when a venue has no dividend file). %market is the exact
    win definition; market = the de-vigged (within-race renormalised) win prob, uniform = 1/field.
    """
    races = predict_order_dists(frame)
    by_runner = {f"{d}_{rn}": g for (d, rn), g in
                 frame.group_by(["date", "race_no"], maintain_order=True)}
    win_rows = []
    for rid, (od, pos) in races.items():
        g = by_runner.get(rid)
        if g is None:
            continue
        n = od.n
        win_p = PoolPricer(od).win()
        mkt = g["market_win_prob"].to_numpy().astype(float)
        mkt = mkt / mkt.sum() if mkt.sum() > 0 else np.full(n, 1.0 / n)
        won = g["won"].to_numpy().astype(int)
        for i, saddle in enumerate(g["runner"].to_list()):
            if won[i] == 1:
                p = pos[int(saddle)]
                win_rows.append({"race_id": rid, "pool": "WIN", "p_model": float(win_p[p]),
                                 "p_market": float(mkt[i]), "p_uniform": 1.0 / n})
    return pl.DataFrame(win_rows)


# ============================================================ leak-free proof
def leak_check(wpq: pl.DataFrame, en: pl.DataFrame, venue: str) -> float:
    """Perturbing a test outcome must not move any model prob (assert_no_lookahead drift = 0)."""
    dates = sorted(wpq["date"].unique().to_list())
    cut = dates[int(0.7 * len(dates))]
    train_dates = [d for d in dates if d < cut]
    framed = stamp_priors(_attach_distance(wpq, en), en, train_dates)
    framed = _ensure_surface(framed)
    tr = framed.filter(pl.col("date") < cut)
    te = framed.filter(pl.col("date") >= cut).head(2000)
    fp = build_fit_predict(n_samples=SEED_NS_FAST)
    return assert_no_lookahead(fp, tr, te, key_cols=("race_id", "runner"),
                               outcome_cols=["won", "placed", "finish_pos"])


def _ensure_surface(frame: pl.DataFrame) -> pl.DataFrame:
    """The default builder reads distance/draw/venue off the frame; guarantee they are present."""
    if "distance" not in frame.columns:
        frame = frame.with_columns(pl.lit(1200).alias("distance"))
    return frame.with_columns(pl.col("distance").fill_null(1200))


# ============================================================ reporting
def _print_table(title: str, scored_by_pool: dict, venue: str):
    print(f"\n{'='*82}\n[{venue}] {title}\n{'='*82}")
    print(f"  {'pool':5s} {'n':>5s}  {'%market':>8s}   {'model_LL':>9s} {'market_LL':>9s} "
          f"{'unif_LL':>8s}  {'p_model':>8s} {'p_mkt':>8s}")
    for pool in POOL_ORDER:
        s = scored_by_pool.get(pool)
        if not s:
            continue
        print(f"  {pool:5s} {s['n_races']:>5d}  {s['pct_of_market']:>7.1%}   "
              f"{s['model_ll']:>9.4f} {s['market_ll']:>9.4f} {s['uniform_ll']:>8.4f}  "
              f"{s['mean_p_model']:>8.4f} {s['mean_p_market']:>8.4f}")


def run_venue(venue: str):
    wpq, en = load_frame(venue)
    wpq = _attach_distance(wpq, en)
    print(f"\n[{venue}] WPQ panel: {wpq.select(['date','race_no']).unique().height} races, "
          f"{wpq.height} runners; energetics history {en.height} runs")

    drift = leak_check(wpq, en, venue)
    print(f"[{venue}] assert_no_lookahead drift = {drift:.2e}  (0 ⇒ strictly-prior, leak-free)")

    # the dividend window defines TEST; everything earlier is TRAIN for the population fit.
    div_path = CACHE / f"dividends_{venue.lower()}.parquet"
    scored_pools: dict[str, dict] = {}

    if div_path.exists():
        dividends = pl.read_parquet(div_path).with_columns(
            pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"))
        test_dates = sorted(dividends["date"].unique().to_list())
        train_dates = [d for d in sorted(wpq["date"].unique().to_list()) if d < min(test_dates)]
        framed = _ensure_surface(stamp_priors(wpq, en, train_dates))
        test_frame = framed.filter(pl.col("date").is_in(test_dates))
        # ALL pools (WIN/PLA + exotics) scored from the de-vigged dividends on the realized combo
        scored = score_pool_book(test_frame, dividends, venue)
        if scored.height:
            scored_pools.update(pool_book_score(scored))
        _print_table("POOL-BOOK skill% (dividend test window, de-vigged, paired fixed-seed)",
                     scored_pools, venue)
    else:
        # no dividend file for this venue: still report the WIN/PLACE sharp anchors on a holdout.
        dates = sorted(wpq["date"].unique().to_list())
        cut = dates[int(0.85 * len(dates))]
        train_dates = [d for d in dates if d < cut]
        framed = _ensure_surface(stamp_priors(wpq, en, train_dates))
        test_frame = framed.filter(pl.col("date") >= cut)
        wa = win_anchor(test_frame)
        if wa.height:
            scored_pools["WIN"] = pool_book_score(wa)["WIN"]
        print(f"[{venue}] no dividends_{venue.lower()}.parquet — exotic pools not scorable from "
              f"dividends; WIN anchor only (holdout from {cut}).")
        _print_table("WIN sharp anchor (holdout, paired fixed-seed)", scored_pools, venue)
    return scored_pools


def main():
    venues = [os.environ["PBG_VENUE"]] if os.environ.get("PBG_VENUE") else ["ST", "HV"]
    print(f"S4 multi-pool gate — model = path-energetics (strictly-prior, NO meta-signal), "
          f"n_samples={N_SAMPLES}, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED','unset')}")
    for v in venues:
        run_venue(v)


if __name__ == "__main__":
    main()
