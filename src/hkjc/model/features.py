"""Leak-free ability features for the ABM — the strongest signals, single-sourced.

These are the per-horse trait measurements the production model rides on, each computed
**leak-free**: a runner's value is the mean/rate over its STRICTLY PRIOR runs only
(expanding mean shifted by one), so a backtest can never see a horse's current or future
result. They are pure transforms over already-loaded frames (no I/O), so the same code
serves the walk-forward backtest and a live race card.

- :func:`form_table` — expanding mean of prior relative-finish. Form is the dominant
  ability signal (≈18% of market skill) and survives the handicap.
- :func:`jockey_table` — trailing win strike-rate of the booked jockey (≈+7pp).
- :func:`style_table` — trailing mean ``rel_early`` (front-runner ↑ / closer ↓), a speed-based
  running-style signal. (The position-based :func:`early_position_table` is what feeds the live
  positional-fade term; ``style`` is currently unused by the production fit — kept as a feature.)

Recency note: an exp-weighted form is +8pp in a bare conditional logit but does NOT
transfer to the ABM (the recency the market prices is already carried by the official
rating, which enters nonlinearly via ``v_c.fixed(band, rtg)``), so the career mean stays.
"""
from __future__ import annotations

import json

import polars as pl

from .energetics import estimate_run_energetics, section_distances


def _norm_date(col: str = "date") -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.replace_all("/", "-")


def form_table(energetics: pl.DataFrame, wpq: pl.DataFrame) -> pl.DataFrame:
    """Leak-free expanding mean of prior relative-finish per horse (all venues count).

    ``relative-finish = 1 − (finish_pos−1)/(field_size−1) ∈ [0,1]``, 1 = won. Returns
    (date, venue, race_no, runner, form); ``form`` is null on debut.
    """
    e = energetics.with_columns(_norm_date().alias("date"))
    w = wpq.with_columns(_norm_date().alias("date"))
    h = (e.select("date", "venue", "race_no", "runner", "horse_id")
         .join(w.select("date", "venue", "race_no", "runner", "finish_pos", "field_size"),
               on=["date", "venue", "race_no", "runner"])
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       (1.0 - (pl.col("finish_pos") - 1) / (pl.col("field_size") - 1)).alias("rf"))
         .sort(["horse_id", "d"]))
    h = h.with_columns(
        ((pl.col("rf").cum_sum().over("horse_id") - pl.col("rf"))
         / (pl.col("rf").cum_count().over("horse_id") - 1)).alias("form"))
    return h.select("date", "venue", "race_no", "runner",
                    pl.when(pl.col("form").is_finite()).then(pl.col("form")).alias("form"))


def prior_starts_table(energetics: pl.DataFrame, wpq: pl.DataFrame) -> pl.DataFrame:
    """Leak-free count of a horse's STRICTLY PRIOR official starts before each race.

    ``prior_starts = cum_count(runs) − 1`` over runs sorted by date (all venues count), so a
    debutant (first official start) gets 0 and a horse's k-th career start gets ``k−1``. Same
    expanding/strictly-prior pattern as :func:`form_table`; contains no information from the
    current or any later run. Returns (date, venue, race_no, runner, prior_starts).

    The debut flag: every history-based ability feature (form, sectional v_c residual, the
    horse's own jockey history) falls back to neutral on a debut, and the official rating is
    fill-null'd to the field median — a fabricated, anti-predictive class signal for genuinely
    unrated first-timers. ``prior_starts == 0`` marks exactly those rows. NOTE: a simple RTG-
    suppression on this flag does NOT convert (the rating slope is tiny and the market is already
    efficient on debuts — see journal); its real use is keying NEW debut signal (barrier-trial
    form), where the horse has no race history but a barrier trial to read.
    """
    e = energetics.with_columns(_norm_date().alias("date"))
    w = wpq.with_columns(_norm_date().alias("date"))
    h = (e.select("date", "venue", "race_no", "runner", "horse_id")
         .join(w.select("date", "venue", "race_no", "runner"),
               on=["date", "venue", "race_no", "runner"])
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
         .sort(["horse_id", "d"]))
    h = h.with_columns(
        (pl.col("d").cum_count().over("horse_id") - 1).alias("prior_starts"))
    return h.select("date", "venue", "race_no", "runner", "prior_starts")


def recency_form_table(energetics: pl.DataFrame, wpq: pl.DataFrame,
                       half_life: float = 2.0) -> pl.DataFrame:
    """Leak-free EXP-WEIGHTED form (recency) — recent runs weighted more (half-life in races).

    Same relative-finish signal as :func:`form_table` but an exponentially-weighted mean (strictly
    prior) instead of the flat expanding mean. Tests whether recency adds over the career mean (it
    helped a bare conditional logit but historically not the ABM, whose rating channel carries recency).
    """
    e = energetics.with_columns(_norm_date().alias("date"))
    w = wpq.with_columns(_norm_date().alias("date"))
    h = (e.select("date", "venue", "race_no", "runner", "horse_id")
         .join(w.select("date", "venue", "race_no", "runner", "finish_pos", "field_size"),
               on=["date", "venue", "race_no", "runner"])
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       (1.0 - (pl.col("finish_pos") - 1) / (pl.col("field_size") - 1)).alias("rf"))
         .sort(["horse_id", "d"]))
    h = h.with_columns(
        pl.col("rf").ewm_mean(half_life=half_life, ignore_nulls=True).over("horse_id").alias("_ewm"))
    h = h.with_columns(pl.col("_ewm").shift(1).over("horse_id").alias("form_rec"))
    return h.select("date", "venue", "race_no", "runner",
                    pl.when(pl.col("form_rec").is_finite()).then(pl.col("form_rec")).alias("form_rec"))


def jockey_table(results_rich: pl.DataFrame) -> pl.DataFrame:
    """Leak-free trailing win strike-rate of the jockey riding each horse, keyed (date, horse_id)."""
    rr = (results_rich
          .with_columns(pl.col("date").cast(pl.Utf8).alias("date"),
                        (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int8).alias("w"),
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .sort(["Jockey", "d"]))
    rr = rr.with_columns(((pl.col("w").cum_sum().over("Jockey") - pl.col("w"))
                          / (pl.col("w").cum_count().over("Jockey") - 1)).alias("jky_sr"))
    return rr.select("date", pl.col("HorseID").alias("horse_id"),
                     pl.when(pl.col("jky_sr").is_finite()).then(pl.col("jky_sr")).alias("jky_sr"))


def trainer_table(results_rich: pl.DataFrame) -> pl.DataFrame:
    """Leak-free trailing win strike-rate of the TRAINER of each horse, keyed (date, horse_id).

    The trainer analogue of :func:`jockey_table` — stable form / intent signal (a yard in form,
    a horse placed to win). Strictly-prior expanding strike rate per trainer.
    """
    rr = (results_rich
          .with_columns(pl.col("date").cast(pl.Utf8).alias("date"),
                        (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int8).alias("w"),
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .sort(["Trainer", "d"]))
    rr = rr.with_columns(((pl.col("w").cum_sum().over("Trainer") - pl.col("w"))
                          / (pl.col("w").cum_count().over("Trainer") - 1)).alias("trn_sr"))
    return rr.select("date", pl.col("HorseID").alias("horse_id"),
                     pl.when(pl.col("trn_sr").is_finite()).then(pl.col("trn_sr")).alias("trn_sr"))


def trainer_hot_table(results_rich: pl.DataFrame, half_life: float = 30.0) -> pl.DataFrame:
    """Leak-free RECENCY-weighted trainer strike-rate — the stable's *hot streak* (intent overlay).

    THEORY §2.3/§2.7: trainer intent ("is the yard in form / the horse placed to win today") is a
    pre-race, leak-free signal the market prices. The flat :func:`trainer_table` is the career strike
    rate; this is the sharper RECENCY-weighted analogue — an exp-weighted mean (half-life in *runs*,
    not days) of the trainer's win indicator over its STRICTLY PRIOR runners. A yard suddenly winning
    races runs hotter than its long-run rate. Strictly-prior (the exp mean is shifted by one), keyed
    (date, horse_id). The flat trainer SR adds only ~0.2pp over jockey+rating (priced); the MOMENTUM
    the market reads is the new signal (ST +0.3pp / HV +0.1pp, ECE improves at both venues).
    """
    rr = (results_rich
          .with_columns(pl.col("date").cast(pl.Utf8).alias("date"),
                        (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Float64).alias("w"),
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .sort(["Trainer", "d"]))
    rr = rr.with_columns(
        pl.col("w").ewm_mean(half_life=half_life, ignore_nulls=True).over("Trainer").alias("_ewm"))
    rr = rr.with_columns(pl.col("_ewm").shift(1).over("Trainer").alias("trn_hot"))
    return rr.select("date", pl.col("HorseID").alias("horse_id"),
                     pl.when(pl.col("trn_hot").is_finite()).then(pl.col("trn_hot")).alias("trn_hot"))


def style_table(sectional: pl.DataFrame) -> pl.DataFrame:
    """Leak-free trailing mean ``rel_early`` per horse (front-runner ↑, closer ↓)."""
    s = sectional.with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("d"))
    s = (s.select("d", "venue", "race_no", "runner", "horse_id", "rel_early")
         .sort(["horse_id", "d"])
         .with_columns(((pl.col("rel_early").cum_sum().over("horse_id") - pl.col("rel_early"))
                        / (pl.col("rel_early").cum_count().over("horse_id") - 1)).alias("style"))
         .with_columns(pl.col("d").dt.strftime("%Y-%m-%d").alias("date")))
    return s.select("date", "venue", "race_no", "runner",
                    pl.when(pl.col("style").is_finite()).then(pl.col("style")).alias("style"))


def early_position_table(run_policy: pl.DataFrame) -> pl.DataFrame:
    """Leak-free trailing mean early running POSITION per horse (1 = led early, 0 = settled last).

    From ``run_policy_dataset``'s ``early_pos_norm`` (normalised early field position; winners
    average 0.95 vs the 0.76 field mean, so HIGH = nearer the front early). The strictly-prior
    expanding mean is the horse's *habitual* early position — its running style as a POSITION,
    distinct from the speed-based :func:`style_table` (the two correlate only ~0.12). Feeds the
    positional-fade term: the non-interacting drift-diffusion over-rates habitual front-runners
    (they burn the anaerobic reserve setting the pace with no cover) and under-rates closers — the
    mean-field shadow of the pace equilibrium (front horses win ~1pp LESS than the physics predicts,
    closers ~1pp MORE; ground-truth-confirmed, residual t≈−7 at both venues).
    """
    rp = run_policy.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"))
    rp = rp.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
    rp = (rp.select("date", "d", "venue", "race_no", "runner", "horse_id", "early_pos_norm")
          .sort(["horse_id", "d"])
          .with_columns(((pl.col("early_pos_norm").cum_sum().over("horse_id") - pl.col("early_pos_norm"))
                         / (pl.col("early_pos_norm").cum_count().over("horse_id") - 1)).alias("early_pos")))
    return rp.select("date", "venue", "race_no", "runner",
                     pl.when(pl.col("early_pos").is_finite()).then(pl.col("early_pos")).alias("early_pos"))


def barrier_trial_table(barrier_trials: pl.DataFrame) -> pl.DataFrame:
    """Leak-free trial speed figure per horse, keyed (horse_id, d) for an as-of join.

    A barrier trial is a horse's only on-track form before its DEBUT (and a fitness read on a
    return from a layoff). Same kinematics as :func:`speed_figure_table`: a trial's average
    velocity ``v_trial = distance / trial_time`` (m/s), expressed relative to a condition par
    ``par(going, distance)`` so trials over different trips/going are comparable —

        trial_fig = v_trial − par(going, distance)        (m/s above/below the trial standard)

    Then the strictly-prior **expanding mean** per horse over its trials (sorted by date). Trials
    never share a date with a race for a horse (mornings vs race days), so a backward as-of join
    on ``d`` by ``horse_id`` is leak-free. ``par`` is a condition constant (not the horse's own
    outcome), so using the full panel for it is leak-free. Returns one row per trial; join the
    LATEST prior row to each race via ``join_asof(..., by="horse_id", strategy="backward")``.

    NOTE the join key: the trial data's ``horse_id`` is the long ``HK_YYYY_XXXX`` registration,
    but the model panels key on the SHORT stable brand (e.g. ``L316``), which is the trial data's
    ``horse_brand`` column — so we emit ``horse_brand`` aliased to ``horse_id``.
    """
    bt = barrier_trials.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("trial_time").cast(pl.Float64, strict=False).alias("ttime"),
        pl.col("distance").cast(pl.Float64, strict=False).alias("dist_m"),
        pl.col("going").cast(pl.Utf8).alias("going"))
    bt = bt.filter((pl.col("ttime") > 0) & pl.col("dist_m").is_not_null()
                   & pl.col("horse_brand").is_not_null())
    bt = bt.with_columns((pl.col("dist_m") / pl.col("ttime")).alias("v_trial"))
    bt = bt.with_columns(pl.col("v_trial").mean().over(["going", "dist_m"]).alias("par"))
    bt = bt.with_columns((pl.col("v_trial") - pl.col("par")).alias("tf"),
                         pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                         pl.col("horse_brand").alias("horse_id"))
    bt = bt.sort(["horse_id", "d"]).with_columns(
        (pl.col("tf").cum_sum().over("horse_id") / pl.col("tf").cum_count().over("horse_id"))
        .alias("trial_fig"))
    return bt.select("horse_id", "d", "trial_fig")


def trial_vc_table(barrier_trials: pl.DataFrame) -> pl.DataFrame:
    """Leak-free per-horse trial *critical velocity* on the RACE v_c scale, for precision-fusion.

    This is the substrate for folding barrier trials into the empirical-Bayes v_c posterior at
    the trial's OWN precision (NOT the additive :func:`barrier_trial_table` proxy, and NOT the
    globally-shrunken sectional-v_c residual channel). For each trial we recover a per-horse
    critical velocity on the *same physical scale as race v_c*:

    - the batch's leader velocity profile (``batch_sect`` split times, shared within a trial batch)
      gives a batch critical velocity ``v_c^batch`` via the same :func:`estimate_run_energetics`
      machinery the race energetics use (the settled late-race velocity the field fades to);
    - each horse ran the *same trial* at its own ``trial_time``, so its critical velocity scales
      with its average pace relative to the batch leader: ``v_c = v_c^batch · batch_time/trial_time``.

    The resulting per-horse ``trial_vc`` has mean ≈ the race v_c mean (~16.82 m/s) — it is on the
    race scale by construction, so a single train-only affine map suffices to put it on the
    race-*residual* scale (done in :func:`hkjc.model.fit`). Then the strictly-prior **expanding
    mean** per horse (sorted by date) and the prior trial **count** ``trial_m`` (how many trials
    that mean averages, = the fusion's number of trial pseudo-observations). Leak-free: trials never
    share a date with a race for a horse (mornings vs race days), so a backward as-of join on ``d``
    by ``horse_id`` is safe; the expanding mean is strictly prior within the trial stream too.

    Returns (horse_id, d, trial_vc, trial_m); join the LATEST prior row to each race via
    ``join_asof(..., by="horse_id", strategy="backward")``. Keyed on the SHORT stable brand
    (``horse_brand``), matching the model panels — same join key as :func:`barrier_trial_table`.
    """
    bt = barrier_trials.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("trial_time").cast(pl.Float64, strict=False).alias("ttime"),
        pl.col("batch_time").cast(pl.Float64, strict=False).alias("btime"),
        pl.col("distance").cast(pl.Float64, strict=False).alias("dist_m"))
    bt = bt.filter((pl.col("ttime") > 0) & (pl.col("btime") > 0) & pl.col("dist_m").is_not_null()
                   & pl.col("horse_brand").is_not_null() & pl.col("batch_sect").is_not_null())

    # batch leader critical velocity from its section split profile (same as the race energetics)
    def _batch_vc(sect_s: str | None, dist: float | None) -> float | None:
        try:
            sect = json.loads(sect_s)
        except (TypeError, ValueError):
            return None
        if not sect or len(sect) < 2 or dist is None:
            return None
        try:
            return float(estimate_run_energetics(sect, section_distances(float(dist), len(sect))).v_c)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return None

    rows = bt.select("batch_sect", "dist_m").to_dicts()
    vc_batch = [_batch_vc(r["batch_sect"], r["dist_m"]) for r in rows]
    bt = bt.with_columns(pl.Series("_vc_batch", vc_batch))
    bt = bt.with_columns(
        (pl.col("_vc_batch") * pl.col("btime") / pl.col("ttime")).alias("_tvc"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
        pl.col("horse_brand").alias("horse_id"))
    bt = bt.filter(pl.col("_tvc").is_not_null())
    bt = bt.sort(["horse_id", "d"]).with_columns(
        (pl.col("_tvc").cum_sum().over("horse_id") / pl.col("_tvc").cum_count().over("horse_id"))
        .alias("trial_vc"),
        pl.col("_tvc").cum_count().over("horse_id").cast(pl.Float64).alias("trial_m"))
    return bt.select("horse_id", "d", "trial_vc", "trial_m")


def collateral_form_table(results_rich: pl.DataFrame, kappa: float = 1.0) -> pl.DataFrame:
    """Leak-free **collateral-form** ability rating — opponent-adjusted, in velocity units (m/s).

    THEORY.md §3b: competition is RELATIONAL, not scalar. Races link horses into a directed
    graph of relative velocities (A ran ``s_A`` against a field, that field ran against other
    fields, …). A horse's ability is the fixed point that propagates these collateral links:
    not just *how fast it ran* (that is :func:`speed_figure_table`), but *how fast it ran
    relative to a field whose own strength is known*. Beating a strong field is worth more than
    the same time against a weak one — the opponent-adjusted (Massey) refinement of the raw figure.

    Construction (a Massey-style relative-velocity rating, solved ONLINE so it is strictly
    leak-free). Each run's par-relative velocity is ``s_i = dist/time − par(going,dist)`` (m/s, the
    same condition-normalised kinematic figure as :func:`speed_figure_table`). Races are processed
    in date order; the field strength of race R is the mean *prior* collateral rating ``R̄`` of its
    runners (unrated newcomers count as neutral 0). A runner's observation from race R is

        obs_i = (s_i − mean_R s) + κ·R̄          (its margin vs the field it ran + the field's strength)

    and a horse's rating is the expanding mean of its ``obs`` over its STRICTLY PRIOR runs. The
    ``κ·R̄`` term is the collateral propagation: a margin posted against a strong field inherits
    that strength (one Gauss–Seidel sweep of the Massey least-squares system in date order). κ=1
    is the least-squares value; κ<1 damps the propagation. The rating emitted FOR a race uses only
    runs BEFORE it — the per-horse running mean is updated only AFTER the race is emitted — so it
    cannot see the current or any later result (same strictly-prior guarantee as :func:`form_table`).
    ``par`` is a race-condition constant (not the horse's own outcome), so the full panel is leak-free.

    Returns (date, venue, race_no, runner, collat_form); ``collat_form`` is null on debut.
    """
    rr = results_rich.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("RaceDistance").cast(pl.Float64, strict=False).alias("dist_m"),
        pl.col("FinishTime").cast(pl.Float64, strict=False).alias("ftime"),
        pl.col("Going").cast(pl.Utf8).alias("going"),
        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
        pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("Venue").cast(pl.Utf8).alias("venue"),
    )
    rr = rr.filter((pl.col("ftime") > 0) & pl.col("dist_m").is_not_null())
    rr = rr.with_columns((pl.col("dist_m") / pl.col("ftime")).alias("v_run"))
    rr = rr.with_columns(pl.col("v_run").mean().over(["going", "dist_m"]).alias("par"))
    rr = rr.with_columns(
        (pl.col("v_run") - pl.col("par")).alias("s"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
    rr = rr.with_columns(
        (pl.col("date") + "_" + pl.col("venue") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("rid"))
    rr = rr.sort("d")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    dates_, venues_, raceno_, runner_, val_ = [], [], [], [], []
    for (_rid,), g in rr.group_by("rid", maintain_order=True):
        hs = g["HorseID"].to_list()
        ss = g["s"].to_numpy()
        d0 = g["date"][0]
        v0 = g["venue"][0]
        rn0 = int(g["race_no"][0])
        rns = g["runner"].to_list()
        # strictly-prior rating each horse carries INTO this race (None = no prior run)
        cur = [(sums[h] / counts[h]) if counts.get(h, 0) > 0 else None for h in hs]
        known = [c for c in cur if c is not None]
        field_r = sum(known) / len(known) if known else 0.0
        for rn, c in zip(rns, cur):
            dates_.append(d0)
            venues_.append(v0)
            raceno_.append(rn0)
            runner_.append(rn)
            val_.append(c)
        smean = float(ss.mean())
        # UPDATE the running per-horse mean AFTER emitting -> preserves the strictly-prior guarantee
        for h, si in zip(hs, ss):
            obs = (float(si) - smean) + kappa * field_r
            sums[h] = sums.get(h, 0.0) + obs
            counts[h] = counts.get(h, 0) + 1
    return pl.DataFrame(
        {"date": dates_, "venue": venues_, "race_no": raceno_, "runner": runner_, "collat_form": val_})


def _distance_band_key(dist: float) -> int:
    """Coarse trip band for like-for-like collateral (a margin at 1200m says less about 2000m)."""
    if dist < 1200:
        return 0
    if dist < 1400:
        return 1
    if dist < 1800:
        return 2
    return 3


def collateral_form_table_v2(
    results_rich: pl.DataFrame,
    kappa: float = 1.0,
    half_life: float | None = None,
    margin_pow: float = 1.0,
    asym: float = 1.0,
    n_sweeps: int = 1,
    band_par: bool = False,
) -> pl.DataFrame:
    """REFINEMENT of :func:`collateral_form_table` — same leak-free online Massey rating, with
    four optional refinements of the competition axis, each conversion-tested as a MARGINAL gain
    over the baked one-sweep rating. All are strictly-prior / online in date order (identical
    leak-free discipline as :func:`collateral_form_table`).

    Parameters
    ----------
    kappa : collateral propagation strength on the field-strength term (1.0 = least-squares).
    half_life : if set, the per-horse rating is an EXP-WEIGHTED mean of its ``obs`` (recent
        meetings count more — a stale opponent rating goes off, à la the recency-form lever)
        instead of the flat expanding mean. ``None`` = expanding mean (the baked behaviour).
    margin_pow : nonlinear margin weighting — the within-race margin ``(s_i − mean_R)`` is raised
        to this power (sign-preserving: ``sign(m)·|m|^p``). ``>1`` rewards big margins more,
        ``<1`` compresses. 1.0 = linear (baked).
    asym : strength-of-schedule ASYMMETRY — weight the field-strength inheritance differently for
        runs you BEAT the field mean (margin>0) vs LOST to it (margin<0). The kept field term is
        ``κ·R̄`` for a beat and ``κ·asym·R̄`` for a loss. asym<1 down-weights "quality of fields
        lost to" (losing to a strong field tells you less than beating one). 1.0 = symmetric (baked).
    n_sweeps : DEEPER PROPAGATION. >1 runs additional leak-free Gauss–Seidel refinement passes over
        the STRICTLY-PRIOR race graph before each new race-date is emitted, propagating opponent
        strength further through the collateral chain toward the true Massey fixed point. The extra
        sweeps only ever touch strictly-prior races, so the rating emitted for a race still uses no
        information from it or any later race. 1 = single online sweep (baked).
    band_par : if True, the par velocity (and the margin) is computed within (going, distance-BAND)
        AND the field-strength inheritance is conditioned on the trip band, so a 1200m margin only
        feeds a horse's rating for nearby trips (ability is partly trip-specific).

    Returns (date, venue, race_no, runner, collat_form); null on debut.
    """
    rr = results_rich.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("RaceDistance").cast(pl.Float64, strict=False).alias("dist_m"),
        pl.col("FinishTime").cast(pl.Float64, strict=False).alias("ftime"),
        pl.col("Going").cast(pl.Utf8).alias("going"),
        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
        pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("Venue").cast(pl.Utf8).alias("venue"),
    )
    rr = rr.filter((pl.col("ftime") > 0) & pl.col("dist_m").is_not_null())
    rr = rr.with_columns((pl.col("dist_m") / pl.col("ftime")).alias("v_run"))
    if band_par:
        rr = rr.with_columns(
            pl.col("dist_m").map_elements(_distance_band_key, return_dtype=pl.Int64).alias("band"))
        rr = rr.with_columns(pl.col("v_run").mean().over(["going", "band"]).alias("par"))
    else:
        rr = rr.with_columns(pl.col("v_run").mean().over(["going", "dist_m"]).alias("par"))
        rr = rr.with_columns(pl.lit(0, dtype=pl.Int64).alias("band"))
    rr = rr.with_columns(
        (pl.col("v_run") - pl.col("par")).alias("s"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
    rr = rr.with_columns(
        (pl.col("date") + "_" + pl.col("venue") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("rid"))
    rr = rr.sort("d")

    # per-horse rating state. With band conditioning, a horse carries a rating PER band; the
    # field-strength it inherits is the band-matched prior rating (fallback to its global mean).
    # The running stat is exp-weighted (decay<1) or a flat expanding mean (decay=1, baked).
    band_sums: dict[tuple, float] = {}      # (horse, band) -> running stat numerator
    band_w: dict[tuple, float] = {}          # (horse, band) -> running stat denominator
    glob_sums: dict[str, float] = {}
    glob_w: dict[str, float] = {}
    decay = 1.0 if half_life is None else 0.5 ** (1.0 / half_life)

    # cache every prior race's runner records so deeper sweeps can re-solve the prior graph.
    # each record: (rid, [(horse, s, band)], smean_per_band dict)
    prior_races: list = []

    def _rating(h: str, band: int) -> float | None:
        if band_par:
            k = (h, band)
            if band_w.get(k, 0.0) > 0:
                return band_sums[k] / band_w[k]
            if glob_w.get(h, 0.0) > 0:
                return glob_sums[h] / glob_w[h]
            return None
        if glob_w.get(h, 0.0) > 0:
            return glob_sums[h] / glob_w[h]
        return None

    def _ingest(records, smean_by_band, w_scale: float, field_src=None):
        """Fold one prior race into the running per-horse stats (used by online emit + resweeps).

        ``field_src`` (h, band)->rating maps the field-strength source. ``None`` = the live online
        state (one Gauss–Seidel pass). For a Jacobi fixed-point resweep, pass the PREVIOUS sweep's
        frozen ratings so opponent strength propagates one more hop through the chain.
        """
        if field_src is None:
            known = [r for h, s, b in records if (r := _rating(h, b)) is not None]
        else:
            known = [field_src[(h, b)] for h, s, b in records if (h, b) in field_src]
        field_r = sum(known) / len(known) if known else 0.0
        for h, s, b in records:
            m = s - smean_by_band[b]
            m_w = (1.0 if m >= 0 else -1.0) * (abs(m) ** margin_pow) if margin_pow != 1.0 else m
            kf = kappa * (1.0 if m >= 0 else asym) * field_r
            obs = m_w + kf
            if band_par:
                k = (h, b)
                band_sums[k] = decay * band_sums.get(k, 0.0) + w_scale * obs
                band_w[k] = decay * band_w.get(k, 0.0) + w_scale
            glob_sums[h] = decay * glob_sums.get(h, 0.0) + w_scale * obs
            glob_w[h] = decay * glob_w.get(h, 0.0) + w_scale

    def _snapshot_ratings(races) -> dict:
        """Frozen (horse, band)->rating from the CURRENT state (used as a Jacobi field source)."""
        snap: dict = {}
        for recs, _smb in races:
            for h, _s, b in recs:
                r = _rating(h, b)
                if r is not None:
                    snap[(h, b)] = r
        return snap

    dates_, venues_, raceno_, runner_, val_ = [], [], [], [], []
    cur_date = None
    for (_rid,), g in rr.group_by("rid", maintain_order=True):
        d_today = g["d"][0]
        if n_sweeps > 1 and d_today != cur_date and prior_races:
            # leak-free DEEPER propagation: Jacobi fixed-point sweeps over the STRICTLY-PRIOR graph.
            # each extra sweep recomputes every prior race's obs using the PREVIOUS sweep's frozen
            # ratings as the field strength, so opponent strength propagates one more hop per sweep.
            for _ in range(n_sweeps - 1):
                prev = _snapshot_ratings(prior_races)
                band_sums.clear()
                band_w.clear()
                glob_sums.clear()
                glob_w.clear()
                for recs, smb in prior_races:
                    _ingest(recs, smb, 1.0, field_src=prev)
            cur_date = d_today
        hs = g["HorseID"].to_list()
        ss = g["s"].to_numpy()
        bands = g["band"].to_list()
        d0 = g["date"][0]
        v0 = g["venue"][0]
        rn0 = int(g["race_no"][0])
        rns = g["runner"].to_list()
        for rn, h, b in zip(rns, hs, bands):
            dates_.append(d0)
            venues_.append(v0)
            raceno_.append(rn0)
            runner_.append(rn)
            val_.append(_rating(h, b))
        # per-band within-race mean (the contemporaneous condition normaliser)
        smean_by_band: dict[int, float] = {}
        for b in set(bands):
            sel = [float(ss[i]) for i in range(len(ss)) if bands[i] == b]
            smean_by_band[b] = sum(sel) / len(sel)
        records = [(h, float(si), b) for h, si, b in zip(hs, ss, bands)]
        _ingest(records, smean_by_band, 1.0)
        if n_sweeps > 1:
            prior_races.append((records, smean_by_band))
    return pl.DataFrame(
        {"date": dates_, "venue": venues_, "race_no": raceno_, "runner": runner_, "collat_form": val_})


def trajectory_trait_table(trajectory: pl.DataFrame) -> pl.DataFrame:
    """Leak-free per-horse TRAJECTORY-SHAPE traits from the per-section velocity path (m/s).

    The sectional v_c / speed-figure channels read a run's *average* energetics; the trajectory
    dataset (``trajectory_dataset_full``) carries the per-section velocity profile, which exposes
    the SHAPE of how a horse runs — mechanics the average hides. Two physically-grounded traits,
    each measured per run then made into a strictly-prior per-horse trait:

    - ``traj_sustain`` — **sustained-speed ability**: mean section velocity / peak section velocity
      (∈(0,1], 1 = perfectly flat cruise). Distinct from peak speed and from positional fade.
    - ``traj_fade`` — **fade resistance**: final-section velocity − peak velocity (≤0, 0 = no late
      deceleration). The kinematic signature of the W′ reserve emptying in the closing stages.

    Leak-free construction (same discipline as :func:`speed_figure_table`): each run's raw trait uses
    only that run's own sections; it is within-race standardized (z-score across the race's runners — a
    contemporaneous condition normaliser, not the horse's outcome); the per-horse value is the
    strictly-prior expanding mean of that z-trait over prior runs. Both null on debut.
    """
    df = trajectory.with_columns(_norm_date().alias("date")).sort(
        "date", "venue", "race_no", "runner", "section")
    run = ["date", "venue", "race_no", "runner", "horse_id", "distance"]
    df = df.with_columns(
        pl.col("velocity").max().over(run).alias("_vmax"),
        pl.col("velocity").mean().over(run).alias("_vmean"),
        pl.col("velocity").last().over(run).alias("_vlast"))
    g = (df.group_by(run, maintain_order=True)
         .agg(pl.col("_vmax").first(), pl.col("_vmean").first(), pl.col("_vlast").first()))
    g = g.with_columns(
        (pl.col("_vmean") / pl.col("_vmax")).alias("sustain"),
        (pl.col("_vlast") - pl.col("_vmax")).alias("fade"))
    rid = ["date", "venue", "race_no"]
    for c in ("sustain", "fade"):
        g = g.with_columns(
            ((pl.col(c) - pl.col(c).mean().over(rid))
             / (pl.col(c).std().over(rid) + 1e-9)).alias(c + "_z"))
    g = g.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")).sort(
        "horse_id", "d")
    out_cols = {"sustain_z": "traj_sustain", "fade_z": "traj_fade"}
    for src, dst in out_cols.items():
        g = g.with_columns(
            ((pl.col(src).cum_sum().over("horse_id") - pl.col(src))
             / (pl.col(src).cum_count().over("horse_id") - 1)).alias(dst))
    return g.select(
        "date", "venue", "race_no", "runner",
        *[pl.when(pl.col(dst).is_finite()).then(pl.col(dst)).alias(dst) for dst in out_cols.values()])


def rating_trajectory_table(rating: pl.DataFrame, k: int = 4) -> pl.DataFrame:
    """Leak-free rating-TRAJECTORY (process) features — the career-cycle shape per horse.

    The official handicap rating is a TIME SERIES per horse, not just the latest POINT. Its
    trajectory encodes the career cycle the point misses. From the per-horse dated rating series
    (``rating_dataset``: ``horse_id, date, rtg``), per rating update:

      - ``rtg_peakgap`` — current rtg − the horse's own prior PEAK rtg (≤0): how far below its own
        best. The one trajectory feature that CONVERTS — it carries an ABSOLUTE career landmark the
        point cannot express (a declining horse runs below what its lagging point rating says); see
        ``PEAKGAP_GAIN_BY_VENUE``. ``rtg_vel`` (Δrtg/k) and ``rtg_accel`` did NOT convert — they merely
        restate recent form (collinear with the form channel); ``rtg_nupd`` (maturity) kept for diagnosis.

    LEAK-FREE: each row summarises the rating history up to and *including* that update; the consumer
    as-of-joins it backward with ``allow_exact_matches=False``, so a race on date ``d`` only sees
    updates strictly before ``d`` (the point ``rtg`` for race ``d`` is the separate join already in the
    frame; the trajectory excludes it).
    """
    r = (rating
         .with_columns(pl.col("date").cast(pl.Date).alias("d"), pl.col("rtg").cast(pl.Float64))
         .drop_nulls(["d", "rtg"]).unique(["horse_id", "d"]).sort(["horse_id", "d"]))
    r = r.with_columns(
        pl.col("rtg").shift(k).over("horse_id").alias("_lag_k"),
        pl.col("rtg").shift(2 * k).over("horse_id").alias("_lag_2k"),
        pl.col("rtg").cum_max().over("horse_id").alias("_peak"),
        pl.col("rtg").cum_count().over("horse_id").alias("_nupd"))
    r = r.with_columns(
        ((pl.col("rtg") - pl.col("_lag_k")) / k).alias("rtg_vel"),
        (((pl.col("rtg") - pl.col("_lag_k")) - (pl.col("_lag_k") - pl.col("_lag_2k"))) / k).alias("rtg_accel"),
        (pl.col("rtg") - pl.col("_peak")).alias("rtg_peakgap"),
        pl.col("_nupd").cast(pl.Float64).alias("rtg_nupd"))
    return r.select("horse_id", "d", "rtg_vel", "rtg_accel", "rtg_peakgap", "rtg_nupd")


def attach_rating_trajectory(df: pl.DataFrame, rating: pl.DataFrame, k: int = 4) -> pl.DataFrame:
    """Strict backward as-of join the trajectory features onto a race frame (needs a ``d`` Date col)."""
    traj = rating_trajectory_table(rating, k=k).sort(["horse_id", "d"])
    out = df.sort(["horse_id", "d"]).join_asof(
        traj, on="d", by="horse_id", strategy="backward", allow_exact_matches=False)
    return out.with_columns(
        pl.col("rtg_vel").fill_null(0.0), pl.col("rtg_accel").fill_null(0.0),
        pl.col("rtg_peakgap").fill_null(0.0), pl.col("rtg_nupd").fill_null(0.0))


def speed_figure_table(results_rich: pl.DataFrame) -> pl.DataFrame:
    """Leak-free trailing SPEED FIGURE per horse — a velocity-based ability signal (m/s).

    Unlike rank-based form (which censors the winning margin), this is grounded in kinematics:
    a run's average velocity ``v_run = RaceDistance / FinishTime`` (m/s), expressed relative to
    a condition standard ``par(going, distance)`` so runs across trips/going are comparable:

        speed_fig = v_run − par(going, distance)        (m/s above/below the standard)

    Then the strictly-prior expanding mean per horse — a trait in real velocity units, so it
    couples to v_c with a kinematic coefficient ≈1 rather than a free gain. ``par`` is a
    race-condition constant (not the horse's own outcome), so using the full panel for it is
    leak-free; the per-horse mean is over prior runs only.
    """
    rr = results_rich.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("RaceDistance").cast(pl.Float64, strict=False).alias("dist_m"),
        pl.col("FinishTime").cast(pl.Float64, strict=False).alias("ftime"),
        pl.col("Going").cast(pl.Utf8).alias("going"),
        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
        pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("Venue").cast(pl.Utf8).alias("venue"),
    )
    rr = rr.filter((pl.col("ftime") > 0) & pl.col("dist_m").is_not_null())
    rr = rr.with_columns((pl.col("dist_m") / pl.col("ftime")).alias("v_run"))
    # condition standard: mean race velocity per (going, distance)
    rr = rr.with_columns(pl.col("v_run").mean().over(["going", "dist_m"]).alias("par"))
    rr = rr.with_columns((pl.col("v_run") - pl.col("par")).alias("sf"),
                         pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
    rr = rr.sort(["HorseID", "d"]).with_columns(
        ((pl.col("sf").cum_sum().over("HorseID") - pl.col("sf"))
         / (pl.col("sf").cum_count().over("HorseID") - 1)).alias("speedfig"))
    return rr.select("date", "venue", "race_no", "runner",
                     pl.when(pl.col("speedfig").is_finite()).then(pl.col("speedfig")).alias("speedfig"))
