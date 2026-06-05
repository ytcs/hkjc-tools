"""Leak-free ability features for the ABM — the strongest signals, single-sourced.

These are the per-horse trait measurements the production model rides on, each computed
**leak-free**: a runner's value is the mean/rate over its STRICTLY PRIOR runs only
(expanding mean shifted by one), so a backtest can never see a horse's current or future
result. They are pure transforms over already-loaded frames (no I/O), so the same code
serves the walk-forward backtest and a live race card.

- :func:`form_table` — expanding mean of prior relative-finish. Form is the dominant
  ability signal (≈18% of market skill) and survives the handicap.
- :func:`jockey_table` — trailing win strike-rate of the booked jockey (≈+7pp).
- :func:`style_table` — trailing mean ``rel_early`` (front-runner ↑ / closer ↓); the
  pace-policy input (dormant by default, PACE_GAIN=0 — see constants).

Recency note: an exp-weighted form is +8pp in a bare conditional logit but does NOT
transfer to the ABM (the recency the market prices is already carried by the official
rating, which enters nonlinearly via ``v_c.fixed(band, rtg)``), so the career mean stays.
"""
from __future__ import annotations

import polars as pl


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
