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
