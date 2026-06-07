"""ABILITY-READ CHECK (task S1 VERIFICATION) — does the precision-weighted FUSION of the ability
instruments land near the de-confounded cruise LEVEL (and beat the noisy Hill SLOPE) at re-emerging
the stripped FORM meta-signal?

The #55/0050 finding: the old Hill-SLOPE ability read barely correlates with held-out form
(ρ≈+0.066 ST / +0.033 HV) while the de-confounded cruise LEVEL re-emerges form near the ceiling
(ρ≈+0.33/+0.40, journal 0049). S1 replaced the hard priority chain with a precision-weighted FUSION
(``hkjc.model.pure_mechanism.fuse_vc_read`` — the single source of truth). This study MEASURES
whether the fused read lands near the de-confounded level.

A LEAN, leak-free identification check (also the reusable harness S5 extends). For every run with a
path-based ability read we assemble THREE candidate ability reads on the SAME aligned rows:

  * ``vc_hill``   — the Hill critical-velocity SLOPE alone (the old coarse read).
  * ``vc_decon``  — the de-confounded cruise LEVEL alone (the 0049 re-emergence channel).
  * ``vc_fused``  — ``fuse_vc_read`` on the panel columns (the PRODUCTION read).

and correlate each against a single leak-free FORM TARGET, alongside the rating FLOOR and the
trailing-form CEILING. EVERY quantity (ability + form + rating) is STRICTLY PRIOR to the run — both
the ability reads (the panel is strictly-prior by construction) and the form target (computed from
finishing positions PRIOR to each run). The form target is used PURELY as a correlation target,
never an input to any ability read.

FORM TARGET (the stripped meta-signal, as a target only): the horse's strictly-prior
RECENCY-WEIGHTED relative finish — ``rf = 1 - (finish_pos-1)/(field_size-1)`` (a normalised finish
in [0,1], 1=win, 0=last), exponentially-weighted with half-life 2 races (the production
FORM_HALF_LIFE=2.0), then SHIFTED one race back so a run's own result never enters its target. This
is exactly the production ``recency_form_table`` signal, recomputed here from ``results_rich`` so the
study has no production-frame dependency.

TRAILING-FORM CEILING: a DIFFERENT prior-form window — the strictly-prior expanding CAREER MEAN of
the same ``rf`` (flat, all prior races, shifted one back). ρ(trailing_form, form_target) is the
AUTOCORRELATION of form (how well form predicts itself across two windows) — the ceiling any
ability read aspires to.

RATING FLOOR: the horse's strictly-prior official handicap rating (``rating_dataset.parquet``,
as-of the race date). ρ(rating, form_target) is the floor a real ability read must clear.

    PYTHONHASHSEED=0 uv run --group bayes python studies/ability_read_check.py

NO market odds, NO win optimization, NO meta-signal as an input. Read-only on src/hkjc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.model.pure_mechanism import build_pure_mechanism_panel, fuse_vc_read

CACHE = Path("data_cache")
FORM_HALF_LIFE = 2.0  # production FORM_HALF_LIFE (races); the recency target's exp half-life.


def _norm_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise the run key (date as YYYY-MM-DD string, race_no/runner ints) for joining."""
    return df.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("race_no").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("runner").cast(pl.Int64, strict=False).alias("runner"),
    )


def form_targets(results_rich: pl.DataFrame, venue: str) -> pl.DataFrame:
    """Two STRICTLY-PRIOR form windows per run, from finishing positions PRIOR to the run only.

      * ``form``          — recency-weighted (exp half-life 2 races) relative-finish, shift(1).
      * ``trailing_form`` — expanding CAREER-MEAN relative-finish (a different prior window), shift(1).

    ``rf = 1 - (finish_pos-1)/(field_size-1)`` ∈ [0,1] (1 = win). Both are shifted one race back so a
    run's OWN finish never enters its own form value (leak-free). Keyed (date, venue, race_no, runner).
    """
    fs = (results_rich.group_by("Date", "Venue", "RaceNumber")
          .agg(pl.len().alias("field_size")))
    rr = (results_rich.filter(pl.col("Venue") == venue)
          .join(fs, on=["Date", "Venue", "RaceNumber"], how="left")
          .with_columns(
              pl.col("Date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
              pl.col("Venue").cast(pl.Utf8).alias("venue"),
              pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
              pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
              pl.col("HorseID").cast(pl.Utf8).alias("horse_id"),
              pl.col("Pla").cast(pl.Float64, strict=False).alias("finish_pos"))
          .filter(pl.col("finish_pos").is_finite() & (pl.col("field_size") > 1)))
    rr = rr.with_columns(
        (1.0 - (pl.col("finish_pos") - 1.0) / (pl.col("field_size") - 1.0)).alias("rf"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_d"))
    rr = rr.sort(["horse_id", "_d", "race_no", "runner"])
    # recency-weighted (exp half-life) — the production form signal, then shift one race back.
    rr = rr.with_columns(
        pl.col("rf").ewm_mean(half_life=FORM_HALF_LIFE, ignore_nulls=True)
        .over("horse_id").alias("_ewm"),
        # expanding career mean of PRIOR runs (a different window): (cumsum-rf)/(cumcount-1).
        (pl.col("rf").cum_sum().over("horse_id") - pl.col("rf")).alias("_csum_prior"),
        (pl.col("rf").cum_count().over("horse_id") - 1).alias("_cn_prior"))
    rr = rr.with_columns(
        pl.col("_ewm").shift(1).over("horse_id").alias("form"),
        pl.when(pl.col("_cn_prior") > 0)
        .then(pl.col("_csum_prior") / pl.col("_cn_prior")).alias("trailing_form"))
    # rf_now = the run's OWN realised relative-finish — the 0050 headline-(a) target (used there as the
    # consequence the prior ability must predict). Reported as a SECOND target for baseline comparison.
    return rr.select("date", "venue", "race_no", "runner", "horse_id",
                     "form", "trailing_form", pl.col("rf").alias("rf_now"))


def rating_floor(venue: str) -> pl.DataFrame:
    """Strictly-prior official handicap rating per run, joined as-of by (horse_id, date).

    ``rating_dataset.parquet`` is the pre-race official rating (a handicapper's read assigned BEFORE
    the race) — the population/handicap FLOOR a path-based ability read must clear. As-of join
    (latest rating on or before the race date) keeps it strictly prior. Venue-agnostic table; the
    venue filter is applied via the run key downstream.
    """
    del venue
    return (pl.read_parquet(CACHE / "rating_dataset.parquet")
            .with_columns(pl.col("date").cast(pl.Date).alias("_rdate"),
                          pl.col("rtg").cast(pl.Float64).alias("rtg"),
                          pl.col("horse_id").cast(pl.Utf8))
            .select("horse_id", "_rdate", "rtg").sort(["horse_id", "_rdate"]))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_venue(venue: str) -> dict:
    traj = (pl.read_parquet(CACHE / "trajectory_dataset_full.parquet")
            .filter(pl.col("venue") == venue))
    rr = pl.read_parquet(CACHE / "results_rich.parquet")

    # (1) strictly-prior path physiology panel (the single source of truth for the ability reads).
    panel = _norm_keys(build_pure_mechanism_panel(traj, rr))

    # (2) the three ability reads on the SAME rows. vc_fused = the production fuse_vc_read; for the
    #     population term we have no per-run vc_pop in this lean panel (the debut fallback is fit
    #     train-only inside the gate) so we pass NaN — fuse_vc_read then weights Hill vs de-confounded
    #     by precision, exactly the production read for a runner with a path-based ability.
    n = panel.height
    nan = np.full(n, np.nan)
    vc_fused = fuse_vc_read(
        panel["vc_hill"].to_numpy(), panel["vc_hill_se"].to_numpy(),
        panel["vc_decon"].to_numpy(), panel["vc_decon_n"].to_numpy(),
        panel["vc_decon_sd"].to_numpy(), nan)
    panel = panel.with_columns(pl.Series("vc_fused", vc_fused, dtype=pl.Float64))

    # (3) the leak-free form targets + the rating floor.
    forms = _norm_keys(form_targets(rr, venue))
    df = panel.join(
        forms.select("date", "venue", "race_no", "runner", "form", "trailing_form", "rf_now"),
        on=["date", "venue", "race_no", "runner"], how="left")
    # as-of rating: latest official rating on or before the race date (strictly prior).
    rtg = rating_floor(venue)
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_d"))
    df = df.sort(["horse_id", "_d"]).join_asof(
        rtg.rename({"_rdate": "_d"}), on="_d", by="horse_id", strategy="backward")

    # A run has a "path-based ability read" iff Hill OR de-confounded is available (vc_fused is then
    # a genuine instrument fusion, not the population fallback). Align ALL correlations on the SAME
    # rows: a path-based read AND a defined form target.
    aligned = df.filter(
        (pl.col("vc_hill").is_finite() | pl.col("vc_decon").is_finite())
        & pl.col("form").is_finite())

    def block(target_col: str) -> dict:
        """All five correlations against one form TARGET, on the SAME aligned rows."""
        t = aligned[target_col].to_numpy()
        return {
            "r_fused": _pearson(aligned["vc_fused"].to_numpy(), t),
            "r_hill": _pearson(aligned["vc_hill"].to_numpy(), t),
            "r_decon": _pearson(aligned["vc_decon"].to_numpy(), t),
            "r_rating_floor": _pearson(aligned["rtg"].to_numpy(), t),
            "r_form_ceiling": _pearson(aligned["trailing_form"].to_numpy(), t),
        }

    return {
        "venue": venue,
        "n": aligned.height,
        "n_hill": int(aligned["vc_hill"].is_finite().sum()),
        "n_decon": int(aligned["vc_decon"].is_finite().sum()),
        # PRIMARY (task spec): target = strictly-prior recency-form; ceiling = form autocorrelation.
        "prior": block("form"),
        # BASELINE-COMPARABLE (0050 headline-a): target = the run's own realised finish rf_now.
        "rfnow": block("rf_now"),
    }


def _print_table(rows: list[dict], key: str) -> None:
    hdr = (f"  {'venue':5s} {'N':>6s} {'r_fused':>9s} {'r_hill':>9s} {'r_decon':>9s} "
           f"{'r_floor':>9s} {'r_ceiling':>10s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        b = r[key]
        print(f"  {r['venue']:5s} {r['n']:6d} {b['r_fused']:+9.3f} {b['r_hill']:+9.3f} "
              f"{b['r_decon']:+9.3f} {b['r_rating_floor']:+9.3f} {b['r_form_ceiling']:+10.3f}")


def main() -> None:
    rows = [run_venue(v) for v in ("ST", "HV")]
    print(f"\n{'='*86}")
    print("ABILITY-READ CHECK — ρ(ability read, FORM target) on the SAME aligned path rows")
    print(f"{'='*86}")
    print("  rows = runs with a path-based ability read (Hill OR de-confounded) AND a defined form")
    print("  coverage: " + "; ".join(
        f"{r['venue']} N={r['n']} (Hill {r['n_hill']}, decon {r['n_decon']})" for r in rows))

    print("\n  TABLE A — TARGET = strictly-prior RECENCY-FORM (exp half-life 2, shift 1) [task spec]")
    print("           (ceiling = career-mean prior form → the AUTOCORRELATION of form)")
    _print_table(rows, "prior")

    print("\n  TABLE B — TARGET = the run's OWN realised finish rf_now [0050 headline-(a), baseline-comparable]")
    print("           (ceiling = prior recency-form → realised finish, as in 0050)")
    _print_table(rows, "rfnow")
    print("\n  0050 baselines (TABLE B comparable): Hill +0.066/+0.033 (ST/HV); decon-level ≈ +0.33/+0.40;")
    print("                  rating floor +0.015/−0.022; form ceiling +0.382/+0.302")

    for r in rows:
        b = r["rfnow"]
        beats_hill = b["r_fused"] > b["r_hill"]
        near_decon = abs(b["r_fused"] - b["r_decon"]) <= 0.05
        print(f"\n  [{r['venue']}] (vs rf_now) fused {'BEATS' if beats_hill else 'does NOT beat'} Hill "
              f"({b['r_fused']:+.3f} vs {b['r_hill']:+.3f}); "
              f"{'LANDS NEAR' if near_decon else 'is OFF'} de-confounded "
              f"({b['r_fused']:+.3f} vs {b['r_decon']:+.3f}).")


if __name__ == "__main__":
    main()
