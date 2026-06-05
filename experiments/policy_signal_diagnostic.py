"""Do the trajectory-measured POLICY traits carry ex-ante signal ORTHOGONAL to form?

The gate that `rel_early` failed (POLICY_MAP: running style as a flat covariate double-counts
form and HURTS). Before coupling any trajectory trait into the sim we must show it adds skill
over the EW-form+jockey baseline in the SAME walk-forward conditional logit feature_scan uses.

This is a SCREEN, not the coupling. A trait that adds nothing as a flat covariate might still
matter inside the sim (style is emergent/positional) — but a trait that DOES add flat signal is
a strong candidate, and the sign/magnitude tells us which mechanism to wire. We test each
leak-free trailing policy trait from run_policy_dataset:

  trail_depleted    propensity to run W' empty (fade + late position loss) — a STAMINA/reserve trait
  trail_fade        mean velocity collapse off peak — the graded version of depletion
  trail_effort_dtg  pace style: where the horse spends its reserve (early/front vs late/closer)
  trail_early_pos   running-position style from the position trajectory (vs the rel_early proxy)
  trail_kick_frac   literal kick timing

Run: uv run python experiments/policy_signal_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from feature_scan import build
from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")
TRAITS = ["trail_depleted", "trail_fade", "trail_effort_dtg",
          "trail_early_pos_norm", "trail_kick_frac_obs"]


def main():
    d = build()
    # join the trajectory-measured trailing policy traits on (date, venue, race_no, runner=HorseNo)
    rp = (pl.read_parquet(CACHE / "run_policy_dataset.parquet")
          .select(["date", "venue", "race_no", "runner"] + TRAITS)
          .with_columns(pl.col("date").cast(pl.Utf8)))
    d = d.join(rp, left_on=["date", "Venue", "RaceNumber", "HorseNo"],
               right_on=["date", "venue", "race_no", "runner"], how="left")
    cov = {t: d[t].is_not_null().mean() for t in TRAITS}
    # fill missing with the column mean (field-neutral) and z-score, matching feature_scan
    for t in TRAITS:
        d = d.with_columns(pl.col(t).fill_nan(None).fill_null(d[t].mean()))
        s = d[t].std()
        d = d.with_columns(((pl.col(t) - d[t].mean()) / (s if s else 1.0)).alias(f"z_{t}"))

    dates = sorted(d["date"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races, {len(folds)} folds")
    print("trait coverage:", {k: round(v, 3) for k, v in cov.items()}, "\n")

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(d, folds, fp, date_col="date", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    floor = harness.aggregate(harness.walk_forward_evaluate(
        d.with_columns((1.0 / pl.col("field_size")).alias("uni")), folds,
        lambda tr, te: te.with_columns(pl.col("uni").alias("q")),
        date_col="date", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")
    market = harness.aggregate(harness.walk_forward_evaluate(
        d, folds, lambda tr, te: te.with_columns((pl.col("market_win_prob") /
            pl.col("market_win_prob").sum().over("race_id")).alias("q")),
        date_col="date", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")

    def line(name, ll):
        sk = floor - ll
        print(f"  {name:38s} {ll:>9.4f} {sk:>+9.4f} {sk / (floor - market):>8.0%}")

    base = ["z_form_ew", "z_jky_sr"]
    print(f"  {'model':38s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    line("EW + jockey (baseline)", skill(base))
    for t in TRAITS:
        line(f"  + {t}", skill(base + [f"z_{t}"]))
    line("EW+jky + ALL policy traits", skill(base + [f"z_{t}" for t in TRAITS]))
    line("MARKET (ceiling)", market)


if __name__ == "__main__":
    main()
