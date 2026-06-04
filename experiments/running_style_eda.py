"""Is there emergent PACE signal beyond form? Cheap ceiling test before sim surgery.

Running style = each horse's leak-free trailing rel_early (tendency to go early). Pace is
emergent: a lone front-runner controls a slow pace (advantage); a closer cashes in when
the field is full of speed (hot pace collapses, it picks up the pieces). We test whether
pace-FIT features — own style vs the FIELD's style — add win-skill beyond form, via the
same walk-forward conditional logit used for the other ability signals.

  style_i        = trailing mean rel_early over horse i's PRIOR runs (leak-free)
  field_speed    = mean style of the OTHER runners (how much early pressure i faces)
  lone_speed     = style_i − max(other styles)   (>0 ⇒ uncontested leader)
  pace_fit       = style_i × (style_i − field_speed)  (lead-when-fast / close-when-slow)

Run: uv run python experiments/running_style_eda.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")


def _norm(col):
    return pl.col(col).cast(pl.Utf8).str.replace_all("/", "-")


def build() -> pl.DataFrame:
    # sectional date is DD/MM/YYYY, wpq is YYYY/MM/DD — parse both to a real Date and join on it
    s = (pl.read_parquet(CACHE / "sectional_dataset.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("d")))
    w = (pl.read_parquet(CACHE / "wpq_dataset.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y/%m/%d", strict=False).alias("d")))
    keys = ["d", "venue", "race_no", "runner"]
    h = (s.select(keys + ["horse_id", "rel_early", "rel_finish", "fade"])
         .join(w.select(keys + ["finish_pos", "field_size", "won"]), on=keys)
         .with_columns(pl.col("d").cast(pl.Utf8).alias("date"),
                       (1.0 - (pl.col("finish_pos") - 1) / (pl.col("field_size") - 1)).alias("rf"))
         .sort(["horse_id", "d"]))
    # leak-free trailing traits (expanding mean over strictly-prior runs)
    def trail(col):
        return ((pl.col(col).cum_sum().over("horse_id") - pl.col(col))
                / (pl.col(col).cum_count().over("horse_id") - 1))
    h = h.with_columns(trail("rel_early").alias("style"), trail("rf").alias("form"))
    h = h.filter(pl.col("style").is_finite() & pl.col("form").is_finite())
    # within-race pace context (uses only trailing traits, so still leak-free)
    rid = ["date", "venue", "race_no"]
    h = h.with_columns(
        ((pl.col("style").sum().over(rid) - pl.col("style")) / (pl.col("field_size") - 1)).alias("field_speed"),
        pl.col("style").max().over(rid).alias("_smax"),
        pl.col("style").sort(descending=True).head(2).last().over(rid).alias("_s2nd"))
    h = h.with_columns(
        (pl.col("style") - pl.when(pl.col("style") == pl.col("_smax"))
         .then(pl.col("_s2nd")).otherwise(pl.col("_smax"))).alias("lone_speed"),
        (pl.col("style") * (pl.col("style") - pl.col("field_speed"))).alias("pace_fit"),
        (pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"))
    # z-score features for a well-conditioned logit
    for c in ["form", "style", "field_speed", "lone_speed", "pace_fit"]:
        h = h.with_columns(((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(f"z_{c}"))
    return h


def main():
    h = build()
    print(f"{h.height} runs, {h['race_id'].n_unique()} races with trailing style+form\n")
    # trait stability: does prior style predict this race's early speed?
    corr = np.corrcoef(h["style"].to_numpy(), h["rel_early"].to_numpy())[0, 1]
    print(f"running-style trait stability (trailing style vs realised rel_early): r={corr:.3f}\n")

    folds = walk_forward_folds(sorted(h["date"].unique().to_list()), n_folds=4, embargo=1, min_train=6)

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(h, folds, fp, date_col="date", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    rows = [
        ("form only", ["z_form"]),
        ("form + style", ["z_form", "z_style"]),
        ("form + lone_speed", ["z_form", "z_lone_speed"]),
        ("form + pace_fit", ["z_form", "z_pace_fit"]),
        ("form + field_speed", ["z_form", "z_field_speed"]),
        ("form + all pace", ["z_form", "z_style", "z_lone_speed", "z_pace_fit", "z_field_speed"]),
    ]
    base = skill(["z_form"])
    print(f"{'features':28s} {'race-win log-loss':>17s} {'Δ vs form-only':>15s}")
    for name, feats in rows:
        ll = skill(feats)
        print(f"{name:28s} {ll:>17.4f} {base - ll:>+15.4f}")
    print("\n(Δ>0 ⇒ the pace feature ADDS win-skill on top of form.)")


if __name__ == "__main__":
    main()
