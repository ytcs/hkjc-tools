"""How close to the market can a well-featurized flexible model get on PUBLIC data alone?

The standing target is >=90% of market. We've found the signals that survive walk-forward: EW
(recency) form (+8pp over career in a logit), jockey strike rate (+7pp), trainer (small). This
stacks them in one conditional logit on the full 3240-race results_rich set and reports market
share — a public-data benchmark independent of the ABM (whose energetics machinery needs the
smaller sectional subset). If this rivals or beats the ABM's ~38%, feature richness is a path
worth as much as mechanistic structure; the two are complementary (ABM adds traffic/pace).

Run: uv run python experiments/best_logit.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")


def build() -> pl.DataFrame:
    d = (pl.read_parquet(CACHE / "results_rich.parquet")
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("Pla").cast(pl.Int64, strict=False),
                       pl.col("WinOdds").cast(pl.Float64, strict=False))
         .drop_nulls(["Pla", "WinOdds"]))
    rid = ["date", "Venue", "RaceNumber"]
    d = d.with_columns(
        pl.len().over(rid).alias("field_size"),
        (pl.col("Pla") == 1).cast(pl.Int8).alias("won"),
        (1.0 / pl.col("WinOdds")).alias("mraw"))
    d = d.with_columns(
        (pl.col("mraw") / pl.col("mraw").sum().over(rid)).alias("market_win_prob"),
        (1.0 - (pl.col("Pla") - 1) / (pl.col("field_size") - 1)).alias("rf"),
        (pl.col("date") + "_" + pl.col("Venue") + "_" + pl.col("RaceNumber").cast(pl.Utf8)).alias("race_id"))

    def trail_mean(col, key):
        return ((pl.col(col).cum_sum().over(key) - pl.col(col))
                / (pl.col(col).cum_count().over(key) - 1))

    d = d.sort(["HorseID", "d"]).with_columns(
        trail_mean("rf", "HorseID").alias("form_career"),
        pl.col("rf").shift(1).ewm_mean(half_life=2.0, ignore_nulls=True).over("HorseID").alias("form_ew"))
    d = d.sort(["Jockey", "d"]).with_columns(trail_mean("won", "Jockey").alias("jky_sr"))
    d = d.sort(["Trainer", "d"]).with_columns(trail_mean("won", "Trainer").alias("trn_sr"))

    gj = float(d["won"].mean())
    d = d.with_columns(pl.col(c).fill_nan(None) for c in ["form_career", "form_ew", "jky_sr", "trn_sr"])
    d = d.with_columns(pl.col("form_career").fill_null(0.5), pl.col("form_ew").fill_null(0.5),
                       pl.col("jky_sr").fill_null(gj), pl.col("trn_sr").fill_null(gj))
    for c in ["form_career", "form_ew", "jky_sr", "trn_sr"]:
        s = d[c].std()
        d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))
    return d


def main():
    d = build()
    dates = sorted(d["date"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races, {len(dates)} meetings\n")

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
        print(f"  {name:30s} {ll:>9.4f} {sk:>+9.4f} {sk/(floor-market):>8.0%}")

    print(f"  {'model':30s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    line("uniform (floor)", floor)
    line("career form", skill(["z_form_career"]))
    line("EW form", skill(["z_form_ew"]))
    line("EW + jockey", skill(["z_form_ew", "z_jky_sr"]))
    line("EW + jockey + trainer", skill(["z_form_ew", "z_jky_sr", "z_trn_sr"]))
    line("career + jockey + trainer", skill(["z_form_career", "z_jky_sr", "z_trn_sr"]))
    line("MARKET (ceiling)", market)


if __name__ == "__main__":
    main()
