"""Coverage report for the barrier-trial backfill vs. the energetics race panel."""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BT = ROOT / "data_cache" / "barrier_trials.parquet"
PANEL = ROOT / "data_cache" / "energetics_dataset_full.parquet"


def main():
    bt = pl.read_parquet(BT)
    panel = pl.read_parquet(PANEL)

    print("=== BARRIER-TRIAL PANEL ===")
    print(f"rows={bt.height}  date range {bt['date'].min()} .. {bt['date'].max()}")
    print(f"unique horses (brand)={bt['horse_brand'].n_unique()}  trial days={bt['date'].n_unique()}")
    null_brand = bt.filter(pl.col("horse_brand").is_null()).height
    null_tt = bt.filter(pl.col("trial_time").is_null()).height
    null_lbw = bt.filter(pl.col("lbw_lengths").is_null()).height
    print(f"parsing gaps: null horse_brand={null_brand}  null trial_time={null_tt} "
          f"({100*null_tt/bt.height:.1f}%)  null lbw_lengths={null_lbw} ({100*null_lbw/bt.height:.1f}%)")

    # cadence: trial days per ISO week
    wk = (bt.select("date").unique()
          .with_columns(pl.col("date").str.to_date().dt.year().alias("y"),
                        pl.col("date").str.to_date().dt.week().alias("w")))
    n_weeks = wk.select(["y", "w"]).unique().height
    print(f"cadence: {bt['date'].n_unique()} trial days over {n_weeks} weeks "
          f"= {bt['date'].n_unique()/n_weeks:.2f} trial days/week")
    dow = (bt.select("date").unique()
           .with_columns(pl.col("date").str.to_date().dt.to_string("%a").alias("dow")))
    print("trial-day weekday dist:", dict(sorted(dow["dow"].value_counts().iter_rows())))

    # trials per horse
    per = bt.group_by("horse_brand").agg(pl.col("date").n_unique().alias("n"))
    print(f"trials per horse: median={per['n'].median()}  mean={per['n'].mean():.2f}  max={per['n'].max()}")

    print("\n=== trial distance dist ===")
    print(bt.group_by("distance").len().sort("distance"))
    print("=== trial going dist ===")
    print(bt.group_by("going").len().sort("len", descending=True))

    # ---- debutant coverage ----
    print("\n=== DEBUTANT COVERAGE ===")
    panel = panel.with_columns(pl.col("date").str.to_date().alias("d"))
    first_race = (panel.group_by("horse_id")
                  .agg(pl.col("d").min().alias("first_race_date")))
    print(f"panel horses={first_race.height}")

    bt2 = bt.with_columns(pl.col("date").str.to_date().alias("trial_d"))
    # earliest trial per brand
    first_trial = bt2.group_by("horse_brand").agg(pl.col("trial_d").min().alias("first_trial_date"))

    # join panel-first-race against trials (brand == horse_id)
    joined = first_race.join(first_trial, left_on="horse_id", right_on="horse_brand", how="left")
    # has any prior trial strictly before first race
    has_prior = joined.filter(
        pl.col("first_trial_date").is_not_null()
        & (pl.col("first_trial_date") < pl.col("first_race_date"))
    )
    # any trial at all (regardless of timing) for brand match
    any_trial = joined.filter(pl.col("first_trial_date").is_not_null())
    n = first_race.height
    print(f"panel horses with ANY brand-matched trial in BT panel: {any_trial.height} "
          f"({100*any_trial.height/n:.1f}%)")
    print(f"panel horses with >=1 trial BEFORE their first race (true debut prior): "
          f"{has_prior.height} ({100*has_prior.height/n:.1f}%)")

    # restrict to horses whose first race is comfortably inside our trial window
    win_lo = bt2["trial_d"].min()
    inwin = first_race.filter(pl.col("first_race_date") >= win_lo)
    inwin_j = inwin.join(first_trial, left_on="horse_id", right_on="horse_brand", how="left")
    inwin_prior = inwin_j.filter(
        pl.col("first_trial_date").is_not_null()
        & (pl.col("first_trial_date") < pl.col("first_race_date"))
    )
    print(f"\nrestricting to panel horses whose FIRST race is on/after {win_lo} "
          f"(trial window covers their debut lead-in): {inwin.height} horses")
    print(f"  of those, with >=1 prior trial: {inwin_prior.height} "
          f"({100*inwin_prior.height/inwin.height:.1f}%)")

    # how many panel horses overlap BT brands at all
    panel_brands = set(panel["horse_id"].unique().to_list())
    bt_brands = set(bt["horse_brand"].unique().to_list())
    print(f"\nbrand overlap: panel={len(panel_brands)} BT={len(bt_brands)} "
          f"intersection={len(panel_brands & bt_brands)}")


if __name__ == "__main__":
    main()
