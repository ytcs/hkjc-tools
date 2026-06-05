"""Does the MARKET misprice the pace/field interaction? (epistemic-arbitrage test)

Every flat covariate is exhausted (form+jockey 41% of market; draw/rating/sectional/policy/weight
each ~0 on top). The remaining edge — if any — must be STRUCTURAL: an interaction a flat model
cannot represent and the philosophy says statistical extrapolators under-price. The one large,
replicating mechanism (POLICY_MAP #2/#6) is PACE CANNIBALISATION on the finite W' reserve:

  - many lead-contesters  => they burn each other early, all fade => front-runners UNDERperform
  - a lone front-runner   => uncontested easy lead, banks W'     => it OVERperforms

The disciplined test does NOT try to out-predict ability (the market wins that). It accepts the
market's win prob as the ability estimate and asks whether the RESIDUAL (won - market_prob) is
SIGNED by the field's pace configuration — i.e. whether the market fails to fully price the
interaction the EOM predicts. If yes, that residual is the trade (fade hot-pace speed, back the
lone speed and the closers behind a hot pace).

Running style is the leak-free trailing early running-position from the TRAJECTORY data (the
policy OUTPUT, not the confounded rel_early speed proxy). Pace pressure = how many genuine
front-runners are entered in the race.

Run: uv run python experiments/pace_pressure_mispricing.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("data_cache")
FRONT_Q = 0.33   # a "front-runner" sits in the front third of the field early, on trailing style


def main():
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("WinOdds").cast(pl.Float64, strict=False),
                        (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int8).alias("won")))
    rid = ["date", "Venue", "RaceNumber"]
    rr = rr.with_columns(pl.len().over(rid).alias("field"),
                         (1.0 / pl.col("WinOdds")).alias("mraw"))
    rr = rr.with_columns((pl.col("mraw") / pl.col("mraw").sum().over(rid)).alias("mkt"))

    # leak-free trailing running-style from the trajectory: low early_pos_norm = habitual front-runner.
    rp = (pl.read_parquet(CACHE / "run_policy_dataset.parquet")
          .select("date", "venue", "race_no", "runner", "trail_early_pos_norm")
          .with_columns(pl.col("date").cast(pl.Utf8)))
    df = rr.join(rp, left_on=["date", "Venue", "RaceNumber", "runner"],
                 right_on=["date", "venue", "race_no", "runner"], how="left")
    # a horse with no trailing style yet is unknown — exclude from the front-runner count (NOT
    # assumed mid): keep its row for outcome accounting but mark style missing.
    df = df.with_columns(pl.col("trail_early_pos_norm").alias("style"))

    # per race: flag front-runners (style in the front third) and count them = pace pressure.
    df = df.with_columns((pl.col("style") < FRONT_Q).fill_null(False).alias("is_front"))
    df = df.with_columns(pl.col("is_front").sum().over(rid).alias("n_front"))

    known = df.filter(pl.col("style").is_not_null())
    print(f"{known.height} runners with trailing style, {known.select(rid).unique().height} races")
    print(f"pace-pressure (n_front per race) distribution: "
          f"{dict(sorted(df.select(rid + ['n_front']).unique()['n_front'].value_counts().iter_rows()))}\n")

    # residual vs market, by role x pace config. positive resid = wins MORE than market expects.
    known = known.with_columns((pl.col("won") - pl.col("mkt")).alias("resid"))

    def bucket(name, sub):
        n = sub.height
        if n < 30:
            print(f"  {name:42s} n={n:>5d}   (too few)")
            return
        r = sub["resid"].to_numpy()
        m, se = r.mean(), r.std() / np.sqrt(n)
        # also express as a ratio: actual wins / market-expected wins
        wr = sub["won"].mean() / sub["mkt"].mean()
        print(f"  {name:42s} n={n:>5d}   resid {m:+.4f} ± {se:.4f}   win/mkt {wr:.2f}")

    print("LONE SPEED (exactly 1 front-runner in the race):")
    bucket("  the lone front-runner", known.filter((pl.col("n_front") == 1) & pl.col("is_front")))
    bucket("  closers behind a lone leader", known.filter((pl.col("n_front") == 1) & ~pl.col("is_front")))

    print("\nHOT PACE (>=3 front-runners contesting the lead):")
    bucket("  a front-runner in the speed duel", known.filter((pl.col("n_front") >= 3) & pl.col("is_front")))
    bucket("  closers behind a hot pace", known.filter((pl.col("n_front") >= 3) & ~pl.col("is_front")))

    print("\nMODERATE PACE (exactly 2 front-runners):")
    bucket("  front-runner (2-way duel)", known.filter((pl.col("n_front") == 2) & pl.col("is_front")))
    bucket("  closers behind 2", known.filter((pl.col("n_front") == 2) & ~pl.col("is_front")))

    print("\nSanity — front-runners overall vs closers overall (should be ~0 if market efficient):")
    bucket("  all front-runners", known.filter(pl.col("is_front")))
    bucket("  all non-front", known.filter(~pl.col("is_front")))


if __name__ == "__main__":
    main()
