"""Re-calibrate the ABM's ability spread to FIT the market (FORM_GAIN x SPEED_NOISE sweep).

The market-residual diagnosis showed the ABM is UNDER-CONFIDENT on ability: every ability axis
(form t=+32, v_c t=+21, w' t=+5.6, rtg t=+2.8) has the market putting MORE probability on strong
horses than the model. The model maps form -> effective v_c via FORM_GAIN, then flattens the
finishing order with SPEED_NOISE race-day noise. Under-confidence => FORM_GAIN too small and/or
SPEED_NOISE too large. Both are real mechanism-calibration constants, so re-fitting them against
OUTCOMES (not against the market) is legitimate — we measure OOS % of market = race-win skill /
market skill, the goodness-of-fit metric.

Speed: the sim is the bottleneck (~10 min full walk-forward), so we SUBSAMPLE test races per fold
(train stays full — the energetics population fit needs it). The relative ordering across the grid
is robust to subsampling; the winner is re-confirmed at full scale separately.

Run: uv run --group bayes python experiments/ability_spread_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import backtest_traffic_st as bt
from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds

TEST_CAP = 180          # test races sampled per fold (train is always full)
SEED = 7
FORM_GAINS = [2.0, 4.0, 6.0, 8.0]
SPEED_NOISES = [0.12, 0.09, 0.06]


def evaluate(df, folds, rng) -> tuple[float, float, float]:
    """Return (race_win_log_loss for model, market, uniform) over subsampled test races."""
    model_rows, mkt_rows, uni_rows = [], [], []
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if train.height == 0 or test.height == 0:
            continue
        rids = test["race_id"].unique().to_list()
        if len(rids) > TEST_CAP:
            keep = set(rng.choice(rids, size=TEST_CAP, replace=False).tolist())
            test = test.filter(pl.col("race_id").is_in(list(keep)))
        preds = bt.fit_predict_traffic(train, test)
        model_rows.append(preds.select("race_id", "won", pl.col("p").alias("q")))
        mkt_rows.append(test.with_columns(
            (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("q"))
            .select("race_id", "won", "q"))
        uni_rows.append(test.with_columns((1.0 / pl.col("field_size")).alias("q"))
                        .select("race_id", "won", "q"))

    def ll(rows):
        d = pl.concat(rows)
        return harness.calibration.race_win_log_loss(d, prob_col="q", winner_col="won", race_col="race_id")
    return ll(model_rows), ll(mkt_rows), ll(uni_rows)


def main():
    df = bt.load_st()
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"ST {df['race_id'].n_unique()} races; sweep on {TEST_CAP}/fold test subsample, N={bt.N_SAMPLES}")
    print(f"baseline FORM_GAIN={bt.FORM_GAIN}, SPEED_NOISE={bt.SPEED_NOISE}\n")
    print(f"  {'FORM_GAIN':>9s} {'SPEED_NOISE':>11s} {'model_LL':>9s} {'%market':>8s}")

    best = None
    for sn in SPEED_NOISES:
        for fg in FORM_GAINS:
            bt.FORM_GAIN, bt.SPEED_NOISE = fg, sn
            rng = np.random.default_rng(SEED)               # same subsample across configs
            m, mkt, uni = evaluate(df, folds, rng)
            pct = (uni - m) / (uni - mkt)
            star = ""
            if best is None or pct > best[0]:
                best = (pct, fg, sn)
                star = "  <-- best"
            print(f"  {fg:>9.1f} {sn:>11.2f} {m:>9.4f} {pct:>7.0%}{star}")
    print(f"\nbest: FORM_GAIN={best[1]}, SPEED_NOISE={best[2]}  ->  {best[0]:.0%} of market "
          f"(baseline was ~44%). Re-confirm at full scale before baking.")


if __name__ == "__main__":
    main()
