"""Fast dev backtest — subsampled walk-forward % of market for rapid mechanism iteration.

Trades precision for speed (fewer folds, subsampled test races, fewer samples) so a mechanism
can be tried in ~1-2 min instead of ~10. Use during development; re-confirm a winner at full
scale (studies/backtest_st.py) before baking a constant.

Philosophy: a mechanism DERIVED from first-principles physics is a priori the right coupling, so
this quick check is mostly a "does it move the needle / not hurt" gate (parsimony), not a
statistical proof. Reusable: load once, add your feature column, call quick_eval(df, jky, **fit_kw).

    from quick_eval import quick_eval
    df, jky = load("ST")
    base = quick_eval(df, jky)                       # baseline
    cand = quick_eval(df, jky, allometric_b=2.0)     # a candidate mechanism
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load  # noqa: F401  (re-exported for convenience)
from hkjc.backtesting import walk_forward_folds
from hkjc.backtesting.scoring import race_win_log_loss
from hkjc.model.fit import fit_predict_traffic


def quick_eval(df, jky_mean, *, n_folds: int = 3, test_cap: int = 150, n_samples: int = 1500,
               seed: int = 7, **fit_kw) -> dict:
    """Subsampled walk-forward. Returns {pct_market, winner_logp, model_ll, n_races}."""
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=n_folds, embargo=1, min_train=1)
    rng = np.random.default_rng(seed)
    mrows, krows, urows, winlp, nrace = [], [], [], [], 0
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if train.height == 0 or test.height == 0:
            continue
        rids = test["race_id"].unique().to_list()
        if len(rids) > test_cap:
            keep = set(rng.choice(rids, size=test_cap, replace=False).tolist())
            test = test.filter(pl.col("race_id").is_in(list(keep)))
        preds = fit_predict_traffic(train, test, jky_mean=jky_mean, n_samples=n_samples, **fit_kw)
        nrace += preds["race_id"].n_unique()
        winlp.append(preds.filter(pl.col("won") == 1)["p"].log().mean())
        mrows.append(preds.select("race_id", "won", pl.col("p").alias("q")))
        krows.append(test.with_columns(
            (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("q"))
            .select("race_id", "won", "q"))
        urows.append(test.with_columns((1.0 / pl.col("field_size")).alias("q"))
                     .select("race_id", "won", "q"))

    def ll(rows):
        return race_win_log_loss(pl.concat(rows), prob_col="q", winner_col="won", race_col="race_id")
    m, k, u = ll(mrows), ll(krows), ll(urows)
    return {"pct_market": (u - m) / (u - k), "winner_logp": float(np.mean(winlp)),
            "model_ll": m, "n_races": nrace}


def main():
    df, jky = load("ST")
    r = quick_eval(df, jky)
    print(f"baseline (kinematic, current constants): {r['pct_market']:.1%} of market "
          f"| winner-logp {r['winner_logp']:.3f} | {r['n_races']} races")


if __name__ == "__main__":
    main()
