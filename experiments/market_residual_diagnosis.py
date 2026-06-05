"""Where does the ABM systematically disagree with the market? (goodness-of-fit diagnosis)

The objective is to FIT the market (bet-skill = goodness-of-fit), not beat it. The principled
way to know which mechanism to deepen next is not to guess — it is to run the current ABM, take
the signed residual against the market, and find which observable mechanism PREDICTS that
residual. Wherever the market systematically assigns more (or less) win probability than our
model, the market is pricing a mechanism the model is missing (or over-weighting). That feature
is the next thing to ground in the physics.

  resid_i = market_prob_i - model_prob_i        (both de-vigged + renormalised within race)

A feature with a positive slope on resid => the market gives MORE probability than the ABM when
that feature is high => the ABM UNDER-weights that mechanism. Negative => the ABM over-weights.
We fit one pooled OLS with race-clustered SEs (honest, since rows within a race are dependent).

This reuses the exact leak-free ST walk-forward pipeline from backtest_traffic_st (so the model
probs are out-of-sample), then joins going / class / weight / freshness as candidate mechanisms.

Run: N_SAMPLES=2500 uv run --group bayes python experiments/market_residual_diagnosis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_traffic_st import fit_predict_traffic, load_st
from hkjc.eval.walkforward import walk_forward_folds

CACHE = Path("data_cache")


def collect_oos_preds() -> pl.DataFrame:
    """Run the ST ABM walk-forward and return per-runner OOS (model prob, market prob, won, feats)."""
    df = load_st()
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    parts = []
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if train.height == 0 or test.height == 0:
            continue
        preds = fit_predict_traffic(train, test)
        parts.append(preds)
    return pl.concat(parts)


def main():
    p = collect_oos_preds()
    # de-vig + renormalise both within race so resid is a clean probability difference
    p = p.with_columns(
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("mkt"),
        (pl.col("p") / pl.col("p").sum().over("race_id")).alias("mdl"))
    p = p.with_columns((pl.col("mkt") - pl.col("mdl")).alias("resid"))
    print(f"OOS runners={p.height}, races={p['race_id'].n_unique()}")
    print(f"mean |resid| = {p['resid'].abs().mean():.4f}  (0 = perfect fit to market)\n")

    # join candidate mechanisms from results_rich (going, class, weight, freshness)
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("ActWt").cast(pl.Float64, strict=False).alias("actwt"),
                        pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt"),
                        pl.col("Going").alias("going"), pl.col("RaceClass").alias("rclass"))
          .select("date", "Venue", "RaceNumber", "runner", "actwt", "bodywt", "going", "rclass"))
    p = p.join(rr, left_on=["date", "venue", "race_no", "runner"],
               right_on=["date", "Venue", "RaceNumber", "runner"], how="left")
    p = p.with_columns((pl.col("actwt") / pl.col("bodywt")).alias("burden"))

    # numeric candidate features (each standardised); plus the model inputs to check weighting
    feats = {
        "form": "form ability (model input — check weighting)",
        "rtg": "official rating (handicap mark)",
        "draw": "barrier draw",
        "field_size": "field size",
        "actwt": "carried weight",
        "bodywt": "body weight",
        "burden": "burden ratio actwt/bodywt (handicapper ignores)",
        "style": "running style (trailing rel_early)",
        "v_c_eb": "model critical velocity",
        "w_prime_eb": "model anaerobic reserve",
    }
    feats = {k: v for k, v in feats.items() if k in p.columns}
    rid = p.select("race_id").to_numpy().ravel()
    y = p["resid"].to_numpy()

    def clustered_slope(x):
        m = ~np.isnan(x)
        xz = (x - np.nanmean(x)) / np.nanstd(x)
        xm = np.column_stack([np.ones(m.sum()), xz[m]])
        yy = y[m]
        beta, *_ = np.linalg.lstsq(xm, yy, rcond=None)
        r = yy - xm @ beta
        xtxi = np.linalg.inv(xm.T @ xm)
        cl = rid[m]
        meat = np.zeros((2, 2))
        for g in np.unique(cl):
            idx = cl == g
            s = xm[idx].T @ r[idx]
            meat += np.outer(s, s)
        cov = xtxi @ meat @ xtxi
        return beta[1], np.sqrt(cov[1, 1])

    print("Univariate slope of (market - model) residual on each mechanism (standardised),")
    print("cluster-robust by race. |t|>~2 => the ABM systematically MIS-weights this mechanism.")
    print(f"  {'mechanism':12s} {'slope':>10s} {'se':>9s} {'t':>7s}   interpretation")
    rows = []
    for k in feats:
        b, se = clustered_slope(p[k].to_numpy().astype(float))
        rows.append((k, b, se, b / se))
    for k, b, se, t in sorted(rows, key=lambda r: -abs(r[3])):
        tag = "" if abs(t) < 2 else ("  <-- ABM UNDER-weights" if b > 0 else "  <-- ABM OVER-weights")
        print(f"  {k:12s} {b:>+10.5f} {se:>9.5f} {t:>7.2f}{tag}   {feats[k]}")

    # categorical: is the ABM mis-calibrated by GOING and by CLASS? (mean resid per level)
    print("\nMean residual by GOING (positive => market backs these more than the ABM):")
    gg = (p.filter(pl.col("going").is_not_null()).group_by("going")
          .agg(pl.len().alias("n"), pl.col("resid").mean().alias("mr"))
          .filter(pl.col("n") >= 200).sort("mr"))
    for r in gg.iter_rows(named=True):
        print(f"  {r['going']:<22s} n={r['n']:>5d}  mean resid {r['mr']:+.4f}")
    print("\nMean residual by CLASS:")
    cc = (p.filter(pl.col("rclass").is_not_null()).group_by("rclass")
          .agg(pl.len().alias("n"), pl.col("resid").mean().alias("mr"))
          .filter(pl.col("n") >= 200).sort("mr"))
    for r in cc.iter_rows(named=True):
        print(f"  {r['rclass']:<22s} n={r['n']:>5d}  mean resid {r['mr']:+.4f}")


if __name__ == "__main__":
    main()
