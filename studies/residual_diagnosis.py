"""Market-residual diagnosis — the modelling agent's core instrument (re-diagnose step).

Run the ABM walk-forward, take the signed residual against the market, and find which
observable mechanism PREDICTS that residual. A positive slope on resid = market gives MORE
probability than the model when that feature is high => the ABM UNDER-weights that mechanism;
negative => OVER-weights. One pooled OLS per mechanism with race-clustered SEs (honest, since
rows within a race are dependent). The biggest |t| is the next thing to ground in the physics.

    resid_i = market_prob_i - model_prob_i        (both de-vigged + renormalised within race)

Run: uv run --group bayes python studies/residual_diagnosis.py
     BEND=0.0030 uv run --group bayes python studies/residual_diagnosis.py   # at a trial constant
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load
from hkjc.backtesting import walk_forward_folds
from hkjc.model.fit import fit_predict_traffic

CACHE = Path("data_cache")


def collect_oos(df, jky_mean, bend, wpg, sim, ab) -> pl.DataFrame:
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    parts = []
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if train.height == 0 or test.height == 0:
            continue
        parts.append(fit_predict_traffic(train, test, jky_mean=jky_mean, bend=bend,
                                         w_prime_gain=wpg, sim=sim, allometric_b=ab))
    return pl.concat(parts)


def clustered_slope(x, y, rid):
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


def main():
    bend = os.environ.get("BEND")
    bend = float(bend) if bend is not None else None
    wpg = float(os.environ.get("W_PRIME_GAIN", "2.0"))
    sim = os.environ.get("SIM", "kinematic")
    ab = float(os.environ.get("ALLOMETRIC_B", "1.0"))
    df, jky_mean = load("ST")
    p = collect_oos(df, jky_mean, bend, wpg, sim, ab)
    p = p.with_columns(
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("mkt"),
        (pl.col("p") / pl.col("p").sum().over("race_id")).alias("mdl"))
    p = p.with_columns((pl.col("mkt") - pl.col("mdl")).alias("resid"))
    print(f"bend={bend if bend is not None else 'default(0.0037)'}  W_PRIME_GAIN={wpg}  "
          f"SIM={sim} ALLOMETRIC_B={ab}  OOS runners={p.height}, races={p['race_id'].n_unique()}")
    print(f"mean |resid| = {p['resid'].abs().mean():.4f}  (0 = perfect fit to market)\n")

    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("ActWt").cast(pl.Float64, strict=False).alias("actwt"),
                        pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt"))
          .select("date", "Venue", "RaceNumber", "runner", "actwt", "bodywt"))
    p = p.join(rr, left_on=["date", "venue", "race_no", "runner"],
               right_on=["date", "Venue", "RaceNumber", "runner"], how="left")

    feats = {"form": "form ability", "rtg": "official rating", "draw": "barrier draw",
             "field_size": "field size", "actwt": "carried weight", "bodywt": "body weight",
             "v_c_eb": "model critical velocity", "w_prime_eb": "model anaerobic reserve"}
    feats = {k: v for k, v in feats.items() if k in p.columns}
    rid = p.select("race_id").to_numpy().ravel()
    y = p["resid"].to_numpy()
    rows = [(k, *clustered_slope(p[k].to_numpy().astype(float), y, rid)) for k in feats]
    print("  mechanism      slope        se       t   interpretation")
    for k, b, se in sorted(rows, key=lambda r: -abs(r[1] / r[2])):
        t = b / se
        tag = "" if abs(t) < 2 else ("  <- UNDER-weights" if b > 0 else "  <- OVER-weights")
        print(f"  {k:12s} {b:>+10.5f} {se:>9.5f} {t:>7.2f}{tag}   {feats[k]}")


if __name__ == "__main__":
    main()
