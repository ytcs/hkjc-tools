"""Is the bodywt residual real mass, or a class/condition proxy? (queue #1 investigation)

The market backs heavier horses MORE than the ABM (bodywt residual t=+8.6, POSITIVE — so it's
body mass as a power/condition signal, not the locomotion-drag effect). Before coupling mass as
a mechanism we must rule out that it's just a proxy for class or the handicap mark. Decisive
tests, all on the same OOS residual:

  1. MULTIVARIATE cluster-robust OLS: resid ~ bodywt + rtg + actwt + field_size + class-dummies.
     If bodywt's PARTIAL t survives controlling for rating + class, it carries ability info
     beyond the handicap mark (a real, priceable mass signal). If it collapses, it's a proxy.
  2. BINNED: mean resid by bodywt quantile (monotone => real; driven by extremes => fragile).
  3. WITHIN-CLASS: does the bodywt slope survive inside a single class?

Run at the baked constants. Run: uv run --group bayes python studies/mass_residual_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load
from hkjc.backtesting import walk_forward_folds
from hkjc.model.fit import fit_predict_traffic

CACHE = Path("data_cache")


def collect_oos(df, jky_mean) -> pl.DataFrame:
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    parts = []
    for fold in folds:
        tr = df.filter(pl.col("date").is_in(fold.train))
        te = df.filter(pl.col("date").is_in(fold.test))
        if tr.height and te.height:
            parts.append(fit_predict_traffic(tr, te, jky_mean=jky_mean))
    return pl.concat(parts)


def cluster_ols(xm, y, rid, names):
    """Multivariate OLS with race-clustered SEs. Returns list of (name, beta, se, t)."""
    beta, *_ = np.linalg.lstsq(xm, y, rcond=None)
    r = y - xm @ beta
    xtxi = np.linalg.inv(xm.T @ xm)
    k = xm.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(rid):
        idx = rid == g
        s = xm[idx].T @ r[idx]
        meat += np.outer(s, s)
    cov = xtxi @ meat @ xtxi
    se = np.sqrt(np.diag(cov))
    return [(names[j], beta[j], se[j], beta[j] / se[j]) for j in range(k)]


def _z(s: pl.Series) -> np.ndarray:
    x = s.to_numpy().astype(float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def main():
    df, jky_mean = load("ST")
    p = collect_oos(df, jky_mean)
    p = p.with_columns(
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("mkt"),
        (pl.col("p") / pl.col("p").sum().over("race_id")).alias("mdl"))
    p = p.with_columns((pl.col("mkt") - pl.col("mdl")).alias("resid"))

    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("ActWt").cast(pl.Float64, strict=False).alias("actwt"),
                        pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt"),
                        pl.col("RaceClass").cast(pl.Utf8).alias("rclass"))
          .select("date", "Venue", "RaceNumber", "runner", "actwt", "bodywt", "rclass"))
    p = p.join(rr, left_on=["date", "venue", "race_no", "runner"],
               right_on=["date", "Venue", "RaceNumber", "runner"], how="left")
    p = p.drop_nulls(["bodywt", "actwt", "rtg", "rclass"])
    print(f"OOS runners with full covariates = {p.height}, races={p['race_id'].n_unique()}\n")

    y = p["resid"].to_numpy()
    rid = p["race_id"].to_numpy()

    # 1) univariate bodywt (reference)
    uni = cluster_ols(np.column_stack([np.ones(p.height), _z(p["bodywt"])]), y, rid, ["const", "bodywt"])
    print(f"univariate bodywt:                t = {uni[1][3]:+.2f}")

    # 2) multivariate: control for rtg, actwt, field_size, and class dummies
    classes = sorted(p["rclass"].unique().to_list())[1:]   # drop first as baseline
    cols = [np.ones(p.height), _z(p["bodywt"]), _z(p["rtg"]), _z(p["actwt"]), _z(p["field_size"])]
    names = ["const", "bodywt", "rtg", "actwt", "field_size"]
    for c in classes:
        cols.append((p["rclass"] == c).to_numpy().astype(float))
        names.append(f"cls:{c}")
    res = cluster_ols(np.column_stack(cols), y, rid, names)
    bw = next(r for r in res if r[0] == "bodywt")
    print(f"multivariate bodywt (control rtg+actwt+field+class): t = {bw[3]:+.2f}  "
          f"(slope {bw[1]:+.5f})")
    print("  -> survives (|t|>~4) => real mass signal beyond the mark; collapses => class proxy\n")

    # 3) binned mean resid by bodywt quantile
    pb = p.with_columns(((pl.col("bodywt").rank() / p.height * 5).ceil().clip(1, 5)).alias("q"))
    print("mean resid by bodywt quintile (positive = market backs more than the ABM):")
    for r in pb.group_by("q").agg(pl.len().alias("n"), pl.col("bodywt").mean().alias("kg"),
                                  pl.col("resid").mean().alias("mr")).sort("q").iter_rows(named=True):
        print(f"  Q{int(r['q'])}  n={r['n']:>5d}  ~{r['kg']:.0f}kg   mean resid {r['mr']:+.4f}")

    # 4) within-class slope (largest class)
    big = p["rclass"].value_counts().sort("count", descending=True)["rclass"][0]
    q = p.filter(pl.col("rclass") == big)
    wc = cluster_ols(np.column_stack([np.ones(q.height), _z(q["bodywt"])]),
                     q["resid"].to_numpy(), q["race_id"].to_numpy(), ["const", "bodywt"])
    print(f"\nwithin-class bodywt slope ({big}, n={q.height}): t = {wc[1][3]:+.2f}")


if __name__ == "__main__":
    main()
