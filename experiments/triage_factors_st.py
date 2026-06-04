"""Cheap triage: which omitted factors hold the most residual info vs our model?

For each candidate factor we measure (a) its correlation with the structural model's
residual ``won − p_traffic`` and (b) the univariate drop in binary log-loss when it
is added on top of ``log p_traffic`` — i.e. information the model is *missing*. We
repeat (b) over ``log p_market`` as an edge check (info the *market* misses; expected
~0). This is an in-sample, univariate triage to PRIORITISE the mechanistic build —
not a walk-forward backtest. Jockey/trainer use leave-one-out strike rates.

Run: uv run python experiments/triage_factors_st.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

CACHE = Path("data_cache")


def _loo_rate(df, key, win="won"):
    """Leave-one-out win rate per group `key`, shrunk to the global mean (k=20)."""
    g = df.group_by(key).agg(pl.col(win).sum().alias("w"), pl.len().alias("n"))
    d = df.join(g, on=key, how="left")
    mu = float(df[win].mean())
    w, n, y = d["w"].to_numpy(), d["n"].to_numpy(), d[win].to_numpy()
    k = 20.0
    return (w - y + k * mu) / (n - 1 + k)


def _ll(beta, x, y):
    z = x @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def _fit_ll(x, y):
    b0 = np.zeros(x.shape[1])
    r = minimize(_ll, b0, args=(x, y), method="L-BFGS-B")
    return _ll(r.x, x, y)


def incremental(base_logit, factor, y):
    """Δ binary log-loss from adding standardised `factor` on top of base_logit."""
    m = ~np.isnan(factor)
    yv, bl, f = y[m], base_logit[m], factor[m]
    f = (f - f.mean()) / (f.std() + 1e-9)
    ones = np.ones(m.sum())
    base = _fit_ll(np.c_[ones, bl], yv)
    full = _fit_ll(np.c_[ones, bl, f], yv)
    return base - full, int(m.sum())


def main():
    panel = pl.read_parquet(CACHE / "oos_panel.parquet").with_columns(pl.col("date").cast(pl.Utf8).alias("d"))
    form = pl.read_parquet(CACHE / "form_dataset.parquet").with_columns(
        pl.col("HorseNo").alias("horse_id"), pl.col("Date").cast(pl.Utf8).alias("d"))

    # leave-one-out jockey / trainer strike rates from the form panel
    form = form.with_columns(pl.Series("jky_wr", _loo_rate(form, "Jockey")),
                             pl.Series("trn_wr", _loo_rate(form, "Trainer")))
    # body-weight delta per horse (this run minus previous), days-since-run, recent form
    form = form.sort(["horse_id", "d"]).with_columns(
        (pl.col("DeclarHorseWt") - pl.col("DeclarHorseWt").shift(1).over("horse_id")).alias("wt_delta"))
    fcols = ["jky_wr", "trn_wr", "wt_delta", "DeclarHorseWt", "days_since_last",
             "win_rate_prior", "avg_finish_prior3"]
    df = panel.join(form.select(["horse_id", "d", *fcols]), on=["horse_id", "d"], how="left")

    y = df["won"].to_numpy().astype(float)
    lt = np.log(np.clip(df["p_traffic"].to_numpy(), 1e-6, 1))
    lm = np.log(np.clip(df["p_market"].to_numpy(), 1e-6, 1))
    resid_model = y - df["p_traffic"].to_numpy()


    # candidates: panel-native + joined form factors
    cand = {
        "draw (in model)": df["draw"].to_numpy().astype(float),
        "v_c_eb (in model)": df["v_c_eb"].to_numpy(),
        "rating": df["rtg"].to_numpy(),
        "jockey_wr": df["jky_wr"].to_numpy(),
        "trainer_wr": df["trn_wr"].to_numpy(),
        "body_wt": df["DeclarHorseWt"].to_numpy().astype(float),
        "wt_delta": df["wt_delta"].to_numpy().astype(float),
        "days_since_run": df["days_since_last"].to_numpy().astype(float),
        "win_rate_prior": df["win_rate_prior"].to_numpy().astype(float),
        "avg_finish_prior3": df["avg_finish_prior3"].to_numpy().astype(float),
    }

    print(f"panel {df.height} runners; factor coverage varies (form join ~56%)")
    print(f"{'factor':20s} {'cov':>5s} {'corr_resid_model':>17s} {'dLL/model':>10s} {'dLL/market':>11s}")
    rows = []
    for name, f in cand.items():
        m = ~np.isnan(f)
        cr = float(np.corrcoef(f[m], resid_model[m])[0, 1])
        dlm, nobs = incremental(lt, f, y)
        dlk, _ = incremental(lm, f, y)
        rows.append((name, nobs, cr, dlm, dlk))
    for name, nobs, cr, dlm, dlk in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:20s} {nobs:5d} {cr:>17.3f} {dlm*1000:>9.2f}m {dlk*1000:>10.2f}m")
    print("\n(dLL in millinats/runner; over-model = info our structural model misses; "
          "over-market ~0 expected. In-sample univariate triage, not a backtest.)")


if __name__ == "__main__":
    main()
