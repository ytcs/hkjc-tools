"""Walk-forward backtest + pace test at HAPPY VALLEY — where geometry says position pays.

The ST pace coupling failed because form already encodes the early-speed edge on a
galloping track. The geometry litmus says HV is different: tight bends + a ~310 m run-in
make position far more decisive (steeper pace AND draw slopes), so running-style may carry
MARGINAL signal there that relative-finish form captures less completely. This reuses the
exact leak-free form/style/energetics machinery from the ST backtest, swapping in HV
geometry and the calibrated HV bend curvature, then sweeps PACE_GAIN with a paired
bootstrap. (HV rail/Course history wasn't scraped, so course defaults to A — a limitation.)

Run: uv run python experiments/backtest_traffic_hv.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.calibration import (CALIBRATED_BEND_CURVATURE, distance_band,
                              fit_energetics_population)
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.traffic_jax import win_probs
from hkjc.track import track_for

import backtest_traffic_st as bt
from backtest_traffic_st import (CACHE, _horse_residuals, _norm, current_form_table,
                                 running_style_table)

BEND = CALIBRATED_BEND_CURVATURE["HV"]
bt.N_SAMPLES = 4000


def load_hv() -> pl.DataFrame:
    e = pl.read_parquet(CACHE / "energetics_dataset.parquet").with_columns(_norm("date").alias("date"))
    w = pl.read_parquet(CACHE / "wpq_dataset.parquet").with_columns(_norm("date").alias("date"))
    r = (pl.read_parquet(CACHE / "rating_dataset.parquet")
         .with_columns(_norm("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("rtg").cast(pl.Float64))
         .drop_nulls(["d", "rtg"]).sort(["horse_id", "d"]))
    keys = ["date", "venue", "race_no", "runner"]
    df = (e.select(keys + ["horse_id", "v_c", "w_prime", "distance"])
          .join(w.select(keys + ["draw", "won", "placed", "finish_pos", "field_size", "market_win_prob"]), on=keys)
          .filter(pl.col("venue") == "HV")
          .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .sort(["horse_id", "d"]))
    df = df.join_asof(r.select("horse_id", "d", "rtg"), on="d", by="horse_id", strategy="backward")
    df = df.with_columns(pl.col("rtg").fill_null(float(r["rtg"].median())))
    df = df.join(current_form_table(), on=keys, how="left").with_columns(pl.col("form").fill_null(0.5))
    df = df.join(running_style_table(), on=keys, how="left")
    df = df.with_columns(pl.col("style").fill_null(pl.col("style").mean()))
    return df.with_columns((pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"))


def fit_predict_hv(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    pop = fit_energetics_population(train, rating_col="rtg")
    res = _horse_residuals(train, pop)
    lam_vc = lambda n: pop.v_c.tau2 / (pop.v_c.tau2 + pop.v_c.sigma2 / n) if n > 0 else 0.0   # noqa: E731
    lam_w = lambda n: pop.w_prime.tau2 / (pop.w_prime.tau2 + pop.w_prime.sigma2 / n) if n > 0 else 0.0  # noqa: E731
    preds = []
    for race_id, g in test.group_by("race_id"):
        g = g.sort("runner")
        dist = int(g["distance"][0])
        rtg = g["rtg"].to_numpy().astype(float)
        form = g["form"].to_numpy().astype(float)
        band = distance_band(dist)
        vc, wp = [], []
        for i, h in enumerate(g["horse_id"].to_list()):
            n, u_vc, u_w = res.get(h, (0, 0.0, 0.0))
            vc.append(pop.v_c.fixed(band, rtg[i]) + lam_vc(n) * u_vc + bt.FORM_GAIN * (form[i] - 0.5))
            wp.append(pop.w_prime.fixed(band, rtg[i]) + lam_w(n) * u_w)
        geom = track_for("HV", dist, "turf")
        style = g["style"].to_numpy().astype(float)
        sd = style.std()
        pace_bias = (style - style.mean()) / sd if sd > 1e-9 else np.zeros_like(style)
        p = win_probs(np.array(vc), np.array(wp), g["draw"].to_numpy().astype(float), dist,
                      track=geom, course="A", size=bt.N_SAMPLES, bend_curvature=BEND,
                      speed_noise=bt.SPEED_NOISE, pace_bias=pace_bias, pace_gain=bt.PACE_GAIN,
                      seed=abs(hash(race_id[0])) % (2**31))
        preds.append(g.with_columns(pl.Series("p", p)))
    return pl.concat(preds)


def per_race_loss(pred: pl.DataFrame) -> dict:
    out = {}
    for rid, g in pred.group_by("race_id"):
        p = np.clip(g["p"].to_numpy().astype(float), 1e-12, None)
        p = p / p.sum()
        won = g["won"].to_numpy().astype(bool)
        out[rid[0]] = -np.log(p[won][0]) if won.any() else np.nan
    return out


def run(df, folds, gain):
    bt.PACE_GAIN = gain
    loss = {}
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if not test.is_empty():
            loss.update(per_race_loss(fit_predict_hv(train, test)))
    return loss


def market_uniform(df, folds):
    mk, un = {}, {}
    for fold in folds:
        test = df.filter(pl.col("date").is_in(fold.test))
        for rid, g in test.group_by("race_id"):
            won = g["won"].to_numpy().astype(bool)
            if not won.any():
                continue
            m = g["market_win_prob"].to_numpy().astype(float)
            m = np.clip(m / m.sum(), 1e-12, 1)
            mk[rid[0]] = -np.log(m[won][0])
            un[rid[0]] = -np.log(1.0 / len(won))
    return mk, un


def main():
    df = load_hv()
    folds = walk_forward_folds(sorted(df["date"].unique().to_list()), n_folds=4, embargo=1, min_train=2)
    print(f"Happy Valley: {df['race_id'].n_unique()} races, {df.height} runners, "
          f"{df['date'].n_unique()} dates, {len(folds)} folds  (bend={BEND})\n")

    mk, un = market_uniform(df, folds)
    base = run(df, folds, 0.0)
    keys0 = sorted(set(base) & set(mk))
    b0 = np.array([base[k] for k in keys0]); m0 = np.array([mk[k] for k in keys0])
    u0 = np.array([un[k] for k in keys0])
    fin = np.isfinite(b0) & np.isfinite(m0)
    print(f"  pace-off:  traffic {b0[fin].mean():.4f}   market {m0[fin].mean():.4f}   "
          f"uniform {u0[fin].mean():.4f}   skill {u0[fin].mean()-b0[fin].mean():+.4f} "
          f"({(u0[fin].mean()-b0[fin].mean())/(u0[fin].mean()-m0[fin].mean()):.0%} of market)\n")

    rng = np.random.default_rng(0)
    print(f"{'PACE_GAIN':>10s} {'mean LL':>9s} {'Δ vs off':>9s} {'95% CI':>20s} {'P(helps)':>9s}")
    print(f"{0.0:>10.2f} {np.nanmean(list(base.values())):>9.4f} {'—':>9s} {'—':>20s} {'—':>9s}")
    for gain in (0.03, 0.06, 0.10):
        on = run(df, folds, gain)
        keys = sorted(set(base) & set(on))
        b = np.array([base[k] for k in keys]); o = np.array([on[k] for k in keys])
        msk = np.isfinite(b) & np.isfinite(o)
        diff = b[msk] - o[msk]
        boot = np.array([rng.choice(diff, diff.size, replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{gain:>10.2f} {o[msk].mean():>9.4f} {diff.mean():>+9.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {(boot>0).mean():>8.1%}")


if __name__ == "__main__":
    main()
