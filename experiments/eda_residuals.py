"""EDA on the Sha Tin traffic-model residuals — learn where the structure helps.

Builds the leak-free out-of-sample panel (traffic vs market predictions + every
feature), then runs a battery: calibration, disagreement (who is right when the
model and market diverge), the decisive **incremental-edge** test (does log(traffic)
lower out-of-sample log-loss *conditional on* log(market)?), residual draw
efficiency, energetics signal, and slices by distance/field. Saves key plots.

Run: uv run --group bayes --group dev python experiments/eda_residuals.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hkjc.eval import harness  # noqa: E402
from hkjc.eval.walkforward import walk_forward_folds  # noqa: E402
from hkjc.model.conditional_logit import conditional_logit_fit_predict  # noqa: E402

from backtest_traffic_st import (fit_predict_market, fit_predict_traffic,  # noqa: E402
                                 load_st)

CACHE = Path("data_cache")
FIG = Path("data_cache/figs")


def build_panel() -> pl.DataFrame:
    """Leak-free OOS panel: every test race scored by traffic + market, with features."""
    df = load_st()
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    rows = []
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if not train.height or not test.height:
            continue
        tr = fit_predict_traffic(train, test).rename({"p": "p_traffic"})
        mk = fit_predict_market(train, test).select("race_id", "runner", "p").rename({"p": "p_market"})
        rows.append(tr.join(mk, on=["race_id", "runner"]))
    panel = pl.concat(rows)
    panel = panel.with_columns(
        (pl.col("p_market") / pl.col("p_market").sum().over("race_id")).alias("p_market"),
        (pl.col("p_traffic") / pl.col("p_traffic").sum().over("race_id")).alias("p_traffic"))
    panel = panel.with_columns(
        pl.col("p_traffic").log().alias("log_traffic"),
        pl.col("p_market").log().alias("log_market"),
        (pl.col("p_traffic").log() - pl.col("p_market").log()).alias("logratio"),
        (1.0 / pl.col("field_size")).alias("p_uniform"))
    panel.write_parquet(CACHE / "oos_panel.parquet")
    return panel


def _reliability(p, y, bins=8):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e9, 1e9
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return np.array(out)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    y = panel["won"].to_numpy()
    print(f"OOS panel: {panel.height} runners, {panel['race_id'].n_unique()} races, "
          f"base win rate {y.mean():.3f}")

    # ---- A. calibration --------------------------------------------------
    section("A. Calibration (reliability) — predicted vs observed win rate")
    for name in ("p_traffic", "p_market", "p_uniform"):
        rel = _reliability(panel[name].to_numpy(), y)
        print(f"  {name:9s}: " + "  ".join(f"{pp:.3f}->{oo:.3f}" for pp, oo, _ in rel))

    # ---- B. disagreement: who's right --------------------------------------
    section("B. Disagreement — bucket by log(traffic/market); realised win rate")
    pq = panel.with_columns(
        pl.col("logratio").qcut(5, labels=[str(i) for i in range(5)]).alias("q"))
    g = pq.group_by("q").agg(
        pl.len().alias("n"), pl.col("logratio").mean().alias("logratio"),
        pl.col("p_market").mean().alias("p_market"), pl.col("p_traffic").mean().alias("p_traffic"),
        pl.col("won").mean().alias("win_rate")).sort("q")
    print("  q  logratio  p_market  p_traffic  win_rate  n   (model>market as q rises)")
    for r in g.iter_rows(named=True):
        print(f"  {r['q']}  {r['logratio']:+.3f}    {r['p_market']:.3f}     "
              f"{r['p_traffic']:.3f}     {r['win_rate']:.3f}   {r['n']}")

    # ---- C. INCREMENTAL EDGE: does traffic add to market? ------------------
    section("C. Incremental edge — walk-forward conditional logit OOS race-win log-loss")
    folds = walk_forward_folds(panel["date"].unique().to_list(), n_folds=4, embargo=1, min_train=1)
    feature_sets = {
        "market only": ["log_market"],
        "market+traffic": ["log_market", "log_traffic"],
        "market+draw": ["log_market", "draw"],
        "market+v_c": ["log_market", "v_c_eb"],
        "traffic only": ["log_traffic"],
    }
    base = None
    for name, feats in feature_sets.items():
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"])
        fm = harness.walk_forward_evaluate(panel, folds, fp, date_col="date",
                                           prob_col="p", outcome_col="won", race_col="race_id")
        ll = harness.aggregate(fm).get("race_win_log_loss", float("nan"))
        if base is None:
            base = ll
        print(f"  {name:16s} log-loss={ll:.4f}   Δ vs market-only={base - ll:+.4f}")

    # ---- D. residual draw efficiency --------------------------------------
    section("D. Draw — empirical win rate vs market-implied, and the residual")
    dr = panel.with_columns((pl.col("won") - pl.col("p_market")).alias("resid"))
    dg = dr.group_by("draw").agg(
        pl.len().alias("n"), pl.col("won").mean().alias("emp_win"),
        pl.col("p_market").mean().alias("mkt_win"), pl.col("p_traffic").mean().alias("trf_win"),
        pl.col("resid").mean().alias("mkt_resid")).sort("draw").filter(pl.col("n") >= 20)
    print("  draw  emp_win  mkt_win  trf_win  mkt_resid  n")
    for r in dg.iter_rows(named=True):
        print(f"   {r['draw']:2d}   {r['emp_win']:.3f}    {r['mkt_win']:.3f}    "
              f"{r['trf_win']:.3f}    {r['mkt_resid']:+.3f}    {r['n']}")
    d, res = dr["draw"].to_numpy().astype(float), dr["resid"].to_numpy()
    sl, ic = np.polyfit(d, res, 1)
    print(f"  market residual vs draw: slope={sl:+.4f}/draw  (≈0 => market prices draw)")

    # ---- E. energetics signal ---------------------------------------------
    section("E. Energetics — does v_c_eb predict finishing, conditional on market?")
    cc = np.corrcoef(panel["v_c_eb"].to_numpy(), panel["finish_pos"].to_numpy())[0, 1]
    print(f"  corr(v_c_eb, finish_pos) = {cc:+.3f}  (neg => higher v_c finishes nearer the front)")
    eq = panel.with_columns([
        pl.col("p_market").qcut(3, labels=["lo", "mid", "hi"], allow_duplicates=True).alias("mq"),
        pl.col("v_c_eb").qcut(2, labels=["v_lo", "v_hi"], allow_duplicates=True).alias("vq")])
    et = eq.group_by(["mq", "vq"]).agg(pl.col("won").mean().alias("win"), pl.len().alias("n")).sort(["mq", "vq"])
    print("  within market tercile, win rate by v_c half:")
    for r in et.iter_rows(named=True):
        print(f"    market={r['mq']:3s} v_c={r['vq']:4s}  win={r['win']:.3f}  n={r['n']}")

    # ---- F. slices --------------------------------------------------------
    section("F. Skill by slice — race-win log-loss (lower better)")
    panel2 = panel.with_columns(pl.when(pl.col("distance") <= 1200).then(pl.lit("sprint"))
                                .when(pl.col("distance") <= 1650).then(pl.lit("mile"))
                                .otherwise(pl.lit("route")).alias("trip"))
    for col in ("trip", "field_size"):
        print(f"  by {col}:")
        for key in sorted(panel2[col].unique().to_list(), key=str):
            sub = panel2.filter(pl.col(col) == key)
            if sub["race_id"].n_unique() < 5:
                continue
            llt = harness.score_predictions(sub, prob_col="p_traffic", race_col="race_id")["race_win_log_loss"]
            llm = harness.score_predictions(sub, prob_col="p_market", race_col="race_id")["race_win_log_loss"]
            llu = harness.score_predictions(sub, prob_col="p_uniform", race_col="race_id")["race_win_log_loss"]
            print(f"    {col}={str(key):7s} races={sub['race_id'].n_unique():3d}  "
                  f"traffic={llt:.3f} market={llm:.3f} uniform={llu:.3f}  trf-skill={llu-llt:+.3f}")

    # ---- plots ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, c in (("p_market", "C0"), ("p_traffic", "C1"), ("p_uniform", "C7")):
        rel = _reliability(panel[name].to_numpy(), y)
        ax[0].plot(rel[:, 0], rel[:, 1], "o-", color=c, label=name.replace("p_", ""))
    ax[0].plot([0, .6], [0, .6], "k--", lw=.8)
    ax[0].set(title="Calibration", xlabel="predicted", ylabel="observed win rate")
    ax[0].legend()
    ax[1].plot(dg["draw"], dg["emp_win"], "o-", label="empirical")
    ax[1].plot(dg["draw"], dg["mkt_win"], "s-", label="market")
    ax[1].plot(dg["draw"], dg["trf_win"], "^-", label="traffic")
    ax[1].set(title="Win rate by draw (ST)", xlabel="draw")
    ax[1].legend()
    ax[2].plot(g["logratio"], g["win_rate"], "o-", label="win rate")
    ax[2].plot(g["logratio"], g["p_market"], "s--", label="market p")
    ax[2].plot(g["logratio"], g["p_traffic"], "^--", label="traffic p")
    ax[2].set(title="Disagreement: who's right", xlabel="log(traffic/market)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_residuals.png", dpi=110)
    print(f"\nwrote {FIG / 'eda_residuals.png'}")


if __name__ == "__main__":
    main()
