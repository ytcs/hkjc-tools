"""Cheap priority scan: which condition/skill signals carry marginal info beyond form?

form_dataset is small (140 races) but carries EVERY signal of interest (jockey, trainer,
carried weight, body weight, gear, running style, days-since). We use it to PRIORITISE what
to scrape at scale — not to make a final claim. In-sample conditional-logit race-win log-loss:
baseline = form proxies (win_rate_prior + avg_finish_prior3); then baseline + each signal.
A drop = the signal adds apparent marginal info on top of form. (In-sample + tiny data, so
treat as a RANKING, not significance.)

Run: uv run python experiments/signal_scan_form.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")


def race_win_ll(pred: pl.DataFrame, prob="q") -> float:
    tot, n = 0.0, 0
    for _, g in pred.group_by("RaceId"):
        q = np.clip(g[prob].to_numpy().astype(float), 1e-9, None)
        q = q / q.sum()
        won = g["won"].to_numpy().astype(bool)
        if won.any():
            tot += -np.log(q[won][0]); n += 1
    return tot / n


def build() -> pl.DataFrame:
    fd = pl.read_parquet(CACHE / "form_dataset.parquet").sort(["HorseName", "Date"])
    # leak-free trailing strike rates (expanding mean of won over strictly-prior runs)
    def trail_sr(key):
        return ((pl.col("won").cum_sum().over(key) - pl.col("won"))
                / (pl.col("won").cum_count().over(key) - 1))
    fd = fd.sort(["Jockey", "Date"]).with_columns(trail_sr("Jockey").alias("jky_sr"))
    fd = fd.sort(["Trainer", "Date"]).with_columns(trail_sr("Trainer").alias("trn_sr"))
    fd = fd.sort(["HorseName", "Date"]).with_columns(
        (pl.col("DeclarHorseWt") - pl.col("DeclarHorseWt").shift(1).over("HorseName")).alias("bwt_chg"),
        (pl.col("Gear") != "--").cast(pl.Int8).alias("has_gear"))
    glob = float(fd["won"].mean())
    fd = fd.with_columns(
        pl.col("jky_sr").fill_null(glob), pl.col("trn_sr").fill_null(glob),
        pl.col("bwt_chg").fill_null(0.0),
        pl.col("days_since_last").fill_null(pl.col("days_since_last").median()),
        pl.col("win_rate_prior").fill_null(glob),
        pl.col("avg_finish_prior3").fill_null(pl.col("avg_finish_prior3").median()))
    # z-score the continuous features
    for c in ["ActWt", "DeclarHorseWt", "bwt_chg", "days_since_last", "win_rate_prior",
              "avg_finish_prior3", "jky_sr", "trn_sr"]:
        s = fd[c].std()
        fd = fd.with_columns(((pl.col(c) - fd[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))
    return fd


def main():
    fd = build()
    print(f"{fd.height} runs, {fd['RaceId'].n_unique()} races (in-sample ranking)\n")
    base = ["z_win_rate_prior", "z_avg_finish_prior3"]

    def ll(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["RaceId"], prob_col="q")
        return race_win_ll(fp(fd, fd))

    b = ll(base)
    print(f"  {'signal added':28s} {'race-win LL':>11s} {'Δ vs form':>10s}")
    print(f"  {'(form baseline)':28s} {b:>11.4f} {'—':>10s}")
    signals = [
        ("+ carried weight (ActWt)", "z_ActWt"),
        ("+ body weight", "z_DeclarHorseWt"),
        ("+ body-weight change", "z_bwt_chg"),
        ("+ gear on", "has_gear"),
        ("+ days since last", "z_days_since_last"),
        ("+ jockey strike rate", "z_jky_sr"),
        ("+ trainer strike rate", "z_trn_sr"),
    ]
    rows = []
    for name, col in signals:
        d = b - ll(base + [col])
        rows.append((name, d))
        print(f"  {name:28s} {b - d:>11.4f} {d:>+10.4f}")
    print("\n  ranked by apparent marginal info (Δ>0 = helps):")
    for name, d in sorted(rows, key=lambda r: -r[1]):
        print(f"    {d:>+.4f}  {name}")
    print("\n(In-sample, 140 races — a PRIORITY ranking for what to scrape at scale, not significance.)")


if __name__ == "__main__":
    main()
