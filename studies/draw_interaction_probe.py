"""Is the residual draw effect an INTERACTION (collision) signature, or pure geometry?

The mean-field bend ground-loss (DRAW_GAIN, baked into wald) neutralised the AGGREGATE draw
residual (t=-26.5 -> -1.5). But pure bend geometry is field-size-INDEPENDENT (the arc is the same
arc no matter how many rivals), whereas the INTERACTION part — boxing, forced wide, blocked on the
rail — should worsen with field size (more bodies to be trapped by). So a single mean coefficient
can be right on average yet wrong per-bucket: over-penalising draw in small fields, under-penalising
in big ones. This measures the draw-residual slope WITHIN field-size buckets. A slope that trends
DOWN (more negative) with field size = the collision fingerprint (the agent-based sim's domain).

Sign: resid = market - model. Model penalises outside draws. If small fields need LESS penalty than
the baked mean, model gives outside too little -> resid positive (slope > 0). If big fields need MORE,
resid negative (slope < 0). Interaction => slope decreasing in field size.

Run: uv run --group bayes python studies/draw_interaction_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load
from residual_diagnosis import clustered_slope, collect_oos


def main():
    df, jky_mean = load("ST")
    p = collect_oos(df, jky_mean, None, 1.0, "wald", 1.0)
    p = p.with_columns(
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("mkt"),
        (pl.col("p") / pl.col("p").sum().over("race_id")).alias("mdl"))
    p = p.with_columns((pl.col("mkt") - pl.col("mdl")).alias("resid"))

    print("draw-residual slope within field-size buckets (interaction => slope falls with field size):")
    print(f"  {'field size':>12s} {'races':>6s} {'draw slope':>11s} {'se':>8s} {'t':>7s}")
    buckets = [("<=8", pl.col("field_size") <= 8), ("9-10", (pl.col("field_size") >= 9) & (pl.col("field_size") <= 10)),
               ("11-12", (pl.col("field_size") >= 11) & (pl.col("field_size") <= 12)), ("13-14", pl.col("field_size") >= 13)]
    for label, cond in buckets:
        q = p.filter(cond)
        rid = q.select("race_id").to_numpy().ravel()
        b, se = clustered_slope(q["draw"].to_numpy().astype(float), q["resid"].to_numpy(), rid)
        print(f"  {label:>12s} {q['race_id'].n_unique():>6d} {b:>+11.5f} {se:>8.5f} {b/se:>7.2f}")


if __name__ == "__main__":
    main()
