"""Scout collateral-form refinements (quick_eval, direction only — over-states ~4x).

Builds the modelling frame ONCE, then swaps in each refined collateral table and re-runs
quick_eval at the baked per-venue COLLAT_GAIN. Marginal test vs the baked one-sweep table.
Confirm any winner with the PAIRED full-scale backtest (PYTHONHASHSEED=0).

    VENUE=HV uv run --group bayes python studies/collat_scout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import VENUE, load  # noqa: E402
from quick_eval import quick_eval  # noqa: E402
from hkjc.model.features import collateral_form_table_v2  # noqa: E402

KEYS = ["date", "venue", "race_no", "runner"]


def swap_collat(df: pl.DataFrame, rr: pl.DataFrame, **kw) -> pl.DataFrame:
    ct = collateral_form_table_v2(rr, **kw)
    out = df.drop("collat_form").join(ct, on=KEYS, how="left")
    return out.with_columns(pl.col("collat_form").fill_null(0.0))


def main():
    df, jky = load(VENUE)
    rr = pl.read_parquet(Path("data_cache") / "results_rich.parquet")
    base = quick_eval(df, jky)
    print(f"{VENUE} baked-collateral baseline: {base['pct_market']:.2%}  ll={base['model_ll']:.4f}")

    configs = {
        "v2-default(=baked)": {},
        "recency hl=4": {"half_life": 4.0},
        "recency hl=6": {"half_life": 6.0},
        "band-par": {"band_par": True},
        "sweeps 3": {"n_sweeps": 3},
        "sweeps 5": {"n_sweeps": 5},
        "sweeps 8": {"n_sweeps": 8},
        "sweeps5+recency4": {"n_sweeps": 5, "half_life": 4.0},
        "sweeps5+bandpar": {"n_sweeps": 5, "band_par": True},
        "sweeps5+rec4+bandpar": {"n_sweeps": 5, "half_life": 4.0, "band_par": True},
    }
    for name, kw in configs.items():
        d2 = swap_collat(df, rr, **kw)
        r = quick_eval(d2, jky)
        dd = (r["pct_market"] - base["pct_market"]) * 100
        print(f"  {name:24s} {r['pct_market']:.2%}  d={dd:+.2f}pp  ll={r['model_ll']:.4f}")


if __name__ == "__main__":
    main()
