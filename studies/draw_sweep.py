"""Calibrate the bend ground-loss (DRAW_GAIN) re-added to the production wald drift.

The re-diagnosis flagged draw as the #1 market residual (t=-26.5) because production wald had
NO draw term (it lived only in the kinematic geometry sim, dropped when wald became production).
This sweeps the fractional-extra-distance-per-lane coefficient. Physics fixes the sign (outside
draws slower); the data fixes the magnitude (absorbs jockeys steering toward the rail).

Run: uv run --group bayes python studies/draw_sweep.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load
from quick_eval import quick_eval

CAP = int(os.environ.get("CAP", "500"))
NS = int(os.environ.get("NS", "4000"))


def main():
    df, jky = load("ST")

    def ev(**kw):
        return quick_eval(df, jky, n_folds=4, test_cap=CAP, n_samples=NS, sim="wald", **kw)["pct_market"]

    print(f"DRAW_GAIN sweep (bend ground-loss in the wald drift)  [cap={CAP} ns={NS}]\n")
    for dg in (0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040):
        print(f"  DRAW_GAIN={dg:.4f}: {ev(draw_gain=dg):.2%}")


if __name__ == "__main__":
    main()
