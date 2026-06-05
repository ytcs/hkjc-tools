"""Re-balance the ability channels at the 56% (recency-form) baseline.

The recency-form switch rebalanced the market residual: form flipped to UNDER-weighted
(t=+11.4), w_prime flipped to OVER-weighted (t=-8.95), v_c_eb is under-trusted (+14.5).
The baked gains (FORM_GAIN=4, W_PRIME_GAIN=2) were calibrated for CAREER form — stale now.
This sweeps the ability gains to find the new interior optimum. Fast probe (pl/wald).

Run: uv run --group bayes python studies/recalib_sweep.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load
from quick_eval import quick_eval

SIM = os.environ.get("SIM", "wald")
CAP = int(os.environ.get("CAP", "400"))
NS = int(os.environ.get("NS", "2500"))


def main():
    df, jky = load("ST")

    def ev(**kw):
        return quick_eval(df, jky, n_folds=4, test_cap=CAP, n_samples=NS, sim=SIM, **kw)["pct_market"]

    base = ev()
    base2 = ev()  # determinism check: must equal base now that the subsample is sorted
    print(f"baseline (FG=4, WPG=2, sn=0.12): {base:.2%}  (repeat {base2:.2%})  [SIM={SIM} cap={CAP} ns={NS}]\n")

    print("FORM_GAIN sweep (form under-weighted -> try higher):")
    for fg in (4.0, 5.0, 6.0, 7.0):
        print(f"  FG={fg}: {ev(form_gain=fg):.1%}")

    print("\nW_PRIME_GAIN sweep (w_prime over-weighted -> try lower):")
    for wpg in (0.0, 0.5, 1.0, 2.0):
        print(f"  WPG={wpg}: {ev(w_prime_gain=wpg):.1%}")

    # the wald sharpness knob is the diffusion coefficient D (NOT speed_noise, which wald ignores).
    # v_c_eb under-trusted (+14.5) => model under-confident => maybe smaller D (sharper). Test at FG=5.
    print("\ndiff_coef (D) sweep at FG=5 (v_c_eb under-trusted -> try smaller D = sharper):")
    for dc in (110.0, 130.0, 150.0, 170.0, 200.0):
        print(f"  D={dc}: {ev(form_gain=5.0, diff_coef=dc):.1%}")


if __name__ == "__main__":
    main()
