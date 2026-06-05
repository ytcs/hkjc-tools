---
id: 0063-form-gain-recalibrated-2-4-residual-diagnosis-dr
title: FORM_GAIN recalibrated 2->4 (residual-diagnosis driven): ST 43.3%->48.2% of market, HV 39%->43%, baked
kind: result
status: open
tags: ["goodness-of-fit", "calibration", "form-gain", "milestone", "market-fit", "diagnose-recalibrate-loop", "standing-order"]
refs: {}
related: []
created: 2026-06-05T02:02:52+00:00
updated: 2026-06-05T02:02:52+00:00
---
FIRST GOODNESS-OF-FIT WIN from the diagnose->recalibrate->bake loop. The market-residual diagnosis
flagged the ABM under-weighting ability (form residual t=+32). ability_spread_sweep (FORM_GAIN x
SPEED_NOISE on a test subsample) found a clean INTERIOR optimum at FORM_GAIN=4.0 (6.0 worse, 8.0 much
worse). Confirmed at FULL SCALE (N=3000, 1932 ST races, leak-free walk-forward):

  FORM_GAIN 2.0 -> traffic skill +0.2355 nats/race = 43.3% of market, ECE 0.0081
  FORM_GAIN 4.0 -> traffic skill +0.2625 nats/race = 48.2% of market, ECE 0.0069 (== market 0.0069)

+4.9pp of market at ST, and ECE now MATCHES the market's own calibration. HV replicates: 39% -> 43%
(+4pp) at FORM_GAIN=4.0 (HV is the 99-race subset; same direction). Baked FORM_GAIN default 2.0->4.0 in
backtest_traffic_st.py (HV imports it).

KEY MECHANISTIC NUANCE: lowering SPEED_NOISE (0.12->0.09->0.06) HURT at every gain. So the model was NOT
globally under-confident -- it specifically under-amplified the GOOD signal (form). Raising FORM_GAIN
sharpens along the right axis; cutting noise sharpens ALL axes incl the noisy sectional v_c -> overfit.
SPEED_NOISE=0.12 confirmed optimal. Lesson: sharpen the validated signal, keep race-day noise honest.

WHY 2.0 was stale: it was calibrated on the old 3-month 161-race subset; the full-history rebuild (3248
races) shifted the optimum. Re-calibrate engine constants after any data-scope change.

METHOD THAT WORKED (keep using): run ABM walk-forward -> resid = market_prob - model_prob (de-vigged,
race-normalised) -> cluster-robust regress on observables -> the biggest |t| mechanism is the next fix.
This replaced guessing-which-covariate with a direct map of where the model misfits the market. Objective
is FIT (bet-skill = goodness-of-fit), not edge. Next: re-diagnose at FG=4 (residuals reshuffle once probs
sharpen); first-pass queue was bodywt/mass t=+11.6 (unused EOM term) and draw/bend over-weight t=-17.
