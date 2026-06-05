"""Baked model calibration constants — the single source for the production ABM.

These are the mechanism-coupling gains the modelling agent calibrates via the
diagnose -> recalibrate -> bake -> re-diagnose loop (market-residual diagnosis). They are
NOT free regressor weights: each maps a measured signal onto a physical term of the
critical-velocity / power-balance model. Change them only with an OOS % of market +
calibration (ECE) check, never to chase in-sample fit.

History / rationale (do not lose):

- FORM_GAIN = 4.0  (m/s of effective v_c per unit form-deviation from the neutral 0.5).
  Form is a far stronger ability measurement than sectional v_c (18% vs 5% of market
  skill) and survives the handicap: a horse that consistently beats its field runs
  effectively faster than its sectional estimate. RECALIBRATED 2.0 -> 4.0 — the
  market-residual diagnosis showed the ABM under-weighted ability (market - model
  residual slope on form t=+32); the ability sweep then found an interior optimum at 4.0
  (6.0 worse, 8.0 much worse), lifting ST 43.3% -> 48.2% of market with ECE 0.0081 ->
  0.0069 (matching the market's own calibration). At FG=4 the form residual flips to
  t=-9.8 (mildly over) — true peak ~3.5, but full-scale strongly favours 4.0 over 2.0.

- SPEED_NOISE = 0.12  (per-horse race-day speed sd, fraction of v_target). Must be
  calibrated to the ability spread or the sim is over-confident. With a real ability
  signal (form) the win is mostly unpredictable, so this is large. KEY: lowering it
  HURTS at every gain — the fix was amplifying the GOOD signal (form), not global
  sharpening (cutting noise also sharpens the noisy sectional v_c and overfits).

- JOCKEY_GAIN = 5.0  (m/s of v_c per unit jockey strike-rate above the field-neutral
  mean). Jockey couples like form — a measurement of the effective v_c the horse
  realises (part better-trip, part the private info that a fancied horse gets a top
  rider). Confirmed in the ABM: paired Δ +0.030 nats/race, interior optimum ~3-6 (gain
  15 over-couples and hurts), lifting the model ~29% -> ~38% of market skill. Baked at
  the plateau centre.

- PACE_GAIN = 0.0  (running-style -> pace policy is OFF). The MECHANISM is real and the
  sim reproduces it (front-runner +0.6pp, closer -0.9pp), but as a PREDICTOR off the
  trailing-rel_early proxy it is FALSIFIED on ST: every PACE_GAIN>0 hurts OOS
  monotonically. The early-speed edge is already absorbed by FORM (and priced by the
  market), so coupling it double-counts. Kept as dormant infrastructure (trajectory-
  measured pace_bias, the venue/distance litmus, the best-response layer). Revisit at HV.

- W_PRIME_GAIN = 2.0  (multiplier on the per-horse anaerobic-reserve residual). The
  market-residual diagnosis showed the ABM over-channels ability through critical velocity
  (v_c/form over-weighted) and UNDER-weighted the W' reserve (w_prime_eb residual t=+13.2).
  This gain amplifies the per-horse reserve spread (the shrunken EB residual u_w), the
  reserve analogue of FORM_GAIN. RECALIBRATED 1.0 -> 2.0: the subsample sweep peaked broadly
  over 1.5-2.5 (3.0 over-amplifies — W' is the noisier estimate), and at full scale 2.0 lifts
  ST 47.7% -> 49.1% of market. The mechanism check confirms it: the w_prime residual HALVED
  (t +13.2 -> +6.4) and the coupled v_c over-weighting also fell (t -8.9 -> -7.0) — the
  ability channel rebalanced from critical-velocity toward reserve, as the mechanism predicts.

- N_SAMPLES = 6000  Monte-Carlo samples per race for the win-probability estimate.

bend_curvature (the geometric draw bias) is per-venue and lives in
``hkjc.model.calibration.CALIBRATED_BEND_CURVATURE`` (HV 0.0023, ST 0.0037). It is the
#1 open residual (draw OVER-weighted, t=-17.4) — the modelling agent's next recalibration.
"""
from __future__ import annotations

import os

# FORM_HALF_LIFE: races. The production form is RECENCY-weighted (exp-weighted relative-finish)
# with this half-life, NOT the career expanding mean. Discovered in the knob scan: recency lifts ST
# 48% -> 56% of market (+7.8pp) — the biggest single lever. Half-life sweep is a clean interior
# optimum at ~2 (career = hl→∞ is worst). Career-average form goes stale over HK's long careers
# (horses improve/decline/change class); recency tracks CURRENT ability and matches how the market
# weights recent runs. See hkjc.model.features.recency_form_table.
FORM_HALF_LIFE = float(os.environ.get("FORM_HALF_LIFE", "2.0"))

# SHRINK_MULT: multiplier on the per-horse sectional v_c empirical-Bayes residual. The knob scan
# showed the sectional v_c is slightly OVER-trusted at 1.0; 0.5 is the peak (career and recency
# scans agree), 0.0 (dropping it) costs only ~0.8pp — the sectional energetics barely earns its
# keep for win once form+rating are in. Baked 0.5.
SHRINK_MULT = float(os.environ.get("SHRINK_MULT", "0.5"))

FORM_GAIN = float(os.environ.get("FORM_GAIN", "4.0"))
SPEED_NOISE = float(os.environ.get("SPEED_NOISE", "0.12"))
JOCKEY_GAIN = float(os.environ.get("JOCKEY_GAIN", "5.0"))
PACE_GAIN = float(os.environ.get("PACE_GAIN", "0.0"))
W_PRIME_GAIN = float(os.environ.get("W_PRIME_GAIN", "2.0"))
# SIM selects the win-probability model:
#   "wald"      — DEFAULT, the production win model. Finishing time = first-passage of a drift-diffusion
#                 (Inverse-Gaussian); the principled, first-principles noise law. One physical parameter
#                 (the diffusion coefficient D); surface-dependent (AWT 2x). ~56% of market.
#   "pl"        — fast-and-dirty closed-form probe (Plackett-Luce / Gumbel softmax); instant, no sampling.
#                 Use for wide parameter sweeps; slightly less accurate than wald.
#   "kinematic" — the JAX traffic sim (car-following + draw geometry). For INTERACTIONS / place+exotic
#                 pricing only — traffic is low-leverage for WIN, where wald matches it ~35x faster.
#   "eom"       — the power-balance EOM (inert recorded infra; underperforms for win — see journal).
# The velocity-multiplicative "probit" mode was dropped (an odd in-between: noise on velocity, not the
# physical finishing-time observable, and a one-shot draw rather than a diffusion).
SIM = os.environ.get("SIM", "wald")

# ALLOMETRIC_B: the body-mass power exponent b in p_c = κ·m^b — the ONE physiological constant for
# how aerobic power scales with muscle mass. Derived from the EOM, not a free gain: v_c =
# p_c/(K_LOCO·m·g) ⟹ at steady state v_c ∝ m^(b−1). The kinematic (production) path applies exactly
# that analytic law as a field-relative factor v_c *= (m/m̄)^(b−1); the SIM="eom" path instead lets
# mass act through the full Newton's-law dynamics (inertia/fade/drag too). The residual diagnosis
# showed the market backs heavier horses MORE (bodywt residual t=+8.6) and a probe confirmed it is
# REAL mass (partial t=+8.8 controlling for rating+class; monotone; survives within-class) ⟹ b>1
# (power scales super-linearly with mass). b=1.0 = no mass effect; calibrate upward. NOTE: the full
# SIM="eom" path currently underperforms the kinematic sim (see journal) — production uses the
# steady-state law. Declared weight is pre-race, so leak-free. 1.0 until calibrated.
ALLOMETRIC_B = float(os.environ.get("ALLOMETRIC_B", "1.0"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "6000"))
