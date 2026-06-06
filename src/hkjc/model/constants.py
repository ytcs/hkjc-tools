"""Baked model calibration constants — the single source of baked model calibration.

Each constant is a mechanism-coupling gain mapping a measured signal onto a physical term
of the critical-velocity / power-balance win model (not a free regressor weight). Values
are set by the leak-free OOS conversion test (% of market + ECE); the findings/stories
behind each number live in journal/, not here. Per-venue dicts + env-var overrides let
fit and sweeps resolve the active value.
"""
from __future__ import annotations

import os

# Recency-form half-life, in races (exp-weighted relative-finish, not the career mean).
FORM_HALF_LIFE = float(os.environ.get("FORM_HALF_LIFE", "2.0"))

# Multiplier on the per-horse sectional v_c empirical-Bayes residual.
SHRINK_MULT = float(os.environ.get("SHRINK_MULT", "0.5"))

# DRAW_GAIN: fractional extra effective distance per draw off the rail (bend ground-loss), in the drift.
DRAW_GAIN = float(os.environ.get("DRAW_GAIN", "0.003"))
# Per-venue bend ground-loss (HV bend-dominated ⇒ larger). ST straight-1000m chute is draw-zeroed in fit.
DRAW_GAIN_BY_VENUE = {"ST": 0.003, "HV": 0.008}
# AWT (dirt) rail bias; fit resolves by SURFACE, overriding the venue dict.
DRAW_GAIN_AWT = float(os.environ.get("DRAW_GAIN_AWT", "0.008"))

# DRAW_FS_GAIN: crowding×draw — field-size scaling of bend ground-loss; interim stopgap, HV-only.
DRAW_FS_GAIN = float(os.environ.get("DRAW_FS_GAIN", "0.0"))
DRAW_FS_GAIN_BY_VENUE = {"ST": 0.0, "HV": 0.25}

# DRAW_VAR_GAIN: draw → diffusion D (trip variance, not the drift); per-horse D_i = D·(1+gain·risk_i). ST-only.
DRAW_VAR_GAIN = float(os.environ.get("DRAW_VAR_GAIN", "0.0"))
DRAW_VAR_GAIN_BY_VENUE = {"ST": 0.5, "HV": 0.0}

# POS_FADE_GAIN: positional-fade on v_c — RETIRED to 0.0, moved to the W′ knob (see POSFADE_W_GAIN).
# Kept as a tunable for diagnosis.
POS_FADE_GAIN = float(os.environ.get("POS_FADE_GAIN", "0.0"))

FORM_GAIN = float(os.environ.get("FORM_GAIN", "5.0"))
# Per-venue form coupling (HV lower than ST); fit falls back to FORM_GAIN for an unknown venue.
FORM_GAIN_BY_VENUE = {"ST": 5.0, "HV": 4.0}
# JOCKEY_GAIN: jockey strike-rate coupling onto effective v_c.
JOCKEY_GAIN = float(os.environ.get("JOCKEY_GAIN", "6.0"))
W_PRIME_GAIN = float(os.environ.get("W_PRIME_GAIN", "1.0"))
# SIM selects the win-probability model:
#   "wald" — DEFAULT production: finishing time = drift-diffusion first-passage (Inverse-Gaussian).
#   "pl"   — fast closed-form probe (Plackett-Luce / Gumbel softmax); full order distribution, no sampling.
SIM = os.environ.get("SIM", "wald")

# ALLOMETRIC_B: body-mass power exponent b; v_c ∝ m^(b−1) (b=1 = no mass effect). Declared weight is
# pre-race ⇒ leak-free.
ALLOMETRIC_B = float(os.environ.get("ALLOMETRIC_B", "1.0"))
# Per-venue mass exponent (ST-only: converts at ST, b=1 at HV).
ALLOMETRIC_B_BY_VENUE = {"ST": 1.1, "HV": 1.0}

# TRIAL_GAIN: additive barrier-trial debut-signal coupling onto v_c. Superseded by TRIAL_PI_RATIO fusion.
# Needs data_cache/barrier_trials.parquet (features.barrier_trial_table).
TRIAL_GAIN = float(os.environ.get("TRIAL_GAIN", "0.0"))
TRIAL_GAIN_BY_VENUE = {"ST": 1.5, "HV": 1.0}

# TRIAL_PI_RATIO: precision-fusion fade-rate knob ρ=π_trial/π_race — folds a barrier trial into the EB
# v_c posterior with a race-count-adaptive weight (no-op at ρ=0). Math derivation: fit.py docstring + journal.
# Needs trial_vc/trial_m columns (features.trial_vc_table) + barrier_trials.parquet.
TRIAL_PI_RATIO = float(os.environ.get("TRIAL_PI_RATIO", "0.0"))
TRIAL_PI_RATIO_BY_VENUE = {"ST": 2.0, "HV": 0.0}

# COLLAT_GAIN: relational opponent-strength coupling — a leak-free Massey collateral rating (m/s vs par),
# within-race centred. fit resolves per-venue unless collat_gain is passed.
COLLAT_GAIN = float(os.environ.get("COLLAT_GAIN", "0.0"))
COLLAT_GAIN_BY_VENUE = {"ST": 1.0, "HV": 2.5}

# COLLAT_CONSTRUCTION_BY_VENUE: per-venue collateral-rating construction (features.collateral_form_table_v2;
# empty dict = baked one-sweep table). Keys:
#   - n_sweeps: Jacobi propagation depth over the strictly-prior race graph (1 = single online sweep).
#   - half_life: recency-weight on per-horse collateral observations, in races (None = flat expanding mean).
#   - band_par: trip-conditioned par + band-matched field-strength inheritance (like-for-like trips).
COLLAT_CONSTRUCTION_BY_VENUE = {
    "ST": {"n_sweeps": 3, "half_life": 4.0, "band_par": True},
    "HV": {},
}

# SUSTAIN_GAIN / FADE_GAIN: trajectory-shape couplings on v_c. SUSTAIN off (carried by form+v_c);
# FADE retired to 0.0, moved to the W′ knob (FADE_W_GAIN). Both kept as tunables for diagnosis.
SUSTAIN_GAIN = float(os.environ.get("SUSTAIN_GAIN", "0.0"))
FADE_GAIN = float(os.environ.get("FADE_GAIN", "0.0"))

# Late-race anaerobic-reserve couplings on the W′ knob (the (1+W′/L) boost), in reserve-metres,
# distance-aware for free. Signs: FADE_W_GAIN > 0 (hold speed ⇒ more reserve), POSFADE_W_GAIN < 0
# (front-runner burns W′ ⇒ less reserve).
FADE_W_GAIN = float(os.environ.get("FADE_W_GAIN", "8.0"))
POSFADE_W_GAIN = float(os.environ.get("POSFADE_W_GAIN", "-80.0"))

# PEAKGAP_GAIN: rating-trajectory own-peak deficit (rtg_peakgap ≤ 0, within-race centred); a process feature.
PEAKGAP_GAIN = float(os.environ.get("PEAKGAP_GAIN", "0.0"))
PEAKGAP_GAIN_BY_VENUE = {"ST": -0.015, "HV": -0.025}

# TRN_HOT_GAIN: trainer hot-streak / intent (recency-weighted stable momentum, within-race centred).
TRN_HOT_GAIN = float(os.environ.get("TRN_HOT_GAIN", "0.0"))
TRN_HOT_GAIN_BY_VENUE = {"ST": 2.0, "HV": 1.0}
N_SAMPLES = int(os.environ.get("N_SAMPLES", "6000"))
