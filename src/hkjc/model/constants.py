"""Baked model calibration constants — the single source for the production ABM.

These are the mechanism-coupling gains the modelling agent calibrates via the
diagnose -> recalibrate -> bake -> re-diagnose loop (market-residual diagnosis). They are
NOT free regressor weights: each maps a measured signal onto a physical term of the
critical-velocity / power-balance model. Change them only with an OOS % of market +
calibration (ECE) check, never to chase in-sample fit.

History / rationale (do not lose):

- FORM_GAIN = 5.0  (m/s of effective v_c per unit form-deviation from the neutral 0.5).
  Form is the strongest ability measurement and survives the handicap: a horse that
  consistently beats its field runs effectively faster than its sectional estimate.
  RECALIBRATED 4.0 -> 5.0 at the recency-form (56%) baseline: switching form to the
  RECENCY-weighted measurement (FORM_HALF_LIFE) flipped the form residual back to UNDER-
  weighted (t=+11.4 at FG=4), because recency form is a sharper, higher-variance signal
  than the career mean it replaced. The sweep found a clean new interior optimum at 5.0
  (6.0 worse, 7.0 falls off a cliff), lifting full-scale ST 55.3% -> 56.2%. History: the
  career-form era ran 2.0 -> 4.0 (market under-weighted ability, t=+32; ECE 0.0081 ->
  0.0069, matching the market's own calibration).

- JOCKEY_GAIN = 5.0  (m/s of v_c per unit jockey strike-rate above the field-neutral
  mean). Jockey couples like form — a measurement of the effective v_c the horse
  realises (part better-trip, part the private info that a fancied horse gets a top
  rider). Confirmed in the ABM: paired Δ +0.030 nats/race, interior optimum ~3-6 (gain
  15 over-couples and hurts), lifting the model ~29% -> ~38% of market skill. Baked at
  the plateau centre.


- W_PRIME_GAIN = 1.0  (multiplier on the per-horse anaerobic-reserve residual = the
  un-amplified natural EB estimate). REVERTED 2.0 -> 1.0 at the recency-form (56%) baseline.
  History: in the career-form era the reserve was UNDER-weighted (w_prime_eb t=+13.2), so we
  amplified it 1.0 -> 2.0 (+1.4pp then). Recency form ABSORBED the reserve signal — the
  re-diagnosis flipped w_prime to OVER-weighted (t=-8.95), so the extra amplification no
  longer earns its keep. The win-prob sweep is flat over WPG 0-2 (W' is a finishing-margin
  effect, low-leverage on WHO wins), so parsimony + the residual sign both say revert to the
  un-amplified 1.0. Mechanism stays wired for place/exotics where margins matter.

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

# DRAW_GAIN: fractional extra effective distance per draw-position off the rail, in the wald/pl
# drift (L_eff = dist·(1 + DRAW_GAIN·(draw−1))). Physics: a wide-drawn horse runs a longer arc round
# the turns (bend ground-loss) => longer first-passage time. The production wald model originally had
# NO draw term (draw lived only in the kinematic geometry sim, dropped when wald became production) —
# the re-diagnosis flagged draw as the #1 residual (t=−26.5). Geometry fixes the sign; the magnitude is
# calibrated and absorbs jockeys steering toward the rail (which mitigates the naive geometric loss).
# 0.0 = no draw effect; CALIBRATED to 0.003 (clean monotone sweep, broad peak 0.003-0.004),
# lifting full-scale ST 56.2% -> 58.1% — the biggest lever since recency form. Sanity check: 0.003/lane
# means draw-14 vs draw-1 runs ~3.9% (~62m) longer over 1600m, which is the right order for the extra
# arc of being ~13 lanes wider round the turns — the calibrated magnitude lands in the geometric
# ballpark, so the mechanism is physical, not a fudge. NOTE (per the user): draw is a proxy for geometry
# AND interaction (horses are not collisionless — boxed-in/blocked on the rail, forced wide). This wald
# term captures only the MEAN-FIELD draw effect; the interaction structure (draw x field-size,
# draw x pace, boxed-in) is the agent-based sim's domain (its original motivation). See hkjc.model.fit.
DRAW_GAIN = float(os.environ.get("DRAW_GAIN", "0.003"))
# PER-VENUE bend ground-loss. The draw penalty is GEOMETRIC, so it must scale with the track: HV is
# bend-dominated (the strongest draw bias in HK) and wants ~2.7x the ST gain — the global ST-calibrated
# 0.003 massively UNDER-penalised draw at HV (re-diagnosis journal 0024: HV draw residual was the #1
# mechanism by far; the per-venue sweep peaks at HV~0.008, ST~0.003, lifting HV 48->53% of market).
# NOTE the sign lesson: a subagent first read the HV residual as "over-weighted, set to 0" — the actual
# sweep proved the OPPOSITE (0 craters HV to 41%); always sweep, never trust a residual-sign label.
# The ST straight 1000m chute (no bends) gets draw_gain ZEROED in fit_predict_traffic (geometry, not a
# constant) — there the residual draw effect is a reverse watering/going gradient, not bend ground-loss.
DRAW_GAIN_BY_VENUE = {"ST": 0.003, "HV": 0.008}
# AWT (Sha Tin dirt, 1650m) has a STRONGER rail bias than turf (dirt kickback, no watering gradient):
# draw 1 wins ~14% vs draw 8 ~1%. The turf-calibrated 0.003 under-penalises it; the AWT-only sweep peaks
# ~0.008 (HV-like), lifting AWT ~43%→49% of market at zero cost to turf. The AWT betting market is itself
# ~33% softer than turf (a more reachable benchmark). fit resolves this by SURFACE, overriding the venue.
DRAW_GAIN_AWT = float(os.environ.get("DRAW_GAIN_AWT", "0.008"))

# DRAW_FS_GAIN: crowding-amplified draw — an INTERIM mean-field stopgap on the draw axis (NOT the final
# word; see below). The bend ground-loss penalty is scaled by field size: dg = draw_g·max(0, 1 +
# DRAW_FS_GAIN·(N−12)), so a wide draw costs MORE in a crowded field (more rivals to be boxed by / forced
# wider) and the penalty vanishes (never reverses — hence the clamp) in a roomy small field. PER-VENUE and
# HV-ONLY: the full-scale PAIRED (fixed-seed) test converts only at HV (0.25 → +0.3pp, 53.8→54.1%; the
# venue where collision/boxed-in physics dominates — tightest bends, the strongest draw bias in HK), and is
# FLAT at ST (0.0pp). LESSON: the 150-race quick_eval wildly OVER-stated this (+1–1.6pp) — only the paired
# full-scale gate (PYTHONHASHSEED fixed so base/candidate share MC seeds) revealed the true ~+0.3pp.
# DEFERRED, NOT REJECTED: draw is deeply coupled to the CONTROL dimension (jockey policy — how the gate is
# used to take up position / navigate the trip), so a regression/mean-field term only captures the
# first-order geometric SHADOW of a geometry×control phenomenon. The richer relational pace-pressure term
# (front-runner fade ∝ rival front-runner count, THEORY §3b) doesn't yet convert for the same reason — the
# mechanism is real but needs geometry + control modelled JOINTLY (a differential game of jockeys on the
# geometry), not because the axis is dead. Treat DRAW_GAIN/DRAW_FS_GAIN as interim until that theory exists.
DRAW_FS_GAIN = float(os.environ.get("DRAW_FS_GAIN", "0.0"))
DRAW_FS_GAIN_BY_VENUE = {"ST": 0.0, "HV": 0.25}

# POS_FADE_GAIN: positional-fade coupling (m/s of effective v_c per unit habitual-early-position above
# the field mean). The production drift-diffusion is NON-interacting, so it over-rates habitual front-
# runners (which burn the anaerobic reserve setting the pace with no draft) and under-rates closers —
# the mean-field shadow of the pace EQUILIBRIUM (a differential game: each jockey spends a finite W' given
# rivals' styles). Ground-truth-confirmed: front horses win ~1pp LESS than the physics predicts, closers
# ~1pp MORE; residual t≈−7 both venues, survives controlling for form/v_c/W'. early_pos high = front, so
# the calibrated gain is NEGATIVE (front -> lower v_c). See features.early_position_table. CALIBRATED to
# -1.0: a clean shared optimum at BOTH venues (ST +1.1pp, HV +1.2pp full-scale), confirming it's a true
# mean-field constant (same sign+magnitude everywhere) — the right signature for an equilibrium effect,
# not a venue quirk. Sharp peak; |gain|>=2 collapses fast (don't over-couple a within-race-centred term).
POS_FADE_GAIN = float(os.environ.get("POS_FADE_GAIN", "-1.0"))

FORM_GAIN = float(os.environ.get("FORM_GAIN", "5.0"))
# PER-VENUE form coupling. The model was originally calibrated on ST and applied globally; a
# per-venue re-diagnosis (journal 0024) showed HV — bend-dominated, the tightest track in HK — wants
# a LOWER form coupling (HV form was over-weighted, residual t=-13.7 vs ST's -8.4). HV optimum ~4.0
# (joint sweep peak), ST stays 5.0. fit_predict_traffic resolves the venue's value unless form_gain is
# passed explicitly (for sweeps). Falls back to FORM_GAIN for an unknown venue.
FORM_GAIN_BY_VENUE = {"ST": 5.0, "HV": 4.0}
# JOCKEY_GAIN 5->6: re-swept after the mass/pos-fade/per-venue bakes; jg=6 is a clean shared optimum at
# BOTH venues (ST +0.4pp, HV +0.35pp; 7 ties, 8 falls off). The jockey signal absorbs trip/positioning
# quality the sectional EB v_c misses — slightly more leverage now that the other channels are sharper.
JOCKEY_GAIN = float(os.environ.get("JOCKEY_GAIN", "6.0"))
W_PRIME_GAIN = float(os.environ.get("W_PRIME_GAIN", "1.0"))
# SIM selects the win-probability model (the interacting agent-based sims were deprecated 2026-06):
#   "wald" — DEFAULT, the production win model. Finishing time = first-passage of a drift-diffusion
#            (Inverse-Gaussian); the principled, first-principles noise law. One physical parameter
#            (the diffusion coefficient D); surface-dependent (AWT 2x).
#   "pl"   — fast closed-form probe (Plackett-Luce / Gumbel softmax); instant, no sampling. Gives the
#            full finishing-order distribution (place/exotics); slightly less accurate than wald for win.
SIM = os.environ.get("SIM", "wald")

# ALLOMETRIC_B: the body-mass power exponent b in p_c = κ·m^b — the ONE physiological constant for how
# aerobic power scales with muscle mass. Derived (not a free gain) from the power-balance steady state
# v_c = p_c/(K_LOCO·m·g) ⟹ v_c ∝ m^(b−1); applied on the wald path as a field-relative factor
# v_c *= (m/m̄)^(b−1). The residual diagnosis showed the market backs heavier horses MORE (bodywt
# residual t=+8.6) and a probe confirmed it is REAL mass (partial t=+8.8 controlling for rating+class;
# monotone; survives within-class) ⟹ b>1 (power scales super-linearly with mass). b=1.0 = no mass
# effect. Declared weight is pre-race, so leak-free.
ALLOMETRIC_B = float(os.environ.get("ALLOMETRIC_B", "1.0"))
# PER-VENUE: the allometric mass effect is REAL and the #1 ST residual (bodywt t≈+8), but converts to
# %market only at ST (b≈1.2, +0.5pp; the persistent mass residual). At HV the coupling does NOT convert
# (b=1.0 best — like several HV residuals it's collinear with what's already priced there), so mass is
# ST-only. fit resolves per-venue unless allometric_b is passed explicitly (for sweeps). b=1 = isometric.
ALLOMETRIC_B_BY_VENUE = {"ST": 1.1, "HV": 1.0}

# TRIAL_GAIN: barrier-trial speed-figure coupling (m/s of v_c per unit trial_fig). The FIRST NEW-DATA
# lever: a barrier trial is the only on-track form a DEBUTANT has, and the market is otherwise efficient
# on debuts via the official handicapper rating — so this is genuinely new signal, not a reweighting of
# what's priced. trial_fig is already a velocity (trial v_c vs a going×distance par, same kinematics as
# the speed figure), so the coefficient is O(1). PER-VENUE (HV benefits more — softer debut market, smaller
# fields): full-scale PAIRED test ST 60.0→60.3% (gain 1.0), HV 54.1→54.9% (gain 1.5); clean interior optima
# both venues (2.5 over-couples). On the SUBSET where it acts (races with a trialed debutant) it is worth
# +3–7pp. NOTE this is a LOWER BOUND: the panel only carries 1 year of trials (2025-06→), so only recent
# debuts are covered — a fuller barrier-trial backfill lifts it further. Needs data_cache/barrier_trials.parquet
# (features.barrier_trial_table, joined as-of on the short stable brand). fit resolves per-venue unless passed.
TRIAL_GAIN = float(os.environ.get("TRIAL_GAIN", "0.0"))
TRIAL_GAIN_BY_VENUE = {"ST": 1.0, "HV": 1.5}
N_SAMPLES = int(os.environ.get("N_SAMPLES", "6000"))
