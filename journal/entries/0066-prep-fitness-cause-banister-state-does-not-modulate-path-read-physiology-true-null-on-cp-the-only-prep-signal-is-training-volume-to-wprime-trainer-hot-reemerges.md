---
title: "PREP/FITNESS causal dimension (Wave-2 #72): does training load → an on-the-day FITNESS STATE → modulate the path-read physiological constants (CP/W' on race day)? Banister impulse-response (τ_fit=42d / τ_fat=7d / k_fat=2.0, literature priors) built strictly-prior from the trackwork gallops; identification = does prep predict a RESIDUAL in per-run CP/W' controlling for the horse's OWN career-mean baseline (build_pure_mechanism_panel). VERDICT (§4b ceiling check, both venues, seed-stable): on CP it is a clean TRUE NULL — the prep block adds +0.0003 ST / +0.0001 HV over a no-prep conditions ceiling on the CP residual, and +0.004 / +0.002 on the CP level; fitness and fatigue separately add ~0. The Banister STATE (fitness−fatigue) adds essentially nothing to ANY observable (cond+banister ≈ cond everywhere). The ONE real prep signal is on W' (Hill reserve D'): prep_signal +0.088 ST / +0.087 HV — but it is carried ENTIRELY by raw training VOLUME (cond+volume +0.236/+0.317 ≈ cond+prep), NOT the fitness state (cond+banister +0.159/+0.211 ≈ cond), and volume is partly confounded with career maturity (n_prior_gallops↔hill_n_dist +0.25 ST) so it is NOT cleanly a fitness→W' modulation. RE-EMERGENCE: trainer-hot DOES re-emerge from the prep latent without being fed in — prep-block→held-out trainer-recent-strike OOS ridge ceiling +0.109 ST / +0.148 HV (raw state↔trn_strike +0.087/+0.088), seed-stable. HONEST READ: the specific causal claim (Banister fitness STATE shifts the day's CP/W') is a TRUE NULL on CP; the only prep→physiology signal is a small, maturity-confounded volume→W' link; the genuine prep payoff is the trainer-hot re-emergence (the deleted TRN_HOT axis is a consequence of prep). Study-only; nothing wired into the generative drift; ridge is a DIAGNOSTIC ceiling, never the model, never baked."
id: 0066-prep-fitness-cause-banister-state-does-not-modulate-path-read-physiology-true-null-on-cp-volume-to-wprime-trainer-hot-reemerges
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["prep", "prep-cause", "fitness", "fatigue", "banister", "impulse-response", "training-load", "trackwork", "gallop", "physiology", "cp", "wprime", "w-prime", "critical-power", "hill", "modulation", "on-the-day", "identification", "residual", "career-baseline", "ceiling", "ceiling-check", "mis-spec-vs-null", "true-null", "s4b", "0054", "0062", "de-confound", "conditions-confound", "training-volume", "maturity-confound", "re-emergence", "reemergence", "trainer-hot", "trn-hot", "trn-hot-gain", "deleted-axis", "wave-2", "causal-network", "out-of-sample", "ridge", "leak-free", "strictly-prior", "first-principles", "no-regression", "no-epicycle", "no-gallop-fusion", "diagnostic-only", "dont-bake", "0050", "0060"]
refs: {"study": "studies/prep_fitness.py", "panel": "src/hkjc/model/pure_mechanism.py (build_pure_mechanism_panel — imported READ-ONLY)", "field": "src/hkjc/model/field.py (deconfounded_cruise_velocity, cover_fraction — imported READ-ONLY)", "identify": "src/hkjc/model/identify.py (settled_cruise_velocity — imported READ-ONLY)", "reuses_pattern": "studies/path_ability_ceiling.py (S1b OOS-ridge oracle), studies/field_pace_ceiling.py (0062 de-confounded ceiling)", "thread_journal": "journal/entries/0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network.md", "s1b_journal": "journal/entries/0054-s1b-path-ability-ceiling-oracle-the-longitudinal-path-is-under-extracted-not-ability-poor-career-oracle-near-form-ceiling.md", "field_ceiling_journal": "journal/entries/0062-s7b-field-pace-ceiling-mis-spec-vs-null-st-s1-pace-is-a-true-null-the-missed-effect-is-in-the-early-pace-gradient.md", "data_trackwork": "data_cache/trackwork.parquet", "data_panel": "data_cache/trajectory_dataset_full.parquet", "data_cond": "data_cache/results_rich.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
PREP/FITNESS — the Wave-2 PREP causal dimension (0060, task #72). The cause is UPSTREAM and physical:
training impulses → an on-the-day FITNESS STATE (fitness − fatigue) → a modulation of the horse's OWN
physiological constants (its CP / W' on race day shift with fitness). This is causal (prep is an
upstream cause of the physiology the path reveals), distinct from the DELETED acausal `gallop-fusion`
regression (which regressed gallop speed-figures onto `v_c` — an epicycle). NEW STUDY-ONLY
`studies/prep_fitness.py`; src/hkjc READ-ONLY (`build_pure_mechanism_panel`, `deconfounded_cruise_velocity`,
`settled_cruise_velocity` imported untouched). NOT wired into the generative drift (a deliberate later
step, like the field dimension's). Lint clean; `PYTHONHASHSEED=0`, reproducible. Full pytest NOT run
(per task).

## THE BANISTER FITNESS STATE (physiological priors — NOT outcome-fitted)
Per training session the impulse is `load = gallop_velocity · worked_distance_m` (m²/s, the kinematic
work proxy of one gallop), summed over the day's gallops. At race day D, over the horse's gallop
sessions s with `sdate < D` (STRICTLY-PRIOR, leak-free — prior gallops only):

    fitness = Σ_s load_s · exp(−(D − sdate_s)/τ_fit)
    fatigue = k_fat · Σ_s load_s · exp(−(D − sdate_s)/τ_fat)
    state   = fitness − fatigue

with the **literature impulse-response constants fixed BEFORE any read**: **τ_fit = 42 d** (slow
positive aerobic-adaptation trace), **τ_fat = 7 d** (fast negative recovery trace), **k_fat = 2.0**
(the canonical Banister negative-gain ratio k₂/k₁ ≈ 2 — a hard gallop hurts before it helps). These
are constants of the model, not knobs tuned to CP or to the win outcome. Coverage: trackwork has 1090
horses, median 76 prior gallops/horse, all overlapping the trajectory panel; with-prep-history runs
ST 10,826 / HV 5,633.

## THE IDENTIFICATION TEST — does prep predict a RESIDUAL in the path-read physiology?
Target = the per-run physiology read controlling for the horse's OWN strictly-prior career baseline:
`cp_resid = vc_run − vc_career` where `vc_run` is the CURRENT-run de-confounded cruise CP (the SAME
estimator `build_pure_mechanism_panel` uses internally, read on the run's own path — leak-free: the
prep PREDICTOR is strictly-prior, `vc_run` is the measured TARGET) and `vc_career` is the career-mean
CP. Also CP level `vc_run` and W' level `wprime_hill` (Hill reserve D' [m]). Raw correlations (n):

| prep → observable | ST corr | HV corr |
|---|---|---|
| state   → CP_resid (vc_run−career) | −0.028 | −0.033 |
| fitness → CP_resid | −0.039 | −0.025 |
| fatigue → CP_resid | −0.035 | −0.007 |
| state   → CP_level (vc_run) | −0.008 | −0.046 |
| state   → W'_level (D') | **+0.069** | −0.045 |
| fitness → W'_level (D') | +0.051 | −0.049 |

The raw prep→CP correlations are ~0 (and, where non-zero, the WRONG sign — fitter horses do NOT show
a higher day-CP). The only non-trivial raw correlation is state→W' on ST (+0.069), inconsistent in
sign across venues. A raw read is not a verdict (PHIL §4b) — apply the ceiling check.

## §4b CEILING CHECK — mis-specification vs TRUE NULL (the non-negotiable discipline)
A model-agnostic OOS ridge CEILING (CV-tuned λ, pooled out-of-fold, the S1b/0062 pattern — a
DIAGNOSTIC probe, NEVER the model) of each physiology observable from the FULL prep block, DE-CONFOUNDED
against a no-prep CONDITIONS ceiling (distance, field-size, class, body-weight, days-since-last-gallop).
**prep_signal = (cond+prep) − cond.** Components swept separately: fitness-only, fatigue-only,
Banister-only (fitness+fatigue+state), and raw VOLUME-only (recent14/recent60/load_total/n_gallops).
Seed-stable across seeds 0/1/7 (W' prep_signal ST +0.088–0.092, HV +0.082–0.087).

| venue | observable | cond | cond+prep | **prep_signal** | cond+fit | cond+fat | cond+banister | cond+volume |
|---|---|---|---|---|---|---|---|---|
| ST | CP_resid (vc_run−career) | +0.333 | +0.333 | **+0.0003** | +0.333 | +0.333 | +0.333 | +0.334 |
| ST | CP_level (vc_run) | +0.583 | +0.586 | **+0.0036** | +0.583 | +0.583 | +0.582 | +0.587 |
| ST | W'_level (D') | +0.151 | +0.238 | **+0.0877** | +0.153 | +0.148 | +0.159 | **+0.236** |
| HV | CP_resid (vc_run−career) | +0.133 | +0.133 | **+0.0001** | +0.133 | +0.133 | +0.127 | +0.139 |
| HV | CP_level (vc_run) | +0.333 | +0.335 | **+0.0017** | +0.333 | +0.332 | +0.332 | +0.334 |
| HV | W'_level (D') | +0.223 | +0.310 | **+0.0867** | +0.215 | +0.213 | +0.211 | **+0.317** |

### THE READS (per PHIL §4b: `ceiling ≈ cond ⇒ TRUE NULL`; `ceiling ≫ our model ⇒ mis-spec`)
- **CP (both the career-residual AND the level): a clean TRUE NULL, both venues.** The full prep block
  adds **+0.0003 / +0.0001** on the CP residual and **+0.004 / +0.002** on the CP level over the
  no-prep conditions ceiling. Even a flexible OOS predictor extracts essentially NOTHING from prep
  about the day's CP beyond what conditions already say. This is not a mis-specified fitness model
  missing a real CP effect — there is no CP-modulation signal in the trackwork to miss. **The fitness
  state does NOT modulate the path-read CP.**
- **The Banister STATE itself is null on EVERY observable.** `cond+banister ≈ cond` everywhere
  (W': +0.159/+0.211 vs cond +0.151/+0.223) — fitness, fatigue and their difference add ~0. The
  specific causal posit (fitness − fatigue shifts the constants) finds no purchase.
- **W' (Hill reserve D') is the one real prep signal (+0.088 ST / +0.087 HV) — but it is TRAINING
  VOLUME, not the fitness state.** `cond+volume` (+0.236 / +0.317) ≈ `cond+prep`, while `cond+banister`
  ≈ `cond`: the entire W' lift is carried by raw training VOLUME (recent load, total load, gallop
  count), not by the impulse-response state. And it is partly a MATURITY CONFOUND: `n_prior_gallops`
  correlates with `hill_n_dist` (+0.25 ST), the very data-width that identifies the noisy Hill
  intercept D' — more-trained horses are more-established horses with more distinct prior distances,
  so a cleaner/larger D'. The direct `n_gallops→W'` link (+0.18 both venues) survives that, so a
  genuine "more training → bigger anaerobic reserve" component is plausible, but it is NOT cleanly a
  fitness-STATE → W' modulation and is confounded enough that it is **not** a clean identification.

**VERDICT: on the headline claim (prep fitness STATE modulates the path-read CP/W') the result is a
TRUE NULL on CP and a small, volume-driven / maturity-confounded effect on W' — NOT a mis-specified
fitness model.** The de-confounded ceiling is the proof it is a true null and not us using the wrong
observable/coupling: the ceiling itself can read nothing more than conditions from prep on CP.

## RE-EMERGENCE — does trainer-hot (the deleted TRN_HOT axis) re-emerge from the prep latent?
Held-out test: a strictly-prior trainer recent-strike (exp-weighted win indicator over the TRAINER's
prior runners, shifted one race back — leak-free, never fed as an input), predicted from the prep
block. **Trainer-hot DOES re-emerge:**

| venue | raw state↔trn_strike | prep-block → trn_strike OOS ridge ceiling |
|---|---|---|
| ST | +0.087 | **+0.109** (seeds 0/1/7: +0.108–0.110) |
| HV | +0.088 | **+0.148** (seeds 0/1/7: +0.147–0.148) |

The prep latent predicts held-out trainer-recent-strike at ρ≈+0.11 ST / +0.15 HV WITHOUT trainer-hot
as an input — a real, seed-stable re-emergence. This is the genuine prep payoff: the trainer-hot
meta-signal is a downstream CONSEQUENCE of the prep cause (a hot trainer is one whose stable's training
load / gallop programme is currently producing fit, recently-galloped runners), exactly as PHIL §5 /
the §4.4 re-emergence map predicts. It re-emerges as a trainer-PROGRAMME signal even though prep does
NOT sharpen the individual horse's day-CP.

## THE HONEST READ (expect a SMALL increment if real — WIN is the diagnostic, not the target)
- The clean, specific finding: **the Banister fitness state does NOT modulate the path-read
  physiology** — CP is a TRUE NULL (§4b-confirmed, both venues, the de-confounded ceiling reads
  nothing), and the only prep→physiology signal is a small training-VOLUME→W' link that is partly a
  career-maturity confound, NOT a fitness-state effect. So the planned mechanism (a multiplicative/
  additive prep shift on the horse's own CP/W') is NOT supported by the trackwork as posed.
- The real value of the dimension is the **trainer-hot RE-EMERGENCE** (+0.11/+0.15 held-out): the
  deleted TRN_HOT axis is a downstream consequence of the prep cause, recovered without being fed in.
  Prep belongs in the causal network as a TRAINER-PROGRAMME / stable-form latent, NOT as a per-horse
  CP/W' modulator.
- POINTERS (PHIL §4.4 — a null is a pointer, not a defeat): (1) the trackwork load proxy
  (`gallop_velocity · worked_distance_m`) may be too coarse — HK gallops are mostly routine AWT pace
  work (mean 7.2 m/s, 200–600 m), so the impulse may carry little physiological information about
  race-day CP; a sharper impulse (relative-to-own-gallop-norm intensity, or workout TYPE — barrier
  trial vs routine) might surface a state effect the current proxy averages away. (2) The CP read is
  the de-confounded cruise; a prep effect might live in a derivative (the W'-burn / fade rate) the
  level read throws away, as 0062 found for the field dimension — a future sweep. But on the level
  reads tested, CP is a true null.

## DISCIPLINE
FIRST-PRINCIPLES — the fitness state is a Banister impulse-response with fixed literature time-constants,
no gain fitted on the win/pool/market outcome; NO meta-signal fed as an input (trainer-hot is a
held-out TARGET only). The ridge is a DIAGNOSTIC ceiling probe (bounding extractable prep structure in
a pre-race physiology observable), NEVER a build, NEVER a production read. Leak-free: the fitness state
uses PRIOR gallops only (session strictly before the race date); the physiology panel is strictly-prior
by construction; the career baseline `vc_career` is the horse's prior-run mean; the trainer-strike
target is shifted one race back (PHIL §7). src/hkjc READ-ONLY; study-only. `PYTHONHASHSEED=0`,
seed-stable. DON'T-BAKE — a decision-gating MEASUREMENT; no production constant changed, nothing wired
into the generative drift.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
