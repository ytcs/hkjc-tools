---
title: "S1c READ-SIDE (#69): read the FULL career θ = (CP, W'_0, P_max) from the dedicated per-surface estimators and run the EOM FORWARD to a MECHANISTIC ability index (EOM-predicted finishing mean velocity, parameter-free given θ) — does it close the gap to the 0054 oracle? VERDICT: NO. The forward-EOM full-θ index reaches ρ(rf_now) ST +0.099 / HV +0.024 — barely above the single-v_c read (+0.083/+0.020), closing only +11% / +3% of the gap to the 0054 oracle (+0.231/+0.157) and far below the form ceiling (+0.312/+0.250). Adding W'_0 over a CP-only forward EOM lifts +0.033 ST / +0.012 HV; adding P_max adds ~0. The full physiological θ does carry a LITTLE more ability than CP alone (mostly via W'), but the FORWARD EOM is itself a COLLAPSE: it maps the rich career velocity PROFILE back onto ONE scalar (the predicted mean velocity), discarding exactly the profile SHAPE (v_open vs v_mean vs v_close) that 0054 showed carries ⅔ of the path's ability signal. So reading full θ does NOT recover the oracle — the bottleneck is the single-scalar SUMMARY (whether v_c or the EOM mean velocity), not which constants are read. DECISION-RELEVANT: the win-prob propagation must consume the multi-dimensional career velocity PROFILE (or the EOM's predicted profile SHAPE, not just its mean), not a richer single index. A forward-EOM mean-velocity index is NOT the fix."
id: 0056-s1c-read-side-full-theta-forward-eom-ability-index-does-not-close-gap-to-oracle-the-eom-collapse-is-the-bottleneck-not-the-vc-read
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor
tags: ["s1c", "read-side", "full-theta", "forward-eom", "eom", "ability-index", "ability-read", "career-aggregation", "empirical-bayes", "eb-shrink", "expanding-prior", "critical-power", "critical-velocity", "w-prime", "p-max", "pmax-opening", "wprime-spent", "sprint-ceiling", "dedicated-surface", "deconfounded-cruise", "hill", "form", "form-target", "rf-now", "re-emergence", "reemergence", "rating-floor", "form-ceiling", "oracle", "under-extracted", "velocity-profile", "scalar-collapse", "leak-free", "strictly-prior", "first-principles", "no-regression", "no-epicycle", "pure-mechanism", "0048", "0052", "0054"]
refs: {"module": "src/hkjc/model/pure_mechanism.py (build_pure_mechanism_panel + per-run P_max/W'_0 + eb_shrink + career profile cols)", "study": "studies/path_full_theta_check.py", "test": "tests/test_pure_mechanism.py", "estimators": "src/hkjc/model/identify.py (pmax_opening_run, pmax_for_ceiling, sprint_ceiling_velocity, wprime_spent_run, hill_fit, settled_cruise_velocity)", "eom_core": "src/hkjc/model/dynamics.py (eom_section_velocities_fast, cp_for_cruise, make_resistances)", "field": "src/hkjc/model/field.py (deconfounded_cruise_velocity, cover_fraction)", "thread_journal": "journal/entries/0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor.md", "oracle_journal": "journal/entries/0054-s1b-path-ability-ceiling-oracle-the-longitudinal-path-is-under-extracted-not-ability-poor-career-oracle-near-form-ceiling.md", "battery_journal": "journal/entries/0048-dedicated-surface-identification-battery-cp-cruise-vs-hill-disagree-by-design-physics-falsification-pmax-aliased-tau-unidentifiable.md", "panel": "data_cache/trajectory_dataset_full.parquet", "rating": "data_cache/rating_dataset.parquet", "mass_going": "data_cache/results_rich.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
S1c READ-SIDE — the critical-path fix proposed by 0054. 0054's ORACLE (a regularized OOS ridge over
the whole strictly-prior career velocity profile) reached ρ(rf_now) ST +0.231 / HV +0.157, ~2.8×/7.8×
the single-v_c fused read (+0.083/+0.020), at 74%/63% of the form ceiling — diagnosing the path as
UNDER-EXTRACTED: the single critical-velocity collapse throws ~⅔ of the path's ability away. This
entry tests the MECHANISTIC (non-regression) version of the proposed fix: read the FULL physiological
θ from the career-aggregated dedicated per-surface estimators and run the EOM FORWARD to an ability
index, and check whether it reaches the oracle. DISCIPLINE: mechanism, not an epicycle — the index is
the EOM-predicted finishing velocity given θ, parameter-free given θ; NOTHING is fit/regressed onto
form (form is a correlation target only). The oracle is the CEILING to reach, never the model.

## WHAT WAS BUILT (src/hkjc/model/pure_mechanism.py — the only file edited in src/)
1. **Per-run P_max and W'_0**, added to `build_pure_mechanism_panel` (the strictly-prior panel),
   reusing the EXISTING dedicated estimators (`hkjc.model.identify`):
   * **P_max [W]** = mean of `pmax_opening_run` (the gate-transient peak, 0048's top surface) and
     `pmax_for_ceiling(sprint_ceiling_velocity)` (the sustained-sprint cross-check). 0048 showed
     opening ≈ 2× ceiling (peak-instantaneous vs sustained); their mean is the least-aliased single
     deployment ceiling the forward EOM wants, bounded to [15, 120] kW.
   * **W'_0 [J]** = `wprime_spent_run(run, cp)` with `cp` = the de-confounded cruise CP for that run
     (cp_for_cruise of the field-de-confounded settled velocity, 0049) — the reserve-spent-above-CP
     surface (0048), bounded to [10, 600] kJ.
   * a per-run cruise **CP/v_c** (the de-confounded cruise velocity, already in the panel).
2. **Career-EB aggregation** (`eb_shrink` + new panel columns `vc_career`/`pmax_career`/
   `wprime0_career` and their prior-run counts): for each horse, the strictly-prior expanding-window
   MEAN of each constant over its PRIOR runs (the run's own path never enters its own θ), to be
   empirical-Bayes-shrunk toward the TRAIN-ONLY population mean by `n/(n+K)`, K=3 prior-run units —
   the physics analog of recency-form EB. Leak-free by construction.
3. **NEW `studies/path_full_theta_check.py`**: assemble the EB-shrunk career full θ per run, run the
   FORWARD EOM (`dynamics.eom_section_velocities_fast`) for the run's KNOWN distance + conditions
   (mass via `body_mass_kg`, going) → the EOM-predicted FINISHING MEAN VELOCITY = the mechanistic
   ability index. Correlate with the SAME leak-free form target as 0052/0054 (run's realised relative
   finish rf_now, both venues, aligned on the SAME path-based-read rows), with the single-v_c fused
   read, the 0054 oracle, the form ceiling, the rating floor.

## THE TABLE (Pearson ρ; aligned path rows; ST N=15,663, HV N=6,304; coverage of career θ ≈ 100% of rows)
TARGET = the run's OWN realised finish rf_now (0052 TABLE B / 0054 headline — directly comparable)
| venue | N | EOM full-θ index | single-v_c (0052) | 0054 oracle | form CEILING | rating floor |
|---|---|---|---|---|---|---|
| **ST** | 15,663 | **+0.099** | +0.083 | +0.231 | +0.312 | −0.001 |
| **HV** | 6,304  | **+0.024** | +0.020 | +0.157 | +0.250 | −0.026 |

(secondary, TARGET = strictly-prior recency-form: EOM full-θ ST +0.260 / HV +0.196 vs single-v_c
+0.188/+0.148 vs oracle +0.560/+0.525 vs ceiling +0.895/+0.917 — same picture: a small lift over the
single read, a long way short of the oracle.)

## MARGINAL VALUE of P_max and W' over a CP-only forward EOM (rf_now, ST / HV)
| forward-EOM θ | ST | HV |
|---|---|---|
| CP-only (career CP, POP P_max + W') | +0.065 | +0.014 |
| CP + W' (POP P_max) | +0.097 | +0.024 |
| CP + P_max (POP W') | +0.066 | +0.012 |
| FULL θ (CP + W' + P_max) | +0.099 | +0.024 |

So **W'_0 carries a real but small marginal lift** (+0.032 ST / +0.010 HV over CP-only) and **P_max
carries ~0** (Δ vs full ≈ +0.001 / −0.001). The full physiological θ is slightly more form-relevant
than CP alone — but the increment is tiny against the oracle gap.

## THE VERDICT — reading FULL θ does NOT close the gap; the EOM-mean COLLAPSE is the bottleneck
The forward-EOM full-θ index reaches +0.099 / +0.024, **+11% / +3%** of the way from the single-v_c
read to the 0054 oracle, and roughly ⅓ of the form ceiling. It does NOT close the gap. The reason is
mechanistically clear and reconciles 0054 exactly:

**The forward EOM is itself a SCALAR COLLAPSE.** 0054 localized the path's ability signal in the
multi-dimensional career velocity PROFILE — the SHAPE relating opening vs cruise vs closing section
speeds (top ridge weights v_open +0.37, v_mean −0.35, v_close +0.26), not any single level. The
forward-EOM ability index, by design, maps the full θ back onto ONE number (the predicted finishing
mean velocity). That single scalar cannot carry the profile shape, so it inherits almost the same
ceiling as the single-v_c read — confirming that the bottleneck is the SINGLE-SCALAR SUMMARY, not
which constants feed it. Reading three constants instead of one helps only insofar as W' perturbs the
mean velocity a little; it does not restore the discarded shape information.

This is the honest, decision-relevant outcome the task asked for: **the EOM full-θ index does NOT beat
the single-v_c read by much and falls well short of the oracle.** Per the task's own framing, that
means the win-prob PROPAGATION — how θ becomes a finishing-order law — is where the richer signal must
enter, NOT a cleverer single ability index. A forward-EOM mean-velocity index is not the fix.

## WHAT THE PROPAGATION REBUILD SHOULD CONSUME (the pointer this hands to the orchestrator)
The +0.231/+0.157 oracle lives in the career velocity PROFILE's SHAPE. To capture it mechanistically
(no regression) the generative drift must read the profile, not a level:
- feed the EB-shrunk career θ into the EOM and let the win/order law read the PREDICTED SECTION-
  VELOCITY PROFILE (the full `eom_section_velocities_fast` vector — opening ramp, cruise, closing
  kick/fade), not just its mean — the closing-vs-cruise differential is where v_close/v_mean live;
- equivalently, let the first-passage/order law be driven by the de-confounded career profile's
  SHAPE (a multi-section drift), so the opening-vs-closing contrast that the oracle weights enters the
  win probability directly. The single drift v_c (or a single EOM-mean index) provably cannot.
The path is NOT exhausted (0054 stands); but the missing extraction is in the propagation's USE of the
profile, not in adding more constants to a scalar read.

## DISCIPLINE / LEAK-FREEDOM
FIRST-PRINCIPLES, no regression of the OUTCOME into the model: the ability index is the EOM-predicted
finishing velocity given θ — parameter-free given θ, no fitted coefficient, no epicycle. Form is a
correlation TARGET only, never an input to any read. Leak-free: every career θ is strictly-prior
(expanding prior-mean, a run's own path excluded — panel construction); the EB population means and
the v_c-only/P_max/W' contrast priors are TRAIN-ONLY (a 60/40 time split); the form targets are
shifted one race back; the rating is an as-of backward join; mass/going are KNOWN pre-race public.
`src/hkjc/model/pure_mechanism.py` is the only src file edited (its sole editor); `dynamics.py`,
`generative.py`, the pricer, `gate.py`, `field.py`, `identify.py` UNTOUCHED. Lint clean
(`ruff check src/hkjc/model/pure_mechanism.py studies/path_full_theta_check.py`). The pure_mechanism
unit suite is GREEN (5 passed) with the no-meta-signal-import guard intact (the new panel columns are
strictly-prior physiology; no meta-signal table imported). DON'T-BAKE — a read-side diagnostic; the θ
builder / production constants are unchanged (the new career columns are emitted by the panel but not
yet wired into the drift, pending the propagation decision). CAVEAT: the index is the EOM finishing
MEAN velocity; a propagation that reads the predicted PROFILE (not the mean) is the untested next step
the oracle implies — this entry only falsifies the "richer single index" route, it does not test the
profile-propagation route.
