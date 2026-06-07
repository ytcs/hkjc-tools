---
title: "S1b PATH ABILITY-CEILING ORACLE (#68, decision-gating): what is the BEST achievable strictly-prior path→form correlation from ALL longitudinal-sectional signals combined, vs the single fused read (0052)? NEW studies/path_ability_ceiling.py assembles a strictly-prior FEATURE MATRIX of every path-derived signal (opening/mid/closing/mean section velocity, velocity range, late deceleration, settled cruise, de-confounded cruise, Hill v_c/W'/n_dist) in TWO leak-free forms — PER-RUN (lag-1, the most-recent-prior run) and CAREER (expanding prior-mean) — plus contemporaneous mass/distance/going, and takes the ORACLE = max OUT-OF-SAMPLE Pearson ρ(regularized-ridge linear combo, leak-free form target) via 5-fold-CV-tuned ridge with POOLED out-of-fold prediction (NOT in-sample R²). VERDICT: UNDER-EXTRACTED, not ability-poor. Against the run's own realised finish rf_now the career oracle reaches ST +0.231 / HV +0.157 vs the single fused read +0.083/+0.020 (a 2.8×/7.8× LIFT) and the rating floor ≈0 — landing at 74%/63% of the form CEILING (+0.312/+0.250). The signal IS in the longitudinal path; the single critical-velocity fusion throws ~⅔ of it away. The lift lives in the CAREER-MEAN RAW SECTION VELOCITIES (career-vel-only oracle +0.202 ST, ≈ the full +0.231), NOT in distance (distance-only +0.017) and NOT in the Hill/de-confounded panel alone (+0.108) — the fusion collapsing the path to one v_c number is the bottleneck. The oracle is robust to horse-grouped CV (ST +0.227 / HV +0.141, ≈ random-split), so no per-horse memorization inflation. RECOMMENDATION: FIX the read (career EB aggregation of the raw section-velocity profile, not just a single fused v_c level) before any pivot — the path is NOT yet exhausted."
id: 0054-s1b-path-ability-ceiling-oracle-the-longitudinal-path-is-under-extracted-not-ability-poor-career-oracle-near-form-ceiling
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor
tags: ["s1b", "oracle", "ability-ceiling", "ability-read", "path-physics", "longitudinal-sectional", "feature-matrix", "career-aggregation", "expanding-mean", "per-run", "ridge", "out-of-sample", "held-out", "k-fold", "cross-validation", "form", "form-target", "recency-form", "rf-now", "re-emergence", "reemergence", "rating-floor", "form-ceiling", "under-extracted", "not-ability-poor", "section-velocity", "settled-cruise", "deconfounded-cruise", "hill", "critical-velocity", "fuse-vc-read", "decision-gating", "leak-free", "strictly-prior", "first-principles", "no-regression", "0049", "0050", "0052"]
refs: {"study": "studies/path_ability_ceiling.py", "reuses": "studies/ability_read_check.py (form_targets, rating_floor, _norm_keys, _pearson)", "module": "src/hkjc/model/pure_mechanism.py (build_pure_mechanism_panel, _per_run path assembly), src/hkjc/model/identify.py (settled_cruise_velocity), src/hkjc/model/field.py (deconfounded_cruise_velocity, cover_fraction)", "thread_journal": "journal/entries/0052-s1-verification-ability-read-fusion-lands-near-deconfounded-level-but-does-not-beat-hill-slope-deconfounded-read-weak-on-broad-population.md", "deconf_journal": "journal/entries/0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges.md", "panel": "data_cache/trajectory_dataset_full.parquet", "rating": "data_cache/rating_dataset.parquet", "mass_going": "data_cache/results_rich.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
S1b — THE DECISION-GATING DIAGNOSTIC. S1 (0052) found the precision-weighted FUSED path ability read
correlates with the leak-free form target at only ρ≈+0.083 (ST) / +0.020 (HV), barely above the
rating FLOOR (≈0) and far below the trailing-form CEILING (+0.31/+0.25). That measured ONE fused
instrument. The prior question, which decides whether to keep mining the path or pivot to new data:
what is the BEST achievable strictly-prior path→form correlation from ALL longitudinal-sectional
signals COMBINED — the ORACLE? NEW `studies/path_ability_ceiling.py`; src/hkjc READ ONLY (untouched);
lint clean; reuses `ability_read_check.py`'s loaders + the SAME leak-free form target so the numbers
are directly comparable to 0052.

## THE HARNESS — a strictly-prior feature matrix in TWO forms, OUT-OF-SAMPLE ridge oracle
FEATURE MATRIX = every signal physically derivable from the longitudinal sectional path + KNOWN
conditions: opening / mid (cruise) / closing / mean section velocity, velocity range, late
deceleration (mid→closing), settled cruise (`identify.settled_cruise_velocity`), de-confounded cruise
(`field.deconfounded_cruise_velocity`), Hill v_c / W' / n_dist (`build_pure_mechanism_panel`), plus
contemporaneous declared mass / carried weight / distance / going (pre-race public, allowed for path
reconstruction, PHIL §7). Built in TWO leak-free forms: (a) PER-RUN = the horse's MOST-RECENT-PRIOR
run's path features (lag-1 — a run's own path never enters its own features); (b) CAREER =
expanding MEAN of each feature over ALL the horse's PRIOR runs. The Hill/de-confounded panel reads
are already strictly-prior expanding by construction.

ORACLE = max OUT-OF-SAMPLE Pearson ρ(regularized linear combo, form target). A ridge with a
5-fold-CV-tuned penalty, fit on TRAIN folds and scored on the HELD-OUT fold; out-of-fold predictions
are POOLED (every row predicted by a model that never saw it) and the λ with the best pooled OOS ρ is
the oracle. NOT an in-sample R² (which would be a meaningless inflated "oracle"). Missing features are
mean-imputed with a missingness indicator so thin-history runners still contribute. Same row
alignment as 0052 (a path-based ability read present: Hill OR de-confounded).

## THE TABLE (Pearson ρ, OUT-OF-SAMPLE; aligned path rows; ST N=15,663, HV N=6,304)
TARGET = the run's OWN realised finish rf_now (0052 TABLE B — the target whose baselines the task quotes)
| venue | N | oracle PER-RUN | oracle CAREER | fused (0052) | rating FLOOR | form CEILING |
|---|---|---|---|---|---|---|
| **ST** | 15,663 | +0.226 | **+0.231** | +0.083 | −0.001 | +0.312 |
| **HV** | 6,304  | +0.173 | **+0.157** | +0.020 | −0.026 | +0.250 |

TARGET = strictly-prior recency-form (half-life 2, shift 1; 0052 TABLE A — ceiling = form autocorrelation)
| venue | N | oracle PER-RUN | oracle CAREER | fused (0052) | rating FLOOR | form CEILING |
|---|---|---|---|---|---|---|
| **ST** | 15,663 | +0.534 | **+0.560** | +0.188 | +0.238 | +0.895 |
| **HV** | 6,304  | +0.507 | **+0.525** | +0.148 | +0.292 | +0.917 |

(The recency-form ceiling +0.90/+0.92 is the form AUTOCORRELATION — a different, near-trivial target;
rf_now is the decision-relevant one, hence the headline.)

## THE VERDICT — UNDER-EXTRACTED, not ability-poor
The career-aggregated oracle (rf_now) reaches **ST +0.231 / HV +0.157**, a **2.8× / 7.8× LIFT** over
the single fused read (+0.083 / +0.020) and far above the rating floor (≈0). It lands at **74% / 63%
of the form CEILING** (+0.312 / +0.250). This is the FIX end of the spectrum, NOT the PIVOT end: the
ability signal IS present in the longitudinal sectional path; the single critical-velocity FUSION
discards roughly two-thirds of it. The per-run (lag-1) and career oracles are close (ST +0.226 vs
+0.231; HV +0.173 vs +0.157) — even one prior run's full path profile beats the fused read; the
career mean adds a little at ST.

## WHERE THE SIGNAL LIVES (feature-subset oracles, ST, rf_now)
- distance only → +0.017 (NOT a distance/velocity-scale artifact).
- contemporaneous conditions only (dist+mass+carried-wt+going) → +0.074 (carried weight = the
  handicap boundary carries a little; legitimate pre-race public).
- CAREER RAW SECTION VELOCITIES only (open/close/mean/mid/range/late-decel) → **+0.202** — essentially
  the whole oracle.
- Hill/de-confounded PANEL only → +0.108 (the panel that feeds the fused read tops out near the
  fused value — confirming the fusion is not the bottleneck's victim, it IS the bottleneck).
- no-conditions (career path + panel, 16 dims) → +0.223 ≈ the full +0.231.
Top standardized ridge weights (career, rf_now): ST = v_open +0.37, v_mean −0.35, v_close +0.26,
v_mid +0.26, W'_hill −0.26, v_c_hill −0.24 (the raw section-velocity PROFILE — its SHAPE, opening
vs mean vs closing — not a single level). HV is flatter and noisier (top: v_close +0.07,
late_decel −0.07, carried-wt +0.04) consistent with HV's thinner Hill coverage and shorter trips.

## WHY THE FUSION MISSED IT (reconciles 0050/0052)
0052 showed the fused read lands near the de-confounded LEVEL but that level is weak as a
broad-population per-run point estimate. S1b explains it: collapsing the path to ONE critical-velocity
number (whether Hill slope, de-confounded level, or their precision blend) is lossy. The form-relevant
ability lives in the multi-dimensional CAREER VELOCITY PROFILE — the relationship among opening,
cruise, and closing section speeds across a horse's prior runs — which a single v_c scalar cannot
carry. The 0050 "swap slope→level" framing was too narrow; the fix is to feed the career-aggregated
section-velocity PROFILE (richer than any one level), de-confounded, into the read.

## RECOMMENDATION (decision-gating outcome)
**FIX the read, do NOT pivot.** Concretely: career-aggregate (EB-shrunk over prior runs) the
de-confounded section-velocity PROFILE — opening / cruise / closing levels and their shape — rather
than a single fused v_c, and let the win drift read that richer ability surface. The +0.23/+0.16
oracle is the realistic upper bound a path-only read can reach (≈¾ of form's own ceiling); the gap
to the ceiling is the part of ability the longitudinal path does NOT see (lateral/positional control,
trip trouble — the CV-lateral-path territory), but that is the LAST mile, not the first. The
longitudinal path is NOT yet exhausted.

## CAVEAT ON THE OOS METHOD (could the oracle be biased?)
The k-fold split is row-level random, so a horse can appear in both train and test folds — a
potential per-horse memorization inflation. A horse-GROUPED 5-fold (all of a horse's runs held out
together) gives ST +0.227 / HV +0.141 — essentially IDENTICAL (ST −0.004, HV −0.016), as expected for
a LINEAR ridge with no per-horse features (it cannot memorize a per-horse offset). So the oracle is
NOT inflated by leakage across the split; if anything the random-split number is a hair generous at
HV. The oracle is also a CEILING in the sense that it CV-selects the best λ on the pooled OOS ρ
(picking the penalty by the held-out score is mildly optimistic, but with 9 coarse λ candidates and
n≫p this is negligible — the per-λ curve is flat near the optimum). Lower bound: the oracle could be
slightly UNDERSTATED because (a) it is purely LINEAR (a non-linear combiner could read interactions
among the velocity profile), and (b) the de-confounded cruise here is the per-run all-distance read,
not the route-only persistence estimator that scored +0.33/+0.40 in 0049. Net: +0.23/+0.16 is a
trustworthy, slightly-conservative estimate of the path-only ceiling.

## DISCIPLINE
FIRST-PRINCIPLES, no regression of the OUTCOME into the model — the ridge is a DIAGNOSTIC oracle
(measuring the achievable correlation), never a build; form is a correlation TARGET only, never an
input to any read. Leak-free: every path feature is strictly-prior (lag-1 or expanding prior-mean,
the run's own path excluded), the form targets are shifted one race back, the rating is an as-of
backward join; conditions are pre-race public. src/hkjc READ ONLY — no model file modified
(`build_pure_mechanism_panel`, `settled_cruise_velocity`, `deconfounded_cruise_velocity` imported
untouched). Lint clean (`ruff check studies/path_ability_ceiling.py`). Full pytest suite NOT run (per
task). DON'T-BAKE — a decision-gating measurement; no production constant changed. The finding is
recorded as-is: the longitudinal sectional path is UNDER-EXTRACTED by the single fused v_c read (the
career-section-velocity-profile oracle reaches ¾ of the form ceiling at ST, 4–8× the fused read both
venues) — the program should FIX the read (career-aggregated de-confounded velocity profile) before
pivoting to new data.
