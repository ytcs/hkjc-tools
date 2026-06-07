---
title: "S1 VERIFICATION: the precision-weighted FUSION ability read (fuse_vc_read) MEASURED on a lean leak-free identification harness (studies/ability_read_check.py, both venues). VERDICT: the fusion does what it was BUILT to do mechanically — where both instruments exist it sits ~2.8× closer to the de-confounded LEVEL than to the noisy Hill SLOPE (mean|fused−decon|=0.30 vs |fused−hill|=0.83 m/s, ST) — but it does NOT beat the Hill read at re-emerging form: against the 0050-comparable target (the run's own realised finish rf_now) the three reads cluster tightly (fused +0.083 / hill +0.098 / decon +0.091, ST; +0.020 / +0.055 / +0.015, HV), all ABOVE the rating floor (≈0.00/−0.03) but far below the form-autocorrelation ceiling (+0.312/+0.250). The fusion LANDS NEAR de-confounded (as designed) but the de-confounded level — measured here as a strictly-prior EXPANDING-MEAN ability read over a horse's ALL prior runs on the BROAD path population (n=15,663 ST / 6,304 HV) — is itself WEAK (≈+0.09/+0.02 vs rf_now), NOT the +0.33/+0.40 of the 0049 route-only split-half PERSISTENCE test. So the S1 fusion is mechanically correct and changes the read in the intended direction, but it does NOT lift the ability→form correlation above the Hill slope on this population, because the +0.33/+0.40 de-confounded re-emergence does not survive being recast as a broad-population strictly-prior point estimate. This is a real finding that bears on S2/S5: the lift S1 was meant to capture lives in the 0049 ROUTE-ONLY split-half setup, not in the per-run de-confounded mean that the panel/fusion actually feed the win drift."
id: 0052-s1-verification-ability-read-fusion-lands-near-deconfounded-level-but-does-not-beat-hill-slope-deconfounded-read-weak-on-broad-population
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor
tags: ["s1", "verification", "ability-read", "fusion", "fuse-vc-read", "precision-weighted", "instrument-fusion", "hill", "hill-slope", "critical-velocity", "deconfound", "deconfounded-level", "cruise", "field-coupling", "form", "re-emergence", "reemergence", "form-target", "recency-form", "rating-floor", "form-ceiling", "autocorrelation", "leak-free", "strictly-prior", "first-principles", "no-regression", "pure-mechanism", "identification-harness", "0049", "0050", "0055"]
refs: {"study": "studies/ability_read_check.py", "module": "src/hkjc/model/pure_mechanism.py (fuse_vc_read, build_pure_mechanism_panel)", "thread_journal": "journal/entries/0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor.md", "deconf_journal": "journal/entries/0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges.md", "panel": "data_cache/trajectory_dataset_full.parquet", "rating": "data_cache/rating_dataset.parquet", "mass_going": "data_cache/results_rich.parquet (Pla + field-size + DeclarHorseWt + ActWt)"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
S1 VERIFICATION MEASUREMENT. S1 (already landed, tests green) replaced the pure-mechanism ability
read's hard priority chain (Hill slope first) with a precision-weighted FUSION of the instruments —
`hkjc.model.pure_mechanism.fuse_vc_read(vc_hill, vc_hill_se, vc_decon, vc_decon_n, vc_decon_sd,
vc_pop)`, the single source of truth feeding the win drift. The #55/0050 finding motivating it: the
Hill SLOPE read barely correlates with form (ρ≈+0.066 ST / +0.033 HV) while the de-confounded cruise
LEVEL re-emerges form near the ceiling (≈+0.33/+0.40, journal 0049). The fusion is meant to land near
the de-confounded level. THIS ENTRY measures whether it does. NEW lean harness
`studies/ability_read_check.py` (also the reusable identification harness S5 extends); src/hkjc READ
ONLY (untouched); lint clean.

## THE HARNESS — lean, leak-free, three reads on the SAME rows
Build the strictly-prior `build_pure_mechanism_panel(traj, results_rich)`. For every run with a
PATH-BASED ability read (Hill OR de-confounded present) assemble three candidate reads on the SAME
aligned rows: `vc_hill` (Hill slope alone), `vc_decon` (de-confounded level alone), `vc_fused`
(`fuse_vc_read` on the panel columns — the PRODUCTION read; `vc_pop` passed NaN so the fusion is a
genuine Hill↔de-confounded precision blend, exactly the production read for a path-based runner).

FORM TARGET (the stripped meta-signal, used PURELY as a correlation target, never an input):
`rf = 1 − (finish_pos−1)/(field_size−1)` ∈ [0,1] (1 = win), the relative finish. Two strictly-prior
windows, both computed from finishes PRIOR to the run (shifted one race back, leak-free):
  * `form`          — exp-weighted RECENCY form, half-life 2 races (the production `FORM_HALF_LIFE`),
                       i.e. the exact production `recency_form_table` signal, recomputed from
                       `results_rich` so the harness has NO production-frame dependency.
  * `trailing_form` — the expanding CAREER-MEAN of `rf` (a DIFFERENT prior window) → the form
                       AUTOCORRELATION ceiling.
RATING FLOOR: the strictly-prior official handicap rating (`rating_dataset.parquet`), as-of join by
(horse_id, date). All five correlations use the SAME aligned rows so they are comparable.

Two targets are reported. TABLE A uses the task-spec strictly-prior recency-form as the target (so
the ceiling is form's own autocorrelation). TABLE B uses the run's OWN realised finish `rf_now` — the
0050 headline-(a) target — so the absolute levels are directly comparable to the 0050 baselines.

## THE TABLES (Pearson ρ; aligned path rows; ST N=15,663, HV N=6,304)
Coverage: ST Hill 8,395 / de-confounded 15,663; HV Hill 1,665 / de-confounded 6,304 (matches 0050).

TABLE A — TARGET = strictly-prior RECENCY-FORM (half-life 2, shift 1); ceiling = form autocorrelation
| venue | N | ρ(fused) | ρ(hill) | ρ(decon) | ρ(rating FLOOR) | ρ(form CEILING) |
|---|---|---|---|---|---|---|
| **ST** | 15,663 | **+0.188** | +0.233 | +0.211 | +0.238 | +0.895 |
| **HV** | 6,304  | **+0.148** | +0.134 | +0.133 | +0.292 | +0.917 |

TABLE B — TARGET = the run's OWN realised finish `rf_now` (0050 headline-(a), baseline-comparable)
| venue | N | ρ(fused) | ρ(hill) | ρ(decon) | ρ(rating FLOOR) | ρ(form CEILING) |
|---|---|---|---|---|---|---|
| **ST** | 15,663 | **+0.083** | +0.098 | +0.091 | −0.001 | +0.312 |
| **HV** | 6,304  | **+0.020** | +0.055 | +0.015 | −0.026 | +0.250 |
| 0050 baseline | — | — | +0.066/+0.033 | ≈+0.33/+0.40 | +0.015/−0.022 | +0.382/+0.302 |

## THE VERDICT — the fusion is mechanically correct but does NOT beat the Hill read here
**The fusion does what it was BUILT to do at the mechanism level.** Where both instruments exist, the
fused read sits ~2.8× closer to the de-confounded LEVEL than to the noisy Hill SLOPE
(mean|fused−decon| = 0.30 vs |fused−hill| = 0.83 m/s at ST; 0.34 vs 0.50 m/s at HV; corr(fused,decon)
0.72 vs corr(fused,hill) 0.52 at ST). The precision weighting DOES down-weight the short-lever-arm
Hill slope toward the de-confounded level — the #55/0050 intent, obtained by mechanism not by hand.

**But the fusion does NOT lift the ability→form correlation above the Hill read.** All three reads
sit in a TIGHT cluster on both targets and both venues — vs `rf_now`: fused +0.083 / hill +0.098 /
decon +0.091 (ST); +0.020 / +0.055 / +0.015 (HV). The fused read LANDS NEAR de-confounded
(|fused−decon| ρ ≤ 0.01 on rf_now, both venues — the design goal) but does NOT beat Hill (it is
fractionally BELOW Hill at ST and below Hill at HV). All three clear the rating floor (≈0.00/−0.03,
reproducing 0050's near-zero floor) but all three sit far below the form-autocorrelation ceiling
(+0.31/+0.25 vs rf_now; +0.90/+0.92 vs prior recency-form).

**The decisive caveat — why "land near de-confounded" did NOT deliver the expected lift.** The
de-confounded LEVEL here (`vc_decon`, a strictly-prior EXPANDING-MEAN of a horse's de-confounded
cruise over ALL its prior runs, on the BROAD path population, n=15,663/6,304) reads only ≈+0.09/+0.02
vs realised finish — it is NOT the +0.33/+0.40 of the 0049 headline. That +0.33/+0.40 was a
DIFFERENT measurement: a ROUTE-ONLY (≥1800 m) split-half PERSISTENCE correlation (early-half mean →
held-out late-half mean, n=414 ST / 296 HV). Recast as the per-run strictly-prior point estimate that
the PANEL and the FUSION actually feed the win drift, on the full population, the de-confounded
re-emergence largely DISAPPEARS. So the fusion lands exactly where it was aimed — but the target it
lands near is weak in the form the production read uses it. The 0050 narrative ("swap slope→level and
the +0.33/+0.40 re-emergence flows in") does NOT hold once "level" is the broad-population per-run
de-confounded mean; the lift lives in the route-only split-half setup, which is not what feeds θ.

## WHAT THIS MEANS FOR S2 / S5 (honest, decision-relevant)
- **S1 closed the mechanism (the read now blends by precision toward the level) but NOT the payoff
  (the ability→form correlation did not rise above the Hill slope).** The expected jump from the
  slope→level swap is NOT realised on the population the drift sees. Proceeding to S2 expecting the
  fusion alone to re-emerge form is not supported by this measurement.
- **The +0.33/+0.40 de-confounded re-emergence is real but SETUP-BOUND** (route-only, split-half
  persistence). If S2/S5 want that signal in the win drift, the panel must reproduce that ESTIMATOR
  (e.g. a route-restricted, multi-reading de-confounded persistence read), not the current
  all-distance expanding mean — OR enrich the path-inferred ability beyond a single per-run level
  (margins surface + closing precision + the pace-interaction layer in the WIN drift, the 0050 §4.4
  pointer), which is the structurally honest route.
- **The harness is the reusable S5 spine.** Lean, leak-free (ability + both form windows strictly
  prior), no production-frame dependency, both venues, lint clean. It can carry any future ability
  read as a new column and re-run the same five-correlation comparison.

## DISCIPLINE
FIRST-PRINCIPLES, no regression, no meta-signal as an INPUT — form is a correlation TARGET only,
never fed to any ability read. Leak-free: the path physiology is strictly-prior by panel construction
(a run's own path never enters its anchor); both form windows are shifted one race back; the rating is
an as-of (backward) join. src/hkjc READ ONLY — no model file modified. `fuse_vc_read` /
`build_pure_mechanism_panel` imported untouched. Lint clean (`ruff check studies/ability_read_check.py`).
Full pytest suite NOT run (per the task). DON'T-BAKE — a verification measurement; no production
constant changed. The finding is recorded as-is: the S1 fusion is mechanically sound and moves the
read toward the level, but does NOT beat the Hill read at re-emerging form on the population the win
drift uses, because the de-confounded level's 0049 re-emergence is bound to a route-only split-half
estimator that the panel/fusion do not reproduce.
