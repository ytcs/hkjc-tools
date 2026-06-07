---
title: "PAYOFF TEST 0050: the PURE-MECHANISM (path-physiology-only) win model — the GoF FALL is to the FLOOR (ST 65.4%→−7.6%, HV 60.0%→+4.6% of market) and FORM DOES NOT RE-EMERGE at the win gate. The decomposition shows the fall is the STRIPPED ABILITY STACK on raced horses (winner −log p: hill Δ+0.58 ST / +0.53 HV, debut Δ+0.66 / +0.34) far more than the debut concession. The path-inferred Hill v_c carries a WEAK held-out-form signal (pearson +0.066 ST / +0.033 HV — above the rating floor +0.015/−0.022 but far below the trailing-form ceiling +0.382/+0.302) and the physics-only win prob recovers only 7%/4% of the empirical form→win gradient (non-monotone). Strictly-prior, leak-free (drift 0), win% MEASURED not optimized."
id: 0050-pure-mechanism-payoff-test-gof-falls-to-floor-form-does-not-reemerge-at-win-gate-weak-ability-signal
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: final
thread: 0046-causal-rebuild-design-cause-not-effect-strip-metasignals-reemergence-validation
tags: ["causal", "causal-rebuild", "pure-mechanism", "payoff-test", "re-emergence", "form", "recency-form", "held-out", "gof-fall", "percent-of-market", "win-gate", "ece", "ability-latent", "v_c", "hill", "critical-velocity", "de-confounded", "field-coupling", "deconfound", "debut", "population-prior", "decomposition", "form-gradient", "form-to-result", "leak-free", "strictly-prior", "assert-no-lookahead", "statespace", "generative", "first-passage", "wald", "measured-not-optimized", "no-odds-gating", "0046", "0047", "0048", "0049", "queue-50"]
refs: {"study": "studies/causal_reemergence.py", "builder": "src/hkjc/statespace/pure_mechanism.py", "pipeline": "src/hkjc/statespace/pipeline.py", "gate": "src/hkjc/statespace/gate.py", "assembly": "src/hkjc/statespace/assembly.py", "design": "docs/CAUSAL_MODEL.md", "design_journal": "journal/entries/0046-causal-rebuild-design-cause-not-effect-strip-metasignals-reemergence-validation.md", "eom_journal": "journal/entries/0047-eom-first-principles-generative-core-bang-cp-bang-physics-replaces-heuristic-level-bias-zero-opening-ramp-captured.md", "identification_journal": "journal/entries/0048-dedicated-surface-identification-battery-cp-cruise-vs-hill-disagree-by-design-physics-falsification-pmax-aliased-tau-unidentifiable.md", "field_journal": "journal/entries/0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges.md", "frame": "studies/backtest_st.py (load)", "panel": "data_cache/trajectory_dataset_full.parquet", "mass_going": "data_cache/results_rich.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
FINAL — the CAUSAL PAYOFF TEST (docs/CAUSAL_MODEL.md §4/§6, thread 0046). The pure-mechanism θ
assembly (`src/hkjc/statespace/pure_mechanism.py`, journal 0047–0049) wired into the production
gate via `build_fit_predict` + `make_pure_mechanism_pre_fit` and RUN on the full walk-forward
(`studies/causal_reemergence.py`, 5 folds, n_samples=4000, both venues). The drift is Hill-CP →
de-confounded cruise → rating-conditioned population mean, with EVERY meta-signal STRIPPED (no
form / jockey / trainer / collateral / peak-gap / speed-fig / trial / gallop / gate-break). Win%
is MEASURED, never optimized; NO odds-gating. Leak-free strictly-prior — `assert_no_lookahead`
drift = 0.00e+00 both venues (perturbing a test outcome moves no win prob). Production `hkjc.model`,
the statespace physics, `pure_mechanism.py`, and `data_cache` UNTOUCHED; new study + journal only.

## (1) THE GoF FALL — to the FLOOR, both venues (MEASUREMENT, not a gate)
% of market on the SAME pool-book gate the production model answers to (paired, same per-race seed):

| venue | production | pure-mechanism | fall | model ECE | mkt ECE | path coverage (Hill / decon / debut) |
|---|---|---|---|---|---|---|
| **ST** | **65.4%** | **−7.6%** | **−73.0 pts** | 0.0238 | 0.0069 | 8,395 / 7,268 / 8,770 of 24,433 |
| **HV** | **60.0%** | **+4.6%** | **−55.4 pts** | 0.0116 | 0.0083 | 1,665 / 4,639 / 8,519 of 14,823 |

The fall is essentially to the UNIFORM floor — at ST the pure model is WORSE than 1/field on the
market gate (−7.6%; its win log-loss 2.558 > uniform 2.517). Stripping the meta-signals removes
~100% of the model's market skill. This is the expected direction (0046 predicted "win% falls then
recovers fundamentally") but the magnitude is total: physiology-only does not discriminate winners
at the win gate.

**The split — it is the STRIPPED ABILITY STACK on RACED horses, not the debut concession.** Mean
−log p on the realized WINNER, by the winner's ability source (lower = sharper on winners):

| winner source | ST pure | ST prod | ST Δ | HV pure | HV prod | HV Δ |
|---|---|---|---|---|---|---|
| **hill** (has a Hill line) | 2.807 | 2.226 | **+0.581** | 2.698 | 2.171 | **+0.527** |
| **decon** (de-conf. cruise) | 2.029 | 2.114 | **−0.085** | 2.189 | 2.226 | **−0.037** |
| **debut** (population mean) | 2.796 | 2.138 | **+0.658** | 2.512 | 2.174 | **+0.339** |

The Hill-anchored RACED horses are where the pure model loses MOST relative to production
(Δ+0.58 ST / +0.53 HV on their winners) — i.e. the loss is the stripped ability STACK (form,
collateral, jockey, …), not merely the debut fallback. Curiously the de-confounded-cruise group is
the ONLY one the pure model wins on (Δ−0.085 ST / −0.037 HV) — the 0049 de-confounded level carries
a hair of real ability there — but it is swamped by the Hill and debut groups. (The debut group's
large Δ is the population mean having no per-runner discrimination at all — production still has
trial/gallop/rating for first-timers.)

## (2) THE HEADLINE — does FORM re-emerge? **Weakly in correlation, NO at the win gate.**

**(2a) Path-inferred ability (strictly-prior Hill v_c) vs HELD-OUT recency-form.** Pearson /
Spearman of the path v_c against the run's realized relative-finish, vs the rating floor and the
trailing-form (autocorrelation) ceiling:

| venue | v_c → held-out form | rating FLOOR | trailing-form CEILING | v_c → form-signal |
|---|---|---|---|---|
| **ST** | **+0.066** / +0.062 | +0.015 / +0.001 | +0.382 / +0.373 | +0.123 |
| **HV** | **+0.033** / +0.029 | −0.022 / −0.022 | +0.302 / +0.289 | +0.070 |

The path ability is WEAKLY but consistently above the population/rating floor (it beats rating
~4× at ST, and rating is ~null) — so the Hill anchor carries SOME genuine, de-confounded ability
that rating does not. But it is an order of magnitude below the trailing-form ceiling (+0.066 vs
+0.382). The arrow it should produce (v_c → the form SIGNAL itself) is +0.123 ST / +0.070 HV — real
but faint. **NB this is WEAKER than the de-confounded-LEVEL re-emergence 0049 reported (+0.33 ST /
+0.40 HV).** The difference is the population: 0049 measured the de-confounded settled-cruise on
ROUTE runs split in time within a horse; here `vc_hill` is the across-distance Hill slope pooled
strictly-prior over ALL the horse's earlier races (often ≤3 distinct distances → a thin, noisy
slope). The de-confounded LEVEL is a better ability read than the Hill SLOPE at this data width.

**(2b) The decisive test — does the physics-only win prob reproduce the form→result gradient
WITHOUT form?** Bin runners by held-out recency-form (quintiles); mean model win-prob vs realized
win-rate across bins:

ST — empirical win% rises 0.019 → 0.050 → 0.066 → 0.089 → **0.175** across form quintiles
(slope +0.247, monotone). PURE-MECH win% is **flat** (0.077 → 0.079 → 0.073 → 0.083 → 0.088;
slope +0.0166, NON-monotone) — **recovers 7% of the empirical gradient**. PRODUCTION (form fed)
recovers 106% (slope +0.261, monotone). HV is the same shape: empirical slope +0.216, pure +0.0093
(**4%**, non-monotone), production +0.231 (107%).

| venue | empirical slope | PURE-MECH (recovers) | PRODUCTION (recovers) | bin-corr pure / prod |
|---|---|---|---|---|
| **ST** | +0.247 | +0.0166 (**7%**, non-mono) | +0.261 (106%) | +0.782 / +0.996 |
| **HV** | +0.216 | +0.0093 (**4%**, non-mono) | +0.231 (107%) | +0.856 / +0.996 |

**⇒ FORM does NOT re-emerge at the win gate.** The physics-only win prob is nearly flat across the
form gradient — it cannot reproduce the 9×-from-bottom-to-top win-rate rise that form encodes. The
bin-level tracking correlation is moderate (+0.78 / +0.86) because the ordering is roughly right,
but the SLOPE — the load-bearing quantity for win discrimination — is essentially zero.

## (3) WHAT RE-EMERGES / WHAT IS STILL MISSING
**Re-emerges (weak but real):** the strictly-prior Hill v_c carries a small de-confounded ability
signal that beats the rating floor (held-out-form corr +0.066 ST / +0.033 HV vs +0.015 / −0.022),
and the de-confounded-cruise group is the one subset where the pure model out-sharpens production on
its winners (Δ−0.085 ST). The 0049 finding (the de-confounded LEVEL re-emerges as held-out ability)
holds — it is just the SLOPE-based `vc_hill` that is too thin to carry it into the win gate.

**Still missing (the gap to recovery):**
1. **The ability READ is too coarse.** `vc_hill` (across-distance slope, ≤3 prior distances on most
   horses) is far noisier than the de-confounded LEVEL (0049: +0.33/+0.40). The payoff path should
   carry the **de-confounded settled-cruise level** (the 0049 estimator) as the primary ability,
   with the Hill slope only where the distance span supports it — that is the single highest-leverage
   fix and it stays pure-mechanism.
2. **Deployment ≠ discrimination.** The field-coupled forward path fits the WITHIN-race velocity
   profile (0049) but the BETWEEN-horse ability spread it implies is compressed (drift clamped to
   12–19 m/s, σ_v ≈ a few tenths) — too tight to separate a 17.5% from a 1.9% win-rate horse. The
   generative law needs the ability variance the meta-signal stack supplies; physiology at this
   measurement precision under-spreads.
3. **Path-fit bar still FAILS vs the descriptive cell-mean** (field-coupled Hill RMSE 1.385 vs 0.456
   ST; 1.099 vs 0.407 HV, bias +1.10 / +0.88 m/s) — the Hill-CP-anchored forward integrator over-runs
   (the literature W′₀/P_max defaults + the static-de-confound CP, not the per-run 0047 inversion).
   Consistent with 0049's note that the static de-confound leaves a +0.7 m/s pace-following residual.
   This bar (already flagged 0047/0049) gates the ability read precision in (1).

## VERDICT
The pure-mechanism causal payoff test is RUN and reported, both venues, leak-free (drift 0). The
GoF fall is to the floor (ST 65.4%→−7.6%, HV 60.0%→+4.6%), driven by the stripped ability stack on
raced horses (winner −log p Δ+0.58/+0.53 on Hill horses) more than the debut concession. FORM does
NOT re-emerge at the win gate: the path Hill v_c beats the rating floor on held-out form (+0.066 vs
+0.015 ST) but sits far below the form ceiling (+0.382) and recovers only 7%/4% of the empirical
form→win gradient. The CAUSE is identifiable and weakly predictive; the meta-signals do NOT yet
re-emerge as consequences strong enough to price winners. The named next step is to carry the 0049
de-confounded LEVEL (not the thin Hill slope) as the ability and to restore the between-horse spread
the generative law lost. DON'T-BAKE — diagnostic measurement; no production constant changed; win%
measured, never optimized.

## DISCIPLINE
NO meta-signal in the model (the builder imports zero of `hkjc.model.features`; θ = Hill-CP /
de-confounded cruise / rating-conditioned population mean only). NO odds-gating — win% is MEASURED
on the gate, never optimized; coupling gains left at default. Form appears ONLY as a held-out
correlation/binning target, never a model input. Strictly-prior leak-free: `assert_no_lookahead`
drift = 0 both venues. Reused the production gate (`gate_score`) and the leak-free frame
(`studies/backtest_st.py::load`) verbatim — same judge the production model answers to. New files:
`studies/causal_reemergence.py` (+ the leak-free assert wired in). `pure_mechanism.py`, the
statespace physics, production `hkjc.model`, and `data_cache` UNTOUCHED. Lint clean.
