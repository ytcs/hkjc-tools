---
title: "Queue #39: ability-fusion COLLAPSE (one precision-weighted v_c posterior) — DON'T-BAKE: the honest fused posterior loses −35pp ST / −32pp HV; the separate win-calibrated gains were each load-bearing beyond the PC1 precision weight, and de-contamination HURTS"
id: 0039-queue-39-ability-fusion-collapse-dont-bake-separate-gains-load-bearing
author: modelling
audience: ["all"]
channel: research
kind: decision
status: draft
thread:
tags: ["calibration", "fusion", "v_c", "ability-latent", "collapse", "pca", "measurement-model", "form", "collateral", "finishing-sectional", "decontamination", "running-style", "goodness-of-fit", "leak-free", "dont-bake", "queue-39", "queue-37"]
refs: {"study": "studies/ability_collapse_probe.py", "fit": "src/hkjc/model/fit.py", "constants": "src/hkjc/model/constants.py", "audit": "journal/entries/0037-vc-latent-factor-audit-catch-all-is-one-ability-factor-collapse.md", "st_legacy_pct": 0.6525, "hv_legacy_pct": 0.6005, "st_collapsed_best_pct": 0.3005, "hv_collapsed_best_pct": 0.2762, "st_ece_legacy": 0.0066, "hv_ece_legacy": 0.0046}
created: 2026-06-06T06:16:29+00:00
updated: 2026-06-06T06:16:29+00:00
---

QUESTION (refactor, gated — DEFAULT-OFF, nothing baked). The #39 factor audit (0037) proved v_c is
ONE over-measured ability latent: PC1 (~25%) fuses recency-form + speed-figure + collateral + the
sectional-v_c EB residual `u_vc` + the finishing-sectional into one factor; production enters them as
a STACK of separately-calibrated ADDITIVE gains (FORM_GAIN, SPEEDFIG, COLLAT_GAIN, SHRINK_MULT·u_vc)
that re-measure the same factor. The proposed refactor: collapse them into ONE exact precision-
weighted ability posterior (each instrument folded onto the v_c RESIDUAL scale via a train-only
affine map at its own precision ρ_k·π_race, sharing ONE denominator — the trial/gallop/finsec fusion
mechanism, extended), and DE-CONTAMINATE the sectional/finsec instrument for running style (the audit
found early_pos~finsec corr ≈0.53; a horse nearer the lead reads a spuriously higher closing
velocity). Win: same %market with FEWER free constants (3 retired), ideally a small de-contamination gain.

METHOD. `fit.fit_predict_traffic` gains a DEFAULT-OFF `ability_fusion="collapsed"` branch (legacy path
byte-identical; the legacy gate reproduces production 65.25/60.05 exactly). The collapsed posterior is
  û = ability_gain · [ n·π_race·u_vc_decon + Σ_k m_k·ρ_k·π_race·(a_k+b_k·s_k) ] / [ π_0 + n·π_race + Σ_k m_k·ρ_k·π_race ]
with form/speedfig/collateral as single CURRENT reads (m=1), trial/gallop/finsec as prior-count
instruments, each `(a_k,b_k)` a TRAIN-ONLY OLS of the v_c residual on the instrument (`fit_instrument_vc_map`).
`ability_gain` is ONE overall coupling scalar (the precision posterior fixes the relative SHAPE; this
sets the magnitude the WIN gate calibrates — the single constant that replaces the per-instrument
gain stack). De-contamination = a train-only OLS slope of the v_c residual on within-race-centred
early_pos, subtracted from the sectional/finsec reads. PAIRED fixed-seed (PYTHONHASHSEED=0), full-
scale, BOTH venues; `assert_no_lookahead` = 0.0 drift on the collapsed path at both venues.
`studies/ability_collapse_probe.py`.

== RESULT: the collapse CRATERS %market at BOTH venues (DON'T-BAKE) ==
PAIRED full-scale, best ability_gain (swept 3–30; decontam ON):

  venue | legacy %mkt  ECE    | collapsed best  ECE    | Δ%mkt   | ability_gain*
  ST    |   65.25%    0.0066  |   30.05%       0.0062  | −35.20pp |  4
  HV    |   60.05%    0.0046  |   27.62%       0.0072  | −32.43pp |  3

The loss is catastrophic and ROBUST: %market is MONOTONICALLY DECREASING in ability_gain above the
optimum (ST 30.0%@4 → 21.2%@8; HV 27.6%@3 → 16.6%@10 → NEGATIVE @15, v_c goes non-positive). So the
fused posterior shape is genuinely LESS win-discriminative, not merely under-scaled — amplifying it
amplifies the WRONG shape. Direct check (HV slice): within-race-centred `corr(v_c, won)` falls from
LEGACY 0.191 to COLLAPSED 0.149–0.154 (and v_c std 1.14 → 0.46–0.56). Even a form-ONLY collapse holds
corr at only 0.149. The honest precision posterior is a worse winner-predictor than the stack.

== WHY: the gains were win-calibrated, not v_c-residual-calibrated ==
The collapse maps each instrument onto the horse's own next-race v_c RESIDUAL (a noisy, biased
target). But the legacy gains are calibrated to the WIN conversion test, and that coupling is much
LARGER than the honest OLS-to-v_c-residual slope: the train-only form slope is b≈0.69, whereas legacy
FORM_GAIN=5 (the effective form coupling is ~7× the honest slope). Form predicts WINNING better than
it predicts a horse's own next sectional v_c residual; the same for collateral. So the additive gains
each do INDEPENDENT WIN WORK beyond what a v_c-residual-target precision fusion can carry. This REFUTES
the pure "one latent" picture at the margin: PC1 is one factor *descriptively* (the audit's correlation/
PCA is correct), but for the WIN order-statistic the channels are NOT redundant — each over-couples the
shared factor in a win-relevant direction the honest fused estimate discards. The audit's own drop-one
ΔR² had warned of this (collat/peakgap convert despite ~0 ΔR², "they sharpen the latent via the tails")
— the tail-sharpening is exactly the win-relevant over-coupling the honest mean-target fusion loses.

== DE-CONTAMINATION HURTS (running style carries real ability) ==
Removing the early-position/pace component from the v_c instrument makes the collapse WORSE at both
venues: no-decon beats decon by +2.19pp ST (32.23 vs 30.05) / +1.35pp HV (28.97 vs 27.62). The
early_pos~v_c correlation is NOT pure tactical contamination — front-running position in HK encodes
real win-relevant ability (gate speed, cruising speed, the jockey's read of the horse), and projecting
it out strips signal. The audit's "context leaks into ability" (PC1 contamination) is a real
DESCRIPTIVE fact, but the leaked component is win-PREDICTIVE, so de-contaminating the ability read for
WIN is net-negative. (This does not touch the separate `pos_fade`/W′ shadow couplings, which model a
DIFFERENT effect — the non-interacting-physics over-rating of front-runners — and stay where they are.)

== VERDICT: DON'T-BAKE. Keep `ability_fusion="legacy"` (production). ==
The collapse is a parsimony idea that fails the gate decisively (−35/−32pp), so the 3-free-constant
reduction (FORM_GAIN×2 + COLLAT_GAIN×2 + SHRINK_MULT → FORM_PI_RATIO + COLLAT_PI_RATIO) is NOT worth
buying — the separate win-calibrated gains are each LOAD-BEARING beyond the PC1 precision weight, and
de-contamination removes win signal. This is the "valid, important finding" PHILOSOPHY §5/§2 describes:
a dead conversion test rejects THIS STRUCTURE (the honest v_c-residual-target precision fusion as a
REPLACEMENT for the win-calibrated gains), not the descriptive one-latent picture nor the fusion
mechanism itself (trial/gallop/finsec still convert as ADDITIONS, because they bring a NEW read the
stack lacked — not a re-measurement of an already-win-calibrated channel).

The refactor lands as DEFAULT-OFF infrastructure only (`ABILITY_FUSION="legacy"` default; new no-op
constants FORM_PI_RATIO/COLLAT_PI_RATIO/SPEEDFIG_PI_RATIO/ABILITY_GAIN/FINSEC_DECONTAM read only on the
collapsed path). Production constants.py baked defaults UNCHANGED; the legacy gate reproduces 65.25/60.05.

LEAK-FREE. `assert_no_lookahead` drift = 0.0 (ST + HV) on the collapsed path; every affine map +
de-contamination slope + population fit is TRAIN-ONLY per fold; instruments strictly prior. The
catastrophic loss is a genuine mechanism result, not a leak artifact.

CAVEATS / what would change the verdict. (1) The collapse targets the v_c RESIDUAL; a fusion that
mapped instruments onto a WIN-relevant ability scale (e.g. a residual-to-win-logit calibration per
instrument) is a different structure not tested here — but that is no longer "one precision posterior",
it re-introduces a per-instrument win calibration (≈ the legacy gains). (2) ability_gain was swept
3–30 venue-shared; a per-instrument ρ sweep was not exhausted (form/collat/finsec=1, speedfig=0), but
the form-ONLY collapse already shows corr 0.149 ≪ 0.191, so no ρ reallocation recovers 35pp. (3) The
DESCRIPTIVE audit stands — this entry only refutes the COLLAPSE as a win-model refactor, confirming the
production additive stack is the right structure for the order statistic even though the factor is one.
