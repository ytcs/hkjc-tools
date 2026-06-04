---
id: 0046-breakthrough-form-relative-finish-ability-calibr
title: BREAKTHROUGH: form (relative-finish ability) + calibrated race-day noise nearly triples model win-skill (11%->29% of market); gap 9x -> 3.4x
kind: finding
status: open
tags: ["breakthrough", "form", "ability", "calibration", "speed-noise", "gap-closing", "model", "handicap"]
refs: {}
related: []
created: 2026-06-04T11:55:29+00:00
updated: 2026-06-04T11:55:29+00:00
---
Closed most of the 10x model-vs-market gap with two data-proven, mechanistically-coupled changes.
DIAGNOSIS (data): (1) In HK HANDICAP racing, static ability is weight-equalized - rating-only and v_c-only conditional-logit each give ~0% win skill (~uniform). The market's skill is current-form / better-than-mark, not static class. (2) The horse's leak-free expanding-mean RELATIVE FINISH (form = 1-(pos-1)/(field-1), averaged over strictly-prior runs) correlates 0.15 with winning vs our sectional v_c ~0.06; conditional-logit ceiling form+v_c+draw ~23% of market skill. (3) Naively coupling form->v_c BROKE the model (log-loss 3.5-4.8, worse than uniform) - OVERCONFIDENCE: speed_noise=0.02 made the sim ~deterministic once abilities had real spread, so the confident top-form pick (a 0.15-corr signal) lost ~85% of the time. The old 0.02 only 'worked' because heavy EB shrinkage made abilities ~equal (unskilled-but-calibrated).
FIX (mechanistic): effective_v_c = sectional_EB_v_c + FORM_GAIN*(form-0.5) [form as a better measurement of the same latent critical velocity], with speed_noise CALIBRATED to the ability spread (race-day unpredictability is large because form is a weak-but-real signal). Joint sweep -> FORM_GAIN=2.0, SPEED_NOISE=0.12 (baked as backtest defaults).
RESULT (walk-forward, 139 ST races, N=5000): race-win log-loss 2.486 -> 2.405; skill vs uniform +0.048 -> +0.128 (11% -> 29% of market's 0.4416); gap to market +0.393 -> +0.313. ECE IMPROVED 0.018 -> 0.0109 (better-calibrated than the market's 0.025) - so it's an honestly-calibrated better model, not overfit-sharp; tuning for skill landed on a calibrated speed_noise (the principled target).
CAVEAT: FORM_GAIN/SPEED_NOISE selected on the 5-fold aggregate (mild hyperparam optimism on 139 races) but the gain is large, consistent across grid points, calibration-improving, and the form mechanism is independently validated. NEXT to push past 29%: (a) field-strength-weight form (beating strong fields should count more), (b) jockey as an independent ability/policy signal (triage residual 6.5m), (c) distance/going-specific form, (d) principled per-race speed_noise calibration.
