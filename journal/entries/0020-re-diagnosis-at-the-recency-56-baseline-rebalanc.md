---
title: Re-diagnosis at the recency (56%) baseline rebalanced the ability gains: FORM_GAIN 4->5, W_PRIME_GAIN 2->1
id: 0020-re-diagnosis-at-the-recency-56-baseline-rebalanc
author: modelling
audience: ["all"]
channel: research
kind: decision
status: confirmed
thread: 
tags: ["recalibration", "form-gain", "w-prime-gain", "re-diagnosis", "recency", "wald", "parsimony"]
refs: {"FORM_GAIN": 5.0, "W_PRIME_GAIN": 1.0, "pct_market_full": 0.562}
created: 2026-06-05T18:53:03+00:00
updated: 2026-06-05T18:53:03+00:00
---
The recency-form switch changed the residual regime, so the gains baked for CAREER form went stale. Fresh market-residual diagnosis (SIM=wald): form flipped to UNDER-weighted (t=+11.4 at FG=4) because recency form is a sharper, higher-variance signal than the career mean; w_prime flipped to OVER-weighted (t=-8.95) because recency form ABSORBED the reserve signal. RECALIBRATED: FORM_GAIN 4->5 (clean interior optimum, +0.9pp full scale; 6-7 fall off a cliff); W_PRIME_GAIN 2->1 (win-prob impact flat over 0-2 -- W' is a finishing-margin effect, low-leverage on WHO wins -- so parsimony + the residual sign both say revert the amplification we added in the career era). diff_coef D=170 already optimal (sharper HURTS, so v_c_eb's under-trust is not a global-confidence problem). Re-diagnosis confirms: form neutralised (->-8.7, the FG optimum edge), v_c_eb +14.5->-4.0, w_prime ->+1.4. METHOD NOTE: SPEED_NOISE is a no-op under wald (wald uses diff_coef, not speed_noise) -- don't sweep it for the production model. Also fixed quick_eval determinism: polars unique() is not order-stable, so the test_cap subsample jittered ~1.7pp run-to-run; sorted the ids.
