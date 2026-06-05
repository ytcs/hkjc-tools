---
title: W_PRIME_GAIN recalibrated 1.0 -> 2.0 (anaerobic reserve under-weighted)
id: 0005-w-prime-gain-recalibrated-1-0-2-0-anaerobic-rese
author: modelling
audience: ["all"]
channel: research
kind: decision
status: confirmed
thread: 
tags: ["calibration", "w_prime_gain", "goodness-of-fit", "reserve"]
refs: {"constants": {"W_PRIME_GAIN": 2.0}, "st_pct_market": 0.491}
created: 2026-06-05T11:57:33+00:00
updated: 2026-06-05T11:57:33+00:00
---
Residual-queue #2: the ABM over-channelled ability through critical velocity (v_c/form OVER) and UNDER-weighted the W' reserve (w_prime_eb residual t=+13.2). Added W_PRIME_GAIN (multiplier on the per-horse EB reserve residual, the stamina analogue of FORM_GAIN) in hkjc.model.constants, applied in fit.py. Subsample sweep peaked broadly 1.5-2.5 (3.0 over-amplifies; W' is noisy). FULL-SCALE confirm: ST 47.7% -> 49.1% of market (LL 2.2576->2.2500, ECE 0.0074->0.0081). MECHANISM CHECK (re-diagnosis at gain 2.0): w_prime residual HALVED t +13.2->+6.4 and the coupled v_c over-weight fell t -8.9->-7.0 -- the ability channel rebalanced from critical-velocity toward reserve, exactly as the mechanism predicts. Baked 2.0. Next lever: mass (bodywt t=+8.6), now addressable via the single-sourced EOM. HV re-confirm pending.
