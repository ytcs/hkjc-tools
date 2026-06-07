---
title: Queue #30: training-gallop precision fusion — HV +0.36pp (baked rho_g=0.5), ST flat (0.0) [CORRECTED]
id: 0038-queue-30-training-gallop-precision-fusion-hv-2-5
author: modelling
audience: ["all"]
channel: research
kind: decision
status: corrected
thread: 
tags: ["calibration", "fusion", "gallop", "trackwork", "v_c", "goodness-of-fit", "per-venue", "leak-free"]
refs: {"constants": {"GALLOP_PI_RATIO_BY_VENUE": {"ST": 0.0, "HV": 0.5}}, "hv_pct_market": 0.6005, "hv_pct_market_base": 0.597, "st_pct_market": 0.652, "hv_ece": 0.0046, "trackwork_horses": 1090}
created: 2026-06-06T05:37:28+00:00
updated: 2026-06-06T05:37:28+00:00
---

> **CORRECTION (2026-06-05, orchestrator).** The original numbers below (HV +2.5pp, 59.7→62.2%) are
> WRONG. They were produced with a wiring bug: the gallop term applied the residual-trained affine
> slope to the **raw** race-scale `gallop_vc` (~16.5 m/s) instead of the residual `gvc − fixed_i`,
> injecting a spurious v_c bump (the same defect #32 caught and fixed for the finishing-sectional
> channel, flagged but left unfixed here). `fit.py` line ~306 was corrected to use `(gvc[i] − fixed_i)`.
> The **true** bug-corrected paired fixed-seed result is **HV +0.36pp (59.7→60.1%), ECE flat** at
> ρ_g=0.5, and **ST −0.41pp** (not −7.8pp — the bump was inflating ST damage too). The bake (HV=0.5)
> is KEPT as a marginal, leak-free, new-data win that does not hurt calibration, per an explicit
> keep/revert decision. Treat the body below as superseded by these numbers.
Queue #30: extended the coherent precision FUSION to training gallops — a THIRD, low-precision observation channel of the same latent v_c beside official runs + barrier trials, mirroring the baked trial fusion (TRIAL_PI_RATIO). NEW LIBRARY CODE: features.gallop_vc_table (leak-free per-horse gallop critical velocity on the RACE v_c scale: peak-furlong velocity -> par(surface,n_splits) figure -> re-centred on the race v_c mean, strictly-prior expanding mean + gallop_m count; backward as-of join, allow_exact_matches=False) and a pi_gallop channel in fit.fit_predict_traffic (gallop_pi_ratio rho_g, gallop_vc_gain), fused ABOVE the SHRINK_MULT layer exactly like the trial term, sharing the posterior denominator with the race/trial/finsec masses. The affine fit_trial_vc_map(signal='gallop_vc') puts the race-scale gallop_vc onto the v_c RESIDUAL scale (the slope absorbs the solo-gallop-pace != race-pace gap). DATA: ran the full backfill (studies/backfill_trackwork_full.py, RESUMABLE + checkpointing — the one-shot backfill got killed by the ~1h task limit, so I made it merge incrementally): data_cache/trackwork.parquet 350 -> 1090 horses / 94,637 gallops; coverage ST 44.3% / HV 38.0% of runners (was ~14%). leak-free confirmed (assert_no_lookahead drift 0.0 both venues).

STRUCTURE FIX during the gate: gallops accrue to m_g~200, so renormalising the precision weight to the m_g=1 debut (as the SPARSE trial channel does) blew a heavily-galloped horse up ~10x and drove some v_c<=0 (Wald mean<=0). The principled fix: a high-count channel uses the RAW posterior weight w_gallop in [0,1] directly — it self-saturates as gallop evidence grows and fades as race evidence n accumulates (the trial debut-renorm is only correct for the sparse channel).

PAIRED FIXED-SEED FULL-SCALE GATE (PYTHONHASHSEED=0, both venues, %market AND ECE):
  HV: 59.69% -> 62.18% (+2.49pp) at rho_g=0.5, ECE 0.0046 -> 0.0045 (BOTH improve). Broad stable plateau rho_g in [0.25,3] all ~+2.5pp (the weight self-saturates); ECE keeps improving to ~0.0041 at rho_g>=1 but %market is identical, so I baked rho_g=0.5 — inside the plateau, respects the noisy-channel LOW-precision prior, not an ECE edge. CONFIRMED on the production backtest_st.py path (baked constants, incl. the intervening finsec channel): HV 59.7 -> 62.2%.
  ST: DEAD and actively HARMFUL — rho_g=0.25 -> -7.8pp, monotonically worse with rho_g. The train-only affine slope is NEGATIVE at ST (b~-0.04): the gallop figure ANTI-correlates with the race v_c residual there. Baked ST=0.0 (a clean no-op — ST backtest 65.2% unchanged).

WHY HV not ST (the theory): exactly the task's hypothesis — HV's barrier-trial axis is SPARSE (TRIAL_PI_RATIO HV=0.0, trials never converted there), so the denser gallop prior is genuinely NEW v_c information at HV; at ST the race+trial+collateral channels already saturate v_c, and the noisy solo-gallop figure only adds anti-signal. A dead first attempt at ST != a dead axis (PHILOSOPHY §2): what to try next at ST below.

BAKED: GALLOP_PI_RATIO_BY_VENUE = {ST: 0.0, HV: 0.5}; gallop_vc_gain default 1.0 (the affine lands residual units). ruff + pytest clean (148 tests, +3 new tests/test_gallop_fusion.py).

NEXT ON THIS AXIS (a dead ST first attempt, not a dead ST axis): (1) RECENCY-weight the gallop expanding mean (exp half-life in sessions) — a stale gallop trace goes off, like recency-form/collateral; the current flat expanding mean over m_g~200 sessions is dominated by old work. (2) Restrict to FAST-WORK / barrier-trial-adjacent gallops only (the additive probe's fresh-work subslice converted 2x) — the slow pipe-openers are pure noise and likely the source of the negative ST slope. (3) A gallop -> D (diffusion/consistency) channel rather than v_c: gallop-figure DISPERSION as a per-horse D_i knob (THEORY §3d/§3e, the EVT field-size argument -> try ST big fields). (4) gallop -> W' (a long fast gallop is a reserve read), distance-aware. The DATA + leak-free join + fusion substrate are now in place for all four.
