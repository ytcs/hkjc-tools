---
title: Three-front push: kinematic-ABM-dead premise overturned, DRAW_FS (HV +0.3pp) banked, pace-interaction + debut-RTG refuted; barrier-trial backfill underway
id: 0026-three-front-push-kinematic-abm-dead-premise-over
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["abm", "interaction", "draw-fs", "crowding", "pace-pressure", "refuted", "conversion-test", "paired-seed", "method", "debut", "prior-starts", "barrier-trials", "new-data", "saturation", "hv"]
refs: {"HV_pct": 0.541, "ST_pct": 0.6, "kinematic_HV_delta_pp": -4.1, "draw_fs_HV_pp": 0.3}
created: 2026-06-05T22:42:06+00:00
updated: 2026-06-05T22:42:06+00:00
---
Three-front push (ABM revamp + new-data + debut). Net model change: HV 53.8->54.1% (+0.3pp), ST 60.0% unchanged. The real value was decisive INTELLIGENCE.

ABM FRONT -- premise OVERTURNED (CORRECTS 0025). The 'kinematic ABM beats wald at HV +2-3pp' claim was STALE -- it predated the per-venue wald bakes. At the current baseline the kinematic sim LOSES to wald at HV by -4.1pp (48.8 vs 52.9, paired quick_eval); its uncalibrated emergent-geometry draw is now worse than wald's tuned per-venue DRAW 0.008. Do NOT promote the slow JAX sim to production. The interaction axis instead pays as MEAN-FIELD shadow terms on the fast wald path (the ABM = theory generator, like pos_fade):
  * BAKED DRAW_FS_GAIN (crowding x draw): dg = draw_g*max(0,1+gain*(N-12)), a wide draw costs more in a crowded field (boxed-in), clamped >=0. PER-VENUE HV-only (HV 0.25 -> +0.3pp; ST flat). The clamp matters: the linear form REVERSED the penalty into a wide-draw bonus in small fields.
  * REFUTED relational pace-pressure (THEORY 3b #1): front-runner W'-burn scaling with rival front-runner count, driven by the converting early_pos signal, as a clean symmetric interaction (contested->fade, lone leader steals it, closers boosted). HV +0.12pp (noise), ST flat. The 4TH flat result on the pace axis (PACE_GAIN, contestedness, eom, pace_press). pos_fade already absorbs the convertible pace signal. Pace axis closed.

METHOD LESSON (important): the 150-race quick_eval WILDLY over-states a knob -- draw_fs read +1-1.6pp on quick_eval, +0.3pp HV/0.0 ST at full scale (~4x). And the full-backtest MC seed is hash(race_id), PYTHONHASHSEED-salted -> +-0.3-0.5pp run-to-run. FIX baked into method: confirm every candidate with a PAIRED full-scale test at PYTHONHASHSEED=0 (base+candidate share per-race seeds; delta is pure mechanism). quick_eval is for scouting sign only.

DEBUT FRONT -- RTG-suppression quick-win does NOT convert (built+tested e2e). Premise real (fab. RTG corr -0.04, anti-predictive) BUT: the v_c rating slope is tiny (~0.04 m/s over the debut spread vs FORM's ~2 m/s), the model is ALREADY calibrated on debuts (mean p 0.065 vs actual win 0.068), and ~35% of 'debutants' carry a REAL handicapper rating (HK rates first-timers off trials/breeding) so uniform suppression discards signal. A debut-jockey gain HURT (HV 33->28%). KEPT the leak-free prior_starts feature (committed) as the debut flag for the real fix. The market is efficient on debuts via existing signal -- debuts need NEW signal, not reweighting.

DATA FRONT -- barrier trials are the genuine next lever (verified LIVE). btresult (SSR HTML, ?Date=DD/MM/YYYY): clean horse_id join, history to 2018, inherently pre-race, trial field DOMINATED by debutants (11/19 sample = unraced foals). trial_time + 3 sectionals + lbw = a real v_c/W' velocity prior for first-timers (same energetics machinery). Prototype studies/newdata_probe.py + samples. A 2022-2026 historical BACKFILL is running -> data_cache/barrier_trials.parquet; NEXT: a leak-free trial-form feature + conversion test (esp. debut subset). Secondary: weather (continuous soil-moisture going proxy, GraphQL Weather op, poll-and-log) and formline (reconstruct collateral-form graph from our localresults).

CONVERGENT CONCLUSION: the mean-field/interaction terms on existing data are saturated (one HV +0.3pp term from a whole pass; pace axis closed). The next real % gain requires NEW DATA -- barrier trials first.
