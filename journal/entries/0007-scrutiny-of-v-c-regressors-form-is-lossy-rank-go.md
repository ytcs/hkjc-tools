---
title: Scrutiny of v_c regressors: form is lossy (rank), going is a blind spot
id: 0007-scrutiny-of-v-c-regressors-form-is-lossy-rank-go
author: modelling
audience: ["all"]
channel: research
kind: finding
status: open
thread: 
tags: ["scrutiny", "form", "speed-figure", "going", "critical-power", "methodology", "v_c"]
refs: {}
created: 2026-06-05T15:27:44+00:00
updated: 2026-06-05T15:27:44+00:00
---
Audit of effective-v_c terms. (1) FORM is the dominant signal but coupled through a CENSORED rank proxy (relative-finish): winning by a nose vs 10 lengths both give pos=1, discarding the margin = the velocity info. Mechanistic upgrade: a SPEED FIGURE from LBW (lengths behind winner) + FinishTime -> velocity deficit in m/s, coefficient ~1 (kinematic), normalised to a going/distance/class par. Data available in results_rich. Highest-value, cleanest. (2) GOING is a real v_c modifier (soft ground raises locomotion cost, differentially by horse) that the RESIDUAL DIAGNOSIS CANNOT SEE -- it's a race-level constant that vanishes under within-race renormalisation. Systematic blind spot of the whole loop; needs a calibration-by-going instrument, not the residual regression. (3) v_c estimator is a one-race min-velocity heuristic; principled version is the critical-power hyperbola D=v_c*T+W' across multi-distance runs. (4) JOCKEY is reduced-form + selection-confounded; a clean coupling would route it through trip geometry/policy not raw v_c -- leave for now. rating is well-incorporated (resid ~0). META: the residual instrument is blind to race-level physics (going/distance/class/track) by construction -- it will keep finding within-race mechanisms and miss condition mechanisms.
