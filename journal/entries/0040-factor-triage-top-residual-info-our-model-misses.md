---
id: 0040-factor-triage-top-residual-info-our-model-misses
title: Factor triage: top residual info our model misses is recent-form/ability (state) and jockey (policy); trainer ~0, rating absorbed
kind: finding
status: open
tags: ["triage", "residual", "jockey", "energetics", "fitness", "enrichment", "priorities"]
refs: {}
related: []
created: 2026-06-04T10:39:49+00:00
updated: 2026-06-04T10:39:49+00:00
---
Cheap in-sample univariate triage (experiments/triage_factors_st.py): for each omitted factor, the drop in binary log-loss when added on top of log(p_traffic) [info our model misses] and log(p_market) [edge check]. Factors joined from cached form_dataset (~56% coverage); jockey/trainer = leave-one-out shrunk strike rate. Millinats/runner, ranked by dLL-over-model:
  avg_finish_prior3  9.40m  (corr_resid -0.130, cov 415)
  jockey_wr          6.52m  (corr +0.125, cov 984; over-market 0.41m - largest market residual)
  win_rate_prior     4.20m  (cov 415)
  days_since_run     1.46m  (over-market 1.75m > over-model - freshness maybe underpriced)
  wt_delta           1.34m  (cov 415)
  draw (in model)    1.04m  (model does NOT fully capture its own draw effect)
  body_wt            0.76m
  rating             0.06m  (ABSORBED - energetics fixed effect already prices it)
  trainer_wr         0.04m  (negligible)
  v_c_eb (in model)  0.00m  (sanity: it IS the model)
SANITY: v_c_eb->0 and rating->~0 confirm the triage is calibrated (the model already absorbs what it should).
INTERPRETATION mapped to the two mechanistic entry points:
(A STATE) Recent-form/ability carries the most residual (avg_finish_prior3 9.4m, win_rate_prior 4.2m) - our sectional-derived (P_c, W') state is INCOMPLETE: the model under-captures demonstrated ability. Biggest lever = improve the physiological STATE estimate (power-balance EOM re-deriving (P_c,W',m); more sectionals; couple body-weight as mass m and freshness/days-since-run into on-the-day state). Note form is a symptom not a mechanism - the mechanistic translation is 'sharpen state estimation'.
(B POLICY) jockey_wr 6.5m is the single clearest omitted MECHANISM = control policy (pace/positioning). It also holds the largest over-market residual (0.41m), consistent with being mostly-but-not-fully priced.
DEPRIORITISE: trainer (~0 over model), body-weight level, rating (absorbed). Freshness + weight-delta are modest but real and freshness may even carry market residual.
CAVEAT: in-sample, univariate, uneven coverage (form factors on 415 rows vs jockey 984) - a PRIORITISATION, not a backtest. PRIORITY ORDER for the mechanistic build: (1) state quality (EOM core + weight/freshness coupling), (2) jockey-as-policy, (3) drafting; trainer last. Confirms: include everything eventually, but state-estimation and jockey first.
