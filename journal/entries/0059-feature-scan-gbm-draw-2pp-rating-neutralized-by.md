---
id: 0059-feature-scan-gbm-draw-2pp-rating-neutralized-by
title: Feature scan + GBM: draw +2pp, rating neutralized by handicap, kitchen-sink GBM overfits
kind: finding
status: open
tags: ["featurization", "draw", "rating", "gbm", "logit", "standing-order"]
refs: {}
related: []
created: 2026-06-04T22:48:45+00:00
updated: 2026-06-04T22:48:45+00:00
---
Post-pivot feature work on the flexible model (results_rich, 3248 races, EW+jockey baseline 41%):
  + draw_norm (raw barrier/field): 43%% (+2pp, real). draw_bias (empirical venue x dband x draw):
    nothing (too sparse). dist_form (distance suitability): nothing.
  Official RATING is a weak win-predictor: corr(rtg, won)=+0.036, winners avg only ~2 pts above
    losers -- the HANDICAP equalises it by design (higher mark -> more weight). Plus the
    rating_dataset is 58%% null on these horses. Not the lever I assumed; deprioritised.
  STRUCTURE upgrade tested: LightGBM (binary won + softmax-over-race). Kitchen-sink 15-feature GBM
    = 40%% < linear logit 43%% -- it splits on weak features (weight/days/trainer/style, all ~0 OOS)
    and overfits. Lesson: signal is concentrated in form_ew + jockey (+ a little draw); undisciplined
    flexibility hurts. The well-specified linear conditional logit on the strong features is still best.
Status vs 90%% target: form-only ~28%% -> EW+jockey ~41%% -> +draw ~43%% (full results_rich); ~53%% on
the cleaner ST-sectional subset. Next levers under test: lean GBM (do interactions help?), and a proper
track-variant speed figure from FinishTime (the principled version of the failed naive margin form).
