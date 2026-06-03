---
id: 0010-results-trained-conditional-logit-model-breaks-t
title: Results-trained conditional-logit model breaks the circularity (works)
kind: result
status: open
tags: ["model", "conditional-logit", "circularity", "win-prediction"]
refs: {"modules": ["hkjc.model.conditional_logit", "hkjc.eval.harness"]}
related: []
created: 2026-06-03T21:51:44+00:00
updated: 2026-06-03T21:51:44+00:00
---
Step 4 shipped hkjc.model.ConditionalLogit: McFadden conditional logit, strength s_i=beta.x_i, within-race win prob = softmax; fit by MLE of actual winners (convex NLL, L-BFGS-B with analytic gradient X^T(p-y); no intercept since a race-constant cancels in softmax; features standardized + mean-imputed; L2 ridge). Market-implied prob is just ONE feature, so the model can diverge from and beat the market rather than reproduce it. conditional_logit_fit_predict plugs straight into the step-2 harness; strengths() feed PlackettLuceModel for Place/Quinella sim. Validated: trained on winners only (no odds), beats uniform by ~37% on multiclass race_win_log_loss out-of-sample across 4 walk-forward folds. NOTE: binary one-vs-rest log-loss can favour flat uniform out-of-sample (loser rows punish confident winners) — judge win models on the multiclass race_win_log_loss, the proper score. Model layer is now a package (base/plackett_luce/conditional_logit/baselines).
