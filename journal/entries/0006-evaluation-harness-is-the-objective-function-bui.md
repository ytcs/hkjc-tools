---
id: 0006-evaluation-harness-is-the-objective-function-bui
title: Evaluation harness is the objective function — build it before optimizing
kind: decision
status: open
tags: ["evaluation", "harness", "walk-forward", "lockbox", "fdr"]
refs: {"modules": ["hkjc.eval.calibration", "hkjc.eval.walkforward", "hkjc.eval.selection", "hkjc.eval.harness"], "doc": "docs/AGENT_ARCHITECTURE.md#3"}
related: []
created: 2026-06-03T21:32:46+00:00
updated: 2026-06-03T21:32:46+00:00
---
Step 2 built hkjc.eval into the loop's objective function: calibration (log-loss/Brier/ECE/RPS + multiclass race_win_log_loss), walk-forward folds with an embargo measured in MEETING-DAYS (not calendar days), a sealed Lockbox that enforces read-once via a persisted access log, and multiple-testing control (Benjamini-Hochberg FDR + Deflated/Probabilistic Sharpe). The harness owns splitting+scoring; the model supplies fit_predict(train,test)->preds. That boundary keeps the judge un-gameable from inside the model.
