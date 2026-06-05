---
title: Noise framing resolved: velocity-mult (=log-normal proportional time-noise) wins; statistical model can't do interactions
id: 0014-noise-framing-resolved-velocity-mult-log-normal
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["noise-model", "finishing-time", "plackett-luce", "interactions", "architecture", "parsimony"]
refs: {}
created: 2026-06-05T17:04:25+00:00
updated: 2026-06-05T17:04:25+00:00
---
Tested finishing-time noise per the standard-theory reframing. FULL-SCALE: velocity-multiplicative constant sigma_v = 48.3% (best). Finishing-TIME ADDITIVE: Gumbel->Plackett-Luce closed form peaks 46.4% @sigma_t=5s; Gaussian->probit 44.9% @8s -- both slightly WORSE. Distance-dependent noise: exp=0 (flat) best; exp=0.5 (the diffusion/first-passage sqrt(D) law) 47.7%, worse -- so the calibration-by-condition distance gradient was small-sample/AWT noise, NOT robust. Parsimony: keep sigma_v flat. RECONCILIATION: constant-sigma_v velocity model IS a finishing-time model -- log-normal noise PROPORTIONAL to time (sigma_T ∝ mu), the standard model for positive performance quantities; data prefers proportional over additive. KEY CAVEAT (user): the purely statistical framework (probit/PL/softmax) CANNOT mechanistically model interactions -- blocking, real-time tactics, jockey policy. That is the agent-based sim's reason to exist (the physics+game-theory core). CORRECTED ARCHITECTURE CONCLUSION: probit = FAST DEV SURROGATE for the WIN-calibration loop (35x, matches because interactions are low-leverage for win); the agent-based SIM stays as the production substrate for interactions/policy/exotics. Not a replacement. pl/time/probit kept as documented fit.py modes. Noise + statistical-reformulation levers EXHAUSTED. Remaining concrete lever: AWT 1650m badly miscalibrated (13%).
