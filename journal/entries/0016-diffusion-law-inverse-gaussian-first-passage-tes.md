---
title: Diffusion law (Inverse-Gaussian first-passage) tested: ties the in-place log-normal noise
id: 0016-diffusion-law-inverse-gaussian-first-passage-tes
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["diffusion", "inverse-gaussian", "wald", "first-passage", "noise-model", "unification", "foundation"]
refs: {}
created: 2026-06-05T17:21:38+00:00
updated: 2026-06-05T17:21:38+00:00
---
Per the diffusion-law correction: finishing time = first-passage of dx=v·dt+sqrt(2D)·dW => INVERSE GAUSSIAN (Wald), strictly positive, Var(T)=2D·L/v^3. Implemented sim='wald' (one physical param, diffusion coef D). FULL-SCALE sweep: peaks D~160-180 -> 47.7%, vs velocity-multiplicative (=log-normal time) 48.2% -- a TIE within noise. Both are correct positive-time models, both one numpy call (equal efficiency); fit marginally favors the log-normal already in place. CONCLUSION: the noise FOUNDATION is validated -- the principled diffusion law agrees with the in-place model to ~0.5pp, so adopting the diffusion framing later (for interactions) costs nothing. UNIFYING PICTURE: each horse = drift-diffusion particle (drift=ability, diffusion=D). Non-interacting limit = independent Inverse-Gaussian finishing times (the fast WIN surrogate). WITH interactions (exclusion=blocking/TASEP, coupling=drafting, policy=mean-field game) = the agent-based sim is the Monte-Carlo of the SAME many-body drift-diffusion. So the fast surrogate and the interacting sim are one model at different interaction orders -- the substrate for the stat-mech program. Keep log-normal noise for now (marginally better, in place); diffusion is the principled foundation for the interaction layer. fit.py keeps probit/pl/time/wald as documented modes.
