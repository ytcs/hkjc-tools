---
title: Model lineup: drop probit (velocity-mult); Wald (diffusion) is production, PL the fast probe
id: 0019-model-lineup-drop-probit-velocity-mult-wald-diff
author: modelling
audience: ["all"]
channel: research
kind: decision
status: confirmed
thread: 
tags: ["wald", "diffusion", "probit", "plackett-luce", "model-lineup", "jax", "n_samples", "first-principles"]
refs: {}
created: 2026-06-05T18:17:38+00:00
updated: 2026-06-05T18:17:38+00:00
---
Per the first-principles mindframe, dropped the probit (velocity-multiplicative) win model -- an odd in-between: noise on velocity not the physical finishing-time observable, and a one-shot draw not a diffusion. Also dropped the additive-Gaussian 'time' mode (allows T<0). LINEUP now: sim='wald' (DEFAULT, the production win model -- drift-diffusion first-passage / Inverse-Gaussian, the principled noise law, one physical D, surface-dependent AWT 2x, ~55.5% of market); sim='pl' (closed-form Plackett-Luce, instant, the fast-and-dirty probe); sim='kinematic' (JAX traffic sim, for INTERACTIONS/exotics only -- traffic low-leverage for win); sim='eom' (inert infra). WALD SPEEDUP (user asked, JAX available): JAX only 1.2x on CPU (no GPU; the IG sampling is memory-bound; a single batched (races,ns,n) call is WORSE, 195s). The real quick win is n_samples: 4000->55.3%, 8000->55.5%, 20000->55.5% -- ~3x faster at 4k for <0.2pp. Production N_SAMPLES=6000 already fine. FURTHER OPTION (not done): the IG is near-Gaussian here (mu/lam~0.02), so a sampling-free Gaussian-approx probit with the diffusion variance could be ~instant -- an approximation to verify. Studies repointed probit->wald; MODEL_STATE + SIM constant updated.
