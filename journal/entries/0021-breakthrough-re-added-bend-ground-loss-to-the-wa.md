---
title: BREAKTHROUGH: re-added bend ground-loss to the wald drift (DRAW_GAIN=0.003) -> ST 56->58% of market
id: 0021-breakthrough-re-added-bend-ground-loss-to-the-wa
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["draw", "bend-ground-loss", "wald", "breakthrough", "interaction", "abm", "re-diagnosis", "first-principles"]
refs: {"DRAW_GAIN": 0.003, "pct_market_full": 0.581, "ece": 0.006}
created: 2026-06-05T18:53:03+00:00
updated: 2026-06-05T18:53:03+00:00
---
The #1 residual at 56% was draw (t=-26.5). ROOT CAUSE: production wald (and pl) had NO draw term at all -- draw/bend_curvature lived ONLY in the kinematic geometry sim, and was silently DROPPED when wald became production. (The naive 'OVER-weights' label was wrong: the model had zero draw signal while the market prices the inside-draw advantage.) FIX (physics, not a regressor): a horse drawn k lanes off the rail runs a longer arc round the turns -> larger effective distance -> longer first-passage time. L_eff = dist*(1+DRAW_GAIN*(draw-1)) in the wald/pl drift. Geometry fixes the SIGN; the magnitude is calibrated and absorbs jockeys steering toward the rail. Clean monotone sweep, broad peak 0.003-0.004; baked 0.003 -> +1.9pp full scale (56.2->58.1%), residual collapsed t=-26.5->-1.5, model ECE 0.0060 (better than market's 0.0069). SANITY CHECK: 0.003/lane => draw-14 vs draw-1 runs ~3.9% (~62m) longer over 1600m -- the right order for the extra arc of being ~13 lanes wider round the turns, so the calibrated magnitude lands in the geometric ballpark (physical, not a fudge). INTERACTION (per user: draw is a proxy for geometry AND collision -- horses are not collisionless, the ABM's original motivation): the residual has a field-size GRADIENT (small fields over-penalised, crowded under) -- the collision fingerprint. But a mean-field field-size modulation (draw_fs_gain, wired+unbaked) recovers only ~0.1pp; the full draw x crowding / boxed-in / draw x pace structure is genuinely the agent-based sim's job (place/exotics). LESSON: a 'parked / low-leverage' verdict can go stale when the production sim changes -- always re-check parked items against the CURRENT production path. bodywt (t=+9.6) is the next such re-check.
