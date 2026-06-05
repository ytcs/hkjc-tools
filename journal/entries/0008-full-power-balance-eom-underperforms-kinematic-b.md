---
title: Full power-balance EOM underperforms kinematic by ~1.4 nats/race (win-prob)
id: 0008-full-power-balance-eom-underperforms-kinematic-b
author: modelling
audience: ["all"]
channel: research
kind: finding
status: confirmed
thread: 
tags: ["eom", "mass", "allometric", "negative-result", "physics-migration", "steady-state"]
refs: {}
created: 2026-06-05T15:45:33+00:00
updated: 2026-06-05T17:57:29+00:00
---
Tried the most-fundamental mass coupling: run the power-balance EOM (hkjc.model.eom) in production so declared mass enters every term via Newton's law, with one allometric constant b. Built dynamics.win_probs + a SIM='eom' path in fit.py (kinematic-calibrated gains FORM_GAIN=4, W_PRIME_GAIN=2). Result on a 42-race slice: EOM winner-logp -3.68 to -4.38 vs kinematic -2.30. Removing drag (cda=0) recovers only ~0.7; lowering tau/noise doesn't help. The EOM's dynamics (accel phase, W'-depletion fade, power smoothing) compress the ability spread the kinematic sim preserves. KEY INSIGHT: the (m/m_field)^(b-1) v_c factor is NOT an arbitrary tack-on -- it is the EOM's exact STEADY-STATE solution (v_c=p_c/(K_LOCO*m*g), p_c=k*m^b). It captures the win-determining mass->v_c physics at kinematic accuracy; the full integrated EOM only adds second-order transients that are currently mis-calibrated and HURT. Making the full EOM competitive needs a from-scratch recalibration (gains up, noise down, fade/drag retuned) starting 1.4 nats behind, uncertain it matches. RECOMMENDATION: bake the steady-state allometric law (first-principles, ready now); treat the full dynamic-EOM migration as a separate greenlit project.
