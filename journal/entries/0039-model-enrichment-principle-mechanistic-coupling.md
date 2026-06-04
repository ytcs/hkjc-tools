---
id: 0039-model-enrichment-principle-mechanistic-coupling
title: Model-enrichment principle: mechanistic coupling only - upgrade kinematic core to a power-balance EOM (mass+drag+drafting); two entry points (state, policy)
kind: design
status: open
tags: ["enrichment", "physics", "energetics", "jockey", "fitness", "mechanistic", "roadmap"]
refs: {}
related: []
created: 2026-06-04T10:29:51+00:00
updated: 2026-06-04T10:29:51+00:00
---
Surveyed the full HKJC info surface (two recon subagents: jockey/trainer; fitness/trackwork/trials/gear/vet; see docs/MODEL_ENRICHMENT.md). User directive (correct): NOTHING tacked on - every factor must couple MECHANISTICALLY into the dynamics, not enter as a covariate/multiplier on the output.
RESOLUTION: exactly two mechanistic entry points. (A) Physiological STATE = the (mass m, aerobic ceiling P_c, anaerobic reserve W') that the equations of motion integrate; (B) Control POLICY = how the agent spends W' over the trip and navigates traffic/ground-loss. Every candidate routes through one of these.
ENABLER: the current core is KINEMATIC (sets v_target=v_c(1+W'/D), caps by traffic) and cannot accept mass or drafting. Upgrade to a per-horse Newtonian POWER-BALANCE EOM, integrated each step (the traffic sim already steps in time): m dv/dt = F_prop - F_resist; F_prop=P(t)/v; P=P_c+anaerobic (int<=W'); W' depletes/recovers; F_resist = locomotion(m) + aero-drag*(1-draft) + bend_loss. Then EVERY factor couples through a real term: weight=m in F=ma; freshness/trial/airway=on-the-day (P_c,W'); drafting=force coupling between agents; jockey=quality of the pace+positioning policy; blinkers=early-pace policy; vet=state gate. Trainer form = weak prior on condition, folded into the fitness state, NOT a side term.
Each addition must reduce to the prior model when its term is neutral (mass-invariant, no draft, neutral policy) - the test that it is coupling, not bolting. This is also the only edge-relevant path: a correct dynamical model makes sharp predictions for off-equilibrium pace/draw/weight/traffic configs the crowd integrates poorly. Build order in the doc: (1) power-balance core, (2) recover policy P(t) from sectionals, (3) couple state factors, (4) couple drafting + jockey-as-policy.
