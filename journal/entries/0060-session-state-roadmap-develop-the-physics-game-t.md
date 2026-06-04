---
id: 0060-session-state-roadmap-develop-the-physics-game-t
title: SESSION STATE + ROADMAP: develop the physics+game-theory ABM, NOT statistical models
kind: milestone
status: open
tags: ["roadmap", "ABM", "game-theory", "policy", "standing-order", "session-state", "direction"]
refs: {}
related: []
created: 2026-06-04T22:55:32+00:00
updated: 2026-06-04T22:55:32+00:00
---
=== CRITICAL DIRECTION (user course-correction, do not relitigate) ===
The physics + game-theory first-principles ABM is the RIGHT structure. Statistical models
(conditional logit, GBM) are a MECHANISM-FREE PLATEAU and a dead end. Evidence this session:
EW-form+jockey+draw logit ~43%% (full results_rich) / ~53%% (ST-sectional subset); a LightGBM was
WORSE (40%%<43%%) because the relationships are ~linear and flexibility just overfits noise features.
The logit asymptotes because it has no representation of the race happening (pace, energy depletion,
tactics, horse-horse interaction) -- exactly where the remaining market gap lives, and exactly what
only a generative physics+game-theory model produces.

ANTI-PATTERN (reinforced): do NOT read "a 2-feature logit currently out-predicts the ABM" or "the
ABM field-interaction adds nothing in the hybrid probe" as "abandon the ABM". Those are verdicts on
the ABM'''s CURRENT rung-1 OPEN-LOOP policy layer (PACE_GAIN=0, kick noise-dominated), which does not
model best-response tactics yet. That is the UNBUILT frontier, not a refutation. I went in circles
chasing statistical featurization (margin/speed-fig, rating, draw, GBM) before the user corrected me.

=== VALIDATED THIS SESSION (keep) ===
- Jockey confirmed at Happy Valley +9pp (replicates ST +7pp), JOCKEY_GAIN=5.0 baked at both venues.
- FULL-HISTORY DATA UNBLOCK (the big infra win, serves the mechanistic path): the ABM was scope-
  capped to 161 races by a hardcoded 3-month build; sectionals exist for all 5 seasons. Built
  energetics_dataset_full + wpq_dataset_full (3248 races, 2911 horses) via build_energetics_full.py
  + build_wpq_full.py; ENERGETICS_FILE/WPQ_FILE env vars, now the DEFAULT in backtest_traffic_st.
  Lifted ST ABM ~38%% -> 43.7%% of market with zero modeling change.
- Per-section POSITION + VELOCITY trajectories exist for every runner (3248 races) and are the raw
  material for the policy/energy layer -- currently thrown away (ABM keeps only v_c, W'). Verified
  reconstruction: winners sustain/kick (fast final section from the front), faders show W'-depletion
  (velocity collapse + position drop in the last section) DIRECTLY OBSERVABLE. section_distances
  helper is correct (first = D-400*(n-1), rest 400m); earlier -350 was my wrong-distance manual error.
- Negative/weak results (do not re-chase): margin/speed-figure form (rank dominates), official rating
  (handicap-neutralised, corr +0.036, 58%% null), trainer/weight/body-wt/days-since/running-style
  (~0 over form), distance-suitability form, empirical draw-bias. Recency EW form (+8pp in bare logit)
  is subsumed by rating inside the ABM. Raw draw adds +2pp in the logit.

=== ROADMAP (next session) ===
1. TRAJECTORY DATASET: per-section velocity + position for all 3248 races (extend the sectional scrape
   to keep per-section detail, not just v_c/W'). Foundational, design-agnostic.
2. MEASURE policy + energy per run from trajectories: effort distribution (early/sustained/kick
   timing), position trajectory (front/stalk/close), and directly-observed W' depletion (the fades).
   Sharper than the current noisy 2-scalar fit; gives per-horse policy tendencies (leak-free trailing).
3. GAME-THEORY LAYER: model each jockey'''s pace/position/kick as a BEST-RESPONSE to the field (others'''
   styles, draw, own v_c/W'), find the equilibrium pace scenario, calibrate against observed pace
   dynamics, couple into the physics EOM + traffic (rungs 2-3 of the policy ladder; infra already in
   traffic_jax/dynamics: pace_bias/pace_gain + kick_frac/kick_overspeed). Field interactions EMERGE.
4. TEST: does the policied ABM capture the field-interaction signal the open-loop ABM could not (the
   probe showed it currently nets to zero), and close more of the gap to market.
Standing order intact: >=90%% of market on public data via the correct MECHANISTIC structure; walls are
the norm, no defeatism.
