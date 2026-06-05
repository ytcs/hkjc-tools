---
id: 0061-trajectory-dataset-policy-w-depletion-measured-p
title: Trajectory dataset + policy/W-depletion measured; pace-mispricing probe: cannibalisation right-sign but underpowered
kind: result
status: open
tags: ["trajectory", "policy", "W-prime", "depletion", "epistemic-arbitrage", "pace", "mispricing", "market-residual", "standing-order", "roadmap"]
refs: {}
related: []
created: 2026-06-04T23:29:34+00:00
updated: 2026-06-04T23:29:34+00:00
---
ROADMAP STEPS 1-2 DONE + epistemic-arbitrage probe (pace mispricing). Built the foundational
per-section trajectory dataset for all 3248 races (build_trajectory_full.py -> trajectory_dataset_full.parquet,
143050 section-rows, velocity+position, zero null) and measured per-run POLICY + observed W'-depletion
(measure_policy.py -> run_policy_dataset.parquet).

MECHANISM VALIDATED (the measures behave exactly as the EOM predicts):
- depleted (fade + late position-loss = tank-empty): winners 0.000 vs losers 0.331. Winners NEVER show
  observable W'-depletion; a third of losers do. Directly-observed reserve exhaustion separates outcomes.
- effort_dtg (distance-to-go centroid of velocity-excess-over-v_c = empirical analog of EOM kick_frac):
  winners 0.466 vs losers 0.511 (winners deploy slightly LATER; sit-and-pounce > early-burn, matching the
  over-aggression-self-defeating sim finding). corr(early_pos, effort_dtg)=-0.23 (front-runners spend early,
  measure is mechanistically correct). corr(effort_dtg, trailing effort_dtg)=+0.22 (pace/kick timing is a
  STABLE leak-free per-horse trait -> a legitimate policy INPUT, not purely race-contingent).

ORTHOGONALITY SCREENS (the gate rel_early failed) -- all ~0 over EW-form+jockey (41% of market):
- policy traits (trail_depleted/fade/effort_dtg/early_pos/kick_frac): each +0..+0.005 skill, ALL together
  42%. Replicates the thesis: style is emergent/positional, must live in the SIM policy, NOT a flat covariate.
- weight/mass mechanism (ActWt, DeclarHorseWt, burden_ratio=ActWt/bodyWt, weight deltas): each ~0, ALL 41%.
  The handicapper is efficient and the market prices it; even the body-weight-ignored burden ratio adds nothing.
  NOTE: traffic_jax (the fast backtest model) is KINEMATIC -- it has NO mass term (mass only in the unused
  numpy DynamicsRaceModel). So mass is a real EOM term sitting unused, but the screen says no flat juice.

EPISTEMIC-ARBITRAGE PROBE (does the MARKET misprice the pace/field interaction?). Accept market win-prob as
the ability estimate; test whether the residual (won - mkt_prob) is signed by field pace config.
- Crude buckets: hot pace (>=3 trailing-front-runners), the front-runners win 0.59x market-implied
  (resid -0.028) and closers behind them 1.09x -- BOTH the predicted cannibalisation sign. But only ~15 such
  races in 5 seasons (trailing-mean style regresses to mid, starving the bucket); n=50, ~1.1 SE. Not established.
- Powered full-data cluster-robust OLS (n=36345, by-race clusters): own_front +0.0033 t=+2.49 (front-runners
  modestly BEAT market -- the known early-speed underbet bias, but economically small & already ~priced in the
  logit); rivals_front -0.0012 t=-2.18; own_front x rivals_front (the cannibalisation interaction) -0.0006
  t=-0.52 -- RIGHT SIGN, NOT SIGNIFICANT. Did not over-bucket to manufacture significance.

VERDICT: every additive signal is exhausted (form+jockey 41%, nothing adds). The one large real mechanism
(pace cannibalisation) is directionally mispriced by the market but too rare/small to establish at 5 seasons.
This MOTIVATES roadmap step 3 with a specific payoff: a game-theory pace-equilibrium would identify the
pace-collapse races far better than a crude trailing-front-runner count (which finds only 15), restoring the
statistical power to test/trade the cannibalisation mispricing. NOT defeatism -- a data/identification wall,
and the flywheel is accruing more races. Anti-pattern intact: do not read the null interaction as 'tactics
dead'; read it as 'crude pace identification is underpowered; the sim is the better identifier'.

Next: (step 3) best-response pace equilibrium -> per-race pace-pressure index from the SIM (not a count) ->
re-test the market-residual mispricing with the sim-identified hot-pace races. If the sim sharpens the
identification, the 0.59x bucket may firm up into a tradeable structural edge (fade sim-flagged hot-pace speed).
