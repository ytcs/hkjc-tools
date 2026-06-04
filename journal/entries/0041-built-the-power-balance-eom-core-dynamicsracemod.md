---
id: 0041-built-the-power-balance-eom-core-dynamicsracemod
title: Built the power-balance EOM core (DynamicsRaceModel): mass/drag/drafting/reserve are now real terms; reduces to kinematic model with drag off
kind: infrastructure
status: open
tags: ["dynamics", "physics", "energetics", "eom", "mechanistic", "mass", "drafting", "model"]
refs: {}
related: []
created: 2026-06-04T10:49:13+00:00
updated: 2026-06-04T10:49:13+00:00
---
Step 1 of the mechanistic enrichment (docs/MODEL_ENRICHMENT.md): replaced the kinematic v_target core with a Newtonian power-balance equation of motion. hkjc.model.dynamics.DynamicsRaceModel, multi-agent, vectorised over samples; integrates m*dv/dt = P(t)/v - [K_LOCO*m*g + 0.5*rho*cda*v^2*(1-draft) + bend_loss], with P=P_c+anaerobic drawn from a finite reserve W' (joules) that depletes above the aerobic ceiling and forces the late fade. Physical params (P_c watts, W' joules, mass kg); from_kinematic(v_c, w_prime, mass) maps via P_c=K_LOCO*m*g*v_c and W'(J)=K_LOCO*m*g*w_prime. Constants G=9.81, RHO=1.2, K_LOCO=0.10. Keeps the traffic structure (car-following velocity cap, lane changes, bend ground-loss) but the velocity now EMERGES from the force balance; drafting is a real drag-relief force when sitting behind a leader.
Every factor now couples as a TERM, not a tacked-on covariate: weight = mass m in F=ma and locomotion; drafting = drag reduction; reserve exhaustion = fade; jockey/tactics will be the pace+positioning policy.
VERIFIED (tests/test_dynamics.py, 5 tests): (1) ACCEPTANCE - with cda=0 and even pace the finish times reproduce the kinematic T=(D-w')/v_c to 0.23% (reduces-to-old-core); (2) mass coupling - at FIXED intrinsic P_c, 470kg->90.8s vs 560kg->108.7s because v_c=P_c/(K_LOCO*m*g) (heavier mechanically slower); (3) reserve - W'=0 gives exactly D/v_c, W'>0 faster; (4) drag - higher cda slower, neutral at 0; (5) inner-draw bias still emerges from bend geometry. 119 tests pass, ruff clean, v1.11.0.
NOTE on mass: from_kinematic recovers v_c by construction (calibrated at that horse's race-day mass) - correct. The mass EFFECT (predicting a different mass) needs the INTRINSIC mass-invariant P_c, estimated by pooling P_c=K_LOCO*g*E[m_i*v_c_i] across a horse's runs at their various masses -> that is step 3 (couple state factors: body-weight as m, freshness as W' recovery). Next: (2) recover the pace policy P(t) from sectional profiles (where jockey lives), then (3) couple weight+freshness into on-the-day (P_c, W').
