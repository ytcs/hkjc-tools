---
title: task #97 physics (EOM-shaped) longitudinal kernel for the 4-feed fusion smoother
id: 0094-task-97-physics-eom-shaped-longitudinal-kernel-f
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060
tags: ["cv", "fusion", "smoother", "eom", "longitudinal", "physkernel", "task97"]
refs: {}
created: 2026-06-07T14:09:24+00:00
updated: 2026-06-07T14:09:24+00:00
---
TASK #97 — replace the white-acceleration Gaussian LONGITUDINAL motion kernel of the 4-feed fusion smoother with a PHYSICS (EOM-shaped) kernel. BASE = #96 cv_fusion_constrained (hard lateral box + joint ordering + geometry down-weight + draw anchor, all kept intact). Study studies/cv_fusion_physkernel.py; paired WHITE vs PHYSICS on R01/R03/R08.

VERDICT (honest, both ways): the physics kernel is a REAL but MODEST win in the form it can defensibly take — and the attempt surfaced a sharp STRUCTURAL lesson about why a strong shape prior CANNOT do more here. NO regression on any validated deliverable. It is NOT a transformative reconstruction upgrade; it smooths the recon and pins the gate transient without adding longitudinal information.

THE EOM SHAPE I ENCODED (from src/hkjc/model/dynamics.py, verified by solving solve_eom for v_c~17): v_EOM(t) = v_c*ramp(t)*(1 + d_fin*kick(arclen)); ramp = 1-exp(-t/2.5) (gate-ramp 0->v_c, ~0.95 v_c by ~6-7s == the EOM opening transient); kick = a localised half-cosine bump over the last 12% of the distance (the single signed W'-kick/fade d.o.f.). Levels (v_c, d_fin) fit PER-HORSE from that horse own arclen feeds in a pre-pass (NO per-horse physiological theta, NO forward-sim, NO other horse, NO outcome) -> the guardrail-2 no-circularity design. Mechanism: the longitudinal velocity sub-state is given an OU mean-reversion toward v_EOM(t) (affine Kalman input), replacing the free white-accel random walk; lateral channel UNTOUCHED (#96).

THE STRUCTURAL LESSON (the §4b honesty the user wanted). A GLOBALLY strong velocity pull (the first thing I tried, tau_revert~1.2s) DID collapse the squiggle (gap-roughness 56->21 on R03) but BROKE the order: R03 winner LOST, late-pole rho +0.10->-0.06, R01 finish-rho +0.73->+0.38. ROOT CAUSE (diagnosed, not a tuning miss): the per-horse VELOCITY DIFFERENCES *are* the running-order signal the RF rank feed carries -- a horse moving through the field (winner climbing 7->1) is exactly a per-horse velocity excess. A stiff shared-cruise template (all horses fit v_c~16.8) smooths those differences away == it erases the overtaking. The squiggle and the order are the SAME degree of freedom; you cannot smooth one without destroying the other with a shared-shape prior. A sweep confirmed no global-pull setting keeps winner 3/3 AND materially smooths -- the winner-safe regime (tau_revert>=8s) is cosmetically inert (gap-roughness ~= white).

THE DEFENSIBLE FIX (shipped): a PHASE-DEPENDENT pull. STRONG EOM-shape pull only in the GATE-RAMP window (first ~7.5s == 3*tau_ramp, where every horse is accelerating from the gate and there is NO running-position signal yet -- first official call is 200-450m), then WEAK pull in cruise/finish (sigma_a 0.55 < white 0.8, light damping that leaves the order-bearing velocity differences alone). Physics pins the opening transient (the genuine 0-20s artifact); the data keeps the mid-race.

RESULTS (WHITE -> PHYSICS; smoothness measured POST-ramp so the real gate transient is not miscounted):
  R01 mile : cruise vel-TV 17.5->12.5, post-ramp 20s swing 2.7->1.8, gap-roughness 6.6->4.3. ALL smoother.
  R03 sprint: vel-TV 16.0->14.3, swing 8.3->7.5, gap-roughness 55.9->38.2. (cruise jerk RMS +0.12 -- minor.)
  R08 route : vel-TV 21.9->19.1, swing 11.1->8.9, gap-roughness 46.5->32.8. ALL smoother.

NO-REGRESSION battery (must NOT degrade -- all PASS):
  winner recovery 3/3 HELD (R01 #3, R03 #7, R08 #6).
  finish-order rho: R01 +0.73->+0.71 (flat), R03 +0.36->+0.36, R08 +0.26->+0.45 (up).
  per-pole order rho: R01 +0.95/+0.82/+0.88/+0.71 -> +0.96/+0.82/+0.88/+0.69 (flat); R03 +0.99/+0.99/+0.10 -> +0.99/+0.99/+0.15; R08 +0.96/+0.95/+0.98/+0.94/+0.60 -> +0.97/+0.96/+0.98/+0.91/+0.66. No material drop.
  lateral rail-vs-wide sign 3/3 HELD (+6.66 8/8 / +6.57 8/10 / +3.50 3/6), gaps unchanged (lateral untouched).
  finish-margin MAE (full-field len): R01 2.59->2.84 (slightly worse), R03 2.68->2.63, R08 2.25->2.00 (improved). Within noise.
  ROBUSTNESS: a +/-30% sweep of the cruise-pull params keeps winner 3/3 and min-pole rho positive on all 3 -- the no-regression guarantee is not a knife-edge (only the smoothing MAGNITUDE varies; R01 gap-roughness is the sensitive one, 4-51 across settings).

FAILURE MODES / honest limits:
  (1) The win is COSMETIC-PLUS, not informational: the physics kernel removes wiggle and pins the gate transient but adds NO new longitudinal information -- the mid-race per-horse order/metres still come entirely from the feeds (it could not, by construction, or it would erase order, see the lesson). It UNIFIES the recon visually with the EOM shape; it does not improve the order rho beyond noise.
  (2) The over-smoothing-hides-a-real-move risk is REAL and is exactly why the strong global version was rejected -- a shared-shape velocity prior strong enough to smooth WILL hide a genuine mid-race surge (it hid winner #3 7->1 on R03). The phase-dependent version sidesteps it by staying weak in cruise, at the cost of a smaller smoothing gain.
  (3) The per-horse fitted v_c come out nearly identical (~16.8) because the aerial FIELD odometer dominates the arclen -- so there is little per-horse cruise differentiation for the template to carry anyway; d_fin picks up some finish kick/fade (e.g. R08 #1/#5 fade -0.18).

CONCLUSION: ship the phase-dependent physics kernel as the v2 longitudinal process model -- it is a free, no-regression smoothing + gate-transient pin that aligns the recon with the project EOM, with the honest caveat that it regularises SHAPE only and does not (cannot, without erasing order) reconstruct mid-race longitudinal information the feeds do not already carry. The clean structural result -- squiggle and running-order are the same velocity d.o.f. -- is the durable finding.

ARTIFACTS: studies/cv_fusion_physkernel.py; parquets data_cache/cv_fusion_trajectories_{R01,R03,R08}_physkernel.parquet; figure data_cache/figs/cv_fusion_physkernel_qc.png (gap-vs-t, WHITE top / PHYSICS bottom, all 3 races). #96 _constrained + banked parquets UNMUTATED; src/hkjc untouched; ruff clean; PYTHONHASHSEED=0. Official=validation-only; draw=allowed initial condition.
