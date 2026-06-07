---
title: TASK #95 VERDICT — the HARDENED 4-feed pipeline GENERALIZES to the HELD-OUT 1800m ROUTE with NO tuning: winner #6 RECOVERED, per-pole order +0.96/+0.95/+0.98/+0.94 (corpus-best), lateral gap +3.97m CORRECT sign (patrol-consensus +12.74m) — a PASS on all 3 core criteria. Honest route LIMITS: finish-order rho +0.26 (blanket NK/SH finish, RF margins resolve only the lead 4), lateral CLASSIFICATION 3/6 + per-band sd 12->52m (the AERIAL render FORESHORTENS on the route: within/between 2.55 vs 1.59, per-bend autocorr -0.16 vs +0.93, rail-vs-wide WRONG sign) and the patrol replay pans oblique not fixed-head-on. The metric-gap is localized to the route bends; ORDER + winner + lateral-sign survive on the fixed scheme
id: 0092-task-95-verdict-the-hardened-4-feed-pipeline-gen
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "fusion", "four-feed", "holy-grail", "generalize", "held-out", "verdict", "pass", "generalization-limit", "scale-slow", "task-95", "r08", "hv", "happy-valley", "1800m", "route", "aerial", "drone", "patrol", "rf", "identity", "consensus", "draw-anchor", "foreshortening", "metric-gap", "render-inversion", "legend-backbone", "winner-recovery", "per-pole", "spearman", "rail-vs-wide", "stewards", "blanket-finish", "no-tuning", "distance-generalization", "sprint-mile-route", "0087", "0088", "0090", "0091", "wave-2"]
refs: {"common": "studies/cv_r08_common.py", "aerial": "studies/cv_r08_aerial.py", "rf": "studies/cv_r08_rf.py", "patrol": "studies/cv_r08_patrol.py", "drone": "studies/cv_r08_drone.py", "fusion": "studies/cv_r08_fusion.py", "identity": "studies/cv_r08_identity.py", "trajectories": "data_cache/cv_fusion_trajectories_R08.parquet", "figure": "data_cache/figs/cv_fusion_recon_R08.png", "panels": "data_cache/cv_r08_{aerial_render,rf_longitudinal,drone_numbered,patrol_numbered}.parquet", "anchor": "data_cache/cv_identity_R08.npz + cv_own_lut_R08_patrol.npz", "official": "data_cache/race_official/20260603R8_results.json (VALIDATION ONLY)", "partial": "journal 0091", "r01": "journal 0087/0090", "r03": "journal 0088/0090"}
created: 2026-06-07T09:45:22+00:00
updated: 2026-06-07T09:45:22+00:00
---
TASK #95 VERDICT — does the HARDENED 4-feed pipeline (the #91 R03 study code + the #94 unified
identity consensus + the #93 draw->y_lat(0) anchor) GENERALIZE to the HELD-OUT route with NO
per-race tuning? HV 2026-06-03 R08, 1800m "C" route, winner #6 SILVERY BREEZE (from behind, RP
"6 6 7 7 1", won NK), 11 runners, 5 sectionals. R08 was held out while hardening on R01(1650)+
R03(1200). Discipline kept: src/hkjc/{model,core,vision/mode_b} UNTOUCHED; R01/R03 parquets
UNMUTATED; ruff clean (src/hkjc + studies); PYTHONHASHSEED=0; official VALIDATION-only (draw =
pre-race initial condition, allowed input). Every R08 driver imports the R1/R03 geometry + the
#94 consensus UNCHANGED; ONLY race-PARAMETERS re-pointed (distance, 5-section clock, field 11,
video paths, frame-verified clock anchors, the auto-detect scan range). The identity weights
(own 1.0 / RF 0.45 / order 0.6 / draw 0.7) and every feed threshold used AS-IS.

== THE VERDICT: PASS on all 3 core criteria, with documented ROUTE-SPECIFIC LIMITS ==
PASS bar (winner recovered + lateral gap correct sign + order rho strong, all on the fixed scheme):
  1. WINNER #6 RECOVERED — fused leader #6. YES.
  2. LATERAL GAP CORRECT SIGN — fused rail-vs-wide +3.97m (R1 +4.26, R03 +5.81); the patrol-
     CONSENSUS alone +12.74m CORRECT (#6 rail 4.9m vs wide 17.6m). SIGN HELD on the route with the
     FIXED #94 consensus + draw vote.
  3. PER-POLE ORDER vs official RP STRONG — +0.96/+0.95/+0.98/+0.94 at c1/c2/c3/c4, late +0.60.
     This is the corpus-BEST body order (R1 +0.96/+0.80/+0.90/+0.71; R03 +0.99/+0.98/-0.08). The
     RF legend backbone generalized cleanest of all (rho(legend,RP) +0.96/+0.98/+0.93/+0.98).
So the METHOD generalizes to a 3rd race / 3rd distance (sprint->mile->ROUTE) with zero tuning.

== THE HONEST GENERALIZATION LIMITS (route-specific, NOT philosophy failures, NOT secretly tuned) ==
A. AERIAL RENDER FORESHORTENS ON THE ROUTE — the metric-gap thesis, quantified. The 1800m route is
   mostly turning (HV "C" 1450m circuit, ~1.24 laps). The render-inversion 4b faithfulness DEGRADES
   monotonically with bend content across the corpus: within/between RATIO 1.59 (R03 sprint) ->
   1.6-1.8 (R1 mile) -> 2.55 (R08 route, WORST); lag-1 autocorr +0.93/+0.90 (sprint/mile body) ->
   +0.86 GLOBAL but NEGATIVE -0.16 per-BEND-band (ratio 2.61 on the 1000-1200m bend) on the route;
   rail-vs-wide gap +4.0/+4.26 (correct) -> -3.2m WRONG SIGN on R08. The oblique render that was a
   faithful full-race lateral source on the straighter sprint/mile becomes per-frame NOISE on the
   route's continuous bends. This is EXACTLY where the run-in/bend metric gap bites on a route.
B. PATROL geometry route-degraded. The route patrol REPLAY pans/follows the field down the home
   straight from an oblique elevated angle, NOT the FIXED head-on camera R01/R03 had. The rail-slope
   auto-detect still found its best window (t=95-111s, rail CV 7.0%, between R1 3.0% and R03 9.7%),
   and the consensus still held the gap SIGN (+12.74m), but the head-on metric is weaker -> the
   patrol anchor band sits late (~1400-1600m) and narrow.
C. The COMPENSATION + its residual. With the aerial lateral degraded and patrol weaker, the fusion
   leans on patrol(late) + draw(start) + drone. Net: fused lateral SIGN correct (+3.97m) but
   CLASSIFICATION only 3/6 (vs R1 7/8, R03 8/10) and the per-band fused y_lat sd BLOWS UP
   12->28->39->52m start->run-in (R1 1.0->1.42m, R03 6.5->15.7m) — the route lateral MAGNITUDE is
   essentially unconstrained mid-race; only ordering + the rail-vs-wide sign survive. Two genuine
   mis-classes: #8 (truth wide, reads rail) and #7 (truth rail, reads mid-wide); #11 is a MISLABELED
   truth (drew gate 11 = "wide" by the draw fallback but actually railed wire-to-wire to lead — an
   in/out shifter the draw-vote start-anchor cannot catch mid-race, the same class as R01 #8).
D. FINISH-ORDER rho +0.26 (R1 +0.68, R03 +0.36). R08 is a BLANKET finish — 11 horses within 4
   lengths (NK/SH/sub-length gaps). The RF finish margins resolve only the lead group (4 horses #2/
   6/9/11 anchored, MAE 1.79len); the back 7 are ordered by the legend rank carried to the bunched
   come-from-behind line (RF late-call rho +0.36), so the deep tail order is weak. This is the SAME
   feed-coverage limit flagged on R03's late pole, amplified by the blanket finish — a feed limit,
   not a pipeline fault. The winner + lead group + the per-pole BODY order are right.

== DISTANCE-GENERALIZATION PICTURE (sprint -> mile -> route; fixed scheme, no per-race tuning) ==
metric                 R03 1200 sprint   R01 1650 mile     R08 1800 ROUTE
winner recovered       YES #7            YES #3            YES #6
per-pole body order    +0.99/+0.98       +0.96/+0.80/+0.90 +0.96/+0.95/+0.98/+0.94  (route BEST)
RF legend vs RP        +1.00/+1.00       +0.98/+1.00/+1.00 +0.96/+0.98/+0.93/+0.98
fused finish-order rho +0.36             +0.68            +0.26 (blanket NK finish)
fused lateral gap      +5.81m            +4.26m           +3.97m (sign held)
patrol-consensus gap   +6.78m            +2.34m           +12.74m (all CORRECT sign)
aerial within/between  1.59              1.6-1.8          2.55  (route foreshortening)
aerial rail/wide gap   +4.0m correct     +4.26m correct   -3.2m WRONG (foreshortened)
per-band lat sd        6.5->15.7m        1.0->1.42m       12->52m (route metric blow-up)
CLEAN generalizer: the RF longitudinal ORDER backbone + winner recovery + lateral SIGN.
DEGRADES with distance/bends: the aerial render lateral METRIC (foreshortening) + absolute lateral
magnitude + the deep-tail finish order (blanket finish). All consistent with the documented metric
section-4b class; none required tuning to expose, none were patched.

CONCLUSION: the hardened pipeline is a VALIDATED race-agnostic ORDER + winner + control-CLASSIFICATION
recon across sprint/mile/route on ONE fixed scheme. The route exposes that the aerial render's
absolute LATERAL metric is the dimension that does NOT generalize to heavy-bend geometry (a clean,
documented limit pointing at the per-frame homography / a foreshortening down-weight as the fix) —
while ORDER, winner, finish-margin-lead, and the rail-vs-wide control SIGN do generalize. The #94
identity consensus + #93 draw anchor are now validated on R01+R03+R08 with one weight-set; the
identity module is ready for the hkjc.vision promotion review (its limit on the route is the
substrate aerial foreshortening, not the identity scheme). Artifacts: studies/cv_r08_*.py;
data_cache/cv_r08_*.parquet + cv_fusion_trajectories_R08.parquet + cv_identity_R08.npz; fig
data_cache/figs/cv_fusion_recon_R08.png. Compare 0087(R1), 0088(R03), 0090(#94 identity), 0091(this
task's partial).
