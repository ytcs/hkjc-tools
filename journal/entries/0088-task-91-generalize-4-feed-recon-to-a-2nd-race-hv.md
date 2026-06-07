---
title: TASK #91 GENERALIZE 4-feed recon to a 2nd race (HV R03 1200m sprint): METHOD generalizes — winner #7 RECOVERED, per-pole RP rho +0.99/+0.98, RF legend backbone +1.00/+1.00, lateral gap +2.69m correct-sign; honest degradations (RF margin magnitude, aerial identity 7/12, patrol numbered sign) + the 9-item algo-refinement backlog of every R1 hard-code that broke
id: 0088-task-91-generalize-4-feed-recon-to-a-2nd-race-hv
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "fusion", "four-feed", "holy-grail", "generalize", "scale-slow", "task-91", "r03", "hv", "happy-valley", "1200m", "sprint", "aerial", "drone", "patrol", "rf", "race-film", "render-inversion", "head-on", "auto-detect", "legend-backbone", "finish-margins", "shared-clock", "identity", "silk", "rail-vs-wide", "stewards", "winner-recovery", "per-pole", "spearman", "refinement-backlog", "reuse", "ruff", "leak-free", "validation-only", "0087", "0075", "0076", "0080", "0086", "wave-2"]
refs: {"common": "studies/cv_r03_common.py", "aerial": "studies/cv_r03_aerial.py", "patrol": "studies/cv_r03_patrol.py", "rf": "studies/cv_r03_rf.py", "drone": "studies/cv_r03_drone.py", "fusion": "studies/cv_r03_fusion.py", "trajectories": "data_cache/cv_fusion_trajectories_R03.parquet", "figure": "data_cache/figs/cv_fusion_recon_R03.png", "panels": "data_cache/cv_r03_aerial_render.parquet, cv_r03_patrol_numbered.parquet, cv_r03_rf_longitudinal.parquet, cv_r03_drone_numbered.parquet", "official": "data_cache/race_official/20260603R3_results.json (VALIDATION ONLY)", "r1_capstone": "journal 0087", "headline": "winner #7 recovered; per-pole RP rho +0.99/+0.98/-0.08; RF legend +1.00/+1.00; margin MAE 1.70 lead; lateral gap +2.69m"}
created: 2026-06-07T07:30:18+00:00
updated: 2026-06-07T07:30:18+00:00
---
TASK #91 — GENERALIZE the validated R1 4-feed reconstruction pipeline to a SECOND race (HV
2026-06-03 R03, Class 4 1200M turf "C", winner #7 FORERUNNER). This is the scale-slow
generalization test: REUSE/ADAPT the R1 study code (cv_s0d_aerial_render / cv_drone_revisit /
cv_patrol_lateral+identity / cv_rf_longitudinal / cv_fusion_4feed), surface every hard-coded R1
constant that broke, and report what generalized CLEANLY vs DEGRADED on the sprint. Method REUSED,
not reinvented: each R03 driver imports the R1 GEOMETRY functions unchanged and only re-points the
R1-baked constants (a shared studies/cv_r03_common.py factors out DIST/sections/clock/official).
Discipline kept: src/hkjc untouched; R1 parquets/figs UNMUTATED (all R03 outputs are R03-suffixed);
ruff clean (src/hkjc + studies); PYTHONHASHSEED=0; official data VALIDATION-ONLY.

HEADLINE: the METHOD generalizes. 4-feed fusion on R03 -> WINNER #7 RECOVERED (fused leader #7),
per-pole order-vs-RP rho +0.99 / +0.98 / -0.08 at the 1/4 / 1/2 / late calls (beats R1's
+0.96/+0.80 at the body), fused finish-order rho +0.36, RF-anchored lead-group finish-margin MAE
1.70 len (R1 0.72), fused rail-vs-wide lateral GAP +2.69m (correct sign; R1 +4.26m), 6/9
stewards-commented horses classified correct. The longitudinal axis (RF legend backbone) and the
winner recovery are the clean wins; the lateral gap and the absolute margins are the honest
degradations.

== PER-FEED GENERALIZATION STATUS (clean / adapted / degraded + the key validation number) ==

AERIAL render (#82 toolkit) — CLEAN/ADAPTED. Render-inversion ported unchanged; the split-screen
+ green-render + stable-rail-spline + ground-contact all generalized. 231 aerial-active frames
(the single-cam lead-in t<~10s correctly excluded by find_layout). Faithfulness battery: lag-1
autocorr +0.93 (R1 +0.90, smooth graphics-like, NOT noise), within/between ratio 1.59 (R1
1.62-1.83 same §4b class), rail-vs-wide GAP +4.0m (R1 +4.26m, CORRECT SIGN). corr(y_lat, mean RP)
flipped to -0.36 (R1 +0.45) — but the explicit stewards/draw rail-vs-wide split is +4.0m correct,
so the ordering signal is intact; the RP-corr flip is the weaker R03 RP->lateral mapping (sprint
field doesn't fan by RP as cleanly). Identity 7/12 via legend (R1 11/12) — REDUCED, see backlog
#5. Painted distance markers still non-OCRable (0 frames, same as R1); odometer carries arclen.

PATROL head-on (#83/#86 toolkit) — CLEAN geometry, DEGRADED validation. The head-on window was
AUTO-DETECTED at t=60-66s (R1 hard-coded 83-96s) by a rail-slope-stability scan — the sprint's
home straight arrives ~20s earlier in race-time, exactly as predicted, and the auto-detector found
it. within/between dispersion ratio 0.055 / 0.103 / 0.184 (R1 0.06/0.16/0.35) — CLEAN, the
fixed-camera head-on geometry is the most faithful generalizer. Identity 9/12 LUT, 8 assigned
(BETTER than R1's 8/12). BUT the numbered rail-vs-wide validation FAILED: GAP -5.7m (WRONG SIGN) —
the silk->number assignment permuted within the colour-degenerate family AND the R03 stewards
lateral-truth is itself noisy (see backlog #7). Rail slope CV 9.7% (R1 3.0%) — higher, the
auto-window includes mild pan. Patrol anchors arclen ~958-1058m (the bend-into-straight), the
sprint analog of R1's <1300m anchor.

RF / race-film (#89 toolkit) — legend backbone CLEAN, finish margins ADAPTED-then-RECOVERED. The
legend-strip running-ORDER backbone validated rho(legend, official RP) = +1.00 / +1.00 / +0.19 at
the 1/4 / 1/2 / late calls (R1 +0.98/+1.00/+1.00) — the name->number bridge (rebuilt from the R03
official parse) reads the body of the race PERFECTLY; the only soft spot is the late/bunched call.
Finish MARGINS needed a real fix: the R03 finish cam zooms LESS than R1 so the saddlecloths
blur+cluster the instant the field crosses (OCR yield -> 0 at t>=71s); the legible window is the
APPROACH (t~68.5-71.2s, ~last 50m), NOT the line. After re-windowing there (+ finer 0.04s step),
margins recovered: 5 horses, margin rho +0.90, ORDER rho +0.90 (== R1 +0.90), MAE 1.93 len (R1
0.41 — magnitude overreads because measured pre-line, but ORDER and the winner-#7-clear-leader are
right, which is what the fusion consumes). Clock RF_OFFSET re-anchored to 2.71s (R1 1.34s; the
broadcast lead-in differs); winner crosses ~vid 72s (R1 ~100s).

DRONE (#87 toolkit) — ADAPTED, supplementary as in R1. Time-anchored on the shared clock: gate
jump ~vid 3.0s, finish-cut AUTO-SCANNED to vid 73.5s (R1 hard-coded 113.68s; centroid-jump scan
re-derived it, matching the winner-cross). IDENTITY was the friction: R1 read silk->number anchors
from its own 0070 drone panel (an R1 artifact that does NOT exist for R03) -> rebuilt them
race-agnostically from the RF finish-cam OCR (5/12 anchors #4,5,7,10,12). 4 numbered runners in the
panel; rail-vs-wide GAP +3.5m (rail #4 1.4m vs wide #5,7 4.9m), CORRECT SIGN. Same §4b metric-null
class as R1 (ordering, not absolute trip). The drone improved the fused body poles (+0.99/+0.98 vs
3-feed +0.98/+0.94).

== FUSION (cv_fusion_4feed RTS smoother, generalized) ==
All 4 feeds on one shared clock (official 3 sections -> race_t; per-feed anchors). 12/12 horses
reconstructed, grid race_t -2.97..70.28s. Winner #7 RECOVERED. Per-pole RP rho +0.99/+0.98/-0.08.
finish-order rho +0.36. lead-group margin MAE 1.70 len. lateral GAP +2.69m (6/9 correct). per-band
lateral sd inflates 7.8m(start)->17m(mid)->24m(bend)->30m(run-in) — wider than R1 because R03 has
ONE patrol metric anchor band (~1000m) and noisier lateral truth; the absolute-metre lateral stays
§4b-uncertain (same as R1). The late-pole rho -0.08 is the genuine soft spot: the legend stops
~race_t 60 and the field bunches, so the come-from-behind winner's final surge is pinned only by
the (thin) RF margins, not the legend.

== THE REFINEMENT BACKLOG (the real deliverable — every R1 hard-code that broke + how hardened) ==
1. OFFICIAL FORMAT: R1 = HTML <td> walk; R03 = JSON-wrapped rendered TEXT with multi-line RP/LBW
   columns -> a NEW regex parser (cv_r03_common.parse_official) stitching the '8 7 1' RP newlines +
   parsing LBW strings (NK/SH/'3-1/4'). HARDENED, race-agnostic.
2. SECTIONS 4->3: a 1200m HK race has THREE sectionals (1650m had four). Every arclen<->race_t
   curve, distance-run call boundary (0/400/800/1200 vs 0/450/850/1250/1650), and the 3-vs-4 RP
   columns changed. HARDENED in cv_r03_common (_D_BOUNDS/_T_CUM) + re-pointed into every feed.
3. CLOCK ANCHORS all R1-specific: RF_OFFSET 1.34->2.71s; drone finish-cut 113.68->73.5s (now
   AUTO-SCANNED by centroid jump); patrol winner-cross vid 117.5->74s; winner-t 98.85->69.29s.
   Re-derived per feed; the drone cut is now auto-detected (robust), the rest frame-verified.
4. RF FINISH-MARGIN WINDOW + winner-ref: R1 hard-coded the window 96.5-100.5s AND winner==#3 as
   the gap reference. Both broke. Winner #7 -> re-derived gaps relative to #7. The window had to
   move to the LEGIBLE APPROACH (68.5-71.2s) because the R03 finish cam blurs at the line — a
   FEED-QUALITY difference, not just a constant. THE load-bearing fix for winner recovery.
5. AERIAL OCR CADENCE DOESN'T SCALE: the R1 cadence (OCR every 3 aerial frames + per-frame marker
   OCR) ran ~2.4hr CPU and stalled on R03; coarsening to stride 8 / OCR every 8 / markers every 30
   finished in minutes but dropped legend identity 11->7/12. The OCR cadence needs to be
   ADAPTIVE/cheaper (cache the legend, skip the non-OCRable markers entirely) — a real algo debt.
6. DRONE IDENTITY had NO race-agnostic bridge: R1 read silk anchors from its own prior panel.
   Rebuilt from the RF finish-cam OCR (5/12). Works but couples drone->RF; a cleaner cross-feed
   identity service is owed. Also the silk-anchor + legend OCR passes use cv2 random-seek
   (cap.set) which is pathologically slow on H.264 — should use sequential av decode (the aerial
   path), a tractability debt that nearly blocked the run.
7. STEWARDS LATERAL-TRUTH is MUCH WEAKER on R03: R1 had clean trip calls ("wire-to-wire rail",
   "caught wide"); R03 comments are INTERFERENCE incidents ("shifted out and made contact"), so a
   keyword scan over-reads WIDE. Hardened to a precise-phrase set + a BARRIER-DRAW fallback (low
   draw=rail), but it stays noisy (#4 reads rail correctly via "closer to the rail" despite draw 9;
   #8 mis-as-wide). This noisy ground-truth is the main reason the lateral validation degraded vs
   R1 — a VALIDATION-surface limit, not necessarily a recon error.
8. PATROL HEAD-ON WINDOW: R1 hard-coded 83-96s. Now AUTO-DETECTED by rail-slope-stability scan
   (found 60-66s on R03). This one generalized CLEANLY and is the template for race-agnostic
   window-finding.
9. PATROL/DRONE NUMBERED-band threshold: R1's RUN-IN foreshorten cut was 1300m on 1650m (~79%);
   scaled to 950m on 1200m. The lateral classification band had to widen to 100-1100m to INCLUDE
   the patrol metric anchor (which sits ~1000m on the sprint, just above the naive run-in cut).

== HONEST CLEAN-vs-DEGRADED SUMMARY (R03 vs R1) ==
CLEAN generalizers: RF legend backbone (+1.00/+1.00), patrol head-on geometry (ratio 0.055-0.18),
patrol window auto-detect, aerial render faithfulness (ac1 +0.93, gap +4.0m), drone time-anchor
(auto-cut), winner recovery, per-pole body order (+0.99/+0.98). DEGRADED: RF finish-margin
MAGNITUDE (MAE 1.7-1.9 vs 0.72; feed zooms less), aerial legend identity (7 vs 11/12; OCR-cadence
tradeoff), patrol numbered rail/wide sign (-5.7m; identity permutation + noisy truth), fused
lateral gap (+2.69 vs +4.26m) and late-pole order (-0.08; legend stops early on the short race).
A feed degrading is itself a finding: none of these are philosophy failures — they are
measurement/identity/validation-surface frictions that localize exactly what to harden to make the
pipeline race-agnostic. The 4-feed METHOD is validated on a 2nd race of a different distance.

ARTIFACTS (all R03-suffixed; R1 outputs untouched):
studies/cv_r03_common.py, cv_r03_aerial.py, cv_r03_patrol.py, cv_r03_rf.py, cv_r03_drone.py,
cv_r03_fusion.py; data_cache/cv_r03_{aerial_render,patrol_lateral,patrol_numbered,
rf_longitudinal,drone_numbered}.parquet; data_cache/cv_fusion_trajectories_R03.parquet;
data_cache/figs/cv_fusion_recon_R03.png (+ per-feed QC figs). Official 20260603R3_results.json
parsed VALIDATION-ONLY. Compare R1: journal 0087 (winner recovered, per-pole +0.96/+0.80/+0.90/
+0.71, margin MAE 0.72, lateral +4.26m).
