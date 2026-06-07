---
title: Identity v2 (#94): a SINGLE FIXED confidence-weighted CONSENSUS (own+RF silk + reliability-gated clock-order + DRAW order, Hungarian) + the draw->y_lat(0) fusion anchor — NO R01 REGRESSION (patrol +2.34m/fused +4.29m, winner #3) AND R03 FIX KEPT (patrol +6.78m/fused +5.81m, winner #7) with the SAME fixed weights; draw->metres = uniform-stall physics (21.3m/14); OCR bottleneck cached-gone; R08 held out
id: 0090-identity-v2-94-a-single-fixed-confidence-weighte
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "identity", "consensus", "hungarian", "silk", "hue-histogram", "emd", "own-lut", "rf-lut", "clock-order", "rank-reliability", "draw", "draw-order", "barrier", "y-lat-zero", "gate-anchor", "fusion-anchor", "fusion", "four-feed", "patrol", "rf", "aerial", "drone", "lateral", "control", "rail-vs-wide", "gap", "r01", "r03", "no-regression", "winner-recovery", "per-pole", "spearman", "fixed-weights", "race-agnostic", "calibration", "stall-width", "ocr-bottleneck", "cache", "stewards", "validation-only", "leak-free", "study-only", "task-94", "task-92", "task-93", "0082", "0086", "0088", "0089", "wave-2"]
refs: {}
created: 2026-06-07T09:05:04+00:00
updated: 2026-06-07T09:05:04+00:00
---
identity v2 (#94) — a SINGLE FIXED confidence-weighted CONSENSUS (no per-race tuning) + the draw->y_lat(0) fusion anchor. Refines the #92 module (0089) which corrected R03 but REGRESSED the known-good R01. THE BAR — no regression on R01, keep the fix on R03 — is MET with the SAME fixed scheme on both races. R08 HELD OUT (not tuned).

studies/cv_identity.py (refined: Identity v2 carries RF-anchor LUT + RF rank backbone + barrier DRAW; new Identity.consensus_assign + silk_distinctness + draw_to_metres) + studies/cv_identity_v2_rerun.py (the re-run harness, both races). study-only; imports cv_rf_longitudinal/cv_patrol_identity/cv_r03_* READ-ONLY; src/hkjc UNTOUCHED (verified); banked R01/R03 parquets UNMUTATED (R01 patrol still 319 rows, R03 still 215); all outputs new/_v2-suffixed; ruff clean (src/hkjc + studies); PYTHONHASHSEED=0; official validation-only (draw is a pre-race INITIAL CONDITION, allowed as input — NOT an outcome).

== THE SINGLE FIXED CONSENSUS (identity = one-to-one Hungarian over 4 weighted votes; SAME weights R01 & R03) ==
cost[track,runner] = W_OWN_SILK*distinct[r]*own_emd + W_RF_SILK*rf_emd + W_ORDER*reliability(t)*order_incons + W_DRAW*draw_incons
  W_OWN_SILK 1.00, W_RF_SILK 0.45, W_ORDER 0.60, W_DRAW 0.70 (FIXED, not hand-tuned per race).
  1. OWN finish-cam silk LUT — restores per-feed fidelity #92 discarded (R01 own 102.5-123s LUT earned +4.1m; R03 own patrol finish-cam LUT). Confidence-weighted per-runner by silk DISTINCTNESS (nearest-neighbour EMD spread): degenerate red/pink ~0, distinct blue/yellow ~1.
  2. RF anchor LUT — the cross-feed shared backbone (0089).
  3. shared-clock ORDER constraint — GATED by rank reliability(race_t) = backbone-depth * body-window taper (decays >80pct race), so it only weights UNBUNCHED windows where the RF rank is trustworthy (the 0089 diagnosis: it helps R03 late, amplifies error on R01 bunched). Race-agnostic, derived from each race's own backbone.
  4. DRAW ORDER — race-INDEPENDENT START lateral order (low draw=rail), a far better order proxy than the head-on DEPTH #92 used. Breaks the silk degeneracy with an always-on signal.

== VALIDATION — no regression, fix kept (same scheme) ==
R01 (no-regression bar): patrol consensus gap +2.34m CORRECT sign (#92 gave WRONG -2.6/-9.5m -> FIXED). FUSED rail-vs-wide gap +4.29m (banked 0084 +3.84m / 0087 +4.26m — HELD, even slightly up; rail #4/8/11/12 4.7m vs wide #1/2/6/9 9.0m, 8/8 commented classified correct). winner #3 RECOVERED. per-pole order vs RP +0.95/+0.82/+0.88/+0.71 — preserved.
R03 (keep-fix bar): patrol consensus gap +6.78m CORRECT (#88 -5.7m, #89 +4.4m -> now +6.78m; rail #1/2/3 3.0m vs wide #7/8/10/11 9.7m). FUSED gap +5.81m (>= the +2.7m bar; #92 was diluted to +1.95m, #88 +2.69m -> the draw anchor RESTORED magnitude; rail 4.25m vs wide 10.06m, 10 covered 8/10 correct). winner #7 RECOVERED. per-pole +0.99/+0.99 body, +0.10 late (the field-bunches-late soft spot, unchanged).
identity coverage: R03 RF-anchor LUT 11/12 + own patrol finish-cam LUT 11/12, 8 head-on tracks assigned. R01 RF-anchor LUT 9/12 + own 102.5-123s LUT 8/12, 9 tracks assigned (all rail/wide validation horses #8/12/2/6 covered).

== DRAW -> y_lat(0) FUSION ANCHOR (#93, the magnitude fix) ==
Added to the smoother as a LOW-SD (1.5m, ~half a stall) lateral observation at the gate-jump race_t=0: y_lat(0)_h = draw->metres, a SECOND metric anchor (start=draw, mid=patrol) so the bounded-lateral-accel prior interpolates the in/out shifts. Its effect is the magnitude restoration: R03 fused gap +1.95m (#92, one ~1000m patrol band) -> +5.81m; R01 held +4.29m. In/out shifters (R01 #8 drawn widest gate 12 but raced inside, #1 drawn rail but mid) are exactly the cases where the start-anchor and the mid patrol-metric DISAGREE and the smoother bridges them — and the fused #8 reads 5.2m rail (CORRECT) even though the patrol-alone consensus mis-read it at 14.5m (the draw-vote/silk-degeneracy residual is washed out by the fusion).

draw->metres CALIBRATION (race-agnostic PHYSICS, no per-race tuning): uniform barrier stalls across the C-course. stall_width = TRACK_WIDTH/N_STALLS = 21.3m/14 = 1.52m; y_lat(draw d) = 0.5*stall_width + (d-1)*stall_width [m from inside rail]; field span 0.8-17.5m. CROSS-CHECKED (reported, never tuned to outcome) vs the validated patrol metric field span: R01 draw-map span 16.7m vs patrol 18.2m; R03 16.7m vs patrol 20.6m — same scale, consistent. (The oblique aerial gate frame WAS extracted — R03 t~3s, numbered stalls 1-13 visible — but pixel-precise stall measurement on an oblique single-rail frame is too noisy to anchor a metric scalar reliably; the uniform-stall physics + the patrol cross-check is the defensible calibration.)

OCR BOTTLENECK confirmed GONE: the per-feed own-LUTs are cached to data_cache/cv_own_lut_R0{1,3}_patrol.npz + the RF/patrol anchors to cv_identity_R0{1,3}.npz; a re-run loads OCR-free in <1s (no RapidOCR). The 0088 2.4hr aerial-OCR cadence stays eliminated (aerial identity = RF rank backbone + silk propagation, no render-numeral re-OCR).

== honest residual / remaining ambiguity ==
* The patrol-ALONE consensus magnitude is still substrate-limited on the red/pink silk family: R01's patrol-alone gap +2.34m < banked +4.1m because #8 (distinct navy+pink, distinctness 0.79) is still mis-assigned to a wide track at the head-on window — the W_DRAW vote pulls it toward its wide gate-12 START position while it actually raced inside. The FUSED result corrects this (#8 -> 5.2m rail) so the deliverable (the fused control classification) is right, but the identity-vote draw term is mildly mismatched to the MID-race window (the draw is a START order). This is the one residual: the draw is a better start-anchor than mid-race identity vote. Not fixed per-race (would be overfitting); the fusion absorbs it.
* R03 late-pole order +0.10 (legend stops early, field bunches at the come-from-behind finish) — unchanged from #88, a feed-coverage limit not an identity fault.
* absolute mid-race per-horse metres + the run-in (>950m) lateral metric remain section-4b-uncertain (same class as all prior feeds); the two metric anchors (start draw + mid patrol) restore the rail-vs-wide GAP magnitude, not the full per-runner absolute trip.

NEXT: this consensus + draw-anchor scheme is now stable on R01+R03 with one fixed weight-set. Ready to run R08 (route, HELD OUT) as the final generalization check before any hkjc.vision identity-service promotion. The draw->y_lat(0) anchor is a clean, philosophy-pure win (pre-race initial condition) and should fold into the fusion smoother permanently.

artifacts: studies/cv_identity.py (consensus_assign + silk_distinctness + draw_to_metres + Identity.draw + _rank_reliability), studies/cv_identity_v2_rerun.py; data_cache/cv_r03_patrol_numbered_v2.parquet (8 runners, +6.78m), cv_patrol_numbered_v2.parquet (R01, 9 runners, +2.34m), cv_fusion_trajectories_R03_v2c.parquet (12 runners, fused +5.81m), cv_fusion_trajectories_v2c.parquet (R01, fused +4.29m), cv_own_lut_R0{1,3}_patrol.npz; figs cv_identity_v2_qc.png + cv_fusion_recon_R03_v2c.png + cv_fusion_recon_4feed_v2c.png. compare 0089 (#92: R03 +4.4 patrol/+1.95 fused, R01 regressed -2.6/-9.5), 0082 (R01 banked +4.1), 0087 (R01 fused +4.26), 0088 (R03 +2.69 fused).
