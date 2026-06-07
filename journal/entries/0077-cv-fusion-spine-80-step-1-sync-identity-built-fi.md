---
title: CV FUSION SPINE #80 STEP 1 — sync+identity built, FIRST cross-feed lateral test is a NULL on the overlap (rho -0.03, cls-agree 56%) and the null LOCALISES cleanly to (a) the bend/run-in FORESHORTENING of the aerial render and (b) the unreliable patrol identity bridge across the camera cut -- NOT a refutation of cross-feed coherence. Both feeds read FAITHFUL on their own clean segments (aerial vs stewards rho +0.9 on the body/home-straight, drops to ~0 on the bend the overlap forces) and their lateral SPREADS agree (~8m aerial / ~10m patrol, identity-free) -- the latent lateral is real and similarly scaled in both instruments; the per-horse ORDERING test fails because the overlap window sits on the worst geometry for the aerial AND the anonymous head-on patrol tracks cannot be bound to numbers across the finish-cam cut. Tells us exactly how to weight + extend: down-weight aerial on the run-in (per 0075), and the fusion needs a continuity-preserving identity anchor (RF time-sync), not the patrol finish-cut img_y bridge.
id: 0077-cv-fusion-spine-80-step-1-sync-identity-built-fi
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "fusion", "four-feed", "holy-grail", "task-80", "step-1", "shared-clock", "sync", "race-time", "odometer", "arclen", "official-sections", "rf-clock", "identity-bridge", "saddlecloth", "ocr", "rapidocr", "finish-cam", "camera-cut", "back-propagate", "anonymous", "head-on", "patrol", "aerial", "render-inversion", "cross-feed", "consistency", "spearman", "rail-vs-wide", "classification", "lateral", "y-lat", "off-m", "control", "control-dimension", "overlap-window", "bend", "run-in", "foreshortening", "mis-specification", "4b", "not-a-null", "coherence", "spread", "stewards", "comments-on-running", "validation-only", "leak-free", "no-model-change", "study-only", "0075", "0076", "0073", "wave-2", "weighting"]
refs: {"study": "studies/cv_fusion_spine.py", "spine_panel": "data_cache/cv_fusion_spine.parquet", "figure": "data_cache/figs/cv_fusion_spine_qc.png", "aerial_panel": "data_cache/cv_s0d_aerial_render.parquet (#82/0075)", "patrol_panel": "data_cache/cv_patrol_lateral.parquet (#83/0076)", "videos": "data_cache/video/{aerial,patrol,full}_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_{results.html,comments.txt} (VALIDATION ONLY)", "overlap_race_t": "65.3-78.3s (arclen ~1100-1290m, the bend)", "patrol_offset": "race_t = patrol_vid_t - 18.65s (finish-cross 117.5s, +-0.4s)", "rf_clock_offset": "race_t = RF_vid_t - 1.34s (manual reads)", "headline": "rho -0.03, cls-agree 56% (NULL); aerial vs stewards +0.93 clean-body / ~0 bend; spread 7.8m/9.9m"}
created: 2026-06-07T03:01:12+00:00
updated: 2026-06-07T03:01:12+00:00
---
STEP 1 of the 4-feed fusion (task #80). Study-only: studies/cv_fusion_spine.py. REUSES the two
validated per-feed panels as observation models (aerial render #82/0075, patrol #83/0076) -- neither
pipeline rebuilt. src/hkjc untouched; the only new video pass is the patrol finish-cam identity OCR.
ruff clean (src/hkjc + studies). Official data VALIDATION-ONLY (sections, finish order, stewards).

(1) SHARED CLOCK -- one race_t axis across feeds.
- Canonical race_t = official sections [27.65,23.30,23.75,24.15] (sum 98.85 == winner 1:38.85).
- AERIAL -> race_t: the panel's odometer x_arclen mapped through the official arclen->time curve,
  PER ROW. Covers race_t 9.2-98.2s. Direct, no anchor needed.
- PATROL -> race_t: no odometer/clock; anchored KINEMATICALLY by the winner (#3) crossing the
  finish line on the photo-finish cut (visually verified vid t=117.5s; auto-detector clamps to it),
  so race_t = patrol_vid_t - 18.65s (residual +-0.4s, single coarse anchor). NOTE the headline run
  reports offset 17.65 from the auto-detector's 116.5s read; the verified visual anchor is 117.5s
  (offset 18.65) -- a 1s clock uncertainty, well within the overlap. Patrol head-on window
  vid 83-96s -> race_t ~65-79s.
- RF/FULL feed CROSS-CHECK: full_20260603_01.mp4 carries an ON-SCREEN RACE CLOCK (top-right,
  "0:58.65", "670m TO FINISH"). Manual reads t_vid 5->0:03.67, 25->0:23.67, 60->0:58.65 give a
  CONSTANT offset 1.33-1.35s == race_t -- an independent confirmation of the canonical clock (the
  stylised broadcast clock font does not OCR with RapidOCR, same as the painted markers in 0075, so
  it is a manual cross-check not a per-frame source). RF is the natural future sync backbone.
- OVERLAP (aerial cap. AND patrol head-on): race_t 65.3-78.3s, width 13.0s. CRUCIAL: this maps to
  arclen ~1100-1290m == the BEND into the home straight, the single WORST region for the aerial
  render (0075 run-in foreshortening), and the patrol head-on fan is the straight-entry. The two
  instruments' coverage only overlaps where the aerial is least reliable.

(2) IDENTITY BRIDGE.
- AERIAL: 11/12 via the legend (reused from the panel's `runner` col; missing #7 only).
- PATROL: anonymous head-on (numbers face away). The feed CUTS (no pixel continuity) to a finish
  camera at t>=105s. Finish-cam saddlecloth OCR (RapidOCR, 2-scale, pooled t=105.5-118.5s) reads
  11/12 numbers -- a strong identity COUNT. BUT binding each number to a head-on TRACK requires its
  rail-relative LATERAL on the finish cam, and that is the failure: the finish cam is SIDE-ON, so
  both image-y AND perpendicular-distance-to-rail conflate lateral with along-track DEPTH/stagger
  (leader low/large, trailer high/small). So the rank hand-off (finish-cam rail-order -> head-on
  track rail-order) is unreliable -- 10/12 tracks get a number but the assignment is wrong for
  several (e.g. #2, officially WIDE + read widest by the aerial, gets bridged to a tight track).
  Bridge fraction: aerial 11/12 identity (faithful); patrol 10/12 tracks numbered but LOW-confidence
  on the lateral binding.

(3) CROSS-FEED CONSISTENCY TEST (the deliverable), on the overlap window, per numbered horse.
- HEADLINE: Spearman rho(aerial y_lat, patrol off_m) = -0.03 (n=9); rail-vs-wide classification
  agreement 56% (~chance). A NULL.
- DIAGNOSIS (the value is in localising it; both contributors are MEASUREMENT mis-spec, NOT a
  coherence refutation):
  (a) The overlap window is the BEND -- aerial faithfulness vs the stewards' Comments-on-Running
      (rail #12/#8/#4/#11 vs wide #2/#6/#9/#1, VALIDATION-ONLY) is rho +0.93 on the clean body
      (arclen<=1300) and +0.93 on the home straight (>=1300), but COLLAPSES to ~0 (+0.14 full-field,
      -0.02 on the bridged subset) on the arclen ~1100-1290m bend the overlap forces. This is exactly
      the 0075 oblique run-in under-determination, now quantified as a window-localised drop.
  (b) The patrol identity bridge is unreliable across the camera cut (above) -- patrol off_m vs
      stewards rho ~0 BECAUSE the numbers are mis-assigned, not because the patrol metric is bad
      (0076 validated the patrol lateral fan independently at within/between 0.06-0.16).
- COHERENCE POSITIVE (identity-free): the two feeds' lateral SPREADS agree -- aerial 7.8m fan vs
  patrol 9.9m fan (both ~half the 21.3m "C" width) -- so both instruments measure a real,
  similarly-scaled latent lateral. The latent is fuseable; the ORDERING test just landed on the
  one window + the one missing piece (continuity-preserving identity) that defeat it.

WHAT THIS TELLS THE FUSION DESIGN (#80 step 2):
- WEIGHT the aerial render DOWN on the run-in/bend (arclen>1300 and the foreshortened transition);
  its faithful regime is the body + home straight, NOT the bend.
- The patrol identity bridge must NOT rely on the finish-cam img_y (side-on depth confound) or the
  camera cut (no continuity). Use a CONTINUITY-PRESERVING anchor: time-sync patrol to the RF feed
  (shared race clock, established here at +-1s) and carry RF/aerial numbered identity into the patrol
  head-on window by race_t + lateral, OR find an overlap window on a CLEANER aerial segment.
- Better overlap: pull a patrol/aerial overlap on the HOME STRAIGHT proper (arclen>1300) where BOTH
  are clean -- here the patrol head-on window happens to sit on the bend; a later patrol window (or
  the RF home-straight) would test the feeds where each is strongest.

ARTIFACTS: studies/cv_fusion_spine.py; data_cache/cv_fusion_spine.parquet (the bridged spine rows:
runner, track, aer_lat, pat_lat per horse on the overlap); data_cache/figs/cv_fusion_spine_qc.png
(shared-clock timeline + overlap band; identity-bridge confidences; aerial-vs-patrol lateral scatter;
text diagnosis). Inputs reused read-only: data_cache/cv_s0d_aerial_render.parquet (#82/0075),
data_cache/cv_patrol_lateral.parquet (#83/0076), data_cache/video/{aerial,patrol,full}_20260603_01.mp4,
data_cache/race_official/20260603R1_{results.html,comments.txt} (VALIDATION ONLY).
