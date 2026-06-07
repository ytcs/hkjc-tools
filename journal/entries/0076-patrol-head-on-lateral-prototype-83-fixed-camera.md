---
title: PATROL head-on lateral PROTOTYPE (#83): FIXED camera => CONSTANT homography (rail slope CV 3.0%, solved ONCE) + HEAD-ON => lateral=horizontal image coord, recovers a STABLE SENSIBLE lateral on ST 2026-06-03 R1 where aerial/drone NULLED. Headline within/between lateral dispersion ratio 0.06 (frame jitter 0.24m/3.98m) to 0.16 (1.5s window 0.63m) on the GEOMETRY, 0.35 even with anonymous cross-linking contamination -- ALL BEAT aerial 0.87 and drone >1. 12/12 field resolved on the run-in (robust >=20-det tracks). Lateral fan (rail ~5-7m / mid ~9-11m / wide ~13-15m) matches the visible spread + stewards commentary (#2 shifts IN from wide barrier; #6 caught wide). The 0071/0070 null was a SUBSTRATE mis-spec (moving cam + oblique projection), not a true null of the control cause. Honest limits: single race, single visible rail (scale anchored not two-rail-solved; ratio is scale-free), anonymous head-on identity inflates the full-track estimator (bridge from finish-cam numbers next).
id: 0076-patrol-head-on-lateral-prototype-83-fixed-camera
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "video", "replay", "patrol", "head-on", "fixed-camera", "constant-homography", "rail", "rail-fit", "slope-cv", "lateral", "control", "control-dimension", "y-lat", "trip", "within-between", "dispersion-ratio", "noise-ratio", "stable", "sensible", "ground-contact", "hoof", "silk", "jockey-detection", "plumb", "depth-aware-tracking", "anonymous", "identity", "number-bridge", "occlusion", "field-fraction", "12-of-12", "vanishing-point", "perspective-rectification", "stewards", "commentary", "interference", "caught-wide", "shifted-across", "de-confound", "mis-specification", "substrate", "4b", "prototype", "study-only", "s0d", "s8", "wave-2", "0060-task", "0070", "0071", "0073", "beats-aerial", "no-model-change", "official-validation-only", "leak-free"]
refs: {"driver": "studies/cv_patrol_run.py", "geometry": "studies/cv_patrol_lateral.py", "figure": "data_cache/figs/cv_patrol_qc.png", "panel": "data_cache/cv_patrol_lateral.parquet (577 rows, 18 tracks, 12 robust)", "video": "data_cache/video/patrol_20260603_01.mp4 (1920x1080 25fps 141.5s)", "official": "data_cache/race_official/20260603R1_results.html", "commentary": "data_cache/race_official/20260603R1_comments.txt", "headon_window": "t=83-96s", "rail": "x=-0.3934*y+393.5 slope CV 3.0%", "ratio": "0.06 (frame) / 0.16 (1.5s) / 0.35 (full-track) vs aerial 0.87", "supersedes_substrate_null": "0071 (aerial y_lat 0.87) / 0070 (drone >1)"}
created: 2026-06-07T01:58:32+00:00
updated: 2026-06-07T01:58:32+00:00
---
PROTOTYPE (study-only, task #83) — the PATROL feed as a control-lateral instrument on ST
2026-06-03 R1 (Class 5, 1650m, TURF "C", width 21.3m). The thesis from 0073: the patrol replay
is the stewards' FIXED high mount the field runs HEAD-ON toward in the home straight, so (1) the
track->image homography is CONSTANT (solve ONCE, the 0071 per-frame rail-refit root cause cannot
occur) and (2) HEAD-ON => lateral = ~the horizontal image coordinate after rectification, not the
oblique rail-offset projection whose per-frame noise nulled aerial/drone (0070/0071). VERDICT: the
thesis HOLDS decisively. All new code in studies/cv_patrol_{lateral,run}.py; src/hkjc/model|core
and vision/mode_b/ UNTOUCHED; ruff clean; official data VALIDATION-only.

## (1) HEAD-ON SEGMENT
The fixed-camera head-on home-straight window is t=83-96s. Verified by eye: before ~83s the camera
still pans off the bend (rail slope flips sign at t<=82); the field runs toward the camera growing
large t=84-100; leaders exit the bottom of frame ~t=101-102, then the feed CUTS to a DIFFERENT
finish-line camera at t>=105 (legible saddlecloth numbers #3,#2,#6,#4 there — a potential identity
bridge, distinct camera). Native frames are 1920x1080@25fps (NOT 720p as 0073 guessed).

## (2) FIXED HOMOGRAPHY — STABLE (the 0071 root cause vanishes)
White left running-rail fit per-frame, taken as the MEDIAN line ONCE (camera fixed): x=-0.3934*y+393.5.
Rail SLOPE CV = 3.0% across the 27-frame window (rock-steady) — the camera is fixed, exactly as the
thesis predicts; the only residual is a +/-80px intercept jitter from horses occluding the rail,
which the solve-once median removes by construction. Single visible rail (no clean right outer rail
in the zoomed head-on frame), so the metric scale is anchored to a ~15m field-fan (cross-checked
inside the 21.3m "C" width); the headline RATIO is scale-invariant regardless.

## (3) DETECTION + GROUND CONTACT
Per-jockey detection by SILK colour (saturated non-turf), dilate+CC to merge each jockey's silk
fragments (raw silk blobs ~40/frame -> ~12-14 jockeys/frame), then a PLUMB down the silk column
through the dark horse body to the turf boundary = the GROUND-CONTACT (hoof/shadow base), projected
to lateral offset from the rail. 646 ground-contacts over 65 frames (avg 9.9/frame).

## (4) THE HEADLINE — within/between lateral dispersion ratio (vs aerial 0.87 / drone >1)
Anonymous depth-aware NN tracks (lateral off + along-track depth gate stops the linker hopping
between horses sharing a lateral band). THREE estimators, ALL well below the aerial 0.87 null:
  (A) frame-to-frame jitter (raw fixed-cam/head-on measurement noise): within 0.24m / between 3.98m
      = RATIO 0.059
  (B) short-window (1.5s) within-horse drift: within 0.63m / between 3.98m = RATIO 0.159
  (C) full-window track std: within 1.66m / between 4.70m = RATIO 0.352 -- INFLATED by anonymous
      cross-linking (an IDENTITY artifact, not a lateral-geometry one; head-on identity is hard).
=> patrol head-on ratio 0.06-0.35, ALL BEAT aerial 0.87 and drone >1. STABLE. The fixed-camera
head-on geometry recovers a stable lateral where the moving-camera aerial/drone substrates could not.

## (5) FRACTION OF FIELD + SENSIBILITY
At the >=20-detection robust cut: 12 long tracks for a 12-horse field = 100% resolved on the run-in
(the contesting/visible field; head-on occlusion is handled by the dense-stride re-detection). The
recovered lateral is SENSIBLE: a clean three-way fan -- rail group ~5-7m, midfield ~9-11m, wide
~13-15m -- matching the visible spread. Per-track early-vs-late trend matches the stewards' commentary
(20260603R1): two tracks drift IN (-1.9m, -3.6m) consistent with #2 SETANTA 'from a wide barrier
shifted across behind runners'; several drift OUT consistent with #6 TELECOM POWER 'caught wide in
the early stages' and the field fanning for home. Finish order 3,12,6,2,9,8,11,4,7,10,5,1 (margins
neck/1.5L/2.25L at the front) used only as validation.

## VERDICT
YES -- the FIXED-CAMERA HEAD-ON PATROL gives a STABLE, SENSIBLE lateral where aerial/drone could not.
The 0071/0070 null was a SUBSTRATE mis-specification (moving camera + oblique projection), NOT a true
null of the control/lateral CAUSE: on the right instrument the lateral signal is clean (within/between
0.06-0.16 on the geometry; 0.35 even with anonymous-linking contamination). HONEST LIMITS: single race;
single visible rail so the metric scale is anchored not two-rail-solved (ratio is scale-free, the
metric readout is approximate); anonymous head-on identity (numbers face away) inflates the full-track
estimator and is the next thing to close (bridge from the t>=105 finish-camera numbers / time-sync to
RF). This unblocks CONTROL (S8) IF a numbered per-horse lateral trip can be bridged. Artifacts:
studies/cv_patrol_lateral.py, studies/cv_patrol_run.py, data_cache/figs/cv_patrol_qc.png,
data_cache/cv_patrol_lateral.parquet.
