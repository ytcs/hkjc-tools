---
title: FOUR-FEED CV DATA-SOURCE MAP — scrutiny of the patrol + rf replay feeds (vs the aerial AVV render + drone we already use). HKJC exposes (≥) four auth-free Akamai replay types per race; each is a DIFFERENT instrument for a DIFFERENT part of the causal program, not redundant. (1) AERIAL = synthetic 3D render (render-inversion path, #82). (2) DRONE = real moving high cam (lateral on the stable back-straight; no global longitudinal). (3) RF (race film) = real SIDE-ON tracking broadcast — its niche is MARGINS/GAPS between horses (direct beaten-lengths → the S3/#60 margins likelihood surface for W'_0/spread), plus a live clock + a '13.47' speed/section readout (a candidate VELOCITY cross-check for the energetics profile), the same legend running-order strip, distance boards ('1000'), and LEGIBLE saddlecloth numbers at the finish; lateral is foreshortened/weak side-on. (4) PATROL = the stewards' INTERFERENCE camera, a FIXED high mount the field runs toward — head-on early (so lateral position = horizontal image coordinate DIRECTLY, no oblique rail-offset projection — the exact math that injected the y_lat noise nulling aerial/drone in 0070/0071), and a FIXED camera ⇒ a CONSTANT homography solved ONCE (no per-frame rail jitter — the 0071 root cause vanishes). Patrol is geometrically the CLEANEST lateral instrument for the contesting/visible horses + captures the literal interference events (bump/checked/carried-wide) that the official stewards' commentary documents (20260603R1_comments.txt: #12 bumped at start, #6 caught wide, #2 wide-barrier-shifted-across, #9 bumped @150m) — built-in cross-validation. Patrol limits: 720p, HEAD-ON OCCLUSION (trailing horses hidden behind leaders), identity hard head-on (numbers face sideways → bridge from RF/finish numbers or sync), best coverage on the finishing straight only. NET: control/lateral → patrol (cleanest geometry) + aerial-render + drone; MARGINS/physiology (W',spread) → RF (NEW surface for S3/#60); velocity → RF readout + aerial clock; identity → RF/aerial/drone finish numbers + aerial legend/gate. The cached patrol/rf are race 2025-04-06 R9 (different race); replay-patrol Akamai URL CONFIRMED 206; the rf token is not 'replay-rf' (404) — likely 'replay-full' (206) or another; needs pinning before a same-race (R1) pull. Scrutiny only; nothing built; no model change.
id: 0073-four-feed-cv-data-source-map-scrutiny-of-the-pat
author: modelling
audience: ["modelling"]
channel: research
kind: observation
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "video", "replay", "feed-map", "data-source", "scrutiny", "patrol", "rf", "race-film", "aerial", "drone", "head-on", "fixed-camera", "constant-homography", "lateral", "control", "control-dimension", "interference", "stewards", "commentary", "bumped", "carried-wide", "margins", "beaten-lengths", "gaps", "s3", "0060-task", "wprime", "spread", "physiology-surface", "velocity", "speed-readout", "energetics", "legend", "running-order", "distance-board", "saddlecloth", "identity", "number-bridge", "occlusion", "720p", "akamai", "replay-patrol", "replay-full", "url-pattern", "different-race", "r9", "needs-r1-pull", "0070", "0071", "0069", "wave-2", "no-model-change"]
refs: {"patrol_video": "data_cache/video/patrol_20250406_09.mp4 (720p, 124s, race 2025-04-06 R9)", "rf_video": "data_cache/video/rf_20250406_09.mp4 (1080p, 123s, race 2025-04-06 R9)", "frames": "/tmp/s0d_pr/{patrol,rf}_*s.png", "stewards_commentary": "data_cache/race_official/20260603R1_comments.txt", "margins_task": "#60 (S3 margins likelihood surface — RF is its data source)", "patrol_url": ".../hdflash/replay-patrol/{YYYY}/{YYYYMMDD}/{NN}/eng/replay-patrol_..._2500kbps.mp4 (confirmed 206)", "docs": "docs/HKJC_DATA.md (replay URL scheme)"}
created: 2026-06-07T01:24:22+00:00
updated: 2026-06-07T01:24:22+00:00
---
User asked to study the other two replay videos (patrol, rf) and scrutinize them carefully for what
else is worth extracting. Both were already cached (data_cache/video/patrol_20250406_09.mp4 720p/124s,
rf_20250406_09.mp4 1080p/123s — race 2025-04-06 R9, a DIFFERENT race than our 2026-06-03 R1). Frames
extracted to /tmp/s0d_pr/{patrol,rf}_*s.png and eyeballed at start / early / turn / run-in / finish.

## RF (race film) — side-on broadcast tracking camera
- Side-on tracking shot of the field from the gate (t=4s, bunched) down the track. Carries: the same
  bottom LEGEND STRIP (running order: 5,3,13,8,6,7,1,2 at t=25s), a race CLOCK (0:23.21) + a "13.47"
  readout (top-right, under the clock — candidate live speed m/s or last-section time; CONFIRM by
  checking if it varies per-frame=speed vs steps=section), a distance board ("1000"), and at the
  finish (t=108s) LEGIBLE saddlecloth numbers (#2,#11,#4,#6...) under the "Photo Finish System" overlay.
- ITS NICHE: side-on, the longitudinal SEPARATION between horses is direct and well-resolved →
  beaten-lengths / GAPS = a clean MARGINS instrument. This is the data the S3/#60 margins likelihood
  surface needs to sharpen W'_0 + the ability spread (a pending PHYSIOLOGY task, not a lateral one).
- WEAKNESS: lateral (rail-vs-wide = depth INTO the frame) is foreshortened → poor for the control
  signal.

## PATROL — the stewards' interference camera (the geometrically clean lateral instrument)
- A FIXED high mount near the finish that the field runs TOWARD: HEAD-ON early (t=25s, horses coming at
  camera, stacked across the track width), elevated-oblique down the straight (t=85s), 3/4 over the
  line (t=108s). Labelled "Patrol Replay". 720p.
- WHY IT MATTERS for control/lateral, on two independent counts:
  1. HEAD-ON ⇒ lateral position is the HORIZONTAL image coordinate DIRECTLY — none of the oblique
     rail-offset projection whose per-frame noise nulled the aerial/drone y_lat (0070/0071).
  2. FIXED camera ⇒ the track→image homography is CONSTANT — solve it ONCE from the (converging) rails
     + distance markers and apply to every frame. The "unstable per-frame rail refit" that was the
     0071 root cause simply does not exist for a fixed mount.
- It is the literal INTERFERENCE camera, so bumping/carried-wide/checked EVENTS are visible and map
  onto the official stewards' running commentary we cached (20260603R1_comments.txt: #12 bumped at
  start; #6 caught wide early; #2 from a wide barrier shifted across behind runners; #9 bumped ~150m).
  That is built-in external cross-validation for the lateral/control read.
- LIMITS: 720p; HEAD-ON OCCLUSION (a horse directly behind the leader is hidden — you get the lateral
  of the VISIBLE/contesting horses, not always all 12 continuously); identity is hard head-on (the
  number cloth faces sideways) so numbers must be bridged from the finish (legible) or time-synced to
  RF/aerial; best coverage is the finishing straight (the field is small/far earlier). So patrol is a
  strong instrument for lateral jockeying + interference of the contesting group on the run home, less
  for a continuous per-horse full-race y_lat trace.

## The four-feed map (each a different instrument; complementary across the WHOLE program)
- CONTROL / lateral cause: PATROL (cleanest geometry — fixed cam + head-on) + AERIAL render-inversion
  (#82, if faithful) + DRONE (real positions, back-straight).
- MARGINS / physiology (W'_0, ability spread): RF side-on gaps → S3/#60.
- VELOCITY / energetics: the RF "13.47" readout (confirm) + the aerial clock/splits.
- IDENTITY: RF + drone + aerial finish-zoom numbers; aerial legend + start-gate anchor.

## Acquisition note
replay-patrol Akamai URL CONFIRMED (206) at the same pattern as aerial/drone:
`.../hdflash/replay-patrol/{YYYY}/{YYYYMMDD}/{NN}/eng/replay-patrol_{YYYYMMDD}_{NN}_eng_2500kbps.mp4`.
The "rf" token is NOT 'replay-rf'/'rf' (both 404); 'replay-full' returns 206 and is the likely race
film. To use patrol/rf WITH the aerial/drone on the SAME race they must be pulled for 2026-06-03 R1
(the cached pair is R9). Scrutiny only — nothing built, no model touched.
