---
title: S0d SYNTHESIS & CORRECTION (reconciles 0067 first-pass + 0068 improved-pipeline; supersedes the 0068 'AERIAL WINS, scale it' verdict). TWO things 0068 got wrong, both material. (1) The aerial 'ρ+0.94' headline is the LEGEND STRIP OCR — and the legend is a DATA OVERLAY (the broadcaster's rendered running-order, ≈ the official chart we already scrape), NOT a physics reconstruction. Reading it confirms the OCR works; it does NOT show CV reconstructing the race, and it is NOT the novel signal. The GENUINE geometric reconstruction (arc-length timing) still gives 0068's OWN spurious −0.50 / the 0067 noise-dominated ρ; the y_lat TRIP is a 7-runner QUALITATIVE anecdote (#12 rail 5.2m / #10 wide 16.1m / #3 mid 11.1m), not a quantitative validation. So the legend's place is a CHEAP DENSE RUNNING-ORDER FEED + a re-id ANCHOR, not a reconstruction win — 'scaling the aerial legend' largely re-acquires order data the charts already give. (2) 0068's 'DRONE WEAK — saddlecloth numbers ILLEGIBLE from altitude, no honest cluster→number bind' is FALSE: it only inspected the high-altitude back-straight segment. FRAME SCRUTINY (frames at /tmp/s0d_scrutiny, t≈120-135s) shows the drone ZOOMS to a close oblique finish shot where numbers are CLEARLY legible (#3,#2,#9,#6,#8,#11 read by eye) — so the drone CAN be numbered non-circularly via finish-zoom OCR + backward label propagation, and being REAL horse positions it is the BETTER path to genuine NUMBERED LATERAL tracks. Further frame scrutiny: the aerial replay is a SPLIT-SCREEN (LEFT = real broadcast tracking camera with real silks + race clock/splits; RIGHT = the synthetic AVV with the legend strip + a top-right MINIMAP + odometer); the start frame (t≈1s) is a REAL gate cam with stalls numbered 1-13 + visible silks (a clean silk↔number anchor); and the tracked paths DROP the field ~140m before the TRUE finish (~t=98-100s, odometer '10M'/'FINISH'), so we never see the winner #3's surge in the recon at all. NET: the open CV problem for BOTH views is the NOVEL LATERAL/CONTROL signal (y_lat trip), still unvalidated; order/finish we can get from the legend or the chart. Plan: aerial keeps 0068's real gains (green-render detect, scale_from_sprites, ocr_every, C-width) and uses the legend as ANCHOR + cheap order data while genuinely extending the recon through the finish + validating y_lat; DRONE becomes PRIMARY for the lateral signal (scale fix + finish-zoom number bridge). Then fusion.
id: 0069-s0d-synthesis-correction-reconciles-0067-first-p
author: modelling
audience: ["modelling"]
channel: research
kind: decision
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["s0d", "cv", "vision", "mode-b", "synthesis", "correction", "reconciliation", "legend", "legend-strip", "data-overlay", "overlay-read", "not-reconstruction", "running-order", "running-position", "official-chart", "re-id-anchor", "reid", "lateral", "y-lat", "control", "control-dimension", "trip", "ground-saved", "unvalidated", "drone", "finish-zoom", "saddlecloth", "number-bridge", "legible", "backward-propagation", "aerial", "split-screen", "real-camera", "minimap", "start-gate", "anchor", "finish-extension", "winner-surge", "frame-scrutiny", "scrutiny", "green-render", "scale-from-sprites", "ocr-every", "c-course", "px-to-m", "overclaim", "supersedes-0068", "0067", "0068", "gated", "s8", "s0c", "external-validation", "honest", "no-regression", "wave-2", "0060"]
refs: {"first_pass": "journal/entries/0067-s0d-cv-methodology-bake-off-aerial-vs-drone-vs-c.md", "improved_pipeline": "journal/entries/0068-s0d-cv-methodology-bakeoff-aerial-virtual-view-wins-legend-running-order-tracks-official-rp-rho-0.94-drone-anonymous-weak-view.md", "frames": "/tmp/s0d_scrutiny/ (aerial+drone key-moment frames; drone finish-zoom numbers crop)", "aerial_video": "data_cache/video/aerial_20260603_01.mp4", "drone_video": "data_cache/video/drone_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_results.html", "tasks": "#78 aerial finish+lateral, #79 drone scale+number-bridge (now primary), #80 fusion"}
created: 2026-06-06T23:58:21+00:00
updated: 2026-06-06T23:58:21+00:00
---
Reconciliation of a tangle: the original S0d agent kept running in the background after the
orchestrator (me) believed it had stalled and took over. It eventually completed a 26-min run with an
improved pipeline and wrote a SECOND journal numbered 0067 (now renumbered **0068**, status
superseded) whose verdict over-claims. This entry is the honest synthesis. The TRAIL is now: **0067**
= first-pass bake-off on a buggy scorecard (de-confounded by hand to a noise-dominated null; its
`data_cache/cv_s0d_*.parquet` row counts 972/2770 were later OVERWRITTEN by 0068's rerun, so 0067's
ref counts are stale); **0068** = improved render-adaptive pipeline + the legend-OCR result; **0069**
(this) = the correction + reframing + plan.

## THE CORE DISTINCTION 0068 BLURRED — overlay-read vs reconstruction
The aerial *Aerial Virtual View* render contains a **legend strip**: number-boxes drawn in live
running order. The broadcaster COMPUTES the running order (from their own tracking/timing) and PRINTS
it. OCR-ing that strip and finding it matches the official running-position sequence at ρ+0.94 is a
statement about OCR fidelity on a **derived data graphic** — it is NOT computer vision reconstructing
the race from the horses' positions. Two consequences:
1. **As a DATA SOURCE the legend is genuinely useful** — a dense (per-frame) running-ORDER feed and a
   re-id ANCHOR to bind sprite tracks to saddlecloth numbers. But running-ORDER is ~what the official
   chart already gives (at 4 poles); the legend just densifies it. It is NOT novel information.
2. **It is NOT the novel CV deliverable.** The reason to do CV at all is the **lateral / control
   signal** — rail-relative y_lat: ground saved/lost, traffic, the cover-then-swoop trip — which is
   INVISIBLE to the chart and to form. That requires GENUINE geometric reconstruction of positions,
   and THAT is exactly what is NOT yet validated: 0067's de-confound and 0068's own caveat both report
   the arc-length/timing rank is spurious (−0.50 / noise), and 0068's y_lat "trips consistent with
   RP" is a 7-runner QUALITATIVE plausibility anecdote, not a quantitative validation against any
   external lateral ground-truth.

So 0068's "AERIAL WINS decisively, recommend scaling (S0c)" is **not supported for the goal that
matters**: it validated the easy, non-novel, overlay-read part and left the hard, novel, lateral part
unproven. Scaling the aerial-legend read mostly re-acquires order data we already have.

## THE DRONE WAS DISMISSED ON AN INCOMPLETE LOOK
0068: "saddlecloth numbers are NOT legible from altitude ... there is NO honest way to bind
cluster→number from the video alone." This inspected only the high-altitude back-straight. **Frame
scrutiny of the drone clip (frames written to /tmp/s0d_scrutiny/):**
- Back-straight (t≈40-80s): high, near-top-down; BOTH white rails clearly visible (so the px→m scale
  is directly calibratable — fixes 0067/0068's 3-4× arc-length inflation + the y_lat width overspill);
  the field is longitudinally strung out so LATERAL rail-vs-wide position is clearly visible — the
  control signal, on REAL horses.
- Finish/run-in (t≈120-135s): the drone ZOOMS to a close oblique tracking shot where SADDLECLOTH
  NUMBERS ARE CLEARLY LEGIBLE — #3, #2, #9, #6, #8, #11 read by eye (with a "Photo Finish System"
  overlay). So the bridge is: OCR numbers in the finish-zoom window → map to silk-colour CLUSTERS →
  propagate labels BACKWARD along each track to the top-down segment. Non-circular, video-only.
Because the drone is REAL horse positions (not a synthetic overlay), once numbered it is the **better
path to a genuine, validated, numbered LATERAL reconstruction** than the aerial sprites (which collapse
into an unresolvable bunch at the run-in).

## MORE FRAME-SCRUTINY FINDINGS (the user's "look harder, did we miss anything")
- The AERIAL replay is a **SPLIT-SCREEN**, only half of which we use: LEFT ≈ a REAL broadcast tracking
  camera (real horses + silks + a race CLOCK and section-split readout — a second within-video view +
  timing); RIGHT = the synthetic AVV (sprites + legend strip + a top-right MINIMAP of field position
  on the track schematic + the odometer). The minimap + legend persist at the run-in when the sprites
  do not.
- START frame (t≈1s) is a REAL starting-gate camera: stalls numbered 1-13, silks visible — a clean
  silk↔number anchor independent of the legend.
- The tracked aerial paths END at ~91s, but the odometer reads ~150M-to-go at t=92 and "10M"/"FINISH"
  at t≈100 — the TRUE finish is ~t=98-100s. We DROP the field ~140m out and NEVER capture the winner
  #3's surge (7th→1st) in the recon; 0068's "winner #3 first" comes from the LEGEND, not from tracking
  #3 crossing the line.

## WHAT IS ACTUALLY TRUE AFTER S0d (the honest scorecard)
- ORDER / FINISH / RUNNING-POSITION: available cheaply (legend overlay ≈ official chart). Not novel.
- SECTION TIMES: aerial odometer/clock give ~0.8s MAE on 400m boundaries (coarse); the left-panel
  clock + splits are an unexploited finer source.
- LATERAL / CONTROL TRIP (the novel goal): UNVALIDATED for both views. Aerial sprites collapse at the
  finish; drone y_lat is mis-scaled + anonymous. This is the real open problem.
- Geometry plumbing (px→m scale, "C"-width, green-render detect): improved and mostly correct now.

## REVISED PLAN
- **Aerial (#78):** keep 0068's gains. Use the legend as a re-id ANCHOR and cheap order-data, NOT as
  the result. Genuinely extend the recon through the true finish (~t=100s), bind 12/12, and VALIDATE
  the y_lat trip (not just order). Optionally exploit the LEFT real-camera panel + its clock.
- **Drone (#79) — now PRIMARY for the novel signal:** fix the inter-rail px→m scale (arc-length→1650m,
  y_lat∈[0,21.3]); build the finish-zoom OCR silk→number bridge; deliver NUMBERED real-horse lateral
  tracks; validate the trip.
- **Fusion (#80):** only once both give a validated lateral read; the legend/start-gate numbers are the
  natural anchor that ties the drone's real-horse tracks to identities.
- CONTROL (S8) and CV scale-out (S0c) remain GATED on a VALIDATED LATERAL reconstruction — order-data
  alone (legend/chart) does not unlock the control cause.

Study-only; `src/hkjc/model` + `src/hkjc/core` untouched. The only `src/` changes are in
`hkjc.vision.mode_b` (render-adaptive detect/scale + `ocr_every`), all study-surface.
