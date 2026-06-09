---
title: Shape-template avatar detection — whole-avatar silhouette + silk verifier (R&D finding)
id: 0113-shape-template-avatar-detection-rnd
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
tags: ["cv", "aerial", "detection", "shape", "template", "silk", "shadows", "bunching", "handoff"]
refs: {"refs": ["0111", "0112"]}
created: 2026-06-08T18:49:51+00:00
updated: 2026-06-08T18:49:51+00:00
---
## TL;DR

The user's SHAPE hypothesis is **viable and, combined with the silk anchor, is the strongest AVV
detector I have measured.** The avatar genuinely has a consistent silhouette — a stubby dark/tan
horse body (~26–45 px long, depending on zoom) with a small bright saturated silk patch on top,
oriented along the rail tangent — and matching that *whole shape* (body component + silk core)
**rejects cast shadows by construction** and **recalls dark-silk runners** that the silk-only route
(0112) missed.

Measured by EYE on annotated overlays (not counts alone): on R08 the shape detector lands **11/11 on
real horses** for the spread frames with high positional accuracy, while the current `detect_avatars`
over-counts (13–24) and **fabricates detections on the rail line / empty turf** (the "right count,
wrong positions" failure — vivid in `shape_extra_f00900.png`). On the **R02A shadow stress** the
shape detector returns **9 detections, all on horse bodies, none on shadows**, versus
`detect_avatars` = **24** (the shadow blow-up). Recall is bounded below only by **bunching** (merged
bodies at the start) and a couple of dark-silk horses inside shadow-merged streaks.

**Recommendation:** promote a **silk-anchored shape detector** — silk patch as the shadow-immune
anchor (0112's load-bearing primitive), the horse-shape body component as the recall booster for
dark silks and the shadow rejector, NMS + the field-count cap as a count-aware (never count-forcing)
top. This dominates both `detect_avatars` (precision/positions) and silk-only (recall).

## What was built

`studies/cv_avatar_shape.py` (study only — imports `detect_avatars` / `track_band` / `render_active`
for comparison; everything else local and adaptive). Pipeline, all derived from the frame:

1. **Channel maps** from the per-frame turf hue: `body` = non-turf, non-bright pixels (dark/tan
   horse — but ALSO cast shadows); `silk` = vivid bright non-turf pixels (the jockey patch ONLY —
   a shadow carries none).
2. **Adaptive scale** = lower-central principal-axis length of the closed body components (the
   avatar zoom changes through the race; measured, not hard-coded). **Orientation** = the rail
   tangent (reused from `aerial_rail`), the travel-direction prior; handles bends.
3. **Body components**: close the body map along the rail tangent to reconnect each avatar's
   head/body/tail fragments into one oriented blob without bridging across-track neighbours.
4. **Shape descriptor per component**: length / aspect vs the horse template, plus whether a vivid
   silk patch sits inside the footprint. Score = length-fit + aspect-fit + silk bonus.
5. **Shadow / rail-band rejection**: a component that is too long (>1.9× scale), too thin
   (aspect > 5), or too big AND has no silk inside is dropped — that is exactly a cast shadow or
   the dark fence-shadow band.
6. **Silk anchors fill in**: vivid silk patches with no surviving body component (bunched /
   fragmented) are kept as real horses; greedy NMS then yields one detection per avatar; the field
   size caps low-score extras (count-AWARE — peaks are never invented, unlike the count-FORCING
   split in `detect_avatars`).

Overlays for every frame in `data_cache/diag/shape_*.png` (green = silk-confirmed, orange =
body-shape only/dark-silk, red × = `detect_avatars` for comparison).

## Measured results (field in parens; shape count / silk-confirmed / detect_avatars)

| frame | field | shape | silk-conf | detect_avatars | by-eye positions |
|-------|-------|-------|-----------|----------------|------------------|
| R08 336 (bunched start) | 11 | 9 | 6 | 11 | all 9 on horses; 2 missed in the tight pack |
| R08 406 | 11 | 11 | 11 | 13 | 11/11 on horses; detect_avatars over by 2 |
| R08 600 (spread) | 11 | 11 | 11 | 11 | 11/11 on silks, well-centred |
| R08 745 (bend) | 11 | 11 | 11 | 11 | on tilted avatars along the curved rail |
| R08 1080 (spread) | 11 | 11 | 11 | 11 | 11/11, crisp on silks |
| R03 700 | 12 | 12 | 12 | 12 | 12/12 on silks, well-centred |
| **R02A 900 (shadows)** | **12** | **9** | **9** | **24** | **all 9 on bodies, ZERO on shadows** |

Unseen spread frames (sanity, not in the brief's set): R08 450/500/900/1500/2000 → shape **11** each
(10–11 silk-confirmed), detect_avatars 12–23. On `shape_extra_f00900.png` detect_avatars dumps a
whole cluster of ×'s onto the green rail line while the shape detector's circles all sit on the pack.

## Does it work?

- **Shape-template detection — yes.** The avatar silhouette is consistent enough across frames AND
  races (R08/R03/R02A) to detect by shape. The rail-tangent orientation prior + close-along-tangent
  handles rotation through bends without a rotated-template bank.
- **Positional correctness — high** (the load-bearing check). Circles land ON the silk/body, not in
  the gaps. This is the decisive advantage over `detect_avatars`, whose count is often right while
  its *positions* are fabricated (rail-line and empty-turf ×'s; the blob-split to hit the count).
- **Shadow robustness — strong (R02A).** The silk-inside-footprint requirement + the over-length /
  over-aspect rejection kills the cast-shadow and fence-shadow false positives that explode the
  dark-body route (9 clean vs 24).

## Recall / precision

- **Precision: high.** Across all frames the kept detections land on real horses by eye; no shadow
  hits, no rail-line hits. The field-count cap removes the few low-score extras without inventing
  positions.
- **Recall: ~75–100%.** 11/11 or 12/12 on every spread/bend frame (R08 406/600/745/1080, R03 700).
  Drops on **bunching** (R08 336: 9/11 — two bodies merge in the tight pack) and on **R02A** (9/12 —
  a few horses sit in shadow-merged streaks or have near-turf-dark silks). This already **beats the
  silk-only ~33–64%** because the body-shape channel recalls the dark-silk runners (the orange
  detections) that have no vivid patch.

## Failure modes

1. **Bunching / overlap** (starts, tight packs): adjacent bodies close into one blob and the silks
   touch → one detection for two horses. The count cap cannot recover a merged pair. This is the
   dominant recall limiter and is shared by every method.
2. **Shadow-merged dark silks (R02A):** a dark-silk horse whose body fuses with its own cast shadow
   becomes an over-length streak with the silk at one end; if the silk is also faint it can be
   rejected as shadow. Trade-off knob: the over-length / aspect thresholds.
3. **Scale drift:** the adaptive scale is robust but pins to its clip floor (26 px) when bodies are
   heavily fragmented at far zoom; a silk-spacing cross-check would firm it up.
4. **Twin detections on one long avatar** on the bend (seen once) — over-eager body+silk both
   surviving NMS; a slightly larger NMS radius along the tangent fixes it.

## Robustness to style drift

Good by construction and verified across three races with different lighting (R08 bright, R02A strong
cast shadows, R03 flat): turf hue read per frame, scale measured per frame, orientation from the rail
geometry, shape thresholds expressed as multiples of the measured scale, vivid-silk thresholds in
HSV with margin, graceful component/silk fallback. The brittle constants are the shape-rejection
multipliers (length 1.9×, aspect 5) — single, data-checkable knobs, not colour/coord tables.

## Recommendation (vs silk-anchor 0112 and the current detector)

1. **Promote a silk-anchored SHAPE detector** to replace `detect_avatars`'s count-forcing core. Keep
   the silk patch as the anchor (0112) — it is the shadow-immune primitive — and add the horse-shape
   body component as (a) the dark-silk recall channel and (b) the shadow/rail-band rejector.
2. **Keep the field size as a count CAP, not a count force** (count-aware): never split a blob to
   manufacture a position. Wrong-but-confident positions poison the downstream 2D reconstruction
   more than an honest missed detection (which the motion model / re-id bridges).
3. **Backstop bunching** with the temporal tracker (carry IDs through a merge) rather than trying to
   resolve it per-frame.
4. **Pair with the legend colour/identity prior** (0112) for free IDs on the runners it lists.

Net: shape-template detection is **the right detection primitive when fused with the silk anchor** —
it fixes both the shadow explosion (dark-body / `detect_avatars`) and the dark-silk recall gap
(silk-only), with markedly better positional fidelity than the current count-forced detector.

Artifacts: `studies/cv_avatar_shape.py`; overlays
`data_cache/diag/shape_R02A_f00900.png` (shadow stress: 9 on bodies, 0 on shadows, vs 24),
`data_cache/diag/shape_extra_f00900.png` (detect_avatars fabricates a cluster on the rail line; shape
stays on the pack), `data_cache/diag/shape_R08_f01080.png` and `shape_R03_f00700.png` (clean
full-field, circles centred on silks), `data_cache/diag/shape_R08_f00336.png` (bunched-start limit).
