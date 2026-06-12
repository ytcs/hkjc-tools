---
title: Horse tagging milestone opened — class-segmentation detector beats blob/shape routes; venue-model + pooled-motion gating
id: 0130-horse-tagging-segmentation-detector
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "detection", "horse-tagging", "segmentation", "r08", "r02a", "r03"]
refs: {"refs": ["0112", "0113", "0129"]}
created: 2026-06-12T00:00:00+00:00
updated: 2026-06-12T00:00:00+00:00
---
## Owner directive

Next milestone: TAG every horse in every frame (stable identity; harder than rails — relative
positions change, horses leave frame, occlusions). Owner flagged the existing blob detector as
possibly the wrong direction, suggested segmentation, and deferred to study findings.

## Pixel study (decides the primitive)

Crops at 4-10x across R08/R02A/R03 show the AVV render palette separates per-pixel:
turf (green hue, lit OR shadow-dark) | body (brown/tan hue) | white (bright desat) |
vivid (saturated non-green) | dark (near-black). The decisive measurements:

- **A cast shadow carries ZERO core-class pixels** (it is darkened TURF — green hue at low
  value, or near-black). The avatar core = body+vivid+small-white is unique to horses, so a
  segmentation detector rejects shadows BY CONSTRUCTION — the failure that breaks
  `detect_avatars` (24 dets on a 12-field at R02A f900) needs no shape heuristics at all.
- **Silks are often mostly WHITE** (R08), so 0112's vivid-only anchor structurally misses them;
  the white channel must be in the core, which makes white FURNITURE the main confounder —
  separated by GEOMETRY, not colour (see gates below).

## Detector (studies/cv_htag_detect.py — `detect_core`)

Core = (vivid | body | white_kept) clusters; jockey MARKERS (vivid/white blobs, merged
ELLIPTICALLY along the local rail tangent: helmet+torso merge, the adjacent lane does not)
are the instance authority: k>=2 markers in one cluster -> split px by nearest marker (bunch
resolution); orphan fragments (head, split cloth) attach inside the oriented avatar ellipse;
fragments/blazes fold by an elliptical det-merge. Count-aware, never count-forced.

White-furniture gates, each from a measured failure:
- rail dashes: ON a fitted rail curve (reuses the rail milestone's per-frame curves —
  locally identical to jockey whites, only the pooled rail structure separates);
- text rows/large glyphs: extent/elongation caps;
- painted on-turf "50M" glyphs (R03 f1700): a pure-white det must abut dark/body horse
  evidence — a horse is a MULTI-CLASS object, a glyph on clean turf is not;
- grandstand/backdrop: the venue-model IN-TRACK test — within ~215 px of the racing rail on
  the HORSE side. Side resolved per-query-local (SVD normal signs are arbitrary across
  queries — orient by physics): (1) away from the fixed rail when seen; (2) else the
  TURF-RICH side (stands abut the rail across non-turf); (3) else distance-only (honest).

## Measured (triage counts; verified by eye on the extremes)

R08: 11-13 dets on 9/10 frames (f600/f745/f336 = 11/11 exact); f800 (rotated bend) = 7 — the
small/far avatars at the top of the rotated view fall under the evidence floors (real recall
gap, bounded). R03: start frame f300 fixed by opening the band to the turf ROI (the 0.42h
band cut clipped the entire start-phase pack); f700/f1700 = 12/12. R02A: visible-set correct
on the shadow-stress frames by eye (f900/f1100 "under-counts" are horses genuinely out of
frame — field-count equality is the WRONG per-frame metric once the field strings out).

**The residual no per-frame gate can close: R02A f400-class CROWD dots** — the rendered
grandstand crowd is thousands of vivid+dark specks, locally identical to silks, and at f400
both rail sides are turf (HV infield), so even the side test degrades. Per the meta-law
(0126): only POOLING separates -> the TRACKER kills it: the camera follows the FIELD, so
horses are near-static on screen while ground-fixed furniture sweeps at the pan rate;
tracklet median ground-relative speed (phase-correlation camera shift on the turf) < 4
px/step = furniture. (Verified sign convention; implemented in studies/cv_htag_track.py.)

## State / next

- `studies/cv_htag_detect.py` (detector + bake-off driver; overlays data_cache/diag/htag/),
  `studies/cv_htag_rails_cache.py` (full-race per-frame rail curves, racing/fixed split ->
  data_cache/_htag_rails_{race}.npz), `studies/cv_htag_track.py` (LOOP 2 tracklets by
  continuity + appearance veto; LOOP 3 pooled-silk-signature chain linking under
  co-occurrence exclusion; furniture filter; full-race overlay render).
- Next: run the tracker end-to-end (R08 first), visual gate + red-team fan-out, then the
  identity layer (legend swatch colours + numbers as the partial anchor, draw order at the
  start) on top of stable tags.
