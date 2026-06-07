---
title: Aerial rail: live-intro trim + solid-rail fallback + spike/positive-perp fixes; A-config coalesce open
id: 0100-aerial-rail-live-intro-trim-solid-rail-fallback
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "rail", "reconstruction", "handoff"]
refs: {"updates": "0099", "open": "A-config fragment coalescing"}
created: 2026-06-07T22:30:00+00:00
updated: 2026-06-07T22:30:00+00:00
---
## Update on the 0099 roadmap (aerial rail backbone)
Done + validated on HV "C" (R08); committed alongside this entry.

- (a) LIVE-INTRO TRIM: new `aerial_layout.render_active(panel)` — the replay opens full-width with the
  live TV broadcast (f0-~330) before the split-screen render. find_layout.aerial_active is unreliable
  there. The clean cross-config signal is the bottom SILK-LEGEND-box count (0 through the whole live
  intro, >=1 for every render frame; validated on "C" and "A"). The tracker now gates on it; the
  start-of-plot garbage is gone.
- (b) SOLID-RAIL FALLBACK: a solid continuous rail has coverage > the 5.0 ceiling and was rejected
  (the mid-race f2028-2156 nmain=0 block). Fix: when the normal 1.0<=cov<=5.0 filter leaves nothing,
  admit high-coverage THIN lines (spread<=5px). The ceiling still rejects the fat field-cluster line
  when a proper rail coexists (f1080). Recovers most of the block.
- GLITCH SPIKES fixed: f770 (both rails far -> the 150px field gate now skips it) and f1280 (the real
  rail momentarily collapsed to cov 0 and a far stub was taken via the un-gated "single" branch — the
  bad-fit field gate now applies to the single-rail case too).
- y_lat is now a POSITIVE distance from the rail (signed lateral is unphysical: the infield side has
  no runners). Store the running-side magnitude, clamped >=0.

Net: the whole-race |perp| band is coherent + physical; residual non-smoothness is the deferred PX
SCALE (camera zoom), to be normalised via known geometry later.

## STILL OPEN
- (c) A-CONFIG single-rail path: course "A" has ONE physical rail, but `enumerate_lines` FRAGMENTS it
  into 2-3 candidate "lines" (sub-segments), so selection picks a fragment / returns None (f2000).
  Needs a FRAGMENT-COALESCING pass (merge near-collinear overlapping lines into one physical rail)
  in enumerate_lines — a core change used by "C" too, so validate on both before landing. The
  single-rail branch + solid-rail fallback are in place; coalescing is the missing piece.
- Metric SCALE normalisation (px->m via known width/geometry) for y_lat. Left-handed-track validation
  (selector is handedness-agnostic — field-median, no hardcoded side — but unvalidated).

## KEY FILES
- hkjc/vision/feeds/aerial_rail.py    select_running_rail (field-median + temporal + gates + fallback)
- hkjc/vision/aerial_layout.py        render_active (live-intro trim signal)
- studies/cv_aerial_track_id.py       whole-race tracker (positive y_lat)
