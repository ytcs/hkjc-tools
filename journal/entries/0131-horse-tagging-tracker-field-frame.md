---
title: Horse tagging LOOP 2/3 — field-frame tracking; phase-correlation "ground reference" FALSIFIED by member accounting; HUD killed by structure not appearance
id: 0131-horse-tagging-tracker-field-frame
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "horse-tagging", "tracking", "identity", "r08"]
refs: {"refs": "0130"}
created: 2026-06-12T00:00:00+00:00
updated: 2026-06-12T00:00:00+00:00
---
## The reference-frame inversion (the session's load-bearing lesson)

The furniture filter was designed as "ground-relative speed ≈ 0 ⇒ furniture", with the
ground's screen motion measured by phase correlation on the turf. First run: top "horse"
chain a fragmented 61 members; 124 "furniture" tracklets dropped. MEMBER ACCOUNTING broke
the story: total dets 14,956; "furniture" held 13,851 — and the biggest "furniture"
tracklets ran f336→f2770 (1,212 members, avatar-sized areas, sitting ON horses when drawn).
**The "furniture" were the horses.** The phase correlation does NOT measure the ground: it
locks onto the FIELD (the avatars are the dominant high-contrast content), so the
"ground-relative" residual was field-relative all along — horses ≈ 1 px/step, real
furniture (ground-fixed, sweeping at the pan rate) HIGH. Numbers pattern-matched the
design's prediction with the labels swapped; only the accounting + looking caught it.

Fix is simpler than the broken design: drop phase correlation entirely; measure the FIELD
shift directly as the median displacement of matched dets per step (11 of ~12 dets ARE the
field), track in the field frame (predict = pos + field-relative velocity + accumulated
field shift), and invert the filter: furniture = field-relative residual > 6 px/step.

## HUD text/odometer — killed by structure, not appearance

Remaining contamination was screen-static HUD: disclaimer text rows (white glyphs gain
dark-adjacency from the fence lines BEHIND them) and the odometer/minimap box (985-3,551 px
blobs + small green-arrow fragments). Three gates, each structural:
- det AREA cap (avatars measure 150-280 px, ~400 max; the minimap segments 985+);
- TEXT-ROW filter: >=3 PURE-white dets sharing a y-line in one frame = a text row (glyph
  rows organize in y; horses are never simultaneously pure-white AND collinear); drop all
  pure-white dets within the row band. (A chain-level static test failed first: linked
  glyph chains WANDER in x across the row — screen-static per fragment, not per chain.)
- the odometer box is an OPAQUE product HUD at a fixed panel position (aerial_layout's odo
  ROI) — zeroed before detection; horses behind it are occluded, so this is honest.

## LOOP 3 (chains) under the exclusion axiom

merge_duplicates (co-temporal + co-located = same horse's split fragments) -> two-tier
links (short gap ≤40f: position-gated in the field frame; long gap ≤400f: pooled-silk-
signature only — out-of-frame re-entries and FINISH-paint dropouts) -> count-aware final
merge (non-overlapping, silk-compatible; a CAP, never forcing).

## R08 measured state (visual gate on the full-race render)

11 chains span ~the whole race (1,221/1,208/1,201/1,198/1,089/969/937/925/921/848/737 of
1,227 frames); f1600/f2000 carry EXACTLY tags 0-10, one per horse, circles centred (by
eye). Residuals, all bounded + diagnosed: (a) rotated-bend recall dip (f800: 6/11 — the
small far avatars, a detection gap from 0130); (b) late-race tag handoffs (FINISH-zone
dropouts split some horses into big-chain + mid-chain pairs that the overlap exclusion
correctly refuses to merge); (c) anonymous tags — the saddlecloth-number binding (legend
swatches + draw order) is the next layer. R02A (crowd stress) + R03 runs pending.

## Cross-race (R03 + R02A)

- **R03** (757 frames, field 12): 12 substantial chains (333-734 members) + fragments;
  f700/f1700 verified by eye — every visible horse tagged once. Generalizes.
- **R02A** (1,126 frames, field 12, the crowd/shadow stress): non-crowd zones CLEAN by eye
  (f900 shadows, f1600, f2400 — one tag per visible horse; top chains 1,073/1,068/1,037
  ≈ full race). **Open class: the early-straight CROWD zone (f340-630)** — ~7 crowd dets/
  frame survive (at f400 both rail sides are turf so the side test is honestly ambiguous,
  0130), and in DENSE clutter the residual filter is defeated by aliasing: a crowd track
  can hop dot-to-dot within the gate, riding the field reference (residual ≈ 0 without
  being one object). 195 crowd tracklets ARE caught by the residual filter; the leak is
  the hop-aliased remainder. The honest closure is POOLED GROUND-FRAME geometry: the
  stands are a fixed band along the track, and sidedness pooled from the frames where the
  fixed rail IS seen (f700+) transports along the rail — Phase-2 (same family as the
  smooth-in-s rail model), not a per-frame bandaid. Also minor: two legend-edge tags
  (f1600 tags 47/65) — legend_top cut misses swatch tops by a few px on some frames.

Artifacts: `studies/cv_htag_track.py` (+detector in 0130's `cv_htag_detect.py`); renders
`data_cache/diag/htag_track_{race}/`; tracks `data_cache/_htag_tracks_{race}.npz`; dets
cache `data_cache/_htag_dets_{race}.npz` (delete after detector changes).
