---
title: Overlay furniture rejected at the pixel source (video-pooled static bands); R08 gold re-frozen after audit; the meta-principles behind the band-aids
id: 0126-overlay-bands-at-source
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "pixtag", "overlay", "tripwire", "r02a", "r08", "principles"]
refs: {"refs": "0125", "task": 104}
created: 2026-06-11T00:00:00+00:00
updated: 2026-06-11T00:00:00+00:00
---
## Owner directive: another wave + back out the GENERAL PRINCIPLES from the band-aid fixes

### The meta-result (measured 5x, both venues)

**Local statistics cannot separate render furniture from honest track structure; only pooling
along the right axis can.** Five local rules were tried and each was vetoed by R08's honest
mess: LOOP-1 line composition (real frag rails grade 0.24 < text 0.55), single-frame tracklet
grade (f818 0.59 vs fence 0.57-0.71), end-window grade (real shredded tips 0.00 = glyph tips),
per-frame glyph rows (rail DASHES and config-A cross-braces ARE squarish rows), per-frame
containment erase (rail dashes lie inside the legend band). The same data separates wide-open
once pooled: tracklet-median grade (>=0.96 vs 0.40), video-level row share (text 0.39-0.91 vs
braces <=0.04), px recurrence (digits ~0.9 vs a crossing rail ~0.02). Every surviving gate is a
pooled gate; every reverted gate was a local gate.

### What landed (this wave)

1. **Overlay-band rejection at the px source** (`aerial_rail`): pass 1 pools per-frame GLYPH
   ROWS (squarish filled comps, >=5 spanning >=0.21W) over the whole video -> STATIC bands
   (share >= `OVERLAY_SHARE_MIN` 0.20: R02A disclaimer 0.84/0.39, R08 legend digits 0.91; brace
   and dash false rows never accumulate). Pass 2 (`_thin_components`): inside a band, a
   glyph-shaped comp's SCREEN-STATIC px (white in >= `OVERLAY_REC_MIN` 0.15 of frames) are
   erased before candidate extraction. Three guards, each forced by a measured veto: rails
   CROSSING a band are not contained; rail dashes inside one are wide-thin; a transient rail
   fragment 8-connected to a static digit keeps its own low-recurrence px (px-level, not
   comp-level, erase). Wired into `clean_pixels(overlay=)`, `extract_rails` (two passes over the
   video) and the pixtag driver.
2. **Weak-vote lineage default** (`thread_rails`): a component with total radius-vote weight <
   `W_VOTE_MIN` 0.05 (noise floor ~0.006; real evidence 0.28-2.6) IGNORES its vote -- the
   primary-lineage default stands. Naming-by-continuity-to-a-neighbor was tried and REVERTED:
   weak components are born at continuity breaks, exactly where threading refused to link, and
   the tripwire caught it flipping R08's f810-816 rotation labels. (Containment hierarchy: no
   evidence -> don't guess; weak evidence -> don't override the prior; pooled evidence -> decide
   once.)

### R08 gold RE-FROZEN (protocol followed)

The px erase changes LOOP-1 output, so the old freeze diverged on exactly 7 frames (geometry
only -- no count or label changes). Each was adjudicated by eye against the gold overlay:
f930/f1374-78 = coverage GAINS (the new curve follows the rail ~100-215 px past where gold
stopped: the frozen end-under-coverage defect healing -- removing digit clutter un-knife-edged
real lines, +209/+357 px at f930/f1374); f838/f1876-78 = <=4 px median shifts where digit-px
fake support was removed. New output equal-or-better everywhere -> re-froze
`rail_tripwire_R08.npz`; tripwire PASSES against the new gold.

### R02A under the wave (owner-reported frames all verified on the re-render)

f630 missing rail -> full pair, correct naming. f642 bad fit -> clean pair fits. f660 weave
into disclaimer -> text rows COMPLETELY clean (no curve, no speckle). f666 swap -> stable
field-adjacent green. f1268-74 drift -> tight on the pair. Tail text zone (f2160-2504): clean.
**Tracklet table consolidated 35 -> 19**, with single pair components spanning f666-f1190
(n=240) and f2000-f2638 (n=320) -- with the furniture gone at the source, continuity holds
across whole race phases and naming rests on big pooled votes. This is the parsimony payoff:
the four LOOP-2 composition gates are now defense-in-depth rather than load-bearing; after
scale-out proves the bands generalize they are candidates for simplification.

### Remaining (Phase-2 geometry, unchanged inventory)

Pair-corridor float on rotated frames; occasional 1-frame dropouts; f708-class smooth-stray;
mid-gap band float. All geometry, all owned by the planned odometer-anchored smooth-in-s global
model (per-frame fits become observations; P2 proper).
