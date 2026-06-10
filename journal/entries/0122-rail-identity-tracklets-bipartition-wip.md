---
title: Rail identity rebuilt — tracklets + co-occurrence bipartition + per-component naming; R02A turn repaired; naming bits WIP
id: 0122-rail-identity-tracklets-bipartition-wip
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "identity", "tracklets", "bipartition", "r02a", "r08", "wip"]
refs: {"refs": "0121", "task": 104}
created: 2026-06-10T00:00:00+00:00
updated: 2026-06-10T00:00:00+00:00
---
## Why (owner findings)

1. **R08 still swapped racing<->fixed mid-video** (owner report; triage confirmed 34 abrupt running-rail
   jumps + 14 sustained wrong-vote stretches; visually verified swaps at f564->566 and f930->956). Root
   cause: `couple_rails` (0121) re-decided identity per frame against a +/-20-frame median — a swap
   SUSTAINED longer than the window becomes its own majority and survives. Identity must be GLOBAL.
2. **Owner principle banked: NO downstream step may assume upstream output is correct** (esp. the far
   rail in config A). The per-frame fit emits hedge/margin slivers through fast turns and — decisively —
   a fabricated 914px ARC through the disclaimer TEXT (R02A f690) that passed every extent-based gate.
3. **Owner naming rule (HKJC_SPECS §1): racing = the WIDER/FURTHER rail** (movable, horses race just
   outside it); fixed = inner/infield boundary.

## What was built (`thread_rails`, replaces `_assign_racing_by_radius` + `couple_rails`)

The unifying principles, each replacing a bandaid:

- **Tracklets (identity = global linking, not per-frame labels).** Confident curves chain frame-to-frame.
  With BOTH rails seen, assignment is RELATIVE (2x2 set-distance pairing — common camera motion cancels;
  ordering cannot flip without a crossing → survives the R08 rotating bend with no threshold). A LONE
  curve extends a tracklet only within a tolerance kept BELOW the config-A pair gap (~15px); where
  continuity genuinely breaks the tracklet ENDS — a wrong link is contained to its own segment.
- **Confidence = extent AND support-occupancy** (`THREAD_CONF_FRAC`, `THREAD_OCC_MIN`): the pixtag
  continuity principle reused at the curve level. Kills the f690 text-arc (long but patchy) and the turn
  slivers as identity/naming evidence; they remain in the OUTPUT (honest short rails) and are labeled by
  a DIRECTED nearest-locus cost (set-cost let a sprawling arc "contain" the rail and steal its name).
- **BAND state (config-A merge).** A lone curve matching BOTH alive tracklets = the close pair merged
  into one fitted band -> identity is DEGENERATE: label the band frames RACING (the band contains the
  racing rail, off <= the pair gap) and RESTART identity at the next split. No coin-flip survives a merge.
- **Naming = one bit per component.** Co-occurring curves are different rails (no-cross gate), so
  co-occurrence edges 2-COLOR each connected component of the tracklet graph; ONE weighted vote per
  component decides which color is racing. Evidence = **side-of-common-center**: fit one LSQ circle to
  the longer curve of a co-occurring pair, compare both curves' median distance to THAT center (wider =
  farther). Sidedness survives the oblique projection; comparing two separately-fit radius MAGNITUDES
  does not (it inverted: T32 "wider" than T33-35 yet "narrower" than T36 — same physical rail).
- **Unified `_split_band`** (midline-side partition + separation gates) replaced `_split_pair`
  (2-means+valley) AND the body of `_decouple_pair` — one mechanism for brace-joined pairs, weaves, and
  crossings. NOTE: the de-weave TRIGGER (`WEAVE_FAC`/`CLOSE_FAC`) is NOT a removable bandaid — it is the
  domain-of-validity condition (on widely-separated rails the robust midline collapses onto the denser
  rail and the split would shred it).

## State (uncommitted-to-main; branch `cv-rail-identity-threads`)

- **R02A f600-688 repaired** (was: green absent/flipping through the turn; owner called f636-724
  "disastrous"). f712-724 repaired (green on the real rail incl. the owner's f720). Diag:
  `studies/cv_rail_swap_triage.py` + `thread_rails.last_debug` (tracklets/edges/votes introspection).
- **R02A naming bits, visually adjudicated** (zone renders /tmp/zone2, venue geometry = field-adjacent
  member is racing): comp 8 (f726-1170) CORRECT after the common-center flip (green = upper/field-side
  member, f730); comp 13 (f1900-2004) CORRECT (f1920); comp 14 (f2066-2638, finish straight) looks
  INVERTED at f2520 (horses BELOW the rail there -> racing should be the lower member); comp 10
  (f1250-1864) unresolved. **f692-710 chain still labeled fixed** (its component's thin f706/708
  evidence; the pair there is REAL — two supported rails — but the bit came out wrong).
- **R08: NOT yet re-verified under the tracklet version** (it was perfect under the first chain version —
  all-two-rail relative assignment, which tracklets preserve, so no regression is EXPECTED, but R08 is
  1226/1227 two-rail so the tracklet graph is one component with massive vote evidence). MUST re-render
  + re-verify + red-team audit before any promotion.

## Open / next (priority order)

1. **Re-render R08 + verify gold frames** (f768/f1736/f1880/f2158, ex-swap zones f564/f930/f2740) under
   the tracklet build; then R02A full re-render; then the §0b red-team fan-out on BOTH.
2. **Per-component naming residuals:** adjudicate comp 10/14 (and the turn minis); candidate fixes:
   (a) DECISIVE cross-gap component linking (2x2 across <=10-step gaps with a margin gate) to pool vote
   evidence across live-cam interludes; (b) if image-space evidence stays thin, the principled next
   anchor is the ODOMETER (rails are functions of arclength s — Phase 2 "smooth in s"), which also
   closes CV_RECON §7.
3. **Owner request — R08 pixel-level regression TRIPWIRE:** freeze the last owner-confirmed-perfect R08
   reconstruction (per-frame rail curves + running labels, e.g. `{keys, pts, running}` npz from
   `track_sequence` at that commit) into `data_cache/` + a pytest that diffs future runs against it
   (tolerance ~1-2px median per curve, exact label match). Add once R08 is re-verified under this build.
4. The f690-class fabricated-arc FIT defect (long smooth arc through text clutter) should eventually be
   rejected UPSTREAM (occupancy gate inside `fit_rails`/`_fit_gather`), not only quarantined downstream.
