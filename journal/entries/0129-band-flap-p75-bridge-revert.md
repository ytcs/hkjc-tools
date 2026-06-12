---
title: Band<->pair flap killed (p75 resolvability + corridor split pool); band-bridge transport REVERTED on a measured migration counterexample
id: 0129-band-flap-p75-bridge-revert
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "identity", "config-a", "r02a"]
refs: {"refs": "0128", "task": 104}
created: 2026-06-12T00:00:00+00:00
updated: 2026-06-12T00:00:00+00:00
---
## Owner report: f1862-1868 "rail bouncing"

Anatomy: green at y~636 (band single hugging the lower member) -> y~622 (split, upper member)
-> back to y~636 (1-frame band relapse at f1866) -> y~620. Two knife-edges:
1. f1854-1862 never resolved the pair: the 8 px gather around the merged fit cannot contain a
   partner ~16 px away, per-frame enumerate could not span the partner's sparse px, and the
   band-split fallback ran on `run_pix` alone.
2. f1866 resolved the pair and the midline split succeeded -- then died 0.6 px below the
   PAIR_SEP floor (median gap 9.8 vs 10.4).

## Fixes

- **Corridor split pool**: the band-split fallback pools `rest` px within 2*PAIR_SEP of the
  merged fit before splitting (the corridor, not the gather band).
- **Resolvability at p75 of the gap profile** (in `_split_band` + the touching-fits
  adjudication): a REAL pair honestly CONVERGES toward one end (f1866: median 9.8 / p75 14.2)
  while a sub-resolution pseudo-pair is tight EVERYWHERE (f630: median 9.4 / p75 9.5) -- the
  median knife-edged them 0.36 px apart. (A 0.8x hysteresis variant was tried first and
  REVERTED: it re-admitted f630's pseudo-pair, whose 2x2 then crossed internally.)
- Result: the pair resolves from f1856 onward, stable to the end of the zone; one honest
  band->split transition replaces the bouncing; the bend pseudo-pairs stay collapsed.

## Band-bridge transport REVERTED (0127's mechanism) — measured counterexample

While re-validating the bend under these changes, the f628 naming regressed and the transport
was put under direct measurement (`/tmp/side_trace2`-style: entry/exit member sides vs the
orientation-chained band curve, interior-overlap-filtered):
- band f560-626 (straight): racing side +4.9 at entry, +8.0 at exit -- transport CONSISTENT;
- band f630-664 (the bend): racing side +3.7 at entry, **-5.8 at exit, with ZERO order flips
  in a sound chain** -- physically impossible under the invariant, UNLESS the reference itself
  crossed a member. It did: the side reference is the MERGED FIT, which migrates across the
  corridor as the pair's support balance shifts through the bend; migration is
  indistinguishable from camera pan without seeing the members (which is the band's defining
  condition). The invariant "side vs the band curve" is NOT crossing-proof.
Three repair rounds (local normals for curved bands, overlap-tangent order continuity,
interior-correspondence filtering -- each fixing a real subsidiary bug) did not and cannot fix
the reference-migration hole. Per the containment hierarchy ("don't guess"), the bridge is
REMOVED; ground truth was re-established from the RAW frame (two genuine white rails ~10 px
apart at f628; an intermediate shadow hypothesis was also eliminated).

## Residual (bounded, documented)

Isolated 1-frame pair components sandwiched between bands (f628, f558) name their members by
the lineage default = coin flip; their members are <= 10-12 px apart, so the worst case is
green on the wrong line of a sub-resolution pair for a single frame. Closing it for real means
identity evidence through the merge that does NOT use the merged fit as a reference -- Phase-2
(smooth-in-s pooled geometry) owns that.

## Validation

R08 tripwire PASSES (band machinery inert in config-C; the p75/pool changes touch only paths
the frozen set never takes -- verified empirically). Suite 193; ruff clean. R02A: zero
same-flag frames; f1850-1876 stable; bend/tail/start zones unchanged from the audited state.
