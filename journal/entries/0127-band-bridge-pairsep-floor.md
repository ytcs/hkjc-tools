---
title: Identity transported through the band (side-order invariance) + PAIR_SEP floor on adjudicated splits — bend-entry naming fixed; final 48-frame fan-out
id: 0127-band-bridge-pairsep-floor
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "identity", "config-a", "r02a", "red-team"]
refs: {"refs": "0126", "task": 104}
created: 2026-06-11T00:00:00+00:00
updated: 2026-06-11T00:00:00+00:00
---
## Final §0b fan-out (48 frames, 6 auditors) on the post-0126 build

44/48 PASS outright; naming polarity correct everywhere EXCEPT one [hi]: the bend ENTRY
(f628-650) had green on the hedge-side member (owner's f630/f666 reports, surviving the wave).
Verified by my own eyes at 4x. Other findings were the known Phase-2 geometry classes (end
under-coverage f616/f706/f686/f696/f1940/f1968; f1912 left-half float; honest occlusion bridges
over the winning-post bed / odometer box).

## Diagnosis — two stacked causes, both structural

1. The bend-entry "pairs" came through the touching-fits adjudication with median gaps of ~4-9
   px — BELOW `PAIR_SEP` (10 px), which the band path treats as ONE observable rail. At that
   separation the 2x2 has no assignment margin: comp f628-630 was internally CROSSED (green on
   the upper member at f628, lower at f630) — un-nameable by any evidence.
2. Even resolvable weak pairs after a band restart had no naming evidence: their radius votes
   are noise (wden 0.003-0.01 vs real 0.22-2.8; raw signs measured 2/3 "correct" = coin flip),
   and the W_VOTE_MIN lineage default is a coin flip too (the band re-anchors the run slot to
   whichever member dominates support).

## Fixes

- **PAIR_SEP floor on adjudicated splits** (`track_sequence`): a touching-fit split is accepted
  only at median gap >= PAIR_SEP — the same floor the band path uses. Sub-floor pseudo-pairs
  collapse to band singles (racing-by-content green: rendering correct by definition). The bend
  f630-f666 and the f1416/f1648 1-frame minis all collapse; the real pair re-resolves from f670
  (47 px) onward.
- **BAND BRIDGE** (`thread_rails`): identity is now TRANSPORTED through the merge. The pair's
  side ORDER is crossing-invariant, and the band keeps 1-step in-tolerance continuity, so the
  member on side s of the orientation-chained band curve at entry is the same physical rail as
  the member on side s at exit. Bridges link entry/exit tracklets; vote-weak PAIR components
  inherit naming via BFS from vote-strong components along bridges. Guards: orientation chain
  killed on gaps > LINK_GAP_MAX; degenerate/same-side readings produce no bridge; only
  PAIR-LIKE components (tracklet co-occurrence >= 50% of member frames) inherit — band-content
  threads (whose "side" is ill-defined) keep the racing-by-content default; vote-strong
  components are never overridden. This is NOT the tripwire-vetoed chaining: every hop is a
  link the threading itself trusts (the vetoed case chained across REFUSED links).
- (also this round, 0126-followup) post-split acceptance no longer re-imposes gmin > NOCROSS —
  the side-split cannot cross; end wobble of the refits is not a crossing (fixed the f1912
  1-frame dropout).

## Validation

- f628: green now on the horse-adjacent member (eye-verified at 4x), named through the bridge
  chain from the strong f670-1190 component. f630-f666: honest band singles. f670+: pair
  correct.
- **R08 tripwire: PASSES untouched** (no bands in config-C: the bridge machinery is inert; the
  PAIR_SEP floor only constrains the touching path, which R08 never takes in the frozen set).
- Suite 193 passed; ruff clean.

## Remaining open inventory (all Phase-2 geometry; nothing naming-bearing)

End under-coverage at frame edges/rotations (f616/f706/f686/f696/f1940/f1968: 60-200 px of
supported rail uncovered); f1912-class corridor float (~13-18 px over half a span); honest
straight bridges across occlusions (winning-post bed, odometer box) that read as overlay
crossings; cosmetic 1 px end steps. All owned by the planned odometer-anchored smooth-in-s
global model (per-frame fits become observations with pooled-geometry outlier gating).
