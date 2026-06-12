---
title: Label pass now enforces the co-occurrence axiom — double-green/double-blue sliver frames fixed (owner f1988); R08 gold re-frozen (label-only)
id: 0128-cooccurrence-axiom-labels
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "identity", "r02a", "r08", "tripwire"]
refs: {"refs": "0127", "task": 104}
created: 2026-06-12T00:00:00+00:00
updated: 2026-06-12T00:00:00+00:00
---
## Owner report: f1988 shows TWO GREEN rails

Root cause: in the f1984-1998 rotation the curves drop below the confidence span, the zone's
tracklets die, and BOTH curves become slivers labeled INDEPENDENTLY by nearest-tracklet -- both
inherited the racing name (G+G at f1988-1992, B+B at f1994). This violates the design's own
co-occurrence axiom ("two curves in one frame are two different rails"), which until now was
enforced everywhere EXCEPT the sliver-label pass. R08's frozen gold carried the same bug
(f810-814: double-green through the rotation).

## Fix (`thread_rails` label pass)

Per-frame reconciliation when two emitted curves share a name:
- a band curve or tracklet MEMBER keeps its name; the sliver takes the complement;
- two same-named SLIVERS are assigned by the 2x2 against the nearest RACING and FIXED loci.
  (v1 -- "flip the higher-cost one" -- killed the duplicates but ALTERNATED the assignment
  across consecutive frames on near-duplicate rotation fragments, R08 f810/f812; the loci are
  stable across frames, so the 2x2 cannot flicker.)

## Validation

- R02A f1984-1998: one G + one B per frame throughout; zero same-flag frames remain in either
  race.
- R08 f810-814: label-only change (exactly the 3 frames the tripwire flagged); the new
  assignment is CONTINUOUS with the audited f806-808 tracklet polarity (G upper fragment
  y~541-576, B lower y~599-635 across the whole rotation) -- strictly better than the frozen
  double-green. **Gold re-frozen; tripwire PASSES.** Suite 193 passed; ruff clean.
