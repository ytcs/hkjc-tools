---
title: R08 re-verified + tripwire frozen; R02A red-team — text-arc tracklets poison the naming vote (one root cause)
id: 0123-rail-audit-tripwire-textarc-poisoning
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "identity", "tripwire", "red-team", "r02a", "r08"]
refs: {"refs": "0122", "task": 104}
created: 2026-06-10T00:00:00+00:00
updated: 2026-06-10T00:00:00+00:00
---
## R08: re-verified under the tracklet build -> TRIPWIRE frozen (owner request, 0122 item 3)

Re-rendered both races from HEAD; verified R08 myself (gold f768/f1736/f1880/f2158, ex-swap zones
f564/566, f930/956, f2740) + a 12-frame 2-batch red-team fan-out. **Identity is correct in every
audited R08 frame** — green on the field-adjacent racing rail, blue on the fixed rail, no swaps
(1222 two-rail / 5 single / only f956 lacks a running rail). The freeze:

- `studies/cv_rail_freeze_r08.py` -> `data_cache/rail_tripwire_R08.npz` (per-frame curves + labels).
- `tests/test_rail_tripwire.py` re-runs `track_sequence` on the cached clean pixels and diffs
  (sym median <= 2 px, p95 <= 15 px, exact label + count match; SKIPs where data_cache absent;
  ~3.5 min). **PASSES.** Re-freeze ONLY after a re-verified visual audit.

R08 residual defects (frozen as-is, upstream classes, NOT identity):
1. **LOOP 1 dropouts**: f564 (far rail visible full-width, 0 px kept -> rails=1 flicker), f956
   (near/racing rail at y~480-570 fully dropped; the lone blue curve was red-team-VERIFIED to be a
   REAL rail passing behind the legend icons, not a UI border).
2. **End under-coverage** (~50-250 px): curve ends stop short of kept support / visible rail on
   rotation frames (f768/f800/f860/f930/f2740) — under-GATHER at ends (not the 13 px trim).
   Phase-2 fill territory.

## R02A: 6-batch red-team fan-out -> ONE root cause for every naming failure

36 frames audited across all phases; every [hi] claim verified by my own eyes; mechanism dug out
via `thread_rails.last_debug` (tracklet/edge/vote dump):

**THE ROOT CAUSE: fabricated disclaimer-text arcs become confident tracklets and the wider-radius
vote then CORRECTLY ranks them as racing.** The text rows sit at the TOP of the frame (y~300-315),
genuinely farther from the bend's common center than the real rail — so the side-of-center
measurement is sound, the EVIDENCE is garbage (PHILOSOPHY §7b made flesh). Occupancy at
THREAD_OCC_MIN=0.55 does NOT separate: these text arcs run occ 0.50-0.85 (R08's gappy bend rails
need the floor <= ~0.6, so raising it trades one failure for another).

- **comp 10 (f1250-1864)**: the merged config-A band (T20, 307 frames) has NO real partner — its
  only co-occurrence edges are to text tracklets T21-T24 (y~301-315). Vote 0.071/0.279 -> the TEXT
  color is named racing, the band goes FIXED. 5/6 audited frames in this zone show the lone band
  curve labeled BLUE; f1650 shows a green "racing rail" drawn THROUGH the Chinese disclaimer
  glyphs (verified). f2160 same class (blue on text, verified).
- **comp 6 (f656-674 turn flips)**: real rail T12 vs text tracklet T13 — first_wider 0.544/1.700
  -> text=racing, real rail=fixed. This is the green/blue alternation the turn batch found
  (f640 G -> f660 B -> f680 G -> f700 B -> f720 G across fragmented mini-components).
- **comp 7 (f692-710, the 0122 leftover)**: at f692 the 907 px text arc passed confidence (occ
  0.55, borderline); the THRESHOLDLESS 2x2 relative assignment at f706 then STITCHED the real
  second rail onto the text tracklet (T15) — the poisoned f692 edge (w=0.214 vs the real pair's
  ~0.02) named the whole chain wrong.
- **Counterfactual (checked against the dump): excluding text edges, the no-evidence default
  (primary lineage = racing) labels comps 6/7/10 correctly.** Comp 8/13/14 + start/tail are
  CORRECT as-is (f2520 re-adjudicated: green IS on the field-adjacent lower member — the 0122
  "comp 14 inverted" suspicion does not reproduce on the fresh render).

Non-naming R02A classes (same fan-out): merged-pair single fit floats mid-gap between the ~10 px
pair (config-A reality below PAIR_SEP; Phase-2 split/fill, the label is the band's job); LOOP 1
text-pixel retention is INTERMITTENT (contrast-dependent, 0-4k px frame to frame); end-trims
behind the legend row; stray kept px on silks/caps.

## Next (priority order)

1. **Kill the f690/f1650/f2160 fabricated-arc class UPSTREAM** (0122 item 4, now urgent — it is
   the single root cause of every R02A naming failure). The physical discriminator is support
   COMPOSITION, not occupancy: a rail's support comes from FEW LONG components (dashes tens-to-
   hundreds px), a text arc gathers DOZENS of tiny glyph blobs. Gate inside
   `_enumerate_curves`/`fit_rails` (e.g. max-component-fraction or component count per length).
   Validate against the R08 tripwire (now the safety net) + R02A re-render.
2. If any residue survives upstream rejection: exclude non-rail-grade curves from VOTE evidence
   (containment, §7b) — the default then names fragmented components correctly.
3. Then re-run the §0b fan-out on R02A; scale-out (R01/R03/R06/AWT); Phase 2 (smooth-in-s
   spanning/fill) closes the mid-gap band float + end under-coverage + LOOP 1 dropouts.
