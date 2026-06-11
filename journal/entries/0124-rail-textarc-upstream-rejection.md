---
title: Text-arc rejection landed as layered composition gates (seed/emit/tracklet); R08 tripwire green, R02A re-audited
id: 0124-rail-textarc-upstream-rejection
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "identity", "tripwire", "red-team", "r02a", "r08"]
refs: {"refs": "0123", "task": 104}
created: 2026-06-10T00:00:00+00:00
updated: 2026-06-10T00:00:00+00:00
---
## The discriminator (measured, both venues)

Occupancy cannot kill the fabricated text arcs (0123): a fully-rendered disclaimer ROW is one
CONTINUOUS strip (occ 0.8-1.0), overlapping R08's gappy bend rails (~0.6). The physical
discriminator is support COMPOSITION about the fitted curve: rail support = THIN curve-aligned
components (dashes: elongation ext/perp-thickness 5-300, where thickness = p90-p10 of
dist-to-curve), text/field support = compact fat blobs (el 1-4). Measured: scattered-glyph arc
comps max el 4.0; real dashed-rail comps from 4.4 (`RAILGRADE_EL_MIN = 4.0`); px-weighted
rail-grade fraction: text arcs 0.00-0.15 vs real rails >= 0.6 dilated (`RAILGRADE_FRAC_MIN
= 0.40`).

## What landed (aerial_rail.py) — layered per PHILOSOPHY 7b, each seam judged independently

1. **Acquisition gate** (`_enumerate_curves`): a seed whose gathered support fails the dilated
   composition gate is never fit -- the 907-922 px text arcs can no longer win the primary slot.
2. **Emit gate** (`track_sequence`): the FINAL support of each curve is judged again (a
   temporally-seeded thread can drift onto the overlay); a dropped curve never seeds the next
   frame.
3. **Tracklet-grade filter** (`thread_rails`): per-tracklet MEDIAN of the UNDILATED member grade
   (undilated keeps English words unmerged, where the gap is widest). Evidence-scaled: pooled
   conviction (med < `RAILGRADE_TRACKLET_MIN` 0.75) needs >= `RAILGRADE_POOL_N` 4 members (every
   real pooled tracklet measures >= 0.96; the disclaimer-row tracklet 0.40, fence/text hybrid
   0.68); a shorter tracklet is convicted only by EXTREME composition (< `RAILGRADE_EXTREME_MIN`
   0.45 -- scattered-glyph minis 0.12 vs the worst real fragmented-rail mini R08 f818 0.59).
   Dead tracklets are wholly contained: no votes, no label inheritance, curves removed.
4. **Label containment** (`thread_rails` label pass): an unmatched non-tracklet curve with patchy
   support (occ < THREAD_OCC_MIN) is uncorroborated on every axis -> dropped. Real rotation
   fragments stay (R08 f2188-2244: unmatched but occ 0.75-1.0).

## What was tried and REVERTED — the tripwire earned its keep

A LOOP-1 line-level composition veto (clean_pixels CONT branch, undilated >= 0.75) killed the
text px at the source on R02A but broke R08: its fragmented real bend rails grade down to 0.24
at the line level (f818/f930/f1344), fully overlapping text rows (<= 0.55). The R08 tripwire
caught it -- 570 violated frames, f818 lost a rail outright. Reverted; LOOP 1 is unchanged this
round (text-px leak documented in-code), containment lives downstream where evidence pools.
Lesson banked in-code: per-frame the classes overlap; per-tracklet they separate.

## Validation

- **R08 tripwire: PASSES** (within frozen tolerances) with all four layers live; suite 193
  passed; ruff clean.
- **R02A re-render + 6-batch red-team fan-out (48 frames, every [hi] re-verified by my own
  eyes)**: **naming polarity is CORRECT in all 48 frames** -- green is field-adjacent everywhere
  it exists, no near/far swap anywhere. Long zone 8/8 PASS, band zone 8/8 PASS (merged-band
  convention held), start/tail naming correct. **The text-arc steal class is GONE**: the f556-574
  row, the 907-922 px arcs, the f1650/f2160 phantoms, and the f656-674 turn alternation no longer
  exist; no curve through text carries naming weight (residual sliver votes w <= 0.001).
- Naming now decided only by live-tracklet evidence: dead = {disclaimer row f556-574,
  glyph mini f2166, fence/text hybrid f2410-2424}; all surviving tracklets grade >= 0.86.

## Residue (documented, bounded — every item eyeballed; none influences naming)

- **2 single-frame text-sliver curves remain**: f2168 (Chinese row, matched sliver: occ 0.90,
  grade 0.56) and f2386 (English row, T32: occ 0.75, grade 0.67) -- locally indistinguishable
  from the worst real fragmented-rail frame (R08 f818: occ 0.90, grade 0.59); the numbers prove
  no local threshold separates them. LOOP-1 appearance work closes them.
- **f348**: single-frame blue arc bowing ~25 px through grass between rail-end touchdowns,
  supported by thin wall-edge fragments (grade 0.96 -- genuinely rail-like locally). Same
  furniture-arc family; vote weight 0.000.
- **End-geometry drift of REAL-rail fits through overlay regions** (rotation frames): f640 green
  end peels ~40-70 px through the English rows (blue stays true on the rail); f674 blue end
  crosses the odometer box; f628-f646 ends overrun the legend boundary (f646 over silk #2).
  Known end classes (0122/0123) fed by the LOOP-1 text/speckle leak.
- **Missing second curve on a visible fixed rail** (f340 start, f616 ~150 px truncation,
  f692/f710 turn, f2200/f2410-f2440 tail): under-acquisition/retention -- honest misses, no
  fabrication; Phase-2 fill + LOOP-1 territory.
- **Un-named rotation fragments default to fixed** (f680/f686: real rail, occ 1.0, no tracklet
  within LABEL_TOL during fast rotation): honest no-evidence default, reads as a 2-frame G->B
  flicker; a tri-state render color would stop it reading as a naming claim.
- **Clean-px speckle on text** (cyan dots, no curve): LOOP-1 leak via the occ >= OCC_HI admit of
  continuous English rows over the dark stands; killing it needs APPEARANCE work in LOOP 1
  (turf-adjacency / glyph texture) under the full re-render + re-audit + re-freeze protocol.

## Follow-up (owner manual review, same day): the END-HOOK class -> `_trim_garbage_ends`

Owner surfaced f640/f666/f668 (green's end bends off the rail through the disclaimer rows) and
f708 (minor end cross onto the fixed rail). Anatomy (measured per 10%-arclen end window): the
hook = a small GLYPH CLUMP at the tip (16-45 px, dilated grade 0.00) which "supports" the bent
deg-3 end -- `_trim_to_support` cannot clip it, and the hook's unsupported bridge becomes a
protected interior gap (f640: tip grade 0.00 + two EMPTY windows + healthy rail). The global
pass could not prevent this because it judges curves as UNITS (existence/identity/naming) and
never inspects intra-curve geometry; an 86%-healthy curve passes every aggregate gate.

Fix: **poisoned-tip amputation** = the CONJUNCTION of garbage tip composition (dilated grade <
RAILGRADE_FRAC_MIN) AND tip bend >= `ENDHOOK_ANG` 12 deg off the inboard tangent (hooks measure
15-27 deg). Either alone is honest and spared: R08's occlusion-shredded gold tips are
garbage-graded but STRAIGHT (f818/f820/f2196 <= 8 deg) and true bend ends are bent but
healthy-graded (f640 partner end: 1.0 @ 18 deg). After the cut, _trim_to_support clips the
orphaned bridge. **Tripwire PASSES untouched; f640/f666/f668 verified fixed on the re-render.**

f708 is NOT this class: its crossed tip is supported by REAL rail px from the wrong rail
(grade 1.0) -- the smooth-stray family (`_decouple_pair`/Phase-2 corridor geometry), documented.
Note: each fix RESHUFFLES which isolated overlay stubs survive (the f2166/f2410-2424 slivers
moved, n=4 tracklet became n=2 and evades the pool rule) -- third independent confirmation that
single-shot overlay stubs are locally indistinguishable from honest fragments; stop chasing them
with local rules.

## Next

1. Re-audit scale-out (R01/R03/R06/AWT) under the gated build.
2. LOOP-1 appearance pass for the text-px leak (above) with re-render + re-audit + re-freeze —
   this is what actually closes the isolated overlay stubs AND the end-hook source.
3. Phase 2 (odometer-anchored smooth-in-s spanning/fill): mid-gap band float, end under-coverage,
   LOOP-1 dropouts (f564/f956), the f708 pair-corridor stray.
