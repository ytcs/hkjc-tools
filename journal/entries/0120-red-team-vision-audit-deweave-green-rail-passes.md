---
title: Red-team vision audit validated; config-A bend weave fixed; GREEN rail passes both venues
id: 0120-red-team-vision-audit-deweave-green-rail-passes
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "red-team", "audit", "deweave", "r02a", "r08"]
refs: {"refs": "0119", "task": 104}
created: 2026-06-10T03:08:30+00:00
updated: 2026-06-10T03:08:30+00:00
---
## Red-team vision audit (fan-out) — validated as the autonomous CV gate; fixed the bend weave

Followed 0119. Built + exercised the **red-team vision audit**: fan out ~6 `general-purpose` subagents in
parallel, each a small frame-batch spanning one race phase, each using NATIVE VISION (self-cropping with a
NEAREST-upscale helper) to judge the rail overlays. Aggregate → VERIFY each checkable claim by eye → fix →
re-render → re-audit. Documented as the standing technique in CV_RECON §0b. Outcome: the method WORKS — it
caught a real defect a single-reviewer 5-frame spot-check had missed, and (after a prompt fix) gives a
trustworthy GREEN-rail census.

### What it caught (and the verify step that mattered)
- **Config-A bend WEAVE (real, GREEN-critical), R02A f690-840.** Three independent batches flagged the two
  threads crossing. Verified at f840: the RUNNING thread had absorbed BOTH close rails into one support
  blob (perp residual-about-fit ~15-20 px vs ~2 px for a clean rail), so its deg-2 fit zig-zagged between
  them — GREEN jumped rails mid-frame. I had previously (wrongly) closed this as an overlay-fill artifact
  by only re-checking f694-710.
- **BLUE/outer hallucination (real, deprioritised).** Outer thread fits thin-continuous HUD edges that
  pixtag cannot distinguish from a rail: the disclaimer text-box TOP edge (f2296/f2366) and oblique-finish
  crossings (f826/f2788). Owner steer: the GREEN racing rail is what downstream uses; BLUE is secondary.
- **False positives (calibration).** Agents over-flagged BLUE "floating through the bottom legend" on R08
  f896/f966/f1036/f1106 — verified FALSE: a real foreground rail runs UNDER the silk-legend HUD. And the
  near/far "swap" calls rested on a heuristic I had mistakenly given them ("racing = rail nearest the
  field") — which is the very trap 0118 falsified (the bunch runs wide). Lesson: trust the agents'
  literal-geometry calls (weave, float, on-text); re-derive their interpretive (near/far) calls.

### Fix — coupled DE-WEAVE (`_decouple_pair`, field-free, non-circular)
When the running support's residual about its OWN fit exceeds ~8 px (`WEAVE_FAC`; one deg-2 cannot follow a
two-rail blob, but a single rail hugs its fit to ~1-4 px even when sharply curved — a curvature-INVARIANT
gate), repartition the UNION of the pair's pixels by NEAREST curve and refit each thread exclusively,
iterating. Two curves cannot own the same pixel, so they relax onto the two separate rails. Removed the old
circle back-projection fill from the overlay driver in the same arc (it extrapolated short slivers into
floating lines on curving rails; the f694-710 "weave" was that fill, not the library).

### Result — GREEN rail audit PASSES both venues (de-weave gate)
- Re-audit (GREEN-only, corrected prompt: judge weave/float/on-artifact, IGNORE near/far): **24/24 frames
  OK** across R02A (incl. the f690-860 weave zone) and R08 (incl. the rotating bend f756-1106 and start/
  finish). Final running residual ~2 px R02A / ~1 px R08.
- R08 NO REGRESSION: the gate is curvature-invariant so crisp separated rails never trigger; de-weave is
  idempotent on already-separated pairs (f2158/f616/f768/f860 visually unchanged).

### Open / handoff
- **Near/FAR identity is signal-ambiguous field-free.** Whether GREEN sits on the RACING vs the FIXED rail
  is decided globally by the radius vote (`_assign_racing_by_radius`); per-frame radius is ~32% noisy (387/
  1222 R08 frames flag "running narrower"), and field-proximity is a known trap. GREEN is verified on a
  CLEAN rail everywhere; whether it is always the correct one of the two needs a better discriminator (or
  the owner's eye). Routed to the owner.
- BLUE/far-rail HUD-edge hallucination (f2296/f2366) + oblique crossings (f826/f2788): deprioritised; the
  Phase-2 global smooth-separation coupling (reject an outer whose offset from the running rail jumps off
  the smooth-in-frame trend) is the principled fix.
