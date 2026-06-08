---
title: Aerial recon validation — A-config single-rail (R02A) runs but degrades; SEP_MIN gate made relative
id: 0109-aerial-recon-A-config-validation-relative-gate
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "reconstruction", "validation", "handoff"]
refs: {"updates": "0108", "task": 101}
created: 2026-06-08T11:30:00+00:00
updated: 2026-06-08T11:30:00+00:00
---
## Aerial recon validation — A-config single-rail race (item 3 continued)

Ran the full pipeline on the cached A-config race (aerial_20260520_02, "R02A"; ~1500 m, 12 runners).
Determined its config by inspection: distance ~1500 m (odometer max), field 12 (detector median), and
crucially the OUTER rail is found in only **28 %** of frames -> the single-rail-dominant test.

### Findings
- **Odometer + field detection generalise.** Green-box odometer: 1335/2256 reads (59 %), arclen
  260-1490 of 1500; in-lane detection median 12 (its exact field size). Absolute arclen reconstructs
  as a clean monotonic diagonal -> the odometer anchor is robust even in A-config.
- **A generalisation BUG found + fixed: the validity gate was R08-scale-tuned.** `SEP_MIN = 60 px`
  (absolute) over-rejected R02A, whose rail separation is only ~61 px (more zoomed-out camera) ->
  whole arclen stretches (600-1100 m) blanked. Made it RELATIVE: `SEP_MIN = 0.45 * sep_global`. R08
  unchanged (0.45*147 = 66 ~ old 60); R02A recovers coverage.
- **Honest DEGRADATION on single-rail.** Even after the gate fix, R02A's per-horse LONGITUDINAL offset
  and LATERAL are markedly noisier than R03/R08. With only 28 % two-rail frames the s_lat calibration
  rests on fewer (and possibly nearer-feature-biased) separations, and the rail backbone is less
  stable -> the reported scale 0.369 m/px (vs 0.153 for R08) may be over-scaled if the sparse "outer"
  is a nearer line, not the true full-width rail. The method RUNS and the arclen is trustworthy, but
  the metric lateral/relative-longitudinal quality drops when two-rail coverage is sparse.

### Net validation status (item 3)
- R03 (1200 m, 12 runners, two-rail): CLEAN generalisation, no tuning. ✓
- R02A (1500 m, 12 runners, single-rail-dominant): RUNS; arclen robust; lateral/longitudinal degrade
  with sparse two-rail; surfaced + fixed the R08-tuned SEP_MIN gate. ✓ (with caveat)
- R06 (13) / R01 (1650 m): one-liners (`RACE=R06`/`R01`), not yet run.
- LEFT-handed: still needs a left-handed source video. `_ry_side` (running-side sign from the field's
  median perp) SHOULD auto-handle it, but unverified.

### Deferred improvement for sparse-two-rail (A-config)
The scale bias under sparse two-rail is the same family as the 0107 anisotropy problem: the metric
ruler is weakly observed. A robust cross-race zoom prior (the AVV product's zoom is similar across
races -> a shared px/m-at-the-rail prior) or a detected absolute fiducial (furlong poles) would pin it.
