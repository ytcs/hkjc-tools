---
title: Aerial recon GENERALISES to R03 (1200 m, 12 runners) — green-box odometer locator + parameterised tracker
id: 0108-aerial-recon-generalises-R03-green-box-odometer
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "reconstruction", "validation", "handoff"]
refs: {"updates": "0107", "task": 101}
created: 2026-06-08T10:30:00+00:00
updated: 2026-06-08T10:30:00+00:00
---
## Aerial recon generalises to a second race (item 3) — no per-race tuning

### Two generalisation blockers fixed
1. **Odometer locator was R08-specific.** `aerial_layout.find_layout` hard-codes the odo ROI at a fixed
   fraction (y 0.27-0.34, x 0.83-0.92). R03's "###M" sits in a DIFFERENT place (inside the green
   minimap box on its left, vs to the right of the box in R08) -> 0 reads. Fix (`studies/cv_build_obs.py`
   `_odo_crop`): DETECT the green minimap box (largest green CC in the top-right) and OCR the whole box
   — read_odometer finds the digit+M pattern wherever it sits. R03: **1275/1515 reads (84%)**, arclen
   150-1190 m of 1200 — same recall as R08. (cv_build_obs.py also replaces the retired-mode_b
   cv_joint_aerial.py for the frames+odo subset the tracker needs.)
2. **Tracker was R08-hardcoded.** `cv_aerial_track_id.py` now has a `RACES` config table (video, dist,
   field, course) selected by `RACE` env; per-race PASS1 cache + diag suffix.

### R03 result (HV "C", 1200 m SPRINT, 12 runners)
- In-lane detection found **median 12 blobs/frame** (R03's exact field size; 635 frames at exactly 12)
  — the count-aware detector adapts to the field with no change.
- Re-link: 27 joins -> top-12 tracks span 477-949 frames.
- Absolute arclen: clean monotonic 150->1200 m. Position-behind-head: coherent dynamics (0 to -35 m).
  2D paths reconstruct; one noisy stretch ~arclen 800-1000 m (a detection-hard turn), rest clean.

So the lateral fix + longitudinal gates + re-link + odometer all transfer to a different
distance/field/race-type on the same HV "C" geometry with only the config-row input.

### Remaining validation (deferred, needs sources)
- R06 (13 runners) / R01 (1650 m) are now one-liners (`RACE=R06`/`R01`) — run when desired.
- A-config SINGLE-RAIL race (aerial_20260520_02): the algorithmically distinct test (no outer rail ->
  s_lat falls back to the constant zoom). Needs its official meta (distance/course/field) fetched.
- LEFT-handed track: needs a left-handed source video (HK turf is right-handed; the running-side sign
  `_ry_side` should auto-handle it, but unverified).
