---
title: Aerial recon LONGITUDINAL — rail-clamp gates + PASS1 caches -> coherent 2D paths (race dynamics emerge)
id: 0106-aerial-longitudinal-clamp-gates-coherent-2d-paths
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "reconstruction", "longitudinal", "handoff"]
refs: {"updates": "0105", "task": 101}
created: 2026-06-08T08:30:00+00:00
updated: 2026-06-08T08:30:00+00:00
---
## Aerial recon LONGITUDINAL — rail-clamp gates make the 2D paths coherent (race dynamics emerge)

With the lateral fixed (0105), the recon noise moved to the per-horse LONGITUDINAL offset. Root cause
(quantified from the cache): the inner-rail polyline's arc-length L_px swings 142->959 px (the detected
span collapses or jumps up to 729 px frame-to-frame), and project_to_polyline's s_px CLAMPS to its
ends — the leader is front-clamped in 18% of frames, back-clamped in 11%, and s_head (p96 of s) jumps
>5 m in 10% of frames, >38 m in 1% (a different horse becomes the p96 reference). These corrupt the
gap (s_head - s) that sets every horse's position-behind-head.

### Fix (studies/cv_aerial_track_id.py)
- **LONGITUDINAL trust gates** in `_recon`: drop the offset when the rail is too short to span the
  field (`L_px < 0.6*median`), the leader is front-clamped (`s_head > L - 25px`), or this point clamps
  to either end (`s < 3` or `s > L - 25px`). Failures -> NaN gaps (honest), not fabricated positions.
- **Per-track smoothing**: offset medfilt w=9 (slow), ylat medfilt w=7 (de-spike perp outliers).
- **Plot only SUBSTANTIAL tracklets** (len >= 60): the ~11 runners are 550-1300-frame tracks; the
  ~156 short fragments are brief spurious detections that only cluttered the paths. NO merging — the
  user warned against aggressive track merging (mis-ID risk).
- **PASS1 detection caches** (`data_cache/cv_trackid_pass1.npz`, `cv_pose_pass1.npz`): the per-frame
  detect+rail decode (~12 min) is cached so the association/recon iterate instantly. `OVERLAYS=0`
  skips the overlay re-decode while iterating.

### Result (HV 2026-06-03 R08, cv_recon_2d.png)
- Absolute arclen (odometer-anchored): clean monotonic diagonal.
- Position-behind-head: **coherent race dynamics** — horses smoothly fan from the leader (0) to ~-40 m
  back, evolving over the race. A few residual step segments remain.
- 2D path (y_lat vs arclen): reconstructed, y_lat mostly 5-15 m.

### Remaining (not yet "perfect")
- FRAGMENTATION: 11 horses -> ~79 len>=60 tracklets (re-acquired after GRACE gaps). Honest without
  merging; a conservative same-lane/same-arclen re-link with a strict gate could consolidate without
  mis-ID, but deferred (the user's caution).
- The longitudinal offset reuses the LATERAL scale (anisotropy: along-track px->m differs). A global
  anisotropy ratio k* (along/lateral, ~const for a fixed-tilt render; measurable on bends where the
  arc pins the along scale) would correct it — a scale stretch, not a noise source, so lower priority.
- A few residual offset steps where s_head's p96 reference still shifts inside the gates.

### Validation TODO (deferred, user notes)
- LEFT-handed track (some races run the other way) — confirm the running-side sign + bend handling.
- A-config single-rail race (HV 2026-05-20 R02) — no outer rail -> s_lat falls back to constant; check.
