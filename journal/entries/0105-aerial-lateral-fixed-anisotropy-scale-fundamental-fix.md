---
title: Aerial recon LATERAL fixed — robust near-constant-zoom s_lat track (fundamental fix); longitudinal is next
id: 0105-aerial-lateral-fixed-anisotropy-scale-fundamental-fix
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "reconstruction", "scale", "handoff"]
refs: {"updates": "0104", "task": 101}
created: 2026-06-08T07:30:00+00:00
updated: 2026-06-08T07:30:00+00:00
---
## Aerial recon LATERAL is fixed — the fundamental scale fix (no EKF needed); longitudinal is the next axis

### The fix (simpler than the 0104 EKF plan)
0104 blamed the residual lateral noise on the unreliable OUTER rail (shadow/marker mis-detected as
the true 22.5 m outer) injecting a wrong scale for whole stretches that no batch smoothing could
reject (systematic, not zero-mean). The EKF/affine-pose plan was overkill. The render's virtual
camera holds a **near-CONSTANT zoom**, so the lateral pixels-per-metre `s_lat = sep_px/22.5` is a
**flat, slowly-varying** curve. So:

1. **Global robust gate** on the two-rail separation against the robust zoom LEVEL (median±MAD): kills
   the wrong-outer spikes in BOTH directions (fake outer too NEAR -> s_lat too small; far background
   line too FAR -> too large) — and crucially this works even in stretches where MOST outers are
   wrong, which a local rolling median cannot (it chases the regional bias — that was the 0104 bug).
2. **Gentle local median** among survivors for the slow drift, then **interp through the rejected
   gaps**. The gated `s_lat(f)` track comes out essentially FLAT (~6.5 px/m) across the whole race.
3. **On-track gate**: with a validated rail+scale, drop any detection beyond the 22.5 m running
   surface (the detector occasionally grabs apron/grandstand blobs at 30-40 m — these were the
   remaining local median spikes at f765-790).

Result (HV 2026-06-03 R08): `s_lat` track flat, **systematic f1700-2150 ramp GONE**, field-median
y_lat sits stably ~8-10 m across the whole race; 492/1227 two-rail measurements rejected as
wrong-outer (40% — confirms how bad the per-frame outer was). Lateral perp tracks (top-3) are now
smooth. Validated in `studies/cv_pose_backbone.py` (now a lateral-scale study with a PASS1 detection
cache `data_cache/cv_pose_pass1.npz` so the gate/recon iterate instantly).

### Ported into the main tracker
`studies/cv_aerial_track_id.py`: new `_robust_zoom_track()` replaces the old `_medfilt(sep_raw, w=15)`
(local median that could not reject the systematic wrong-outer stretches). Lateral ylat in the recon
is now physical everywhere.

### What's left — the LONGITUDINAL axis (now the dominant noise)
With lateral fixed, the recon's noise is now in the per-horse LONGITUDINAL offset (cv_recon_res.png
bottom panel: step discontinuities; cv_recon_2d.png middle/right still spaghetti). Root causes:
- `project_to_polyline` returns `s_px` as cumulative arc-length from the rail polyline's vertex 0,
  whose ORIGIN moves frame-to-frame as the detected rail span shifts. `(s_head - s)` cancels the
  common origin in principle, but `s_head` itself CLAMPS/jumps when the leader projects past the
  rail's visible end (AUDIT: `s_px=0.0`, offset -130 m spikes at f840/f1100s).
- The longitudinal offset reuses the LATERAL `scale_f` — anisotropy makes along-track px->m differ
  from lateral px->m, so the offset metric is the wrong ruler.
- NOTE: part of the downward drift in the offset panel is REAL race dynamics (backmarkers falling
  behind the leader), not noise — don't over-smooth it away.

### Next step
Fix the longitudinal: (a) a consistent arc-length origin (project onto the rail with a stable origin,
or anchor s to the odometer-tracked head and measure gaps robustly, gating the clamped-endpoint
frames out); (b) an along-track scale distinct from the lateral one (the odometer rate gives
metres/frame for the head -> calibrate along-track px->m directly). Absolute arclen (odometer-anchored,
left panel) is already a clean monotonic diagonal — it's the per-horse RELATIVE offset that needs the
robust along-track ruler.
