---
title: Aerial rail reconstruction promoted to the library (byte-identical R08 port)
id: 0117-aerial-rail-reconstruction-promoted-to-the-libra
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: decision
status: open
thread: 
tags: ["cv", "aerial", "rail", "library", "promotion", "port", "handoff"]
refs: {"refs": "0116", "task": 104}
created: 2026-06-09T08:01:17+00:00
updated: 2026-06-09T08:01:17+00:00
---
## Rail reconstruction PROMOTED to the library (verified byte-identical on R08)

Follow-up to 0116. The 0115/0116 aerial rail-reconstruction redesign is now promoted from the study
scripts into the library, in ONE rail module (no module sprawl, per owner directive):
**hkjc.vision.feeds.aerial_rail** now carries the full pipeline alongside the retained primitives:
  * PRIMITIVES (kept; reused by aerial_detect + studies): line_candidates, fit_curve_pca, enumerate_lines, _gather_inliers.
  * RECONSTRUCTION (the current promoted path): clean_pixels (LOOP 1 long-smooth-curve gate),
    track_sequence / fit_rails (LOOP 2 temporal fit-gather), build_kappa / backproject_fixed_r / radius_at (global geometry).
  * LEGACY (demarcated, retained): select_running_rail — the per-frame field-proximity selector, superseded
    by track_sequence. Kept ONLY because the not-yet-ablated full aerial pipeline (cv_aerial_track_id.py,
    cv_pose_backbone.py) still rides on it. Slated for removal once the new perception is wired in (integrate-then-ablate).

The studies/cv_rail_exp_{pixtag,fit,global}.py are now thin I/O + diagnostic-overlay drivers over the library.

### Verification (faithful port, R08 = gold standard)
  * LOOP 1 cache BYTE-IDENTICAL to the pre-port baseline (0/1227 frames differ; total kept px 4,556,929 both).
  * LOOP 2 triage IDENTICAL: R08 0-rail 0 | 2-rail 1226 | total 1227 | f768 rails 2 (the protected bend).
  * VISUAL re-confirmation on full-res overlays: f768 (BEND1) + f2158 (BEND2) — running(green)+outer(blue)
    follow the concentric arc edge-to-edge. (Per CV discipline: visual is the gate; the counts are triage.)

### Hygiene (replace old, no two copies)
Deleted the now-dead duplicate rail scratch: cv_rail_track3.py + cv_rail_track4.py (the FALSIFIED v3 tracker,
0115), cv_rail_audit_dump.py (its audit), cv_rail_exp_coupled.py (the constant-offset dead-end), and
cv_aconfig_diag.py (a rail diagnostic on a stale select_running_rail signature). The journal is the durable trail.

Lint: added a per-file N803/N806 ignore for aerial_rail.py (CV kernel math notation: W=width, P/Q=point
arrays, R=radius, J=Jacobian), matching the existing harville/conditional_logit/generative precedent; the two
N802 functions were renamed snake_case (_cw, backproject_fixed_r).

### Open / next
Bring R02A (config-A single visible rail + a slightly different render style) up to the R08 standard via the
same experiment->visual-inspect loop, on a separate branch. R08 is the frozen gold standard (must not regress).
