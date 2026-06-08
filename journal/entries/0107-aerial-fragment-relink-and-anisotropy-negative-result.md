---
title: Aerial recon — conservative fragment re-link consolidates the 11 horses; longitudinal anisotropy is NOT recoverable
id: 0107-aerial-fragment-relink-and-anisotropy-negative-result
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "reconstruction", "tracking", "handoff"]
refs: {"updates": "0106", "task": 101}
created: 2026-06-08T09:30:00+00:00
updated: 2026-06-08T09:30:00+00:00
---
## Aerial recon — fragment re-link (item 1, DONE) + anisotropy along-scale negative result (item 2)

### Item 1 — conservative fragment RE-LINK (consolidates the 11 horses) — DONE
A horse whose detection drops out for > GRACE (24 frames) restarted as a new tid -> 167 tracklets for
11 horses. New post-PASS2 step in `cv_aerial_track_id.py`: re-join fragment B after A when their
ODOMETER-PREDICTED metric trajectories are continuous across the gap. Predict A forward by the
odometer delta (`A.end_arclen + (odo(B.start) - odo(A.end))`) and gate on |Δarclen| <= 6 m AND
|Δy_lat| <= 3 m, with one-successor/one-predecessor greedy-by-cost assignment (union-find). The
LATERAL gate (3 m — horses hold their lane) is the strong, safe discriminator; the odometer makes
long holes (up to 250 frames ~10 s) linkable. NO blind merging — strict + unambiguous, per the user's
mis-ID caution. Result: 52 joins -> top-11 tracks now span **953-1802 frames** (longest = 83 % of the
race, up from 552-1299). Race dynamics (position behind leader) are now long and continuous.

### Item 2 — longitudinal ANISOTROPY along-scale is NOT reliably recoverable (negative result)
The longitudinal offset reuses the LATERAL px->m scale; under the anisotropic top-down render the
along-track scale differs by a ratio k* = s_along/s_lat. Tried to measure k* two ways on bend frames
(where curvature should pin the along scale), both FAILED:
- **circle-fit radius_px / R**: k* swung 0.39-1.50 and got WORSE as the arc-fit improved. Cause: a
  track-plane CIRCLE projects to an ELLIPSE under anisotropy, so circle-fitting the radius is
  confounded by the very anisotropy being measured.
- **affine-ICP |col0|/|col1|**: even worse (p50 6.97, p90 ~20). The along-track direction is
  DEGENERATE — ICP nearest-point matching lets correspondences slide along the smooth rail, so the
  along scale is unobservable from the rail alone.
Root reason: the virtual camera TRACKS the field, so the leader is ~pixel-stationary -> the odometer
(our one absolute along ruler) gives no pixel-motion handle either. The along-track metric scale is
fundamentally weakly-observable from the rail + odometer. DECISION: keep the isotropic offset
(k*=1) — qualitatively correct race dynamics in "lateral-equivalent metres", with a known bounded
scale uncertainty — rather than apply a bogus, unstable k*. Proper fix (deferred, needs a new
fiducial): detect painted furlong/distance poles for an absolute along ruler, or calibrate the
end-point gaps from official race-result margins.

### Item 3 — generalisation validation: IN PROGRESS
Building a minimal odometer obs (frames+odo via the promoted `aerial_ocr.read_odometer`, since the old
`cv_joint_aerial.py` builder imports the retired mode_b) for R03 (HV "C", 1200 m sprint, 12-13
runners) to test field-size + distance generalisation on the same geometry. The cached A-config
single-rail race (aerial_20260520_02) lacks documented config; left-handed needs a left-handed source.
