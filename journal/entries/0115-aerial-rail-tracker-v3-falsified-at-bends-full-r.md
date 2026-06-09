---
title: Aerial rail tracker v3 falsified at bends — full-res audit, root causes, first-principles redesign plan
id: 0115-aerial-rail-tracker-v3-falsified-at-bends-full-r
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "rail", "tracker", "falsification", "redesign", "handoff"]
refs: {"supersedes": ["0110", "0114"], "task": 104}
created: 2026-06-08T20:00:00+00:00
updated: 2026-06-08T20:00:00+00:00
---
## The aerial rail tracker (v3) does NOT solve the bends — full-res audit falsifies 0114

This supersedes **0114** (temporal orientation-free rail tracker — claimed "R08 whole race, both bends,
no collapse") and **0110** (field-consensus recon — claimed "R08+R03 root-solved + validated"). Both
over-stated: they were validated on a downscaled separation PLOT (`railT3/_tracking.png`), not the
full-resolution per-frame geometry. A frame-by-frame audit of `data_cache/diag/{pixtag,railrecon}/`
(the full-res dump from `studies/cv_rail_audit_dump.py`, which runs the SAME `cv_rail_track3.forward_track`)
shows the bend geometry is wrong where the plot looked smooth. This is the §0 "numbers lie by omission"
trap: `sep` (median distance between the two curves) stays plausible even when both curves are short
crossed stubs. The valid content of 0110/0114 is folded in below so nothing is lost.

### How this was verified (visual + quantitative + objective — not a spot check)
- **Objective whole-race span map** (ran `forward_track` over `_railprep_R08_330_2100`, measured each
  output curve's x-extent vs frame width W): one contiguous catastrophic region **f734–860** where the
  run-rail spans **0.19W** and the outer **0.16W** (i.e. ~16–19% of frame width — stubs). Everywhere
  else both span 0.82–0.99W. So the failure is sharply localised to BEND1.
- **Rotation probe** (rail principal-axis angle per frame): pre-bend straight swings 5.4° total
  (0.3°/frame, 0 px in-panel translation); **BEND1 swings 35.3° (up to 12.4°/frame) and the rail
  translates up to 201 px between consecutive frames.** The late bend (BEND2, f1950–2100) swings only
  3.3° — it barely rotates, yet still fails (different cause, below).
- **pixtag montage over f752–800**: candidate rail pixels trace BOTH rails edge-to-edge throughout the
  collapse. Pixels present, fit collapsed ⇒ the bend failure is ALGORITHMIC, not a pixel-tag deficit.

### Confirmed root causes
1. **BEND1 collapse = the K=5 sliding window's "rail is ~stationary in-panel" assumption breaks under
   camera ROTATION.** The window co-registers pixels across 5 frames; at 12°/frame + 201 px/frame the
   accumulated pixels are at different orientations/positions, and the `GATE=18 px` perpendicular
   assignment rejects the moved pixels → buffer starves below `MIN_PIX` → coast on stale state →
   collapse (f734–860). This is the core modelling error: v3 tracks the rail in IMAGE space and assumes
   temporal stationarity THERE, but the rails are static in the WORLD; what is smooth is the CAMERA
   POSE, not the image-space rail.
2. **Inner/outer FLIP (f768/f770)** — no persistent global identity. `run` is re-decided every frame by
   nearest-to-field-median; when the collapsed stubs sit by the field the label flips, and the two
   INDEPENDENT curves cross (nothing enforces concentricity).
3. **Curves never reach the frame edge — structural.** `fit_curve_pca` builds the polyline over
   `t∈[t_lo,t_hi]` = min/max projection of the ASSIGNED pixels only; no extrapolation to the content
   box. Bites whenever support is partial (occlusion, starvation).
4. **Outer-rail STARVATION (BEND2 / f2158)** — a separate failure: the outer rail runs low into the dark
   boundary + the bottom-legend exclusion band, so it is pixel-starved; the tracker fits a short stub
   from the few survivors (≥MIN_PIX) instead of falling back to the full-span parallel offset. Compounded
   by the legend horizontal CUT (`cv_rail_prep` `y1=_legend_top−8`) clipping real outer-rail pixels where
   the rail dips into the legend y-band (the f840 case), and by pixtag false positives.

### pixtag precision issues (per-pixel discriminator can't reject same-signature non-rail)
The tagger keys on brightness+whiteness+thinness per pixel, so it latches onto everything sharing that
signature: **horse silk patches (every frame), the disclaimer text box, the minimap track-outline, the
"FINISH" vertical overlay**, and it loses the real outer rail to the **bottom-legend horizontal cut**.
The unifying truth (matches the user's framing): a real rail pixel belongs to a LONG, CONNECTED,
ELONGATED, LOW-CURVATURE component spanning the frame; every false source is the opposite shape (compact
blob / short cluster / closed loop / short orthogonal segment). The right gate is component-level, not
per-pixel — and it replaces the region cuts (so the outer rail survives inside the legend band).

### Still-valid content carried forward from 0110/0114 (do NOT relitigate)
- **Orientation-free curve** (`fit_curve_pca`: minor=poly(major) in the PCA frame) is correct — a bend is
  low local curvature and never extrapolates to ±3000 px the way the old `y=f(x)` did. KEEP.
- **Pixel-tagging physics audit**: a morphological top-hat/ridge filter is a DEAD END (turf mowing
  stripes are thin bright ridges); brightness+whiteness+thinness+length is the right discriminator, and
  the geometric tracker should tolerate an imperfect tagger. KEEP — but precision must improve (above).
- **render_active dissolve gates** (0111, already promoted to `aerial_layout`) are unaffected and stand.
- **Field-consensus factorisation** (0110: replace one fragile per-frame reference with an N-horse
  consensus nuisance + one absolute anchor) is a sound IDEA; but its "validated at bends" claim is void
  because it rides on the (broken) rail scale at the bend. Re-validate after the rail redesign.

### First-principles redesign — experiments (robust, not bandaids)
Reframe: rails are STATIC rigid world structures (two concentric curves, fixed ~22.5 m separation, known
HV "C" bend radius, permanent inner/outer identity); the CAMERA POSE is the smooth thing. Track the smooth
thing, enforce the rigid geometry.
- **R1 — Coupled concentric two-rail model.** One centreline + signed half-separation: inner=c−h·n̂,
  outer=c+h·n̂. Concentric/non-crossing by construction; identity a carried sign. Success: zero crossings
  race-wide; smooth separation; f768/f770 flip gone.
- **R2 — Full-span fit with carried curvature prior.** Evaluate the polynomial across the whole content
  box, regularised toward the previous curvature. Success: run/outer span ≥0.9W on ≥99% of frames.
- **R3 — Motion-compensated prediction (kills the bend collapse).** Drop the stationary-window
  assumption: estimate the inter-frame rail warp (similarity/affine or sparse flow), warp the previous
  curve forward, assign+fit the CURRENT frame's pixels (K=1–2; pixtag already spans every frame).
  Success: f734–860 span ≥0.9W. **Extension/prize:** promote the warp to a tracked ground→image
  HOMOGRAPHY → delivers the §7 bend metric (absolute lateral metres) the whole revamp exists for (S9 drag).
- **R4 — Pixtag precision by component geometry.** Connected-component elongation+length+low-curvature
  gate (rejects silks/text/minimap/finish); mask avatars (0113 shape dets) + static HUD overlays at the
  source; REPLACE the bottom-legend horizontal cut with the component gate so the outer rail survives in
  the legend band. Success: outer recall through legend band restored (f840/f2158 outer ≥0.9W); ~0 false
  pixels on silks/text/HUD; no real-rail loss.
- **R5 — Generalise**: re-validate on R03 + R02A with the FULL-RES audit dump (never the separation plot).
  R02A (single visible rail) is R1's degenerate-mode test.

Sequence: R4 (clean inputs) → R1+R2 (coupled, spanning) → R3 (motion-comp / homography) → R5 (generalise),
each gated on the objective span/flip/separation metrics above — so we never again "validate" on a metric
that hides the geometry.

Artifacts: span/rotation probes `/tmp/probe_span.py` `/tmp/probe_rot.py` (logic to fold into a study);
audit overlays `data_cache/diag/{pixtag,railrecon}/` (R08, full-res, full race).
