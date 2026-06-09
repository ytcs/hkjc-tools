---
title: Pixtag continuity gate (thin-OR-continuous) kills the R02A green-on-artifacts class
id: 0119-pixtag-continuity-gate-kills-green-on-artifacts
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "r02a", "pixtag", "continuity", "handoff"]
refs: {"refs": "0118", "task": 104}
created: 2026-06-09T23:53:59+00:00
updated: 2026-06-09T23:53:59+00:00
---
## Pixtag continuity gate — a rail is THIN and/or CONTINUOUS (replaces the coverage/adaptive-thr keep)

Phase 1 of the next-wave plan (CV_RECON §10). Goal: reduce mis-tagged pixels in `clean_pixels`
(LOOP 1) WITHOUT regression, to fix the **green-on-artifacts** pathology (the racing thread latched
onto the disclaimer-text band / grandstand-base edge / odometer minimap — R02A f340, f2052, f2070).
Driven by full-res VISUAL inspection; metrics for triage only. Branch `cv-aerial-r02a-recon`.

### What the pixels showed (visual, owner-confirmed)
At f340 the AZURE rail (turf level, beside the horses) is the correct racing rail; the GREEN thread
was fit to the bottom edge of the translucent grandstand/disclaimer band ~80 px higher — NOT a rail.
At f2052/f2070 the green shot a spurious diagonal into the HUD text. Root = PIXTAG: disclaimer-text
glyphs + the grandstand-base STEP edge survive `clean_pixels` and the fit acquires them.

### The line-descriptor probe (why the old gate leaked)
For every spanning line on the pathology frames + the R08 hard frames I measured spread, coverage,
and a new **occupancy** = fraction of along-major-axis bins (nb=20) holding support. Two genuine rail
classes emerged, cleanly separable from artifacts by ONE rule:
  * **CRISP rails** (R08 everywhere, R02A straights): spread ≤ 1.24 px. At bends they go GAPPY
    (R08 f734 occ 0.67, f2158 occ 0.58) but stay razor-thin (spread ~0.7). Thinness alone certifies.
  * **NOISY/ARCING rails** (R02A bend arcs): thick (spread 4–5) but FULLY CONTINUOUS (occ ≥ 0.83,
    usually 1.0).
  * **Artifacts** (text / dense blobs / minimap): spread 1.6–6.5 AND occ ≤ 0.75 (text scatters into
    word/glyph gaps); fat field/grandstand blobs are occ-high but spread > ceiling.
The OLD gate (`thin_ok` adaptive-thr OR `solid_ok` cov≥2) leaked both ways: thin text passed the
thin test, and worse — a thin TEXT line set the adaptive `thr` low and knife-edged the real rail out
(f340 rail spread 4.58 > thr 4.0). [Note: this retires the COV_SOLID/adaptive-thr keep added in 0118.]

### FIX (first-principles, in `aerial_rail.clean_pixels`)
Keep a spanning line iff **`spread ≤ S_THIN` (unmistakably thin) OR `occ ≥ OCC_HI` (one continuous
curve)**, capped at the 2D-block ceiling `s_ceil`. `S_THIN_FAC = 0.00126·W` (~1.2 px, set above R08's
crisp rails ≤0.96 and below R02A thin-text ~1.6); `OCC_HI = 0.80`. Continuity is the owner's stated
discriminator (rail ~0.96–1.0 vs text ~0.3–0.5); thinness is the safety net that rescues the gappy
bend rails a flat occupancy floor would wrongly drop. `_occupancy()` is computed in `enumerate_lines`
beside spread/coverage (no module sprawl).

### Result (VISUAL gate, full-res; counts for triage)
  * **R02A green-on-artifacts RESOLVED** — f340/f2052/f2070 now show only the correct racing rail; the
    text/grandstand/minimap band is clean (dropped). Verified on the pixtag overlays (text → red) and
    the fit overlays (no HUD-band curve). Broad sweep (f590/840/1136/1386/1638/2138/2388/2638) clean,
    no new pathology; two-rail config-A straights still resolve the close pair.
  * **R02A zero-rail 22 → 0** (every render-active frame now has the racing rail). Rails/frame
    0:0 / 1:651 / 2:475 (singles are the genuine config-A reality — far rail coincident/out-of-band).
  * **R08 GOLD HELD** — f768/f2158/f734/f800/f860 all two-rail, visually re-confirmed concentric
    edge-to-edge. Zero-rail 0. Net frame-level change: 5 frames slipped two→one — f560/562/564 are a
    text-artifact REMOVAL (improvement; the old "second rail" was the disclaimer band), f816/f956 are a
    genuine GAPPY FAR rail dropped (occ in the gate gap) — left for Phase 2 fill, not chased with a
    gate bandaid (would re-admit text).

### Open / handoff (→ Phase 2)
Remaining pathologies are now squarely Phase-2 (global non-crossing smooth-curve second pass):
(1) rail-separation **weave/crossing** f694–710 (per-frame association, no global non-crossing
constraint — unchanged by pixtag); (2) R02A single-rail FILL where the far rail is faint/gappy
(f816/f956 R08, many R02A singles) — the far rail is smooth in odometer-s and present in neighbours,
so a global s-coupled model recovers it. NO circles, NO field. Image-space first.
