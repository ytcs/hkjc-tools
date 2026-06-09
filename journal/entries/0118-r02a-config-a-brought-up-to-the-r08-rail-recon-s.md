---
title: R02A (config-A) brought up to the R08 rail-recon standard (two first-principles fixes)
id: 0118-r02a-config-a-brought-up-to-the-r08-rail-recon-s
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "rail", "r02a", "config-a", "validation", "handoff"]
refs: {"refs": "0117", "task": 104}
created: 2026-06-09T09:37:47+00:00
updated: 2026-06-09T09:37:47+00:00
---
## R02A (config-A) brought up to the R08 standard — two first-principles fixes

Follow-up to 0117 (library promotion). Goal: make the config-A race R02A (HV course A, 30.5 m, the
WIDEST rail position; slightly different render style) reconstruct as well as the R08 gold standard,
driven by full-res VISUAL inspection on a dense frame subset (metrics for triage only). Done on
branch `cv-aerial-r02a-recon`. Two distinct failure modes were found by looking at the pixels +
clean-pixel-cloud analysis, and fixed at the root in the library (hkjc.vision.feeds.aerial_rail):

### Mode 1 — running/outer LABEL inversion (the dominant failure, back-straight block f1158-1882)
On the wide zoomed-out config-A view BOTH rails sit far apart. The running-rail label TEMPORALLY
LOCKED onto the FAR (upper) rail while the field demonstrably hugs the NEAR (lower) rail (clean-pixel
clouds: far rail y~300, near rail y~600 with 4200+ px, field at y~620, yet running was assigned to
y~300). The fit geometry was fine -- only the inner/outer LABEL was inverted, carried by the temporal
seed through ~360 frames.
FIX: re-assert RUNNING = the rail the field hugs every frame (the field always races along the inner
rail), with hysteresis (swap only if the partner is clearly <0.7x nearer the field). Temporal seeding
still stabilises the rail GEOMETRY; it no longer locks the LABEL. No-op on R08 (running already
field-nearest). -> running-far-from-field frames ~419 -> 201.

### Mode 2 — striped near rail DROPPED on sharp bends (knife-edge spread gate)
On config-A bends the striped near rail was dropped by clean_pixels. Root cause: the adaptive spread
threshold takes best = the THINNEST spanning line, but a thin FAR line (grandstand/far rail, spread
~2.6-2.8) set best low -> thr ~4.6-4.8, knife-edging the real striped near rail (spread 4.8, cov 2.5)
just over the boundary -> kept on f1760, dropped on f1700/f1800 (144 of 4500 rail px) by 0.01 px.
FIX (first-principles): SOLID COVERAGE (cov >= COV_SOLID) is the unambiguous rail signature -- a
dense-support spanning line IS a rail regardless of the thin-line-driven thr, so keep it up to the
2D-block ceiling. Purely ADDITIVE (text cov<2 stays out), so R08 cannot lose rails. Plus _gather_curve:
for the curving high-spread rails, iterate fit->gather with a spread-scaled band so the deg-2 parabola
WALKS along the sharp arc and recovers the curved ends a single PCA fit clips (R08's thin rails never
enter this branch). -> recovered the near rail across the whole bend/back-straight block.

### Result (VISUAL gate, full-res)
  * R02A running-rail-far-from-field frames: 439 -> 2 (f2098 borderline). Zero-rail 20 -> 1 (f2060).
    Rails/frame now 0:1, 1:630, 2:495 of 1126 (single-rail frames are the genuine config-A reality:
    the outer rail is often out of the visible band, journal 0109).
  * Visually confirmed correct across EVERY phase: start (f360+ top-down), early bends (f520-700),
    the back-straight block (f1000-1882, the old catastrophe), late bends (f1900-2300), run-in +
    finish (f2400-2638). The running rail now follows the field's rail everywhere.
  * R08 NO REGRESSION: 0-rail 0 / 2-rail 1226 / f768=2 / f2158=2 (216 clean frames gained pixels from
    the additive solid-coverage rule but the fit is unchanged at the rail level; f768/f2158 BEND
    overlays visually re-confirmed concentric edge-to-edge). Both fixes are no-ops / additive on R08.

### Open / handoff
Minor residual: f2060 (rail present on a bend but unfit), f2098 (borderline). The config-A "outer"
is sometimes the grandstand/hedge boundary rather than a true outer rail (the running rail -- the one
that matters for recon -- is correct); this is the 0109 sparse-outer caveat, not a regression. Handed
to the human operator for the gating manual review. Next: merge to main after review; then wire the
promoted perception into the 4-feed fusion + ablate the legacy select_running_rail.
