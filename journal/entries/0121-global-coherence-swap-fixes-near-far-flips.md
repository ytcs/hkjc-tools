---
title: Global coherence pass (couple_rails) — orientation-free swap fixes near/far flips; turn-drift remains
id: 0121-global-coherence-swap-fixes-near-far-flips
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "coherence", "phase2", "r02a", "r08"]
refs: {"refs": "0120", "task": 104}
created: 2026-06-10T06:01:37+00:00
updated: 2026-06-10T06:01:37+00:00
---
## Global coherence second pass — orientation-free near/far SWAP works; image-space repair does not

Owner dense-inspection surfaced a punch-list the sparse audit missed: vertical GLITCHES (R02A f642/f668),
residual WEAVES (f704-746/f1986), and WRONG-RAIL / near-far FLIPS (f1900-66, R08 f588). Built the Phase-2
global temporal-coherence pass `couple_rails` (runs in `track_sequence` after the racing-rail vote).

### What works — orientation-free swap (KEPT)
Fingerprint each frame's running rail by `COUPLE_N=7` points sampled along it (ordered by x-then-y, so it
is ORIENTATION-FREE — the earlier y-at-fixed-x sampling went NaN on a vertical rail and could not see it).
Build a robust per-point reference (median over a +/-20-frame window, 2-pass outlier-trimmed), and per frame
SWAP running<->outer when the outer fits the reference better. This relabel is low-risk and fixed the
near/far flip class: focused red-team re-audit confirms **R02A f1900/1920/1940/1966 and R08 f588 now OK**
(green on the correct rail, no crossover), f1986 + both broad samples (R02A+R08) clean, no regressions.

### What does NOT work — image-space curve REPAIR (REMOVED)
The first version also REBUILT a gross-outlier running curve from the reference. That re-hit the journal-0115
rotation trap: through a fast camera turn the rail sweeps near-VERTICAL while the +/-frame window is mostly
horizontal, so the median reference is a horizontal blur and the rebuild FABRICATED a floating horizontal
curve (f642 made worse). Removed the rebuild — an image-space frame-window reference is simply not valid
under camera rotation. The principled fix for the turn frames is an ODOMETER-ANCHORED / rotation-compensated
reference (CV_RECON Phase 2 "smooth in s"; journal 0115 R3) — a larger build, deferred.

### State (focused red-team audit, 24 frames, GREEN = downstream signal)
  * FIXED: near/far flips (f1900-66, f588), prior weaves (f826 earlier via de-weave; f1986). Broad R02A +
    R08 samples clean.
  * RESIDUAL (2 frames, [med], localized curve-end drift, NOT floats):
      - f642: green drifts into the grass/hedge margin at the early top-down TURN (rotation hard-spot).
      - f720: green's LEFT end over-curves and peels below the rail into infield turf (deg-2 end over-extension).

### Open / handoff
Turn-region drift (f642 class) needs the odometer-anchored coherence (the real Phase-2; owner to weigh the
build). f720-style end peel-off is a deg-2 over-curvature at the bend end (candidate: clamp the fit to its
pixel support / cap end curvature). Then resume scale-out (R01/R03/R06/AWT + left/right-hand configs).
