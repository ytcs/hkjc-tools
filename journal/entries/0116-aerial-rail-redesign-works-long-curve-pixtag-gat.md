---
title: Aerial rail redesign works — long-curve pixtag gate + temporal fit-gather tracker (R08 validated)
id: 0116-aerial-rail-redesign-works-long-curve-pixtag-gat
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "rail", "tracker", "temporal", "pixtag", "handoff"]
refs: {"refs": "0115", "task": 104}
created: 2026-06-08T22:00:00+00:00
updated: 2026-06-09T08:00:00+00:00
---
## The 0115 redesign WORKS — pixtag long-curve gate + temporal fit-gather rail tracker (R08)

Follow-up to **0115** (which falsified v3 at the bends and laid out R1–R5). The redesign is built and
visually validated on R08 across every race phase. Two new study scripts (not yet promoted):
`studies/cv_rail_exp_pixtag.py` (LOOP1, pixel cleaning) and `studies/cv_rail_exp_fit.py` (LOOP2, rail fit).

### LOOP1 — pixtag precision by the LONG-SMOOTH-CURVE gate (R4)
The per-pixel brightness+whiteness+thinness tagger latches on silks/text/minimap/FINISH/legend-numbers.
Fix (the user's framing, no epicycles): a rail is ONE long, thin, low-curvature CONNECTED curve; group
thin components into smooth curves (`aerial_rail.enumerate_lines`) and keep only pixels on a curve
spanning >=0.30 of the content width with median perpendicular spread <=4 px. This REPLACES the bottom-
legend horizontal cut, so the outer rail SURVIVES the legend band (fixes the f840 outer-rail clip).
Verified by eye: disclaimer text / minimap / FINISH drop out; rails kept edge-to-edge incl. the legend band.
DEAD END recorded: a static-overlay PERSISTENCE mask fails — the camera holds the rail ~fixed mid-frame
on straights so the RAILS are the most persistent pixels, not the HUD (the minimap is saturated colour and
never passes the whiteness gate anyway). Cleaned pixels cached to `data_cache/_railclean_R08.npz`.

### LOOP2 — rail fit (R1+R2+R3, the parts that were actually needed)
Per-frame, on clean pixels:
1. **Fit each rail INDEPENDENTLY** (not the 0115-Exp-B constant-offset coupled model — that is an epicycle
   that misplaces at the bend, since concentric arcs are NOT a constant PIXEL offset apart under the oblique
   camera; confirmed by eye on f768/f776).
2. **fit + gather + refit**: seed the running rail (line nearest the field median), gather ALL clean pixels
   within 8 px of the curve and refit (bridges fragmentation -> full coverage), remove them, fit the partner
   from the residual.
3. **NO extrapolation** — fit over pixel support only; clean rail pixels already span the frame, and blind
   box-extension shot tilted segments diagonally (the f600 X-crossing artifact).
4. **Partner accepted only if it does not CROSS the running rail** (min gap >20 px, median <340). This is the
   ONE robust discriminator at bends — parallelism / constant-gap tests FAIL there (concentric arcs have
   different chord directions); a spurious diagonal / FINISH line always crosses.
5. **Temporal SEEDING** (`track_sequence`): carry the previous frame's curves as the gather seed for the
   next frame. The rail moves smoothly so the prev curve is a good seed -> bridges fragmented/occluded bend
   frames -> the running rail follows the arc instead of collapsing to a stub. Crucially this seeds the
   gather but REFITS on the CURRENT frame's pixels only -> no smearing (the exact failure that killed v3's
   K=5 raw-pixel window). Single-rail frames: 65 (per-frame) -> 1 (temporal) of 1227.

### Visual validation (full-res overlays, `data_cache/diag/exp_railfit/`, all 1227 frames)
Inspected the metric-RANKED worst frames + a broad sample across every phase. Correct on:
start (336–550), BEND1 (768–900, the 0115 catastrophe — now the running+outer follow the diagonal arc),
back straight (900–1300), mid (1300–1700), BEND2 (1950–2300), run-in + finish (2400–2784, no spurious
diagonal). Metric used ONLY to triage which frames to LOOK at (per discipline: numbers lie for CV).

### Status / open
- R08 visually correct across all phases on the inspected set; HANDED to the human for exhaustive final review.
- Honest residuals: 1 acquisition frame (<2 rails); the OUTER rail is faint/absent in some mid frames where
  it is genuinely out of the band (a single correct running rail there, not a failure); coverage is lower at
  the bend (clean-pixel residue), placement still correct by eye.
- NOT yet: promoted to the library / wired into the 4-feed fusion; validated on R03 + R02A; the absolute bend
  METRIC (§7) — the homography (R3 extension) is still the route to metric lateral, deferred.
- Discipline note (twice-learned this session): span / on-rail FRACTION both passed while the curves were
  visibly misplaced. CV correctness is a VISUAL verdict on full-res pixels; metrics are triage only.

## ADDENDUM (Direction 1 done) — colour-context white rejection fixes f808-822
The only residual region (f808-822, running curve hooking into the silk-legend) is fixed at the ROOT in
pixtag, not by a box: a rail-white sits on TURF, a legend NUMBER/SLEEVE (and a jockey SILK) white sits
ADJACENT to vivid saturated colour. So reject white within ~16px of a vivid non-turf colour blob
(s>140,v>80, hue far from turf). Drops the legend sleeves that bent the fit AND silk sleeves on horses,
while KEEPING the outer rail that merely dips into the legend BAND (it is on turf, not touching a swatch).
Visually verified: f808-822 running rail now continues to the corner (no hook); whole-race running-rail
support<0.6 frames 7->4 (the 4 are f810-816, the steepest bend, where the curve correctly BRIDGES the now-
sparse legend band -- low support is the metric penalising a correct bridged gap, confirmed by eye). No
regression across phases. Direction 2 (rigid-body temporal-consistency backstop) remains the recommended
NEXT robustness layer but has nothing visible left to correct after this fix.

## ADDENDUM 2 (full-span via GLOBAL geometry, not local extension)
Local extension to make short segments span the frame was a dead end: concentric-completion needs a
partner (fails config-A single rail); arc/poly self-extension LOOPS on near-straight rails (fit a finite
R to a straight sliver -> the outer rail curled across the frame, f360). All reverted.
CORRECT approach (owner-directed): reconstruct the GLOBAL track geometry, map it back per frame.
  * GLOBAL = the rail's signed image-curvature profile kappa(s) vs odometer arclength s, taken as the
    sliding-window MEDIAN over the WELL-SEEN (long-span) frames -> smooth; short/bent frames inherit
    kappa from their well-seen neighbours at nearby s (cv_rail_exp_global.py Stage 1; built inside
    cv_rail_exp_fit.build_kappa).
  * MAP BACK = per frame, pin the rail RADIUS R=1/kappa(s) and fit only the POSE (circle centre, 2-DOF
    Gauss-Newton) to the rail's own pixels, then draw the arc across the content box
    (_backproject_fixedR). On straights kappa~=0 -> R=inf -> a LINE (cannot loop). Pinning R from the
    GLOBAL profile (not the local sliver) is what both kills the loop AND spans the short frames with the
    correct curvature.
  * Generalises to config-A: a single visible rail just takes its R from kappa(s) -- no partner needed.
Validated VISUALLY (full-res): R08 all phases span correctly, straights are lines (f360 loop gone),
f810-822 fixed, finish clean. R02A (config-A, single rail) running rail spans across start/bends/run-in
from one rail -- the generalisation gap is closed. Residual: R02A shadow-heavy steep-bend frames
(~f1990) rough, a minor finish-curve artifact (~f2630). Diag: exp_railfit{,_R02A}/, exp_pixtag{,_R02A}/.
Studies: cv_rail_exp_{pixtag,fit,global}.py. Caches: _railclean_{R08,R02A}.npz, _railkappa_R08.npz.

## ADDENDUM 3 (adaptive spread + hard-coded-parameter audit)
R02A start (f390 etc.) failed: the pixtag long-curve gate rejected ALL spanning lines because R02A's
render is noisier -- its rail line has perpendicular spread ~4.5-6.3 px vs R08 ~0.7-3, just above the
fixed MAX_SPREAD=4 -> 0 kept pixels -> fit nothing. The full-width rail (ext=949) was there but barely
too thick.
FIX = ADAPTIVE spread (owner-suggested, rejection-rate driven): the running rail is ALWAYS present and
spanning, so set the spread tolerance RELATIVE to the frame's OWN thinnest spanning line:
thr = min(CEIL=9, max(BASE=4, best_spread + MARGIN=2)). Crisp R08 stays tight (best~0.7 -> thr=4);
noisy R02A loosens (best~4.5 -> thr~6.5). In the loosened regime (spread>BASE) require coverage>=2 so a
sparse aligned-TEXT row (R08 disclaimer: spread~6 but coverage~0.4) is still rejected. The same adaptive
rule was applied to the FIT's _enumerate_curves (it had a duplicate fixed spread>4 that re-rejected the
R02A rail -> R02A 0-rail frames 111 -> 29 of 1126; R08 unchanged 1225/1227 two-rail). f390 now spans.
Remaining R02A 0-rail (29, ~2.6%) are the hardest shadow/occlusion frames (rail faint near the strung
field; pixtag sometimes keeps the bottom legend strip instead) -- scattered, flagged for review.

### Hard-coded-parameter audit (futureproofing vs render drift) -- owner-requested
ROBUST already (relative/data-driven): fractions of content size (MIN_EXT_FRAC, top-band 0.15ph, kappa
window=race/40), turf-relative brightness (turf_med+28) & hue (dh>20 from turf), adaptive spread,
coverage ratio, k-sigma robust fits.
BRITTLE (to fix; classed):
 (1) absolute PIXEL distances (break on resolution/zoom change): SPREAD_* , ADJ=16, gather/seed 8/12,
     MIN_RAIL=25, no-cross 20/340, R_LINE=8000, aerial_rail MIN_EXT/MAX_WIDTH/band_px/t_margin.
     -> express ALL as multiples of ONE per-frame SCALE = px-per-metre from the known rail separation.
 (2) absolute BRIGHTNESS/COLOUR (break on palette restyle): brightness floor 150, s<75 whiteness,
     VIVID_S/V 140/80, find_layout min_legend_sat=120, render_active TRACK_LAP_MIN=240 / TOP_GREEN_MAX
     (0111 already flags these as resolution-calibrated). -> make them PERCENTILES of the frame's own
     intensity/saturation distribution.
 (3) structural: HALF=960 -> frame.width//2.
Principle: two anchors (a measured px-per-metre scale + per-frame intensity/sat percentiles) collapse
nearly all magic numbers; what remains should be dimensionless physical ratios + true track facts
(width, bend radius from hkjc.core). Highest-value first: the px-per-metre scale anchor.

## ADDENDUM 4 (de-brittling IMPLEMENTED -- the audit acted on)
The hard-coded-parameter audit (Addendum 3) is now implemented in the two study scripts, calibrated to
REPRODUCE current behaviour at the AVV scale (verified zero/near-zero regression), so it is robust to
render drift:
 * DISTANCES -> fractions of content width W (resolution-invariant): spread base/margin/ceiling,
   silk-adjacency, min-component, gather/seed bands, no-cross gap, max rail separation, straight-vs-arc
   radius, min rail count. (cv_rail_exp_pixtag + cv_rail_exp_fit; _W()/_mr() derive px per-frame.)
 * COLOUR -> turf-relative / per-frame (palette-invariant): brightness = turf_gray_med + 1.2*turf_std
   (dropped the brittle 150 floor that only suited R02A by luck); whiteness s < 0.55*turf_sat_med;
   vivid s > 0.95*turf_sat_med & v > 0.6*turf_val_med & hue-far. Replaces absolute 150/75/140/80.
 * STRUCTURAL: HALF=960 -> frame.width//2; fit mask + default ROI now data-derived (no 1080/960/150/700).
   Fit's duplicate fixed spread>4 -> the same adaptive rule.
Verified on the regenerated caches: R08 running rail found on ALL 1227 frames (0-rail=0; 2-rail 1225->1204,
the 21 are faint-OUTER frames, running intact); R02A 0-rail 29->24; f390 start fix + R08 bends (global-kappa)
+ f776 text-drop all hold. SCOPED OUT (flagged, NOT changed -- risk to the production 4-feed pipeline):
the shared library aerial_rail.enumerate_lines constants and aerial_layout.render_active gates (the latter
already journal-flagged 0111 as resolution-calibrated) -- to relativise at promotion time.
Remaining genuine constants are now dimensionless ratios (elongation, coverage, aspect) + true track facts.

## ADDENDUM 5 (colour de-brittle CORRECTED -- std inflates on bends; use turf MEDIAN)
First colour de-brittle (Addendum 4) REGRESSED R08 f768 (lost the outer rail; 2-rail 1225->1204). Root
cause found by measuring: tying the brightness threshold to turf STD is wrong -- on bend frames the band
includes the dark fence/shadows, inflating turf_std (f768 std 44 vs 30 on straights), so turf_med+1.2*std
= 198 vs the old 173 -> over-tightened -> outer rail dropped. FIX: make the offset relative to the turf
MEDIAN (robust, not contaminated by dark non-turf): brightness = turf_med * 1.19 (~ old +28 at R08, and
NOT inflated on bends). Vivid/whiteness factors re-set to reproduce the old absolutes at R08's palette
(VIVID_S_FAC 1.08~140, VIVID_V_FAC 0.50~80, WHITE_S_FAC 0.58~75). Result: R08 RESTORED (2-rail 1226 >=
original 1225, f768=2 verified) AND R02A IMPROVED (the turf-median brightness suits R02A's darker turf far
better than the old absolute 150 floor: 2-rail 317->490, 0-rail 29->20). Lesson: relativise to ROBUST
per-frame stats (median), never to a spread that non-target structure contaminates.
