---
title: Horse tagging LOOP 2/3 — field-frame tracking; phase-correlation "ground reference" FALSIFIED by member accounting; HUD killed by structure not appearance
id: 0131-horse-tagging-tracker-field-frame
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "horse-tagging", "tracking", "identity", "r08"]
refs: {"refs": "0130"}
created: 2026-06-12T00:00:00+00:00
updated: 2026-06-12T00:00:00+00:00
---
## The reference-frame inversion (the session's load-bearing lesson)

The furniture filter was designed as "ground-relative speed ≈ 0 ⇒ furniture", with the
ground's screen motion measured by phase correlation on the turf. First run: top "horse"
chain a fragmented 61 members; 124 "furniture" tracklets dropped. MEMBER ACCOUNTING broke
the story: total dets 14,956; "furniture" held 13,851 — and the biggest "furniture"
tracklets ran f336→f2770 (1,212 members, avatar-sized areas, sitting ON horses when drawn).
**The "furniture" were the horses.** The phase correlation does NOT measure the ground: it
locks onto the FIELD (the avatars are the dominant high-contrast content), so the
"ground-relative" residual was field-relative all along — horses ≈ 1 px/step, real
furniture (ground-fixed, sweeping at the pan rate) HIGH. Numbers pattern-matched the
design's prediction with the labels swapped; only the accounting + looking caught it.

Fix is simpler than the broken design: drop phase correlation entirely; measure the FIELD
shift directly as the median displacement of matched dets per step (11 of ~12 dets ARE the
field), track in the field frame (predict = pos + field-relative velocity + accumulated
field shift), and invert the filter: furniture = field-relative residual > 6 px/step.

## HUD text/odometer — killed by structure, not appearance

Remaining contamination was screen-static HUD: disclaimer text rows (white glyphs gain
dark-adjacency from the fence lines BEHIND them) and the odometer/minimap box (985-3,551 px
blobs + small green-arrow fragments). Three gates, each structural:
- det AREA cap (avatars measure 150-280 px, ~400 max; the minimap segments 985+);
- TEXT-ROW filter: >=3 PURE-white dets sharing a y-line in one frame = a text row (glyph
  rows organize in y; horses are never simultaneously pure-white AND collinear); drop all
  pure-white dets within the row band. (A chain-level static test failed first: linked
  glyph chains WANDER in x across the row — screen-static per fragment, not per chain.)
- the odometer box is an OPAQUE product HUD at a fixed panel position (aerial_layout's odo
  ROI) — zeroed before detection; horses behind it are occluded, so this is honest.

## LOOP 3 (chains) under the exclusion axiom

merge_duplicates (co-temporal + co-located = same horse's split fragments) -> two-tier
links (short gap ≤40f: position-gated in the field frame; long gap ≤400f: pooled-silk-
signature only — out-of-frame re-entries and FINISH-paint dropouts) -> count-aware final
merge (non-overlapping, silk-compatible; a CAP, never forcing).

## R08 measured state (visual gate on the full-race render)

11 chains span ~the whole race (1,221/1,208/1,201/1,198/1,089/969/937/925/921/848/737 of
1,227 frames); f1600/f2000 carry EXACTLY tags 0-10, one per horse, circles centred (by
eye). Residuals, all bounded + diagnosed: (a) rotated-bend recall dip (f800: 6/11 — the
small far avatars, a detection gap from 0130); (b) late-race tag handoffs (FINISH-zone
dropouts split some horses into big-chain + mid-chain pairs that the overlap exclusion
correctly refuses to merge); (c) anonymous tags — the saddlecloth-number binding (legend
swatches + draw order) is the next layer. R02A (crowd stress) + R03 runs pending.

## UPDATE (0132 work, R08): start pathology fixed — global second pass + upstream pixels

Owner flagged the early frames as pathological (over-tagging: 13-24 tags for an 11-field) and
directed: no-teleport + a global second pass from global info, and "better upstream pixels beat
downstream patches." Three layers, in that spirit:

1. **Upstream pixels.** (a) AREA is the count authority, markers only LOCATE — a cluster splits
   into k horses only when its area supports k (a single white horse fragments into 2-3
   vivid/white blobs; marker count alone over-segmented the bunched start). (b) Connect ONE
   avatar's fragments (silk+blaze+tail, split by the dark saddle) with an ORIENTED close along
   the travel axis before clustering — within-horse fragments align along travel, adjacent horses
   are offset across-track, so it doesn't bridge lanes. (c) Avatar SCALE refined from
   marker-bearing blobs (real avatars), then TEMPORALLY SMOOTHED across frames (the zoom is
   smooth) — the zoomed-out bend was reading rail/text furniture as scale 38 when avatars were
   ~13 px, collapsing the whole tiny pack into 1 detection. (`cv_htag_dets_cache.py` two-pass.)
2. **Global second pass = exactly N=11 no-teleport tracks** (`propagate_ntracks`). The camera
   common mode is a per-step SIMILARITY transform (pan+ZOOM+rotation) — a translation-only field
   shift could not model the start/finish zoom (horses move radially) and lost the bunched start.
   The KEY architectural lesson: a single anchor must COAST across the unreliable bend, which is
   fragile; instead seed N tracks per STABLE SEGMENT (runs where det count ≈ N) and STITCH
   segments left-to-right by pooled silk signature (+ position where they overlap). Identity then
   survives the bend/finish WITHOUT crossing them per-frame.
3. **Reliability filter on output**: where det count is far from N (the zoomed-out bend, the
   finish paint) the per-frame matches are unreliable — identity is carried THROUGH (no-teleport
   state) but those positions are not EMITTED (a wrong grab is worse than an honest gap).

Plus two propagation refinements: appearance (last-frame silk ΔE) as a tie-break in the match
cost, and an in-step ASSIGNMENT ITERATION (predict with the carried camera M, re-estimate M from
the matches, re-assign) to recover the fast-camera-motion frames (run-in tilt) where the carried
prediction had gated out the right detection.

**Validation — the right quality metric is silk-signature CONSISTENCY, not a position-jump
count.** A field-relative position-jump audit OVER-COUNTS in zoom zones (it doesn't normalise for
the zoom changing all relative distances) and reads real run-in surges as "jumps". The clean test:
a tag that swaps horses changes its SILK; map each emitted (frame,tag) back to its detection's
silk signature and measure each tag's dispersion across the race. Result (R08): **9 of 11 tags
spread < 6 LAB (one horse, held the whole race); the 2 "high-spread" tags (0, 2) are
MULTI-TONE silks** (magenta+white; yellow+red+white) whose dominant-colour cluster flips
frame-to-frame — confirmed by cropping their horse at f600/f1500/f2068/f2200 (same silk
throughout), NOT swaps. (This is why the veto uses the LAST-frame silk, temporally coherent,
not the pooled median, which would veto a two-tone silk's off-tone frames.) An EM field-stabilized
re-association was built and REVERTED: it cannot undo a sustained swap (both sides of a path
crossing are locally smooth, so it stays put) and there were almost no real swaps to fix.

Result (R08, full render + silk audit): **11 tags spanning the race, one per horse; the start is
fixed and stable** (f400↔f440 identity persists per silk; zero >11-tag frames, was 13-24); the
mid + finish-line convergence (f2620/f2680 across the 100/50 m paint) hold one tag per horse by
eye. Coverage 915/953/963 of 1227 after the reliability filter (the ~270 dropped frames are
honest gaps in the zoomed-out bend + finish paint where detection is unreliable). Residual: the
single near-total-detection-failure frame f816 (bridged, not emitted). Studies: detector
`cv_htag_detect.py`, scale-smoothed cache `cv_htag_dets_cache.py`, global pass `cv_htag_track.py`.

## UPDATE 2 (R08): camera model corrected; track-width-anchored scale fixes run-in swaps

Pushing further surfaced two corrections and a real fix.

**The camera FOLLOWS THE FIELD (corrected model).** ORB on the masked background turf gives the
true ground motion: scale 1.0000 everywhere, translation ~31 px/step median (the turf sweeps; the
camera tracks the field, so horses are ~static on screen). A turf-flow camera transform was built
(`cv_htag_camera.py`) and wired in — and REVERTED: applying the GROUND transform to a horse
predicts it moves WITH the turf, exactly backwards (coverage 915->897). The track-matched estimate
(the FIELD bulk motion — the right reference for horse prediction) is what the tracker uses.

**Real run-in swaps traced to a SCALE-ESTIMATION FEEDBACK FAILURE.** A silk-spread audit (a swap
changes the silk) found genuine swaps at f2144-2212: tag 9 was a cyan silk at f2144, a different
silk after the run-in. Cause: the run-in over-detects (15-20 blobs / 11 horses) because the
avatar SCALE collapses there (14.6 vs the true ~30) — and it's a FEEDBACK loop (small scale ->
over-segment -> small marker-bearing blobs -> small scale estimate), self-reinforcing across a
whole zone so the temporal median can't recover it. Over-detection -> the reliability filter drops
those frames -> a gap the tracks swap across.

**Fix: anchor the avatar scale to the rail TRACK-WIDTH** (`cv_htag_dets_cache.py::_tw_anchored_scale`).
The racing<->fixed rail perpendicular distance is a DETECTION-INDEPENDENT zoom proxy; avatar scale
~ K / track_width (avatars shrink as the oblique bend view spreads the rails), K calibrated by the
robust median of raw_scale*track_width. This breaks the feedback loop — scale ~28-30 at the run-in
instead of 14.6. **Result: coverage 903->940/984/989; the run-in swap cluster resolved** (silk
jump tag 3 100->6, tag 4 131->34, tag 9's run-in jump gone); 8 tags silk-spread < 5, the 2
"high-spread" tags remain visually-confirmed multi-tone silks (0 magenta+white, 2 yellow+red+white).

**Remaining residual (classical ceiling):** the DISCLAIMER-TEXT-OCCLUDED run-in (f2176-2240) — the
semi-transparent overlay text sits OVER the pack, firing false detections on partially-visible
horses; scale cannot merge these and masking the text would erase the horses under it. This is the
honest case for a learned detector (torch 2.12 + CUDA available) trained on the pipeline's own
clean-zone pseudo-labels (avatars are consistent synthetic sprites; ~900 reliable frames of
labels). Detection — not identity — is the ceiling.

## Cross-race (R03 + R02A)

- **R03** (757 frames, field 12): 12 substantial chains (333-734 members) + fragments;
  f700/f1700 verified by eye — every visible horse tagged once. Generalizes.
- **R02A** (1,126 frames, field 12, the crowd/shadow stress): non-crowd zones CLEAN by eye
  (f900 shadows, f1600, f2400 — one tag per visible horse; top chains 1,073/1,068/1,037
  ≈ full race). **Open class: the early-straight CROWD zone (f340-630)** — ~7 crowd dets/
  frame survive (at f400 both rail sides are turf so the side test is honestly ambiguous,
  0130), and in DENSE clutter the residual filter is defeated by aliasing: a crowd track
  can hop dot-to-dot within the gate, riding the field reference (residual ≈ 0 without
  being one object). 195 crowd tracklets ARE caught by the residual filter; the leak is
  the hop-aliased remainder. The honest closure is POOLED GROUND-FRAME geometry: the
  stands are a fixed band along the track, and sidedness pooled from the frames where the
  fixed rail IS seen (f700+) transports along the rail — Phase-2 (same family as the
  smooth-in-s rail model), not a per-frame bandaid. Also minor: two legend-edge tags
  (f1600 tags 47/65) — legend_top cut misses swatch tops by a few px on some frames.

Artifacts: `studies/cv_htag_track.py` (+detector in 0130's `cv_htag_detect.py`); renders
`data_cache/diag/htag_track_{race}/`; tracks `data_cache/_htag_tracks_{race}.npz`; dets
cache `data_cache/_htag_dets_{race}.npz` (delete after detector changes).
