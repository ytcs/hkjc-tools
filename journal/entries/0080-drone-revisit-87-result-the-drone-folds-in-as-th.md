---
title: DRONE REVISIT #87 RESULT — the drone folds in as the 3rd FULL-RACE feed (time-anchored on the shared clock, arclen 0->1640m), and delivers the FIRST per-horse cross-feed lateral AGREEMENT (drone vs aerial rho +0.88..+0.93, cls 100%, leave-one-out +0.83..+0.91) -- but the per-frame within/between dispersion test STILL NULLS (ratio 1.04 run-in / 2.6 home-straight / 5.2 bend, all >=1), so the new toolkit beats 0070 on between-runner ORDERING + cross-feed REDUNDANCY but NOT on absolute per-runner TRIP metres. The strong +0.93 sits on the arclen 773-1000m BEND (both feeds order moderately-faithful there: aerial-vs-stew +0.71, drone +0.46); the genuinely clean home-straight overlap (1270-1490m) NULLS (-0.58) BECAUSE the aerial itself degrades there (aerial-vs-stew +0.04, the 0075/0077 foreshortening) -- confirming 0077's structural finding that no arclen band is clean for BOTH at once. Honest mixed result: a real 3rd feed + first redundant cross-feed positive (ordering), still a §4b mis-spec on the absolute metric.
id: 0080-drone-revisit-87-result-the-drone-folds-in-as-th
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "drone", "fusion", "four-feed", "holy-grail", "task-87", "third-feed", "shared-clock", "race-t", "arclen", "time-anchor", "coverage", "census", "straightness", "bend", "home-straight", "run-in", "stable-rail", "rail-spline", "ground-contact", "shadow", "y-lat", "lateral", "control", "within-between", "dispersion-ratio", "cross-feed", "redundant", "coherence", "spearman", "leave-one-out", "rail-vs-wide", "classification", "stewards", "comments-on-running", "silk-number-bridge", "reid", "foreshortening", "under-determined", "4b", "mis-specification", "not-a-null", "null-on-metric", "positive-on-ordering", "0070", "0075", "0076", "0077", "wave-2", "external-validation", "validation-only", "leak-free", "no-model-change", "study-only"]
refs: {"study": "studies/cv_drone_revisit.py", "panel": "data_cache/cv_drone_numbered.parquet", "figure": "data_cache/figs/cv_drone_revisit_qc.png", "drone_panel_readonly": "data_cache/cv_s0d_drone.parquet (0070)", "aerial_panel_readonly": "data_cache/cv_s0d_aerial_render.parquet (0075)", "drone_video": "data_cache/video/drone_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_comments.txt (VALIDATION ONLY)", "partial": "0079-drone-revisit-87-partial-census-the-drone-is-the", "anchors": "race_t=(vid_t-3.0)*0.8931; finish-cut vid 113.68s=winner 98.85s", "headline": "cross-feed rho +0.88..+0.93 (mid-bend, LOO-robust); dispersion ratio 1.04-5.2 (>=1, metric null)"}
created: 2026-06-07T03:39:44+00:00
updated: 2026-06-07T03:39:44+00:00
---
TASK #87 — fold the DRONE in as the 3rd lateral feed: census coverage, retry a validated lateral
with the 0075 toolkit (stable rail + ground-contact), run the back-straight cross-check vs aerial.
ST 2026-06-03 R1, 1650m turf "C" (21.3m). Study-only: studies/cv_drone_revisit.py. src/hkjc/model|
core|vision/mode_b UNTOUCHED; the 0070 drone panel + 0075 aerial panel reused READ-ONLY (the 0070
silk->number bridge reused as per-number colour anchors, NOT re-OCR'd); new panel written to
data_cache/cv_drone_numbered.parquet. Official data (sections, finish, stewards comments)
VALIDATION-ONLY. ruff clean (src/hkjc + studies).

STEP 1 — COVERAGE CENSUS (the headline structural gain). 0070 said the drone has "no global
longitudinal axis" (the field-relative odometer totals only ~379m). FIX: anchor on TIME, not the
odometer. Two frame-verified real-time events -> an affine drone_vid_t -> race_t map:
  gate jump (centroid speed surges 2->12 px/frame, vid ~3.0s) -> race_t 0
  hard CUT to the photo-finish replay (centroid jump 350px, vid 113.68s) -> winner crosses 98.85s
=> race_t = (vid_t - 3.0) * 0.8931. Then race_t -> arclen via the official-section curve (the same
curve the fusion spine uses, 0077). RESULT: the drone covers race_t -2.7 -> 98s == arclen 0 ->
1640m -- THE WHOLE RACE. Per-frame rail-straightness (PCA across/along ratio) classifies it:
  * START + accel: arclen 0-450m, mixed straight/turn (vid 3-34)
  * the long BEND: arclen ~470-1130m, straightness 0.3-0.6 (vid 36-78) -- the sweeping turn
  * a STRAIGHT patch arclen ~1150-1190m (vid 80-82)
  * the CLEAN HOME STRAIGHT: arclen ~1270-1566m, straightness 0.9-1.0 (vid 88-108) -- OVERLAPS the
    aerial's validated clean home-straight (>=1300m, 0075) AND the run-in patrol+aerial both struggle
  * run-in to the line goes oblique (vid 109-113, foreshortened) then CUT to the replay.
So the drone IS a genuine 3rd FULL-COVERAGE feed once time-anchored, with a clean straight that
co-locates with the aerial's clean zone -- the redundant overlap 0077 said the aerial+patrol pair
lacked. (#80 holy-grail: this is the 3rd feed folded in, not yet the 4th/RF margins.)

STEP 2 — VALIDATED LATERAL, new toolkit (stable per-window rail + ground-contact). The 0075 recipe
ported to the panning drone: pool the inner-rail boundary points across a WINDOW (camera ~static)
and fit ONE rail line (NOT 0070's per-frame turf-boundary refit, the instability root cause); use
each sprite's GROUND-CONTACT (feet/shadow base, lowest extent) not the body centroid; scale by the
per-frame DT corridor == 21.3m; identity via the 0070 silk->number colour anchors. WITHIN/BETWEEN
y_lat DISPERSION RATIO per clean zone (0070 null was ratio>1):
  mid-bend   (773-1000m): within 2.56 / between 0.49 = 5.19   (rail too curved -> single line poor)
  home-str  (1270-1490m): within 2.44 / between 0.93 = 2.62
  run-in    (1520-1647m): within 2.14 / between 2.06 = 1.04   (best; near the boundary)
HONEST NULL on STEP 2: the ratio STILL does NOT beat 1 in any window. The toolkit helps -- the
run-in between-runner spread jumps to 2.06m (vs the bend's 0.49m) and the ratio falls to ~1.04 (vs
0070's >1 and the per-frame swings 0.8->6.3m) -- but a single straight-line per-window rail on a
PANNING camera is still not a stable enough reference for a clean per-frame absolute trip. This is
the same CLASS of §4b limit as the aerial (0075): the absolute per-runner TRIP metres are
mis-specified; the BETWEEN-RUNNER ORDERING is the recoverable signal.

drone-vs-stewards (rail=0/wide=1) Spearman: mid-bend +0.46, home-str -0.07, run-in +0.39 -- positive
where the drone geometry is least degraded, consistent with the ordering (not metric) being real.

STEP 3 — the BACK-STRAIGHT REDUNDANT CROSS-CHECK vs the aerial render (the prize). Per numbered
horse, median drone y_lat vs median aerial render y_lat on the overlapping band, Spearman + rail/
wide class-agree, WITH a 0077-style localiser (is the AERIAL itself faithful vs stewards on THAT
band?):
  mid-bend  (773-1000m):  rho +0.93, cls 100%  | aerial-vs-stew +0.71, drone-vs-stew +0.68
  home-str (1270-1490m):  rho -0.58, cls  33%  | aerial-vs-stew +0.04, drone-vs-stew +0.11
  run-in   (1520-1647m):  rho +0.32, cls  56%  | aerial-vs-stew +0.39, drone-vs-stew -0.14
The mid-bend +0.93 is ROBUST: leave-one-out rho +0.83..+0.91 (mean +0.88), not driven by one point;
drone rail horses #8/#11/#12 read 3.1-3.8m, wide #9 widest 5.5m. This is the FIRST time two
INDEPENDENT feeds agree per-horse on lateral ORDERING at rho>0.8 -- the redundant cross-feed
coherence 0077's bend overlap (patrol identity bottleneck, rho -0.03) could NOT deliver. BUT the
honest caveat: it sits on arclen 773-1000m, a BEND (not the cleanest geometry), where BOTH feeds
order only moderately-faithfully (aerial +0.71). The genuinely CLEAN home-straight overlap
(1270-1490m) NULLS (-0.58) -- and the localiser shows WHY: the aerial itself is unreliable there
(aerial-vs-stew +0.04; e.g. it reads the wire-to-wire rail leader #12 at 19.2m, plainly wrong),
the 0075/0077 home-straight-entry FORESHORTENING. So this CONFIRMS 0077's structural finding from
the 3rd feed's side: there is no single arclen band where the aerial is clean AND the drone is clean
enough for a strong redundant per-horse test by re-windowing alone.

NET / VERDICT (mixed, honest):
  + The drone is now a 3rd FULL-RACE feed on the shared clock (time-anchored, arclen 0-1640m) with a
    clean straight co-located with the aerial's clean zone -- a real structural gain for #80 fusion.
  + FIRST redundant cross-feed per-horse lateral AGREEMENT (drone vs aerial rho +0.88..+0.93, cls
    100%, LOO-robust) -- on the arclen 773-1000m band. The between-runner ORDERING signal is real and
    coherent ACROSS two independent instruments.
  - The per-frame within/between DISPERSION test still NULLS (ratio >=1 everywhere; best 1.04) -- the
    new toolkit does NOT recover a clean absolute per-runner TRIP from the panning drone. Same §4b
    metric mis-spec class as the aerial render (single under-determined rail + a moving camera).
  - The CLEANEST-geometry overlap (home straight) nulls because the AERIAL degrades there -- so the
    cross-feed prize is earned on a bend, not the clean straight; 0077's "no clean-for-both band"
    structural finding is reaffirmed, now from 3 feeds.
NEXT (for #80): the drone's run-in/home-straight is the REAL-position fanned finish that patrol+aerial
both lose -- worth a proper per-frame homography (recover the painted-marker pixel anchors or use the
oblique run-in rail as a 2nd fiducial) to convert the drone ordering into a metric there; and the
drone's clean home straight should be FUSED with the patrol head-on (0076) which is metric-clean on
the straight-entry but loses identity -- the drone's number bridge + the patrol metric are
complementary. RF margins remain the 4th feed to fold.

ARTIFACTS: studies/cv_drone_revisit.py; data_cache/cv_drone_numbered.parquet (4957 rows: numbered
drone y_lat per window with ground-contact + body variants); data_cache/figs/cv_drone_revisit_qc.png
(census band on race_t/arclen; stable-rail + ground-contact overlay; per-zone dispersion ratio;
drone-vs-aerial cross-feed scatter). Reused READ-ONLY: data_cache/cv_s0d_drone.parquet (0070),
data_cache/cv_s0d_aerial_render.parquet (0075), data_cache/video/drone_20260603_01.mp4,
data_cache/race_official/20260603R1_comments.txt (VALIDATION ONLY).
