# 0333 Stage A build, first milestone: the render-only core (segments, joint state, list with fault chains, P(horse) and detection terms, one objective, global moves); crops for the coordinator's go/no-go

Date 2026-11-24. Mode BUILD (Stage A of journal/research/recon_core_redesign.md, owner-approved). Branch: the builder's
worktree branch off origin/main 056a212. Nothing merges until Stage A passes its frozen test.

## Setup
- **Design memo:** docs/RECON.md §4, "The core" (replaces the identify / fuse_video / fuse_final sections). It lists the
  constants, the cut rules, the objective, the moves, the outputs and the status of 0316's 14 decision points.
- **Code:** src/hkjc/recon/core.py (new); pipeline.py's identify / fuse_video / fuse_final rewritten on it.
  - assoc.py and identity.py deleted; fusion.py keeps Obs, Prior, Path and with_tail.
  - Tests: tests/test_core.py (new), others rewritten by a forked helper (see Records).
- **Races:** only the ten milestone races, run from reflinked copies of main's upstream artifacts in the worktree's
  data_cache/s_stageA/runs/<cell>/<key> (journal/research/s_stageA_build/stage.sh). The 0312 cases also ran under
  their mask cells (20231001_ST_1 lg1, 20240505_ST_10 lg0, 20260301_ST_2 lg1). No other race was opened.
- **Process:** every race job through scripts/slot.sh, at most two at a time. Waits were on explicit PIDs or on the
  queue files. One queued job was killed by its PID (not started). Unit tests of test_core.py ran outside slot.sh
  (seconds each).
- **Viewer:** a helper agent wrote scratchpad stageA/view.py (render panel with labels, the list row, the product order,
  crops beside the official silks), and BEFORE images from main's outputs. AFTER images come from the same script.

## Owner and coordinator rulings applied during the build
- Compute is not a gate for Stage A.
- Known occluders (list band, disclaimer): a box overlapping one gets P(horse) `P_EDGE`; tracker continuity across an
  occluder's edge is a weak link; a horse projected there is unobserved (detection term, below).
- Detector recall varies by frame: the detection term uses a frame-level detection probability from the frame's own
  box count, so an undetected horse in a poorly detected frame is little evidence.

## Findings during the build (development on the milestone races; post hoc, not gates)
- **J must see positions the detector could have shown.** Without a detection term, 20241201_ST_10's first run put
  the list's leader #3 18 m ahead of every box (unobserved, sd 0.8 from the finish camera) and an unlisted horse on
  the leading avatar. The detection term charges a path that projects into the visible panel without a box.
- **Fault-chain emissions.** A frozen or omitted state that emits 1 always beats an ok state whose facts carry the
  misread floor: the frozen chain then sits at P ~ 1 on a clean list (unit test). Fixed: facts are taken relative to a
  clearly kept order; frozen rows make each adjacent fact a coin flip; an omitted horse's unlisted fact says nothing.
  With a coin flip for omission (first version), a 60 s omission cost ~800 units and J preferred the shifted labels.
- **20240511_ST_7 #9 (list omits the 3rd horse ~10-90 s): J prefers the truth, the search does not reach it.**
  jtest.py relabels the core's segments inside a window by the rank shift that inserts #9 3rd
  (4->9, 1->4, 7->1, 3->7, 10->3, 5->10, 6->5, 9->6). On the solution before the global moves, the [12, 70.85] s shift
  has J 80,764.8 against 81,343.6 found (no camera), and 80,787.5 against 81,367.4 with the camera. The list term
  falls 1,600 -> 1,196. The assignment step cannot propose it: every horse's path is tight on its shifted avatar, so a
  one-horse move costs the kinematic term of the whole chain.
  - Window proposals (each horse's own labels within 10 / 30 s left out, no rank votes) raise J here.
  - Permutation moves (a camera cut's label permutation carried over a stretch) find a smaller gain (12.4-14.4 s)
    and stop at J 80,918.8, still above the shifted labelling's 80,764.8.
  - Missing: an insertion move (put an omitted horse at rank r over a stretch, shifting every horse behind it).

## Changes made after seeing milestone cases (disclosed; not tuned on a gate)
- After 20241201_ST_10: the detection term (with the coordinator's occluder and recall rulings).
- After 20240511_ST_7: the omission emission (an omitted horse's unlisted fact says nothing), `P_OMIT_IN` 1e-3 ->
  1e-4, the frozen emission (coin flip per fact), window proposals, permutation moves and the insertion move.
- jtest.py's "truth" labelling is the TV-derived permutation (0322's TV order at 42.7 s), applied to the core's
  segments; it was used only to decide whether J or the search failed.
- Recall ceiling (silk 0.95, bends 0.86, disc 0.985) from the coordinator's recall audit (0330).

## Milestone results (code 294fc4f + 631e05f, cells "m2"; movements, not verdicts)
**The final-product rows of m2 are invalid**: until 2cfae52 the final product's assignment step and view maps used
render s without the metric bias that its state step applied, so every final proposal was mis-scored (J still guarded
each acceptance, so finals stayed near their best start labels). Judge the video-only products; finals are being
recomputed (cells "m3"). 2cfae52 also restricts the label-free finish registration to clean boxes: on 20241201_ST_10
its earliest "crossings" were list-icon ghost tracklets (cy 441-476 px, boxes 31-80 px; registration t_w 81.34 s
against tracklet 2's 82.57 s), the source of the ~1 s camera-vs-path conflict.
Main = the stored product (pod_B3g / product dirs; for the 0312 cases s_ms/runs/main/<cell>). Times are race seconds.

| Case | Main (before) | Core (after) |
|---|---|---|
| 1. 20240511_ST_7 #9 | #9 on t11 (last avatar) 20-47 s; absent / vague at 60-80 s | #9 on t2 (3rd avatar) 20-80 s, both products; order at 42.7 s 2, 8, 9, 4, 1, 7, 3, 10. Reached by the insertion move "#9 at rank 3 over 12.4-71.0 s" (81 segments, J 81,093 -> 80,393). Video sectional MAE 0.159 -> 0.081 s |
| 2. 20241201_ST_10 | labels flip between t0/t3 (#13) and t3/t7 (#8) at 76.5-77.8 s | video (m2): #13 on t3, #8 on t7, #11 on t10 at 76.5-80 s, no flips through the camera burst |
| 3. 20231001_ST_1 (U, lg1) | U: #10 on t6 at 71 s, #8 absent 72-74 s | #8 on t11 69-90 s; #10 on t12 from 71 s (leaving the list band); U and lg1 identical from 71 s |
| 4. 20240505_ST_10 (U, lg0) | U and lg0 differ at 27 s (#3 on t2 vs absent) | U and lg0 identical at every crop time: #3 on t6 to ~27 s, then t8; #1 on t11, then t6 |
| 5. 20260301_ST_2 (U, lg1) | lg1 swaps #10/#11 (t1/t7) 15-42 s | video: U #10 t7, #11 t1; lg1 still swaps them 15-30 s (the m2 final showed lg1 = U, invalid, see above) |
| 6. 20251126_HV_3 (frozen list) | video: #1, #3 right behind #2 through 20-40 s; final: #3 absent | m2 pending (m1 code, video = final: #11, #12, #10 at the front, #1 7th on t7 throughout, #3 on t0 at the back) |
| 7. 20250226_HV_8 #7/#11 (expected unchanged) | final: #7 on t3, #11 absent at 35-44.6 s; video: #7 on t3, #11 on t9 throughout | #7 on t9, #11 on t3, #8 on t8 at 24.6-55 s in both products |
| 8. 20260419_ST_4 #2 (expected unchanged) | #2 on t13 (topmost box) 72.6-78 s | #2 on t2 (lat 9.4) 72.6-78 s |
| 9. 20241020_ST_9 #10 (expected unchanged) | #10 ~5-7 m behind #13/#8 at 14.6-15.8 s | m2 pending (video solve still running at report time) |
| 10. 20240612_HV_8 #4 (expected unchanged) | #4 on t1 at 8.9 s, 0.2 m behind #9 | #4 and #8 unlabelled at 8.9-11 s; #4's path 2.6 m behind #9 at 8.9 s, 1.8 m at 9.5 s |

**Whole-race label churn, U against the masked cell** (share of rows labelled in either run whose labels differ):

| Race (cell) | core video | core final | main video | main final |
|---|---|---|---|---|
| 20231001_ST_1 (lg1) | 13.3 % | 13.2 % | 2.9 % | 2.4 % |
| 20240505_ST_10 (lg0) | 0.5 % | 0.5 % | 7.5 % | 3.2 % |
| 20260301_ST_2 (lg1) | 16.0 % | 9.0 % | 16.7 % | 13.8 % |

20231001_ST_1's churn is #2 <-> #3 over 10-77 s (1,083 rows): U's insertion moves put #3 / #5 at rank 8 (J gains
12-65), lg1's did not. Mechanism (diagnosed, not yet changed): the omitted state emits 1 for an unlisted fact, while
an unlisted horse abreast of the 8th listed one has p ~ 1/2 in the ok state, so the omission chain switches on for
abreast back markers after a few seconds and the list stops ordering them.

**Accepted global moves per cell** (all J-lowering; horses, window, segments, gain):
- 20240511_ST_7: permutation at 14.4 s over 12.4-14.4 s (6 segments, 424); insert #9 at rank 3 over 12.4-71.0 s (81,
  701); permutation at 69.0 s over 58.8-69.0 s (7, 172).
- 20231001_ST_1 U: insert #5 at rank 8 over 12.2-70.2 s (4, 65, video + camera); final: insert #3 at rank 8 over
  0.0-70.2 s (6, 63), #5 at rank 8 over 0.0-69.2 s (4, 12), #3 at rank 8 over 0.0-70.2 s (4, 50). lg1: none.
- 20260419_ST_4: insert #14 at rank 10 over 52.2-83.8 s (12, 3); #12 at rank 10 over 52.2-83.8 s (19, 350); #14 at
  rank 12 over 11.0-83.8 s (36, 7); #6 at rank 13 over 11.0-83.8 s (22, 54); final: #14 at rank 10 over 0-90 s (38,
  56), #5 at rank 10 over 45.6-90 s (19, 53), permutation at 45.6 s (1, 33).
- 20260301_ST_2 U: permutations at 51.4 s (6, 52), 51.6 s over 41.4-51.6 s (39, 30); final at 56.8 s (2, 570). lg1:
  insert #9 at rank 8 over 53.0-85.7 s (5, 64) and over 50.8-85.7 s (12, 67); final permutations at 55.6 s over
  41.4-55.6 s (31, 416) and 51.6 s (6, 176).
- 20240505_ST_10: U permutation at 43.4 s (2, 110); lg0 permutation at 40.6 s onward (11, 95).
- 20240612_HV_8: permutation at 16.6 s over 8.8-16.6 s (2, 60).
- 20250226_HV_8, lg1 20231001_ST_1: none.

**Gates (validate, video-only).** 20240511_ST_7 improves (sectional 0.159 -> 0.081 s). 20241201_ST_10 (m1 code) is
worse than main: sectional 0.278 -> 0.397 s, behind-winner 0.347 -> 0.664 s. There the finish camera's crossings sit
~1 s before the first solve's path crossings for 11 of 12 horses, and J_s rises ~2.5k -> 9.6k when they enter.

**Compute.** The replaced stages took 170-6,500 CPU s per cell (main: 1-5 s). The time is in the global moves, each
a full state step (joint Newton, per-horse marginals, in-view projection). Not runnable on the 60-race dev list or
block A as is: an optimisation pass and pruning of the move enumeration come before step (c).

## Records
- Library: the branch commit(s) listed in the report.
- Scripts: journal/research/s_stageA_build/ (stage.sh, run1.py, runall.sh, dbg_fin.py, jtest.py).
- Runs and logs: the worktree's data_cache/s_stageA/ (runs/, logs/, q_*.txt).
- Crops: /tmp/claude-1000/-home-ytc-Projects-hkjc-tools/7c78352b-fd6f-4198-993e-a1bc75fe6c86/scratchpad/stageA/
  (<race>_before_*.png / _after_*.png with .txt order and box lines; run_before.sh, run_after.sh).
- Interface check (`runtest.py`, the harness's `RaceRun.run(stages=identify, fuse_video, fuse_final, validate)`):
  20250226_HV_8 under legend=10@0 ran through `run()` with status ok; both label files carry `prob`. The unmasked
  check failed on a malformed mask.txt from stage.sh (fixed) and was not rerun before this report. Its final `prob` was
  0 everywhere: the metric-bias bug above (fixed in 2cfae52).
- Process disclosures: one wait loop contained a `pgrep -f` test (it killed nothing); pytest (single thread, ~30 s)
  and the viewer runs (seconds, CPU, the last ones with GPU devices hidden as slot.sh does) ran outside slot.sh; jobs
  killed by explicit PID: a queued debug job (never started), the insertion prototype, and the m1 / m2 queues A, B, D
  and job E when code changed under them.
