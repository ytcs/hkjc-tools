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

## Re-submission after the Gate 1 NO-GO (code 8d8d048; cells m7, HV_3 m8)
Fixes, in the coordinator's order:
1. **Omission state** (606930b): an omitted horse explains only a lead of more than `OMIT_MARGIN` (2.4 m) over the
   last listed horse (fact likelihood max(q(d), 1 - q(d + 2.4))); a horse level with the 8th listed one gains nothing
   from it. Chains run over every list row (vectorised). Insertion candidates now come from rank jumps at stretch
   edges, not from the omission posterior of the current labelling (1b12c65).
2. **Posterior** (606930b, 211e2d0): per segment over every labelling the solve scored with a full state step, weighted
   exp(-(J - J_best)/(2 T_POST)), T_POST 10; abreast-pair swaps (the stretch, and carried to the race's end / from its
   start) are always scored.
3. **`prob`** in both label files (631e05f); the m2 files the coordinator read predate it.
4. **Large moves**: every accepted move with its segments and gain is listed below.
5. **Compute** (606930b, f80c537): precomputed Hessian positions, quadratic line search, vectorised and scaled
   forward-backward, move candidates screened by a truncated state step (best 3 get a full step). Replaced stages per
   cell (CPU s, unmasked unless stated): HV_8 35, HV_8 2024 39, 0312-1 59 / lg1 48, 0312-2 58 / lg0 62, HV_3 84,
   ST_7 97, 0312-3 185 / lg1 254, ST_4 335, ST_10 651, ST_9 2,809. Before: 170-6,500. ST_9 and ST_10 are still far
   from "tens of seconds".
6. Also: the start solve counts rows of a held near-cloth read (contract.legend_frozen) as frozen with P 0.9
   (8d8d048), after HV_3 followed the frozen list under the m7 code; only HV_3 has such a window among the cells.

Milestone cells (video products; movements):
- 1. ST_7: #9 on t2, 3rd, 20-80 s; order at 42.7 s 2, 8, 9, 4, 1, 7, 3, 10, 5, 6.
- 2. ST_10: #13 on t3, #11 on t7, #8 on t4 at 76.5-80 s, no flips. Gates: behind-winner 0.347 -> 0.184 s, finish rho
  0.867 -> 0.956, sectional 0.278 -> 0.530 s (worse).
- 3. 0312-1: #8 on t11 from 70 s, #10 on t12 from 71 s in U and lg1; #2 9th at 71-75 s in both (list 10, 7, 8, 6, 4,
  1, 9, 5).
- 4. 0312-2: U and lg0 identical (#3 t6 then t8, #1 t11 then t6).
- 5. 0312-3: U and lg1 identical (#10 t7, #11 t1). prob 1.0 on both: the swap alternatives cost dJ +6,229 (the
  abreast stretch 11.0-41.4 s, mostly lateral and detection terms: the tracklets keep distinct lanes past 41 s) and
  +3,958 (to the race's end, mostly the list once the pair separates).
- 6. HV_3: #11, #12, #10 behind #2 throughout the frozen window, #1 7th (t7), #3 at the back: the order of the live
  lists at 15 s and 45 s.
- 7. HV_8: #7 t9, #11 t3, #8 t8, 24.6-55 s.
- 8. ST_4: #2 on t2 (wide).
- 9. ST_9: #10 on t6 as main, 7 m behind #13 at 14.6 s; #8 now on t7 (lat 13-14) 3.5 m ahead of its main position.
- 10. HV_8 2024: #4 and #8 unlabelled at 8.9-9.5 s; #4 on t6 at 11-14 s (lilac crop; list at 11 s has #4 3rd, product
  3rd).

Churn U vs masked (video / final): 0312-1 1.1 / 1.1 % (main 2.9 / 2.4), 0312-2 0.0 / 0.0 % (main 7.5 / 3.2),
0312-3 1.5 / 1.4 % (main 16.7 / 13.8). Prob on 0312-3's pair rows over 12-52 s: median 1.000, share < 0.9 0.000 in both
arms.

Accepted global moves (segments, gain in J), m7 / m8 logs:
- ST_7: permutation 12.4-14.4 s (6, 424); insert #9 at rank 3 over 12.1-71.0 s (81, 372); swap #9/#10 over 71.0-97.1 s
  (1, 376).
- ST_4: insert #12 at rank 10 over 11.0-48.6 s (13, 233); permutations at 48.6 s over 11.0-48.6 (10, 84), 43.4-48.6
  (5, 25), 48.6-56.6 (10, 216).
- ST_9: inserts #10 at rank 5 over 0-63.4 s (21, 156), #10 at rank 9 over 65.2-98.4 s (18, 309), #14 at rank 11 over
  12.0-50.2 s (20, 184), #3 at rank 5 over 50.2-69.0 s (27, 36); permutations (10, 171; 2, 13; 2, 2; 3, 1; 2, 14);
  swaps #2/#14 (10, 19), #3/#7 (2, 24), #6/#10 (2, 244), #8/#14 (6, 23), #9/#10 over 0-60.6 s (11, 628).
- 0312-3 U: permutations at 51.4 s (6, 48), 51.6 s over 41.4-51.6 s (39, 93); lg1: (1, 15), (8, 381), (13, 166).
- 0312-2 lg0: permutation from 40.6 s (11, 95). HV_8 2024: permutation (2, 39 and 60), swap #4/#11 (2, 58).
- HV_8, 0312-1 (U, lg1): none.

## Round 2 after the coordinator's second NO-GO (code 2af4437, cells m14)
- **Case 5 (20260301_ST_2): the core reproduces the RENDER, which disagrees with the patrol.** In the render the box
  the core labels #10 (t7, then t2; lat 3.7-6.3) carries #10's silk (navy body, red cap, light-blue/red sleeves) and
  the inside box it labels #11 (t1) carries #11's silk (green, orange sash): crop
  scratchpad/stageA/case5_render_silks_46_52s.jpg (render panels at 46 and 52 s). The list orders 10 ahead of 11 at
  every row 40-60 s, and the wide box leads the inside one by 3.7-5.3 m from 40 to 80 s (render s). The swap IS scored
  on re-fitted paths: the whole-race swap (11.0-85.7 s) leaves J_lat unchanged (2,183 both) and costs dJ +3,969 =
  list +3,052, along-track +837, detection +52; the stretch-only swap (11.0-41.4 s) costs +6,229, mostly lateral
  (+4,174), because each label then jumps lanes at 41.4 s. The coordinator's patrol reading (#10 inside #11 at
  41.9-51.9 s) is a render-vs-real conflict that render-only evidence cannot carry (Stage C, patrol).
- **Case 2 (20241201_ST_10), sectional.** The worsening was all at the finish boundary: every horse's video path
  crossed 1.0-1.7 s after its official time, interior boundaries no worse. The render's own boxes reach the line late
  (#3's last labelled box: d 3.2 m at 82.3 s; official 81.2 s) after the render camera change near 78 s. Main's paths
  were pulled onto the official times by the finish camera, whose clock comes from the label-free registration; in
  this race the registration's earliest crossings are list-icon tracklets, whose projected crossings fall near the
  official times. 2cfae52 had excluded those boxes, so the camera clock followed the late render (sectional 0.530 s).
  Reverted in 45dc97c: sectional 0.314 s (main 0.278), behind-winner 0.481 s (main 0.347). The render metric after
  camera changes stays unmodelled (decision point 12).
- **Case 2, #4 at 54 s.** Under the m14 code #4 is labelled on the 4th avatar (s 921.2, sd 0.2) where the list names
  #14; #14's path sits 0.1 m behind #5 (abreast, cheap), and the unlisted #4 4 m ahead of the 8th listed horse is
  explained by the omission state (entry about 18, exit about 14). In the m11 run #4 had no box at 54 s and its
  interpolated path ran 1.2 m ahead of #2, within the order facts' abreast tolerance. Not resolved: the list's
  statement about #4 is outweighed by that avatar's kinematic continuity, and no appearance term separates them.
- **Search shortcuts and stability.** With 12 screened candidates per family (2d12e57) the churn of 20240505_ST_10
  lg0 rose 0.0 -> 16.8 % and of 20260301_ST_2 lg1 1.5 -> 14.2 % (a J-lowering permutation was screened out); 40
  (858c202) restores 0.0 % and 1.6 %. ST_7's insertion was likewise dropped at 12.
- **Compute (CPU s, replaced stages, m14):** HV_8 28, HV_8 2024 40, 0312-1 68 / 61, 0312-2 69 / 110, HV_3 98, ST_7
  105, 0312-3 217 / 221, ST_4 297, ST_10 514, ST_9 890. Profiles: ST_9 (the 2,809 s run, before the screening cap):
  3,900 screening state steps in 17 screening calls took 471 of 679 profiled s, the fault chains 186 s (since replaced
  by scans), the MILP 44 s. ST_10 (video only, 149 profiled s, m13 code): full state steps 44 s, unaries 40 s
  (kinematic blocks 35 s), window proposals 27 s, screening 24 s, MILP 17 s.
- **Milestone cells (m14, video):** 1: #9 3rd (t2) 20-80 s. 3: #8 t11 / #10 t12, U = lg1, #2 9th at 71-75 s. 4: U =
  lg0. 5: U = lg1 (#10 t7/t2 wide, #11 t1 inside), prob 1.0. 6: live-list order through the frozen window. 7: #7 t9,
  #11 t3, #8 t8. 8: #2 t2 wide. 9: #10 t6 as main. 10: #4 t6 at 11-14 s. Churn U vs mask 1.1 / 0.0 / 1.6 % (main
  2.9 / 7.5 / 16.7).

## Convergence of the global solve (coordinator priority, 2026-11-24 evening)

- **Measurement (frozen code 6e7faff, video evidence with the cell's finish camera; tools multistart.py, crossj.py,
  whystuck.py, unarycheck.py in journal/research/s_stageA_build).** Each (race, cell) solved from several starts:
  the solver's own init, the other cells' stored labellings, two seeded perturbations of the init's end (10 random
  two-runner exchanges over one segment's span). Results in the reply to the coordinator and in
  data_cache/s_stageA (ms_before.log, ms_q*.log, triage.log). Of 20 tune-split U-vs-lg pairs, 15 had one side's own
  evidence prefer the other side's labels (search failure, 8-2,276 J); one (20250716_HV_6 lg1) is an evidence
  difference (half of the list masked decides it).
- **Causes found.** (1) The assignment step's unaries are not a linearisation of J: on 20241013_ST_2 the list-vote
  term predicted +35 / +80 / +29 / +56 for single relabellings whose exact J_list change was -3 / +5 / 0 / 0, and
  single moves that lower J by 25-71 were never proposed. Dropping the list votes from the step unaries made spreads
  worse (ST_2 U init end 42,763 -> 43,168), so they stay as the proposal's steering. (2) The gaps between ends are
  cascades: split into label-connected parts, each part alone raises J (HV_9 U vs lg0: 27 parts summing to +2,611
  for a joint -2,142), so no single-family move reaches them. (3) J does not depend on the state step's warm start
  (ST_2: cold vs warm within 10 J), so the ends are true local minima of the move set.
- **Slot breach.** One whystuck.py run went through slot.sh while both diagnostic queues held slots (3 slots for a
  few minutes). Not repeated.
- **Declared before results: acceptance of the fix.** (a) On the bench (20241013_ST_2 U/lg1, 20240501_HV_9 U,
  20241109_ST_9 lg1, 20251012_ST_4 lg1, 20240207_HV_4 lg1) the product's solve ends within MIN_GAIN of J_ref (the
  lowest J any source has reached on that evidence) on at least 4 of 6, and its median gap falls by >= 80 %.
  (b) multistart.py on the new code (the product's solve, then the same with each extra start given as a start):
  every end <= the J of its given start; median end-J spread over race-cells falls by >= 80 % against the frozen
  code, and no race-cell's spread grows. (c) CPU of the replaced stages reported per race (not gated).
  (d) The evidence-subset inits must not use the harness's 10 s legend-mask grid.
- **Fix (24cbf3d, 3f7728d).** Five inits (all evidence; list vote only; reads + silk only; list rows of alternate
  7 s blocks with reads + silk, each parity), each descended by assignment steps; the global move families from
  every distinct end; then crossovers between the lowest end and every other end (runs of label-connected
  difference parts, screened like the other families), descending again until none lowers J. Rejected on the way
  (journal/research/s_stageA_build/parked_*.patch, bench logs): proposal parts (no change on ST_2 U); wider abreast
  swaps (U J -286 on ST_2, lg1 unmoved, CPU x1.7); dropping list votes from the step unaries (worse ends); global
  moves from only the 2 lowest assignment-step ends (lost 335 J on ST_9 lg1 and 435 J on ST_4 lg1: the ranking
  after assignment steps does not predict the ranking after global moves); seeded random kicks (no gain on ST_2 lg1
  for +50 s).
- **Bench (video evidence; J_ref = lowest J any source reached before the fix):**

  | race cell | frozen init end | J_ref | 3f7728d end | CPU frozen -> 3f7728d |
  |---|---|---|---|---|
  | 20241013_ST_2 lg1 | 42,455 | 41,981 | 41,881 | 22 -> 140 |
  | 20240501_HV_9 U | 68,095 | 65,956 | 66,099 | 57 -> 616 |
  | 20241109_ST_9 lg1 | 72,369 | 71,736 | 71,450 | 35 -> 416 |
  | 20240207_HV_4 lg1 | 75,184 | 74,860 | 74,942 | 12 -> 170 |
  | 20241013_ST_2 U | 42,763 | 42,640 | 42,467 | 28 -> 168 |
  | 20251012_ST_4 lg1 (24cbf3d) | 83,492 | 83,103 | 83,056 | 196 -> 747 |

  Acceptance (a): 4 of 6 at or below J_ref (HV_9 U +143, HV_4 lg1 +82); median gap 431 -> 0. Met.
  Profile (ST_9 lg1, 290 s): screening quick steps 85 s, MILP 68 s (106 calls), window costs 56 s, unaries 55 s.
- **Final product.** Global moves in the final solve (on top of the multi-start) lowered final J by 11 (ST_2 U, +71
  CPU s) and 84 (20231001_ST_1 U, +211 s); kept off. Crossovers run in every solve.
- **Merged listfault (a6c5177) as 0c72d55.** Clean merge; 187 tests pass.
- **Solver CPU, exact (5062cb3, c2c4cf2).** MILP pairs whose cost exceeds every link reward of their segment get no
  variable (never optimal; step MILPs 0.3-1.6 s -> 0.06-0.3 s, same solutions); unaries cached by (state,
  labelling); lateral window columns cached by band; fault-chain scans on separate entry arrays (bit-exact on random
  chains, 2.1x); banded LAPACK calls without the finiteness scan. Bench (4 race-cells, merged code 0c72d55): same
  labels, J and marginals; CPU 230/224/156/340 -> 191/132/105/239 s (sum 951 -> 667). Remaining profile (ST_9 lg1):
  screening state steps 110 s (fault chains 69 s of it), kinematic unaries 49 s, window costs 38 s, MILP 17 s.
- **Dev-list P1 runs.** before = 6e7faff (runs/p1before), after = c2c4cf2 (runs/p1after; a few half-b cells may come
  from the 0c72d55 export, claimed exact), cells U + lg0/lg1 (silk) or dc0/dc1 (disc).
- **Finish sectional (f788005).** On the 60 dev races (main's video paths, findt.py / finbias.py) the render's metric
  bias at the line exceeds the last boundary's: silk median +1.07 m (tune half; check half +1.68; in time the silk
  finish crossing is 0.08 s later than the 400 m bias implies, t about 3.9 over 40 races), disc -0.50 m (sd 2.4, not
  distinct from 0). A withheld finish now takes the last visible boundary's bias plus that step (silk 1.07, disc 0).
  Offline estimate on main's F cells (finstep_eval.py, dt - step / v): silk check half median |dt| 0.125 -> 0.091 s,
  p90 0.390 -> 0.351; tune half 0.117 -> 0.123, p90 0.483 -> 0.425. To be measured on F cells run with the code.
  It changes only cells with the finish withheld (not P1's cells).
- **Case 6 regression and pair exchanges (9a94af5).** The Gate-1 rerun of 20251126_HV_3 on c2c4cf2 kept #7/#9 swapped
  for the whole race (coordinator confirmed by eye): kept J 81,825.1, true labelling 81,811.7 (list -12, silk -6). The
  two are 3-4 lateral units apart, so no abreast swap proposed it, and the linearised cost change ranked the
  race-long exchange 511th (+85,616 against a true -13.4). New family: any two horses over the whole race, cut->end and
  start->cut (cut windows only where the two paths are within 3 m / 4 units at the cut), screened by the truncated
  state step alone (its value matched the full step within 0.6 J on the case). Variants (bench J vs c2c4cf2 / CPU s):
  in every descent 0 / -48 / -367 / -74, CPU 1,216; in a last descent from the lowest end 0 / -14 / -367 / -74, CPU
  628 (c2c4cf2: 667) -> kept the last-descent variant. Case 6 now 81,803.8 with #7/#9 right. Screen check: 23 pair
  exchanges screened below J on case 6 (every-descent run), 0 not confirmed by the full step.
- **Dev runs restarted on 9a94af5** (stale c2c4cf2 rows moved to runs/p1after_c2c4cf2_stale); split c/d 20 races
  each (coordinator), e/f 10 each (mine, after the Gate-1 reruns of cases 6, 4, 1, 3, 5).
- **Case 5 search miss and chain moves (WIP, not committed; wip_chain_relabel.patch).** Gap U vs lg1 under U's
  evidence 2,228 J: three tracker chains (A 44-83 s, B 47-83 s, C 48-83 s) permuted; U left C's tail (segment 105,
  253 detections, P(horse) 0.97) as clutter. The approved cut re-matching (paths-gated) could not reach it: U's paths
  for #4/#9/#11 are 4-11 m apart at the 56.8 s cut. Tracker-chain moves (exchanges and 3-cycles of whole chains and
  from the change point, carrying unambiguous continuations incl. clutter; partners: the 8 chains closest by boxes
  within 1 s of the change) plus a chain-aware exact relabel pass (segments >= 10 detections) in the last descent:
  case 5 U 93,960 -> 91,968 (bar 91,732, not met: +236), #10/#11 unchanged at P 1.000 (swap dJ +505 under video
  evidence), CPU 2,644 s; bench J 0 / -314 / 0 / -377 vs 9a94af5, CPU 628 -> 1,130 s.
- **Linearised vs one-horse refit (relabel candidates, full-step dJ):** case 5 U (257 of 1,028 evaluated): Spearman
  0.800 -> 0.884, exact-negative in top 50: 1/2 -> 2/2; ST_2 U (old labels): 0.938 -> 0.939 (1 exact-negative).
- **Miss rate on c/d races (9a94af5, 5 races):** gaps (lg0, lg1) HV_1 0/+455, HV_9 -1,033/-295, HV_3 -0.4/-16,
  HV_2 0/+65, 20260503_ST_4 +406/-38 -> 3 of 5 miss; P3 flips at p >= 0.99 pooled 3,396 / 65,472 (5.2 %).
- **Case 5 lg1 final flip.** In the pipeline the lg1 final's second solve scored its start 2 (the video labelling)
  at 99,879 and kept an end at 90,372: under final evidence the video labelling is 9,507 J worse, and the final
  re-labels 11-83 s wholesale (#10/#11, #4/#9/#11, #1/#4/#5, ...). My offline final-evidence reconstruction
  (finaleval.py) does not yet reproduce the pipeline's J (103,167 for the stored final labels), so the term
  attribution is open.
