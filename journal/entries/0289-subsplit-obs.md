# 0289 Official 200 m sub-splits as fuse_final observations: declared verdict FAIL (kinks)

Date 2026-11-23. Fast declared round (0285 fix 2). Base main 25e1f3b; origin/main later moved to aff0d30, which
changes no path maths (only appended columns), so the t/s comparison stands.
- Declaration: journal/research/s_subobs/DECLARATION.txt (md5 9b1f6568 at first write, commit 02d0403), plus amendment 1
  (scorer check), written before any on-arm number.
- Scripts: pick.py, replay.py, run.sh, score.py, scorer_check2.py. Outputs: s_subobs/out/ (summary.txt, P1/P2 rows).
- The change tested: subobs_25e1f3b.patch, with tests test_subsplit_obs.py.txt. It is not committed to the library.

## Change tested (HKJC_SUBSPLIT_OBS=1, not adopted)
- The visible official sub-split times (600 / 200 m to go) enter official_observations as soft crossings, feed
  "official", tid 35_000 + horse.
- The sd is SUBSPLIT_SD = 0.44 m, priced from 0285 (b)'s pooled bound on the independent official point noise.
- The sub-split points are also anchors for the metric bias. A withheld point sets no bias.
- New mask Mask(subsplit=k) withholds one point. It is exclusive of outcome and boundary, and existing specs render
  unchanged.

## Run
- 40 races (dev201 and gate149; 10 per set and venue at an even stride; turf; none of the sealed 25).
- Local fuse_final + validate replays from the stored fuse_video.pkl: 320 replays at 2.4-4.6 s each, one process,
  2 G cap.
- Scorer check: off_U all-halves p90 is 1.490 %, against a declared 1.18 % ± 0.3 pp.
  - 0272's own rows for these 40 races give the same 1.490 % (n 1882), as do the stored products, so the scorer is
    exact and this sample simply sits high.
  - Recorded as amendment 1, then the round continued.

## Result (off_U vs on with the point withheld; paired race-bootstrap 90 % CI)
| | off | on | delta [90 % CI] | rule | |
|---|---|---|---|---|---|
| P1 bracketing-halves p90 frac (n 1882) | 1.490 % | 1.480 % | -0.010 [-0.118, +0.056] pp | CI upper < +0.03 pp | **miss** |
| P2 withheld-point p90 \|e_rel\| (n 941) | 3.238 m | 3.074 m | -0.165 [-0.268, +0.045] m | CI upper < +0.05 m | ok |
| P2 median \|e_rel\| | 0.651 m | 0.646 m | -0.005 [-0.023, +0.006] m | info | |
| G1 H1 interior \|err\| median / p90 (n 471) | 0.126 / 0.288 s | 0.126 / 0.288 s | CI upper +0.000 / +0.000 | < +0.005 / +0.010 | ok |
| G1f H1 finish median / p90 (n 468) | 0.155 / 0.441 s | 0.153 / 0.438 s | CI upper +0.002 / +0.008 | same | ok |
| G2 kinks at boundaries, median / p95 | 0.0096 / 0.0374 | 0.0095 / 0.0392 m/s | x1.05 | | ok |
| **G2 kinks at sub-split points, median / p95** | 0.0040 / 0.0203 | **0.0128 / 0.0488 m/s** | x1.05 | | **FAIL** |
| G3 visible interior \|r\| median / p90 | 0.0290 / 0.1048 s | 0.0287 / 0.1038 s | ≤ +0.02 s | | ok |
| G3 finish > 0.02 s | 0 | 0 | | | ok |
| G4 plumbing (on_H == off_H at 400 m withheld; off_S0 == off_U) | | | 13/13, true | | ok |

- **Verdict: FAIL.**
  - The guard that fails is G2 at the sub-split points.
  - P1 also misses non-inferiority, because its CI is wide.
  - No expansion, because the declared rule allows one only when every guard holds. The switch is not committed.
- By point and venue (information):
  - 600 m withheld: P2 HV 2.80 → 2.65 m, ST 4.22 → 4.09 m.
  - 200 m withheld: P2 HV 2.12 → 2.15 m, ST 3.88 → 3.74 m.
  - P1 is lower in every cell (ST 2.10 → 2.05 %, 1.89 → 1.82 %).
- In-sample (information only; circular because the points are inputs): all-halves p90 1.490 → 0.889 %; sub-split e_rel
  SD 2.54 → 0.97 m.
- det_labels_final changed in 31 of 40 races (2 to 1,583 detections). The officials at 200 m spacing feed the
  skeleton, the anchor check and the final E-step.

## Reading
- **The kink is real and is the price of the point pull.**
  - On the on arm the sub-split-point kink (median 0.013, p95 0.049 m/s) is about 1.3x the existing boundary kink
    (0.0095 / 0.039 m/s).
  - The off arm has almost none at those points, because nothing pins the path there.
- **The held-out gain is on the right side but not resolved.**
  - At the withheld point, p90 falls 0.17 m, but the CI touches +0.05.
  - The gain comes from the identity changes driven by the other point, not from the withheld point itself.
- **Next options**, which would need a new declaration:
  - a smoother pull: the same obs with the acceleration prior relaxed locally, or obs spread over ±0.2 s;
  - or SUBSPLIT_SD at the boundary sd, 0.6 m. That is not a free fit: the boundary price applies to the same quantised
    structure.
  - Either way G2 stays at x1.05.

## Library
- Committed: Mask(subsplit=k) as evaluation tooling (owner, 2026-11-23: the masked-evaluation extension stays),
  rebased onto aff0d30.
- No stage reads sub-splits. So Mask(subsplit=k) only marks the run dir (mask.txt and the guard), and outputs are
  unchanged.
