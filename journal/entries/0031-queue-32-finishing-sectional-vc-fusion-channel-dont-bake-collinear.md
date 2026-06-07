---
title: "Queue #32: finishing-sectional v_c fusion channel — DON'T BAKE (collinear with whole-race v_c; the closing read IS the existing v_c)"
id: 0031-queue-32-finishing-sectional-vc-fusion-channel-dont-bake-collinear
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: draft
thread:
tags: ["queue-32", "finishing-sectional", "phase-structure", "theory-3e", "fusion", "precision-fusion", "v_c", "collinear", "dont-bake", "closing-phase", "conversion-test"]
refs: {"study": "studies/finishing_sectional_probe.py", "feature": "src/hkjc/model/features.py:finishing_sectional_vc_table", "fit": "src/hkjc/model/fit.py", "constant": "FINSEC_PI_RATIO", "queue_item": 32, "panel": "data_cache/trajectory_dataset_full.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
HYPOTHESIS (THEORY §3e phase structure). The CLOSING phase (final ≈400 m) is the least
tactically-contaminated read of raw aerobic capacity — every horse maximally extended, W′ emptying,
nobody holding back — so the finishing-sectional velocity should be the HIGHEST-precision single read
of v_c, sharper than the trip-average. Fold a horse's strictly-prior finishing-sectional v_c reads
into the EB v_c posterior at their OWN precision π_fin=ρ_fin·π_race, the temporal analogue of the
baked trial/gallop fusion (give each PHASE its own precision).

CONSTRUCTION (leak-free). `features.finishing_sectional_vc_table(trajectory, mode, going)` on
`trajectory_dataset_full` (per-section velocity, 39,256 runs, full 4-yr, ST 82.9% / HV 90.4% runner
coverage). Per run: closing-section velocity = velocity at `section == n_sections` (the final 400 m).
Two modes: `raw` = the closing velocity itself; `kick` = the PACE-ADJUSTED residual
`v_close − v_mean_run` (the closing kick net of the run's own average). Par-normalise on
(going, distance) — a panel-level condition constant, NOT the horse's outcome ⇒ leak-free — re-centre
onto the race-v_c mean (the closing velocity mean 16.90 ≈ race v_c mean 16.82, so it lands on the race
scale by construction). Then the strictly-prior EXPANDING mean per horse (cum-sum shifted by one) +
prior count `finsec_m`. Fused in `fit.fit_predict_traffic` as a 4th channel exactly like gallops: a
train-only affine map (`fit_trial_vc_map`, signal=finsec_vc) puts the race-scale signal on the v_c
RESIDUAL scale; `finsec_m` precision mass enters the SHARED posterior denominator with trials+gallops;
weight clamped ≤1. `FINSEC_PI_RATIO` defaults 0.0 — production untouched (no-op).

LEAK-FREE PROOF. (i) finsec_vc is null exactly where finsec_m==0 (no prior closing read) — the
shift-by-one expanding mean. (ii) The study attaches it with a STRICT backward as-of join
(`allow_exact_matches=False`), so a race never sees its own or any later closing sectional. (iii) Par
is a contemporaneous condition constant. (iv) DECISIVE: ρ_fin=0 reproduces the base bit-for-bit
(ST 65.25%/0.0066, Δ +0.00%) under the paired fixed-seed — the channel is a clean no-op when off, so
the on-deltas are pure mechanism on shared per-race MC seeds.

THE PAIRED GATE (PYTHONHASHSEED=0, full-scale, 5-fold walk-forward, % of market | ECE):

  ST (1,952 races):  base 65.25% | 0.0066
    raw  ρ=0.5 / 1 / 2:  65.42 / 65.46 / 65.30   ECE +0.0003 / +0.0004 / +0.0008   (ECE WORSE — fail)
    kick ρ=0.5 / 1 / 2:  65.47 / 65.49 / 65.31   ECE −0.0001 / +0.0001 / +0.0007
       → only kick ρ=0.5 nominally passes BOTH (+0.22pp, ECE −0.0001) — inside MC noise.
  HV (1,296 races):  base 59.69% | 0.0046
    raw  ρ=0.5 / 1 / 2:  59.64 / 59.64 / 59.64   ECE −0.0004 (flat-to-NEGATIVE on %market — fail)
    kick ρ=0.5 / 1 / 2:  59.64 / 59.64 / 59.64   ECE −0.0003 (flat-to-NEGATIVE on %market — fail)

VERDICT: **DON'T BAKE.** Leave FINSEC_PI_RATIO = 0.0 at both venues (production no-op, already so).
- HV is DEAD-FLAT (−0.05pp %market both modes) — the channel adds nothing.
- ST's best cell (kick ρ=0.5: +0.22pp, ECE −0.0001) is a single non-robust GATE-PASS sitting INSIDE
  the ±0.3–0.5pp MC-noise band the methodology warns of; the adjacent ρ=1.0 already flips ECE positive,
  and `raw` worsens ECE at every ρ. Not a real, robust both-margin improvement on both venues.

WHY (the collinearity, with evidence — a FLAT result is the valuable finding here). The whole-race v_c
already in the posterior IS essentially the closing read: `energetics.estimate_run_energetics` defines
`v_c = min` of the FINAL TWO sections, so per-run corr(closing-section velocity, race v_c) = **0.96**.
The "finishing sectional as a high-precision v_c channel" is therefore NOT a new axis — it is the
channel the model already shrinks into v_c. Folding it back in is near-redundant, hence the flat gate.
The ONE genuinely independent sub-axis is the pace-adjusted closing KICK (`v_close − v_mean_run`), which
is ~orthogonal to whole-race average velocity (corr 0.09); that is why `kick` is the only mode showing
any ST flicker — but the flicker is sub-noise. A dead TERM, not a dead AXIS.

WHAT TO TRY NEXT (the axis stays open; this rejects the STRUCTURE, not the phase mechanism):
1. Closing-kick RESIDUAL on the W′ knob, not v_c. The clean closing kick is the late anaerobic
   reserve emptying (THEORY §3d/§3e), the SAME phenomenon as the baked FADE_W_GAIN (fade on W′). The
   finishing kick likely belongs on W′ (distance-aware via W′/L), where it is NOT collinear with the
   v_c the closing split already feeds — and where the existing fade term shows it converts. The raw
   v_c channel here double-counts v_c; the residual-on-W′ is the non-redundant home.
2. Phase-SPECIFIC precision / a true 3-phase Wald (start/cruise/closing drifts) rather than one channel
   bolted on a homogeneous law — the §3e program proper. The single-channel fusion can't express that
   the closing read should DISPLACE (not supplement) the cruise-contaminated v_c.
3. Pace-CONTEXT-adjusted closing velocity: normalise the closing split by the pace it was run INTO
   (the field's mid-race tempo), isolating the horse's closing ability from the race shape — a sharper
   read than the own-run mean used in `kick`.

IMPLEMENTATION NOTE (caught a real bug — relevant to the gallop channel). The first gate run was
catastrophic (ST 65→−187%) because the per-horse fusion term applied the affine map to the RAW
race-scale finsec_vc (~16.85) instead of the rating-band RESIDUAL — `fit_trial_vc_map` residualises x
internally (`x = x − fixed`) for non-`trial_fig` signals, so the prediction MUST subtract the same
per-horse fixed effect; passing raw injected a spurious ≈+5 m/s v_c bump. Fixed:
`fmap_a + fmap_b·(finsec_vc − pop.v_c.fixed(band, rtg))`, term clamped ≤1 (finsec_m is a large per-horse
count, mean ~10, so the unclamped m_f=1-normalised weight over-weights multi-read horses). ⚠️ The
GALLOP channel (`gallop_term = … (gmap_a + gmap_b·gvc[i])`, fit.py) has the IDENTICAL raw-signal bug,
latent only because GALLOP_PI_RATIO=0 everywhere (never exercised). It will misfire the same +5 m/s way
if ever switched on — flagged for whoever takes up the gallop fusion (queue #30) to apply the same
`(gvc − fixed)` correction before baking.

Production untouched (FINSEC_PI_RATIO 0.0 both venues, no constants baked). MODEL_STATE not updated
(orchestrator serializes).
