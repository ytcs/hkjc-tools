---
title: "Queue #29 (T3+T4): peak-gap mean-reversion ASYMMETRY (HV passes thin, ST flat) + unified allometric b (split is load-bearing — DON'T unify)"
id: 0031-queue-29-peakgap-mean-reversion-asymmetry-and-unified-allometric-b
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: draft
thread:
tags: ["peakgap", "mean-reversion", "asymmetry", "career-landmark", "process-feature", "allometric", "mass", "allometric-b", "parsimony", "venue-split", "conversion-test", "queue-29", "ECE", "draft"]
refs: {"feature": "features.rating_trajectory_table", "knobs": ["fit.rtg_peakgap_asym_gain", "fit.rtg_peakgap_vel_gain", "fit.allometric_b"], "constants": ["PEAKGAP_ASYM_GAIN / PEAKGAP_ASYM_GAIN_BY_VENUE", "PEAKGAP_VEL_GAIN / PEAKGAP_VEL_GAIN_BY_VENUE", "ALLOMETRIC_B_BY_VENUE"], "studies": ["studies/peakgap_reversion_probe.py", "studies/allometric_b_probe.py"], "ST_base_pct": 0.6525, "HV_base_pct": 0.5969, "HV_asym_best_gain": -0.03, "HV_asym_best_pct": 0.6063, "HV_asym_best_d_ece": -0.0007, "queue_item": 29}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---

Two independent sub-parts, gated separately. Both paired fixed-seed (PYTHONHASHSEED=0), full scale,
both venues, %market | ECE. Base for each gate is the CURRENT production path (not zero).

=============================================================================================
## T3 — peak-gap as career MEAN-REVERSION (not a static deficit)

HYPOTHESIS (THEORY §3d career-phase / mean-reversion). The baked PEAKGAP_GAIN enters the own-peak
rating deficit (rtg − own prior peak, ≤0) as a FLAT linear coupling — a static handicap. The refinement:
a deficit is a MEAN-REVERSION signal. A horse far below its career peak is often an improving/returning
horse the market UNDER-rates (reversion UP toward peak); a horse at a fresh peak may be over-bet
(regression DOWN). So the coupling may be ASYMMETRIC around the career landmark, OR the rating's recent
TREND/velocity (recovering vs declining) may carry the signal instead of the raw gap LEVEL. Per the
MODEL_STATE refuted list, raw rating velocity/accel was refuted as generic velocity (restates form);
only the peak LANDMARK converts — so the LIVE hypothesis is the asymmetry STRUCTURE around the landmark,
not generic velocity.

CONSTRUCTION (additive, leak-free, default NO-OP).
- Two new knobs on `fit.fit_predict_traffic`, both stacking ON TOP of the live static peakgap term:
  - `rtg_peakgap_asym_gain` (a) — an EXTRA coupling on the BELOW-peak (raw ≤0) portion only,
    within-race centred: `v_c += gain·(min(rtg_peakgap,0) − field_mean)`. A non-zero gain ⇒ the
    below-peak slope DIFFERS from the at-peak (flat) slope = the asymmetry around the landmark.
  - `rtg_peakgap_vel_gain` (b) — the rating velocity toward/away from peak (`rtg_vel` = Δrtg/k),
    within-race centred, on v_c (the reversion TREND rather than the gap LEVEL).
- Constants `PEAKGAP_ASYM_GAIN[_BY_VENUE]`, `PEAKGAP_VEL_GAIN[_BY_VENUE]`, default 0.0 ⇒ production
  untouched. Both columns (`rtg_peakgap`, `rtg_vel`) are the EXISTING strictly-prior as-of feature
  (`features.rating_trajectory_table` → `attach_rating_trajectory`, backward join
  `allow_exact_matches=False`, same one already baked for the static term). The new knobs only
  RE-WEIGHT that leak-free column (min/centre/velocity) — NO new data path, NO new join. EB population
  stays train-only per fold. Determinism confirmed (PYTHONHASHSEED=0 reproduces to the digit).

THE GATE (paired fixed-seed, FULL scale; base = CURRENT static peakgap; %market | ECE):

  ST (1952 races, 24433 runners; 24.2% below-peak)   base 65.25% | 0.0066
    (a) asym=-0.03                                         64.90% | 0.0055   (Δ% -0.35, ΔECE -0.0011)
    (a) asym=-0.02                                         65.19% | 0.0060   (Δ% -0.06, ΔECE -0.0006)
    (a) asym=-0.01                                         65.30% | 0.0063   (Δ% +0.05, ΔECE -0.0003)  PASS (marginal)
    (a) asym=+0.01                                         65.01% | 0.0065   (Δ% -0.24, ΔECE -0.0001)
    (a) asym=+0.02                                         64.64% | 0.0063   (Δ% -0.61, ΔECE -0.0003)
    (b) vel=-0.02                                          65.31% | 0.0070   (Δ% +0.06, ΔECE +0.0004)  ECE worse
    (b) vel=-0.01                                          65.28% | 0.0069   (Δ% +0.03, ΔECE +0.0003)  ECE worse
    (b) vel=+0.01/+0.02                                    ~65.2  | ~0.0066   (Δ% ≤0)
  → ST: FLAT. Asymmetry buys at most +0.05% (noise); velocity is null/ECE-worse. Clean CONFIRMATION
    that the ST static term is already sufficient — exactly what the refuted-list note predicts.

  HV (1296 races, 14823 runners; 27.1% below-peak)   base 59.69% | 0.0046
    (a) asym=-0.05                                         59.72% | 0.0055   (Δ% +0.03, ΔECE +0.0009)  ECE worse
    (a) asym=-0.04                                         60.33% | 0.0052   (Δ% +0.64, ΔECE +0.0006)  ECE worse
    (a) asym=-0.035                                        60.52% | 0.0045   (Δ% +0.84, ΔECE -0.0002)  PASS
    (a) asym=-0.03                                         60.63% | 0.0039   (Δ% +0.94, ΔECE -0.0007)  PASS (best ECE)
    (a) asym=-0.025                                        60.64% | 0.0047   (Δ% +0.95, ΔECE +0.0000)  PASS (best %)
    (a) asym=-0.02                                         60.61% | 0.0056   (Δ% +0.92, ΔECE +0.0010)  ECE worse
    (a) asym=-0.01                                         60.28% | 0.0047   (Δ% +0.59, ΔECE +0.0001)  ECE worse
    (a) asym=+0.01                                         58.93% | 0.0049   (Δ% -0.76, ΔECE +0.0002)  ECE worse
    (b) vel=-0.02                                          60.06% | 0.0045   (Δ% +0.37, ΔECE -0.0001)  PASS (weaker)
    (b) vel=-0.01                                          59.87% | 0.0045   (Δ% +0.18, ΔECE -0.0001)  PASS (weaker)
    (b) vel=+0.01/+0.02                                    <base  | worse                              wrong sign

  → HV: a real but THIN signal. Steeper below-peak coupling (asym<0 = MORE reversion-up credit) lifts
    %market ~+0.9% with ECE neutral-to-better at asym∈[-0.035,-0.025]. SIGN is right and consistent
    (negative asym = credit the below-peak horse as reverting UP). BUT the ECE is JUMPY across the grid
    (−0.0007 at −0.03, but +0.0010 at −0.02 and +0.0009 at −0.05), i.e. the dual-gate PASS sits on a
    narrow ECE knife-edge near the ±0.0003–0.0005 sampling floor.

THE DECISIVE CONTROL — is it ASYMMETRY, or just "HV's static gain is under-tuned?"
rtg_peakgap is ≤0, so a steeper below-peak slope is nearly a steeper OVERALL slope (73% of HV runners
are at-peak=0 and carry no gap signal). So I swept the SYMMETRIC plain `PEAKGAP_GAIN` at HV:
    PEAKGAP_GAIN=-0.025 (baked)   59.7% | 0.0046
    PEAKGAP_GAIN=-0.035           60.3% | (—)
    PEAKGAP_GAIN=-0.045           60.6% | 0.0056   ← same +0.9% %market as the asym term…
    PEAKGAP_GAIN=-0.055           60.6% | (—)
  A SYMMETRIC larger HV gain reaches the SAME +0.9% %market — BUT at ECE 0.0056 (+0.0010, FAILS the
  dual gate). The asymmetry term reaches the same %market at ECE 0.0039 (−0.0007, PASSES). So the
  asymmetry is NOT a different %market story — the *market lift is just "HV peakgap is under-coupled"* —
  but the asymmetric STRUCTURE achieves that lift with materially BETTER calibration than steepening
  everything symmetrically. Reading: HV genuinely wants more below-peak (reversion-up) coupling, and
  loading it ASYMMETRICALLY (not also steepening the at-peak/regression-down side) is what keeps the
  calibration clean. That is a (weak) vindication of the mean-reversion structure over the static term
  AT HV — but it is thin and ECE-fragile.

RECOMMENDATION (T3).
- ST: **DON'T-BAKE.** Flat for both asymmetry and velocity — the static term is sufficient (confirms
  the refuted-list note: only the LANDMARK converts; no asymmetry/velocity structure to add at ST).
- HV: **DON'T-BAKE (lean), CANDIDATE-IF-PUSHED.** `PEAKGAP_ASYM_GAIN_BY_VENUE["HV"] = -0.03` passes the
  paired dual gate (+0.94% %market, ECE −0.0007) and beats a symmetric retune on calibration — a real
  but MARGINAL, ECE-knife-edge result. I would NOT bake it on the strength of one ECE-jumpy grid: the
  +0.9% is mostly "HV static peakgap is under-tuned" and the asymmetry's only clean advantage is the
  ECE, which is grid-fragile. Defer to the orchestrator; if anything, the cleaner first move is to
  re-examine the HV STATIC PEAKGAP_GAIN tune (it may simply want to be slightly larger with an ECE
  guard), and only adopt the asymmetric form if it survives a finer/independent-seed ECE check.
- Production stays NO-OP (both knobs 0.0 at both venues); constants baked-defaults unchanged.

WHAT TO TRY NEXT (T3).
1. Decompose at-peak vs below-peak as TWO independent gains (full two-slope fit) to measure the
   regression-DOWN (at/near-peak, over-bet) side directly — here it's folded into the static term;
   the asym knob only touches the below-peak side.
2. Residualise rtg_peakgap against recency-form before coupling, to be sure the HV below-peak lift is
   the LANDMARK and not form leaking back in (the refuted-list worry).
3. An independent-seed ECE replication of HV asym=-0.03 before any bake — the dual-gate PASS is on a
   ±0.0005 ECE margin.

=============================================================================================
## T4 — unify the allometric exponent b across venues

HYPOTHESIS (THEORY §3d allometric mass + constants note "venue-invariant physiology, ST-only by
leverage"). `ALLOMETRIC_B_BY_VENUE={ST:1.1, HV:1.0}` couples declared mass to v_c via v_c ∝ m^(b−1).
The note claims the physiological exponent b is ONE number and the venue split is an ARTIFACT of where
mass-leverage bites (HV's other channels already price mass), not real physiology. So a SINGLE unified
b should hold both venues — a parsimony win (one fewer free constant). Test candidate unified b ∈ {1.0,
1.05, 1.1} applied to BOTH venues vs the {ST:1.1, HV:1.0} baseline.

CONSTRUCTION. No new code — `fit.fit_predict_traffic(allometric_b=)` already overrides the per-venue
dict when not None (mass is leak-free: declared weight is published pre-race; field-relative to the
geometric-mean mass on the wald path). `studies/allometric_b_probe.py` runs the paired gate per venue:
base = `allometric_b=None` (the per-venue split), candidate = a single b for both.

THE GATE (paired fixed-seed, FULL scale; %market | ECE):

  ST (baked b=1.1)    base (split) 65.25% | 0.0066
    unified b=1.0                  64.75% | 0.0071   (Δ% -0.50, ΔECE +0.0005)  WORSE both
    unified b=1.05                 65.13% | 0.0073   (Δ% -0.12, ΔECE +0.0007)  WORSE both
    unified b=1.1                  65.25% | 0.0066   (== base; b=1.1 IS ST's value)

  HV (baked b=1.0)    base (split) 59.69% | 0.0046
    unified b=1.0                  59.69% | 0.0046   (== base; b=1.0 IS HV's value)
    unified b=1.05                 59.97% | 0.0047   (Δ% +0.28, ΔECE +0.0000)  %market up, ECE ~flat
    unified b=1.1                  60.02% | 0.0055   (Δ% +0.33, ΔECE +0.0009)  %market up but ECE WORSE

THE VERDICT — the split is LOAD-BEARING; do NOT unify.
- To unify at a SINGLE b you must pick ONE value for both venues:
  - b=1.0 unified → ST loses −0.50% %market (ECE +0.0005). FAILS at ST.
  - b=1.1 unified → HV gains +0.33% %market BUT ECE worsens +0.0009. FAILS the DUAL gate at HV.
  - b=1.05 unified → ST −0.12%/ECE worse AND HV ECE essentially flat; a compromise that improves
    NEITHER venue on the dual gate (ST worse both metrics).
- There is NO single b that holds both venues on the dual %market+ECE gate. b=1.1 helps HV's %market
  but breaks HV's (tighter, 0.0046) calibration; b=1.0 protects HV's calibration but costs ST. The
  venue split is doing real work.

INTERESTING NUANCE: HV's %market actually WANTS more mass (b=1.1: +0.33%), echoing the ST direction —
consistent with the "venue-invariant physiology" idea at the level of the DRIFT. What kills unification
is HV CALIBRATION: pushing HV mass to b=1.1 over-disperses the win prob (ECE +0.0009), the same
"variance lever over-disperses the already-tighter HV field" failure mode seen for the per-horse-D
knobs (queue #28 FIGVAR, DRAW_VAR_GAIN). So the constants-note story is HALF right: the *drift*
physiology may indeed be venue-invariant (HV's mean wants b>1 too), but the venue split is NOT a pure
leverage artifact you can drop — it is the calibration guard that keeps HV from over-dispersing. The b
exponent is doing double duty (drift physiology + an implicit HV dispersion control), and the two pull
apart at HV.

RECOMMENDATION (T4).
- **DON'T-BAKE a unified b. Keep `ALLOMETRIC_B_BY_VENUE={ST:1.1, HV:1.0}`.** The split is load-bearing:
  no single b passes the dual gate at both venues. Documented above with the gate numbers.
- Parsimony is NOT free here — unifying costs ≥−0.5% at ST (b=1.0) or +0.0009 ECE at HV (b=1.1).
- WHAT TO TRY NEXT if the parsimony win is still wanted: absorb HV's mass-leverage shortfall elsewhere
  (field-size or distance leverage, per the note's hypothesis) and RE-TEST a unified b=1.1 — i.e. give
  HV a mass×field-size or mass×distance term so the venue difference is explained by leverage, then see
  if b can unify WITHOUT the HV ECE break. That is the real test of "venue-invariant physiology, split
  is leverage" and is the only path to the parsimony win; the flat-unify tested here rejects it.

=============================================================================================
LEAK-FREE CAVEATS (both sub-parts). All features are the EXISTING strictly-prior as-of columns
(`rtg_peakgap`/`rtg_vel` via backward `allow_exact_matches=False`; declared mass is a pre-race public
boundary condition). The new knobs only re-weight those columns — no new join, no new data path. The
EB energetics population is fit on each walk-forward fold's TRAIN window only (unchanged). All gates are
paired fixed-seed (PYTHONHASHSEED=0, shared per-race MC seeds), full scale, both venues; verified
deterministic on re-run. No bakes performed — both new knobs default 0.0 (production untouched); the
orchestrator serializes any bake.
