---
title: "Queue #31: START-phase gate-break trait — drift channel DON'T-BAKE (priced/collinear); D channel ST-sprint BOTH↑ but sub-noise, HV ECE-worse"
id: 0030-queue-31-gate-break-trait-drift-channel-don-t-bak
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread:
tags: ["theory", "phase-structure", "queue-31", "gate-break", "start-phase", "draw-residual", "diffusion", "D-knob", "collinearity", "dont-bake", "draft"]
refs: {"study": "studies/gate_break_probe.py", "feature": "src/hkjc/model/features.py:gate_break_table", "fit": "src/hkjc/model/fit.py", "constants": "src/hkjc/model/constants.py", "queue_item": 31, "theory": "THEORY.md §3e", "draw_earlypos_corr": 0.06, "gate_break_earlypos_corr": 0.998}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — orchestrator serializes any bake + MODEL_STATE update. Production left a NO-OP
(all GATE_BREAK_* gains default 0.0). Recommendation: DON'T-BAKE both channels as-is.

## HYPOTHESIS (THEORY §3e — the START phase as a distinct regime)
The homogeneous Wald law treats the trip as one drift-diffusion gun-to-wire. The START is a
distinct phase: a horse that breaks well and secures early position gains a real, repeatable
edge — it should reach v_c sooner (shorter effective L / higher early drift) AND a clean break
is low-variance while a troubled break injects variance D. This should matter MOST where the
start is a bigger fraction of L (sprints) and where traffic at the break is worst (big fields).
The trait must be the horse's BREAK SKILL, not a draw restatement — so early position
RESIDUALISED against draw.

## CONSTRUCTION (leak-free) — features.gate_break_table(run_policy, wpq)
Per run: gate_break_run = early_pos_norm − E[early_pos_norm | draw_norm], draw_norm =
(draw−1)/(field_size−1), and E[·|draw_norm] = a + b·draw_norm a single population affine fit
over the full panel. Then the strictly-prior EXPANDING MEAN per horse (sorted by date) =
gate_break, with prior-run COUNT gate_break_n for EB shrinkage. In fit.py the trait is
shrunk n/(n+3) toward 0 (= field-average break) and within-race centred.

LEAK-FREE PROOF: (1) per-horse trait is the expanding mean of STRICTLY PRIOR runs only (shift
discipline, like form/early_pos/trajectory tables). (2) The draw→position affine is a population
constant over a RANDOM, pre-race-public boundary condition (draw) — not the horse's outcome —
same leak status as the trial `par` and the trajectory within-race z-score. (3) Mechanical
check: `assert_no_lookahead` on the full gate-break fit path (both channels active) returns max
prediction drift = 0.0 (predictions invariant to corrupting test outcomes). (4) Population EB
fit is train-only per walk-forward fold (backtest_st loader, unchanged discipline).

## THE COLLINEARITY FINDING (the key result, §3e-predicted)
Draw explains ALMOST NONE of HKJC rank-normalised early position: corr(early_pos_norm,
draw_norm) = 0.06; the per-band draw→early-pos slope is ~+0.03–0.08 (and weakly POSITIVE — wide
horses sit fractionally more forward, a style/selection artifact, not a break penalty). The bend
ground-loss, NOT the break, is the dominant draw mechanism in this data. Consequence:
residualising against draw removes essentially nothing, so gate_break ≈ centred early_pos
(corr(gate_break_run, early_pos_norm) = 0.998). The break trait is therefore HIGHLY COLLINEAR
with the existing positional-fade `early_pos` term and with the form channel — it can only earn
its keep through a DIFFERENT mechanism/channel (early-phase drift or D), not as new ability info.

## THE PAIRED GATE (PYTHONHASHSEED=0, full-scale walk-forward, shared per-race MC seeds)
Channel (a) — effective-L / DRIFT bonus (good break ⇒ shorter L, scaled by start_frac=min(1,
1200/L) so it bites in sprints):

  ST FULL    base 65.25% | ECE 0.0066 →  gain 0.05/0.1/0.2/0.4 = 64.96/64.14/61.32/51.03%
             (Δ% −0.29/−1.11/−3.93/−14.22), ECE all WORSE. Negative gains also cost %market
             (−0.15/−0.84/−3.65) though they IMPROVE ECE (0.0066→0.0050 at −0.1).
  ST sprint  base 69.12% → all gains LOWER %market (−0.12…−16.24). ST route same.
  HV FULL    base 59.69% → gain 0.05/0.1 = 58.58/56.75% (Δ −1.11/−2.94). Decisively negative.
  ⇒ DRIFT channel DON'T-BAKE on BOTH venues, both signs (monotone away from base in %market).

Channel (b) — DIFFUSION D modulation (good break ⇒ lower D / more predictable):

  ST FULL    base 65.25% | ECE 0.0066 →  gain 0.1/0.25/0.5 = 65.31/65.38/65.34% | 0.0064/0.0062/
             0.0058  (Δ% +0.06/+0.13/+0.09, ΔECE −0.0001/−0.0004/−0.0008) — BOTH↑ at every gain.
  ST sprint  base 69.12% | 0.0087 →  gain 0.5 = 69.34% | 0.0086  (Δ% +0.22, ΔECE −0.00004) BOTH↑.
  ST route   base 61.73% | 0.0080 →  +0.1–0.13% %market but ECE WORSENS (+0.0004) — fails BOTH.
  HV FULL    base 59.69% | 0.0046 →  gain 0.25/0.5 = 60.01/60.08% | 0.0053/0.0052  (Δ% +0.32/+0.39,
             ΔECE +0.0007/+0.0006) — %market UP but ECE WORSE ⇒ fails the BOTH gate. Negative gain
             (−0.25) improves ECE (−0.0007) but costs %market (−0.31).
  ⇒ D channel: ST passes BOTH gates (sprint-led, +0.13% full / +0.22% sprint, ECE flat-to-better);
    HV FAILS (ECE worsens). The split is exactly the §3e/EVT field-size prediction (the D-knob
    pays in big-field ST, null/wrong at small-field HV — same signature as the baked DRAW_VAR_GAIN
    ST-only split, and consistent with journal 0029: HV has the lowest CV already).

## INTERPRETATION
- §3e gets a partial theory-confirms-data: the START-phase signal lives on the VARIANCE knob
  (clean break = lower D), NOT the drift — putting it on drift double-counts the already-priced
  forward-position/form channel and craters %market. And it pays where §3e says (ST sprints /
  big fields), null at HV. The drift no-op + D-sprint-positive pattern is the §3e
  "mean→drift, temperament→D" decomposition resolving in favour of the D half.
- BUT the ST D gain is SUB-NOISE: +0.13% full / +0.22% sprint is well inside the ±0.3–0.5pp MC
  band the methodology caveat names (and quick_eval over-states ~4×; this IS the full-scale gate,
  so it is not inflated — it is simply tiny). ECE moves are 1e-4. Not a confident BAKE.

## RECOMMENDATION: DON'T-BAKE (both channels), keep the axis open.
- Drift channel: REJECT this structure (priced/collinear — the break adds nothing as new drift).
- D channel: ST-only is directionally right and §3e-consistent but the magnitude is sub-noise;
  not worth a bake on its own. HV explicitly fails (ECE). Park as "real mechanism, marginal
  leverage on win" pending a sharper trait.

## WHAT TO TRY NEXT (if the axis is revisited)
1. The trait is a near-restatement of early_pos because DRAW barely moves HKJC early position.
   A sharper break signal needs the actual GATE-to-first-call position CHANGE, not the settled
   early level — i.e. the §3e "first-call rank minus draw" using the raw RunningPosition call
   points (results_rich.RunningPosition) rather than run_policy's already-settled early_pos_norm.
   That would decouple it from the positional-fade term.
2. break×FIELD-SIZE interaction on the D channel ONLY, ST big fields (the EVT corollary): the
   GATE_BREAK_FS_GAIN hook exists but was left at 0.0 here — sweep it on ST sprints with large N.
3. break → D only on AWT (journal 0029: AWT carries a real distance-independent variance FLOOR =
   the "bad-break costs the same seconds at any L" mechanism §3e names) — the D channel's natural
   home if any.
4. interaction with the positional-fade W′ term (POSFADE_W_GAIN): the break is the EARLY control
   decision whose cost is paid LATE (the §3e intertemporal coupling) — a break×pos_fade cell, not
   a standalone gate trait.

## ARTIFACTS (production untouched — all gains default 0.0)
- features.gate_break_table — leak-free draw-residualised break trait + EB count.
- fit.fit_predict_traffic gate_break_gain (drift, start_frac×field-size scaled) +
  gate_break_d_gain (diffusion) + gate_break_fs_gain — all NO-OP at 0.0.
- constants GATE_BREAK_GAIN[_BY_VENUE]/FS_GAIN/D_GAIN/LREF — all 0.0.
- studies/gate_break_probe.py — paired fixed-seed probe, ST distance-band breakout.
