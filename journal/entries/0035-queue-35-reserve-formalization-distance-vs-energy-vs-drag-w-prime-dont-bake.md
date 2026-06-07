---
title: "Queue #35: Is the reserve W′ ENERGY or DISTANCE? Distance-W′ is SUFFICIENT — energy/drag reweighting FLAT-to-worse both venues; deployment-speed (sit-and-kick) interaction sub-noise, DON'T-BAKE"
id: 0035-queue-35-reserve-formalization-distance-vs-energy-vs-drag-w-prime-dont-bake
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread:
tags: ["theory", "energetics", "w-prime", "reserve", "critical-power", "hills-law", "queue-35", "distance-vs-energy", "drag", "deployment", "sit-and-kick", "pos-fade", "dont-bake", "draft"]
refs: {"study": "studies/reserve_formalization_probe.py", "fit": "src/hkjc/model/fit.py", "energetics": "src/hkjc/model/energetics.py", "constants": "src/hkjc/model/constants.py", "queue_item": 35, "theory": "THEORY.md §2.1/§3b/§3d/§3e", "panel": "data_cache/trajectory_dataset_full.parquet"}
created: 2026-06-05T00:00:00+00:00
updated: 2026-06-05T00:00:00+00:00
---
DRAFT — orchestrator serializes any bake + MODEL_STATE update. Production left a NO-OP
(W_PRIME_DEPLOY_GAIN=0.0; reserve recovery unchanged = the `dist` form). Recommendation:
distance-W′ is SUFFICIENT; DON'T-BAKE energy/drag reweighting OR the deployment interaction.

## HYPOTHESIS (the foundational choice behind the W′ knob)
Production recovers the reserve as a DISTANCE: W′ = Σ_k (v_k − v_c)^+ · t_k (metres run above
critical velocity v_c = min of the final two section velocities), fed into the drift as a
fractional boost v_eff = v_c·(1 + W′/L) — Hill's distance-reserve law. But ENERGY, not distance,
is the conserved quantity, and the metres a Joule of reserve buys FALL as deployment velocity
rises (drag ∝ v², reduced gross efficiency supramaximal). So the SAME excess-velocity integral
spent at a higher speed should count for LESS. Two facets of one hypothesis:
  (a) reweight the recovery: energy W′_E = Σ(v−v_c)^+·v·t, drag W′_D = Σ(v−v_c)^+·v²·t;
  (b) couple the W′/L leverage to HOW the reserve is deployed (front-runner early-burn vs
      closer late-kick) — the sit-and-kick interaction, at EQUAL recovered W′.

## CONSTRUCTION (leak-free, all NO-OP default — production untouched)
- THREE per-run recoveries from the SAME sectionals (`trajectory_dataset_full.parquet`, per-section
  velocity+time). The `dist` recovery reproduces the baked `energetics_dataset_full.w_prime`
  BIT-FOR-BIT (max abs diff 0.00e+00 on 24,433 ST + 14,823 HV runs — verified in-study). `energy`
  and `drag` rescaled to the `dist` population MEAN so the (1+W′/L) boost magnitude is matched —
  we test the WEIGHTING SHAPE, not a gain change. The swapped per-run w_prime flows through the
  EXISTING EB pooling (train-only ⇒ strictly-prior) + (1+W′/L) drift boost, so leak-freeness is
  IDENTICAL to the baseline; the mean-rescale is a single full-panel population constant (public
  sectionals, not any horse's own outcome — same leak status as a `par` constant).
- DEPLOYMENT interaction: additive fit param `w_prime_deploy_gain` (default 0.0 = exact baseline),
  boost = (W′/L)·(1 + gain·epos), epos = strictly-prior centred habitual early position (>0 front /
  early-deploy, <0 closer / late-deploy) — the SAME leak-free trail trait already feeding pos_fade.
- LEAK-FREE PROOF: `assert_no_lookahead` on the full fit path returns drift 0.00e+00 at both
  deploy_gain=0.0 and 0.5, BOTH venues (predictions invariant to corrupting test outcomes). The
  `dist` no-op is bit-for-bit. fit.py edits are purely additive (one default-0 param + a wp_boost
  array that is `wp/dist` when the gain is 0).

## THE PAIRED GATE (PYTHONHASHSEED=0, FULL-scale walk-forward, shared per-race MC seeds)

RESERVE RECOVERY — dist vs energy vs drag (% of market | ECE):
                  ST                         HV
  dist     65.25% | 0.0066  (baseline)   59.69% | 0.0046  (baseline)
  energy   65.23% | 0.0067  (−0.01%,ECE↑) 59.65% | 0.0046  (−0.03%, ECE −0.0001)
  drag     65.23% | 0.0065  (−0.02%,ECE↓) 59.63% | 0.0045  (−0.06%, ECE −0.0001)

  ⇒ Energy/drag reweighting is FLAT-to-slightly-WORSE on %market at BOTH venues. The %market deltas
    (−0.01 to −0.06pp) are deep inside the ±0.3–0.5pp MC band (and quick_eval over-states ~4× — this
    IS the full-scale gate, so not inflated, simply null). ECE moves are ±1e-4. NO gate win: energy
    and drag never improve %market, so the BOTH-must-improve gate fails on the %market leg alone.

RESERVE RECOVERY by ST distance band (W′/L bites hardest in sprints — the sharpest test):
  sprint   dist 69.1% | ene 69.1% | dra 69.2%   (Δ ≈ 0.0% all)
  mile     dist 65.2% | ene 65.2% | dra 65.2%   (Δ ≈ 0.0%)
  middle   dist 55.9% | ene 55.9% | dra 55.9%   (Δ ≈ 0.0%, drag ECE −0.001)
  staying  dist −71.0% (n tiny, ignore)
  ⇒ Even in SPRINTS, where the W′/L boost has the most leverage and the reweighting should bite
    hardest, the three recoveries are INDISTINGUISHABLE. The weighting shape carries no win-order signal.

DEPLOYMENT-SPEED interaction — the sit-and-kick test (W′ × early-position, dist recovery):
                       ST                          HV
  gain=0   (base)  65.25% | 0.0066            59.69% | 0.0046
  gain −0.5        65.22% | 0.0060 (ECE↓)     59.70% | 0.0054 (ECE↑)
  gain −0.25       65.26% | 0.0060  BOTH↑*    59.72% | 0.0051 (ECE↑)
  gain +0.25       65.22% | 0.0065            59.65% | 0.0040 (%↓, ECE↓)
  gain +0.5        65.19% | 0.0066 (ECE↑)     59.61% | 0.0040 (%↓, ECE↓)
  ⇒ The ONLY "BOTH↑" cell is ST gain=−0.25 (+0.01% / ECE −0.0006) — a +0.01pp %market move that is
    pure noise. And it does NOT replicate at HV (where gain=−0.25 IMPROVES %market +0.03% but ECE
    WORSENS +0.0005 — fails BOTH). The signs are inconsistent across venues: ST wants gain<0 for ECE,
    HV wants gain>0 for ECE but at a %market cost. No coherent, venue-shared, gate-passing setting.
    (*ST gain=−0.25 = a closer (epos<0) gets a LARGER reserve boost — directionally the "reserve
    deployed late/low-speed buys more metres" intuition, but the magnitude is sub-noise.)

## INTERPRETATION — why distance-W′ is SUFFICIENT (theory-confirms-the-current-structure)
1. The reweighting is FLAT because the three recoveries are near-monotone transforms of each other
   AFTER the per-horse EB POOLING and within-race normalisation. W′ enters the win order-statistic
   only through the RANKING of the (1+W′/L) boost within a race; energy/drag re-weight the excess
   integral but preserve that ranking almost exactly (a horse with a big distance-reserve also has a
   big energy/drag-reserve — they correlate ~0.9+). The win model never sees the absolute Joules, only
   the field-relative boost order, which the reweighting barely perturbs. So the DISTANCE form is not
   "wrong physics" — it is the right SUFFICIENT STATISTIC for the win order-statistic, and the
   energy/drag refinement is a distinction the win target cannot resolve.
2. This is consistent with the W_PRIME_GAIN=1.0 revert (journal 0005 / MODEL_STATE): W′ is largely a
   MARGIN effect, not a win-prob effect — recency-form already absorbed the reserve signal. A
   reformalization of a low-win-leverage knob inherits that low leverage. The reserve's win-relevant
   content is in WHO has more of it (the EB-pooled level), not in the velocity-weighting of how it was
   spent.
3. The sit-and-kick coupling is already (partially) captured: the front-runner early-burn vs closer
   late-kick asymmetry is the baked POSFADE_W_GAIN=−80.0 (positional-fade ON the W′ knob, journal
   §3d T1) — a front-runner's habitual early position DEBITS the reserve. The extra W′×epos
   MULTIPLICATIVE interaction tested here is collinear with that already-baked ADDITIVE term, which is
   why it adds nothing and flips sign across venues.

## RECOMMENDATION: distance-W′ is SUFFICIENT. DON'T-BAKE.
- Reserve recovery: KEEP the `dist` form. Energy and drag are physically motivated but the win
  order-statistic cannot tell them apart (flat at both venues, flat even in ST sprints). This
  EMPIRICALLY answers the foundational question: distance IS the correct formalization OF THE
  RESERVE FOR WIN — not because energy is wrong, but because the reweighting is a sufficient-statistic
  no-op under the field-relative boost ranking.
- Deployment interaction (`w_prime_deploy_gain`): DON'T-BAKE. The one BOTH↑ cell (ST −0.25) is
  sub-noise and venue-inconsistent; the mechanism it gropes for is already on the W′ knob as
  POSFADE_W_GAIN. Leave at 0.0.
- Production untouched (all defaults NO-OP). Mechanism not refuted as PHYSICS — refuted as a
  WIN-PROB LEVER (the energy/drag distinction may still matter for MARGIN / place-exotic pricing,
  where the absolute reserve magnitude, not just its rank, enters — untested here).

## CAVEATS
- This gates WIN only (the order-statistic / first-for-first). Energy-weighting could still help
  place/exotic margins where the whole finishing-order spacing matters — out of scope here.
- v_c per run = min of final two section velocities (the production estimator); an easy winner
  easing down under-reads v_c, inflating the excess integral identically for all three modes — the
  reweighting cannot fix a v_c-estimator bias, only re-weight a fixed (v−v_c)^+ profile.
- ST staying band (−71%) is a tiny-sample artifact (few staying races), ignored in the verdict.

## ARTIFACTS (production untouched — all gains default 0.0)
- studies/reserve_formalization_probe.py — paired fixed-seed probe: three recoveries from the
  trajectory panel (bit-for-bit dist check), ST distance-band breakout, deployment sweep, both venues.
- fit.fit_predict_traffic `w_prime_deploy_gain` (default 0.0) + `wp_boost` array (= wp/dist at gain 0).
- constants W_PRIME_DEPLOY_GAIN / _BY_VENUE — 0.0.
- src/hkjc/model/energetics.py — the `dist` recovery (unchanged); energy/drag live in the study only.
