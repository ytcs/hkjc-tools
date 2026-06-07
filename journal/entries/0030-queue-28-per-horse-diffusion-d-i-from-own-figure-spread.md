---
title: "Queue #28: per-horse D_i from a horse's own figure-spread — converts ST (+0.19% +ECE, BAKE), fails HV ECE gate (DON'T-BAKE)"
id: 0030-queue-28-per-horse-diffusion-d-i-from-own-figure-spread
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: draft
thread: 
tags: ["theory", "variance", "diffusion", "D-heterogeneity", "queue-28", "figure-spread", "speed-figure", "conversion-test", "third-knob", "EVT", "field-size", "draft"]
refs: {"feature": "features.speed_figure_dispersion_table", "knob": "fit.figvar_gain", "constant": "FIGVAR_GAIN / FIGVAR_GAIN_BY_VENUE", "study": "studies/per_horse_diffusion_probe.py", "ST_base_pct": 0.6525, "ST_best_gain": 0.15, "ST_best_pct": 0.6544, "ST_d_pct": 0.0019, "ST_d_ece": -0.0008, "HV_base_pct": 0.5969, "HV_gain02_pct": 0.6048, "HV_gain02_d_ece": 0.0009, "queue_item": 28}
created: 2026-06-06T03:30:00+00:00
updated: 2026-06-06T03:30:00+00:00
---
HYPOTHESIS (THEORY §3d/§3e — the D-heterogeneity knob, the variance third knob applied PER HORSE).
Every banked signal moves the MEAN drift; the only variance lever so far (DRAW_VAR_GAIN) is a
geometry/field effect, not the horse's own consistency. A horse whose past speed figures are tightly
clustered is a low-variance runner; one whose figures swing wildly is genuinely less predictable, so
its diffusion D_i should scale with the spread of its OWN prior figures. Because the winner is an
order statistic, variance is as load-bearing as the mean — a high-D horse upsets more than its drift
implies. Queue #29 (the CV∝1/√L study) established D is CONSTANT in L on turf, so this is
HETEROGENEITY across horses around that stable population baseline, not a distance effect.

CONSTRUCTION.
- Feature `features.speed_figure_dispersion_table` (mirrors `speed_figure_table`): per-run figure
  `sf = v_run − par(going,dist)` (m/s vs a condition standard), then the per-horse trait = the
  strictly-prior STANDARD DEVIATION of `sf` over the horse's PRIOR runs (expanding std shifted by one,
  computed Welford-style from lagged running sums of `sf` and `sf²`). SHRINKAGE toward the population
  dispersion σ_pop (panel mean of the per-horse stds, ≈0.147 m/s) with weight `n/(n+3)`; a horse with
  ≤1 prior run gets σ_pop exactly (neutral). Population: mean 0.148, IQR 0.120–0.159, fat tail to 2.2
  m/s — real per-horse heterogeneity in figure-consistency.
- Knob `fit.fit_predict_traffic(figvar_gain=)` wired into the wald variance beside `draw_var_gain`:
  `D_i = D·(1 + gain·(disp_i/disp̄ − 1))`, field-MEAN-normalised so the population-average horse keeps
  the baseline D — a pure REDISTRIBUTION of variance, conserving the field-mean D (so global
  calibration is preserved). Clamped ≥0.05·D. Swept param, default `FIGVAR_GAIN=0.0` ⇒ NO-OP. Baked
  default OFF at BOTH venues (`FIGVAR_GAIN_BY_VENUE={"ST":0,"HV":0}`) — production untouched.

LEAK-FREE PROOF. Direct check (PYTHONHASHSEED=0): for a 56-run career horse, the emitted `fig_disp`
at each run equals the hand-computed shrunk std over EXACTLY that run's strictly-prior figures (match
to 1e-6; n≤1 → σ_pop). `par` is a race-condition constant (not the horse's own outcome) ⇒ leak-free,
same discipline as the baked speed_figure / collateral channels. The feature joins on
(date,venue,race_no,runner); the population σ_pop is computed over the full panel but is a dispersion
CONSTANT (the shrink target), not a per-horse outcome — identical to how `par` uses the full panel.
Population component carries no per-fold future leakage into the per-horse std (which is strictly
prior). The gate below trains the energetics population on each walk-forward fold's TRAIN window only.

THE GATE (paired fixed-seed, PYTHONHASHSEED=0, FULL scale, both venues; %market | ECE):

  ST (1952 races, 24433 runners)            base 65.25% | 0.0066
    figvar_gain=0.10                              65.42% | 0.0064   (Δ% +0.17, ΔECE −0.0002)  PASS
    figvar_gain=0.15                              65.44% | 0.0058   (Δ% +0.19, ΔECE −0.0008)  PASS *best*
    figvar_gain=0.20                              65.42% | 0.0058   (Δ% +0.17, ΔECE −0.0008)  PASS
    figvar_gain=0.25                              65.38% | 0.0058   (Δ% +0.13, ΔECE −0.0008)  PASS
    figvar_gain=0.30                              65.28% | 0.0059   (Δ% +0.03, ΔECE −0.0007)  PASS
    figvar_gain=0.40                              65.04% | 0.0052   (Δ% −0.21, ΔECE −0.0014)  %market down
    figvar_gain=1.00                              61.04% | 0.0061   (Δ% −4.21)                overshoot
    figvar_gain=−0.40                             62.08% | 0.0065   (Δ% −3.17)                wrong sign craters

  HV (1296 races, 14823 runners)            base 59.69% | 0.0046
    figvar_gain=0.05                              59.99% | 0.0050   (Δ% +0.30, ΔECE +0.0004)  ECE WORSE
    figvar_gain=0.10                              60.23% | 0.0049   (Δ% +0.54, ΔECE +0.0003)  ECE WORSE
    figvar_gain=0.15                              60.43% | 0.0049   (Δ% +0.74, ΔECE +0.0003)  ECE WORSE
    figvar_gain=0.20                              60.48% | 0.0056   (Δ% +0.79, ΔECE +0.0009)  ECE WORSE
    figvar_gain=0.40                              60.20% | 0.0055   (Δ% +0.51, ΔECE +0.0009)  ECE WORSE
    figvar_gain=−0.40                             54.03% | 0.0043   (Δ% −5.66)                wrong sign craters

EVT FIELD-SIZE BREAKOUT (ST, split at field_size 12; band by SCORING, gain=0.15; %market | ECE):
    BIG  (≥12, ~20440 runners)   base 66.06% | 0.0064   cand 66.14% | 0.0064   (Δ% +0.09, ΔECE ~0)
    SMALL(<12,  ~3747 runners)   base 62.00% | 0.0183   cand 62.37% | 0.0175   (Δ% +0.36, ΔECE −0.0008)
  The ST gain is broad-based, not confined to big fields. Naively the EVT order-statistic argument
  (§3e) predicts D-heterogeneity pays MOST in big fields; here the %market lift is actually a touch
  larger in ST SMALL fields, though they start from a much worse-calibrated base (ECE 0.0183 vs 0.0064
  — small ST fields are the under-served calibration cell), so the term is partly fixing that. BIG
  fields still improve on both metrics. Reading: the per-horse-D redistribution helps ST broadly, with
  the cleanest calibration repair in the poorly-calibrated small-ST cell — consistent with "variance
  is load-bearing in the order statistic" without being a pure big-field effect. (Caveat: the band is
  by scoring only; the fit trains on full fields. A field-size INTERACTION of the gain is untested.)

INTERPRETATION.
- The MECHANISM is real and the SIGN is unambiguous at both venues: positive gain (erratic horses get
  MORE D_i) helps %market; negative gain craters both. The market does NOT already fully price
  figure-consistency into the win prob — there is a recoverable per-horse variance signal.
- ST: a clean, STABLE interior optimum — gains 0.10–0.30 all pass the DUAL gate (both %market and ECE
  improve), a plateau peaking ≈0.15 (+0.19% %market, ECE −0.0008). Small in %market but a real, paired,
  full-scale dual-metric improvement on a broad plateau — the signature of a genuine lever, not noise.
- HV: FAILS the dual gate at every positive gain. %market wants the term (up to +0.8%) but ECE
  degrades monotonically (+0.0003 to +0.0009) even at gain=0.05. HV's smaller fields + tighter handicap
  blanket finishes mean the population already sits at a lower residual spread (queue #29: HV CV below
  ST at every shared distance), so redistributing variance per-horse over-disperses HV and breaks its
  (already tighter, ECE 0.0046) calibration. This is the same ST-converts / HV-null venue split the EVT
  field-size argument predicts for variance levers (cf. DRAW_VAR_GAIN, ST-only), here sharpened to
  "HV actively mis-calibrates."

RECOMMENDATION.
- ST: **BAKE-CANDIDATE** `FIGVAR_GAIN_BY_VENUE["ST"] = 0.15` (range 0.10–0.20 equivalent; passes the
  paired full-scale dual gate). Defer to the orchestrator (serializes all bakes). Note it stacks ON TOP
  of the live ST base (65.25%) which already carries DRAW_VAR_GAIN=0.5 — the two D-knobs (geometry/draw
  and own-figure-spread) co-exist and the figure-spread term still adds, so they are not redundant.
- HV: **DON'T-BAKE.** Keep `FIGVAR_GAIN_BY_VENUE["HV"] = 0.0`. The dual gate fails on ECE at all gains.
- Production stays NO-OP as shipped (both venues 0.0); nothing in constants.py baked-defaults changed.

WHAT TO TRY NEXT if a fuller variance lever is wanted.
1. Couple to the figure-residual-AFTER-class/rating (regress sf on rtg-band, take the residual std) so
   the dispersion is not confounded with a horse simply moving up/down in class — the §spec's suggested
   refinement; may sharpen the ST gain and possibly rescue an HV variant.
2. Field-size INTERACTION of the gain (figvar_gain × field_size) rather than the by-scoring band — to
   test the EVT prediction properly inside the fit, and possibly let HV's small fields opt out.
3. AWT-specific D_i: queue #29 found AWT carries a real distance-independent variance FLOOR (σ0≈0.46s);
   per-horse D_i may pay most on AWT (big fields + intrinsic extra diffusion) — untested here (the
   per-venue ST/HV split folds AWT into ST's surface branch via awt_diff_mult, but a dedicated AWT gain
   is unexplored).
