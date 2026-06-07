---
title: "v_c latent-factor audit: the channel stack is ONE over-measured ability factor + a sparse long tail — COLLAPSE, not multi-attribute reformulation"
id: 0037-vc-latent-factor-audit-catch-all-is-one-ability-factor-collapse
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: draft
thread:
tags: ["v_c", "factor-analysis", "pca", "catch-all", "ability-latent", "measurement-model", "collapse", "redundancy", "collinearity", "retrodiction", "audit", "architecture", "context-leak", "jockey", "trainer", "queue-31", "queue-32", "queue-29", "queue-37"]
refs: {"study": "studies/vc_factor_audit.py", "out": "studies/out/vc_factor_audit.txt", "fit": "src/hkjc/model/fit.py", "features": "src/hkjc/model/features.py", "constants": "src/hkjc/model/constants.py", "loader": "studies/backtest_st.py:load", "theory": "docs/THEORY.md#3d"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
QUESTION (architectural, descriptive — no code touched, nothing baked). The production drift v_c_eff
has accreted ~10 channels (fit.py vc.append, lines ~324-331): rating anchor, sectional-v_c EB residual
`u_vc`, recency form, jockey, speed figure, barrier-trial (fig + fusion v_c), training-gallop v_c,
finishing-sectional v_c, collateral-form, rating peak-gap, trainer-hot, positional-fade/early-pos,
gate-break, allometric mass. Is v_c a CATCH-ALL — one over-measured ability latent dressed as physics —
or genuinely MULTI-ATTRIBUTE (distinct latents warranting a multi-attribute generative reformulation)?

METHOD. `studies/vc_factor_audit.py` rebuilds the per-runner STRICTLY-PRIOR channel matrix from
`backtest_st.load("ST")` / `load("HV")` — the SAME columns fit.py consumes — adds `u_vc` (own v_c minus
the rating-band fixed pop mean, the term fit.py shrinks), `logmass` (the allometric driver), and the
finishing-sectional v_c (raw + kick, queue #32) which load() doesn't join. 16 channels. Within-RACE
z-score (the field-relative view the model acts on; nulls→0=field-average, the fit fill convention).
Then: full pairwise corr; correlation-PCA (SVD); phys-vs-context split; drop-one ΔR² on the winner
(LPM); retrodiction of the recent collinearity deaths. ST 1952 races / 24,433 runners; HV 1296 /
14,823. (Descriptive audit — population fit on the full panel, not a leak-free backtest.)

== EFFECTIVE DIMENSIONALITY (the crux) ==
PCA variance ladder is FLAT — no dominant factor. PC1 explains only 24.7% (ST) / 25.6% (HV); to reach
80% of channel variance needs **9 of 16** factors at BOTH venues (90%: 11; 95%: 12-13). BUT the
unexplained 75% is NOT genuine attributes — it is per-channel MEASUREMENT NOISE. The signal lives in
PC1, and PC1 is one thing:

  ST  PC1 24.7%  PC2 13.1%  PC3 8.2%  PC4 7.6%  PC5 6.4% ...  (→80% at PC9, of 16)
  HV  PC1 25.6%  PC2 13.3%  PC3 8.5%  PC4 7.5%  PC5 6.4% ...  (→80% at PC9, of 16)

The high nominal dimensionality is the SIGNATURE of a catch-all measured by many NOISY instruments, not
of many attributes: every ability instrument loads PC1 with the same sign, and the rest of each
channel's variance is idiosyncratic noise that scatters across PC4-PC16 (one near-private factor each).

== PC1 = THE ABILITY FACTOR (loading table, identical across venues) ==
Channels loading PC1 (|load| sorted, ST / HV):
  finsec_vc    +0.41 / +0.43   ABILITY  (closing-phase read)
  finsec_kick  +0.39 / +0.41   ABILITY
  early_pos    +0.39 / +0.39   CONTEXT  (!! positional-fade)
  gate_break   +0.39 / +0.39   CONTEXT  (!! start-phase)
  form         +0.31 / +0.27   ABILITY
  collat_form  +0.31 / +0.34   ABILITY
  speedfig     +0.30 / +0.28   ABILITY
  u_vc         +0.23 / +0.22   ABILITY
  jky_sr       +0.18 / +0.16   CONTEXT
  --- everything below ~0.06 (effectively off PC1): ---
  trial_vc, trial_fig, gallop_vc, rtg_peakgap, rtg, trn_hot, logmass

So PC1 fuses form + speed figure + collateral + sectional-v_c residual + the finishing-sectional reads
into ONE factor — these are REDUNDANT MEASUREMENTS of a single ability latent. The off-PC1 channels are
NOT a second ability attribute; they are either (a) a different OBSERVATION TYPE of the same latent that
is too SPARSE/NOISY to align (trial_vc 99% present but trial-pair corr only 0.50; gallop_vc 33%
coverage, ~0 corr with everything — its own private PC5), or (b) genuinely DISTINCT but tiny axes:
  - rtg (the priced anchor) + rtg_peakgap own PC3 (loadings +0.55/+0.59 and +0.56/+0.64) — the
    rating/career-landmark axis sits APART from the form-cluster ability factor (rtg~rtg_peakgap corr
    0.22/0.30; rtg~form only 0.08/0.03). This is the one clean "distinct latent" and it is exactly the
    process-feature THEORY §3d/§3e predicted earns its keep by adding an ABSOLUTE career landmark.
  - gallop_vc owns PC5 alone (+0.80) — orthogonal because it is mostly missing + noisy, not a new
    attribute.
  - trn_hot owns its own factor (PC7 ST +0.82 / PC6 HV +0.68) — isolated intent overlay.
  - jky_sr scatters to PC8/PC9 — isolated context.
  - trial_fig/trial_vc share PC2/PC4 — the debut-form observation channel, distinct only because it is a
    DIFFERENT-INSTRUMENT read (trials), not a different latent.

== PHYS vs CONTEXT: context LEAKS into the ability factor ==
The category error is REAL and measured. early_pos (positional-fade) and gate_break load PC1 as HARD as
the ability reads (+0.39 each, ABOVE form's +0.31). RMS|load| on PC1: ability 0.236 vs **context 0.288**
(context loads PC1 MORE than ability, both venues). early_pos/gate_break are not horse-ability — they are
a STYLE/jockey-policy trait — yet they sit in the dominant ability factor. (Mechanically: early_pos
correlates 0.53/0.59 with finsec_vc — a horse that sits closer to the lead reads a higher closing
velocity; the v_c instrument is contaminated by running style.) jky_sr/trn_hot, by contrast, stand
APART (PC1 loads 0.18/0.05; they own PC7-PC9) — those two contextual channels do NOT leak into ability;
they are weak but clean separate signals. So the leak is specifically the POSITION/style channels
(early_pos, gate_break) into the v_c read, not jockey/trainer.

== INDEPENDENT CONTRIBUTION TO THE WINNER (drop-one ΔR², LPM full R²=0.088 ST / 0.074 HV) ==
LOAD-BEARING (independent work on the winner):
  u_vc        ΔR² .0323 / .0352   the sectional EB residual is BY FAR the most independent channel
  form        ΔR² .0068 / .0062
  jky_sr      ΔR² .0067 / .0056   (context, but clean independent work — earns its separate place)
  rtg         ΔR² .0015 / .0005
  trial_fig   ΔR² .0012 / .0001
  logmass     ΔR² .0007 / .0005
  rtg_peakgap ΔR² .0000 / .0015   (HV-only independent — the career landmark, HV's declining horses)
COLLINEAR BALLAST (≈0 independent ΔR², redundant given the rest):
  collat_form, speedfig, finsec_vc, finsec_kick, gallop_vc, trial_vc, trn_hot, early_pos, gate_break
  — all ΔR² ≤ 0.0007. These move the winner-fit by nothing once u_vc+form+jky are in.

Reconcile with production %-market: collat_form/peakgap DID convert on the BAKE test (HV +2.3/+2.4pp)
even though their drop-one ΔR² here is ~0. The reconciliation IS the thesis: they convert by SHARPENING
THE SAME ability estimate (a better measurement of PC1, esp. for the lightly-raced / opponent-strength
tails the form channel reads poorly), NOT by adding an independent attribute. That is precisely the
"measure the latent better" lever (PHILOSOPHY §8), and it is the COLLAPSE picture, not the
multi-attribute picture: more instruments on one latent, each paying via the order-statistic tails.

== RETRODICTION (validates the analysis — the factor structure PREDICTED the deaths) ==
R² of each recently-died feature regressed on all OTHER channels (high ⇒ already-spanned ⇒ predicted
dead). Result (ST / HV):
  finsec_vc (#32, raw)         R² 0.944 / 0.956  SPANNED → predicted dead ✓ (the closing read IS v_c)
  finsec_kick (#32, "ortho")   R² 0.939 / 0.952  SPANNED → predicted dead ✓ (NOT orthogonal — its corr
                                with finsec_vc is 0.95; the "kick" mode did NOT escape the span)
  gate_break (#31)             R² 0.997 / 0.998  SPANNED → predicted dead ✓ (corr 0.998 w/ early_pos —
                                draw-residualisation barely moves it; draw explains ~0 of early position)
  early_pos (#37, sit-and-kick R² 0.997 / 0.998  SPANNED → predicted dead ✓
    base)
  rtg_peakgap (#29 parent)     R² 0.072 / 0.112  INDEPENDENT → NOT spanned ✓ (the career-landmark axis
                                stands apart — and is exactly the one that converts)
A clean 5/5 retrodiction: every channel that died of collinearity projects into already-spanned factor
space; the ONE survivor (peakgap) is the one orthogonal axis. The factor structure predicts the
collinearity deaths and the conversions. This validates the method.

== VERDICT: (A) COLLAPSE ==
v_c is ONE over-measured ability latent, NOT a multi-attribute object. PC1 (24-26%) is the only shared
factor and it fuses form + speed-figure + collateral + sectional-v_c + finishing-sectional + (leaked)
position into a single dimension; the remaining channels are either redundant reads of it, a different
INSTRUMENT (trials/gallops) of the same latent, or one genuinely-distinct but tiny axis (rating/peakgap,
PC3). A multi-attribute generative reformulation is NOT warranted — there are not multiple horse-ability
latents hiding in the stack; there is one latent measured by ~6 collinear instruments plus a sparse
long tail. The honest fix is a MEASUREMENT MODEL:

1. COLLAPSE the redundant ability gain-stack (form, speedfig, collat_form, sectional u_vc, finsec) into
   ONE precision-weighted ability posterior — which fit.py's docstring ALREADY says it is a linearisation
   of (§3d "v_c is a precision-weighted posterior"). The audit confirms that is literally true: they are
   one factor. The win is to make the fusion EXACT (one latent, each instrument at its own precision)
   rather than a stack of separately-calibrated additive gains that re-measure the same factor and double-count.
   The instrument channels (trial_vc, gallop_vc) already fit this as separate-precision reads of the SAME
   latent — extend that fusion to absorb form/speedfig/collat/finsec; drop the redundant additive gains.
2. MOVE the non-physiological signals to a SEPARATE layer, OUT of v_c. early_pos/gate_break are a
   style/policy trait leaking into the ability read (PC1 leak; corr 0.53-0.63 with the v_c instruments) —
   a category error: a horse-attribute factor carrying a jockey/position-policy signal. They belong on the
   W′/D knob or a control layer (THEORY §3d already retired pos_fade off v_c onto W′; the audit says
   FINISH that move and pull gate_break with it — gate_break is just early_pos, so it should not be a
   separate v_c channel at all). jky_sr and trn_hot are weak but CLEAN separate signals (they don't leak
   into PC1, jky does independent winner-work ΔR²~.006) — keep them as a thin context layer, not folded
   into ability.
3. KEEP rtg/rtg_peakgap as the one genuinely-distinct axis (PC3) — the priced rating anchor + the
   career-landmark process feature. This is the only place the stack carries a second real dimension, and
   it is small (PC3 8%); it does NOT justify a multi-attribute REFORMULATION, just its own clean term.

The attributes the model actually carries: ONE ability latent v_c (≈ PC1: form/figure/collateral/
sectional, measured by races + trials + gallops + closing splits as separate-precision instruments), ONE
career-trajectory axis (rtg/peakgap, PC3), and a thin CONTEXT layer (jockey, trainer-intent) — plus a
STYLE/control trait (early-pos family) that must leave v_c. That is a one-factor ability + measurement
model + a small distinct rating axis — option (A), justified by the numbers: PC1≈25%, 9-factor
dimensionality that is noise not attributes, context out-loading ability on PC1, and a 5/5 retrodiction.

CAVEATS. Descriptive structure only (corr/PCA/LPM), not a conversion test — a COLLAPSE refactor must
still pass the paired fixed-seed full-scale %-market+ECE gate before any bake (this entry gates nothing).
The drop-one ΔR² ranks RELATIVE importance under a linear-probability winner model; it under-credits
channels that pay via the order-statistic tails (collat/peakgap convert despite ~0 ΔR²) — which is itself
the COLLAPSE evidence (they sharpen one latent, not add one). gallop_vc is only 33% covered; its isolation
is sparsity, not a distinct attribute. u_vc/logmass were fit on the full panel for this descriptive read.
