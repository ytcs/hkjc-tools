---
title: "Finishing-time LAW: DIFFUSIVE-SUFFICIENT — no material jump/incident tail; AWT #33 floor is larger D, not jump mass"
id: 0032-finishing-time-law-diffusive-sufficient-no-material-jump-tail-awt-floor-is-larger-D
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: confirmed
thread:
tags: ["theory", "diffusion", "inverse-gaussian", "wald", "finishing-time", "tail", "jump", "incident", "birnbaum-saunders", "AWT", "D-floor", "queue-33-followup", "stochasticity", "theory-2.8", "theory-3e", "descriptive"]
refs: {"study": "studies/finishing_time_law.py", "plots": ["studies/out/ft_turf_vs_awt_tail.png", "studies/out/ft_uppertail_qq.png", "studies/out/finishing_time_densities.png", "studies/out/ft_trouble_rate.png"], "panel": "data_cache/results_rich.parquet", "prior": "0029-queue-33-cv-t-1-l-confirmed-on-turf-d-roughly-co", "n_runners": 36853, "n_bands": 23, "bulkig_p99_cov_median": 1.07, "gamma_p99_cov_median": 1.05, "verdict": "DIFFUSIVE-SUFFICIENT"}
created: 2026-06-06T03:30:00+00:00
updated: 2026-06-06T03:30:00+00:00
---
DESCRIPTIVE population study (NOT a predictive feature → no as-of/leak-free constraint; read-only on
production — touched no fit.py/constants.py/features.py). Question (THEORY §2.8, §3e; follow-up to #33
/0029): is the stochastic part of finishing time T **diffusive** (Inverse-Gaussian first-passage, thin
scaling tail) or does it carry a material **jump/incident** component (heavy asymmetric upper tail from
troubled trips; a distance-independent variance floor)? Panel: results_rich.parquet, 36,853 runners on
GOOD/GOOD-TO-FIRM, binned into 23 tight (surface×venue×distance×going) bands with n≥250. Per band: par =
within-band mean T; residual ε = T−par; all families fit by MLE to y = ε−min(ε)+δ with loc fixed at 0
(so the comparison is SHAPE/TAIL only, not location). Study + 4 plots: studies/finishing_time_law.py.

=== VERDICT: DIFFUSIVE-SUFFICIENT. No material jump/incident tail. ===
The stochastic finishing-residual is well-described by a single diffusive-family law. The empirical upper
tail is NOT materially heavier than a body-anchored diffusion predicts; the AWT variance floor of #33 is
**uniformly larger D**, NOT a jump component. An IG+jump (or Birnbaum–Saunders) production law is NOT
warranted by a tail mis-fit.

=== AIC/BIC: the production law (IG) is the WORST diffusive fit; Gamma wins; jump does NOT win. ===
Per-band AIC (ΔAIC vs best). Family of best AIC across 23 bands: **Gamma 21, LogNormal 1, IG+Jump 1.**
  • Inverse-Gaussian (the SIM=wald law) is the WORST or near-worst single family in EVERY band — ΔAIC
    vs Gamma of +100…+900. The IG SHAPE is a mediocre description of the residual BODY; Gamma/LogNormal
    fit the centre far better (lognorm typically 2nd).
  • The explicit 2-component **IG-core + shifted-exponential "trouble"** mixture does NOT win on AIC
    (loses to Gamma in 22/23) — but only because its CORE is IG, which is the handicap above, NOT
    because the data lacks a tail it could fill. AIC here measures the core, not the incident tail —
    exactly the trap the task warned of, so the verdict is weighted on TAIL CALIBRATION, below.
NOTE this is a refinement of the production picture: the wald LAW's MOMENTS are vindicated (#33: CV∝1/√L,
Var∝L), but its DISTRIBUTIONAL SHAPE is closer to Gamma/LogNormal than to strict IG. Both are diffusive,
positively-skewed, single-component laws — the win order-statistic is moment- and tail-driven, so this is
not a production defect, just the true shape of the noise.

=== TAIL CALIBRATION (the verdict weight): the diffusive tail is ≈calibrated; no jump room. ===
Upper-tail coverage = empirical exceedance / nominal at the fitted q-quantile (1.0 = calibrated;
>1 = empirical tail HEAVIER than model = under-cover; <1 = over-cover). Median across the 23 bands:
  family                         p95 ratio   p99 ratio
  BulkIG (IG fit to lower 92%)      0.85        1.07     ← the proper diffusive NULL
  Gamma (best body fit)             0.73        1.05
  InvGauss (full-sample 3-param)    0.35        0.45     ← OVER-covers: 3-param IG inflates its
                                                            own tail to absorb the body misfit
  IG+Jump (bulk core + add-on)      0.64        1.21     ← OVER-covers; the jump knob pins at its
                                                            bound and over-fattens, not a clean tail
KEY READ: once IG is anchored to the BODY (BulkIG = IG on the lower 92%), its OWN intrinsic positive
skew produces ≈ the right amount of upper-tail mass — median p99 coverage 1.07, i.e. only ~7% under at
the 99th pctile, NOT the explosive under-coverage a genuine jump world requires. Gamma is the same (1.05).
The unconstrained IG+jump (pi capped 0.30, tau floored 1.0s) hits BOTH bounds in ~17/23 bands and then
OVER-covers (p99 1.21, several bands 2.5–7.6×) — the optimizer is using the "trouble" mass to widen the
whole upper half, not to add a discrete rare spike. A real incident world would let pi settle at a small
interior value with a clean tail-coverage repair; instead the bound binds against a tail that is already
≈full → the data does NOT want a separable jump component. Upper-tail QQ (ft_uppertail_qq.png) confirms:
full-sample IG and Birnbaum–Saunders points sit ON the diagonal through the 80–99th pctiles in every band,
except 1–2 ISOLATED freak outliers per band (a 73 s / 32 s run, ~1 in 4,000 — a genuine pulled-up/eased
horse) that are real incidents but far too rare to move the law. Densities (finishing_time_densities.png)
show a smooth unimodal right-skewed body with NO secondary mode or shoulder on the right — no jump bump.

=== ASYMMETRY: positive, but only IG-grade — consistent with diffusive skew, not excess incidents. ===
Robust-standardized ε: skew>0 in all 23 bands; upper/lower tail ratio (z99/|z01|) = 1.2…2.5 (median ~1.8);
frac >3·MAD ≈ 1.3–3.1% vs frac <3·MAD ≈ 0%. The asymmetry is REAL (troubled trips only slow, never speed,
so the residual is one-sidedly right-heavy) but its MAGNITUDE is exactly what a body-anchored IG/Gamma
already emits (their p99 coverage ≈1.0). A few bands show inflated raw skew (ST-1200 30.8, HV-1800-GtF
12.0) driven entirely by the lone freak outliers above, not a systematic tail. So the asymmetry is the
diffusion's OWN skew plus a negligible-probability true-incident sprinkle — not a material jump mass.

=== AWT FLOOR RESOLUTION (the sharpest single discriminator — ties directly to #33). ===
#33 found AWT carries a distance-INDEPENDENT variance floor (σ0≈0.46 s, ~36% of AWT-sprint variance) and
left open whether it is larger D (uniform) or jump/kickback (upper-tail). ANSWER: **larger D (uniform),
NOT jump.** ft_turf_vs_awt_tail.png: after dividing out the scale (robust-standardize each band's ε by its
own median/MAD), the turf and AWT upper-tail SURVIVAL curves LIE ON TOP OF EACH OTHER out to z≈4 (AWT-1200
tracks turf-1200 almost exactly; AWT-1800 tracks turf-1800). AWT's extra variance is a UNIFORM scale
inflation (a fatter spread at every quantile → larger D), not a distinct heavy upper tail. The matched-band
spread table agrees: AWT-1200 up/lo asymmetry 1.36 vs turf-1200 1.31 (indistinguishable); AWT routes
(1650/1800) show a mildly larger up/lo (1.51–1.77 vs turf ~1.34) but the STANDARDIZED tail still overlaps
turf, so that is the longer-trip skew, not a surface-specific jump. Both surfaces share a tail heavier than
Gaussian (all curves sit well above the N(0,1) ref) — but that heaviness IS the diffusive IG/Gamma skew,
captured by BulkIG's ≈1.0 coverage. So #33's "if any D(L) floor is ever added it belongs on AWT" stands,
and it should be a larger/floored **D** (the baked awt_diff_mult=2.0 surface-D mechanism), NOT a jump term.

=== MODELLING IMPLICATIONS ===
1. The SIM=wald STOCHASTIC MECHANISM (diffusion → positively-skewed first-passage time) is the RIGHT
   class. The data's noise is diffusive: thin-scaling moments (#33) AND a tail that a body-anchored
   diffusion calibrates. There is NO separable distance-independent jump/incident component of material
   size — the start-transient/traffic noise THEORY §3e posits exists, but it manifests as variance
   (mean/scale), not as a detachable heavy tail beyond the diffusion's own skew.
2. The fitted SHAPE is closer to Gamma/LogNormal than to strict Inverse-Gaussian (IG is the worst body
   fit). This is a SHAPE refinement, not a defect: the win model is an order statistic driven by the mean
   drift, the Var∝L law (vindicated #33), and the tail — all of which IG carries adequately. Swapping the
   first-passage law to Birnbaum–Saunders/Gamma is NOT recommended on this evidence: it would trade the
   physical first-passage derivation for a marginally better body fit with NO tail-calibration gain
   (Gamma p99 1.05 ≈ BulkIG 1.07) and no order-statistic benefit. Parsimony + theory-leads rejects it.
3. AWT #33 floor = larger/uniform D (confirmed here), so the open "AWT D(L) floor" item should be
   implemented (if ever) as an additive surface-D term (Var += σ0²-equiv on the D knob), NOT an IG+jump
   law. Keep it on the existing diffusion-surface mechanism.

=== HYPOTHESIS for a FUTURE GATED experiment (estimate, do not claim a win) ===
An **IG+jump (compound-Poisson/shifted-exponential "trouble") production law is NOT worth a gated
experiment** on this evidence — the marginal tail under-coverage is ~5–7% at p99 (BulkIG/Gamma both ≈1.0)
and there is no clean separable incident mass to model; the explicit mixture over-fits when forced.
The ONE residual diffusive-shape question worth a CHEAP future probe (lower priority than the §3e phase
levers / fusion): whether replacing the per-band IG SHAPE assumption with the empirically-better
positively-skewed family materially sharpens the win-probability ORDER STATISTIC via the per-horse
variance (D_i, #28/0030) channel — i.e. does the better-calibrated body+tail shape change argmin
probabilities enough to convert? Estimated potential: SMALL (the moment law already drives the order
statistic and is vindicated; shape past the moments is a 2nd-order effect on argmin). Gate it only if a
cheaper lever queue empties. The incident/"racing luck" axis is better pursued as a per-horse leak-free
TRAIT on D_i (a horse's own trouble-proneness raises its variance), NOT as a population jump law — which
is exactly the #28/0030 D_i direction already in flight, and it routes the (mild, real) one-sided
asymmetry to where it is predictive rather than merely descriptive.

CAVEAT: descriptive population study (par = band-mean T; within-band shape). It characterizes the LAW's
shape/tail and the turf-vs-AWT tail, not per-horse D_i heterogeneity (#28's job) nor a leak-free predictive
claim. The freak-outlier incidents (~1/4,000) are real but population-negligible.
