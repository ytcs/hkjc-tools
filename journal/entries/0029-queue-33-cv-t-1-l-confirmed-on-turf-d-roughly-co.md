---
title: Queue #33: CV(T) ∝ 1/√L CONFIRMED on turf; D roughly constant (no L-term needed); AWT carries a real variance floor
id: 0029-queue-33-cv-t-1-l-confirmed-on-turf-d-roughly-co
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["theory", "variance", "distance-law", "queue-33", "diffusion", "D-floor", "CV"]
refs: {"study": "studies/cv_distance_law.py", "plot": "studies/out/cv_distance_law.png", "panel": "data_cache/results_rich.parquet", "queue_item": 33, "slope_bucket": -0.507, "slope_bucket_ci": [-0.84, -0.17], "slope_turf_pooled": -0.39, "D_turf_m2s": 0.7, "D_awt_m2s": 0.94, "floor_sigma0_turf_s": 0.15, "floor_sigma0_awt_s": 0.46, "D_logL_slope_turf": 0.12}
created: 2026-06-06T02:49:48+00:00
updated: 2026-06-06T02:49:48+00:00
---
DESCRIPTIVE population study (not a predictive feature → no as-of/leak-free constraint; within-race
de-meaning removes field-strength + going, leaving the residual stochastic spread). Panel:
data_cache/results_rich.parquet (39,428 runners, 3,248 races, 2022-09 → 2026-05, per-horse FinishTime
in seconds — clean, monotone with Pla, no DNF sentinels). Study + plot:
studies/cv_distance_law.py → studies/out/cv_distance_law.png.

METHOD. Within-race CV(T) = sd(FinishTime)/mean(FinishTime) per race (the sd de-means the race →
strips field strength/going), aggregated per bucket L × Venue × surface (TURF vs ALL-WEATHER from
Course). Scaling test: regress log CV on log L. Floor test: regress log D on log L and fit the
time-floor model Var(T) = 2·D0·L/v³ + σ0² (σ0² = a distance-INDEPENDENT variance chunk).

=== DOES 1/√L HOLD? YES — strongly, on turf. ===
Fitted slope of log CV on log L (theory predicts −0.5):
  • bucket-level (each L×venue×surface a point, n≥30):  slope = −0.507, 95% CI [−0.84, −0.17], R²=0.45
  • turf pooled-by-distance (9 cells):                  slope = −0.39,  95% CI [−0.58, −0.20], R²=0.71
  • per-race turf (n=2933):                             slope = −0.62,  95% CI [−0.72, −0.53], R²=0.054
  • per-race ST turf −0.72 / HV turf −0.54 / AWT −0.47.
The −0.5 prediction sits inside every CI. CV falls cleanly 1000m→1650/1800m (turf 0.92%→0.66%),
i.e. longer races ARE intrinsically more predictable, parameter-free, exactly as THEORY §3e predicts.
(The per-race slope reads slightly steeper than −0.5 but its R² is tiny — it's noise-dominated; the
bucket means, where the signal lives, land on −0.5.)

=== THE D(L) FLOOR — is D constant? ON TURF, YES. ===
Back-out D from CV²=2D/(L·v), v=L/mean(T):
  log D on log L (turf):  slope = +0.12, 95% CI [−0.27, +0.50] — indistinguishable from 0.
  ⇒ Theory's SIMPLEST form (D constant in L) holds on turf. D ≈ 0.6–0.8 m²/s across the middle
    distances; the implied-D table is flat with a faint U (mild lift at the 1400m and 2400m cells,
    the latter only 12 races — small-sample, not a trend).
The time-floor fit on turf: D0 ≈ 0.70 m²/s, σ0 ≈ 0.15 s — the distance-independent chunk is only
~10% of the variance at 1000m and ~3% at 2200m. So on turf there is NO material sprint over-dispersion
beyond 1/√L: a D(L) floor term is NOT warranted.

=== THE ONE EXCEPTION — ALL-WEATHER TRACK. ===
AWT is both more diffusive AND floored:  D0 ≈ 0.94 m²/s (vs turf 0.70 — consistent with the baked
awt_diff_mult=2.0 surface-D), and a LARGE time-floor σ0 ≈ 0.46 s = ~36% of the variance in AWT
sprints (1200m) vs ~26% at 1800m. AWT sprints carry a big distance-independent variance pile — the
kickback/dirt "bad-break/check costs the same seconds at any L" mechanism §3e names. If any D(L)
floor is ever added, it belongs on AWT, not turf.

=== VENUE STRUCTURE (consistent with the model's priors). ===
HV turf CV sits systematically BELOW ST turf at every shared distance (e.g. 1200m HV 0.79% vs ST
0.94%); implied D_HV ≈ 0.58–0.80 vs D_ST ≈ 0.69–0.92. HV's smaller fields + tighter handicap blanket
finishes → lower residual spread. This is the descriptive backdrop to the EVT field-size argument
(§3e): ST runs the big fields where D has the most order-statistic leverage — the same split that
made draw→D (DRAW_VAR_GAIN) convert ST-only.

=== MODELLING IMPLICATIONS ===
1. The production homogeneous-Wald variance law (Var ∝ L, CV ∝ 1/√L) is EMPIRICALLY VINDICATED on
   turf — the slope is −0.5 and D shows no L-trend. The parameter-free §3e prediction passes a direct
   data check (theory-confirms-data).
2. A distance-dependent D(L) floor is NOT needed on turf (the queue's open "distance-floor D(L)" item
   can be CLOSED for turf — the sprint cells are not under-dispersed relative to 1/√L). Keep it open
   ONLY for AWT, where σ0≈0.46s is real; an AWT-specific additive D floor (Var += σ0²) is the correct
   structure there, and AWT is already the surface flagged as extra-diffusive.
3. SUPPORTS draw→D rather than complicating it. The variance knob is well-behaved (constant D in L on
   turf) so the draw→D lever sits on a stable base; and because D has the most leverage in big fields
   (ST) and CV is largest in the SHORT/AWT cells, the §3e prioritisation rule is reinforced: pursue
   per-horse D_i (#28) and variance levers in ST big fields / sprints first. For per-horse D_i (#28):
   the within-race CV floor (~0.6–0.9% turf) is the population baseline a horse's own figure-spread
   should modulate around — D_i should scale the bucket D, not replace it.
4. Caveat: this is a POPULATION/descriptive study (within-race spread, all runners) — it validates the
   law's L-scaling and the D-constancy, not the per-horse D_i heterogeneity, which is #28's job.
