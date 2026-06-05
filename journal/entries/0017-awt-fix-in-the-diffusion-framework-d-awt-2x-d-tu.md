---
title: AWT fix in the diffusion framework: D_awt = 2x D_turf (no ad-hoc velocity knob)
id: 0017-awt-fix-in-the-diffusion-framework-d-awt-2x-d-tu
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["awt", "diffusion", "surface", "noise-model", "ability-transfer", "calibration"]
refs: {}
created: 2026-06-05T17:38:39+00:00
updated: 2026-06-05T17:38:39+00:00
---
AWT (1650m) was 13% of market (over-confident + weak ability transfer). FIRST attempt used an ad-hoc multiplier on velocity speed_noise -- wrong framework (user). CORRECT: in the diffusion (Wald) model the noise IS the diffusion coefficient D; AWT is simply a more diffusive surface (kickback, less reliable turf-trained signal) => larger D. Calibrated (Wald, D_turf=170): AWT D = 2x turf D is optimal -> AWT 16.7%->23.7% (vs 12.7% velocity baseline), overall 47.4->47.7%, turf unchanged 48.8%. A clean physical statement (AWT 2x as diffusive), one surface-dependent physical parameter, not a tuned velocity knob. REMAINING: AWT 23.7% << turf 48.8% is the genuine ability-TRANSFER weakness (model ranks AWT winners at percentile 0.59 vs 0.70 turf) -- the turf-dominated form/v_c doesn't apply to all-weather; surface-specific ability is the fix but data-starved (AWT ~6% of races). TIMING: wald sampling 1.6x probit (Inv-Gaussian vs Gaussian), both ~30-40s full-scale; closed-form PL is ~2000x faster (no sampling) -- the instant proxy for sweeps. ARCHITECTURE: this is the case for adopting diffusion as THE noise model (physical, condition-dependence is real, interaction-ready); ~0.5pp turf cost vs velocity-mult. Eventually the agent-based sim's integrator should use per-timestep sqrt(2D)dW (non-interacting limit = Wald).
