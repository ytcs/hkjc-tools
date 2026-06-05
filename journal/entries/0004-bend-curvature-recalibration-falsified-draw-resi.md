---
title: Bend_curvature recalibration FALSIFIED -- draw residual is low-leverage
id: 0004-bend-curvature-recalibration-falsified-draw-resi
author: modelling
audience: ["all"]
channel: research
kind: deadend
status: superseded
thread: 
tags: ["calibration", "bend_curvature", "draw", "negative-result", "residual-queue"]
refs: {}
created: 2026-06-05T08:54:27+00:00
updated: 2026-06-05T18:53:03+00:00
---
Residual-queue #1 (draw OVER-weighted t=-17.4) is NOT fixed by bend_curvature. Subsample sweep nominated 0.0030 (48->50%) but at FULL scale 0.0030 gives 47.9% vs 0.0037's 47.7% (+0.2pp, within noise) and ECE WORSE (0.0087 vs 0.0074). %market is on a flat ridge over bend in [0.0030,0.0050] -- the geometry knob is already near-optimal. Interpretation: the draw residual is real but LOW-LEVERAGE for goodness-of-fit (it shifts mid-pack mass, not the winner's prob, which dominates %market). Keep bend_curvature(ST)=0.0037. Next: re-diagnose and target a higher-leverage mechanism (w_prime t=+12.9 or mass t=+9.0), or test a draw x distance/field interaction rather than the flat bend constant.
