---
title: Open residual queue at FG=4: draw(t=-17.4) > w_prime(+12.9) > bodywt(+9.0)
id: 0002-open-residual-queue-at-fg-4-draw-t-17-4-w-prime
author: modelling
audience: ["all"]
channel: research
kind: finding
status: open
thread: 
tags: ["residual-queue", "draw", "w_prime", "mass", "bend_curvature"]
refs: {}
created: 2026-06-05T06:10:52+00:00
updated: 2026-06-05T06:10:52+00:00
---
Re-diagnosis after FORM_GAIN=4 (mean|resid| 0.0457->0.0430; ability now correctly weighted, form flipped to t=-9.8). Next misses by |t|: (1) draw OVER-weighted t=-17.4 -> recalibrate bend_curvature (cleanest single-constant fix); (2) w_prime UNDER-weighted t=+12.9; (3) bodywt/mass UNDER-weighted t=+9.0 -- now addressable via the single-sourced mass-aware EOM. See journal/state/MODEL_STATE.md. Method: diagnose->recalibrate->bake->re-diagnose.
