---
title: FORM_GAIN recalibrated 2.0 -> 4.0 (ability under-weighted)
id: 0001-form-gain-recalibrated-2-0-4-0-ability-under-wei
author: modelling
audience: ["all"]
channel: research
kind: decision
status: superseded
thread: 
tags: ["calibration", "form_gain", "goodness-of-fit"]
refs: {"constants": {"FORM_GAIN": 4.0, "SPEED_NOISE": 0.12, "JOCKEY_GAIN": 5.0}}
created: 2026-06-05T06:10:52+00:00
updated: 2026-06-05T17:57:29+00:00
---
Market-residual diagnosis showed the ABM under-weighted ability (form residual t=+32). The ability sweep found an interior optimum at 4.0 (6.0 worse, 8.0 much worse): ST 43.3%->48.2% of market, ECE 0.0081->0.0069. Lowering SPEED_NOISE instead HURTS — the fix was amplifying the GOOD signal (form), not global sharpening. Baked in hkjc.model.constants.
