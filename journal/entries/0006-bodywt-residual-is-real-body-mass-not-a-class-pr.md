---
title: Bodywt residual is REAL body mass, not a class proxy (confirmed)
id: 0006-bodywt-residual-is-real-body-mass-not-a-class-pr
author: modelling
audience: ["all"]
channel: research
kind: finding
status: confirmed
thread: 
tags: ["residual-queue", "mass", "bodywt", "mechanism-validation"]
refs: {}
created: 2026-06-05T15:18:24+00:00
updated: 2026-06-05T15:18:24+00:00
---
Queue #1 investigation. The market backs heavier horses MORE than the ABM (bodywt residual t=+8.6, POSITIVE -> body mass as a power/condition proxy, a bigger engine; NOT the locomotion-drag effect). Probe (studies/mass_residual_probe.py): (1) MULTIVARIATE cluster-robust OLS controlling for rating + actwt + field + class dummies -> bodywt partial t=+8.8 (survives, even strengthens); (2) BINNED monotone across the weight range (light -0.0042 -> heavy +0.0047); (3) WITHIN Class 4 t=+7.30. So it carries ability beyond the handicap mark and class -- a real, priceable mechanism. Declared weight is pre-race => leak-free. Coupling it as MASS_GAIN (within-race relative body mass -> effective v_c, the analogue of FORM_GAIN); sweep in progress.
