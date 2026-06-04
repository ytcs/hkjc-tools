---
id: 0052-speed-figure-margin-form-does-not-beat-ordinal-r
title: Speed-figure / margin form does NOT beat ordinal rank form (null result)
kind: finding
status: open
tags: ["form", "speed-figure", "margin", "null-result", "featurization"]
refs: {}
related: []
created: 2026-06-04T19:33:17+00:00
updated: 2026-06-04T19:33:17+00:00
---
Hypothesis: ordinal form (1-(Pla-1)/(N-1)) discards margin; FinishTime should sharpen it.
Re-scraped results_rich WITH per-horse FinishTime+LBW (0 nulls, 39320 rows). Built two figures:
  (a) cohort speed fig: distance/time standardized within (venue,dband,going) -> only 10% of market.
      FALSIFIED: absolute time conflates race pace+class with ability; standardizing recovers
      class, not horse edge. Ordinal form is pace-immune (within-race relative); this is not.
  (b) within-race standardized time margin (faster than race mean=higher), pace/going-immune by
      construction -> 27% of market, vs ordinal 28%. Combining ordinal+margin = 28% (+0.0006, noise).
Conclusion: RANK strictly dominates MARGIN. In HK small competitive fields, margin-beyond-rank is
pace/tactical noise, not stable ability. Finishing ORDER already captures the ability ordering.
Third wall (after weight/trainer/etc nulls); per standing order, walls are the norm -> next lever.
experiments/speed_figure_form.py
