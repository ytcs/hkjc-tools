---
id: 0058-probe-verdict-abm-field-interaction-edge-is-not
title: Probe verdict: ABM field-interaction edge is NOT localized -- redundant in every race slice
kind: finding
status: open
tags: ["ABM", "field-interaction", "probe", "strategy", "standing-order"]
refs: {}
related: []
created: 2026-06-04T22:42:12+00:00
updated: 2026-06-04T22:42:12+00:00
---
Per-race ABM advantage over the EW+jockey logit (logit_loss - abm_loss; +ve = ABM better),
1910 ST races, sliced by the covariates where field interaction should bite hardest:
  lone_speed terciles:  -0.061 / -0.044 / -0.037   (ABM loses everywhere; corr +0.018)
  n_front (pace duel):  -0.073 / -0.026 / -0.046   (no monotone pattern; corr +0.021)
  field_size:           -0.067 / -0.047 / -0.038   (corr +0.034)
  distance:  sprints 1000-1200 -0.0095 (n=148, near parity) else ~-0.05; corr -0.015.
All |corr| < 0.035. There is NO race shape where the mechanistic structure beats the per-horse
logit -- the aggregate-zero was genuine redundancy, not dilution of a localized edge. The only
near-parity niche is short sprints (raw speed dominates, little traffic/energetics to model), and
even there the ABM does not win. VERDICT: the user'''s pre-pivot gate is cleared -> pivot to the
flexible feature model is evidence-justified. Keep the ABM code as a benchmark + potential
engineered-feature source (lone-speed/pace-pressure indices), not the primary predictor.
