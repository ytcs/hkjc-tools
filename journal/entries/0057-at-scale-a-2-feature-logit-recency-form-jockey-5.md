---
id: 0057-at-scale-a-2-feature-logit-recency-form-jockey-5
title: At scale, a 2-feature logit (recency form + jockey) = 53%% of market and the ABM adds NOTHING
kind: finding
status: open
tags: ["ABM", "logit", "featurization", "strategy", "standing-order", "breakthrough"]
refs: {}
related: []
created: 2026-06-04T21:46:14+00:00
updated: 2026-06-04T21:46:14+00:00
---
Full-data hybrid stack, 1932 ST races (paired bootstrap on per-race win log-loss):
  ABM raw 44%% | EW+jockey logit 53%% | ABM+EW+jockey 53%% | market 100%%.
  EW+jockey BEATS ABM raw: Δ +0.0480 nats/race [+0.0331,+0.0632] P=100%%.
  ABM adds nothing on top of EW+jockey: Δ +0.0002 [-0.0040,+0.0045] P=53%% (statistically zero).
This REVERSES the 139-race result (where the ABM helped a data-starved logit). At scale the flexible
model learns the signal directly and the mechanistic energetics/traffic/rating coupling is redundant
given two simple per-horse features. Likely because the official rating the ABM ingests is itself
mostly a function of recent finishing positions (EW form) + connections (jockey), so EW+jockey proxies
it. STRATEGIC INFLECTION: the gains live in FEATURIZATION, and a flexible model extracts them better
than hand-built mechanistic coupling. The ABM was the scaffold that surfaced these signals and beat
one-factor form, but the path to 90%% is feature richness (rating done right, conditions-suitability,
draw, weight, class moves, breeding, field-pace structure as engineered features), not more ABM
machinery. CAVEAT to probe: the ABM''''s field-interaction edge may exist in specific race shapes
(lone speed, small fields) even though it nets to zero in aggregate.
