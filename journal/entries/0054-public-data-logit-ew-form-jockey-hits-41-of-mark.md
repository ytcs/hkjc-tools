---
id: 0054-public-data-logit-ew-form-jockey-hits-41-of-mark
title: Public-data logit (EW form + jockey) hits 41%% of market, BEATS standalone ABM (~38%%)
kind: finding
status: open
tags: ["benchmark", "logit", "ABM", "hybrid", "strategy", "standing-order"]
refs: {}
related: []
created: 2026-06-04T20:07:51+00:00
updated: 2026-06-04T20:07:51+00:00
---
best_logit.py, 3248 ST+HV races, walk-forward conditional logit:
  career form 28%% | EW form 36%% | EW+jockey 41%% | EW+jockey+trainer 41%% (trainer ~0).
A two-feature flexible model already EXCEEDS the full mechanistic ABM (~38%% form+jockey), and we
have not yet added proper rating, conditions-suitability, draw, weight, class moves, or field-pace.
Two implications: (1) 90%% of market is clearly not data-blocked -- feature richness is a live path.
(2) The ABM and logit capture ability DIFFERENTLY (recency helps logit, not ABM). The ABM'''s unique,
non-substitutable value is FIELD INTERACTION (a horse'''s chance depends on the others: pace pressure,
traffic, draw x geometry) which a per-horse logit cannot represent. STRATEGIC DIRECTION: hybrid --
feed ABM-derived interaction features (predicted win prob / pace-pressure / traffic-trouble) into a
flexible calibrated model alongside per-horse signals (EW form, jockey). Best of both structures.
(Caveat: conditional_logit_fit_predict blew up on the collinear career+jockey+trainer combo -> needs
ridge regularization; EW+jockey is stable and monotone.)
