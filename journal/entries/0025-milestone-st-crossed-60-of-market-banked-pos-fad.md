---
title: MILESTONE: ST crossed 60% of market. Banked pos-fade/mass/AWT/jockey; mean-field surrogate now saturating
id: 0025-milestone-st-crossed-60-of-market-banked-pos-fad
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["milestone", "60-percent", "pos-fade", "mass", "allometric", "awt", "jockey", "per-venue", "saturation", "conversion-test", "interaction-layer", "new-data", "debut", "game-theory"]
refs: {"ST_pct": 0.6, "HV_pct": 0.54}
created: 2026-06-05T21:53:56+00:00
updated: 2026-06-05T21:53:56+00:00
---
Autonomous multi-front push. Current: ST 60.0% / HV ~54% of market (wald, full-scale; project start ~48%). BANKED this session (all committed): per-venue DRAW/FORM (HV +4.9pp), POS_FADE_GAIN=-1.0 (the pace-equilibrium shadow, +0.5/0.6pp), allometric mass on wald ST b=1.1 (the persistent #1 ST residual, +0.5pp; HV off), per-SURFACE AWT draw 0.008 (AWT 43->49%; AWT market ~33% softer), JOCKEY_GAIN 5->6 (+0.4/0.3pp). Deepened THEORY.md §3b (the competition game as a differential/mean-field game; why Elo failed = competition is RELATIONAL not scalar; positional-fade = the equilibrium's mean-field shadow; 6 ranked testable predictions).
REFUTED this session (residual signal that does NOT convert to %market -- the recurring lesson that the quick_eval conversion test, NOT the residual t-stat, is the gate): opponent-strength/Elo (collinear w/ rtg), carried-weight actwt, rest-days (days-since-RACE proxy -- per user, blind to daily training; needs training-load data), weight-trajectory bw_dev (won-corr~0), contestedness x pos_fade (theory pred #1, early_pos>0.85 proxy), HV form-by-distance (marginal +0.2-0.4pp).
SATURATION: the cheap per-horse mean-field levers are banked/refuted. The push to 70% now forks into (A) the INTERACTION layer (the ABM -- it already BEATS wald at HV +2-3pp; the substrate for the pace/draw x style couplings the theory predicts) and (B) NEW DATA scraping (barrier trials for debuts, training-load for rest, weather for environment, formline for competition -- all cataloged in HKJC_DATA.md). Quick win still open: a debut-specific fix (RTG is ANTI-predictive for fill-null debut ratings; jockey is the strong debut signal; the debut market is genuinely soft, winner-rank 0.44 vs 0.68) -- bounded ~0.3-0.5pp, needs a prior_starts feature.
