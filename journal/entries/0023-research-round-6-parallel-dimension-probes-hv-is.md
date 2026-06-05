---
title: Research round (6 parallel dimension probes): HV is broken (#1 lever, ~10pp); pace + rest-days are real medium leads
id: 0023-research-round-6-parallel-dimension-probes-hv-is
author: modelling
audience: ["all"]
channel: research
kind: finding
status: open
thread: 
tags: ["research-round", "per-venue", "HV", "draw", "pace", "lone-leader", "days-since-run", "opponent-adjusted", "interactions", "sectionals", "headroom"]
refs: {"HV_pct": 0.476, "ST_pct": 0.579, "HV_draw_t": -31.9}
created: 2026-06-05T20:17:30+00:00
updated: 2026-06-05T20:17:30+00:00
---
Dispatched 6 read-only subagents across the THEORY dimensions to measure market-residual headroom at the 58% (ST) baseline. Findings, ranked by leverage:
[BIG, >=5pp] PER-VENUE CALIBRATION — HV is broken. HV = 47.6% of market vs ST 57.9% (-10.3pp). TWO agents converged: (a) the ST-calibrated global DRAW_GAIN=0.003 massively OVER-penalises draw at HV (HV draw residual t=-31.9 vs ST t=-1.4; slope 25x larger, uniform across HV courses); HV form also over-weighted (t=-13.7 vs -8.4). (b) the kinematic ABM BEATS wald by +2-3pp at HV (matched samples) but LOSES ~4pp on ST — divergence is draw-driven (kinematic better on wide draws in large fields, worse on inside draws via a rail-blocking artefact). So HV wants its own DRAW_GAIN (~0) + FORM_GAIN, and the ABM may be the right HV win model. Also: ST 1000m straight chute has the draw SIGN WRONG (inner draws disadvantaged at the chute, model penalises wide) -> distance-conditional draw fix (152 races, 46% bucket). Small ST fields/sprints over-confident.
[MEDIUM, 1-3pp] PACE — front-runner/lone-leader residual is real (FR t~9-13 both venues, survives form; lone-leader market-model gap +1.3pp at HV; PackedFront contested-pace HV-specific). CRUCIAL: the OLD pace falsification was for the `style` feature (corr only 0.12 with the positional proxy trail_early_pos_norm), so a lone-leader / early-position coupling in WALD is genuinely UNTESTED and physically grounded (lone leader saves energy, controls pace).
[SMALL, 0.5-1.5pp] CONDITION — days-since-run REST PREMIUM (28-60d t=+5.9, survives 7-covariate multivariate; >120d penalty t=-3.3). Body-weight DELTA = noise. Gear = data-limited (not in results_rich). Mechanism: muscle recovery peaks 4-8wk.
[SMALL, ~0.3pp] SECTIONALS — dense mean-velocity as speedfig adds ~+0.3pp over the existing parked speedfig; market is largely efficient w.r.t. sectionals; the CP-W' across-runs hyperbola is NOISE here (too few trip-varied prior runs). CV trajectories unlikely to help WIN (maybe draw geometry).
[NOT A LEVER now] OPPONENT-ADJUSTED ABILITY — residual correlation real (t~4.6, leak-safe, control-stable) but does NOT convert to %market in any leak-free test (collinear ~0.70 with own rtg; the rating channel already prices it). The proper test is a self-consistent fixed-point rating (Elo/BT) REPLACING form, not a bolt-on add-on — a bigger build, uncertain payoff.
[NOT A LEVER on ST] INTERACTIONS for WIN — kinematic ABM net NEGATIVE on ST (~-4pp, over-penalises inside draws); its value is HV win + exotics/place, not ST win.
META-LESSON: the production model was calibrated ENTIRELY on ST and applied globally -> HV and ST-1000m are mis-served. Same 'parked verdict goes stale when the production path changes' lesson, now at the venue/distance level. NEXT STEP IS THE USER'S CALL (no changes made; all probes read-only).
