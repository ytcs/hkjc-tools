---
title: Draw & the interaction axis are DEFERRED not rejected: draw is geometry × CONTROL coupled (needs a joint jockey-policy game, not a regression term)
id: 0027-draw-the-interaction-axis-are-deferred-not-rejec
author: modelling
audience: ["all"]
channel: research
kind: decision
status: confirmed
thread: 
tags: ["draw", "control", "geometry", "interaction", "deferred", "theory", "game-theory", "jockey-policy", "competition", "mean-field", "abm-deprecation"]
refs: {}
created: 2026-06-05T23:02:52+00:00
updated: 2026-06-05T23:02:52+00:00
---
User steer (important framing correction): the draw effect — and the pace/competition interaction axis generally — must be DEFERRED, not REJECTED.

WHY the simple treatments fail: draw is not a static geometric covariate. It is deeply coupled to the CONTROL dimension — the jockey's policy (how the gate is used to take up position, when to commit, how the trip is navigated). A regression / mean-field term (DRAW_GAIN, DRAW_FS_GAIN, a scalar contestedness×pos_fade, the relational pace-pressure term) can only capture the FIRST-ORDER GEOMETRIC SHADOW of a fundamentally GEOMETRY × CONTROL phenomenon. That is why those probes don't convert — the mechanism is real, we just haven't modelled it deeply enough. The first-attempt ABM did reproduce the qualitative draw/trip characteristics (a point in favour of the picture), even though its hand-tuned implementation was low-leverage for win and was deprecated from the codebase.

DIRECTION: the proper treatment is a joint differential / mean-field GAME of jockeys navigating the geometry (the THEORY §3b competition game), with draw as a boundary condition on that game — geometry + control modelled TOGETHER. Until that theory exists, treat DRAW_GAIN/DRAW_FS_GAIN as INTERIM stopgaps and keep the axis OPEN. This is the most promising piece of deep theory-building left — a theory build, not a regression probe.

Action taken: re-framed THEORY.md (Appendix), MODEL_STATE (section A + a DEFERRED-not-refuted note), and the DRAW_FS_GAIN constant comment from 'interaction axis closed / refuted' to 'deferred pending a geometry×control theory'. The ABM-sim CODE deprecation stands (greenlit; physics preserved in THEORY.md), but the AXIS is open.
