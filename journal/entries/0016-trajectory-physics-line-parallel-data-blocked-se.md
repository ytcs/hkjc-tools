---
id: 0016-trajectory-physics-line-parallel-data-blocked-se
title: Trajectory/physics line: parallel & data-blocked; sectional source is a JS SPA
kind: decision
status: open
tags: ["trajectory", "physics-model", "sectional", "data-blocked", "parallel-line"]
refs: {"doc": "docs/TRAJECTORY_RESEARCH_PLAN.md"}
related: []
created: 2026-06-04T00:44:09+00:00
updated: 2026-06-04T00:44:09+00:00
---
Planned the dense-trajectory line as a SECONDARY parallel line (carryover stays main - simpler). Orthogonal but feeds carryover's weakest dependency (RQ6 leg Trio model). DATA IS BLOCKING; modelling is notes only. Tier 1 (sectional splits+positions, low effort): the 'Sectional Time & Position' page is a JS SPA (legacy .aspx 302s to it, no server-side table; no obvious data XHR on first load). ACTION: browser spike to capture the deferred data XHR/GraphQL, then build a SectionalSource and a cheap harness kill/confirm (do sectional features beat market where draw+weight couldn't?). We already have RunningPosition (sectional positions) + SpeedPro energy; missing piece is split TIMES. Tier 2 (video trajectory, high effort): find replay stream URLs (browser), then CV pipeline (detect->track->camera/track homography->saddlecloth ID); ground-lost = highest hidden signal. MODELLING NOTES: CP-W' energetics (latent v_c, W', efficiency, accel, preferred pace; stable per-horse; hierarchical Bayesian fit regularizes small samples); pace-counterfactual prediction = the edge mechanism vs efficient win market. Full plan: docs/TRAJECTORY_RESEARCH_PLAN.md.
