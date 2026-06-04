---
id: 0055-hybrid-stack-validated-abm-recency-44-and-found
title: Hybrid stack validated (ABM+recency = 44%%) AND found the ABM 20x data bottleneck
kind: finding
status: open
tags: ["hybrid", "ABM", "data", "bottleneck", "standing-order", "energetics"]
refs: {}
related: []
created: 2026-06-04T20:14:52+00:00
updated: 2026-06-04T20:14:52+00:00
---
hybrid_stack.py: generated the ABM OOS win prob via its own walk-forward, stacked log(ABM_p) with
EW form + jockey in a 2nd-level conditional logit. On the SAME races:
  ABM raw 40%% | EW+jockey logit 39%% | ABM+EW 44%% | ABM+EW+jockey 32%% (overfit) | market 100%%.
ABM+EW beats BOTH alone (+4pp over ABM) -> the ABM field-interaction signal and recency form are
partly ORTHOGONAL; the hybrid is the right architecture. Adding jockey on top collapses it because
jockey is already INSIDE the ABM (collinear, unregularised logit overfits on small n) -> stack only
the ABM'''s orthogonal complements, not what is already mechanistic; the stacker needs ridge.
BUT the decisive discovery: this ran on only 139 races. The energetics_dataset covers just 2026-03..05
(161 ST races / 15 dates) because of a HARDCODED 3-month build range -- while results_rich has 3248
races / 5 seasons. HKJC serves sectionals across the whole span (probed). The ABM was never data-tapped;
it was scope-capped. build_energetics_full.py rebuilds the panel over all 3248 races (validated: it
reproduces existing v_c/w_prime exactly). This 20x expansion is the highest-leverage move for the 90%%
target -- tight ABM calibration + a hybrid stack powered on ~3000 races, not 139.
