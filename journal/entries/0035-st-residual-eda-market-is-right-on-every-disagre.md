---
id: 0035-st-residual-eda-market-is-right-on-every-disagre
title: ST residual EDA: market is right on every disagreement; only faint residual is energetics (v_c), strongest at middle distances
kind: finding
status: open
tags: ["eda", "residuals", "traffic-model", "sha-tin", "market-efficiency", "energetics", "draw", "incremental-edge"]
refs: {}
related: []
created: 2026-06-04T09:50:17+00:00
updated: 2026-06-04T09:50:17+00:00
---
Battery of EDA on the leak-free OOS panel (experiments/eda_residuals.py; data_cache/oos_panel.parquet, 1769 runners / 139 ST races; plot data_cache/figs/eda_residuals.png).

A. CALIBRATION. Both traffic and market are honestly calibrated (track the diagonal). But the market is far SHARPER: it spreads win prob 0.006->0.27, traffic only 0.02->0.16. Traffic is calibrated-but-compressed => it cannot discriminate enough to beat the market. Both slightly overconfident at the very top.

B. DISAGREEMENT (the decisive chart). Bucketing by log(traffic/market) into quintiles, realised win rate tracks p_MARKET almost exactly and is essentially FLAT in p_traffic:
  q0 (market>>traffic): market .189 traffic .062 -> won .167  (market right)
  q4 (traffic>>market): market .010 traffic .087 -> won .011  (market right)
  win-rate by quintile (.167,.116,.065,.031,.011) ~ market (.189,.109,.056,.028,.010); traffic ~flat .06-.09.
  => Whenever the model disagrees with the market, the market is right. Traffic disagreements are noise.

C. INCREMENTAL EDGE (walk-forward conditional logit OOS race-win log-loss):
  market only 2.1260 | market+traffic 2.1527 (WORSE -0.027) | market+draw 2.1343 (worse) | market+v_c 2.1238 (+0.0023, marginal) | traffic only 2.4700.
  => Adding the full traffic prob HURTS. Draw adds nothing. v_c (critical velocity) is the ONLY feature that (barely) improves on market.

D. DRAW. Market residual (won - p_market) vs draw has slope -0.003/draw ~ 0 => the market PRICES DRAW efficiently; no linear draw inefficiency to exploit. Post-calibration the traffic model own draw effect in real (ability-spread) fields is modest/flat.

E. ENERGETICS (the one glimmer). corr(v_c_eb, finish_pos) = -0.117 (higher v_c -> better finish, weak but real). Within EVERY market tercile, higher-v_c half wins ~1-1.4pp more (hi: .174 vs .164; mid: .060 vs .046; lo: .017 vs .007). Small but CONSISTENT residual the market does not fully price — and it is the energetics, not the draw/traffic geometry.

F. WHERE THE PHYSICS IS REAL (slices, traffic skill vs uniform): mile +0.136, route +0.091, sprint -0.040. The structural model HELPS at middle/route trips and HURTS at sprints — consistent with critical-velocity/stamina mattering over distance while gate-speed/break dominates sprints (ST 1000m is the straight chute anyway). Field-size slices noisy.

VERDICT: No standalone or incremental edge from the interacting traffic model over the efficient ST win market; the market is right on disagreements and already prices draw. The single physically-meaningful residual is the energetics (v_c), faint (~1pp) and strongest at middle distances. Pull that thread: (a) sharpen the energetics signal (better v_c/W estimation, going/pace adjustment, more sectional data via the flywheel), (b) test v_c as a small additive feature to a market-anchored model on middle-distance races specifically, (c) the win-prediction market is closed - revisit the structural model for EXOTICS/positioning (where the crowd is weaker) rather than win.
