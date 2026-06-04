---
id: 0027-trip-trouble-gate-negative-video-not-justified-c
title: Trip-trouble gate NEGATIVE -> video NOT justified; close the model-edge line
kind: decision
status: open
tags: ["trouble", "video-gate", "market-efficiency", "model-edge-closed", "strategy"]
refs: {"experiment": "experiments/validate_trouble_model.py"}
related: []
created: 2026-06-04T05:04:37+00:00
updated: 2026-06-04T05:04:37+00:00
---
Gate experiment (144 races, 16 meetings; 179 troubled-last-start rows). Prior-run trouble from stewards' running comments (blocked/no-clear-run/checked, ~12% rate). Walk-forward race_win_log_loss: market 2.069 baseline; clogit[mkt] 2.078 (-0.4%); clogit[mkt,trouble_last] 2.154 (-4.1%); +rate 2.154 (-4.1%). Trouble HURTS. Likely cause: the 'unlucky last time' angle is the most famous handicapping angle -> crowd already (over)bets it, so troubled horses are if anything OVER-rated, not under. GATE VERDICT: the stewards' trouble notes are a stronger signal than a first CV pass would extract; they don't beat the market, so video re-deriving the lateral/ground-lost signal won't either -> DO NOT build the video CV pipeline (saved weeks). FOURTH consecutive negative (0011 crude, 0023 rating/form, 0025 sectional-physics, now trouble): the HK win market is efficient w.r.t. EVERY extractable signal. STRATEGIC CONCLUSION: close the model-edge hunt; the carryover STRUCTURAL edge is the only viable line and is compounding live data via the Modal flywheel. Caveat: samples small (3-month windows) but the 4x consistency + efficient-market prior make this the right resource call. Revisit only with full-season data or a fundamentally new data source the market truly cannot see.
