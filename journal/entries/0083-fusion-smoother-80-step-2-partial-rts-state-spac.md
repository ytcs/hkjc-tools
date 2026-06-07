---
title: FUSION smoother #80 step 2 PARTIAL — RTS state-space runs end-to-end on all 12 horses, first reconstructed-race trajectories produced
id: 0083-fusion-smoother-80-step-2-partial-rts-state-spac
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: []
refs: {}
created: 2026-06-07T04:55:46+00:00
updated: 2026-06-07T04:55:46+00:00
---
PARTIAL (anti-stall): studies/cv_fusion_smoother.py — a per-horse numpy Kalman forward-filter + RTS backward-smoother on the shared race_t grid (5Hz, 448 nodes, race_t 9.2-98.6s) — RUNS end-to-end fusing all 3 validated feeds. State z=[s,s_dot,l,l_dot]; physics motion prior = bounded longitudinal accel (sigma_a 0.8 m/s^2) + bounded lateral accel (0.35 m/s^2 -> slow sideways drift), applied JOINTLY in the smoother, no per-feed pre-filter. Obs models: aerial odometer arclen (field-shared, sd 3m) + y_lat (sd 6m, inflated to 12m >1300m foreshortening); drone arclen (sd 8m) + y_lat (sd 7m); PATROL = metric anchor y_lat (sd 1.2m/conf). INGEST: aerial 2626 obs/11 runners, drone 4957/10, patrol 319/8 -> UNION 12/12 identity. FIRST NUMBERS: tight/wide gap SURVIVES fusion +3.85m (rail #4/8/11/12 mean 5.36m vs wide #1/2/6/9 mean 9.21m) vs stewards/patrol +4.1m, now covering 8/8 commented horses (patrol alone was 4/8). Full result + figure + per-band uncertainty census to follow after a prior-strength pass (the run-in band must show the honest widening). Artifacts: cv_fusion_trajectories.parquet (5193 rows), figs/cv_fusion_recon.png.
