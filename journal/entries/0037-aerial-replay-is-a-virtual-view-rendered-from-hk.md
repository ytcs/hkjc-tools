---
id: 0037-aerial-replay-is-a-virtual-view-rendered-from-hk
title: Aerial replay is a VIRTUAL VIEW rendered from HKJC RF/GPS positioning; dense trajectory data exists upstream - hunt the feed before CV
kind: finding
status: open
tags: ["video", "trajectory", "tier-b", "data-acquisition", "energetics"]
refs: {}
related: []
created: 2026-06-04T10:11:58+00:00
updated: 2026-06-04T10:11:58+00:00
---
Inspected data_cache/aerial_sample.mp4 (1080p/25fps). The HKJC aerial replay is an Aerial Virtual View - a SYNTHETIC 3D render (clean silk-icon markers, schematic track with rail lines + distance markers 420M/920M, saddlecloth legend), NOT drone footage. Disclaimer (accuracy may be affected by mobile-phone signal interference; refer to Real Replay) reveals it is rendered from an RF/GPS horse-positioning system.
IMPLICATION (reorders TRAJECTORY_RESEARCH_PLAN Tier-2): dense per-horse (x,y) trajectory data ALREADY EXISTS upstream and the hard real-footage CV (moving camera, occlusion, re-ID) is solved by HKJC. Highest-leverage cheap step = hunt the underlying position FEED (JSON/websocket of (x,y) per horse per tick, like the live-odds GraphQL) via browser/network recon of the aerial-view page. If it exists -> dense ~25Hz lateral+longitudinal trajectories, NO CV: sharper v_c/W AND the lateral-lane data Tier-B needs to calibrate the interacting model. Fallback CV on the virtual view is far easier than broadcast footage (rendered icons + distance-marker calibration) but uses a moving 3D camera (per-frame homography); the view is lossy (RF noise) - fine for lane/ground-loss calibration, not sub-metre truth.
GATE: build only after confirming the energetics residual (faint v_c edge, middle distances; journal 0035) is worth sharpening. Doc updated.
