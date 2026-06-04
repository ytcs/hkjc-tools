---
id: 0014-m1-shipped-jackpot-capture-store-carryover-ledge
title: M1 shipped: jackpot capture store + carryover ledger (data flywheel)
kind: result
status: open
tags: ["jackpot-carryover", "M1", "data-flywheel", "capture"]
refs: {"modules": ["hkjc.data.jackpot_ticks", "hkjc.data.carryover_ledger"], "runner": "experiments/capture_jackpots.py"}
related: []
created: 2026-06-04T00:34:16+00:00
updated: 2026-06-04T00:34:16+00:00
---
Built milestone M1 of the carryover plan. JackpotCaptureStore (data.jackpot_ticks) polls live_jackpots and persists the C/G/estDividend time series per meeting (append-only parquet; no dedup -> we want the trajectory for the dilution study). carryover_ledger: settled_jackpot_result joins get_race_dividends (won/not-won/dividend/combo) to the last pre-close capture; build_carryover_rows computes eff_takeout/agg_edge per leg; CarryoverLedger upserts a durable dataset keyed by (date,venue,pool,leg). Runner experiments/capture_jackpots.py has capture/settle modes for a cron. Verified end-to-end (dry run persisted an upcoming-meeting DT snapshot; values null until pools open). NEXT: schedule it (B11) so we accumulate real events, plus B13 Racing-News backfill for frequency priors (Gate 1).
