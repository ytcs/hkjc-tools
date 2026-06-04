---
id: 0015-jackpot-capture-scheduling-self-gating-loop-in-h
title: Jackpot capture scheduling: self-gating loop in HK time; Claude cron is not the season tool
kind: decision
status: open
tags: ["scheduling", "ops", "timezone", "jackpot-carryover", "M1"]
refs: {"runner": "experiments/capture_jackpots.py"}
related: []
created: 2026-06-04T00:44:09+00:00
updated: 2026-06-04T00:44:09+00:00
---
How to dispatch M1 capture. Claude's CronCreate is SESSION-BOUND, fires in the USER'S LOCAL timezone (not HK), and recurring jobs auto-expire after 7 days -> fine for ad-hoc 'capture tonight's meeting' while a session is live, NOT a season-long pipeline. Chosen approach: a self-gating LOOP runner (experiments/capture_jackpots.py loop) deployed under tmux/systemd (or OS cron in UTC), which (a) decides WHEN to poll in HK time via zoneinfo Asia/Hong_Kong (correct regardless of host clock), (b) only PERSISTS when jackpot pools are genuinely open (non-null investment) so off-meeting/holiday polls are cheap no-ops. TZ discipline: captured_at stored UTC, partition date = server's HK meeting date, scheduler gating computed in HK time. HK = UTC+8 no DST; day cards ~05:00-10:00 UTC, night ~10:45-15:30 UTC. Verified: skips when no pools open; HKT resolves UTC+8.
