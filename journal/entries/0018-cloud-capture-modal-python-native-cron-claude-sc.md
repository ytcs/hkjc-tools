---
id: 0018-cloud-capture-modal-python-native-cron-claude-sc
title: Cloud capture: Modal (Python-native cron); Claude scheduling can't run when machine is off
kind: decision
status: open
tags: ["cloud", "modal", "scheduling", "ops", "jackpot-carryover"]
refs: {"files": ["cloud/modal_app.py", "src/hkjc/data/capture.py"]}
related: []
created: 2026-06-04T00:53:57+00:00
updated: 2026-06-04T00:53:57+00:00
---
Researched scheduling for the always-on flywheel. Claude CronCreate/loop are LOCAL/session-bound; Anthropic Routines run in the cloud but MIN INTERVAL 1 HOUR (too coarse for 5-min polling). So Claude has no managed cloud cron for this. Chosen: MODAL — Python-native (our package runs unchanged), reliable cron scheduled in HK time (modal.Cron timezone=Asia/Hong_Kong), persistent Volume, $30/mo free credits >> this tiny workload. Built cloud/modal_app.py (capture every 5min HK 10-23h self-gating; settle nightly 23:30 HKT) + cloud/README.md. Extracted shared ops into hkjc.data.capture (capture_once/settle_meeting/settle_recent) so local runner and cloud share one implementation. GitHub Actions is the fallback but cron is unreliable (10-30min delays/drops) and private-repo schedules need a paid plan.
