---
id: 0025-flywheel-deployed-modal-jackpot-capture-is-live
title: Flywheel DEPLOYED: Modal jackpot capture is live (stevenytc workspace)
kind: result
status: open
tags: ["flywheel", "deployed", "modal", "jackpot-carryover", "live-data"]
refs: {"app": "hkjc-jackpot-capture", "file": "cloud/modal_app.py"}
related: []
created: 2026-06-04T03:31:06+00:00
updated: 2026-06-04T03:31:06+00:00
---
cloud/modal_app.py deployed to Modal (app hkjc-jackpot-capture, stevenytc workspace). Image built from pyproject + add_local_python_source(hkjc). Schedules (HK time via modal.Cron timezone=Asia/Hong_Kong): capture every 5 min HK 10:00-23:59 (self-gating: only persists when local jackpot pools open), settle nightly 23:30 HKT. Persists to Modal Volume 'hkjc-data'; pull via `modal volume get hkjc-data /carryover_ledger.parquet ./data_cache/`. Verified end-to-end with `modal run` -> container imported hkjc, called live_jackpots, self-gated ('no open jackpot pools' at Thu 11:30 HKT). First real carryover data expected at the next HK meeting (HV Wed typically runs Triple Trio). This unblocks the carryover main line: live C/G/estimatedDividend trajectory -> identify +EV windows + calibrate dilution/cover (the data the historical backtest could not provide). Cost ~a few $/mo, within Modal free credits.
