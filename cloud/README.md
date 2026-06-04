# Cloud capture (always-on, no local machine needed)

The carryover flywheel must capture the live jackpot/odds trajectory during HK
meetings even when your laptop is off. This runs it server-side.

## Why Modal (the recommendation)

We researched the options:

| Option | Always-on? | Python? | Sub-hour cron? | Cost | Verdict |
|---|---|---|---|---|---|
| Claude `CronCreate` / `loop` | ❌ local/session-bound | — | yes | — | not cloud |
| Anthropic **Routines** | ✅ | — | ❌ **1-hour minimum** | sub usage | too coarse |
| **Modal** | ✅ | ✅ native | ✅ | **$30/mo free credits** | **chosen** |
| GitHub Actions | ✅ | ✅ | nominal 5-min, **unreliable** (10–30 min delays, drops) | free public / paid private | fallback |
| AWS Lambda + EventBridge | ✅ | ✅ | ✅ | free-tier | viable, more setup |

Modal wins: our package runs unchanged, cron is reliable and **scheduled in HK
time**, a persistent Volume holds the data, and this workload (a few hundred tiny
polls/week) costs ~nothing against the free credits.

## Deploy

```bash
pip install modal
modal token new                      # one-time auth
modal deploy cloud/modal_app.py      # registers the scheduled functions
```

That's it — `capture` runs every 5 min during HK 10:00–23:59 (self-gating, so
off-meeting polls are no-ops) and `settle` runs nightly at 23:30 HKT.

## Get the data back

```bash
modal volume get hkjc-data /carryover_ledger.parquet ./data_cache/
modal volume get hkjc-data /jackpot_ticks ./data_cache/ --recursive
```

Then analyse locally as usual (`CarryoverLedger("data_cache/carryover_ledger.parquet").read()`).

## Notes / first-deploy checks
- Modal `Cron(timezone=...)` schedules in HK time directly — no UTC math.
- `volume.commit()` after writes / `volume.reload()` before reads keep the shared
  Volume consistent across invocations.
- On first deploy, run `modal run cloud/modal_app.py` to smoke-test one capture.
- Values only populate while a meeting's pools are open; expect "no open jackpot
  pools — skipped" between meetings.

## GitHub Actions fallback
If you'd rather not add Modal: a scheduled workflow (`on: schedule: - cron`) can
run `uv run python experiments/capture_jackpots.py capture` and commit the parquet
back to the repo. Caveats: GitHub cron is best-effort (often 10–30 min late, can be
skipped under load — bad for catching the late-money trajectory), and **scheduled
workflows on private repos require a paid plan**.
