"""Always-on cloud capture for the Jackpot Carryover flywheel (Modal).

Why Modal: Python-native (our package runs unchanged), reliable sub-hour cron,
persistent Volumes, and a free $30/mo credit tier that dwarfs this tiny workload —
so the flywheel runs without your machine being on. (Claude's own scheduling can't
do this: CronCreate/loop are local/session-bound; Anthropic Routines cap at a
1-hour minimum. GitHub Actions is the fallback but its cron is unreliable and needs
a paid plan for private repos.)

Deploy:
    pip install modal && modal token new
    modal deploy cloud/modal_app.py
    modal volume get hkjc-data /carryover_ledger.parquet ./data_cache/   # pull data

Schedules are in **HK time** via Modal's Cron timezone support, so they line up
with race meetings directly. Capture self-gates (only writes when pools are open),
so polling outside meetings is a cheap no-op.
"""
import modal

app = modal.App("hkjc-jackpot-capture")
volume = modal.Volume.from_name("hkjc-data", create_if_missing=True)
DATA = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")   # deps stay in sync with the project
    .add_local_python_source("hkjc")                # ship the local package source
)


@app.function(
    image=image, volumes={DATA: volume}, timeout=180,
    schedule=modal.Cron("*/5 10-23 * * *", timezone="Asia/Hong_Kong"),  # HK meeting hours
)
def capture():
    from hkjc.data.capture import capture_once
    from hkjc.data.ticks import JackpotCaptureStore, OddsTickStore

    volume.reload()
    js = JackpotCaptureStore(f"{DATA}/jackpot_ticks")
    os_ = OddsTickStore(f"{DATA}/odds_ticks")
    stored = capture_once(js, os_)
    volume.commit()
    print("stored jackpot snapshot + win odds" if stored else "no open jackpot pools — skipped")


@app.function(
    image=image, volumes={DATA: volume}, timeout=600,
    schedule=modal.Cron("30 23 * * *", timezone="Asia/Hong_Kong"),  # nightly, after cards
)
def settle():
    from hkjc.data.capture import settle_recent
    from hkjc.data.carryover_ledger import CarryoverLedger
    from hkjc.data.ticks import JackpotCaptureStore

    volume.reload()
    js = JackpotCaptureStore(f"{DATA}/jackpot_ticks")
    led = CarryoverLedger(f"{DATA}/carryover_ledger.parquet")
    n = settle_recent(js, led)
    volume.commit()
    print(f"settled {n} captured meetings into the ledger")


@app.local_entrypoint()
def main():
    """`modal run cloud/modal_app.py` — one-off capture to smoke-test the deploy."""
    capture.remote()
