# CLAUDE.md — working guide for this repo

HKJC race reconstruction: race facts + the four public replay videos -> per-horse 2D paths.
The library is deliberately lean — two packages:

- `hkjc.data`  — everything read/downloaded from HKJC (`race_info`, replay videos, silks, track
  geometry, racecard, live, form, ...). Acquisition only.
- `hkjc.recon` — race info + videos -> paths (`RaceRun`, cached stages). Method/state:
  **`docs/RECON.md`** (read it first). Prior modelling/betting code was removed 2026-11 (in git
  history) and will be rebuilt on top of the reconstructed paths later.

## Principles

- **Feeds are proposals; the path is a posterior.** Every video is an incomplete, noisy proposal;
  paths come from Bayesian fusion with physics (bounded acceleration) and geometry (rails, course
  width) priors — not one master feed plus corrections.
- **Official outcome data is validation-only** (placings, times, sectionals, running positions).
  Pre-race facts (distance, course, draw, silks) may be inputs.
- **Look at the pixels.** Validate CV by reprojecting the reconstruction onto raw frames; numbers
  hide failures. Fixed acceptance gates in `recon/validate.py` — never loosen after seeing numbers;
  report failures as failures.
- **Record rejected approaches** (and why) in docstrings/docs so they are not re-tried.

## Running

```bash
uv sync --group vision
PYTHONHASHSEED=0 uv run pytest -q            # tests
uv run ruff check src tests                  # lint (must stay clean)
scripts/capped.sh uv run --group vision python -m hkjc.recon 20260603 HV 8   # one race
scripts/run_validation_set.sh                # 10-race validation set
```

Heavy jobs MUST run under `scripts/capped.sh` (systemd MemoryMax; the machine has 15 GB and the
session has crashed from OOM before). `/tmp` is RAM-backed tmpfs shared with other agents — keep
scratch small and clean it. Data/caches live in `data_cache/` (git-ignored); detector weights in
`models/`. GPU: RunPod is authorized (state the price before creating anything billable).

## Journal

`journal/entries/NNNN-*.md` is the durable research trail (plain markdown with a short header).
Add an entry for each substantive finding; `docs/RECON.md` holds the current consolidated state.
