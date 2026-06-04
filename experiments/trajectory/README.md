# experiments/trajectory — CV trajectory pipeline (SKELETON)

Scaffold for extracting **per-horse** dense track trajectories from race **video
pixels** (worst case: no positional feed), to feed the physics energetics model
(`v_c`, `W'`) that today is fit from only 2–4 sectional splits.

Full design + rationale + roadmap: [`../../docs/trajectory/CV_PIPELINE.md`](../../docs/trajectory/CV_PIPELINE.md).

## Status

Stubs only. Every stage function raises `NotImplementedError` with an implementation
hint. The package **imports cleanly with no heavy deps** (cv2 / torch / ultralytics /
pyav / pandas / scipy are lazy-imported *inside* functions). Only the pure-stdlib
`schema.py` dataclasses are eagerly importable.

## Stages (data flow)

```
ingest → detect → track → identify → homography → project → export(+qa)
```

| File | Stage | Default tool |
|---|---|---|
| `ingest.py` | 1 Ingest: video → timestamped frames + race context | pyav (exact PTS) / cv2 |
| `detect.py` | 2 Detect: horse/marker boxes | YOLO v11 (→ RT-DETR for occlusion) |
| `track.py` | 3 Track: tracklets (local IDs) | BoT-SORT (→ ByteTrack/OC-SORT) |
| `identify.py` | 4 Identify: tracklet → saddlecloth/runner | PARSeq OCR + silks prior + Hungarian |
| `homography.py` | 5 Calibrate: pixel↔world H | markers (Mode B) / landmark+chain (Mode A) |
| `project.py` | 6/7 Project + Trajectory: (x,y)→(d,off), smooth | centreline spline from `track.py`; Kalman/RTS |
| `export.py` | 8/9 Export + QA | Parquet; QA vs published sectionals |
| `schema.py` | output schema | stdlib dataclasses |
| `pipeline.py` | orchestration | wires the stages (stubbed) |

> `track.py` here is the **pipeline stage** (tracking), not `src/hkjc/track.py` (the
> world geometry). This package consumes that geometry **read-only** and never edits `src/`.

## Recommended path

1. **Phase 0 — find the upstream `(x,y)` feed first** (Aerial Virtual View network
   recon). If it exists, skip CV entirely.
2. **If pixels only: start with Mode B** (virtual render) — near-rectified, legible
   numbers, exact on-screen markers → ~80% of the hard CV disappears.
3. **Mode A** (moving broadcast camera) is a research project; gate it behind a
   confirmed signal and accept a lower accuracy ceiling.

## Output schema

`RaceTrajectory` → per `RunnerTrajectory` → list of `TrajPoint(t, d, off, speed,
accel, x, y, gap, conf)` + `TrafficEvent`s, in `track.TrackGeometry` coordinates
(`d` = metres-from-finish, `off` = lateral metres outward from the current inner
rail). See `schema.py` for the dataclasses and the documented dict form.

## Smoke test (imports cleanly)

```python
from experiments.trajectory import RaceContext, RaceTrajectory, TrajPoint
ctx = RaceContext(venue="ST", date="2026-06-01", race_no=1, distance=1200.0)
# stage modules import without cv2/torch/ultralytics installed; calling a stage raises NotImplementedError
```
