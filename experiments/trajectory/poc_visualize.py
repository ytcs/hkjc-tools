"""ONE-RACE CV proof-of-concept — overlay horse detections+tracks on the real footage.

Purpose: let a human EYEBALL extraction quality before any scale-out. This is deliberately
minimal and self-contained (NOT the production pipeline): a stock COCO detector (YOLO 'horse'
class) + the tracker's built-in ByteTrack, drawn back onto the video, plus a pixel-space
trajectory plot. NO homography / track-coordinate projection yet — the first quality gate is
"are horses reliably detected and is identity stable across the run?". Projection to (d, off)
comes only once detection/tracking look good.

Run (ephemeral heavy deps, nothing added to pyproject):
  uv run --with ultralytics --with opencv-python-headless --with matplotlib \
          --with av --with imageio-ffmpeg \
      python experiments/trajectory/poc_visualize.py data_cache/video/rf_20250406_09.mp4

Decoding uses PyAV (system ffmpeg) for robustness; cv2 only draws. Outputs into
data_cache/video/poc_out/: annotated_*.mp4, montage_*.png, tracks_*.png + stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _frames(video, start, stop, stride):
    """Yield (frame_idx, BGR ndarray) for frames in [start, stop) at the given stride."""
    import av
    container = av.open(str(video))
    vs = container.streams.video[0]
    idx = -1
    for frame in container.decode(vs):
        idx += 1
        if idx < start:
            continue
        if idx >= stop:
            break
        if (idx - start) % stride != 0:
            continue
        yield idx, frame.to_ndarray(format="bgr24")
    container.close()

import os

HORSE_CLASS = 17          # COCO 'horse'
MAX_FRAMES = int(os.environ.get("POC_MAXFRAMES", "450"))
STRIDE = int(os.environ.get("POC_STRIDE", "2"))
WINDOW = (0.45, 0.95)     # fraction of the video to scan (skip parade, keep the run+finish)
TRAIL = 25                # frames of trail to draw
IMGSZ = int(os.environ.get("POC_IMGSZ", "640"))    # inference resolution; drone needs ~1920 (tiny horses)
CONF = float(os.environ.get("POC_CONF", "0.25"))   # detection threshold; lower for small objects
MODEL = os.environ.get("POC_MODEL", "yolov8n.pt")


def _color(i: int):
    rng = np.random.default_rng(i * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def main():
    import av
    import cv2
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    video = Path(sys.argv[1] if len(sys.argv) > 1 else "data_cache/video/rf_20250406_09.mp4").resolve()
    out = video.parent / "poc_out"
    out.mkdir(parents=True, exist_ok=True)
    tag = video.stem

    probe = av.open(str(video))
    vs = probe.streams.video[0]
    fps = float(vs.average_rate or 25.0)
    total = vs.frames or int(float(vs.duration * vs.time_base) * fps)
    W, H = vs.codec_context.width, vs.codec_context.height
    probe.close()
    start = int(total * WINDOW[0]); stop = int(total * WINDOW[1])
    print(f"video {tag}: {W}x{H}, {fps:.1f}fps, {total} frames ({total/fps:.0f}s); "
          f"scanning {start}..{stop} stride {STRIDE}, cap {MAX_FRAMES}")

    model = YOLO(MODEL)
    writer = imageio.get_writer(str(out / f"annotated_{tag}.mp4"),
                                fps=max(fps / STRIDE, 8), codec="libx264", quality=7,
                                macro_block_size=8)

    tracks: dict[int, list] = {}     # id -> [(frame_idx, cx, cy, conf)]
    montage_saved = []
    processed = 0
    for fidx, frame in _frames(video, start, stop, STRIDE):
        if processed >= MAX_FRAMES:
            break
        res = model.track(frame, persist=True, classes=[HORSE_CLASS], conf=CONF,
                          iou=0.5, imgsz=IMGSZ, tracker="bytetrack.yaml", verbose=False)
        boxes = res[0].boxes
        n_det = 0
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            n_det = len(ids)
            for tid, (x1, y1, x2, y2), cf in zip(ids, xyxy, confs):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tracks.setdefault(int(tid), []).append((processed, float(cx), float(cy), float(cf)))
                col = _color(int(tid))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(frame, f"#{tid}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                trail = [(p[1], p[2]) for p in tracks[int(tid)] if p[0] > processed - TRAIL]
                if len(trail) > 1:
                    cv2.polylines(frame, [np.array(trail, np.int32)], False, col, 2)
        cv2.putText(frame, f"f{fidx}  det={n_det}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if processed in (0, MAX_FRAMES // 3, 2 * MAX_FRAMES // 3) or (n_det >= 6 and len(montage_saved) < 4):
            pth = out / f"montage_{tag}_{processed:03d}.png"
            cv2.imwrite(str(pth), frame); montage_saved.append(pth)
        processed += 1
    writer.close()

    # stats + pixel-space trajectory plot
    lengths = sorted((len(v) for v in tracks.values()), reverse=True)
    long_tracks = [t for t in lengths if t >= 10]
    print(f"\nprocessed {processed} frames | {len(tracks)} track-ids | "
          f"lengths(top10)={lengths[:10]} | tracks≥10frames={len(long_tracks)}")
    print(f"median det/frame≈{np.median([sum(1 for v in tracks.values() if any(p[0]==f for p in v)) for f in range(processed)]):.1f}")

    fig, ax = plt.subplots(figsize=(12, 7))
    for tid, pts in tracks.items():
        if len(pts) < 8:
            continue
        a = np.array([(p[1], p[2]) for p in pts])
        c = np.array(_color(tid)) / 255.0
        ax.plot(a[:, 0], a[:, 1], "-", color=c, lw=1.5, alpha=0.9)
        ax.scatter(a[0, 0], a[0, 1], color=c, s=18)
        ax.text(a[-1, 0], a[-1, 1], f"#{tid}", color=c, fontsize=8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_title(f"{tag}: pixel-space horse tracks (PoC, no homography) — {len(long_tracks)} stable tracks")
    ax.set_xlabel("image x (px)"); ax.set_ylabel("image y (px)")
    fig.tight_layout(); fig.savefig(out / f"tracks_{tag}.png", dpi=110)
    print(f"\nwrote: {out}/annotated_{tag}.mp4, tracks_{tag}.png, {len(montage_saved)} montage PNGs")


if __name__ == "__main__":
    main()
