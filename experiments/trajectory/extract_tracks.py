"""STAGE 1-3 — ingest + detect + track over the OVERHEAD drone window, dump raw tracks.

The drone MP4 (HV 2026-06-03 R1) is a multi-shot edit: a continuous OVERHEAD drone
follow-shot from the start (~t=0, gate) to ~t=102s (home turn), then it cuts to
ground-level broadcast finish cameras. Only the overhead window is homography-friendly,
so we scan ONLY [T0, T1] (seconds).

Detector decision (from detect_probe.py): plain YOLO yolov8x @ imgsz=1920, conf~0.15 is
the best config on CPU — SAHI tiling gave no accuracy gain at 3x the cost. Detection is
near-perfect (12/12) when the field is laterally spread, and collapses (~2) when the field
bunches single-file going away on a bend; the tracker's motion model bridges those.

We also crop each detection's silk ROI (upper third of the box) and store its mean
Lab colour, for the IDENTITY stage (silk-colour matching — saddlecloth numbers are NOT
legible in the overhead view).

Output: poc_out/tracks_raw_<tag>.npz with per-detection rows
  (frame_idx, t_sec, track_id, x1,y1,x2,y2, conf, foot_x, foot_y, L,a,b)
plus an annotated mp4 and a pixel-space plot.

Run:
  cd /home/ytc/Projects/hkjc-tools && uv run --with ultralytics \
      --with opencv-python-headless --with av --with imageio --with imageio-ffmpeg \
      --with matplotlib python experiments/trajectory/extract_tracks.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

VIDEO = Path("/home/ytc/Projects/hkjc-tools/data_cache/video/drone_20260603_01.mp4")
OUT = VIDEO.parent / "poc_out"
HORSE = 17

T0 = float(os.environ.get("EX_T0", "3.0"))      # race jump ~t=4-5s; start a touch earlier
T1 = float(os.environ.get("EX_T1", "102.0"))    # overhead window ends ~102s (then broadcast)
STRIDE = int(os.environ.get("EX_STRIDE", "3"))  # process every Nth frame (25fps -> ~8.3Hz)
IMGSZ = int(os.environ.get("EX_IMGSZ", "1920"))
CONF = float(os.environ.get("EX_CONF", "0.15"))
MODEL = os.environ.get("EX_MODEL", "yolov8x.pt")
MAXF = int(os.environ.get("EX_MAXF", "100000"))
WRITE_VIDEO = os.environ.get("EX_VIDEO", "1") == "1"   # set 0 to skip annotated mp4 (saves RAM)
MONT_EVERY = int(os.environ.get("EX_MONT_EVERY", "20"))


def _color(i):
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

    OUT.mkdir(parents=True, exist_ok=True)
    tag = VIDEO.stem

    container = av.open(str(VIDEO))
    vs = container.streams.video[0]
    fps = float(vs.average_rate or 25.0)
    W, Hh = vs.codec_context.width, vs.codec_context.height
    f0, f1 = int(T0 * fps), int(T1 * fps)
    print(f"video {tag}: {W}x{Hh} {fps:.1f}fps; overhead window t[{T0},{T1}]s "
          f"= frames [{f0},{f1}] stride {STRIDE}; model {MODEL} imgsz {IMGSZ} conf {CONF}")

    model = YOLO(MODEL)
    writer = None
    if WRITE_VIDEO:
        writer = imageio.get_writer(str(OUT / f"annotated_overhead_{tag}.mp4"),
                                    fps=max(fps / STRIDE, 8), codec="libx264", quality=7,
                                    macro_block_size=8)

    rows = []          # detection records
    tracks = {}        # id -> list[(proc_idx, foot_x, foot_y)]
    montage = []
    idx = -1
    proc = 0
    for frame in container.decode(vs):
        idx += 1
        if idx < f0:
            continue
        if idx >= f1 or proc >= MAXF:
            break
        if (idx - f0) % STRIDE != 0:
            continue
        img = frame.to_ndarray(format="bgr24")
        t_sec = idx / fps
        res = model.track(img, persist=True, classes=[HORSE], conf=CONF, iou=0.5,
                          imgsz=IMGSZ, tracker="bytetrack.yaml", verbose=False)
        b = res[0].boxes
        n = 0
        if b is not None and b.id is not None:
            ids = b.id.cpu().numpy().astype(int)
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            n = len(ids)
            for tid, (x1, y1, x2, y2), c in zip(ids, xyxy, cf):
                fx, fy = (x1 + x2) / 2.0, y2            # foot = bottom-centre (ground contact)
                # silk ROI = upper third of box (rider torso)
                ry1, ry2 = int(y1), int(y1 + (y2 - y1) * 0.45)
                rx1, rx2 = int(x1), int(x2)
                roi = img[max(0, ry1):max(1, ry2), max(0, rx1):max(1, rx2)]
                if roi.size:
                    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
                else:
                    lab = np.array([0, 128, 128], float)
                rows.append((idx, t_sec, int(tid), float(x1), float(y1), float(x2),
                             float(y2), float(c), float(fx), float(fy),
                             float(lab[0]), float(lab[1]), float(lab[2])))
                tracks.setdefault(int(tid), []).append((proc, float(fx), float(fy)))
                col = _color(int(tid))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(img, f"{tid}", (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                tr = [(p[1], p[2]) for p in tracks[int(tid)] if p[0] > proc - 30]
                if len(tr) > 1:
                    cv2.polylines(img, [np.array(tr, np.int32)], False, col, 2)
        cv2.putText(img, f"t={t_sec:5.1f}s f{idx} det={n}", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if writer is not None:
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if proc % MONT_EVERY == 0:
            p = OUT / f"ovr_montage_{tag}_{proc:04d}.png"
            cv2.imwrite(str(p), img); montage.append(p)
        proc += 1
        if proc % 10 == 0:
            print(f"  proc {proc}  t={t_sec:5.1f}s  det={n}  ids={len(tracks)}", flush=True)
        if proc % 40 == 0:   # checkpoint npz so a partial run is usable
            np.savez_compressed(OUT / f"tracks_raw_{tag}.npz", rows=np.array(rows, float),
                                cols=np.array(["frame", "t", "id", "x1", "y1", "x2", "y2",
                                               "conf", "fx", "fy", "L", "a", "b"]),
                                fps=fps, W=W, H=Hh, T0=T0, T1=T1, stride=STRIDE)
    if writer is not None:
        writer.close()
    container.close()

    arr = np.array(rows, dtype=np.float64)
    np.savez_compressed(OUT / f"tracks_raw_{tag}.npz", rows=arr,
                        cols=np.array(["frame", "t", "id", "x1", "y1", "x2", "y2",
                                       "conf", "fx", "fy", "L", "a", "b"]),
                        fps=fps, W=W, H=Hh, T0=T0, T1=T1, stride=STRIDE)

    # stats
    per_frame = {}
    for r in rows:
        per_frame.setdefault(r[0], 0)
        per_frame[r[0]] += 1
    dets = np.array(list(per_frame.values()))
    lengths = sorted((len(v) for v in tracks.values()), reverse=True)
    print(f"\n== STAGE 1-3 DONE ==")
    print(f"processed {proc} frames | {len(rows)} detections | {len(tracks)} track-ids")
    print(f"det/frame: median={np.median(dets):.1f} mean={dets.mean():.1f} "
          f"max={dets.max()} min={dets.min()} | frames with>=10 det: "
          f"{(dets>=10).sum()}/{len(dets)} ({100*(dets>=10).mean():.0f}%)")
    print(f"track lengths (top15): {lengths[:15]}")
    print(f"tracks >=20 frames: {sum(1 for l in lengths if l>=20)}; "
          f">=50: {sum(1 for l in lengths if l>=50)}")

    # pixel-space plot
    fig, ax = plt.subplots(figsize=(13, 7))
    for tid, pts in tracks.items():
        if len(pts) < 15:
            continue
        a = np.array([(p[1], p[2]) for p in pts])
        c = np.array(_color(tid)) / 255.0
        ax.plot(a[:, 0], a[:, 1], "-", color=c, lw=1.3, alpha=0.85)
        ax.text(a[-1, 0], a[-1, 1], f"{tid}", color=c, fontsize=7)
    ax.set_xlim(0, W); ax.set_ylim(Hh, 0); ax.set_aspect("equal")
    ax.set_title(f"{tag}: overhead pixel-space foot-point tracks "
                 f"({sum(1 for l in lengths if l>=15)} tracks >=15 frames)")
    ax.set_xlabel("image x (px)"); ax.set_ylabel("image y (px)")
    fig.tight_layout(); fig.savefig(OUT / f"ovr_tracks_{tag}.png", dpi=110)
    print(f"wrote tracks_raw_{tag}.npz, annotated_overhead_{tag}.mp4, "
          f"ovr_tracks_{tag}.png, {len(montage)} montages")


if __name__ == "__main__":
    main()
