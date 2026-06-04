"""Detection probe: compare plain-YOLO vs SAHI-tiled inference on overhead drone frames.

The drone overhead view has ~12 horses at ~30-50 px each in a 1920x1080 frame, clustered
mid-frame. Stock yolov8n@640 detects ZERO. This probe quantifies, on a handful of
representative overhead frames, how many of the 12 horses each config finds, and writes
annotated crops so a human can eyeball box quality. Goal: pick the detection config for the
full pass.

Run:
  cd /home/ytc/Projects/hkjc-tools && uv run --with ultralytics --with sahi \
      --with opencv-python-headless python experiments/trajectory/detect_probe.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

VIDEO = Path("/home/ytc/Projects/hkjc-tools/data_cache/video/drone_20260603_01.mp4")
OUT = VIDEO.parent / "poc_out"
HORSE = 17  # COCO horse

# representative overhead frames (fractions of duration) — all confirmed overhead drone
PROBE_FRACS = [0.05, 0.15, 0.30, 0.40, 0.50, 0.60, 0.68]


def grab_frame(frac):
    import av
    container = av.open(str(VIDEO))
    vs = container.streams.video[0]
    total = vs.frames
    target = int(total * frac)
    idx = -1
    out = None
    for frame in container.decode(vs):
        idx += 1
        if idx >= target:
            out = frame.to_ndarray(format="bgr24")
            break
    container.close()
    return target, out


def run_plain(model, img, imgsz, conf):
    r = model.predict(img, classes=[HORSE], conf=conf, imgsz=imgsz, iou=0.5, verbose=False)
    b = r[0].boxes
    if b is None or len(b) == 0:
        return []
    return [(*xy, c) for xy, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())]


def run_sahi(det_model, img, tile, overlap, conf):
    from sahi.predict import get_sliced_prediction
    res = get_sliced_prediction(
        img, det_model,
        slice_height=tile, slice_width=tile,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        postprocess_type="GREEDYNMM", postprocess_match_threshold=0.5,
        verbose=0,
    )
    out = []
    for o in res.object_prediction_list:
        if o.category.id != HORSE:
            continue
        bb = o.bbox
        out.append((bb.minx, bb.miny, bb.maxx, bb.maxy, o.score.value))
    return out


def draw(img, dets, color=(0, 0, 255)):
    import cv2
    im = img.copy()
    for x1, y1, x2, y2, c in dets:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(im, f"{c:.2f}", (int(x1), int(y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(im, f"n={len(dets)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return im


def main():
    import cv2
    from ultralytics import YOLO
    from sahi import AutoDetectionModel

    OUT.mkdir(parents=True, exist_ok=True)
    model_name = os.environ.get("PROBE_MODEL", "yolov8x.pt")
    print(f"model = {model_name}")
    ymodel = YOLO(model_name)               # triggers download -> known weights path
    weights_path = str(ymodel.ckpt_path)
    print(f"weights = {weights_path}")

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights_path,
        confidence_threshold=float(os.environ.get("PROBE_SAHI_CONF", "0.20")),
        device="cpu", image_size=640,
    )

    configs_plain = [("plain1920", 1920, 0.10), ("plain1280", 1280, 0.10)]
    sahi_tile = int(os.environ.get("PROBE_TILE", "512"))
    sahi_ov = float(os.environ.get("PROBE_OV", "0.25"))

    summary = []
    for frac in PROBE_FRACS:
        fidx, img = grab_frame(frac)
        if img is None:
            continue
        row = {"frac": frac, "fidx": fidx}
        for name, imgsz, conf in configs_plain:
            t = time.time()
            d = run_plain(ymodel, img, imgsz, conf)
            row[name] = len(d)
            row[name + "_t"] = time.time() - t
            cv2.imwrite(str(OUT / f"probe_{name}_{int(frac*100):02d}.png"), draw(img, d))
        t = time.time()
        ds = run_sahi(sahi_model, img, sahi_tile, sahi_ov, float(os.environ.get("PROBE_SAHI_CONF", "0.20")))
        row[f"sahi{sahi_tile}"] = len(ds)
        row[f"sahi{sahi_tile}_t"] = time.time() - t
        cv2.imwrite(str(OUT / f"probe_sahi{sahi_tile}_{int(frac*100):02d}.png"), draw(img, ds, (0, 200, 0)))
        summary.append(row)
        print(f"frac {frac:.2f} f{fidx}: "
              + "  ".join(f"{k}={row[k]}" for k in row if not k.endswith("_t") and k not in ("frac", "fidx"))
              + f"  | times "
              + "  ".join(f"{k.replace('_t','')}={row[k]:.1f}s" for k in row if k.endswith("_t")))

    print("\n=== SUMMARY (detections out of 12 horses) ===")
    keys = [k for k in summary[0] if k not in ("frac", "fidx") and not k.endswith("_t")]
    print("frac   fidx  " + "  ".join(f"{k:>10}" for k in keys))
    for r in summary:
        print(f"{r['frac']:.2f}  {r['fidx']:5d}  " + "  ".join(f"{r[k]:>10}" for k in keys))


if __name__ == "__main__":
    main()
