"""CV feasibility probe — "watch the races" (path-extraction vs gait analysis).

Disposable scratch (studies/). Consolidates the 2026-06 CV feasibility recon into one
reproducible script. It does NOT touch src/hkjc/model — it only reads the existing
sample videos in data_cache/video/ and the hkjc.vision schema, and prints a verdict.

What it establishes (run with --group vision):

  1. The overhead "aerial/virtual" view (what the hkjc.vision scaffold targets) is the
     WRONG substrate for off-the-shelf CV: horses are ~20px, top-down, packed and
     motion-blurred against uniform turf -> generic YOLO returns ~0 horses on most
     frames (see data_cache/video/poc_out/extract_run.log: 0 det on 88/111 frames,
     longest track 19 frames, 134 phantom ids for a ~12 field). The (d,off) projection
     in traj_drone_*.npz "works" but coverage is far too sparse to be a feature.

  2. The broadcast / RF SIDE view is the tractable substrate: the SAME generic YOLOv8x
     finds ~7 horses + ~11 jockeys per frame (full field), boxes 200-460px wide, on-
     screen saddlecloth numbers + a silks legend for identity. This is where
     detect->track->identify is actually viable.

  3. GAIT from IN-RACE footage is hard: at racing pace the side view has heavy
     horse-on-horse occlusion, motion blur, and rail-occluded lower legs (see the
     /tmp horse crop this script writes). Clean gait needs slow, isolated subjects ->
     PARADE-RING or BARRIER-TRIAL walk-up footage, a different (and lighter) source.

Usage:
    uv run --group vision python studies/cv_feasibility_probe.py

Outputs (under data_cache/video/poc_out/cv_probe/): annotated side-frame, biggest-horse
crop, and a printed comparison table. Heavy deps (cv2/ultralytics) are lazy-imported.
"""
from __future__ import annotations

import collections
import os

OUT = "data_cache/video/poc_out/cv_probe"
SIDE_VIDEO = "data_cache/video/rf_20250406_09.mp4"        # broadcast / RF side view
OVERHEAD_VIDEO = "data_cache/video/drone_20260603_01.mp4"  # aerial/virtual overhead view


def _detect_frame(model, frame, conf=0.25):
    """Return (counts_by_class, horse_boxes) for one BGR frame."""
    r = model(frame, conf=conf, verbose=False)[0]
    names = r.names
    counts = collections.Counter(names[int(c)] for c in r.boxes.cls.tolist())
    horses = []
    for b in r.boxes:
        if names[int(b.cls)] == "horse":
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            horses.append((x1, y1, x2, y2, float(b.conf)))
    return counts, horses, r


def main() -> None:
    import cv2
    from ultralytics import YOLO

    os.makedirs(OUT, exist_ok=True)
    model = YOLO("yolov8x.pt")  # generic COCO model: has a 'horse' class, no fine-tune

    print("=" * 72)
    print("CV FEASIBILITY PROBE — overhead vs side view, generic YOLOv8x (COCO)")
    print("=" * 72)

    results = {}
    for label, path, fidx in [
        ("OVERHEAD (aerial/virtual)", OVERHEAD_VIDEO, 1500),
        ("SIDE (broadcast/RF)", SIDE_VIDEO, 150),
    ]:
        if not os.path.exists(path):
            print(f"  [skip] {label}: {path} missing")
            continue
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  [skip] {label}: could not read frame {fidx}")
            continue
        counts, horses, r = _detect_frame(model, frame)
        sizes = sorted(((int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2, _ in horses),
                       key=lambda wh: -wh[0] * wh[1])
        print(f"\n{label}  ({path}, frame {fidx})")
        print(f"  detections by class : {dict(counts)}")
        print(f"  horse boxes (w x h) : {sizes[:8]}")
        results[label] = (len(horses), counts.get("person", 0), sizes[0] if sizes else (0, 0))
        # save an annotated frame + the biggest-horse crop (gait substrate check)
        ann = r.plot()
        tag = "overhead" if "OVERHEAD" in label else "side"
        cv2.imwrite(f"{OUT}/probe_{tag}.png", ann)
        if horses and "SIDE" in label:
            x1, y1, x2, y2, _ = max(horses, key=lambda h: (h[2] - h[0]) * (h[3] - h[1]))
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(f"{OUT}/biggest_horse_crop.png", crop)
            print(f"  biggest-horse crop  : {crop.shape} -> {OUT}/biggest_horse_crop.png")
            print("    (inspect it: in-race side view has horse-on-horse occlusion +")
            print("     motion blur + rail-occluded legs -> NOT clean enough for gait.")
            print("     Gait needs parade-ring / barrier-trial walk-up footage instead.)")

    print("\n" + "-" * 72)
    print("VERDICT")
    print("-" * 72)
    ov = results.get("OVERHEAD (aerial/virtual)")
    sd = results.get("SIDE (broadcast/RF)")
    if ov and sd:
        print(f"  overhead: {ov[0]} horses, {ov[1]} jockeys, biggest box {ov[2]}")
        print(f"  side    : {sd[0]} horses, {sd[1]} jockeys, biggest box {sd[2]}")
        print("  => Side view is the tractable substrate; overhead is not (off-the-shelf).")
    print("""
  Path-extraction: feasible ONLY on the side view, and only with a fine-tuned
    detector + saddlecloth-OCR identity + per-frame homography. The (d,off) output
    is a TRAILING per-horse trait (computed from the race itself), so it is
    leak-free for FUTURE races the same way form is: ground_lost / path_efficiency
    averaged over a horse's past runs, joined on horse_id. High ceiling, high cost.

  Gait: needs a different source (parade-ring / barrier-trial walk-up), not in-race.
    A single-frame conformation/soundness read is pre-race & leak-free by construction,
    and is the only CV signal that helps DEBUTANTS (blank form). Lower integration
    cost per race once the footage is sourced, but the footage-sourcing + animal-pose
    model is the open piece.
""")


if __name__ == "__main__":
    main()
