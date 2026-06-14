"""Train a single-class AVV avatar detector (Ultralytics YOLO, already a `vision` dep).

Trained on the pipeline's own clean-zone pseudo-labels (cv_htag_nn_labels.py). The avatars are
consistent synthetic sprites, so a nano model converges fast; high imgsz keeps the small (15-40 px)
avatars resolvable. CPU only here (no CUDA at runtime) — runs in the background.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_nn_train.py
Out:  data_cache/htag_yolo_{RACE}/runs/.../weights/best.pt
"""
from __future__ import annotations

import os

RACE = os.environ.get("RACE", "R08")          # a race, or "joint" for the multi-race detector
DATA = f"data_cache/htag_yolo_{RACE}/data.yaml"


def main():
    import torch
    from ultralytics import YOLO

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training avatar detector on {dev}; data={DATA}")
    try:
        model = YOLO("yolo11n.pt")            # pretrained init (auto-download)
    except Exception as exc:                   # offline -> from scratch
        print(f"pretrained unavailable ({exc}); training from scratch")
        model = YOLO("yolo11n.yaml")
    epochs = int(os.environ.get("HTAG_EPOCHS", 40))
    model.train(
        data=DATA, epochs=epochs, imgsz=960, batch=16, device=dev, patience=10,
        cache=True, workers=4, project=f"data_cache/htag_yolo_{RACE}/runs", name="avatar",
        exist_ok=True, verbose=True,
        # small, high-contrast synthetic sprites: mild aug, no mosaic distortion late
        mosaic=0.5, close_mosaic=8, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,
        degrees=15.0, translate=0.05, scale=0.4, fliplr=0.0, flipud=0.0,
        # degrees/scale aug simulate the bend rotation + zoom-out the clean-zone labels lack;
        # NO flips (the render's shadow/orientation geometry is consistent, a cue to keep)
    )
    print("done; best weights under data_cache/htag_yolo_%s/runs/avatar/weights/best.pt" % RACE)


if __name__ == "__main__":
    main()
