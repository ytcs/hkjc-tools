"""Probe: can we segment the white running rail cleanly in the overhead frames?

The inner running rail (the bright continuous white fence the field races along) is the
key homography landmark. This probe extracts a few overhead frames, thresholds for the
white rail, and overlays the detected rail pixels + a fitted rail polyline so a human can
judge whether rail-anchored homography is viable.

Run:
  cd /home/ytc/Projects/hkjc-tools && uv run --with opencv-python-headless --with av \
      --with matplotlib python experiments/trajectory/rail_probe.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

VIDEO = Path("/home/ytc/Projects/hkjc-tools/data_cache/video/drone_20260603_01.mp4")
OUT = VIDEO.parent / "poc_out"
FRACS = [0.05, 0.20, 0.40, 0.60]


def grab(frac):
    import av
    c = av.open(str(VIDEO)); vs = c.streams.video[0]
    tgt = int(vs.frames * frac); idx = -1; out = None
    for f in c.decode(vs):
        idx += 1
        if idx >= tgt:
            out = f.to_ndarray(format="bgr24"); break
    c.close(); return out


def white_mask(img):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # white = low saturation, high value
    m = cv2.inRange(hsv, (0, 0, 175), (180, 60, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def main():
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    for frac in FRACS:
        img = grab(frac)
        if img is None:
            continue
        m = white_mask(img)
        vis = img.copy()
        vis[m > 0] = (0, 0, 255)
        # connected components, keep large thin ones (rails)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if area > 800 and max(w, h) > 300:   # long structure
                ys, xs = np.where(lab == i)
                cv2.putText(vis, f"rail?{area}", (int(xs.mean()), int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(OUT / f"rail_probe_{int(frac*100):02d}.png"), vis)
        print(f"frac {frac}: white px={int((m>0).sum())} components={n}")
    print("wrote rail_probe_*.png")


if __name__ == "__main__":
    main()
