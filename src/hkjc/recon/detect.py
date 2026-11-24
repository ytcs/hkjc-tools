"""Avatar detection on the AVV render (self-supervised YOLO) + per-detection silk colour.

The detector was trained on the RIGHT HALF of the 1080p frame cropped above the legend strip, so it
is fed exactly that; box centres are returned in render-PANEL coords (``PanelBox``) so they compose
with the camera poses.
"""

from __future__ import annotations

import glob
import os

import numpy as np

CONF = 0.25
IOU = 0.5
LEGEND_BAND = 62  # px below the legend top occupied by number boxes + silk icons


def default_weights() -> str:
    env = os.environ.get("HKJC_AVATAR_WEIGHTS")
    if env:
        return env
    if os.environ.get("HKJC_DEVICE", "cpu").startswith("intel:"):
        return _openvino_model()  # Intel iGPU/NPU via OpenVINO (opt-in; centres within 0.5 px)
    if os.environ.get("HKJC_DEVICE", "cpu").startswith("cuda") and os.path.exists(
        "models/avv_avatar_yolo11n.pt"
    ):  # GPU path (opt-in; numerics differ slightly from the CPU ONNX path)
        return "models/avv_avatar_yolo11n.pt"
    for f in (
        "models/avv_avatar_yolo11n.onnx",
        "models/avv_avatar_yolo11n.pt",
    ):  # ONNX: faster on CPU
        if os.path.exists(f):
            return f
    for pat in (
        "runs/**/htag_yolo_joint/**/weights/best.pt",
        "**/htag_yolo_joint/**/weights/best.pt",
    ):
        c = glob.glob(pat, recursive=True)
        if c:
            return max(c, key=os.path.getmtime)
    raise FileNotFoundError("avatar detector weights not found (models/avv_avatar_yolo11n.pt)")


OV_IMGSZ = (544, 960)  # the render panel (945x535) letterboxed: a FIXED input for OpenVINO


def _openvino_model() -> str:
    """OpenVINO export of the avatar detector (created once from the .pt weights); box centres agree
    with the CPU ONNX path within 0.5 px."""
    out = "models/avv_avatar_yolo11n_openvino_model"
    if not os.path.exists(out):
        from ultralytics import YOLO

        YOLO("models/avv_avatar_yolo11n.pt").export(
            format="openvino", imgsz=list(OV_IMGSZ), half=False, verbose=False
        )
    return out


class AvatarDetector:
    def __init__(self, weights: str | None = None, imgsz: int = 960):
        from ultralytics import YOLO

        from .ortsession import no_spin

        no_spin()
        self.model = YOLO(weights or default_weights())
        self.imgsz = imgsz

    def raw(self, panel):
        """The detector's boxes on the render PANEL (945x535; ~2x fewer pixels than the letterboxed half frame):
        (xyxy (N, 4), conf (N,)) arrays, or None for no box."""
        dev = os.environ.get("HKJC_DEVICE", "cpu")
        ov = dev.startswith("intel:")
        r = self.model.predict(
            panel,
            conf=CONF,
            iou=IOU,
            imgsz=OV_IMGSZ if ov else self.imgsz,
            verbose=False,
            rect=not ov,
            device=dev,
        )[0]
        if len(r.boxes) == 0:
            return None
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    @staticmethod
    def drop_legend(raw, legend_row: int) -> np.ndarray:
        """``raw`` boxes -> (N, 6) [cx, cy, w, h, conf, 0] in panel px; legend icon band dropped."""
        if raw is None:
            return np.zeros((0, 6))
        xyxy, cf = raw
        cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
        keep = ~((cy > legend_row - 4) & (cy < legend_row + LEGEND_BAND))
        out = np.c_[cx, cy, xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1], cf, np.zeros(len(cf))]
        return out[keep]

    def __call__(self, full_img, box, legend_row: int) -> np.ndarray:
        """-> (N, 6) [cx, cy, w, h, conf, 0] in PANEL px. ``legend_row`` is panel-relative."""
        # Full panel: horses are visible in the turf strip below the legend; detections inside the
        # legend icon band are dropped (the detector fires on the silk icons).
        half_x = full_img.shape[1] // 2
        band = full_img[:, half_x:]
        r = self.model.predict(band, conf=CONF, iou=IOU, imgsz=self.imgsz, verbose=False)[0]
        if len(r.boxes) == 0:
            return np.zeros((0, 6))
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 + half_x - box.x0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 - box.y0
        keep = (
            ~((cy > legend_row - 4) & (cy < legend_row + LEGEND_BAND))
            & (cy >= 0)
            & (cy < box.y1 - box.y0)
            & (cx >= 0)
            & (cx < box.x1 - box.x0)
        )
        out = np.c_[cx, cy, xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1], cf, np.zeros(len(cf))]
        return out[keep]


def silk_hist(panel, det, bins: int = 12) -> np.ndarray:
    """Normalised hue histogram (+ white/black bins) of the vivid/white pixels inside a detection box
    — a rotation-invariant colour signature of the jockey silk/cap seen from above."""
    import cv2

    cx, cy, w, h = det[:4]
    x0, x1 = int(max(0, cx - w / 2)), int(min(panel.shape[1], cx + w / 2 + 1))
    y0, y1 = int(max(0, cy - h / 2)), int(min(panel.shape[0], cy + h / 2 + 1))
    hsv = cv2.cvtColor(panel[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    hh, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    turf = (hh > 28) & (hh < 60) & (s > 70) & (v > 60) & (v < 200)  # render turf
    vivid = (s > 90) & (v > 70) & ~turf
    white = (s < 40) & (v > 190)
    black = v < 40
    hist = np.zeros(bins + 2)
    if vivid.any():
        hist[:bins] = np.bincount((hh[vivid] * bins // 180).clip(0, bins - 1), minlength=bins)
    hist[bins] = white.sum()
    hist[bins + 1] = black.sum() * 0.3  # dark body/shadow is common to all: down-weight
    t = hist.sum()
    return hist / t if t > 0 else hist


SILK_NH = 12  # hue bins (x2: pastel / vivid) of the silk feature


def _silk_bins(hsv, dark_w):
    hh, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    white = (s < 40) & (v > 170)
    dark = v < 60
    past = (s >= 40) & (s < 110) & (v > 120) & ~dark
    viv = (s >= 110) & (v > 80)
    hb = (hh * SILK_NH // 180).clip(0, SILK_NH - 1)
    h = np.r_[
        np.bincount(hb[past], minlength=SILK_NH),
        np.bincount(hb[viv], minlength=SILK_NH),
        white.sum(),
        dark_w * dark.sum(),
    ].astype(float)
    t = h.sum()
    return h / t if t > 0 else h


def silk_feature(panel, det) -> np.ndarray:
    """Silk signature of a render avatar: pastel/vivid hue bins + white + (down-weighted) dark,
    excluding turf and horse-brown pixels (in ``silk_hist`` the brown body dominated and pastel silks
    counted as white, leaving self-vs-other matching at chance)."""
    import cv2

    cx, cy, w, h = det[:4]
    x0, x1 = int(max(0, cx - w / 2)), int(min(panel.shape[1], cx + w / 2 + 1))
    y0, y1 = int(max(0, cy - h / 2)), int(min(panel.shape[0], cy + h / 2 + 1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(2 * SILK_NH + 2)
    hsv = cv2.cvtColor(panel[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    hh, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    turf = (hh > 28) & (hh < 60) & (s > 50) & (v > 50) & (v < 210)
    brown = (hh <= 25) & (s > 60) & (s < 200) & (v > 40) & (v < 160)  # horse body / mane
    return _silk_bins(hsv[~turf & ~brown], dark_w=0.3)


def silk_prototype(gif_path) -> np.ndarray:
    """The same signature from the official silk image (a pre-race input; alpha-masked GIF)."""
    import cv2
    from PIL import Image

    a = np.array(Image.open(gif_path).convert("RGBA"))
    m = a[..., 3] > 0
    hsv = cv2.cvtColor(np.ascontiguousarray(a[..., :3][:, :, ::-1]), cv2.COLOR_BGR2HSV)[m]
    return _silk_bins(hsv, dark_w=0.3)


def bgsub_blobs(panel, back, valid, *, thr=60, min_a=40, max_a=1500, merge_px=16.0):
    """Surface-agnostic horse detection from the STATIC WORLD MAP: the map warped into the frame
    (``back``, horses averaged out) is differenced with the frame; blobs = horses (+ shadows).
    Shadow pixels dim all channels by a similar ratio (hue kept) and are excluded from the blob
    centre. Works where the turf-trained YOLO fails (AWT dirt). -> (N, 6) [cx, cy, w, h, conf, 1]."""
    import cv2

    p = panel.astype(np.float32)
    b = back.astype(np.float32) + 1.0
    d = np.abs(p - b).sum(2)
    fg = (d > thr) & (valid > 0) & (back.max(2) > 0)
    ratio = p / b
    rm = ratio.mean(2)
    shadow = (rm < 0.85) & ((ratio.max(2) - ratio.min(2)) < 0.12)
    m = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    n, lab, st, _cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if not (min_a <= a <= max_a and w < 80 and h < 80):
            continue
        ys, xs = np.nonzero(
            (lab[y : y + h, x : x + w] == k)
            & fg[y : y + h, x : x + w]
            & ~shadow[y : y + h, x : x + w]
        )
        if len(xs) < 10:
            continue
        cov = np.cov(np.c_[xs, ys].T) if len(xs) > 2 else np.eye(2)
        w, V = np.linalg.eigh(cov)
        out.append([x + xs.mean(), y + ys.mean(), len(xs), V[0, 1], V[1, 1], 4 * np.sqrt(w[1])])
    # one avatar can split into fragments (silk vs body colour vs shadow gap). Union blobs whose
    # centres are < merge_px apart, OR close ALONG the frame's dominant body axis (doubled-angle mean
    # of the blobs' major axes; horses run parallel) but tight ACROSS it: head-to-tail fragments sit
    # ~25 px apart, while horses abreast must not merge.
    ax = np.array([0.0, 0.0])
    L = 0.0
    if out:
        Ob = np.array(out)
        ang = np.arctan2(Ob[:, 4], Ob[:, 3])
        m2 = np.array([np.sum(Ob[:, 2] * np.cos(2 * ang)), np.sum(Ob[:, 2] * np.sin(2 * ang))])
        a0 = 0.5 * np.arctan2(m2[1], m2[0])
        ax = np.array([np.cos(a0), np.sin(a0)])
        L = float(np.median(Ob[:, 5]))
    c = [list(o[:3]) for o in out]

    def close(a, b):
        dv = np.array([b[0] - a[0], b[1] - a[1]])
        if np.hypot(*dv) < merge_px:
            return True
        al, ac = abs(dv @ ax), abs(dv[0] * ax[1] - dv[1] * ax[0])
        return al < 0.8 * L and ac < 0.2 * L

    merged = True
    while merged and len(c) > 1:
        merged = False
        P = np.array([[o[0], o[1]] for o in c])
        D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1)) + np.eye(len(c)) * 1e9
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        i = j = None
        for a_, b_ in order[: len(c) * 4]:
            if a_ < b_ and close(c[a_], c[b_]):
                i, j = int(a_), int(b_)
                break
        if i is not None:
            a, b = c[i], c[j]
            w = a[2] + b[2]
            c[i] = [(a[0] * a[2] + b[0] * b[2]) / w, (a[1] * a[2] + b[1] * b[2]) / w, w]
            del c[j]
            merged = True
    res = [[x, y, 2.4 * np.sqrt(a), 2.4 * np.sqrt(a), 0.5, 1.0] for x, y, a in c]
    return np.array(res).reshape(-1, 6)
