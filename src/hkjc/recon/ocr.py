"""Burnt-in text on the replays, read with RECOGNITION-ONLY OCR (RapidOCR, no text detection) on
fixed-position crops: faster than det+rec and more accurate on these boxes (det splits "850M"
into fragments).

AVV render (1080p): left-panel race clock, the odometer box ("850M" = leader's distance-to-go,
a round-up display: it ticks V+10 -> V as the leader REACHES V), the legend (top-8 running order,
8 fixed-pitch number boxes).
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from . import video

AVV_ODO = (286, 320, 1606, 1700)  # y0, y1, x0, x1 (absolute, 1080p)
_CLOCK = re.compile(r"(\d):(\d\d)[.:](\d\d)")
_ODO = re.compile(r"(\d{1,4})\s*M", re.IGNORECASE)


@lru_cache(maxsize=1)
def _engine():
    import os

    from rapidocr_onnxruntime import RapidOCR

    from .ortsession import no_spin

    no_spin()
    n = int(os.environ.get("HKJC_OCR_THREADS", "-1"))  # parallel races: cap threads per process
    eng = RapidOCR(intra_op_num_threads=n, inter_op_num_threads=1) if n > 0 else RapidOCR()
    dev = os.environ.get("HKJC_DEVICE", "cpu")
    if dev.startswith("intel:"):  # the SAME PP-OCRv4 recogniser compiled for the Intel iGPU/NPU
        eng.text_rec.session = _OVSession(eng.text_rec.session.session._model_path, dev)
    if os.environ.get("HKJC_OCR_DEVICE", "cpu").startswith("cuda"):  # opt-in; NOT bit-exact vs CPU
        eng.text_rec.session = _CudaSession(eng.text_rec.session.session._model_path)
    return eng


class _CudaSession:
    """Drop-in for RapidOCR's ONNX session: the same recogniser on ONNX Runtime's CUDA provider
    (needs onnxruntime-gpu). Fails loudly rather than falling back to the CPU provider."""

    def __init__(self, model_path):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):  # CUDA/cuDNN from the nvidia-* wheels, if present
            ort.preload_dlls()
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("HKJC_OCR_DEVICE=cuda but ONNX Runtime has no CUDA provider")
        self.name = self.session.get_inputs()[0].name

    def __call__(self, x):
        return self.session.run(None, {self.name: x})


class _OVSession:
    """Drop-in for RapidOCR's ONNX session: same model file, OpenVINO-compiled (dynamic width)."""

    def __init__(self, model_path, dev):
        import openvino as ov

        core = ov.Core()
        self.model = core.compile_model(core.read_model(model_path), dev.split(":", 1)[1].upper())

    def __call__(self, x):
        return [self.model(x)[0]]


def rec_batch(crops, scale: int = 3) -> list[tuple[str, float]]:
    """Recognition-only OCR of many crops in ONE batched call."""
    import cv2

    if not crops:
        return []
    ups = [cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) for c in crops]
    rec = _engine().text_rec
    rec.rec_batch_num = 32
    res, _ = rec(ups)
    return [(str(t), float(c)) for t, c in res]


def _parse_clock(txt, _conf=1.0) -> float | None:
    m = _CLOCK.search(txt.replace(" ", ""))
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100 if m else None


def _parse_odometer(txt, conf, max_m: int = 4000) -> float | None:
    m = _ODO.search(txt.replace(" ", ""))
    if not m or conf < 0.6:
        return None
    v = float(m.group(1))
    return v if v <= max_m else None


def _legend_slot_crops(panel) -> list:
    lt = video.legend_top(panel)
    W = panel.shape[1]
    pitch = W / 8.8
    out = []
    for k in range(8):
        cx = int(0.064 * W + k * pitch)
        out.append(panel[max(0, lt - 4) : lt + 34, max(0, cx - 22) : cx + 16].copy())
    return out


def _parse_legend(res, n_field: int = 14) -> np.ndarray:
    row = np.zeros(8, int)
    for k, (txt, conf) in enumerate(res):
        dg = "".join(ch for ch in txt if ch.isdigit())
        if dg and conf > 0.6 and 1 <= int(dg) <= n_field:
            row[k] = int(dg)
    return row


def _legend_boxes(panel):
    """Pixels of the 8 legend number boxes (opaque; they change only on overtakes)."""
    lt = video.legend_top(panel)
    W = panel.shape[1]
    pitch = W / 8.8
    crops = [
        panel[
            max(0, lt - 4) : lt + 34,
            max(0, int(0.064 * W + k * pitch) - 22) : int(0.064 * W + k * pitch) + 16,
        ]
        for k in range(8)
    ]
    return np.concatenate([c.reshape(-1, 3) for c in crops])


def read_render_hud_cached(
    cache, *, n_field: int = 14, clock_every: int = 10, legend_every: int = 3
):
    """Per unique render frame from a ``video.FrameCache``: [frame, clock|NaN, odometer] and
    [frame, legend slots x8] (left->right = running positions 1..8; 0 = unread). The clock (from the
    cached left-panel crops) is video-locked, read every ``clock_every`` frames; the odometer is
    OCR'd only when its crop changes; the legend every ``legend_every`` frames.
    Two passes (speed, results identical): the re-read rules depend on PIXELS only, so pass 1 picks
    the crops to read, pass 2 OCRs them in large batches (all crops of one kind share one size, so
    batch padding is identical)."""
    x0, _x1, y0, _y1 = cache.box
    oy0, oy1, ox0, ox1 = AVV_ODO
    clock_crops, odo_crops, leg_crops = [], [], []
    plan = []  # per frame: (i, clock idx or -1, odo idx (latest), legend idx (latest))
    prev_o, prev_l, io, il = None, None, -1, -1
    for n, (i, p) in enumerate(cache):
        ic = -1
        if n % clock_every == 0:
            clock_crops.append(np.asarray(cache.clock[n]))
            ic = len(clock_crops) - 1
        oc = p[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0]
        if prev_o is None or float(np.abs(oc.astype(np.int16) - prev_o).mean()) > 2.0:
            odo_crops.append(oc.copy())
            io = len(odo_crops) - 1
            prev_o = oc
        if n % legend_every == 0:
            # the number boxes are opaque: re-OCR only when they change (overtakes), ~10x fewer calls
            lb = _legend_boxes(p)
            if (
                prev_l is None
                or lb.shape != prev_l.shape
                or float(np.abs(lb.astype(np.int16) - prev_l).mean()) > 8.0
            ):
                leg_crops.append(_legend_slot_crops(p))
                il = len(leg_crops) - 1
                prev_l = lb
        plan.append((i, ic, io, il))
    clocks = [_parse_clock(t, c) for t, c in rec_batch(clock_crops)]
    odos = [_parse_odometer(t, c) for t, c in rec_batch(odo_crops)]
    flat = rec_batch([c for slots in leg_crops for c in slots])
    legs = [_parse_legend(flat[8 * k : 8 * k + 8], n_field) for k in range(len(leg_crops))]
    rows, leg = [], []
    for i, ic, io, il in plan:
        c = clocks[ic] if ic >= 0 else np.nan
        o = odos[io] if io >= 0 else np.nan
        rows.append([i, np.nan if c is None else c, np.nan if o is None else o])
        leg.append([i, *(legs[il] if il >= 0 else np.zeros(8, int))])
    return np.array(rows), np.array(leg)

