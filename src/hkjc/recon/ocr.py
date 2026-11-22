"""Burnt-in text on the replays, read with RECOGNITION-ONLY OCR (RapidOCR, no text detection) on
fixed-position crops: faster than det+rec and more accurate on these boxes (det splits "850M"
into fragments).

AVV render (1080p): left-panel race clock, the odometer box ("850M" = leader's distance-to-go,
a round-up display: it ticks V+10 -> V as the leader REACHES V), the legend (top-8 running order,
8 fixed-pitch number boxes).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np

from . import video

AVV_ODO = (286, 320, 1606, 1700)  # y0, y1, x0, x1 (absolute, 1080p)
AVV_CLOCK = (288, 342, 800, 945)  # y0, y1, x0, x1 (absolute, 1080p): the left-panel race clock
CLOCK_EVERY = 10  # the clock is read on every 10th unique render frame (video-locked)
CLOCK_SHIFTS = range(-16, 97, 4)  # px: vertical offsets searched when the clock is not at AVV_CLOCK
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


def _legend_slot_x(W: int, n_slots: int = 8) -> list:
    """Column centres of the legend's number boxes. A field of 8+ shows 8 slots at pitch W/8.8; a smaller
    field of N spreads its N slots about the same centre (digits centred at 0.466 W) at a wider pitch,
    (211 - 13 N) / 945 W: the digit centres of a 7-runner (ST 2024-04-14 R8) and a 6-runner legend (ST
    2025-04-13 R1) on the 945 px panel sit 120 and 133 px apart (the line gives 107 px at N = 8, measured 108)."""
    if n_slots >= 8:
        return [int(0.064 * W + k * W / 8.8) for k in range(8)]
    pitch = (211 - 13 * n_slots) / 945 * W
    return [int(0.4635 * W + (k - (n_slots - 1) / 2) * pitch) for k in range(n_slots)]


def _legend_slot_crops(panel, n_slots: int = 8) -> list:
    lt = video.legend_top(panel)
    return [
        panel[max(0, lt - 4) : lt + 34, max(0, cx - 22) : cx + 16].copy()
        for cx in _legend_slot_x(panel.shape[1], n_slots)
    ]


def _parse_legend(res, n_field: int = 14) -> np.ndarray:
    row = np.zeros(8, int)
    for k, (txt, conf) in enumerate(res):
        # NFKC folds circled and fullwidth digits to ASCII ("④" -> "4", "⑪" -> "11"); then ASCII digits only
        dg = "".join(ch for ch in unicodedata.normalize("NFKC", txt) if "0" <= ch <= "9")
        if dg and conf > 0.6 and 1 <= int(dg) <= n_field:
            row[k] = int(dg)
    return row


def read_render_hud_cached(cache, *, n_field: int = 14, n_slots: int = 8, clock_every: int = CLOCK_EVERY):
    """Per unique render frame from a ``video.FrameCache``: [frame, clock|NaN, odometer|NaN] and
    [frame, legend slots x8] (left->right = running positions 1..8; 0 = unread; ``n_slots``: slots shown,
    the field size when under 8). The clock (from the cached left-panel crops) is video-locked, read every
    ``clock_every`` frames; the odometer and the legend are read on every frame. No pixel-change test
    skips a read: the legend's number boxes are not opaque (the track shows through and around them), so a
    slot changed by an overtake can differ less from its last read than an unchanged slot differs between
    frames, and a skipped read holds a stale number.
    The crops are OCR'd in large batches (all crops of one kind share one size, so batch padding is
    identical and each read equals the crop's read on its own)."""
    x0, _x1, y0, _y1 = cache.box
    oy0, oy1, ox0, ox1 = AVV_ODO
    frames, clock_at, clock_crops, odo_crops, leg_crops = [], [], [], [], []
    for n, (i, p) in enumerate(cache):
        frames.append(i)
        if n % clock_every == 0:
            clock_at.append(n)
            clock_crops.append(np.asarray(cache.clock[n]))
        odo_crops.append(p[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0].copy())
        leg_crops.append(_legend_slot_crops(p, n_slots))
    clock = np.full(len(frames), np.nan)
    for n, (t, c) in zip(clock_at, rec_batch(clock_crops)):
        x = _parse_clock(t, c)
        clock[n] = np.nan if x is None else x
    odo = [_parse_odometer(t, c) for t, c in rec_batch(odo_crops)]
    flat = rec_batch([c for slots in leg_crops for c in slots])
    m = min(8, n_slots)  # crops per legend read; unshown slots stay 0
    rows = [[i, clock[n], np.nan if odo[n] is None else odo[n]] for n, i in enumerate(frames)]
    leg = [[i, *_parse_legend(flat[m * n : m * n + m], n_field)] for n, i in enumerate(frames)]
    return np.array(rows), np.array(leg)


def read_clock_located(path, frames, *, box=AVV_CLOCK, shifts=CLOCK_SHIFTS):
    """Race-clock reads at video ``frames`` with the clock crop moved to where the clock is. A sponsor box
    (LONGINES) above the clock pushes it ~45 px below the fixed crop on some broadcasts. Each vertical
    shift of ``box`` is scored by the number of reads that agree on the clock's zero frame (frame - 25 clock
    within 2 frames of their median); the centre of the widest run of best-scoring shifts is kept (the
    clock whole inside the crop). -> (clock seconds or NaN per frame, shift px; NaN everywhere when no
    shift gives 2 agreeing reads)."""
    import av

    y0, y1, x0, x1 = box
    lo, hi = min(shifts), max(shifts)
    want = {int(f) for f in frames}
    strips = {}
    with av.open(str(path)) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                strips[i] = fr.to_ndarray(format="bgr24")[max(0, y0 + lo) : y1 + hi, x0:x1].copy()
            if i >= max(want):
                break
    off = max(0, y0 + lo)
    blank = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)  # a frame past the video's end reads nothing
    dys = sorted(shifts)
    reads, score = [], []
    for dy in dys:
        crops = [strips[int(f)][y0 + dy - off : y1 + dy - off] if int(f) in strips else blank for f in frames]
        v = np.array([np.nan if (x := _parse_clock(t, c)) is None else x for t, c in rec_batch(crops)])
        ok = np.isfinite(v)
        z = np.asarray(frames, float)[ok] - 25.0 * v[ok]
        reads.append(v)
        score.append(int((np.abs(z - np.median(z)) <= 2.0).sum()) if ok.sum() >= 2 else 0)
    score = np.array(score)
    top = np.flatnonzero(score == score.max())
    run = max(np.split(top, np.flatnonzero(np.diff(top) > 1) + 1), key=len)
    k = int(run[len(run) // 2])
    if score[k] < 2:
        return np.full(len(frames), np.nan), 0
    return reads[k], dys[k]
