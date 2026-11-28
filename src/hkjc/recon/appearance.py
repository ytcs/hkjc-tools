"""Render appearance of the silk render: the learned avatar model's per-detection runner likelihoods (``produce``,
the ``appearance`` stage) and their term in the core's objective (``costs``), discounted for the correlation of
repeated detections of one avatar, as a (segments x runners) cost in J.

Producer (``produce`` -> ``appearance.npz``, rows = dets.npz rows). Weights under ``MODEL_DIR``: the crop and silk
towers (``head.pt``), their input standardisation (``norm.npz``), the render temperature (``temperature.json``) and a
frozen ImageNet ResNet-18 (``resnet18.pth``). Per detection:
- crop: a square of side 1.5 x max(w, h) centred on the box, from the render panel zero-padded 64 px, resized to
  48 x 48 (INTER_AREA, BGR); frames are decoded with ``video.decode_from`` and only frames with a detection become
  pixels.
- features: a soft palette histogram (16 fixed colours in Lab, sigma 15, over a 3 x 2 grid of the crop,
  sqrt-normalised; 96) and the ResNet-18's layer3 average-pooled to 2 x 2 plus layer4's global average (1536, float16)
  on the crop upsampled to 96 px (INTER_CUBIC). The ResNet runs through OpenVINO on the CPU at f32 in fixed batches of
  ``BATCH``.
- the crop tower maps the standardised features to a 64-dim unit embedding; the silk tower maps each runner's
  official silk (composited on grey 128: the ResNet features of its 96 px resize, plus palette histograms of cap, body,
  sleeves and the whole silk) to another; logit = scale x cosine, ``logp`` = log-softmax of logit / T over the runners
  with a silk.
- alignment guard: ``detect.silk_feature`` on the decoded panel must reproduce the stored silk.npz feature
  (``silk_mismatch`` counts the detections where it does not).
Stored: ``logp`` (n, R) float32 (R = the race's runners in order; NaN column for a runner without a silk, NaN row for a
detection whose frame was not decoded), ``logit`` (n, R), ``emb`` (n, 64) float16, ``ok`` (n,), ``nos``, ``hid``,
``has_gif`` (R,), ``T_render``, ``dets_md5`` and ``n_dets`` (the dets.npz it was computed from; ``load`` checks them),
``silk_mismatch``, ``model_md5``.

Cost term: ``appearance.npz``'s ``logp``, the calibrated per-detection log p(runner | crop) over the race's runners
``nos``, with ``ok`` and ``has_gif``.

Per detection d and runner h, with P_d the box's P(horse) (``core.p_horse``): the softmax has no "not a horse"
outcome, so a box that is not a horse reads as uniform:

  log r_dh = log(P_d G p_dh + (1 - P_d))     (G runners with an official silk; r = 1 for a runner without one and
                                              for rows the model did not score)

The unit of correlated evidence is the avatar's tracker tracklet tau (observe's ``members``; a detection in no
tracklet is its own): the same avatar renders the same way frame after frame, so each detection carries weight
w = 1 / (1 + (n_tau - 1) RHO), n_tau the tracklet's P(horse)-weighted count of scored rows, and the tracklet's pooled
evidence saturates at about 1 / RHO detections. The tracklet's read is

  L_tau,h = sum_{d in tau} w log r_dh,   q_tau = (1 - EPS) softmax(L_tau) + EPS / H

EPS is the share of avatars that render like another runner's silk (systematic, not averaged away by pooling).
Appearance says WHICH runner an avatar is, not WHETHER it is a horse: a box that is not a horse (a list icon, a
glyph) can look like any runner's silk, so the tracklet's cost is taken against its best-matching runner,

  CT_tau,h = -2 log(q_tau,h / max_h' q_tau,h')  >= 0,

and the labelled-versus-clutter balance stays with P(horse) and the kinematic terms. The core's segments cut
tracklets, so CT is split over the tracklet's segments additively: segment k carries its own rows' evidence -2 L_kh
plus its share a_k (of the tracklet's P(horse)-weighted scored rows) of CT_tau,h + 2 L_tau,h. A tracklet whose
segments share one label costs exactly CT_tau,h however it is cut, and segments of a tracklet that changes avatar
each keep their own read. RHO and EPS were fitted by maximum likelihood of q on development races the model never
trained on, against the stored product's confident tracklet labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

RHO = 0.48
EPS = 0.031


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load(app_path, dets_path) -> dict:
    """``appearance.npz``'s arrays, after checking that its rows are this run's detections: its ``n_dets`` and
    ``dets_md5`` (the dets.npz it was computed from) must match ``dets_path``; raises ValueError otherwise (rows of
    another detection set would read other boxes' likelihoods)."""
    with np.load(app_path) as za:
        app = {k: za[k] for k in ("logp", "ok", "nos", "has_gif")}
        n_app, md5_app = int(za["n_dets"]), str(za["dets_md5"])
    with np.load(dets_path) as zd:
        n = len(zd["rows"])
    md5 = _md5(dets_path)
    if n_app != n or md5_app != md5:
        raise ValueError(f"{app_path} was computed from another dets.npz (n_dets {n_app} md5 {md5_app}; "
                         f"this run's dets.npz: {n} rows, md5 {md5})")
    return app


def log_ratios(app, det_idx, horses, p_horse):
    """Per tracks row: (log r_dh (rows x runners), scored mask)."""
    horses = np.asarray(horses, int)
    H = len(horses)
    det_idx = np.asarray(det_idx, int)
    nos = np.asarray(app["nos"], int)
    gif = np.asarray(app["has_gif"], bool)
    lp = np.asarray(app["logp"], float)[det_idx]
    valid = np.asarray(app["ok"], bool)[det_idx] & np.isfinite(lp[:, gif]).all(1)
    col = np.searchsorted(horses, nos)
    use = gif & (col < H) & (horses[np.minimum(col, H - 1)] == nos)
    G = int(use.sum())
    P = np.asarray(p_horse, float)
    logr = np.zeros((len(det_idx), H))
    if G and valid.any():
        q = np.exp(np.where(valid[:, None], lp[:, use], 0.0))
        q /= q.sum(1, keepdims=True)
        logr[:, col[use]] = np.log(P[:, None] * G * q + (1.0 - P[:, None]))
        logr[~valid] = 0.0
    return logr, valid


def tracklet_ids(members, n_rows) -> np.ndarray:
    """Per tracks row: its tracklet index (rows in no tracklet get their own)."""
    trk = np.full(n_rows, -1)
    for i, m in enumerate(members):
        trk[np.asarray(m, int)] = i
    lone = trk < 0
    trk[lone] = len(members) + np.arange(lone.sum())
    return trk


def costs(app, det_idx, seg, n_seg, horses, p_horse, members) -> np.ndarray:
    """(segments x runners) appearance cost C_kh (above) for the core's segments ``seg`` (per tracks row; each
    segment lies in one tracklet)."""
    H = len(horses)
    seg = np.asarray(seg, int)
    logr, valid = log_ratios(app, det_idx, horses, p_horse)
    trk = tracklet_ids(members, len(seg))
    n_trk = int(trk.max()) + 1 if len(trk) else 0
    pv = np.asarray(p_horse, float) * valid
    cnt = np.bincount(trk, pv, n_trk)
    w = 1.0 / (1.0 + (np.maximum(cnt, 1.0)[trk] - 1.0) * RHO)
    Lt = np.zeros((n_trk, H))
    np.add.at(Lt, trk, w[:, None] * logr)
    q = (1.0 - EPS) * np.exp(Lt - logsumexp(Lt, axis=1, keepdims=True)) + EPS / H
    lift = 2.0 * (np.log(q.max(1, keepdims=True)) - np.log(q)) + 2.0 * Lt
    L = np.zeros((n_seg, H))
    np.add.at(L, seg, w[:, None] * logr)
    seg_trk = np.zeros(n_seg, int)
    seg_trk[seg] = trk
    share = np.bincount(seg, pv, n_seg) / np.maximum(cnt[seg_trk], 1e-12)
    return -2.0 * L + share[:, None] * lift[seg_trk]


# ------------------------------------------------------------------------------------------------------------- producer
MODEL_DIR = Path("models/render_appearance")
CROP, PAD, BATCH, CHUNK = 48, 64, 128, 1024  # crop px, panel padding px, ResNet batch, crops per feature chunk
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet input standardisation
STD = np.array([0.229, 0.224, 0.225], np.float32)
PAL_RGB = np.array([(245, 245, 245), (180, 180, 180), (90, 90, 90), (20, 20, 20), (210, 30, 30), (120, 20, 40),
                    (240, 130, 170), (245, 140, 30), (245, 225, 40), (160, 210, 40), (20, 140, 60), (120, 190, 235),
                    (30, 80, 200), (20, 30, 80), (110, 40, 150), (120, 75, 40)], np.uint8)
PAL_SIGMA = 15.0  # Lab distance scale of the soft palette assignment


def _pal_lab():
    import cv2

    p = cv2.cvtColor(PAL_RGB[None], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    p[:, 0] *= 100 / 255
    p[:, 1:] -= 128
    return p


def _lab(img_bgr):
    import cv2

    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L[..., 0] *= 100 / 255
    L[..., 1:] -= 128
    return L


def _palhist(L, mask=None):
    """Soft palette histogram of Lab pixels (sqrt-normalised, 16)."""
    px = L.reshape(-1, 3) if mask is None else L[mask]
    if not len(px):
        return np.zeros(16, np.float32)
    d2 = ((px[:, None, :] - _pal_lab()[None]) ** 2).sum(-1)
    w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * PAL_SIGMA ** 2))
    w /= w.sum(1, keepdims=True)
    h = w.sum(0)
    return np.sqrt(h / h.sum()).astype(np.float32)


def palette(crops, sub=64):
    """(n, 48, 48, 3) BGR uint8 -> (n, 96): the palette histogram of each crop's 3 x 2 cells of 16 x 24 px."""
    import cv2

    out = []
    P = _pal_lab()
    for s in range(0, len(crops), sub):
        c = crops[s:s + sub]
        n = len(c)
        L = cv2.cvtColor(c.reshape(n * CROP, CROP, 3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(n, CROP, CROP, 3)
        L[..., 0] *= 100 / 255
        L[..., 1:] -= 128
        px = L.reshape(n, 3, 16, 2, 24, 3).transpose(0, 1, 3, 2, 4, 5).reshape(n, 6, 384, 1, 3)
        d2 = (px[..., 0] - P[:, 0]) ** 2 + (px[..., 1] - P[:, 1]) ** 2 + (px[..., 2] - P[:, 2]) ** 2
        w = np.exp(-(d2 - d2.min(-1, keepdims=True)) / (2 * PAL_SIGMA ** 2))
        w /= w.sum(-1, keepdims=True)
        h = w.sum(2)
        out.append(np.sqrt(h / h.sum(-1, keepdims=True)).astype(np.float32).reshape(n, 96))
    return np.concatenate(out) if out else np.zeros((0, 96), np.float32)


def crop48(pad, cx, cy, w, h):
    """The detection's square crop from the padded panel ``pad``, resized to 48 x 48."""
    import cv2

    s = 1.5 * max(w, h)
    a0, b0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    a1, b1 = a0 + int(round(s)), b0 + int(round(s))
    return cv2.resize(pad[b0 + PAD:b1 + PAD, a0 + PAD:a1 + PAD], (CROP, CROP), interpolation=cv2.INTER_AREA)


def _resnet():
    """The frozen ResNet-18 (eval) and its feature graph (layer3 pooled to 2 x 2 + layer4 global average)."""
    import torch
    import torchvision

    m = torchvision.models.resnet18(weights=None)
    m.load_state_dict(torch.load(MODEL_DIR / "resnet18.pth"))
    m.eval()

    class Head(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            m = self.m
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            x = m.layer2(m.layer1(x))
            l3 = m.layer3(x)
            l4 = m.layer4(l3)
            return torch.cat([torch.nn.functional.adaptive_avg_pool2d(l3, 2).flatten(1), l4.mean((2, 3))], 1)

    return Head(m).eval()


def _norm_input(imgs_bgr96):
    x = np.asarray(imgs_bgr96)[..., ::-1].astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(0, 3, 1, 2)


def _cnn_torch(head, imgs_bgr96):
    """list of 96 x 96 BGR uint8 -> (n, 1536) float16 (torch, CPU)."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(imgs_bgr96), BATCH):
            x = torch.from_numpy(_norm_input(np.stack(imgs_bgr96[i:i + BATCH])).copy())
            out.append(head(x).numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 1536), np.float16)


def _cnn_openvino(head, threads):
    """(n, 96, 96, 3) BGR uint8 -> (n, 1536) float16 through the ResNet graph compiled by OpenVINO (CPU, f32, fixed
    batches of BATCH)."""
    import openvino as ov
    import torch

    om = ov.convert_model(head, example_input=torch.zeros(BATCH, 3, 96, 96))
    net = ov.Core().compile_model(om, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": str(threads),
                                              "ENABLE_CPU_PINNING": False})

    def run(imgs):
        out = []
        for i in range(0, len(imgs), BATCH):
            x = _norm_input(imgs[i:i + BATCH])
            nb = len(x)
            if nb < BATCH:
                x = np.concatenate([x, np.zeros((BATCH - nb,) + x.shape[1:], np.float32)])
            out.append(net(np.ascontiguousarray(x))[0][:nb].astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, 1536), np.float16)

    return run


def silk_inputs(head, paths):
    """Silk-tower inputs of official silk GIFs: (ResNet features of the 96 px resize (n, 1536) float16, palette
    histograms of cap, body, sleeves and the whole silk (n, 64))."""
    import cv2
    from PIL import Image

    ims, cols = [], []
    for p in paths:
        a = np.array(Image.open(p).convert("RGBA")).astype(np.float32)
        al = a[..., 3:4] / 255
        rgb = (a[..., :3] * al + 128 * (1 - al)).astype(np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        ims.append(cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_LINEAR))
        L = _lab(bgr)
        m = a[..., 3] > 0
        reg = np.zeros((4,) + m.shape, bool)
        reg[0, 0:10] = True
        reg[1, 10:38, 9:21] = True
        reg[2, 10:38, 0:9] = True
        reg[2, 10:38, 21:30] = True
        reg[3] = True
        cols.append(np.concatenate([_palhist(L, reg[i] & m) for i in range(4)]))
    return _cnn_torch(head, ims), np.array(cols, np.float32).reshape(-1, 64)


def _towers():
    """(crop and silk towers' state, input standardisation, render temperature, weight md5s)."""
    import torch

    sd = torch.load(MODEL_DIR / "head.pt")
    norm = dict(np.load(MODEL_DIR / "norm.npz"))
    T = float(json.loads((MODEL_DIR / "temperature.json").read_text())["render"])
    md5 = {p: _md5(MODEL_DIR / p)[:8] for p in ("head.pt", "norm.npz", "temperature.json", "resnet18.pth")}
    return sd, norm, T, md5


def extract(video_path, rows, box, silk_feat, threads):
    """48 px crops of every detection; decoded mask ``ok``; alignment-guard mismatch count."""
    import av
    import cv2

    from . import detect, video

    x0, x1, y0, y1 = (int(b) for b in box)
    crops = np.zeros((len(rows), CROP, CROP, 3), np.uint8)
    ok = np.zeros(len(rows), bool)
    if not len(rows):
        return crops, ok, 0
    fr = rows[:, 0].astype(int)
    order = np.argsort(fr, kind="stable")
    fs = fr[order]
    want = set(fs.tolist())
    last = max(want)
    bad = 0
    with av.open(str(video_path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = threads
        for i, f in video.decode_from(c, min(want)):
            if i > last:
                break
            if i not in want:
                continue
            panel = f.to_ndarray(format="bgr24")[y0:y1, x0:x1]
            pad = cv2.copyMakeBorder(panel, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)
            a, b = np.searchsorted(fs, [i, i + 1])
            for j in order[a:b]:
                cx, cy, w, h = rows[j, 1:5]
                crops[j] = crop48(pad, cx, cy, w, h)
                ok[j] = True
                if not np.array_equal(detect.silk_feature(panel, rows[j, 1:5]).astype(np.float32), silk_feat[j]):
                    bad += 1
    return crops, ok, bad


def log_softmax_scores(logit, has, ok, T):
    """(logp, logit) over the race's runners from the towers' scaled cosines ``logit`` (n, runners with a silk): NaN
    columns for runners without a silk (``has`` False), NaN rows where ``ok`` is False."""
    n, R = len(ok), len(has)
    full = np.full((n, R), np.nan, np.float32)
    full[:, has] = logit
    z = logit / T
    z = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    logp = np.full((n, R), np.nan, np.float32)
    logp[:, has] = z
    logp[~ok] = np.nan
    full[~ok] = np.nan
    return logp, full


def produce(out, dets_path, silk_path, video_path, runners, threads=1) -> str:
    """Write ``appearance.npz`` to ``out`` (path) for the detections in ``dets_path`` (their ``rows`` and panel
    ``box``) of the aerial video ``video_path``; ``silk_path`` the run's silk.npz (alignment guard); ``runners``
    [(saddlecloth no, horse id, official silk GIF path or None)] in race order; ``threads`` the decoder's and
    OpenVINO's threads. -> a log line."""
    import torch

    prev = torch.get_num_threads()
    torch.set_num_threads(1)  # the towers' and the silk ResNet's float results do not depend on the process's threads
    try:
        return _produce(out, dets_path, silk_path, video_path, runners, threads)
    finally:
        torch.set_num_threads(prev)


def _produce(out, dets_path, silk_path, video_path, runners, threads):
    import cv2
    import torch

    D = np.load(dets_path)
    rows, box = D["rows"], D["box"]
    silk_feat = np.load(silk_path)["feat"]
    assert len(silk_feat) == len(rows), "silk.npz rows are not dets.npz rows"
    sd, norm, T, md5 = _towers()
    head = _resnet()
    crops, ok, bad = extract(video_path, rows, box, silk_feat, threads)
    n = len(rows)
    feat = np.zeros((n, 1536), np.float16)
    pal = np.zeros((n, 96), np.float32)
    cnn = _cnn_openvino(head, threads)
    idx = np.flatnonzero(ok)
    for s in range(0, len(idx), CHUNK):
        ii = idx[s:s + CHUNK]
        pal[ii] = palette(crops[ii])
        feat[ii] = cnn(np.stack([cv2.resize(crops[i], (96, 96), interpolation=cv2.INTER_CUBIC) for i in ii]))
    nos = np.array([r[0] for r in runners], int)
    hid = np.array([r[1] for r in runners])
    has = np.array([r[2] is not None for r in runners], bool)
    g_cnn, g_col = silk_inputs(head, [r[2] for r in runners if r[2] is not None])
    crop_w, crop_b = sd["crop.0.weight"], sd["crop.0.bias"]
    gif_w, gif_b = sd["gif.weight"], sd["gif.bias"]
    with torch.no_grad():
        X = np.concatenate([feat.astype(np.float32), pal], 1)
        X = (X - norm["MU0"]) / norm["SD0"]
        e = torch.nn.functional.normalize(torch.nn.functional.linear(torch.from_numpy(X.astype(np.float32)), crop_w,
                                                                     crop_b), dim=-1)
        GX = np.concatenate([g_cnn.astype(np.float32), g_col], 1)
        GX = (GX - norm["GMU"]) / norm["GSD"]
        gg = torch.nn.functional.normalize(torch.nn.functional.linear(torch.from_numpy(GX.astype(np.float32)), gif_w,
                                                                      gif_b), dim=-1)
        lg = (sd["scale"] * e @ gg.T).numpy()
        emb = e.numpy().astype(np.float16)
    logp, logit = log_softmax_scores(lg, has, ok, T)
    emb[~ok] = np.nan
    with open(out, "wb") as fh:
        np.savez_compressed(fh, logp=logp, logit=logit, emb=emb, ok=ok, nos=nos, hid=hid, has_gif=has,
                            T_render=np.float64(T), dets_md5=_md5(dets_path), n_dets=n, silk_mismatch=bad,
                            model_md5=json.dumps(md5))
    return f"   appearance: {n} detections, {int(ok.sum())} scored, {int(has.sum())}/{len(has)} official silks, " \
           f"{bad} silk-feature mismatches"
