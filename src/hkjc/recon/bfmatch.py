"""Brute-force Hamming matching of ORB descriptors, index-identical to ``cv2.BFMatcher(NORM_HAMMING)``.

Opt-in (``HKJC_MATCH_DEVICE``: ``cuda`` or ``cpu``): the neighbour tables of every frame pair a stage will match
are computed up front in ONE spawned helper process (CUDA never initialises in the parent, whose forked
pools must stay CUDA-free) and handed to ``register.match``. Unset: OpenCV inside ``register.match``.

Exactness: distances are popcounts <= 256, computed as |a| + |b| - 2 a.b on 0/1 bit vectors; every partial
sum is an integer <= 256, exact in fp16 and fp32. OpenCV's tie order (equal distance -> lower train index
first; cross-check keeps the lowest-index best in both directions) is reproduced with a composite key."""

from __future__ import annotations

import os

import numpy as np

K_NN = 4  # register.match's knnMatch k


def device() -> str | None:
    return os.environ.get("HKJC_MATCH_DEVICE") or None


def _tables(dev: str, des: list, jobs: list) -> dict:
    """{(i, j, cross): table}; cross -> (ia, ib) int64 arrays, else ti (N, min(K_NN, M)) int64."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    dt = torch.float16 if dev.startswith("cuda") else torch.float32
    bits = {}

    def bits_of(k):
        if k not in bits:
            if len(bits) > 64:
                bits.clear()
            bits[k] = torch.from_numpy(np.unpackbits(des[k], axis=1)).to(dev).to(dt)
        return bits[k]

    out = {}
    for i, j, cross in jobs:
        a, b = bits_of(i), bits_of(j)
        N, M = a.shape[0], b.shape[0]
        if N == 0 or M == 0:
            out[(i, j, cross)] = (np.zeros(0, np.int64),) * 2 if cross else np.zeros((N, 0), np.int64)
            continue
        D = a.float().sum(1, keepdim=True) + b.float().sum(1)[None, :] - 2.0 * (a @ b.T).float()
        Di = torch.round(D)
        if not torch.equal(Di, D):
            raise RuntimeError("bfmatch: non-integer Hamming distance")
        D = Di.to(torch.int64)
        if cross:
            ab = (D * M + torch.arange(M, device=dev)[None, :]).argmin(1)
            ba = (D * N + torch.arange(N, device=dev)[:, None]).argmin(0)
            q = torch.arange(N, device=dev)
            ok = ba[ab] == q
            out[(i, j, cross)] = (q[ok].cpu().numpy(), ab[ok].cpu().numpy())
        else:
            key = D * M + torch.arange(M, device=dev)[None, :]
            _, ti = torch.topk(key, min(K_NN, M), dim=1, largest=False, sorted=True)
            out[(i, j, cross)] = ti.cpu().numpy()
    return out


def tables(des: list, jobs: list) -> dict:
    """Neighbour tables for ``jobs`` [(i, j, cross)] over descriptor arrays ``des``; {} when not opted in."""
    dev = device()
    if not dev or not jobs:
        return {}
    jobs = list(dict.fromkeys(jobs))
    if not dev.startswith("cuda"):
        return _tables(dev, des, jobs)
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_tables, (dev, des, jobs))
