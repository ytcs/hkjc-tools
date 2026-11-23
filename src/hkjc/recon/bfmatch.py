"""Brute-force Hamming matching of ORB descriptors, index-identical to ``cv2.BFMatcher(NORM_HAMMING)``.

Opt-in (``HKJC_MATCH_DEVICE``: ``cuda`` or ``cpu``): the neighbour tables of every frame pair a stage will match
are computed up front in a helper interpreter (CUDA never initialises in the parent, whose forked pools must stay
CUDA-free): a fresh one per call, or with ``HKJC_MATCH_HELPER=1`` one kept for the race and handed to ``register.match``. Unset: OpenCV inside ``register.match``.

Exactness: distances are popcounts <= 256, computed as |a| + |b| - 2 a.b on 0/1 bit vectors; every partial
sum is an integer <= 256, exact in fp16 and fp32. OpenCV's tie order (equal distance -> lower train index
first; cross-check keeps the lowest-index best in both directions) is reproduced with a composite key."""

from __future__ import annotations

import os

import numpy as np

K_NN = 4  # register.match's knnMatch k


def device() -> str | None:
    return os.environ.get("HKJC_MATCH_DEVICE") or None


def helper_on() -> bool:
    """``HKJC_MATCH_HELPER=1``: one helper interpreter per race (started by ``warm``, closed by ``close``) instead of a
    fresh one per ``tables`` call; the features stage also feeds its feature workers from a detector process. Off by
    default (opt-in until checked on a pod)."""
    return os.environ.get("HKJC_MATCH_HELPER", "0") == "1"


def _tables(dev: str, des: list, jobs: list) -> dict:
    """{(i, j, cross): table}; cross -> (ia, ib) int64 arrays, else ti (N, min(K_NN, M)) int64."""
    from .cudasched import blocking_sync

    blocking_sync(dev)  # (the helper interpreter, before its first CUDA call)
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
    return _tables_subprocess(dev, des, jobs) if helper_on() else _tables_oneshot(dev, des, jobs)


def _tables_oneshot(dev: str, des: list, jobs: list) -> dict:
    """``_tables`` in a fresh interpreter (``python -c``, arguments and result pickled over pipes): CUDA
    initialises there, never in the caller, whose forked pools must stay CUDA-free. Unlike a
    multiprocessing spawn it never re-imports the caller's main module."""
    import pickle
    import subprocess
    import sys

    code = (
        "import pickle, sys; from hkjc.recon.bfmatch import _tables; "
        "sys.stdout.buffer.write(pickle.dumps(_tables(*pickle.load(sys.stdin.buffer))))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], input=pickle.dumps((dev, des, jobs)), capture_output=True
    )
    if r.returncode:
        raise RuntimeError(f"bfmatch helper failed: {r.stderr.decode(errors='replace')[-2000:]}")
    return pickle.loads(r.stdout)


_HELPER = None  # the persistent helper interpreter: (Popen, device, owner pid)

_HELPER_CODE = """
import os, pickle, sys
dev = sys.argv[1]
out = os.fdopen(os.dup(1), "wb")  # the reply channel; anything printed goes to stderr
os.dup2(2, 1)
import torch
if dev.startswith("cuda"):
    torch.zeros(1, device=dev)  # initialise CUDA before the first request
from hkjc.recon.bfmatch import _tables
inp = sys.stdin.buffer
while True:
    try:
        req = pickle.load(inp)
    except EOFError:
        break
    try:
        res = (True, _tables(dev, *req))
    except Exception as e:
        res = (False, f"{type(e).__name__}: {e}")
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()  # other races share the GPU between requests
    pickle.dump(res, out)
    out.flush()
"""


def warm():
    """Start the helper interpreter now (torch import and CUDA initialisation overlap the caller's work) when
    the tables will be computed on CUDA; it serves every later ``tables`` call of this process."""
    dev = device()
    if dev and dev.startswith("cuda") and helper_on():
        _helper(dev)


def _helper(dev: str):
    global _HELPER
    import os
    import subprocess
    import sys

    if _HELPER is not None and (_HELPER[1] != dev or _HELPER[0].poll() is not None or _HELPER[2] != os.getpid()):
        if _HELPER[2] == os.getpid():
            _close()
        _HELPER = None
    if _HELPER is None:
        pr = subprocess.Popen([sys.executable, "-c", _HELPER_CODE, dev], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        _HELPER = (pr, dev, os.getpid())
        import atexit

        atexit.register(_close)
    return _HELPER[0]


def close():
    """Stop this process's helper interpreter, if any (a later ``tables`` call starts a new one)."""
    _close()


def _close():
    global _HELPER
    import os

    if _HELPER is not None and _HELPER[2] == os.getpid():
        pr = _HELPER[0]
        _HELPER = None
        try:
            pr.stdin.close()
            pr.wait(timeout=30)
        except Exception:
            pr.kill()


def _tables_subprocess(dev: str, des: list, jobs: list) -> dict:
    """``_tables`` in the helper interpreter (``python -c``, started once per process and kept; arguments and
    result pickled over pipes): CUDA initialises there, never in the caller, whose forked pools must stay
    CUDA-free. Unlike a multiprocessing spawn it never re-imports the caller's main module."""
    import pickle

    pr = _helper(dev)
    try:
        pickle.dump((des, jobs), pr.stdin)
        pr.stdin.flush()
        ok, res = pickle.load(pr.stdout)
    except (EOFError, BrokenPipeError, OSError) as e:
        _close()
        raise RuntimeError(f"bfmatch helper died: {type(e).__name__}: {e}") from e
    if not ok:
        raise RuntimeError(f"bfmatch helper failed: {res}")
    return res
