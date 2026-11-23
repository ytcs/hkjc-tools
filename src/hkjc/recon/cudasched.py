"""Opt-in blocking synchronisation for CUDA (``HKJC_CUDA_BLOCKING_SYNC=1``).

By default a CUDA context spin-waits for the GPU while it has fewer contexts than cores, so every process waiting
on the GPU (avatar detector, matcher helper, OCR recogniser, film and ruler YOLO) can hold a CPU core busy. Blocking
sync makes the waiting thread sleep instead. It sets the device's primary context flags through the driver API
before the process creates the context; the runtimes (torch, ONNX Runtime) share that primary context. The
scheduling mode does not change any computed value."""

from __future__ import annotations

import os

CU_CTX_SCHED_BLOCKING_SYNC = 0x04
_DONE = False


def blocking_sync(dev: str) -> None:
    """Called in a process right before its first CUDA use on ``dev`` ('cuda' or 'cuda:N'); no-op unless the switch is
    on and ``dev`` is CUDA. Raises if the driver refuses (the context already exists, no driver)."""
    global _DONE
    if _DONE or not dev.startswith("cuda") or os.environ.get("HKJC_CUDA_BLOCKING_SYNC", "0") != "1":
        return
    import ctypes

    cu = ctypes.CDLL("libcuda.so.1")
    set_flags = getattr(cu, "cuDevicePrimaryCtxSetFlags_v2", None) or cu.cuDevicePrimaryCtxSetFlags
    ordinal = int(dev.split(":", 1)[1]) if ":" in dev else 0
    d = ctypes.c_int()
    for name, call in (
        ("cuInit", lambda: cu.cuInit(0)),
        ("cuDeviceGet", lambda: cu.cuDeviceGet(ctypes.byref(d), ordinal)),
        ("cuDevicePrimaryCtxSetFlags", lambda: set_flags(d, CU_CTX_SCHED_BLOCKING_SYNC)),
    ):
        r = call()
        if r:
            raise RuntimeError(f"HKJC_CUDA_BLOCKING_SYNC: {name} returned CUresult {r}")
    _DONE = True
