"""A minimal, dependency-free GPU "holder" process.

Parks a tiny CUDA context + allocation on one GPU so the card shows up as busy in
``nvidia-smi`` (and to other users' "find a free GPU" tooling), then idles until
its TTL elapses or it receives SIGTERM/SIGINT. It deliberately uses the CUDA
*driver* API via ``ctypes`` (``libcuda``, always present on a GPU host) rather
than torch/cupy, so it runs from any Python — the repo's main venv has no torch.

The parent (``gpu_reservation``) pins the target card by exporting
``CUDA_VISIBLE_DEVICES=GPU-<uuid>`` before spawning this, so device ordinal 0 is
always the intended GPU.

Run::

    python -m backend.services.runtime.gpu_holder --uuid GPU-xxxx --mib 256 --ttl-seconds 43200

Exit codes: 0 = clean stop (TTL/SIGTERM); 2 = could not acquire the GPU (no
libcuda, driver error) — the manager treats that as a failed reservation.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys
import time

_CUDA_SUCCESS = 0


def _load_libcuda() -> "ctypes.CDLL | None":
    for name in ("libcuda.so.1", "libcuda.so", "libcuda.dylib", "nvcuda.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _acquire(cuda: "ctypes.CDLL", mib: int) -> "tuple[ctypes.c_void_p, ctypes.c_void_p] | None":
    """cuInit → cuDeviceGet(0) → cuCtxCreate → cuMemAlloc(mib). Returns (ctx, ptr) or None."""
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

    if cuda.cuInit(0) != _CUDA_SUCCESS:
        return None
    dev = ctypes.c_int(0)
    if cuda.cuDeviceGet(ctypes.byref(dev), 0) != _CUDA_SUCCESS:
        return None
    ctx = ctypes.c_void_p()
    if cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != _CUDA_SUCCESS:
        return None
    ptr = ctypes.c_void_p()
    nbytes = max(1, int(mib)) * 1024 * 1024
    # A context alone already marks the card busy; the allocation makes the hold
    # visibly non-trivial. A failed alloc (e.g. card now full) is non-fatal — the
    # context still holds the card.
    cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    return ctx, ptr


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Park a tiny CUDA hold on one GPU until TTL/SIGTERM.")
    parser.add_argument("--uuid", default=os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                        help="GPU UUID (for logging only; pinning is via CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--mib", type=int, default=256, help="MiB to allocate (visibility; default 256).")
    parser.add_argument("--ttl-seconds", type=int, default=0,
                        help="Auto-release after N seconds (0 = no expiry; rely on SIGTERM).")
    args = parser.parse_args(argv)

    cuda = _load_libcuda()
    if cuda is None:
        print("gpu_holder: libcuda not found — cannot reserve", file=sys.stderr, flush=True)
        return 2
    held = _acquire(cuda, args.mib)
    if held is None:
        print(f"gpu_holder[{args.uuid}]: CUDA init/context failed — cannot reserve",
              file=sys.stderr, flush=True)
        return 2

    stop = {"flag": False}

    def _handle(_signum, _frame):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    deadline = (time.time() + args.ttl_seconds) if args.ttl_seconds > 0 else None
    print(f"gpu_holder[{args.uuid}]: holding {args.mib}MiB pid={os.getpid()} "
          f"ttl={args.ttl_seconds or 'none'}", flush=True)
    while not stop["flag"]:
        if deadline is not None and time.time() >= deadline:
            print(f"gpu_holder[{args.uuid}]: TTL elapsed — releasing", flush=True)
            break
        time.sleep(2.0)
    # held context/allocation drop on process exit → card freed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
