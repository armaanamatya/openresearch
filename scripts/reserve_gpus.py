#!/usr/bin/env python
"""reserve_gpus.py — hold currently-free GPUs on a shared host (best-effort).

Examples::

    # hold 4 currently-free cards for 12h (default)
    python scripts/reserve_gpus.py reserve --count 4
    # hold specific cards (by index or UUID)
    python scripts/reserve_gpus.py reserve --gpus 4,5,6,7
    # hold everything that's free right now
    python scripts/reserve_gpus.py reserve --all-free --ttl-hours 6
    # see what's held + what's free
    python scripts/reserve_gpus.py status
    # let them go
    python scripts/reserve_gpus.py release --all
    python scripts/reserve_gpus.py release --gpus 4,5

Only reserves cards that are FREE right now — it never evicts another user's job.
Holds are TTL-bounded and auto-release; our own reproductions can still lease a
reserved card (the allocator recognizes the holder PIDs). Etiquette: holding
idle cards on a shared box is antisocial — keep the TTL tight.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.services.runtime.gpu_reservation import GpuReservationManager  # noqa: E402
from backend.services.runtime.local_gpu_allocator import (  # noqa: E402
    discover_gpus,
    free_devices,
)

_REGISTRY = _REPO_ROOT / "runs" / ".gpu_reservations.json"


def _manager() -> GpuReservationManager:
    return GpuReservationManager(_REGISTRY, repo_root=_REPO_ROOT)


def _resolve_to_uuids(tokens: list[str]) -> list[str]:
    """Map a mix of GPU indices (``4``) and UUIDs (``GPU-...``) to UUIDs."""
    devices = discover_gpus()
    by_index = {d.index: d.uuid for d in devices}
    known_uuids = {d.uuid for d in devices}
    out: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in known_uuids:
            out.append(tok)
        elif tok.isdigit() and int(tok) in by_index:
            out.append(by_index[int(tok)])
        else:
            print(f"reserve_gpus: unknown GPU '{tok}' — skipping", file=sys.stderr)
    return out


def _ttl_remaining(expires_at: str) -> str:
    if not expires_at:
        return "none"
    try:
        secs = (datetime.fromisoformat(expires_at) - datetime.now(timezone.utc)).total_seconds()
    except ValueError:
        return "?"
    if secs <= 0:
        return "expired"
    h, m = divmod(int(secs) // 60, 60)
    return f"{h}h{m:02d}m"


def cmd_reserve(args: argparse.Namespace) -> int:
    mgr = _manager()
    kwargs: dict = {"hold_mib": args.hold_mib, "ttl_seconds": int(args.ttl_hours * 3600)}
    selectors = [args.count is not None, bool(args.gpus), args.all_free]
    if sum(selectors) != 1:
        print("reserve: pass exactly one of --count, --gpus, or --all-free", file=sys.stderr)
        return 2
    if args.count is not None:
        kwargs["count"] = args.count
    elif args.gpus:
        kwargs["uuids"] = _resolve_to_uuids(args.gpus.split(","))
    else:
        kwargs["all_free"] = True

    created = mgr.reserve(**kwargs)
    if not created:
        print("reserve: no cards reserved (none free, all already held, or holder failed).")
        return 1
    for r in created:
        print(f"reserved GPU {r.index} ({r.uuid[:24]}…) pid={r.pid} ttl={_ttl_remaining(r.expires_at)}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    mgr = _manager()
    if args.all == bool(args.gpus):
        print("release: pass exactly one of --all or --gpus", file=sys.stderr)
        return 2
    if args.all:
        released = mgr.release(all=True)
    else:
        released = mgr.release(uuids=_resolve_to_uuids(args.gpus.split(",")))
    print(f"released {len(released)} hold(s).")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    mgr = _manager()
    res = mgr.list_reservations()
    held_uuids = {r.uuid for r in res}
    print(f"== reservations ({len(res)}) ==")
    for r in sorted(res, key=lambda x: x.index):
        print(f"  GPU {r.index:>2}  {r.uuid[:24]}…  pid={r.pid}  ttl={_ttl_remaining(r.expires_at)}")
    devices = discover_gpus()
    free = free_devices(devices, own_pids=frozenset(r.pid for r in res))
    free_unheld = [d for d in free if d.uuid not in held_uuids]
    print(f"== free & unheld ({len(free_unheld)}) ==")
    print("  " + (", ".join(f"GPU {d.index}" for d in free_unheld) or "(none)"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("reserve", help="Reserve currently-free GPUs.")
    pr.add_argument("--count", type=int, default=None, help="Reserve this many free cards.")
    pr.add_argument("--gpus", default="", help="CSV of GPU indices or UUIDs to reserve.")
    pr.add_argument("--all-free", action="store_true", help="Reserve every currently-free card.")
    pr.add_argument("--ttl-hours", type=float, default=12.0, help="Auto-release after N hours (default 12).")
    pr.add_argument("--hold-mib", type=int, default=256, help="MiB parked per card (default 256).")
    pr.set_defaults(func=cmd_reserve)

    prel = sub.add_parser("release", help="Release held GPUs.")
    prel.add_argument("--all", action="store_true", help="Release everything we hold.")
    prel.add_argument("--gpus", default="", help="CSV of GPU indices or UUIDs to release.")
    prel.set_defaults(func=cmd_release)

    pst = sub.add_parser("status", help="Show reservations + free cards.")
    pst.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
