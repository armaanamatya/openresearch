"""verify_run.py — sanity-check a launch dir produced by dev.ps1 / dev.sh.

Walks a logs/<TS>/ folder and validates the contract documented in
docs/design/unified-logging-launcher.md and tier2-observability-plan.md.
Reports a per-section pass/fail checklist. Exits non-zero if anything is
missing or malformed — suitable for CI smokes.

Usage::

    python scripts/verify_run.py logs/20260517-183936
    python scripts/verify_run.py --latest      # newest under ./logs/

Checks performed:
  - meta.json: present, valid JSON, ended_at set, runs_root absolute.
  - manifest.json: present, valid JSON, no zero-byte text files.
  - server/: backend.log and frontend.log present and non-empty.
  - pipeline.log / pipeline.jsonl (Tier 2a, when an agent ran).
  - prj_<id>/: dashboard_events.jsonl present; agents/<NN>-<id>/ each
    contains the full Tier 2b set (prompt.md, trace.log, tool_calls.jsonl,
    result.txt, meta.json), with meta.status ∈ {ok, error, interrupted}.

Stdlib only — no third-party deps. Safe to invoke from any Python 3.11+.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable


REQUIRED_AGENT_FILES = (
    "prompt.md",
    "trace.log",
    "tool_calls.jsonl",
    "result.txt",
    "meta.json",
)


class Check:
    """Bag of pass/fail messages for one section of the report."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed: list[str] = []
        self.failed: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)


def check_meta(root: Path, c: Check) -> None:
    p = root / "meta.json"
    if not p.exists():
        c.fail("meta.json missing")
        return
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        c.fail(f"meta.json invalid JSON: {exc}")
        return
    if not meta.get("ended_at"):
        c.fail(
            "meta.json ended_at is empty — the launcher did not clean up. "
            "Was the process kill -9'd?"
        )
    else:
        c.ok(
            f"ended_reason={meta.get('ended_reason', '?')} "
            f"backend_exit={meta.get('backend_exit')} "
            f"frontend_exit={meta.get('frontend_exit')}"
        )
    runs_root = meta.get("runs_root", "")
    if not runs_root:
        c.fail("meta.json runs_root is empty")
    elif not Path(runs_root).is_absolute():
        c.fail(f"meta.json runs_root is not absolute: {runs_root!r}")
    else:
        c.ok(f"runs_root = {runs_root}")


def check_manifest(root: Path, c: Check) -> None:
    p = root / "manifest.json"
    if not p.exists():
        c.fail("manifest.json missing (launcher cleanup didn't run)")
        return
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        c.fail(f"manifest.json invalid JSON: {exc}")
        return
    c.ok(
        f"indexes {m.get('file_count', 0)} files "
        f"({m.get('total_bytes', 0)} bytes)"
    )
    for entry in m.get("files", []):
        path = entry.get("path", "")
        size = entry.get("size", 0)
        if path.endswith((".log", ".jsonl", ".txt", ".md")) and size == 0:
            c.fail(f"zero-byte text file: {path}")


def check_server_logs(root: Path, c: Check) -> None:
    server_dir = root / "server"
    if not server_dir.exists():
        c.fail("server/ missing")
        return
    for name in ("backend.log", "frontend.log"):
        p = server_dir / name
        if not p.exists():
            c.fail(f"server/{name} missing")
        elif p.stat().st_size == 0:
            c.fail(f"server/{name} is empty (process never started?)")
        else:
            c.ok(f"server/{name} ({p.stat().st_size} bytes)")


def check_pipeline_log(root: Path, c: Check) -> None:
    """Tier 2a — pipeline.log + pipeline.jsonl appear once an agent run starts."""
    log = root / "pipeline.log"
    jsonl = root / "pipeline.jsonl"
    if not log.exists() and not jsonl.exists():
        c.ok(
            "(no pipeline.log/.jsonl — no agent run launched this session; "
            "this is expected for server-only verifications)"
        )
        return
    if log.exists():
        if log.stat().st_size > 0:
            c.ok(f"pipeline.log ({log.stat().st_size} bytes)")
        else:
            c.fail("pipeline.log present but empty")
    else:
        c.fail("pipeline.log missing while pipeline.jsonl exists")
    if jsonl.exists():
        valid = invalid = 0
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
                valid += 1
            except json.JSONDecodeError:
                invalid += 1
        if invalid:
            c.fail(f"pipeline.jsonl has {invalid} malformed lines")
        else:
            c.ok(f"pipeline.jsonl has {valid} valid JSON records")


def check_projects(root: Path, c: Check) -> None:
    """Each prj_<id>/ should have dashboard_events.jsonl; agents/ is optional."""
    prj_dirs = sorted(root.glob("prj_*"))
    if not prj_dirs:
        c.ok(
            "(no prj_*/ — no pipeline workspace created this session; the "
            "Tier 1 'prj_ co-locates with server/' contract was not exercised "
            "by this run)"
        )
        return
    c.ok(f"found {len(prj_dirs)} project workspace(s)")
    for p in prj_dirs:
        de = p / "dashboard_events.jsonl"
        if not de.exists():
            c.fail(f"{p.name}/dashboard_events.jsonl missing")
        elif de.stat().st_size == 0:
            c.fail(f"{p.name}/dashboard_events.jsonl empty")
        agents_dir = p / "agents"
        if agents_dir.exists():
            _check_agent_dirs(agents_dir, c, prefix=p.name)
        else:
            c.ok(
                f"{p.name}: no agents/ (pre-Tier-2b code, or REPROLAB_LOG_DIR/"
                "REPROLAB_RUNS_ROOT was unset when the pipeline ran)"
            )


def _check_agent_dirs(agents_root: Path, c: Check, *, prefix: str) -> None:
    invocations = sorted(p for p in agents_root.iterdir() if p.is_dir())
    if not invocations:
        c.fail(f"{prefix}/agents/ exists but is empty")
        return
    c.ok(f"{prefix}/agents/: {len(invocations)} invocation(s)")
    for inv in invocations:
        label = f"{prefix}/agents/{inv.name}"
        missing = [f for f in REQUIRED_AGENT_FILES if not (inv / f).exists()]
        if missing:
            c.fail(f"{label}: missing {missing}")
            continue
        try:
            meta = json.loads((inv / "meta.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            c.fail(f"{label}/meta.json invalid: {exc}")
            continue
        status = meta.get("status", "?")
        if status not in {"ok", "error", "interrupted"}:
            c.fail(f"{label}: unexpected status={status!r}")
            continue
        result = (inv / "result.txt").read_text(encoding="utf-8")
        if status == "ok" and not result.strip():
            c.fail(f"{label}: status=ok but result.txt is empty")
            continue
        # tool_calls.jsonl format
        tj = inv / "tool_calls.jsonl"
        bad_lines = 0
        for line in tj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
        if bad_lines:
            c.fail(f"{label}/tool_calls.jsonl: {bad_lines} malformed lines")
            continue
        c.ok(
            f"{label}: status={status} msg_count={meta.get('msg_count')} "
            f"retries={meta.get('retries')}"
        )


CHECKS: list[tuple[str, Callable[[Path, Check], None]]] = [
    ("meta", check_meta),
    ("manifest", check_manifest),
    ("server", check_server_logs),
    ("pipeline", check_pipeline_log),
    ("projects", check_projects),
]


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.latest:
        candidates = sorted(p for p in Path("logs").glob("*/") if p.is_dir())
        if not candidates:
            print("no logs/<TS>/ dirs found", file=sys.stderr)
            raise SystemExit(1)
        return candidates[-1].resolve()
    if args.path is None:
        print("provide a path or --latest", file=sys.stderr)
        raise SystemExit(2)
    return Path(args.path).resolve()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="verify a logs/<TS>/ launch dir")
    ap.add_argument("path", nargs="?", help="path to logs/<TS>/")
    ap.add_argument(
        "--latest",
        action="store_true",
        help="use the newest logs/<TS>/ under ./logs/",
    )
    args = ap.parse_args(argv)

    root = _resolve_root(args)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    print(f"verifying: {root}\n")

    results: list[Check] = []
    for name, fn in CHECKS:
        c = Check(name)
        fn(root, c)
        results.append(c)

    for c in results:
        print(f"== {c.name} ==")
        for msg in c.passed:
            print(f"  OK   {msg}")
        for msg in c.failed:
            print(f"  FAIL {msg}")
        print()

    total_ok = sum(len(c.passed) for c in results)
    total_fail = sum(len(c.failed) for c in results)
    print(f"summary: {total_ok} passed, {total_fail} failed")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
