"""Stage-transition monitor for a running pipeline.

Tails dashboard_events.jsonl, runner.stderr.log, and pipeline.log, emitting
one stdout line per:
  - agent_completed for any agent in the wait-set
  - gate result event
  - runner / pipeline failure signal (Traceback, FAILED, TransientError, etc.)

Exits cleanly when every agent in the wait-set has completed (so the parent
Monitor's notification stream naturally terminates and the orchestrator
knows we're done).

Usage:
    python _monitor_stages.py <log_dir> <project_id> <agent_id>[,<agent_id>...]

Example:
    python _monitor_stages.py logs/20260519-001647 prj_9cbac43dcba7d926 \\
        paper-understanding,artifact-discovery
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# tuneables: cheap local poll
POLL_SECONDS = 2.0
FAILURE_NEEDLES = (
    "X FAILED:",
    "Traceback",
    "TransientError",
    "blocked_requires_human",
    "STOPPED at",
)
PIPELINE_NEEDLES = (
    "[Gate",
    "Running .* Agent",  # not actually used as a regex — substring match
)


def _tail_new(path: Path, last_size: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], last_size
    size = path.stat().st_size
    if size <= last_size:
        return [], last_size
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], last_size
    return text[last_size:].splitlines(), size


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: _monitor_stages.py <log_dir> <project_id> <agent_id>[,<agent_id>...]", file=sys.stderr)
        return 2

    log_dir = Path(argv[1])
    project_id = argv[2]
    wait_set: dict[str, bool] = {a: False for a in argv[3].split(",")}

    dash = log_dir / project_id / "dashboard_events.jsonl"
    runner = log_dir / project_id / "runner.stderr.log"
    pipeline = log_dir / "pipeline.log"

    last_dash = last_run = last_pipe = 0
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"BEGIN monitor at {start_iso} watching {sorted(wait_set)}", flush=True)

    while True:
        # Stage completions + gate results — the primary signal.
        lines, last_dash = _tail_new(dash, last_dash)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = e.get("event")
            aid = e.get("agentId")
            if evt == "agent_completed" and aid in wait_set and not wait_set[aid]:
                wait_set[aid] = True
                task = (e.get("agent") or {}).get("currentTask", "")[:80]
                print(f"STAGE_COMPLETE: {aid} :: {task}", flush=True)
            elif evt == "verification_gate_result":
                stage = e.get("stage", "?")
                status = e.get("status", "?")
                detail = (e.get("detail") or "")[:80]
                print(f"GATE_EVENT: {stage} status={status} detail={detail}", flush=True)

        # Failure surface in the runner subprocess stderr.
        lines, last_run = _tail_new(runner, last_run)
        for line in lines:
            stripped = line.strip()
            for needle in FAILURE_NEEDLES:
                if needle in stripped:
                    print(f"RUNNER_ERROR: {stripped[:160]}", flush=True)
                    break

        # Root-logger summary lines that the orchestrator emits.
        lines, last_pipe = _tail_new(pipeline, last_pipe)
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if (
                "Agent " in stripped and " completed " in stripped
            ) or "[Gate" in stripped or "rubric-verifier" in stripped:
                # Trim the timestamp prefix to keep notifications compact.
                if " :: " in stripped:
                    payload = stripped.split(" :: ", 1)[1]
                else:
                    payload = stripped
                print(f"PIPELINE: {payload[:140]}", flush=True)

        if all(wait_set.values()):
            print("DONE: all watched stages complete; monitor exiting", flush=True)
            return 0

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
