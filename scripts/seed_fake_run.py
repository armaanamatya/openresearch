#!/usr/bin/env python3
"""Seed the fake runs the Playwright e2e suite asserts against.

``e2e/reports.spec.ts`` and ``e2e/lab-smoke-interactive.spec.ts`` referenced a
``scripts/seed-fake-run.sh`` that was never committed (audit 2026-06-10) — the
seeded specs could only ever pass on the machine where that script lived. This
is that seeder, rebuilt on the REAL backend builders
(:mod:`backend.agents.worker_reports`) so the seeded artifacts can never drift
from the schema the UI reads.

Creates (idempotently, under --runs-root, default ./runs):
  - ``seed_reports_test``   — failed run with 3 worker reports (one failed with
    an "SDK success-with-no-text" blocker + a nonzero-exit command — the exact
    shapes ``reports.spec.ts`` asserts).
  - ``prj_diffusion_smoke`` — failed run (library row outside the Completed filter).
  - ``prj_completed_smoke`` — completed run with a final_report.json (library
    row inside the Completed filter).

Usage:
    .venv/bin/python scripts/seed_fake_run.py [--runs-root runs]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents.worker_reports import (  # noqa: E402
    build_extended_worker_report,
    finalize_worker_report,
    open_worker_report,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _demo_status(run_dir: Path, *, status: str, title: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "projectId": run_dir.name,
        "outputDir": str(run_dir),
        "runMode": "rlm",
        "status": status,
        "paperId": run_dir.name,
        "paperTitle": title,
        "paper": {"id": run_dir.name, "title": title},
        "startedAt": _now(),
        "updatedAt": _now(),
        "completedAt": _now(),
        "primitiveProvider": "stub",
        "seeded": True,
    }
    (run_dir / "demo_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_reports_run(runs_root: Path) -> None:
    run_dir = runs_root / "seed_reports_test"
    _demo_status(run_dir, status="failed", title="Seeded worker-reports fixture")
    # Fresh reports each seed so re-runs don't accumulate duplicates.
    # open_worker_report writes reports/worker_reports/<uuid>.json,
    # reports/worker_reports.jsonl, AND a legacy run-root jsonl — reset all.
    import shutil

    reports_dir = run_dir / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    (run_dir / "worker_reports.jsonl").unlink(missing_ok=True)

    def _worker(agent_id: str, status: str, commands=None, **finalize_kw) -> None:
        report = build_extended_worker_report(
            run_id=run_dir.name,
            worker_type="rlm_primitive",
            agent_id=agent_id,
            project_dir=run_dir,
            model="claude-sonnet-4-6",
            provider="anthropic",
            status="running",
            assignment={"summary": f"RLM primitive: {agent_id}"},
        )
        if commands:
            # Top-level report.commands is what the UI report-rail renders
            # (command-exit-code testid) — NOT execution_summary.commands.
            report["commands"] = commands
        open_worker_report(run_dir, report)
        finalize_worker_report(run_dir, report, status=status, **finalize_kw)

    _worker("implement_baseline", "completed",
            execution_summary={"summary": "baseline generated"},
            raw_text='{"ok": true, "code_path": "code/"}')
    _worker(
        "run_experiment", "failed",
        blockers=[{
            "title": "SDK success-with-no-text",
            "description": "claude-agent-sdk returned success with an empty body; "
                           "auth/SDK failure, not Docker (see CLAUDE.md sandboxes note).",
            "severity": "critical",
            "source": "sdk",
            "suggested_fix": "check ANTHROPIC_API_KEY / claude login",
        }],
        errors=[{"message": "experiment exited nonzero", "recoverable": True}],
        commands=[{"command": "python train.py", "exit_code": 1, "duration_ms": 4200}],
        execution_summary={"summary": "training crashed at import"},
        raw_text='{"success": false, "error": "exit 1"}',
        error="experiment exited nonzero",
    )
    _worker("verify_against_rubric", "blocked",
            blockers=[{
                "title": "no metrics to grade",
                "description": "experiment produced no metrics.json",
                "severity": "high",
                "source": "rubric_guard",
            }],
            raw_text='{"graded": false}')
    print(f"[seed] {run_dir.name}: 3 worker reports")


def seed_library_runs(runs_root: Path) -> None:
    diffusion = runs_root / "prj_diffusion_smoke"
    _demo_status(diffusion, status="failed", title="Diffusion smoke (seeded)")
    print(f"[seed] {diffusion.name}: failed")

    completed = runs_root / "prj_completed_smoke"
    _demo_status(completed, status="completed", title="Completed smoke (seeded)")
    (completed / "final_report.json").write_text(json.dumps({
        "verdict": "partial",
        "reproduction_summary": "seeded fixture run",
        "baseline_metrics": {"accuracy": 0.5},
        "rubric": {"overall_score": 0.5},
        "mode": "rlm",
        "started_at": _now(),
        "completed_at": _now(),
    }, indent=2), encoding="utf-8")
    print(f"[seed] {completed.name}: completed + final_report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "projects", nargs="*",
        help="ignored (back-compat: historical invocations named projects; "
             "all fixtures are always seeded)",
    )
    args = parser.parse_args(argv)
    runs_root = args.runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    seed_reports_run(runs_root)
    seed_library_runs(runs_root)
    print("[seed] done — e2e seeded specs are now runnable against a live backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
