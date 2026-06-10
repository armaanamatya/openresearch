"""PaperBench subcommand: drives bundle inspection and dry-mode runs.

Mode 1 (no API key required):
    openresearch paperbench list
    openresearch paperbench summary --paper-id ftrl

Mode 2 (dry run — no LLM, placeholder submission only):
    openresearch paperbench run --paper-id ftrl [--bundles-root third_party/paperbench]

Status JSON is persisted to ``<runs_root>/paperbench/<run_group_id>/status.json``
so the frontend can poll it without coupling to the in-process Python state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.evals.paperbench import (
    code_development_ceiling,
    create_submission_manifest,
    load_paperbench_bundle,
    summarize_rubric,
)
from backend.evals.paperbench.bundle import PaperBenchBundle, PaperBenchBundleError


DEFAULT_BUNDLES_ROOT = Path("third_party/paperbench")


# Published PaperBench BasicAgent baselines (Tables 11 + 15 of the PaperBench paper,
# OpenAI April 2025). Used purely for display next to our score; never treated as
# ground truth for our pipeline.
PUBLISHED_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "ftrl": {
        "claude_3_5_sonnet_basicagent": {"mean": 0.093, "se": 0.010},
        "o1_basicagent": {"mean": 0.017, "se": 0.008},
        "o3_mini_basicagent": {"mean": 0.003, "se": 0.002},
        "gpt_4o_basicagent": {"mean": 0.030, "se": 0.017},
    },
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_bundles_root(value: str | None) -> Path:
    raw = Path(value) if value else DEFAULT_BUNDLES_ROOT
    return raw.expanduser().resolve()


def _load_bundle_or_exit(bundles_root: Path, paper_id: str) -> PaperBenchBundle:
    try:
        return load_paperbench_bundle(bundles_root, paper_id)
    except PaperBenchBundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def cmd_paperbench_list(args: argparse.Namespace) -> int:
    bundles_root = _resolve_bundles_root(getattr(args, "bundles_root", None))
    if not bundles_root.is_dir():
        print(f"error: bundles root does not exist: {bundles_root}", file=sys.stderr)
        return 2
    bundles: list[dict[str, Any]] = []
    for entry in sorted(bundles_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            bundle = load_paperbench_bundle(bundles_root, entry.name)
        except PaperBenchBundleError as exc:
            bundles.append({"paper_id": entry.name, "error": str(exc)})
            continue
        bundles.append(
            {
                "paper_id": bundle.paper_id,
                "metadata": bundle.metadata(),
                "has_addendum": bundle.addendum_path.is_file(),
                "rubric_path": str(bundle.rubric_path),
            }
        )
    json.dump({"bundles_root": str(bundles_root), "bundles": bundles}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_paperbench_summary(args: argparse.Namespace) -> int:
    bundles_root = _resolve_bundles_root(getattr(args, "bundles_root", None))
    bundle = _load_bundle_or_exit(bundles_root, args.paper_id)
    rubric = bundle.rubric()
    summary = summarize_rubric(rubric)
    code_ceiling = code_development_ceiling(rubric)
    payload: dict[str, Any] = {
        "paper_id": bundle.paper_id,
        "metadata": bundle.metadata(),
        "rubric_summary": summary.to_dict(),
        "code_development_ceiling": code_ceiling,
        "blacklist_entries": list(bundle.blacklist_entries()),
        "published_baselines": PUBLISHED_BASELINES.get(bundle.paper_id, {}),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_paperbench_status(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root).expanduser().resolve()
    status_path = runs_root / "paperbench" / args.run_group_id / "status.json"
    if not status_path.is_file():
        print(f"error: status not found: {status_path}", file=sys.stderr)
        return 2
    sys.stdout.write(status_path.read_text(encoding="utf-8"))
    if not status_path.read_text(encoding="utf-8").endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_paperbench_run(args: argparse.Namespace) -> int:
    bundles_root = _resolve_bundles_root(getattr(args, "bundles_root", None))
    bundle = _load_bundle_or_exit(bundles_root, args.paper_id)

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_group_id = args.run_group_id or _make_run_group_id(bundle.paper_id)
    run_dir = runs_root / "paperbench" / run_group_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rubric = bundle.rubric()
    rubric_summary = summarize_rubric(rubric).to_dict()
    code_ceiling = code_development_ceiling(rubric)

    status: dict[str, Any] = {
        "run_group_id": run_group_id,
        "paper_id": bundle.paper_id,
        "bundle_root": str(bundle.root),
        "runs_root": str(runs_root),
        "mode": "dry",
        "status": "pending",
        "started_at": _utcnow(),
        "updated_at": _utcnow(),
        "completed_at": None,
        "attempts": [],
        "rubric_summary": rubric_summary,
        "code_development_ceiling": code_ceiling,
        "published_baselines": PUBLISHED_BASELINES.get(bundle.paper_id, {}),
        "blacklist_entries": list(bundle.blacklist_entries()),
        "mean_score": None,
        "standard_error": None,
        "n_attempts": 0,
        "error": None,
    }
    _write_status(run_dir, status)

    # Emit the run handle immediately so callers (the Next.js API route, CI
    # scripts) get a stable id to poll even if the pipeline later fails.
    json.dump(
        {"run_group_id": run_group_id, "status_path": str(run_dir / "status.json")},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()

    try:
        _run_dry(bundle, run_dir, status)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any failure into status JSON
        status["status"] = "failed"
        status["error"] = repr(exc)
        status["updated_at"] = _utcnow()
        status["completed_at"] = _utcnow()
        _write_status(run_dir, status)
        print(f"error: paperbench run failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_dry(bundle: PaperBenchBundle, run_dir: Path, status: dict[str, Any]) -> None:
    """Produce a placeholder submission tree and validate it.

    Useful for demos / CI where no API key is available. Demonstrates the
    PaperBench submission contract end-to-end without invoking any LLM.
    """

    submission_dir = run_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    reproduce_sh = submission_dir / "reproduce.sh"
    if not reproduce_sh.is_file():
        reproduce_sh.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "echo 'paperbench dry-run placeholder; replace with real pipeline output'\n",
            encoding="utf-8",
        )
    os.chmod(reproduce_sh, 0o755)
    readme = submission_dir / "README.md"
    if not readme.is_file():
        readme.write_text(
            f"# Placeholder submission for paper `{bundle.paper_id}`\n\n"
            "This directory was produced by `openresearch paperbench run` in dry mode.\n"
            "This is a dry-run submission; no pipeline outputs were generated.\n",
            encoding="utf-8",
        )
    manifest = create_submission_manifest(
        bundle.paper_id,
        submission_dir,
        metadata={"mode": "dry", "run_group_id": status["run_group_id"]},
        write=True,
    )
    status["status"] = "succeeded"
    status["completed_at"] = _utcnow()
    status["updated_at"] = _utcnow()
    status["attempts"] = [
        {
            "attempt_id": f"{status['run_group_id']}-dry",
            "seed": None,
            "status": "succeeded",
            "submission_dir": str(submission_dir),
            "submission_validation": _validation_to_dict(manifest.validation),
        }
    ]
    status["n_attempts"] = 1
    _write_status(run_dir, status)


def _validation_to_dict(validation: Any) -> dict[str, Any]:
    return {
        "ok": validation.ok,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "total_bytes": validation.total_bytes,
        "file_count": validation.file_count,
        "committed_bytes": validation.committed_bytes,
    }


def _write_status(run_dir: Path, status: dict[str, Any]) -> None:
    path = run_dir / "status.json"
    path.write_text(json.dumps(status, indent=2, sort_keys=False), encoding="utf-8")


def _make_run_group_id(paper_id: str) -> str:
    suffix = uuid.uuid4().hex[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"pb_{paper_id}_{stamp}_{suffix}"


def add_paperbench_subparser(subparsers: argparse._SubParsersAction) -> None:
    pb = subparsers.add_parser("paperbench", help="PaperBench bundle inspection and dry-mode runs.")
    pb_sub = pb.add_subparsers(dest="pb_cmd", required=True)

    listing = pb_sub.add_parser("list", help="List vendored PaperBench bundles.")
    listing.add_argument("--bundles-root", default=None)
    listing.set_defaults(func=cmd_paperbench_list)

    summary = pb_sub.add_parser("summary", help="Show rubric breakdown and code-only ceiling for a paper.")
    summary.add_argument("--paper-id", required=True)
    summary.add_argument("--bundles-root", default=None)
    summary.set_defaults(func=cmd_paperbench_summary)

    status = pb_sub.add_parser("status", help="Print the latest status JSON for a run group.")
    status.add_argument("--run-group-id", required=True)
    status.set_defaults(func=cmd_paperbench_status)

    run = pb_sub.add_parser("run", help="Start a PaperBench dry run (placeholder submission only).")
    run.add_argument("--paper-id", required=True)
    run.add_argument("--bundles-root", default=None)
    run.add_argument("--run-group-id", default=None, help="Override the auto-generated run group id.")
    run.set_defaults(func=cmd_paperbench_run)


__all__ = [
    "PUBLISHED_BASELINES",
    "add_paperbench_subparser",
    "cmd_paperbench_list",
    "cmd_paperbench_run",
    "cmd_paperbench_status",
    "cmd_paperbench_summary",
]
