#!/usr/bin/env python3
"""Promote one GEPA-accepted mutation into a reviewable PR.

Usage:
    python scripts/promote_gepa_mutation.py runs/_gepa/<ts>/<mutation_id>

Refuses promotion when:
- ``validation_run.json`` is missing or its ``status != "passed"`` (§4.6)
- ``before.txt`` / ``after.txt`` / ``pr_body.md`` are missing
- The mutation's ``component_id`` is not in the immutable-safe REGIONS dict
  (defense in depth — the adapter already refuses, but we re-check at the
  promotion seam)

Promotion writes a patch to the source module that owns ``default_text`` for
the component and opens a PR via ``gh pr create`` tagged ``gepa-mutation``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from backend.agents.optimization.mutable_regions import REGIONS

# Where each component's default_text lives in source. The promotion script
# patches the constant in-place at this path.
COMPONENT_SOURCE: dict[str, tuple[str, str]] = {
    "improvement.orchestrator.body": (
        "backend/agents/prompts/improvement.py",
        "IMPROVEMENT_ORCHESTRATOR_PROMPT",
    ),
    "improvement.pool_generation.body": (
        "backend/agents/prompts/improvement.py",
        "ADAPTIVE_POOL_GENERATION_PROMPT",
    ),
    "improvement.rerank.body": (
        "backend/agents/prompts/improvement.py",
        "ADAPTIVE_RERANK_PROMPT",
    ),
    "improvement.orchestrator_round_n.body": (
        "backend/agents/prompts/improvement.py",
        "IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT",
    ),
    "baseline_agent.body": (
        "backend/agents/prompts/baseline_implementation.py",
        "BASELINE_IMPLEMENTATION_PROMPT",
    ),
    "root_system.decomposition_example": (
        "backend/agents/rlm/system_prompt.py",
        "_DECOMPOSITION_EXAMPLE",
    ),
}


def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]  # noqa: F821
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def _read_required(path: Path) -> str:
    if not path.exists():
        _die(f"missing required artifact: {path}")
    return path.read_text(encoding="utf-8")


def _read_required_json(path: Path) -> dict:
    return json.loads(_read_required(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutation_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the patch + PR body without opening a PR")
    args = parser.parse_args(argv)

    mut = Path(args.mutation_dir).resolve()
    if not mut.is_dir():
        _die(f"not a directory: {mut}")

    meta_path = mut / "metadata.json"
    if not meta_path.exists():
        _die(f"missing {meta_path} (component_id, mutation_id)")
    meta = _read_required_json(meta_path)
    component_id = meta.get("component_id")
    if component_id not in REGIONS:
        _die(f"unknown component_id {component_id!r}; not in REGIONS")
    if component_id not in COMPONENT_SOURCE:
        _die(f"no source-file mapping for {component_id!r} (promotion not yet wired)")

    before = _read_required(mut / "before.txt")
    after = _read_required(mut / "after.txt")
    pr_body = _read_required(mut / "pr_body.md")

    val_path = mut / "validation_run.json"
    if not val_path.exists():
        _die("validation_run.json missing — §4.6 requires a held-out real run before promotion")
    validation = _read_required_json(val_path)
    if validation.get("status") != "passed":
        _die(f"validation_run.json status={validation.get('status')!r}; only 'passed' may promote")

    # Sanity check: the registry's current default_text must equal ``before``
    # (otherwise main has drifted since the optimization run and the patch is stale).
    if REGIONS[component_id].default_text != before:
        _die(
            "registry default_text has drifted since optimization; "
            "main has changed under this mutation. Re-run optimization."
        )

    src_path_rel, var_name = COMPONENT_SOURCE[component_id]
    src_path = Path(src_path_rel)
    if not src_path.exists():
        _die(f"source file missing: {src_path}")

    src = src_path.read_text(encoding="utf-8")
    # Replace the first occurrence of the variable assignment containing ``before``.
    # Sentinel-based replacement is more robust than regex against multi-line strings.
    needle = before
    if needle not in src:
        _die(
            f"could not locate before-text verbatim in {src_path} (assignment to "
            f"{var_name!r}); the constant may have been edited."
        )
    new_src = src.replace(needle, after, 1)

    branch = f"gepa-mutation/{component_id.replace('.', '-')}-{mut.name}"
    if args.dry_run:
        print(f"--- dry-run for {component_id} → {src_path} ({var_name})")
        print(f"would branch: {branch}")
        print(f"--- pr_body.md ---\n{pr_body}")
        return 0

    src_path.write_text(new_src, encoding="utf-8")
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", str(src_path)], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"gepa-mutation: {component_id} ({mut.name})"],
        check=True,
    )
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)
    subprocess.run(
        [
            "gh", "pr", "create",
            "--title", f"GEPA mutation: {component_id} ({mut.name})",
            "--body", pr_body,
            "--label", "gepa-mutation",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
