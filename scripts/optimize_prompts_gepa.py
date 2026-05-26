#!/usr/bin/env python3
"""Driver for one GEPA optimization run.

Loads trainset/valset, validates G2 (held-out ≥30%), constructs the adapter,
and calls ``gepa.optimize(...)``. Writes the §4.5 artifact tree.

See ``docs/superpowers/specs/2026-05-26-gepa-phase0-audit.md`` §0.6 for the
CLI flag spec.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python scripts/optimize_prompts_gepa.py` from repo root
# without requiring PYTHONPATH=.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.agents.optimization.eval_budget import EvalBudgetEnforcer
from backend.agents.optimization.gepa_adapter import (
    SURFACES,
    ReproLabGEPAAdapter,
)
from backend.agents.optimization.mutable_regions import REGIONS


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_paper_list(path: Path) -> list[dict]:
    """Load arxiv IDs from a JSON file and pair each with its archetype label."""
    raw = json.loads(path.read_text())
    ids: list[str] = raw if isinstance(raw, list) else raw.get("papers", [])
    out = []
    for pid in ids:
        archetype = "other"
        af = Path("tests/fixtures/papers") / pid / "archetype.txt"
        if af.exists():
            archetype = af.read_text().strip()
        out.append({"paper_id": pid, "archetype": archetype})
    return out


def _validate_split(train: list[dict], val: list[dict], held_out: str) -> None:
    train_ids = {r["paper_id"] for r in train}
    val_ids = {r["paper_id"] for r in val}
    overlap = train_ids & val_ids
    if overlap:
        raise SystemExit(f"trainset and valset overlap: {sorted(overlap)}")
    if held_out in train_ids or held_out in val_ids:
        raise SystemExit(f"held-out paper {held_out!r} appears in train/val (contamination)")
    total = len(train) + len(val)
    if total == 0:
        raise SystemExit("trainset+valset is empty")
    val_frac = len(val) / total
    if val_frac < 0.30:
        raise SystemExit(f"G2 violation: valset is {val_frac:.1%} of total, need ≥30%")


def _seed_candidate(surface: str) -> dict[str, str]:
    """Initial candidate = current default text for every component in the surface."""
    return {cid: REGIONS[cid].default_text for cid in SURFACES[surface]}


def _git_snapshot(out_path: Path) -> None:
    """Write the current diff against main as ``source_snapshot.patch``."""
    try:
        diff = subprocess.run(
            ["git", "diff", "main", "--", "backend/", "scripts/"],
            capture_output=True, text=True, check=False,
        ).stdout
        out_path.write_text(diff, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        out_path.write_text(f"<snapshot failed: {exc}>\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=["A", "B", "C"], required=True)
    parser.add_argument("--task-lm", default="openai/gpt-5")
    parser.add_argument("--reflection-lm", default="openai/gpt-5")
    parser.add_argument("--trainset", type=Path, required=True)
    parser.add_argument("--valset", type=Path, required=True)
    parser.add_argument("--held-out", required=True, help="arxiv id for §4.6 validation")
    parser.add_argument("--max-metric-calls", type=int, default=100)
    parser.add_argument("--max-merge-invocations", type=int, default=5)
    parser.add_argument("--max-usd-per-eval", type=float, default=0.50)
    parser.add_argument("--max-parallel-pods", type=int,
                        default=int(os.environ.get("REPROLAB_GEPA_MAX_PARALLEL_PODS", "2")))
    parser.add_argument("--cache-strategy", choices=["scoped", "disabled"], default="scoped")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    surface = {"A": "root_system", "B": "improvement", "C": "baseline_agent"}[args.lane]

    train = _load_paper_list(args.trainset)
    val = _load_paper_list(args.valset)
    _validate_split(train, val, args.held_out)

    run_dir = args.run_dir or Path("runs/_gepa") / _utc_ts()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "lane": args.lane,
        "surface": surface,
        "task_lm": args.task_lm,
        "reflection_lm": args.reflection_lm,
        "seed": args.seed,
        "max_metric_calls": args.max_metric_calls,
        "max_merge_invocations": args.max_merge_invocations,
        "max_usd_per_eval": args.max_usd_per_eval,
        "cache_strategy": args.cache_strategy,
        "trainset_size": len(train),
        "valset_size": len(val),
        "held_out": args.held_out,
        "components": list(SURFACES[surface]),
    }, indent=2), encoding="utf-8")
    (run_dir / "trainset.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (run_dir / "valset.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    _git_snapshot(run_dir / "source_snapshot.patch")

    budget = EvalBudgetEnforcer(
        max_usd=args.max_usd_per_eval,
        max_parallel_pods=args.max_parallel_pods,
    )
    adapter = ReproLabGEPAAdapter(
        surface=surface,
        opt_run_dir=run_dir,
        task_lm=args.task_lm,
        budget=budget,
        cache_strategy=args.cache_strategy,
    )

    try:
        import gepa  # type: ignore[import-not-found]
    except ImportError:
        print(
            "ERROR: gepa not installed. Pin lives in backend/requirements.txt;\n"
            "       install via Docker (the local brew Py3.14 has a broken pyexpat ABI\n"
            "       that blocks pip — see docs/superpowers/specs/2026-05-26-gepa-phase0-audit.md §0.1).",
            file=sys.stderr,
        )
        return 2

    result = gepa.optimize(
        seed_candidate=_seed_candidate(surface),
        trainset=train,
        valset=val,
        adapter=adapter,
        task_lm=args.task_lm,
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_metric_calls,
        use_merge=True,
        max_merge_invocations=args.max_merge_invocations,
        run_dir=str(run_dir),
        cache_evaluation=False,  # we manage cache via surface_salt; gepa-level cache is a poison vector
        seed=args.seed,
    )

    # Persist Pareto front + mutations log (gepa writes most into its own
    # run_dir; we mirror the spec §4.5 names for human discoverability).
    (run_dir / "pareto_front.jsonl").write_text(
        "\n".join(json.dumps(c, default=str) for c in getattr(result, "pareto_candidates", [])),
        encoding="utf-8",
    )
    print(f"GEPA optimization complete. Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
