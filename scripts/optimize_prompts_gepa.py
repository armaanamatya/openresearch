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

from backend.agents.optimization.cost_tracker import LMCostTracker
from backend.agents.optimization.eval_budget import EvalBudgetEnforcer
from backend.agents.optimization.gepa_adapter import (
    SURFACES,
    ReproLabGEPAAdapter,
)
from backend.agents.optimization.mutable_regions import REGIONS
from backend.agents.optimization.reflection_lms import (
    make_api_key_reflection_lm,
    make_claude_oauth_reflection_lm,
)


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
    parser.add_argument(
        "--reflection-backend",
        choices=["api-key", "claude-oauth"],
        default="api-key",
        help="api-key: gepa.lm.LM(reflection_lm) for cost tracking. "
             "claude-oauth: route reflection through Claude subscription (no API balance needed, $0).",
    )
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

    # Reflection LM: a callable matching gepa's LanguageModel protocol.
    # API-key backend threads each call through LMCostTracker so the
    # final reflection_cost.json reflects actual reflection-LM spend.
    cost_tracker = LMCostTracker(run_dir=run_dir)
    if args.reflection_backend == "claude-oauth":
        reflection_lm_obj = make_claude_oauth_reflection_lm(model_name=args.reflection_lm)
    else:
        reflection_lm_obj = make_api_key_reflection_lm(args.reflection_lm, tracker=cost_tracker)

    result = gepa.optimize(
        seed_candidate=_seed_candidate(surface),
        trainset=train,
        valset=val,
        adapter=adapter,
        task_lm=args.task_lm,
        reflection_lm=reflection_lm_obj,
        max_metric_calls=args.max_metric_calls,
        use_merge=True,
        max_merge_invocations=args.max_merge_invocations,
        run_dir=str(run_dir),
        cache_evaluation=False,  # we manage cache via surface_salt; gepa-level cache is a poison vector
        seed=args.seed,
    )

    # Persist final reflection-LM spend. Includes total_metric_calls from
    # GEPAResult so callers can sanity-check call count against the tracker.
    snapshot = cost_tracker.snapshot()
    (run_dir / "reflection_cost.json").write_text(
        json.dumps({
            "total_cost_usd": snapshot.total_cost_usd,
            "total_tokens_in": snapshot.total_tokens_in,
            "total_tokens_out": snapshot.total_tokens_out,
            "tracked_call_count": snapshot.call_count,
            "gepa_total_metric_calls": getattr(result, "total_metric_calls", None),
            "model": args.reflection_lm,
            "backend": args.reflection_backend,
        }, indent=2),
        encoding="utf-8",
    )

    # Persist Pareto front: gepa.GEPAResult exposes val_aggregate_scores
    # parallel to .candidates. Write a JSONL with the score for each candidate.
    with (run_dir / "pareto_front.jsonl").open("w", encoding="utf-8") as f:
        for i, cand in enumerate(result.candidates):
            score = result.val_aggregate_scores[i] if i < len(result.val_aggregate_scores) else None
            f.write(json.dumps({"idx": i, "score": score, "candidate": cand}, default=str) + "\n")

    # Bridge: write one proposed_mutations/<id>/ dir per non-seed candidate
    # that scored ≥ the seed. Promotion script reads these.
    seed_score = result.val_aggregate_scores[0] if result.val_aggregate_scores else 0.0
    seed_cand = _seed_candidate(surface)
    mut_root = run_dir / "proposed_mutations"
    mut_root.mkdir(exist_ok=True)
    proposed = 0
    for i, cand in enumerate(result.candidates):
        if i == 0:  # seed
            continue
        score = result.val_aggregate_scores[i] if i < len(result.val_aggregate_scores) else 0.0
        if score < seed_score:
            continue
        # One mutation dir per (component_id that actually changed).
        for cid, after_text in cand.items():
            before_text = seed_cand.get(cid, "")
            if before_text == after_text:
                continue
            mid = f"cand{i}-{cid.replace('.', '-')}"
            d = mut_root / mid
            d.mkdir(exist_ok=True)
            (d / "metadata.json").write_text(json.dumps({
                "component_id": cid,
                "mutation_id": mid,
                "candidate_index": i,
                "val_aggregate_score": score,
                "seed_val_aggregate_score": seed_score,
            }, indent=2), encoding="utf-8")
            (d / "before.txt").write_text(before_text, encoding="utf-8")
            (d / "after.txt").write_text(after_text, encoding="utf-8")
            (d / "pr_body.md").write_text(_render_pr_body(
                cid=cid, mid=mid, score=score, seed_score=seed_score,
                run_dir=run_dir, before=before_text, after=after_text,
            ), encoding="utf-8")
            proposed += 1

    print(f"GEPA optimization complete. Artifacts: {run_dir}")
    print(f"  candidates: {len(result.candidates)}, proposed_mutations: {proposed}")
    print(f"  next: python scripts/optimize_prompts_gepa.py validate "
          f"--mutation-dir <one of {mut_root}/*>")
    return 0


def _render_pr_body(*, cid: str, mid: str, score: float, seed_score: float,
                     run_dir: Path, before: str, after: str) -> str:
    delta = score - seed_score
    return (
        f"# GEPA mutation: `{cid}`\n\n"
        f"**Mutation id:** `{mid}`\n"
        f"**Optimization run:** `{run_dir.name}`\n"
        f"**Val score:** {score:.4f} (seed {seed_score:.4f}, Δ {delta:+.4f})\n\n"
        f"## Audit checklist (G6 + G7)\n"
        f"- [ ] Side-by-side diff reviewed below\n"
        f"- [ ] Can a human read this prompt at a glance? (G7 readability)\n"
        f"- [ ] `validation_run.json` `status == \"passed\"`\n"
        f"- [ ] No immutable region addressed (G3)\n\n"
        f"## Before ({len(before)} chars)\n```\n{before[:2000]}{'...[truncated]' if len(before) > 2000 else ''}\n```\n\n"
        f"## After ({len(after)} chars)\n```\n{after[:2000]}{'...[truncated]' if len(after) > 2000 else ''}\n```\n"
    )


# ---------------------------------------------------------------------------
# `validate` subcommand — runs one real eval against the held-out paper +
# writes validation_run.json that promote_gepa_mutation.py reads (§4.6).
# ---------------------------------------------------------------------------


def validate_mutation(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run §4.6 validation for one mutation")
    parser.add_argument("--mutation-dir", type=Path, required=True)
    parser.add_argument("--held-out", required=True, help="arxiv id of held-out paper")
    parser.add_argument("--task-lm", default="openai/gpt-5")
    parser.add_argument("--max-usd", type=float, default=1.00)
    args = parser.parse_args(argv)

    mut = args.mutation_dir.resolve()
    meta = json.loads((mut / "metadata.json").read_text())
    component_id = meta["component_id"]
    after_text = (mut / "after.txt").read_text()

    # Determine surface from component_id.
    surface = None
    for s, cids in SURFACES.items():
        if component_id in cids:
            surface = s
            break
    if surface is None:
        print(f"ERROR: unknown component_id {component_id}", file=sys.stderr)
        return 2

    # Run one eval against held-out via the adapter's subprocess entrypoint.
    val_run_dir = mut / "validation_run"
    val_run_dir.mkdir(exist_ok=True)
    adapter = ReproLabGEPAAdapter(
        surface=surface,
        opt_run_dir=mut,
        task_lm=args.task_lm,
        budget=EvalBudgetEnforcer(max_usd=args.max_usd, max_wall_clock_seconds=3600),
        cache_strategy="disabled",  # §4.5: validation always disables cache
    )
    candidate = {component_id: after_text}
    batch = [{"paper_id": args.held_out, "archetype": "other"}]
    try:
        result = adapter.evaluate(batch=batch, candidate=candidate, capture_traces=True)
        score = float(result.scores[0]) if result.scores else 0.0
        status = "passed" if score > 0.0 else "failed"
    except Exception as exc:  # noqa: BLE001
        score, status = 0.0, "failed"
        (val_run_dir / "exception.txt").write_text(str(exc), encoding="utf-8")

    (mut / "validation_run.json").write_text(json.dumps({
        "status": status,
        "score": score,
        "held_out": args.held_out,
        "task_lm": args.task_lm,
        "component_id": component_id,
    }, indent=2), encoding="utf-8")
    print(f"validation: status={status} score={score:.4f}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    # Subcommand dispatch: `validate ...` runs validate_mutation; everything
    # else is the optimization driver.
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        sys.exit(validate_mutation(sys.argv[2:]))
    sys.exit(main())
