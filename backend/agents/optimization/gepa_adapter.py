"""ReproLab GEPA adapter — implements the ``gepa.core.adapter.GEPAAdapter`` contract.

Each ``evaluate`` call spawns one subprocess per (candidate, paper) eval. The
subprocess inherits the GEPA candidate text via the
``REPROLAB_GEPA_CANDIDATE_JSON`` env var, the surface salt via
``REPROLAB_PRIMITIVE_CACHE_SURFACE_SALT``, and writes its artifacts to a
candidate-scoped directory under ``runs/_gepa/<ts>/candidates/<hash>/evals/<paper>/``.

Subprocess isolation is intentional: ``run_pipeline_rlm`` is not designed for
in-process concurrent invocation (existing ``run_isolated`` wrapper assumes a
single SDK client per process). Subprocess-per-eval inherits the existing
isolation, runs the spawn through a thin entrypoint, and re-establishes the
``PromptOverrideContext`` inside the child.

`gepa` library is an optional import — this module type-checks and unit-tests
without it; only the ``optimize_with_gepa`` entry point actually requires
``import gepa`` at call time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.agents.optimization.eval_budget import EvalBudgetEnforcer
from backend.agents.optimization.mutable_regions import REGIONS
from backend.agents.optimization.trace_minimizer import build_record

logger = logging.getLogger(__name__)

# The 5 GEPA surfaces this adapter knows how to optimize. Each surface lists
# the component_ids that are joint-optimized when that lane is selected.
SURFACES: dict[str, tuple[str, ...]] = {
    "improvement": (
        "improvement.orchestrator.body",
        "improvement.pool_generation.body",
        "improvement.rerank.body",
        "improvement.orchestrator_round_n.body",
    ),
    "root_system": ("root_system.decomposition_example",),
    "baseline_agent": ("baseline_agent.body",),
}


def candidate_hash(candidate: dict[str, str]) -> str:
    """Stable short hash of a candidate dict for cache + dir namespacing."""
    blob = json.dumps(candidate, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class EvalRequest:
    paper_id: str
    archetype: str
    run_dir: Path  # candidate-scoped, isolated


@dataclass
class EvalResult:
    score: float
    record: dict
    output: dict = field(default_factory=dict)


class ReproLabGEPAAdapter:
    """GEPA contract impl. See spec §4."""

    def __init__(
        self,
        *,
        surface: str,
        opt_run_dir: Path,
        task_lm: str = "openai/gpt-5",
        budget: EvalBudgetEnforcer | None = None,
        cache_strategy: str = "scoped",  # "scoped" | "disabled"
    ) -> None:
        if surface not in SURFACES:
            raise ValueError(
                f"unknown surface {surface!r}; choose from {sorted(SURFACES)}"
            )
        self.surface = surface
        self.opt_run_dir = Path(opt_run_dir)
        self.task_lm = task_lm
        self.budget = budget or EvalBudgetEnforcer.from_env()
        if cache_strategy not in ("scoped", "disabled"):
            raise ValueError(f"cache_strategy must be 'scoped' or 'disabled'")
        self.cache_strategy = cache_strategy

    # ------------------------------------------------------------------
    # GEPA contract
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Run one candidate against ``batch`` papers; return ``EvaluationBatch``.

        ``batch`` is a list of dicts with at least ``{"paper_id": str,
        "archetype": str}``. Trainset/valset loaders produce this shape.
        """
        self._validate_candidate(candidate)
        chash = candidate_hash(candidate)
        candidate_dir = self.opt_run_dir / "candidates" / chash
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "overrides.json").write_text(
            json.dumps(candidate, indent=2), encoding="utf-8"
        )

        scores: list[float] = []
        outputs: list[dict] = []
        trajectories: list[dict] = []

        for row in batch:
            paper_id = row["paper_id"]
            archetype = row.get("archetype", "other")
            eval_dir = candidate_dir / "evals" / paper_id
            eval_dir.mkdir(parents=True, exist_ok=True)
            result = self._run_one_eval(
                candidate=candidate,
                candidate_hash=chash,
                req=EvalRequest(paper_id=paper_id, archetype=archetype, run_dir=eval_dir),
            )
            scores.append(result.score)
            outputs.append(result.output)
            if capture_traces:
                trajectories.append(result.record)
            self._append_jsonl(self.opt_run_dir / "scores.jsonl", {
                "candidate_hash": chash,
                "paper_id": paper_id,
                "score": result.score,
            })

        return _make_evaluation_batch(scores=scores, outputs=outputs, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Build per-component reflective examples from a prior ``evaluate`` call.

        ``eval_batch.trajectories`` already contains §4.4-shape records (built
        by ``trace_minimizer.build_record`` in ``_run_one_eval``). Group them
        by component.
        """
        trajectories: list[dict] = list(getattr(eval_batch, "trajectories", None) or [])
        out: dict[str, list[dict]] = {cid: [] for cid in components_to_update}
        for rec in trajectories:
            cid = rec.get("component_id")
            if cid in out:
                out[cid].append(rec)
        # Persist for human inspection (G6 audit trail).
        for rec in trajectories:
            self._append_jsonl(self.opt_run_dir / "reflective_dataset.jsonl", rec)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_candidate(self, candidate: dict[str, str]) -> None:
        allowed = set(SURFACES[self.surface])
        for cid, text in candidate.items():
            if cid not in allowed:
                raise ValueError(
                    f"candidate proposes mutation to {cid!r} which is not in "
                    f"surface {self.surface!r} (allowed: {sorted(allowed)})"
                )
            if cid not in REGIONS:
                raise ValueError(f"unknown component_id {cid!r}")
            budget = REGIONS[cid].char_budget
            if len(text) > budget:
                raise ValueError(
                    f"candidate text for {cid!r} is {len(text)} chars; "
                    f"exceeds budget {budget}"
                )

    def _run_one_eval(
        self, *, candidate: dict[str, str], candidate_hash: str, req: EvalRequest
    ) -> EvalResult:
        env = os.environ.copy()
        env["REPROLAB_GEPA_CANDIDATE_JSON"] = json.dumps(candidate)
        env["REPROLAB_GEPA_PAPER_ID"] = req.paper_id
        env["REPROLAB_GEPA_RUN_DIR"] = str(req.run_dir)
        env["REPROLAB_GEPA_TASK_LM"] = self.task_lm
        if self.cache_strategy == "disabled":
            env["REPROLAB_PRIMITIVE_CACHE"] = "disabled"
        else:
            # Scoped: namespace by (candidate, surface) so two candidates
            # evaluating the same paper don't share each other's cache.
            env["REPROLAB_PRIMITIVE_CACHE_SURFACE_SALT"] = (
                f"{candidate_hash}:{self.surface}"
            )

        start = time.time()
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.agents.optimization.eval_entrypoint",
                ],
                env=env,
                check=False,
                timeout=self.budget.max_wall_clock_seconds,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logger.warning(
                "gepa_eval[%s/%s]: timeout after %.0fs", candidate_hash, req.paper_id, elapsed
            )
            return EvalResult(
                score=0.0,
                record=self.budget.timeout_record(
                    component_id=next(iter(candidate.keys()), "unknown"),
                    example_id=req.paper_id,
                    elapsed_s=elapsed,
                ),
            )

        # Read each component's §4.4 record; one paper produces ONE record
        # per component in the candidate. Adapter aggregates them and uses
        # the mean score as the per-paper score for the Pareto axis.
        records: list[dict] = []
        for cid in candidate.keys():
            rec = build_record(
                component_id=cid,
                example_id=req.paper_id,
                paper_archetype=req.archetype,
                run_dir=req.run_dir,
            )
            records.append(rec)
        # Pareto axis = per-paper Hermes-clamped score (max over components,
        # since any mutation that yields a good run "wins" the paper).
        score = max((r["score"] for r in records), default=0.0)
        # Aggregate record for the trajectory channel — pick the component
        # whose mutation produced the score.
        primary = max(records, key=lambda r: r["score"], default={"component_id": "unknown"})
        return EvalResult(score=score, record=primary, output={"records": records})

    def _append_jsonl(self, path: Path, obj: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")


def _make_evaluation_batch(
    *, scores: list[float], outputs: list[dict], trajectories: list[dict]
) -> Any:
    """Construct ``gepa.core.adapter.EvaluationBatch`` if available, else a stub."""
    try:
        from gepa.core.adapter import EvaluationBatch  # type: ignore[import-not-found]

        return EvaluationBatch(scores=scores, outputs=outputs, trajectories=trajectories)
    except ImportError:
        @dataclass
        class _StubBatch:
            scores: list[float]
            outputs: list[dict]
            trajectories: list[dict]

        return _StubBatch(scores=scores, outputs=outputs, trajectories=trajectories)


__all__ = [
    "EvalRequest",
    "EvalResult",
    "ReproLabGEPAAdapter",
    "SURFACES",
    "candidate_hash",
]
