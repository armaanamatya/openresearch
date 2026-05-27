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
        """Build per-component reflective examples in gepa's canonical shape.

        Gepa's reflection LM expects each example as exactly three keys:
        ``{"Inputs", "Generated Outputs", "Feedback"}`` — see
        ``gepa.adapters.default_adapter.DefaultAdapter.make_reflective_dataset``.
        Returning our raw §4.4 record would feed the reflection LM an
        unstructured trace dump and produce nonsense mutations.

        Mapping from our §4.4 record to gepa's canonical 3-key shape:
        - ``Inputs`` ← rubric-before snapshot + weak-leaves the candidate
          was meant to address (what was the situation).
        - ``Generated Outputs`` ← the candidate prompt text that ran +
          metrics digest (what did the candidate do).
        - ``Feedback`` ← rubric delta, Hermes status, repair counts,
          forced-iteration warnings (what to learn from).

        Gepa updates ONE component per reflection cycle even when the
        candidate has multiple components (its DefaultAdapter asserts
        ``len(components_to_update) == 1``). We don't assert (defensive)
        but the grouping naturally yields the same effect.
        """
        trajectories: list[dict] = list(getattr(eval_batch, "trajectories", None) or [])
        out: dict[str, list[dict]] = {cid: [] for cid in components_to_update}

        for rec in trajectories:
            cid = rec.get("component_id")
            if cid not in out:
                continue
            out[cid].append(_to_gepa_reflective_record(rec, candidate.get(cid, "")))

        # Persist the raw §4.4 records for human inspection (G6 audit trail).
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


def _to_gepa_reflective_record(rec: dict, candidate_text: str) -> dict:
    """Map one §4.4-shape record into gepa's canonical 3-key reflective shape.

    The reflection LM consumes the dict literally — keys must match
    ``{"Inputs", "Generated Outputs", "Feedback"}`` exactly. ``Inputs`` and
    ``Generated Outputs`` are structured dicts (canonical pattern from
    gepa adapter docs); ``Feedback`` is a natural-language instruction the
    LM treats as the actionable signal.

    Why dict-shape (Phase 2 hardening): the canonical gepa adapter pattern
    (`gepa.adapters.default_adapter.DefaultAdapter`) returns dicts so the
    reflection LM sees field names directly. String-flattening worked but
    forced the LM to parse our prose; dicts let it operate on labelled
    structure.
    """
    inp = rec.get("input") or {}
    trace = rec.get("execution_trace") or {}
    hermes = trace.get("hermes") or {}

    candidate_truncated = candidate_text[:1500]
    weak_leaves = (inp.get("weak_leaves_before") or [])[:5]
    repair_summaries = (trace.get("repair_summaries") or [])[:5]

    inputs = {
        "paper_archetype": rec.get("paper_archetype", "unknown"),
        "rubric_overall_before": float(inp.get("rubric_overall_before", 0.0)),
        "rubric_areas_before": inp.get("rubric_areas_before", {}),
        "weak_leaves_before": weak_leaves,
        "prior_results_digest": str(inp.get("current_results_digest", ""))[:400],
    }
    generated_outputs = {
        "candidate_prompt_text": candidate_truncated,
        "candidate_truncated": len(candidate_text) > 1500,
        "candidate_metrics": rec.get("candidate_output", {}),
        "rubric_overall_after": float(trace.get("rubric_overall_after", 0.0)),
        "rubric_delta_areas": trace.get("rubric_delta_areas", {}),
        "run_experiment_success": bool(trace.get("run_experiment_success", False)),
    }
    feedback = (
        f"Hermes-clamped score: {float(rec.get('score', 0.0)):.3f}\n"
        f"Repair attempts: {len(repair_summaries)} ({'; '.join(repair_summaries)[:400]})\n"
        f"Forced-iteration warnings: {trace.get('forced_iteration_warnings', 0)}\n"
        f"Blanket declines: {trace.get('blanket_decline_count', 0)}\n"
        f"Hermes status: {hermes.get('status', 'unavailable')}\n"
        f"Unsupported claims: {json.dumps(hermes.get('unsupported_claims', []), default=str)[:400]}\n"
        "Improve the prompt so that weak leaves move toward pass, Hermes status "
        "stays grounded, repair count drops, and the rubric overall climbs."
    )

    return {
        "Inputs": inputs,
        "Generated Outputs": generated_outputs,
        "Feedback": feedback,
    }


def _make_evaluation_batch(
    *, scores: list[float], outputs: list[dict], trajectories: list[dict]
) -> Any:
    """Construct ``gepa.core.adapter.EvaluationBatch`` if available, else a stub.

    Real gepa.EvaluationBatch has 4 fields: ``outputs``, ``scores``,
    ``trajectories``, ``objective_scores``. We don't decompose the score into
    multiple objectives (Pareto axis is per-paper, not per-objective per spec
    §7), so ``objective_scores`` mirrors ``scores`` as a single-objective view.
    """
    try:
        from gepa.core.adapter import EvaluationBatch  # type: ignore[import-not-found]

        return EvaluationBatch(
            scores=scores,
            outputs=outputs,
            trajectories=trajectories,
            objective_scores=None,
        )
    except ImportError:
        @dataclass
        class _StubBatch:
            scores: list[float]
            outputs: list[dict]
            trajectories: list[dict]
            objective_scores: list[float] | None = None

        return _StubBatch(
            scores=scores,
            outputs=outputs,
            trajectories=trajectories,
            objective_scores=None,
        )


__all__ = [
    "EvalRequest",
    "EvalResult",
    "ReproLabGEPAAdapter",
    "SURFACES",
    "candidate_hash",
]
