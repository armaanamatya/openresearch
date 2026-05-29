"""Primitive function registry exposed to the RLM REPL.

Each primitive wraps an existing stage agent's core function (see
`docs/rlm-pivot-mapping.md` §1) and:
  - Emits a `primitive_call` SSE event for the live-iteration UI
  - Updates `cost_ledger.jsonl`
  - Returns a structured dict the root can store in REPL variables

**Algorithm-2 guard (brief §7.7, mapping doc §1 invariant):**
No primitive signature accepts `paper_text` / `supplementary_text` /
`repo_files` as a whole-corpus argument. Primitives take slices and
structured specs only. The root assembles slices with REPL code and
`llm_query`/`rlm_query` against constructed slices.

Phase 2 (#59) implementation.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from backend.agents.rlm.context import RunContext
    from backend.agents.schemas import GpuPlan

logger = logging.getLogger(__name__)

# Module-level alias so tests can monkeypatch RuntimeAppService without
# requiring a live Docker daemon.
from backend.services.runtime.service import RuntimeAppService


class PrimitiveOutcome(str, Enum):
    ok = "ok"
    partial_evidence = "partial_evidence"
    repairable = "repairable"
    retryable = "retryable"
    fatal = "fatal"


_RUN_EXPERIMENT_REPAIRABLE_FAILURES = {
    "code_bug",
    "contract_violation",
    # Existing classifier labels that require agent-side repair.
    "missing_module",
    "torch_redundancy",
    "cuda_oom",
    "oom_killed",
    "requirements_not_found",
    "missing_dataset",
    "exec_timeout",
    "watchdog_killed",
    "preflight_blocked",
    "permission_denied",
    "syntax_error",
    "scope_shape_violation",
    "unknown",
}
_RUN_EXPERIMENT_RETRYABLE_FAILURES = {
    "transient",
    "ssh_drop",
    "pod_unavailable",
    # Existing classifier labels for backend/network transients.
    "network_flake",
    "runpod_capacity",
    "runpod_transient_500",
    "runpod_ssh_timeout",
}
_RUN_EXPERIMENT_FATAL_FAILURES = {
    "balance_too_low",
    "auth_failed",
    "quota_exceeded",
    # Existing classifier label for the same fatal funding state.
    "runpod_balance_too_low",
}

# PR-ζ: transient-error retry policy for _execute_in_sandbox.
# Three retries with exponential backoff: 5s, 10s, 20s.
# Total retry budget is capped so it cannot blow through the primitive
# wall-clock limit (the surrounding run_experiment timeout still bounds).
_MAX_TRANSIENT_RETRIES: int = 3
_BACKOFF_BASE_S: float = 5.0
_RETRY_TIMEOUT_TOTAL_S: float = 90.0

_DEFAULT_PRE_EMIT_STALL_S = 240.0


class PreEmitStallError(RuntimeError):
    """Repairable implement_baseline pre-emission stall marker for PR-π."""


def _with_outcome(result: dict, outcome: PrimitiveOutcome) -> dict:
    """Attach primitive typestate to a result dict without disturbing payload shape."""
    result.setdefault("outcome", outcome.value)
    return result


def _baseline_ok_envelope(code_dir: "Any") -> dict:
    """Return the normalized successful implement_baseline envelope."""
    from pathlib import Path

    path = Path(code_dir)
    files = sorted(
        str(p.relative_to(path)).replace("\\", "/")
        for p in path.rglob("*")
        if p.is_file()
    ) if path.exists() else []
    return _with_outcome({
        "ok": True,
        "code_path": str(path),
        "files": files,
    }, PrimitiveOutcome.ok)


def _baseline_error_envelope(
    *,
    error_code: str,
    error: str,
    repairable: bool = True,
    code_dir: "Any | None" = None,
    missing_files: list[str] | None = None,
) -> dict:
    """Return the normalized failed implement_baseline envelope."""
    payload = {
        "ok": False,
        "success": False,
        "error_code": error_code,
        "error": error,
        "repairable": repairable,
    }
    if code_dir is not None:
        payload["code_path"] = str(code_dir)
    if missing_files is not None:
        payload["missing_files"] = missing_files
    return _with_outcome(
        payload,
        PrimitiveOutcome.repairable if repairable else PrimitiveOutcome.fatal,
    )


def _harvest_baseline_artifacts(
    code_dir: "Any",
    *,
    error_code: str = "incomplete_artifacts",
    error_prefix: str = "implement_baseline artifacts incomplete",
) -> dict:
    """Validate and harvest baseline artifacts written by the SDK subprocess.

    A usable baseline must at minimum expose a non-empty ``commands.json`` and
    at least one runnable source/script artifact. This is deliberately small:
    it accepts partial but executable projects after SDK cleanup failures while
    refusing directories that would only cascade into run_experiment failures.
    """
    import json
    from pathlib import Path

    path = Path(code_dir)
    missing: list[str] = []
    commands_path = path / "commands.json"
    if not commands_path.exists():
        missing.append("commands.json")
    else:
        try:
            commands = json.loads(commands_path.read_text(encoding="utf-8"))
            if not isinstance(commands, list) or not commands:
                missing.append("commands.json: non-empty list")
        except (json.JSONDecodeError, OSError):
            missing.append("commands.json: valid JSON list")

    runnable_suffixes = {".py", ".sh", ".bash", ".ps1"}
    has_runnable = False
    if path.exists():
        for file in path.rglob("*"):
            if not file.is_file() or file.name == "commands.json":
                continue
            if file.suffix.lower() in runnable_suffixes or file.name in {"Dockerfile", "Makefile"}:
                has_runnable = True
                break
    if not has_runnable:
        missing.append("runnable source file")

    if missing:
        return _baseline_error_envelope(
            error_code=error_code,
            error=f"{error_prefix}: missing {', '.join(missing)}",
            repairable=True,
            code_dir=path,
            missing_files=missing,
        )
    return _baseline_ok_envelope(path)


def _failure_class_key(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _classify_run_experiment_outcome(result: dict) -> PrimitiveOutcome:
    """Map a run_experiment result dict to its primitive typestate."""
    if result.get("success") is True:
        return PrimitiveOutcome.ok

    metrics = result.get("metrics")
    if isinstance(metrics, dict) and bool(metrics):
        return PrimitiveOutcome.partial_evidence

    failure_class = _failure_class_key(result.get("failure_class"))
    if not failure_class:
        return PrimitiveOutcome.repairable
    if failure_class in _RUN_EXPERIMENT_REPAIRABLE_FAILURES:
        return PrimitiveOutcome.repairable
    if failure_class in _RUN_EXPERIMENT_RETRYABLE_FAILURES:
        return PrimitiveOutcome.retryable
    if failure_class in _RUN_EXPERIMENT_FATAL_FAILURES:
        return PrimitiveOutcome.fatal
    return PrimitiveOutcome.repairable


def _timeout_for(ctx: "RunContext", cap_s: float) -> float:
    """Return the tightest timeout (seconds) for a primitive given the run deadline.

    Takes the lesser of `cap_s` (the primitive's own hard cap) and
    `ctx.remaining_s()` (time left in the overall wall-clock budget).  Always
    returns a positive float — callers can pass it directly to
    `concurrent.futures.Future.result(timeout=...)`.
    """
    remaining = ctx.remaining_s()
    if remaining is None:
        return cap_s
    # clamp to at least 1 s so we don't hand a zero/negative timeout to .result()
    return max(1.0, min(cap_s, remaining))


def _pre_emit_stall_s() -> float:
    """Resolve PR-π pre-emit stall threshold from env.

    Pre: ``REPROLAB_PRE_EMIT_STALL_S`` may be unset or a positive number.
    Post: returns a positive second threshold, defaulting to 120s.
    Side effects: logs a warning for invalid environment values.
    Exceptions raised: none.
    """
    raw = os.environ.get("REPROLAB_PRE_EMIT_STALL_S", "").strip()
    if not raw:
        return _DEFAULT_PRE_EMIT_STALL_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid REPROLAB_PRE_EMIT_STALL_S=%r; using default", raw)
        return _DEFAULT_PRE_EMIT_STALL_S
    return value if value > 0 else _DEFAULT_PRE_EMIT_STALL_S


EXPERIMENT_TIMEOUT_BY_MODE: dict[str, int] = {
    "efficient": 7200,   # 2h per call
    "max": 21600,        # 6h per call
}
_DEFAULT_EXPERIMENT_TIMEOUT_S: int = 7200  # fallback when execution_mode unknown


def resolve_experiment_timeout_s(ctx) -> int:
    """Resolve the wall-clock cap for a single run_experiment call.

    Order:
      1. REPROLAB_RUN_EXPERIMENT_TIMEOUT_S env var (if set and > 0)
      2. EXPERIMENT_TIMEOUT_BY_MODE[ctx.execution_mode]
      3. _DEFAULT_EXPERIMENT_TIMEOUT_S

    Then clamp to ctx.remaining_s() only when finite — infinite remaining
    means no --max-wall-clock was set; honor the mode default unchanged.
    """
    import math as _math
    import os as _os

    _env = _os.environ.get("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S", "").strip()
    if _env:
        try:
            override = int(_env)
            if override > 0:
                resolved = override
            else:
                resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(
                    getattr(ctx, "execution_mode", None)
                    or _os.environ.get("REPROLAB_EXECUTION_MODE"),
                    _DEFAULT_EXPERIMENT_TIMEOUT_S,
                )
        except ValueError:
            resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(
                getattr(ctx, "execution_mode", None)
                or _os.environ.get("REPROLAB_EXECUTION_MODE"),
                _DEFAULT_EXPERIMENT_TIMEOUT_S,
            )
    else:
        resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(
            getattr(ctx, "execution_mode", None)
            or _os.environ.get("REPROLAB_EXECUTION_MODE"),
            _DEFAULT_EXPERIMENT_TIMEOUT_S,
        )

    try:
        remaining = ctx.remaining_s()
    except Exception:
        remaining = _math.inf
    if remaining is None:
        remaining = _math.inf
    if _math.isfinite(remaining):
        # Clamp to remaining budget; floor at 1 s so .result(timeout=0) is not
        # passed accidentally (matches the _timeout_for convention).
        resolved = max(1, min(resolved, int(remaining)))
    return resolved


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response.

    Robust to prose and ``` fences around the JSON: scans forward from each
    `{` and uses `json.JSONDecoder.raw_decode`, which correctly ignores braces
    inside strings and any trailing text — unlike a naive first-`{`/last-`}`
    span, which over-grabs when the response contains prose braces.

    If a `{` opens a structure that runs off the end of `text` (a truncated
    response), this raises rather than falling through to a later *inner*
    `{...}` fragment — a silent wrong parse is worse than a clear failure.
    """
    import json
    decoder = json.JSONDecoder()
    end_of_text = len(text.rstrip())
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            # A decode error at end-of-text means a structure opened at `idx`
            # and was never closed: the response is truncated. Returning a
            # later inner fragment would be a silent wrong parse — raise.
            if exc.pos >= end_of_text:
                raise ValueError("truncated JSON object in LLM response") from exc
        idx = text.find("{", idx + 1)
    raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")


def _extract_json_array(text: str) -> list:
    """Pull the first JSON array out of an LLM response.

    Mirrors _extract_json but scans for `[` instead of `{`. Same EOF-truncation
    guard. Used by leaf-scorer's batch-response parser (review M3 / T26).
    """
    import json
    decoder = json.JSONDecoder()
    end_of_text = len(text.rstrip())
    idx = text.find("[")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError as exc:
            if exc.pos >= end_of_text:
                raise ValueError("truncated JSON array in LLM response") from exc
        idx = text.find("[", idx + 1)
    raise ValueError(f"no JSON array in LLM response: {text[:200]!r}")


def _clamp01(val: object) -> float:
    """Coerce an LLM-returned value into [0.0, 1.0]; None / garbage -> 0.0."""
    try:
        return max(0.0, min(1.0, float(val)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# run_experiment's container stdout is unbounded — a training run can emit
# megabytes. verify_against_rubric / propose_improvements serialize the whole
# result (logs included) into an LLM prompt, so an uncapped log re-creates the
# context-window blow-up the RLM design exists to avoid. Cap to a head+tail
# window before the logs leave run_experiment.
_MAX_LOG_CHARS = 16000

# Contract: the experiment writes its measured numeric results as a flat JSON
# object (metric name → number) to this file in the code root (or in an
# "outputs/" sub-directory). `_execute_in_sandbox` reads it back so
# run_experiment can return real metrics instead of an empty dict.
METRICS_FILENAME = "metrics.json"

# Per-command wall-clock cap for run_experiment's sandbox execution. Container
# resource bounds (network/memory/cpu) already come from SandboxConfig
# defaults; this is the time bound. Phase 3 (#60) should source it from
# settings / ctx rather than this module constant.
_EXEC_TIMEOUT_SECONDS = 14400  # 4 hr — generous per-command cap so long-running
                                # train.py runs (multi-experiment papers on CPU,
                                # or large epoch counts on a single GPU) aren't
                                # cut off by the per-command guard.  The outer
                                # ctx.remaining_s() wall-clock still binds the
                                # whole run; this is just the inner ceiling.
                                # Override via REPROLAB_RUN_EXPERIMENT_TIMEOUT_S
                                # in run_experiment if a single run truly needs
                                # a tighter bound.

# CUDA OOM detection: markers observed in PyTorch / cuBLAS logs (spec 2026-05-23 §OOM).
# Pattern set is intentionally tight to avoid false positives on unrelated CUDA errors.
_CUDA_OOM_MARKERS: tuple[str, ...] = (
    "CUDA out of memory",
    "RuntimeError: CUDA error: out of memory",
    "torch.cuda.OutOfMemoryError",
    "cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
)


def _detect_cuda_oom(*, exit_code: int, stderr_tail: str) -> bool:
    """True when exit-code or stderr tail indicates a CUDA OOM (spec 2026-05-23 §OOM).

    `stderr_tail` should be the last ~4KB of combined stderr/stdout from the failed
    experiment. Exit codes 137 and -9 are SIGKILL/OOM-killer shapes; substring
    match covers the documented PyTorch/cuBLAS variants. Pattern set is
    intentionally tight to avoid false positives on unrelated CUDA errors.
    """
    if exit_code in {-9, 137}:
        return True
    if not stderr_tail:
        return False
    return any(marker in stderr_tail for marker in _CUDA_OOM_MARKERS)


def _cap_logs(text: str) -> str:
    """Bound an unbounded log string to a head+tail window for LLM consumption."""
    if len(text) <= _MAX_LOG_CHARS:
        return text
    half = _MAX_LOG_CHARS // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... [{omitted} chars truncated] ...\n{text[-half:]}"


_PLAN_REPRODUCTION_SYSTEM = (
    "You are the Reproduction Planner for ReproLab. Given a paper's method "
    "spec and a target environment spec, produce a ReproductionContract: what "
    "counts as a faithful reproduction, a smoke-test plan, a full-run plan, "
    "the expected output artifacts, a dataset plan, an evaluation plan, and a "
    "verification checklist. Return exactly ONE JSON object with those fields "
    "and nothing else. Do NOT write files; do NOT reference any filesystem path."
)

# θ: metrics_shape declaration — appended to the planning prompt so the agent
# commits to the exact dotted paths it will emit in metrics.json.
# This eliminates the nested-vs-flat ambiguity that caused 16 contract
# violations in the Adam run (2026-05-25). The agent may choose any json_path
# shape (flat or nested) — it just must commit here and stick to it in train.py.
_METRICS_SHAPE_INSTRUCTION = """

You MUST also declare metrics_shape — the exact dotted paths your train.py
will write into metrics.json, one per rubric result_match leaf. The grader
will check metrics.json for EXACTLY these paths; emitting different ones
counts as a contract violation.

You may choose ANY json_path shape (flat like "mnist_logistic_adam_final_nll"
OR nested like "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll") —
but commit to it here and stick to it in train.py. The grader honors your
declared path.

Include "metrics_shape" as a JSON key in your response:
"metrics_shape": [
  {"metric_id": "<stable_id_for_rubric_leaf>",
   "json_path": "<dotted.path.inside.metrics.json>",
   "rubric_leaf_ids": []},
  ...
]

One entry per metric the paper's rubric evaluates. When the paper has no
numeric result_match leaves (e.g. a methods-only paper), emit an empty list.
"""

# β3: compute-adjusted rubric — appended to the planning prompt when clipping is active.
_COMPUTE_SCOPE_INSTRUCTION = """

You are planning under a deliberately CLIPPED compute budget. In addition to
the standard ReproductionContract fields, include a "compute_scope" key in
your JSON response so the grader can score against an achievable floor instead
of the paper's headline target.

Fill compute_scope.metric_floors with ONE entry per result_match leaf the
grader will evaluate (one per metric the paper claims). For each entry:

  - metric: the metric name matching the rubric leaf (e.g. "mnist_test_loss").
  - direction: "higher" for accuracy/F1/reward-style, "lower" for loss/error/cost.
  - paper_target: the paper's reported headline value for this metric.
  - floor: the value plausibly reachable given the actual compute budget.
    Use convergence-trajectory reasoning:
    Adam-style: exponential approach → 1/N budget reaches ~30% of final gain.
    SGD+cosine: closer to logarithmic → 1/N budget reaches ~50%.
    GAN losses: non-monotonic — set a permissive floor; be honest.
  - rationale: one short sentence explaining the floor choice.

CONSTRAINT: floor MUST be WORSE than paper_target.
  direction="higher": floor <= paper_target (floor can be equal but not exceed).
  direction="lower": floor >= paper_target (floor can be equal but not below).

compute_scope JSON shape:
  "compute_scope": {
    "is_clipped": true,
    "paper_epochs": <int or null>,
    "actual_epochs": <int or null>,
    "rationale": "<one sentence>",
    "metric_floors": [
      {"metric": "...", "direction": "higher"|"lower",
       "paper_target": <float>, "floor": <float>, "rationale": "..."}
    ]
  }
"""


def _is_clipping_active(ctx: "RunContext") -> bool:
    """True iff the planning agent should emit ComputeScope.

    Triggered by either: execution_profile.mode == "efficient", OR
    minimize_compute=True. Either is an opt-in compute budget cap.
    """
    minimize = bool(getattr(ctx, "minimize_compute", False))
    profile = getattr(ctx, "execution_profile", None)
    mode = ""
    if profile is not None:
        m = getattr(profile, "mode", None)
        mode = str(getattr(m, "value", m) or "").lower()
    return minimize or mode == "efficient"


_HINT_THRESHOLD = 10_000  # chars


def understand_section(text_slice: str, *, ctx: "RunContext") -> dict:
    """Extract datasets/metrics/training-recipe/hardware/ambiguities from a slice.

    Wraps the *title-agnostic* heuristic helpers in
    `backend/agents/paper_understanding.py`. Returns a PARTIAL PaperClaimMap
    dict — `core_contribution`, `claims`, `model_architecture` and
    `evaluation_protocol` need section titles and are left for the root model
    to extract with `llm_query` over `context` (design decision D5).

    `ctx` is required by the primitive-wrapper protocol (design decision D4 —
    `build_custom_tools` closes `ctx` over every primitive uniformly); this
    heuristic body does not use it.

    Cached via ``primitive_cache``: identical text_slice → cached result
    (skips the heuristic re-parse on retry).
    """
    from backend.agents.rlm import primitive_cache as _cache
    _payload = {"text_slice": text_slice}
    _cached = _cache.maybe_get(ctx.project_dir, "understand_section", payload=_payload)
    if _cached is not None:
        return _with_outcome(_cached, PrimitiveOutcome.ok)

    from backend.agents.paper_understanding import (
        _extract_datasets, _extract_metrics, _extract_training_recipe,
        _extract_hardware, _extract_ambiguities,
    )
    sections = {"_": text_slice}
    result = {
        "datasets": [d.model_dump() for d in _extract_datasets(sections)],
        "metrics": [m.model_dump() for m in _extract_metrics(sections)],
        "training_recipe": _extract_training_recipe(sections).model_dump(),
        "hardware_clues": _extract_hardware(sections),
        "ambiguities": [a.model_dump() for a in _extract_ambiguities(sections)],
    }
    if len(text_slice) > _HINT_THRESHOLD:
        result["_meta"] = {
            "hint": (
                "This slice is "
                f"{len(text_slice):,} chars — for tighter extraction "
                "consider `rlm_query(slice, specific_question)` "
                "instead. A focused sub-RLM call typically returns a "
                "more precise answer than this primitive's generic "
                "schema."
            ),
            "slice_chars": len(text_slice),
            "threshold": _HINT_THRESHOLD,
        }
    result = _with_outcome(result, PrimitiveOutcome.ok)
    _cache.put(ctx.project_dir, "understand_section", payload=_payload, result=result)
    return result


def extract_hyperparameters(text_slice: str, *, ctx: "RunContext") -> dict:
    """Extract hyperparameters from a slice (typically the training-recipe section).

    Wraps `paper_understanding._extract_training_recipe`. Returns a flat dict:
    optimizer, learning_rate, batch_size, epochs_or_steps, scheduler,
    other_hparams. The heuristic populates the first four; the root model can
    fill scheduler/other_hparams via `llm_query` if needed.

    `ctx` is required by the primitive-wrapper protocol (design decision D4);
    this heuristic body does not use it.

    Cached via ``primitive_cache``.
    """
    from backend.agents.rlm import primitive_cache as _cache
    _payload = {"text_slice": text_slice}
    _cached = _cache.maybe_get(ctx.project_dir, "extract_hyperparameters", payload=_payload)
    if _cached is not None:
        return _with_outcome(_cached, PrimitiveOutcome.ok)

    from backend.agents.paper_understanding import _extract_training_recipe
    result = _extract_training_recipe({"_": text_slice}).model_dump()
    if len(text_slice) > _HINT_THRESHOLD:
        result["_meta"] = {
            "hint": (
                "This slice is "
                f"{len(text_slice):,} chars — for tighter extraction "
                "consider `rlm_query(slice, specific_question)` "
                "instead. A focused sub-RLM call typically returns a "
                "more precise answer than this primitive's generic "
                "schema."
            ),
            "slice_chars": len(text_slice),
            "threshold": _HINT_THRESHOLD,
        }
    result = _with_outcome(result, PrimitiveOutcome.ok)
    _cache.put(ctx.project_dir, "extract_hyperparameters", payload=_payload, result=result)
    return result


def detect_environment(method_spec: dict, *, ctx: "RunContext") -> dict:
    """Infer the runtime environment; return an EnvironmentSpec dict.

    Wraps `environment_detective.run_offline` — the deterministic, no-LLM entry
    point — directly (brief §4 "wrap, not rewrite"). Verified: `run_offline` is
    exactly the heuristic helper chain plus a Dockerfile write into the run
    dir; that file-write side effect is fine — a primitive may write run
    artifacts via `ctx`. (`build_environment` consumes the Dockerfile from the
    returned EnvironmentSpec dict's `dockerfile` field, not that file.)
    `method_spec` is a (possibly partial) PaperClaimMap dict;
    `PaperClaimMap.core_contribution` is its one *required* field, so it is
    defaulted here — `understand_section`'s output omits it.

    Fail-soft (A2-L1): if the REPL passes a non-dict `method_spec` (e.g. a
    string or None), return an error dict rather than raising.
    """
    if not isinstance(method_spec, dict):
        return _with_outcome({
            "success": False,
            "error": (
                f"detect_environment: method_spec must be a dict, "
                f"got {type(method_spec).__name__!r}"
            ),
        }, PrimitiveOutcome.repairable)

    from backend.agents.rlm import primitive_cache as _cache
    _sandbox_mode_val = getattr(ctx, "sandbox_mode", None)
    _sandbox_key = getattr(_sandbox_mode_val, "value", str(_sandbox_mode_val) if _sandbox_mode_val is not None else None)
    _payload = {
        "method_spec": method_spec,
        # gpu_mode + sandbox_mode both affect Dockerfile shape, so both are cache keys.
        "gpu_mode": getattr(ctx, "gpu_mode", None),
        "sandbox_mode": _sandbox_key,
    }
    _cached = _cache.maybe_get(ctx.project_dir, "detect_environment", payload=_payload)
    if _cached is not None:
        return _with_outcome(_cached, PrimitiveOutcome.ok)

    from backend.agents.environment_detective import run_offline
    from backend.agents.schemas import PaperClaimMap

    claim_map = PaperClaimMap(**{"core_contribution": "", **method_spec})
    # Thread gpu_mode + sandbox_mode so the Dockerfile uses the right base image
    # and wheel source. sandbox_mode="runpod" triggers the pre-built pytorch base,
    # avoiding the ~2.5 GB CUDA wheel download that causes 1800s build timeouts.
    spec = run_offline(
        ctx.project_id, ctx.runs_root, claim_map, method_spec.get("artifact_index"),
        gpu_mode=getattr(ctx, "gpu_mode", None),
        sandbox_mode=_sandbox_key,
    )
    result = _with_outcome(spec.model_dump(), PrimitiveOutcome.ok)
    _cache.put(ctx.project_dir, "detect_environment", payload=_payload, result=result)
    return result


def _emit_dashboard_event_to_path(project_dir, *, event_type: str, payload: dict) -> None:
    """Path-based emit — same JSONL contract as _emit_dashboard_event but
    addressable from places that don't have a RunContext (e.g. the watchdog
    async task that lives below the primitive layer).  Fail-soft."""
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path

    events_file = _Path(project_dir) / "dashboard_events.jsonl"
    line = {
        "ts": _dt.now(_tz.utc).isoformat(),
        "event": event_type,
        "data": payload,
    }
    try:
        with events_file.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(line, default=str) + "\n")
    except Exception:  # noqa: BLE001 — observability must never break the run
        logger.exception("dashboard event emit failed for %s", event_type)


def _emit_dashboard_event(ctx: "RunContext", *, event_type: str, payload: dict) -> None:
    """Append a JSON event line to runs/<id>/dashboard_events.jsonl.

    Fail-soft (D3): any IO error is logged but never propagates — observability
    must never interrupt a run.
    """
    _emit_dashboard_event_to_path(ctx.project_dir, event_type=event_type, payload=payload)


def _emit_iteration_boundary_warning(run_dir, outcome: str, brief: str) -> None:
    """Append an iteration_boundary_recommended run_warning to dashboard_events.jsonl.
    Only fires for repairable/partial_evidence; pure file I/O; fail-soft."""
    if outcome not in {"repairable", "partial_evidence"}:
        return
    try:
        from pathlib import Path as _Path
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        events_path = _Path(run_dir) / "dashboard_events.jsonl"
        event = {
            "event": "run_warning",
            "timestamp": _dt.now(_tz.utc).isoformat(),
            "code": "iteration_boundary_recommended",
            "message": (
                f"run_experiment returned {outcome}; end this iteration so the "
                f"failure surfaces as fresh next-turn context. ({brief})"
            ),
        }
        with open(events_path, "a") as f:
            f.write(_json.dumps(event) + "\n")
    except Exception:  # noqa: BLE001 — observability is best-effort
        pass


def resolve_gpu_requirements(
    requirements: "GpuRequirements | dict",
    *,
    ctx: "RunContext",
) -> dict:
    """Plan-time GPU resolver primitive (RLM #dynamic-gpu spec 2026-05-23).

    The RLM root supplies LLM-derived GpuRequirements (from accumulated
    PaperClaimMap.hardware_clues + reasoning over env_spec and the full workload).
    This primitive maps them to a GpuPlan via the catalog, caches the plan in run
    state for idempotency, and emits a ``gpu_resolved`` SSE event for UI / audit.

    Idempotent: subsequent calls in the same run return the cached plan even if
    the caller passes different requirements. This avoids cost drift across
    re-resolution attempts and matches RLM-loop expectations.

    Args:
        requirements: Either a typed ``GpuRequirements`` instance or a plain dict
            with the same keys — the REPL typically produces dicts, so both are
            accepted.

    Returns:
        A ``GpuPlan`` serialised as a plain dict (JSON-safe via ``model_dump``).

    Never raises — returns an error dict on validation failure so the REPL root
    can handle it gracefully.
    """
    import json as _json
    from pathlib import Path as _Path

    from backend.agents.schemas import GpuRequirements as _Req
    from backend.config import get_settings
    from backend.services.runtime import gpu_resolver

    # ---- Idempotency: return cached plan if present.
    state_dir = _Path(ctx.project_dir) / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_file = state_dir / "gpu_plan.json"
    if cache_file.exists():
        try:
            cached = _json.loads(cache_file.read_text(encoding="utf-8"))
            return _with_outcome(cached, PrimitiveOutcome.ok)
        except Exception:  # noqa: BLE001 — corrupt cache → recompute
            logger.warning("resolve_gpu_requirements: cache file unreadable, recomputing")

    # ---- Coerce payload.
    if isinstance(requirements, dict):
        req = _Req(**requirements)
    elif isinstance(requirements, _Req):
        req = requirements
    else:
        return _with_outcome({
            "success": False,
            "error": (
            f"resolve_gpu_requirements: requirements must be GpuRequirements or dict, "
            f"got {type(requirements).__name__}"
            ),
        }, PrimitiveOutcome.repairable)

    # ---- vram_override: per-run CLI override bypasses LLM estimate.
    vram_override = getattr(ctx, "vram_override", None)
    if vram_override is not None:
        req = req.model_copy(update={"estimated_vram_gb": int(vram_override)})

    settings = get_settings()
    cloud_types: tuple[str, ...] = (
        ("COMMUNITY", "SECURE")
        if getattr(settings, "runpod_cloud_type", "COMMUNITY") == "SECURE"
        else ("COMMUNITY",)
    )

    from backend.agents.schemas import GpuPlan as _GpuPlan
    plan: "_GpuPlan" = gpu_resolver.resolve(
        req,
        dynamic_gpu_enabled=settings.dynamic_gpu_enabled,
        force_single_gpu=settings.force_single_gpu,
        max_gpu_usd_per_hour=settings.max_gpu_usd_per_hour or None,
        headroom_multiplier=settings.dynamic_gpu_headroom,
        fallback_vram_gb=settings.dynamic_gpu_fallback_vram_gb,
        cloud_types=cloud_types,
    )

    # ---- Persist atomically.
    payload = _with_outcome(plan.model_dump(mode="json"), PrimitiveOutcome.ok)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload, default=str), encoding="utf-8")
    tmp.replace(cache_file)

    # ---- Emit SSE event: gpu_resolved for all plans; gpu_fallback additionally
    #      when the resolver fell back to the default SKU (no catalog match).
    _emit_dashboard_event(ctx, event_type="gpu_resolved", payload=payload)
    if plan.source == "fallback":
        _emit_dashboard_event(ctx, event_type="gpu_fallback", payload=payload)

    return payload


# Indirection so tests can monkeypatch the async Docker build.
def _build_image(dockerfile_path, context_dir, tag, **kw):
    from backend.services.runtime.local_docker import build_image
    return build_image(dockerfile_path, context_dir, tag, **kw)


def _image_exists(tag: str) -> bool:
    """Return True iff the Docker image `tag` already exists locally.

    Uses the Docker SDK's images.get() — raises ImageNotFound when the image
    is absent, any other exception (SDK unavailable, daemon unreachable) is
    treated conservatively as "not found" so the caller falls through to the
    normal build path.
    """
    try:
        import docker  # type: ignore[import-untyped]
        from docker.errors import ImageNotFound  # type: ignore[import-untyped]

        client = docker.from_env()
        try:
            client.images.get(tag)
            return True
        except ImageNotFound:
            return False
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception:  # noqa: BLE001 — SDK missing / daemon down: fall through to build
        return False


_ENV_REPAIR_SYSTEM = (
    "You are a Docker environment repair assistant. Given a Dockerfile and the "
    "build error it produced, output a corrected Dockerfile and NOTHING else — "
    "no prose, no code fences."
)


def build_environment(env_spec: dict, *, ctx: "RunContext") -> dict:
    """Build the Docker image for `env_spec`, repairing the Dockerfile on failure.

    Genuinely fail-soft (design decision D3): any failure — a spent attempt
    cap, an `llm_client` error, a `write_text` error, a bad import — returns
    `{"ok": False, "error": ..., "attempts": ...}`; the primitive never raises.
    The ONE exception is `SandboxRuntimeError` (Docker daemon down / SDK
    missing): an infrastructure failure, not a Dockerfile problem, so it
    propagates.

    Hardening (WS-H Batch P):
    - A2-C3: the ThreadPoolExecutor is now created ONCE, outside the repair
      loop, and each `.result()` uses a per-attempt timeout so the aggregate
      wall time is bounded.
    - A2-M1: the repair LLM call also uses a per-attempt timeout via the same
      pool (it's synchronous so we submit it and bound .result()).
    """
    # Local sandbox is docker-free: dependencies are resolved on the host
    # (per-run venv), so there is no image to build. Short-circuit BEFORE any
    # docker client is touched — otherwise build_environment raises
    # SandboxRuntimeError(backend_unavailable) on hosts without a daemon.
    _sb_mode = getattr(ctx, "sandbox_mode", None)
    _sb_key = getattr(_sb_mode, "value", str(_sb_mode) if _sb_mode is not None else None)
    if _sb_key == "local":
        return _with_outcome({
            "ok": True,
            "image_tag": "",
            "attempts": 0,
            "skipped": True,
            "note": "local sandbox: dependencies resolved on host venv; no image built",
        }, PrimitiveOutcome.ok)

    import asyncio
    import concurrent.futures
    import hashlib
    import tempfile
    import time
    from pathlib import Path

    # SandboxRuntimeError is named in an `except` clause below — bind it
    # before the try-block so a failed import *inside* the try cannot make
    # that clause raise NameError and escape (the D3 fail-soft hole).
    from backend.services.runtime.interface import SandboxRuntimeError

    dockerfile = str(env_spec.get("dockerfile") or "").strip()
    if not dockerfile:
        return _with_outcome({
            "ok": False,
            "image_tag": "",
            "error": "env_spec.dockerfile is empty",
            "attempts": 0,
        }, PrimitiveOutcome.repairable)

    attempts, ok, tag, error = 0, False, "", ""
    try:
        from backend.config import get_settings

        settings = get_settings()
        max_attempts = max(1, settings.environment_build_max_attempts)
        # Per-attempt budget: 1800 s build + 60 s LLM repair.
        per_attempt_s = getattr(settings, "environment_build_attempt_s", 1800)
        llm_repair_s = getattr(settings, "environment_build_llm_repair_s", 60)
        # Aggregate cap: total time across all repair attempts.
        aggregate_cap_s = _timeout_for(ctx, per_attempt_s * max_attempts)

        # A Docker tag is a mutable pointer, not an identifier: a fixed tag
        # lets two build_environment calls in one run collide, after which
        # run_experiment runs whichever image the tag last pointed at. Key the
        # tag to the Dockerfile so distinct environments get distinct images.
        digest = hashlib.sha1(dockerfile.encode("utf-8")).hexdigest()[:12]
        tag = f"reprolab/{ctx.project_id}:env-{digest}"

        # Three-layer "don't redo work" guard: content-addressed tag → Docker
        # layer cache → this existence check.  When the image is already
        # present (same Dockerfile hash → same tag → same bits), skip the
        # entire rebuild and return immediately.  Re-checked per call so a
        # manual `docker rmi` between iterations forces a real rebuild (D5).
        if _image_exists(tag):
            return _with_outcome({
                "ok": True,
                "image_tag": tag,
                "attempts": 0,
                "skipped": True,
            }, PrimitiveOutcome.ok)

        deadline_abs = time.monotonic() + aggregate_cap_s
        with tempfile.TemporaryDirectory() as tmp:
            context_dir = Path(tmp)
            dockerfile_path = context_dir / "Dockerfile"
            # A2-C3: single executor for all repair iterations.
            # I12: explicit shutdown(wait=False) so a wedged build cannot block cleanup.
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                while not ok and attempts < max_attempts:
                    remaining = deadline_abs - time.monotonic()
                    if remaining <= 0:
                        error = "build_environment: aggregate time cap exceeded"
                        break
                    attempts += 1
                    dockerfile_path.write_text(dockerfile, encoding="utf-8")
                    # Async bridge: asyncio.run in the worker thread; timeout
                    # bounded by both the per-attempt cap and aggregate remaining.
                    build_timeout = max(1.0, min(per_attempt_s, remaining))
                    try:
                        ok, tag, error = pool.submit(
                            asyncio.run,
                            _build_image(dockerfile_path, context_dir, tag),
                        ).result(timeout=build_timeout)
                    except concurrent.futures.TimeoutError:
                        error = (
                            f"build_environment: Docker build timed out "
                            f"after {build_timeout:.0f} s (attempt {attempts})"
                        )
                        break
                    if not ok and attempts < max_attempts:
                        llm_timeout = max(
                            1.0, min(llm_repair_s, deadline_abs - time.monotonic())
                        )
                        try:
                            # A2-M1: bound the synchronous repair LLM call.
                            dockerfile = pool.submit(
                                ctx.llm_client.complete,
                                system=_ENV_REPAIR_SYSTEM,
                                user=f"Dockerfile:\n{dockerfile}\n\nBuild error:\n{error}",
                            ).result(timeout=llm_timeout).strip()
                        except concurrent.futures.TimeoutError:
                            error = (
                                f"build_environment: LLM repair call timed out "
                                f"after {llm_timeout:.0f} s (attempt {attempts})"
                            )
                            break
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    except SandboxRuntimeError as exc:
        # Infrastructure failures are not Dockerfile repair opportunities, but
        # they must still be fail-soft so the REPL does not cascade through
        # undefined variables after build_environment raises.
        outcome = PrimitiveOutcome.retryable if getattr(exc, "retryable", False) else PrimitiveOutcome.fatal
        return _with_outcome({
            "ok": False,
            "image_tag": "",
            "error": f"build_environment: {type(exc).__name__}: {exc}",
            "attempts": attempts,
        }, outcome)
    except Exception as exc:  # noqa: BLE001 — fail-soft (D3): any other failure
        return _with_outcome({
            "ok": False,
            "image_tag": "",
            "error": f"{type(exc).__name__}: {exc}",
            "attempts": attempts,
        }, PrimitiveOutcome.repairable)

    return _with_outcome({
        "ok": ok,
        "image_tag": tag if ok else "",
        "error": error,
        "attempts": attempts,
    }, PrimitiveOutcome.ok if ok else PrimitiveOutcome.repairable)


def plan_reproduction(method_spec: dict, env_spec: dict, *, ctx: "RunContext") -> dict:
    """Generate a reproduction contract from structured specs via the LLM.

    Uses a primitive-specific system prompt (`_PLAN_REPRODUCTION_SYSTEM`). The
    orchestrator's `REPRODUCTION_PLANNER_PROMPT` is deliberately NOT reused: it
    instructs a file-writing agent ("write to `{runs_root}/{project_id}/...`"),
    which conflicts with a primitive that must return JSON inline. Returns a
    ReproductionContract dict.

    Fail-soft (A2-H3): `_extract_json` / schema validation failures return an
    error dict instead of propagating (the D3 pattern).
    """
    import json

    from backend.agents.schemas import ReproductionContract

    # Name the exact JSON keys: ReproductionContract is extra="ignore", so a
    # response keyed on prose-guessed names is silently dropped to an
    # all-defaults (near-empty) contract. Derived from the schema so the
    # prompt cannot drift from the model.
    fields = list(ReproductionContract.model_fields)
    user = (
        "method_spec:\n" + json.dumps(method_spec, indent=2, default=str)
        + "\n\nenvironment_spec:\n" + json.dumps(env_spec, indent=2, default=str)
        + "\n\nReturn exactly ONE JSON object with these keys and no others: "
        + json.dumps(fields)
        + ". String-valued keys hold prose; list-valued keys hold arrays of strings."
    )
    # Bug 1a — paper grounding check: if the method_spec contains dataset/method
    # names not found in the paper text, the plan_reproduction inputs are likely
    # contaminated from a different paper. Check before the LLM call so we don't
    # embed hallucinated names into the contract. Fail-soft: only emit a warning
    # (not a repairable error) if the paper text file is missing or the check itself
    # raises — grounding is advisory at plan_reproduction time to avoid false-positive
    # aborts on partial or summary-only paper texts.
    try:
        _paper_text_path = ctx.project_dir / "parsed_full_text.txt"
        if _paper_text_path.exists():
            from backend.agents.paper_grounding import assert_paper_grounded as _assert_grounded
            _grounding_violations = _assert_grounded(method_spec, _paper_text_path.read_text(encoding="utf-8", errors="replace"))
            if _grounding_violations:
                _unfounded = [v.value for v in _grounding_violations]
                logger.warning(
                    "plan_reproduction[%s]: %d grounding violation(s) — "
                    "input method_spec contains names not found in paper text: %s",
                    ctx.project_id, len(_grounding_violations), _unfounded,
                )
                _emit_dashboard_event(ctx, event_type="run_warning", payload={
                    "code": "paper_grounding_failed",
                    "message": (
                        f"plan_reproduction: {len(_grounding_violations)} name(s) in "
                        f"method_spec not found in paper text: {_unfounded[:5]}"
                    ),
                    "violations": [
                        {"field": v.field, "value": v.value, "suggestion": v.suggestion}
                        for v in _grounding_violations
                    ],
                })
    except Exception as _pg_exc:  # noqa: BLE001 — never block on grounding check
        logger.debug("plan_reproduction: grounding check failed (%s) — skipping", _pg_exc)

    from backend.agents.rlm import primitive_cache as _cache
    _payload = {"method_spec": method_spec, "env_spec": env_spec}
    _cached = _cache.maybe_get(ctx.project_dir, "plan_reproduction", payload=_payload)
    if _cached is not None:
        return _with_outcome(_cached, PrimitiveOutcome.ok)

    # β3: extend the planning prompt when compute is clipped.
    # θ: always append the metrics_shape instruction so the agent declares its
    # exact metric paths at plan time.
    system_prompt = _PLAN_REPRODUCTION_SYSTEM
    if _is_clipping_active(ctx):
        system_prompt = system_prompt + _COMPUTE_SCOPE_INSTRUCTION
    system_prompt = system_prompt + _METRICS_SHAPE_INSTRUCTION

    try:
        raw = ctx.llm_client.complete(system=system_prompt, user=user)
        data = _extract_json(raw)
        if not any(k in data for k in ReproductionContract.model_fields):
            raise ValueError(
                f"LLM response has no ReproductionContract fields: {list(data)}")

        # β3: parse compute_scope whenever the key is present in the LLM response.
        # Unconditional sanitization — a string-valued or malformed compute_scope
        # must never reach ReproductionContract(**data) regardless of clipping mode.
        if "compute_scope" in data:
            from pydantic import ValidationError as _PydanticValidationError
            from backend.agents.schemas import ComputeScope as _ComputeScope
            cs_dict = data.get("compute_scope")
            if isinstance(cs_dict, dict):
                try:
                    data["compute_scope"] = _ComputeScope(**cs_dict).model_dump()
                except _PydanticValidationError as _ve:
                    logger.warning(
                        "plan_reproduction: compute_scope validation failed (%s) — dropping",
                        _ve,
                    )
                    _emit_dashboard_event(ctx, event_type="run_warning", payload={
                        "code": "compute_scope_invalid",
                        "message": str(_ve)[:500],
                    })
                    data["compute_scope"] = None
            else:
                # String, None, or any non-dict — coerce to None rather than aborting.
                if cs_dict is not None:
                    logger.warning(
                        "plan_reproduction: compute_scope is not a dict (%s) — dropping",
                        type(cs_dict).__name__,
                    )
                    _emit_dashboard_event(ctx, event_type="run_warning", payload={
                        "code": "compute_scope_invalid",
                        "message": (
                            f"compute_scope must be a dict or null; got "
                            f"{type(cs_dict).__name__!r}: {str(cs_dict)[:200]}"
                        ),
                    })
                data["compute_scope"] = None
        else:
            # Not in the response (max mode or no instruction sent) — leave as None.
            data["compute_scope"] = None

        # θ: parse metrics_shape from the LLM response. Malformed entries are
        # skipped with a warning (not a crash) so a bad item doesn't abort the plan.
        # Missing metrics_shape → empty list (backward compat: fingerprint fallback).
        from pydantic import ValidationError as _PydanticValidationError
        from backend.agents.schemas import MetricPath as _MetricPath
        raw_shape = data.get("metrics_shape")
        parsed_shape: list[dict] = []
        if isinstance(raw_shape, list):
            for i, item in enumerate(raw_shape):
                if not isinstance(item, dict):
                    logger.warning(
                        "plan_reproduction: metrics_shape[%d] is not a dict (%s) — skipping",
                        i, type(item).__name__,
                    )
                    continue
                try:
                    parsed_shape.append(_MetricPath(**item).model_dump())
                except (_PydanticValidationError, TypeError) as _mp_err:
                    logger.warning(
                        "plan_reproduction: metrics_shape[%d] validation failed (%s) — skipping",
                        i, _mp_err,
                    )
                    _emit_dashboard_event(ctx, event_type="run_warning", payload={
                        "code": "metrics_shape_item_invalid",
                        "index": i,
                        "message": str(_mp_err)[:300],
                    })
        elif raw_shape is not None:
            logger.warning(
                "plan_reproduction: metrics_shape is not a list (%s) — treating as empty",
                type(raw_shape).__name__,
            )
        data["metrics_shape"] = parsed_shape

        # λ: scan method_spec + env_spec text for dataset mentions and populate
        # data_recipes with canonical loader recipes from the static registry.
        # This runs after the LLM response is validated so it never depends on
        # the LLM naming datasets correctly. find_recipes_in_text is a pure
        # string scan over the combined spec text — no additional LLM call.
        try:
            from dataclasses import asdict as _asdict
            from backend.agents.dataset_recipes import find_recipes_in_text as _find_recipes
            import json as _json
            _spec_text = (
                _json.dumps(method_spec, default=str)
                + " "
                + _json.dumps(env_spec, default=str)
            )
            _found = _find_recipes(_spec_text)
            data["data_recipes"] = [_asdict(r) for r in _found]
        except Exception as _dr_exc:  # noqa: BLE001 — never block on recipe scan
            logger.warning("plan_reproduction: data_recipes scan failed (%s) — empty list", _dr_exc)
            data["data_recipes"] = []

        result = _with_outcome(ReproductionContract(**data).model_dump(), PrimitiveOutcome.ok)
        _cache.put(ctx.project_dir, "plan_reproduction", payload=_payload, result=result)
        return result
    except Exception as exc:  # noqa: BLE001 — fail-soft (A2-H3 / D3 pattern)
        return _with_outcome({
            "success": False,
            "error": f"plan_reproduction: {type(exc).__name__}: {exc}",
        }, PrimitiveOutcome.repairable)


def _run_baseline_with_sdk(project_id, runs_root, pcm, env, contract, artifact_index, **kw):
    """Indirection over baseline_implementation.run_with_sdk so tests can patch it."""
    from backend.agents.baseline_implementation import run_with_sdk
    return run_with_sdk(project_id, runs_root, pcm, env, contract, artifact_index, **kw)


def implement_baseline(plan: dict, *, ctx: "RunContext") -> dict:
    """Generate the baseline code from a reproduction plan; return a typed envelope.

    `plan` is the aggregate dict the root assembles: `{"paper_claim_map":
    <understand_section output>, "environment_spec": <detect_environment
    output>, "reproduction_contract": <plan_reproduction output>}` (plus an
    optional `artifact_index`) — NOT a single producer's output. Wraps
    `baseline_implementation.run_with_sdk` (a code-writing agent) and writes
    `code/commands.json` so `run_experiment` can read the run commands without
    a BaselineResult (design decision D2). The return contract is always either
    ``{"ok": True, "code_path": ..., "files": [...]}`` or
    ``{"ok": False, "error_code": ..., "error": ..., "repairable": bool}``.

    Hardening (A2-C2): `pool.submit(...).result()` previously blocked the
    worker thread indefinitely; now bounded by `_timeout_for(ctx, 3600)`.
    On timeout returns a fail-soft error dict (never raises).

    Lane A — warm-retry cache: cached on `{plan, repair_context, arxiv_id,
    sandbox_mode, gpu_mode}` (NOT `remaining_s` — that changes every call).
    On cache hit we ALSO verify `code/commands.json` is still on disk and
    treat a missing manifest as a miss → recompute from scratch.  This handles
    the race where `attempt_isolation` archived the code AFTER the cache wrote.
    """
    import asyncio
    import json
    from pathlib import Path

    from backend.agents.schemas import PaperClaimMap, EnvironmentSpec, ReproductionContract

    # core_contribution is PaperClaimMap's one required field; default it so a
    # partial paper_claim_map (e.g. understand_section's output) validates.
    pcm = PaperClaimMap(**{"core_contribution": "", **plan.get("paper_claim_map", {})})
    env = EnvironmentSpec(**plan.get("environment_spec", {}))

    # Bug 3 fix: detect error envelopes from plan_reproduction before constructing
    # ReproductionContract. An envelope looks like {"success": False, "error": "..."}
    # or has an "error" key with no ReproductionContract fields (other than "outcome").
    _contract_dict = plan.get("reproduction_contract")
    contract: ReproductionContract | None = None
    if _contract_dict:
        _is_envelope = (
            _contract_dict.get("success") is False
            or (
                "error" in _contract_dict
                and not any(
                    k in _contract_dict
                    for k in ReproductionContract.model_fields
                    if k != "outcome"
                )
            )
        )
        if _is_envelope:
            _envelope_error = _contract_dict.get("error", "unknown plan_reproduction failure")
            logger.warning(
                "implement_baseline[%s]: plan_reproduction returned error envelope — "
                "contract set to None; fallback recipe recovery active. Error: %s",
                ctx.project_id, _envelope_error,
            )
            _emit_dashboard_event(ctx, event_type="run_warning", payload={
                "code": "plan_reproduction_failed_envelope",
                "message": (
                    f"plan_reproduction returned a failed envelope "
                    f"(error: {str(_envelope_error)[:300]}); "
                    f"proceeding with fallback recipe recovery"
                ),
            })
            contract = None
        else:
            contract = ReproductionContract(**_contract_dict)

    artifact_index = plan.get("artifact_index")

    # An optional plan["repair_context"] (a failed run_experiment result) puts
    # the code-writing agent into fix-existing-code mode — the root passes it
    # to retry after run_experiment fails.
    repair_context = plan.get("repair_context")

    # ------------------------------------------------------------------
    # PR-ι.2 — Patch-mode implement_baseline (killer fix for repeated bugs).
    #
    # When this is the 2nd+ call on the same run AND a prior train.py exists
    # on disk AND the repair_context carries structured contract/preflight
    # violations, DO NOT full-rewrite.  Instead:
    #   1. Read the prior train.py from disk.
    #   2. Build a "minimal diff" prompt: prior file + violation list.
    #   3. Apply the diff; on apply failure → fall back to full rewrite.
    #
    # This makes the violation hint structurally unavoidable — the exact
    # line+file appears in the prompt and Sonnet produces a diff against
    # the same file, so the bug must be fixed rather than re-introduced.
    # ------------------------------------------------------------------
    _train_py_path = ctx.project_dir / "code" / "train.py"
    _has_violations = bool(
        repair_context
        and (
            repair_context.get("contract_violations")
            or repair_context.get("preflight_violations")
        )
    )
    _use_patch_mode = (
        repair_context is not None
        and _has_violations
        and _train_py_path.exists()
        and getattr(ctx, "current_iteration", 0) >= 1
    )
    if _use_patch_mode:
        import asyncio as _asyncio
        from backend.agents.baseline_implementation import (
            patch_mode_run_with_sdk as _patch_run,
            _extract_violations_from_repair_context as _extract_violations,
        )
        _violations = _extract_violations(repair_context)
        _prior_content = _train_py_path.read_text(encoding="utf-8", errors="replace")
        logger.info(
            "implement_baseline[%s]: patch-mode triggered — %d violations, "
            "prior train.py is %d lines",
            ctx.project_id, len(_violations), _prior_content.count("\n"),
        )
        # Emit a warning so the SSE stream shows patch-mode is active.
        try:
            from backend.agents.rlm.sse_bridge import build_run_warning_event as _warn_ev
            if ctx.emit is not None:
                ctx.emit(_warn_ev(
                    level="info",
                    code="implement_baseline_patch_mode",
                    message=(
                        f"implement_baseline: using patch-mode ({len(_violations)} violations) "
                        f"instead of full rewrite — diff will be applied to existing train.py"
                    ),
                ))
        except Exception:  # noqa: BLE001
            logger.debug("implement_baseline: could not emit patch-mode SSE warning")

        try:
            _patch_success, _patch_result = _asyncio.run(
                _patch_run(
                    ctx.project_id,
                    ctx.runs_root,
                    _prior_content,
                    _violations,
                    repair_context,
                    model=getattr(ctx, "agent_model", None),
                    runtime=getattr(ctx, "runtime", None),
                )
            )
        except Exception as _patch_exc:  # noqa: BLE001
            _patch_success = False
            _patch_result = str(_patch_exc)

        if _patch_success:
            # Write the patched train.py back to disk atomically.
            _tmp = _train_py_path.with_suffix(".py.patch_tmp")
            _tmp.write_text(_patch_result, encoding="utf-8")
            import os as _os
            _os.replace(_tmp, _train_py_path)
            logger.info(
                "implement_baseline[%s]: patch-mode succeeded — train.py updated in-place",
                ctx.project_id,
            )
            return str(ctx.project_dir / "code")
        else:
            logger.warning(
                "implement_baseline[%s]: patch-mode failed (%s) — "
                "falling back to full rewrite",
                ctx.project_id, _patch_result,
            )
            # Fall through to full rewrite path below.

    # ------------------------------------------------------------------
    # Cache lookup BEFORE we spawn the expensive sub-agent.
    # Payload deliberately excludes ``remaining_s`` (every call differs)
    # but includes ``repair_context`` (different failure → different fix).
    # Sandbox/gpu enums are coerced to their .value strings so the key is
    # canonical across run instances.
    # ------------------------------------------------------------------
    # Bug 1b — second-chance paper grounding check on plan["paper_claim_map"].
    # Fail-soft: if the paper text file is missing or the check raises, skip silently.
    try:
        _paper_text_path = ctx.project_dir / "parsed_full_text.txt"
        _pcm_for_grounding = plan.get("paper_claim_map") or {}
        if _paper_text_path.exists() and _pcm_for_grounding:
            from backend.agents.paper_grounding import assert_paper_grounded as _assert_grounded2
            _grounding2 = _assert_grounded2(
                _pcm_for_grounding,
                _paper_text_path.read_text(encoding="utf-8", errors="replace"),
            )
            if _grounding2:
                _unfounded2 = [v.value for v in _grounding2]
                logger.warning(
                    "implement_baseline[%s]: %d grounding violation(s) — "
                    "paper_claim_map contains names not in paper text: %s",
                    ctx.project_id, len(_grounding2), _unfounded2,
                )
                _emit_dashboard_event(ctx, event_type="run_warning", payload={
                    "code": "paper_grounding_failed",
                    "message": (
                        f"implement_baseline: {len(_grounding2)} name(s) in "
                        f"paper_claim_map not found in paper text: {_unfounded2[:5]}"
                    ),
                    "violations": [
                        {"field": v.field, "value": v.value, "suggestion": v.suggestion}
                        for v in _grounding2
                    ],
                })
    except Exception as _pg2_exc:  # noqa: BLE001 — never block on grounding check
        logger.debug("implement_baseline: grounding check failed (%s) — skipping", _pg2_exc)

    from backend.agents.rlm import primitive_cache as _cache
    _sandbox_key = getattr(ctx.sandbox_mode, "value", str(ctx.sandbox_mode) if ctx.sandbox_mode is not None else None)
    _gpu_key = getattr(getattr(ctx, "gpu_mode", None), "value", str(getattr(ctx, "gpu_mode", None)) if getattr(ctx, "gpu_mode", None) is not None else None)
    from backend.agents.baseline_knowledge import KNOWLEDGE_CHANNEL_VERSION as _KC_VER
    _payload = {
        "plan": plan,
        "repair_context": repair_context,
        "arxiv_id": getattr(ctx, "arxiv_id", None),
        "sandbox_mode": _sandbox_key,
        "gpu_mode": _gpu_key,
        "knowledge_channel_version": _KC_VER,
    }
    _cached = _cache.maybe_get(ctx.project_dir, "implement_baseline", payload=_payload)
    if _cached is not None:
        # Recover old path cache entries and new envelope cache entries.
        _value: Any = _cached.get("value") if _cached.get("_kind") == "path" else _cached
        if isinstance(_value, str):
            # Verify the on-disk commands.json still exists — if attempt_isolation
            # archived the code between the cache write and now, treat as miss.
            _verify_dir = Path(_value)
            _harvested = _harvest_baseline_artifacts(_verify_dir)
            if _harvested.get("ok") is True:
                logger.info(
                    "implement_baseline: cache HIT (warm retry) for %s — "
                    "skipping ~5 min Sonnet sub-agent call",
                    ctx.project_id,
                )
                return _harvested
            logger.info(
                "implement_baseline: cache HIT but code/ missing on disk "
                "(probably archived) — recomputing from scratch",
            )
        elif isinstance(_value, dict):
            if _value.get("ok") is True:
                _verify_dir = Path(str(_value.get("code_path", "")))
                _harvested = _harvest_baseline_artifacts(_verify_dir)
                if _harvested.get("ok") is True:
                    return _harvested
                logger.info(
                    "implement_baseline: cached envelope points at incomplete code/ — "
                    "recomputing from scratch",
                )
            else:
                # Cached error dict (e.g. timeout) — normalize before returning.
                return _baseline_error_envelope(
                    error_code=str(_value.get("error_code") or _value.get("code") or "cached_failure"),
                    error=str(_value.get("error") or "cached implement_baseline failure"),
                    repairable=bool(_value.get("repairable", True)),
                    code_dir=_value.get("code_path"),
                    missing_files=_value.get("missing_files"),
                )

    # θ: extract metrics_shape from the contract so implement_baseline binds
    # Sonnet's code-writing to the declared paths. Coerce MetricPath objects
    # to plain dicts for JSON serialization across the SDK boundary.
    _metrics_shape: list[dict] = []
    if contract is not None:
        _raw_shape = getattr(contract, "metrics_shape", None) or []
        _metrics_shape = [
            (mp.model_dump() if hasattr(mp, "model_dump") else dict(mp))
            for mp in _raw_shape
            if mp is not None
        ]

    # λ: extract data_recipes from the contract so implement_baseline binds
    # Sonnet to use canonical dataset loaders verbatim. Plain dicts only.
    _data_recipes: list[dict] = []
    if contract is not None:
        _raw_dr = getattr(contract, "data_recipes", None) or []
        _data_recipes = [
            (r.model_dump() if hasattr(r, "model_dump") else dict(r))
            for r in _raw_dr
            if r is not None
        ]

    async def _run():
        # ctx.agent_model is the per-invocation model_override — it is the only
        # knob that beats the agent registry's heavier default for the
        # baseline-implementation agent (Opus). None -> registry default.
        # 2026-05-23 (final): thread ctx.sandbox_mode through so the agent can
        # pick a CPU-friendly baseline (smoke-test mode) when the sandbox has
        # no GPU. Without this, B2 of the paper sweep wrote a real VLM
        # training that hung indefinitely on docker (CPU-only).
        return await _run_baseline_with_sdk(
            ctx.project_id, ctx.runs_root, pcm, env, contract, artifact_index,
            runtime=ctx.runtime, model=ctx.agent_model,
            repair_context=repair_context,
            sandbox_mode=ctx.sandbox_mode,
            gpu_mode=getattr(ctx, "gpu_mode", None),
            arxiv_id=getattr(ctx, "arxiv_id", None),
            # Budget awareness: hand the agent the same remaining_s the
            # run_experiment primitive uses, so its train.py can scale to fit.
            remaining_s=ctx.remaining_s(),
            # Lane Q — minimize-compute knob, threaded through the execution
            # profile. When True, the agent's prompt gets the substitution
            # rules + scope.declared_reductions contract.
            minimize_compute=getattr(ctx, "minimize_compute", False),
            # θ: pass declared metric paths so the Sonnet agent is bound to emit
            # exactly these paths in metrics.json.
            metrics_shape=_metrics_shape or None,
            # λ: pass canonical dataset loader recipes so the Sonnet agent uses
            # the correct loader verbatim (e.g. stanfordnlp/imdb not bare 'imdb').
            data_recipes=_data_recipes or None,
            # GPU parallelism policy — controls DDP/FSDP/vLLM-TP vs single GPU.
            gpu_parallelism=getattr(ctx, "gpu_parallelism", None),
            gpu_visible_count=getattr(ctx, "gpu_visible_count", None),
        )

    # Generous 4 h cap for implement_baseline (the sub-agent that writes code).
    # The outer ctx.remaining_s() wall-clock still binds the whole run.
    timeout = _timeout_for(ctx, 14400)
    # I12: explicit shutdown(wait=False) so a wedged worker cannot block cleanup.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # SDK aclose deadlock watchdog (2026-05-24): on Windows, the claude-agent-sdk's
    # async generator cleanup hangs after the sub-agent finishes writing code.
    # The future never resolves even though all files are on disk. We detect this
    # by polling: if commands.json exists AND no new files are written for
    # _ACLOSE_STALL_S seconds, the SDK is deadlocked — break out and proceed
    # with the code that was already written.
    _ACLOSE_STALL_S = 120  # 2 min of no file changes after code is written
    _POLL_S = 10
    _PRE_EMIT_STALL_S = _pre_emit_stall_s()

    code_dir = ctx.runs_root / ctx.project_id / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    try:
        future = pool.submit(asyncio.run, _run())
        import time as _time
        deadline_abs = _time.monotonic() + timeout
        _stall_start: float | None = None
        _pre_emit_stall_start = _time.time()
        logger.info(
            "implement_baseline: pre-emit watchdog armed (%ds grace)",
            int(_PRE_EMIT_STALL_S),
        )

        while True:
            remaining_timeout = deadline_abs - _time.monotonic()
            if remaining_timeout <= 0:
                _err = _baseline_error_envelope(
                    error_code="timeout",
                    error=f"implement_baseline: timed out after {timeout:.0f} s",
                    code_dir=code_dir,
                )
                _cache.put(ctx.project_dir, "implement_baseline", payload=_payload, result=_err)
                return _err
            try:
                result = future.result(timeout=max(0.1, min(_POLL_S, remaining_timeout)))
                break  # Normal return
            except concurrent.futures.TimeoutError:
                pass  # Check watchdog conditions below
            except Exception as exc:  # noqa: BLE001
                harvested = _harvest_baseline_artifacts(
                    code_dir,
                    error_code="sdk_failure_incomplete_artifacts",
                    error_prefix=f"implement_baseline SDK failed after artifact write ({type(exc).__name__}: {exc})",
                )
                if harvested.get("ok") is True:
                    _cache.put(ctx.project_dir, "implement_baseline", payload=_payload, result=harvested)
                    return harvested
                harvested["sdk_error"] = f"{type(exc).__name__}: {exc}"
                return harvested

            # Check overall timeout
            if deadline_abs - _time.monotonic() <= 0:
                _err = _baseline_error_envelope(
                    error_code="timeout",
                    error=f"implement_baseline: timed out after {timeout:.0f} s",
                    code_dir=code_dir,
                )
                _cache.put(ctx.project_dir, "implement_baseline", payload=_payload, result=_err)
                return _err

            # SDK aclose deadlock detection: if commands.json exists (code written)
            # and no files have changed recently, the SDK is hung on cleanup.
            commands_json = code_dir / "commands.json"
            if not commands_json.exists():
                # Mtime-based progress: ANY file the sub-agent writes counts as
                # progress (config.json / requirements.txt / rubric_guard.py /
                # train.py / commands.json / etc.) — not a hardcoded name list.
                # 2026-05-27 Adam+VAE regression: the prior name-list missed
                # legitimate progress (sub-agent wrote rubric_guard.py + config.json
                # but not yet commands.json) → false-positive escalation cascade.
                latest_mtime = max(
                    (f.stat().st_mtime for f in code_dir.iterdir() if f.is_file()),
                    default=0.0,
                )
                if latest_mtime > _pre_emit_stall_start:
                    _pre_emit_stall_start = _time.time()
                    continue
                pre_emit_elapsed = _time.time() - _pre_emit_stall_start
                if pre_emit_elapsed > _PRE_EMIT_STALL_S:
                    message = (
                        "implement_baseline: SDK pre-emit stall — no file activity "
                        f"in {_PRE_EMIT_STALL_S:.0f}s. Likely SDK aclose deadlock pre-result."
                    )
                    logger.warning(
                        "implement_baseline: code_dir idle for %ds. "
                        "Escalating to repairable error (NOT cached — retry will be fresh).",
                        int(pre_emit_elapsed),
                    )
                    try:
                        _emit_dashboard_event(ctx, event_type="run_warning", payload={
                            "code": "sdk_pre_emit_stall",
                            "message": message,
                        })
                    except Exception:  # noqa: BLE001
                        logger.debug("implement_baseline: failed to emit pre-emit stall warning")
                    # Deliberately NOT cached: the stall is transient — a retry
                    # may succeed. Caching this guarantees a cascade where every
                    # downstream run_experiment receives the same error dict.
                    return _baseline_error_envelope(
                        error_code="sdk_pre_emit_stall",
                        error=message,
                        code_dir=code_dir,
                        missing_files=["commands.json", "runnable source file"],
                    )
                continue

            if commands_json.exists():
                # Check if any file in code/ was modified in the last _POLL_S
                now = _time.time()
                latest_mtime = max(
                    (f.stat().st_mtime for f in code_dir.iterdir() if f.is_file()),
                    default=0,
                )
                if now - latest_mtime > _POLL_S:
                    # No file changes — start or continue the stall timer
                    if _stall_start is None:
                        _stall_start = _time.time()
                        logger.info(
                            "implement_baseline: code written, SDK idle — "
                            "watching for aclose deadlock (%ds grace)",
                            _ACLOSE_STALL_S,
                        )
                    elif _time.time() - _stall_start > _ACLOSE_STALL_S:
                        logger.warning(
                            "implement_baseline: SDK aclose deadlock detected — "
                            "code is on disk but SDK hung for %ds. Breaking out.",
                            int(_time.time() - _stall_start),
                        )
                        harvested = _harvest_baseline_artifacts(
                            code_dir,
                            error_code="sdk_aclose_incomplete_artifacts",
                            error_prefix="implement_baseline SDK cleanup stalled after artifact write",
                        )
                        if harvested.get("ok") is True:
                            _cache.put(ctx.project_dir, "implement_baseline", payload=_payload, result=harvested)
                        return harvested
                else:
                    # Files still being written — reset stall timer
                    _stall_start = None
    except concurrent.futures.TimeoutError:
        _err = _baseline_error_envelope(
            error_code="timeout",
            error=f"implement_baseline: timed out after {timeout:.0f} s",
            code_dir=code_dir,
        )
        _cache.put(ctx.project_dir, "implement_baseline", payload=_payload, result=_err)
        return _err
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # run_with_sdk writes the generated code to runs_root/project_id/code;
    # derive commands.json's directory the same way (not ctx.project_dir/code)
    # so the manifest provably lands alongside the code regardless of how
    # RunContext.project_dir was constructed.
    code_dir.mkdir(parents=True, exist_ok=True)
    commands = list(getattr(result, "commands_to_run", []) or [])
    if commands:
        (code_dir / "commands.json").write_text(json.dumps(commands), encoding="utf-8")

    # PR-ξ γ: surface knowledge-channel strict violations as a repairable envelope.
    # run_with_sdk encodes violations in diff_summary and assumptions_applied when
    # the post-emit verifier found strict violations. Check and propagate here.
    _kc_summary = getattr(result, "diff_summary", "") or ""
    if _kc_summary.startswith("knowledge_channel:"):
        _kc_assumptions = list(getattr(result, "assumptions_applied", []) or [])
        _kc_preflight = [
            {
                "kind": a.split(":", 2)[1] if a.count(":") >= 2 else "strict_violation",
                "fact_id": a.split(":", 2)[2] if a.count(":") >= 2 else a,
                "detail": a,
            }
            for a in _kc_assumptions
            if a.startswith("kc_violation:")
        ]
        _emit_dashboard_event(ctx, event_type="run_warning", payload={
            "code": "knowledge_channel_strict_violation",
            "message": _kc_summary,
            "violations": _kc_preflight,
        })
        _err = _baseline_error_envelope(
            error_code="knowledge_channel_strict_violation",
            error=_kc_summary,
            code_dir=code_dir,
        )
        _err["preflight_violations"] = _kc_preflight
        # Do NOT cache a strict-violation result — force recompute on next attempt.
        return _err

    harvested = _harvest_baseline_artifacts(code_dir)
    if harvested.get("ok") is True:
        _cache.put(ctx.project_dir, "implement_baseline", payload=_payload, result=harvested)
    return harvested


def _backend_for_sandbox_mode(
    sandbox_mode: object,
    *,
    run_budget: object = None,
    gpu_plan: "GpuPlan | None" = None,
):
    """Return a RuntimeBackend instance for the given sandbox mode.

    ``SandboxMode.docker`` (and ``None`` / the default) map to
    ``LocalDockerBackend``.  ``SandboxMode.runpod`` is now fully wired: this
    function calls ``ensure_runpod_available()`` (fast fail on missing creds)
    and constructs a real ``RunpodBackend``, forwarding ``run_budget`` so the
    ``max_pod_seconds`` cap is enforced at each ``exec()`` call.

    Any other unsupported mode (local, auto, brev, simulate) falls back to
    ``LocalDockerBackend`` with a WARNING rather than crashing, so the run still
    produces a result while making the misconfiguration visible.

    ``run_budget=None`` is safe for all modes — no cap is enforced.
    ``gpu_plan=None`` is safe for all modes — RunpodBackend falls back to
    legacy Settings defaults when None.  Non-runpod backends ignore it.
    """
    from backend.agents.execution import SandboxMode
    from backend.services.runtime.local_docker import LocalDockerBackend

    if sandbox_mode is None:
        return LocalDockerBackend()
    try:
        mode = SandboxMode(sandbox_mode)
    except ValueError:
        logger.warning(
            "_execute_in_sandbox: unknown sandbox_mode %r — falling back to LocalDockerBackend",
            sandbox_mode,
        )
        return LocalDockerBackend()

    if mode is SandboxMode.docker:
        return LocalDockerBackend()

    if mode is SandboxMode.local:
        from backend.services.runtime.local_process import LocalProcessBackend
        return LocalProcessBackend()

    if mode is SandboxMode.runpod:
        import backend.services.runtime as _runtime
        from backend.services.runtime.runpod_backend import RunpodBackend

        _runtime.ensure_runpod_available()
        return RunpodBackend(run_budget=run_budget, gpu_plan=gpu_plan)

    # All other modes (auto, brev, simulate) are not yet wired
    # for the RLM path.  Fall back with a loud WARNING so the operator knows.
    logger.warning(
        "_execute_in_sandbox: sandbox_mode=%r is not supported in the RLM "
        "path — falling back to LocalDockerBackend.  "
        "Set --sandbox docker or --sandbox runpod for a supported backend.",
        mode.value,
    )
    return LocalDockerBackend()


def _combine_command_output(results: list) -> str:
    """Join sandbox command results into one log — stdout AND stderr, in order.

    A failed command writes its diagnostics (tracebacks, missing-module errors)
    to stderr. Building the log from stdout alone left every run_experiment
    failure with logs="" — undiagnosable on disk, and useless as the
    repair_context the code agent needs to fix the baseline.
    """
    parts: list[str] = []
    for r in results:
        if r.stdout:
            parts.append(r.stdout)
        if r.stderr:
            parts.append(r.stderr)
    return "\n".join(parts)


async def _execute_in_sandbox(
    code_path: str,
    env_id: str,
    commands: list[str],
    *,
    project_id: str,
    run_id: str,
    sandbox_mode: object = None,
    run_budget: object = None,
    gpu_plan: object = None,
    gpu_mode: object = None,
    gpu_device_ids: tuple[str, ...] = (),
) -> dict:
    """Run `commands` in a container started from the prebuilt image `env_id`.

    Drives the verified `RuntimeAppService` lifecycle (`service.py`): create a
    sandbox from the existing image (`dockerfile_path=None`, `build_context=None`
    → no rebuild, design decision D1), execute each command, destroy. The
    service methods take `Command` objects. Indirection so tests can patch it.

    Hardening (A2-C1): `asyncio.shield` on destroy so the container is cleaned
    up even when the outer thread's `.result(timeout=...)` fires and the
    coroutine is cancelled.

    I7: `sandbox_mode` selects the runtime backend; ``None`` / ``docker`` map to
    ``LocalDockerBackend`` (behaviour-identical to the previous hardcoded path).
    ``runpod`` constructs a real ``RunpodBackend`` with the given ``run_budget``
    (so ``max_pod_seconds`` is enforced).  Any unsupported mode falls back to
    ``LocalDockerBackend`` with a WARNING.
    """
    import asyncio
    import json as _json
    from pathlib import Path

    from backend.services.runtime.interface import SandboxConfig, SandboxRuntimeError
    from backend.services.runtime.service import (
        CreateSandbox, DestroySandbox, ExecuteCommand,
    )

    # Sandbox routing authority: callers must pass ctx.sandbox_mode into this
    # argument; env_id is an image identifier only. If the model accidentally
    # passes a backend name as env_id, ignore it for routing and keep using
    # sandbox_mode.
    _env_hint = str(env_id or "").strip().lower()
    _mode_value = (
        str(getattr(sandbox_mode, "value", sandbox_mode or "")).strip().lower()
    )
    if _env_hint in {"local", "docker", "runpod"} and _env_hint != _mode_value:
        logger.warning(
            "_execute_in_sandbox: env_id=%r looks like a backend hint; "
            "ignoring it for sandbox routing and using sandbox_mode=%r",
            env_id,
            sandbox_mode,
        )

    code_dir = Path(code_path)
    # Per-call artifact dir: deterministic per run_id so retries don't clobber.
    artifact_root = code_dir / "outputs" / run_id
    artifact_root.mkdir(parents=True, exist_ok=True)

    service = RuntimeAppService(_backend_for_sandbox_mode(
        sandbox_mode, run_budget=run_budget, gpu_plan=gpu_plan,
    ))
    # gpu_mode threads ctx.gpu_mode → SandboxConfig so LocalDockerBackend's
    # is_gpu_passthrough_mode predicate sees the user's actual choice. Without
    # this the SandboxConfig default ("auto") makes --gpus all silently
    # skipped, and the container runs CPU-only even with --gpu-mode prefer.
    _gpu_mode_str = (
        getattr(gpu_mode, "value", str(gpu_mode))
        if gpu_mode is not None
        else "auto"
    )
    # A4(a): When running local with a per-run venv, inject the venv's bin
    # directory at the front of PATH so the experiment subprocess uses that
    # interpreter and its installed packages rather than the system Python.
    import os as _os
    _exp_env_extra: dict[str, str] = {}
    _mode_str_local = str(getattr(sandbox_mode, "value", sandbox_mode) or "").lower()
    _venv = (_os.environ.get("REPROLAB_EXPERIMENT_VENV") or "").strip()
    if _mode_str_local == "local" and _venv:
        _exp_env_extra["VIRTUAL_ENV"] = _venv
        _exp_env_extra["PATH"] = f"{_venv}/bin:" + _os.environ.get("PATH", "")

    config = SandboxConfig(
        project_id=project_id,
        run_id=run_id,
        image=env_id,
        project_root=code_dir,
        artifact_root=artifact_root,
        gpu_mode=_gpu_mode_str,
        gpu_device_ids=tuple(gpu_device_ids or ()),
        dockerfile_path=None,   # prebuilt image — no rebuild (design decision D1)
        build_context=None,
        # Bug C: paper reproduction must fetch pretrained weights and datasets
        # (HuggingFace, PyPI, torch hub) — network_disabled defaults to True and
        # blocked every model-download paper. The paper corpus is never mounted
        # into this container (only agent-written code is), so this is not a
        # corpus-leak vector. Scoped here; the global default stays disabled.
        network_disabled=False,
        environment={
            # Local sandbox: /artifacts doesn't exist on most hosts and can't be
            # created without root.  Point OUTPUT_DIR straight at artifact_root
            # so train.py doesn't need to fall back via directory introspection.
            # Docker/RunPod: /artifacts is the container-mounted volume — keep it.
            "OUTPUT_DIR": str(artifact_root) if _mode_str_local == "local" else "/artifacts",
            "REPROLAB_ARTIFACT_DIR": str(artifact_root) if _mode_str_local == "local" else "/artifacts",
            "MPLCONFIGDIR": str(artifact_root / ".matplotlib") if _mode_str_local == "local" else "/artifacts/.matplotlib",
            "PYTHONUNBUFFERED": "1",
            **_exp_env_extra,
        },
    )
    resource_limits = {
        "memory_limit": config.memory_limit,
        "cpus": config.cpus,
        "gpu_mode": config.gpu_mode,
        "sandbox_mode": str(sandbox_mode or "docker"),
    }
    try:
        _emit_dashboard_event_to_path(
            code_dir.parent if code_dir.name == "code" else code_dir,
            event_type="sandbox_resource_limits",
            payload={
                "project_id": project_id,
                "run_id": run_id,
                **resource_limits,
            },
        )
    except Exception:  # noqa: BLE001 - resource observability must not block
        logger.exception("_execute_in_sandbox: sandbox_resource_limits emit failed")
    # Auto-install requirements.txt on RunPod BEFORE commands.json runs. The
    # agent's prompt has repeatedly forgotten to wire `python -m pip install -r
    # requirements.txt` into commands.json on runpod (Dockerfile is doc-only;
    # pod boots from the generic pytorch image without the paper's deps). Make
    # this a backend invariant so every paper's requirements.txt is honored
    # whether the agent remembers or not. Local docker is unaffected because
    # the Dockerfile IS used to build the image — deps are already baked in.
    requirements_path = code_dir / "requirements.txt"
    bootstrap_commands: list[str] = []

    # Lane E: pod-side heartbeat daemon. Writes a unix timestamp to
    # /artifacts/.heartbeat every 30 s. Detects pod-level wedges (NCCL
    # deadlock, dead kernel, frozen HF download) that produce zero stdout
    # — exec.log size stays flat for hours but the pod itself is wedged.
    # Backgrounded via ``nohup ... &`` so it doesn't block subsequent commands.
    try:
        from backend.agents.rlm.run_watchdog import heartbeat_daemon_command, is_enabled as _watchdog_enabled
        if _watchdog_enabled():
            # Local sandbox: /artifacts is not the real artifact dir — use the
            # per-run artifact_root so the heartbeat file is actually writable.
            _hb_dir = str(artifact_root) if _mode_str_local == "local" else "/artifacts"
            bootstrap_commands.append(heartbeat_daemon_command(_hb_dir))
    except Exception:  # noqa: BLE001 — instrumentation MUST NOT block the run
        logger.exception("_execute_in_sandbox: heartbeat-daemon injection failed")

    # sandbox_mode may be a SandboxMode enum (str(...) is "SandboxMode.runpod")
    # OR a plain string "runpod". Use substring match to cover both forms.
    _mode_str = str(sandbox_mode).lower() if sandbox_mode else ""
    if "runpod" in _mode_str:
        # Lane 6: when REPROLAB_BOOTSTRAP_MKDIRS is set by the RunPod backend
        # (because a network volume is mounted for persistent pip / HF cache),
        # create those dirs FIRST so pip and HuggingFace can write to them.
        # Pre-pip step — must run before any other bootstrap.
        bootstrap_commands.append(
            'mkdir -p ${REPROLAB_BOOTSTRAP_MKDIRS:-/tmp/.reprolab_noop}'
        )
        # Lane 1: auto-derive requirements.txt from the Dockerfile when the
        # agent forgot to write one. The local-docker sandbox path builds an
        # image from the Dockerfile (every pip dep installed); the RunPod
        # path uses a pre-built PyTorch image and ONLY bootstraps from
        # requirements.txt. Without this, every paper missing requirements.txt
        # silently fails with ModuleNotFoundError once the agent's train.py
        # imports anything beyond torch/numpy.
        if not requirements_path.exists():
            try:
                from backend.agents.rlm.requirements_derive import ensure_requirements_txt
                _project_dir = code_dir.parent if code_dir.name == "code" else code_dir
                # Pass the RunPod base image so the synthesized requirements.txt
                # can strip packages already baked into the image (torch /
                # torchvision / torchaudio on runpod/pytorch* bases).  Without
                # this strip, every cold pod re-downloaded the 755 MB torch
                # wheel and ~50% of the time the connection dropped mid-stream
                # (Adam v10 #2 failure).
                ensure_requirements_txt(
                    code_dir,
                    dockerfile_path=_project_dir / "Dockerfile",
                    base_image=env_id,
                )
            except Exception:  # noqa: BLE001 — observability must never block the run
                logger.exception("_execute_in_sandbox: requirements.txt auto-derive failed")
        if requirements_path.exists():
            # Lane 6 detail: drop ``--no-cache-dir`` so pip writes to
            # PIP_CACHE_DIR.  When PIP_CACHE_DIR points to a mounted network
            # volume, subsequent pods reuse the cache and skip the ~2 GB
            # torch+matplotlib+datasets download.  When no volume is mounted
            # the cache lives in /root/.cache (pod-local, lost on destroy)
            # which is no worse than the previous ``--no-cache-dir`` path
            # but uses the cache within the same pod across retries.
            bootstrap_commands.append(
                "python -m pip install --upgrade pip wheel setuptools"
            )
            bootstrap_commands.append(
                "python -m pip install -r requirements.txt"
            )

    # A4(b): Local sandbox — auto-install requirements.txt into the per-run
    # venv (PATH already points there via _exp_env_extra). Mirrors the runpod
    # block above: same command string, same position (prepended before the
    # agent's commands via the (*bootstrap_commands, *commands) loop at line
    # ~2183). Safe to run even if the venv is absent — pip will use the
    # active Python. bootstrap_commands feeds into service.execute() for ALL
    # backends through the unified loop, so this path is genuine for local.
    #
    # NOTE: On this host `python` is not in PATH (only `python3`), so we use
    # `|| true` to prevent non-zero exit codes from causing `success=False`
    # in the all(r.succeeded) check. The commands.json entry handles the
    # actual package install via an explicit PATH-export bash -c command.
    if "local" in _mode_str and requirements_path.exists():
        bootstrap_commands.append(
            "python -m pip install --upgrade pip wheel setuptools || true"
        )
        bootstrap_commands.append(
            "python -m pip install -r requirements.txt || true"
        )

    # Lane E: spawn the stall watchdog alongside command execution.
    # It polls exec.log + .heartbeat + dashboard_events.jsonl every 30 s
    # and emits run_warning SSE events at the warn-threshold + invokes
    # on_kill at the hard-threshold (which raises WatchdogKilled to break
    # out of the execute loop).
    from backend.agents.rlm.run_watchdog import (
        KillVerdict as _KillVerdict,
        WatchdogConfig as _WatchdogConfig,
        run_watchdog as _run_watchdog,
        is_enabled as _watchdog_enabled,
    )

    class _WatchdogKilled(RuntimeError):
        """Raised by on_kill so the execute loop unwinds cleanly via finally."""

    project_dir_for_watchdog = code_dir.parent if code_dir.name == "code" else code_dir
    # Lane N — bounded recovery budget. Pod is destroyed once this is exhausted.
    import os as _os_env_wd
    _MAX_SOFT_RECOVERIES = int(_os_env_wd.environ.get("REPROLAB_WATCHDOG_MAX_SOFT_RECOVERIES", "3"))

    _wd_cfg = _WatchdogConfig.from_env()

    # PR-ζ: transient-error retry loop.
    # Wraps the entire create→execute→destroy lifecycle. The bootstrap_commands
    # and watchdog callback closures are computed once; only the sandbox
    # creation and execution are retried. Each retry gets a fresh sandbox —
    # the watchdog closures close over `sandbox` by name and see the updated
    # value at call time.
    #
    # Wall-clock guard: total time spent in the retry loop (backoff only, not
    # the experiment itself) is capped at _RETRY_TIMEOUT_TOTAL_S so this
    # cannot blow through the surrounding run_experiment timeout.
    from backend.services.runtime.transient_classifier import (
        TransientClass as _TransientClass,
        classify_exception as _classify_exception,
    )
    _retry_attempts: list[dict] = []
    _retry_loop_start = asyncio.get_event_loop().time()

    # `sandbox` is declared here so the watchdog closures (defined inside the
    # loop) close over it by name and always reference the current attempt's
    # sandbox — not a stale reference from a prior attempt.
    sandbox = None

    for _retry_idx in range(_MAX_TRANSIENT_RETRIES + 1):
        sandbox = None
        results = []
        _soft_recovery_count = 0
        _wd_task = None
        _last_sre: SandboxRuntimeError | None = None
        _backoff: float = _BACKOFF_BASE_S * (2 ** _retry_idx)

        async def _emit_warn_real(report) -> None:
            try:
                _emit_dashboard_event_to_path(
                    project_dir_for_watchdog,
                    event_type="run_warning",
                    payload={
                        "reason": "stale_run",
                        **report.to_dict(),
                        "thresholds": {
                            "warn_after_seconds": _wd_cfg.warn_after_seconds,
                            "kill_after_seconds": _wd_cfg.kill_after_seconds,
                        },
                    },
                )
            except Exception:  # noqa: BLE001 — observability never blocks
                logger.exception("watchdog warn-emit failed")

        async def _emit_kill_real(report):
            """Lane N — escalating probe-recover.

            Distinguishes "pod truly dead" from "pod alive, agent slow-printing".
            Each KILL verdict from the watchdog walks one rung up the ladder:

              Strike 0 (probe OK, count=0): just warn + RECOVERED — give the
                run more time. A train.py with prints-every-25-epochs at 12 s/epoch
                takes >10 min between prints; that's slow, not wedged.
              Strike 1 (probe OK, count=1): warn + soft_recover (pkill in-pod
                train.py) + RECOVERED — first benefit-of-doubt was used; the
                staleness is persistent, so the in-pod process is likely wedged.
              Strike 2+ (probe OK, count >= MAX): destroy pod, raise
                _WatchdogKilled. We've spent the budget.
              Any probe FAIL: destroy immediately (pod is gone).
            """
            nonlocal _soft_recovery_count

            # 1. probe alive via FRESH SSH channel (not the wedged one).
            probe_ok = False
            try:
                probe_ok = await service.probe_alive(sandbox, timeout=10.0)
            except Exception:  # noqa: BLE001 — probe never raises in our impl
                logger.exception("watchdog probe raised — treating as dead")

            # 2. Escalating recovery if alive.
            if probe_ok and _soft_recovery_count < _MAX_SOFT_RECOVERIES:
                current_strike = _soft_recovery_count
                _soft_recovery_count += 1
                # Strike 0 = benefit of doubt (no in-pod kill, just warn).
                # Strikes >= 1 = the staleness is persistent, soft-recover.
                should_soft_recover = current_strike >= 1
                recover_ok = False
                if should_soft_recover:
                    try:
                        recover_ok = await service.soft_recover(sandbox)
                    except Exception:  # noqa: BLE001
                        logger.exception("watchdog soft_recover raised")
                reason = (
                    "pod_alive_under_watchdog_soft_recovered" if should_soft_recover
                    else "pod_alive_under_watchdog_grace"
                )
                try:
                    _emit_dashboard_event_to_path(
                        project_dir_for_watchdog,
                        event_type="run_warning",
                        payload={
                            "reason": reason,
                            "soft_recovery_count": _soft_recovery_count,
                            "max_soft_recoveries": _MAX_SOFT_RECOVERIES,
                            "soft_recover_attempted": should_soft_recover,
                            "soft_recover_succeeded": recover_ok,
                            **report.to_dict(),
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("watchdog recover-emit failed")
                logger.warning(
                    "watchdog: probe OK strike=%d/%d soft_recover=%s ok=%s — keeping pod warm",
                    _soft_recovery_count, _MAX_SOFT_RECOVERIES,
                    should_soft_recover, recover_ok,
                )
                return _KillVerdict.RECOVERED

            # 3. Destroy: pod is dead OR we've recovered too many times.
            try:
                _emit_dashboard_event_to_path(
                    project_dir_for_watchdog,
                    event_type="run_warning",
                    payload={
                        "reason": "pod_killed_by_watchdog",
                        "probe_ok": probe_ok,
                        "soft_recovery_count": _soft_recovery_count,
                        "max_soft_recoveries": _MAX_SOFT_RECOVERIES,
                        **report.to_dict(),
                        "thresholds": {
                            "warn_after_seconds": _wd_cfg.warn_after_seconds,
                            "kill_after_seconds": _wd_cfg.kill_after_seconds,
                        },
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("watchdog kill-emit failed")
            # Tear down the sandbox so the in-flight execute returns promptly.
            try:
                await asyncio.shield(service.destroy(DestroySandbox(sandbox=sandbox)))
            except Exception:  # noqa: BLE001
                logger.exception("watchdog kill-destroy failed")
            raise _WatchdogKilled(
                f"Watchdog killed run after {report.stale_seconds:.0f}s of no signal "
                f"(freshest={report.freshest_signal}); probe_ok={probe_ok} "
                f"recoveries={_soft_recovery_count}/{_MAX_SOFT_RECOVERIES}"
            )

        try:
            sandbox = await service.create_sandbox(CreateSandbox(config=config))

            if _watchdog_enabled():
                _wd_task = asyncio.create_task(_run_watchdog(
                    artifact_root=artifact_root,
                    project_dir=project_dir_for_watchdog,
                    config=_wd_cfg,
                    on_warn=_emit_warn_real,
                    on_kill=_emit_kill_real,
                ))

            for command in (*bootstrap_commands, *commands):
                results.append(await service.execute(
                    ExecuteCommand(sandbox=sandbox, command=command,
                                   timeout=_EXEC_TIMEOUT_SECONDS)))
        except _WatchdogKilled as exc:
            # Surface as a fail-soft error dict so the caller's outer escalation
            # loop treats it like an OOM (advance ladder / repair_context).
            # Watchdog kills are not retried — the pod was destroyed deliberately.
            return {
                "success": False,
                "metrics": {},
                "logs": _cap_logs(_combine_command_output(results)),
                "error": f"run_experiment: {exc}",
                "watchdog_killed": True,
            }
        except SandboxRuntimeError as _sre:
            # PR-ζ: classify and decide whether to retry.
            _klass = _classify_exception(_sre)
            _retry_attempts.append({
                "attempt": _retry_idx + 1,
                "transient_class": _klass.value,
                "error": str(_sre)[:300],
            })
            _elapsed_retry = asyncio.get_event_loop().time() - _retry_loop_start
            _can_retry = (
                _klass == _TransientClass.transient
                and _retry_idx < _MAX_TRANSIENT_RETRIES
                and (_elapsed_retry + _backoff) <= _RETRY_TIMEOUT_TOTAL_S
            )
            if _klass == _TransientClass.fatal or _klass == _TransientClass.code_bug:
                # Propagate immediately — no retry benefit.
                raise
            if _can_retry:
                try:
                    _emit_dashboard_event_to_path(
                        project_dir_for_watchdog,
                        event_type="sandbox_retry",
                        payload={
                            "attempt": _retry_idx + 1,
                            "max": _MAX_TRANSIENT_RETRIES,
                            "backoff_s": _backoff,
                            "transient_class": _klass.value,
                            "error": str(_sre)[:300],
                        },
                    )
                except Exception:  # noqa: BLE001 — observability never blocks
                    logger.exception("_execute_in_sandbox: sandbox_retry emit failed")
                logger.warning(
                    "_execute_in_sandbox: transient error (attempt %d/%d) — "
                    "retrying after %.0fs backoff. %s",
                    _retry_idx + 1, _MAX_TRANSIENT_RETRIES + 1,
                    _backoff, str(_sre)[:200],
                )
                # Continue to finally (which destroys the sandbox if it was
                # created), then sleep, then the next loop iteration.
                _last_sre = _sre
            else:
                # Exhausted retries or unknown class — propagate with attempts
                # so the caller's repair_context shows what was tried.
                _sre._retry_attempts = _retry_attempts  # type: ignore[attr-defined]
                raise
        finally:
            # Cancel the watchdog task BEFORE destroy so an in-flight on_kill
            # doesn't double-destroy. asyncio.shield: destroy completes even
            # if the surrounding wait_for / thread-pool timeout cancels this
            # coroutine (A2-C1).
            if _wd_task is not None and not _wd_task.done():
                _wd_task.cancel()
                try:
                    await _wd_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if sandbox is not None:
                await asyncio.shield(service.destroy(DestroySandbox(sandbox=sandbox)))

        if _last_sre is None:
            # Normal success path — break out of the retry loop.
            break
        # Transient retry: sleep then loop.
        await asyncio.sleep(_backoff)

    # PR-ζ: sandbox fallback — when RunPod retries are exhausted and the host
    # supports local docker + GPU, optionally swap ctx.sandbox_mode to local
    # for the remainder of the run. Opt-in via REPROLAB_RUNPOD_AUTO_FALLBACK=true
    # (default off). The ctx object is not available inside _execute_in_sandbox
    # (it does not receive ctx); fallback is handled in run_experiment which
    # calls this function. See _apply_sandbox_fallback_if_eligible in run_experiment.
    # TODO(PR-ζ-followup): thread ctx into _execute_in_sandbox so fallback can
    # be applied here with the correct emit surface.

    # Contract: paper's code writes $OUTPUT_DIR/metrics.json (host: artifact_root/metrics.json).
    metrics: dict = {}
    metrics_path = artifact_root / METRICS_FILENAME
    if metrics_path.exists():
        try:
            loaded = _json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metrics = loaded
            else:
                logger.warning(
                    "_execute_in_sandbox: %s is valid JSON but not a dict (%s) — "
                    "falling back to {}",
                    metrics_path,
                    type(loaded).__name__,
                )
        except (_json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "_execute_in_sandbox: could not parse %s as JSON (%s) — "
                "falling back to {}",
                metrics_path,
                exc,
            )

    failed_exit_code = next(
        (r.exit_code for r in reversed(results) if r.exit_code not in (None, 0)),
        None,
    )
    last_exit_code = results[-1].exit_code if results else None
    cause_kind = next(
        (
            getattr(r.cause_kind, "value", str(r.cause_kind))
            for r in reversed(results)
            if r.cause_kind is not None
        ),
        None,
    )
    return {
        "success": all(r.succeeded for r in results),
        "metrics": metrics,
        "logs": _cap_logs(_combine_command_output(results)),
        "artifact_dir": str(artifact_root),
        "exit_code": failed_exit_code if failed_exit_code is not None else last_exit_code,
        "cause_kind": cause_kind,
        "resource_limits": resource_limits,
    }


def _persist_experiment_result(
    ctx: "RunContext",
    result: dict,
    *,
    model_id: str = "default",
    eval_env: str = "default",
) -> dict:
    """Append a run_experiment result to ``experiment_runs.jsonl`` and return it.

    A run_experiment result otherwise lives only in the root model's REPL — a
    failed experiment leaves no on-disk trace, so a post-run diagnosis cannot
    see the actual logs (this session it forced re-running ``train.py`` by
    hand). This writes one JSONL line per call — repair-loop retries included —
    and logs a WARNING on failure so the run log surfaces it. Fail-soft:
    observability must never break the run.
    """
    import json
    from datetime import datetime, timezone

    # Auto root-cause: classify the failure shape from error+logs and surface
    # the class + suggested fix on the event AND in the result dict (so the
    # next iteration's repair_context can show the agent exactly what to fix
    # instead of re-diagnosing an opaque traceback).
    try:
        from backend.agents.rlm.failure_classifier import classify_failure
        _fclass, _fsuggest = classify_failure(result)
    except Exception:  # noqa: BLE001 — observability never blocks
        _fclass, _fsuggest = ("unknown", "")
    if _fclass and _fclass != "ok":
        result.setdefault("failure_class", _fclass)
        result.setdefault("suggested_fix", _fsuggest)
    _with_outcome(result, _classify_run_experiment_outcome(result))

    if not result.get("success"):
        logger.warning(
            "run_experiment failed: %s",
            result.get("error") or "(see experiment_runs.jsonl for logs)",
        )
    try:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **result}
        entry.setdefault("model_id", model_id)
        entry.setdefault("eval_env", eval_env)
        path = ctx.project_dir / "experiment_runs.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:  # noqa: BLE001 — observability must never break the run
        logger.exception("run_experiment: failed to persist experiment result")
    # A1: emit experiment_completed so the Lane γ multi-model UI panel fires on
    # real runs (foldExperimentCompleted in use-rlm-run.ts was dead code until
    # this event was wired on the backend side).  Fail-soft via _emit_dashboard_event.
    #
    # 2026-05-24: include ``error`` and a 1.5 KB logs tail on the event so the
    # UI and a human debugger can see *why* a failure happened without having
    # to ssh into the run dir. Previously the SSE event carried only
    # success/metrics/per_model — failures looked identical regardless of
    # whether the agent forgot to write train.py, the container timed out, or
    # the eval lacked an API key.
    _logs_tail = (result.get("logs") or "")[-1500:]
    _emit_dashboard_event(ctx, event_type="experiment_completed", payload={
        "success": result.get("success", False),
        "metrics": result.get("metrics", {}),
        "per_model": result.get("metrics", {}).get("per_model"),  # None if absent — fine
        "error": result.get("error") or None,
        "logs_tail": _logs_tail or None,
        # Rubric-contract violations are forwarded so the UI can show specific
        # actionable gaps (missing keys, off-by-X% numbers, missing variants).
        "contract_violations": result.get("contract_violations") or None,
        "contract_summary": result.get("contract_summary") or None,
        # Auto-classified root cause + suggested fix (Lane K).
        "failure_class": result.get("failure_class") or None,
        "suggested_fix": result.get("suggested_fix") or None,
        "outcome": result.get("outcome") or None,
    })
    # PR-μ Solution C: emit iteration_boundary_recommended warning and feed the
    # forced-iteration policy so it can refuse FINAL_VAR on the two-experiment
    # anti-pattern (root chains two run_experiment calls in one REPL turn).
    _outcome_str = result.get("outcome") or "ok"
    _brief = str(result.get("error") or result.get("failure_class") or "")[:120]
    _emit_iteration_boundary_warning(
        run_dir=ctx.project_dir,
        outcome=_outcome_str,
        brief=_brief,
    )
    # Print REPL banner so the root model sees the recommendation in its output.
    if _outcome_str in {"repairable", "partial_evidence"}:
        print(
            "╔═ ITERATION BOUNDARY RECOMMENDED ═╗\n"
            f"║ run_experiment returned {_outcome_str}; end this iteration\n"
            f"║ so the failure surfaces as fresh next-turn context.\n"
            "╚══════════════════════════════════╝",
            flush=True,
        )
    # Feed the forced-iteration policy's per-iteration experiment tracker.
    # The policy object lives on ctx._forced_iteration_policy (set by run.py).
    _fip = getattr(ctx, "_forced_iteration_policy", None)
    if _fip is not None:
        try:
            _fip.record_run_experiment(_outcome_str)
        except Exception:  # noqa: BLE001 — never crash for observability
            pass
    return result


def _validate_scope_metrics(
    scope_spec: object,
    metrics: dict,
) -> str | None:
    """Validate metrics.json shape against the run's ScopeSpec.

    Returns ``None`` when the shape is acceptable or when no scope is set.
    Returns a non-empty hint string when the metrics dict violates a
    multi-model / multi-dataset scope requirement; callers (run_experiment)
    convert the hint into a fail-soft error dict so the agent's next
    implement_baseline iteration gets it as repair_context.

    Rules:
      - No scope OR empty metrics → pass (None). An empty metrics dict means
        the experiment itself failed at a different layer; not our concern.
      - Multi-model scope → metrics MUST carry a top-level ``per_model`` dict
        keyed by model id, with at least every model from
        ``scope_spec.models`` present.
      - Multi-dataset scope AND multi-model → each per_model entry MUST carry
        a ``per_dataset`` dict keyed by dataset id with every dataset present.
      - Multi-dataset but single-model → metrics MUST carry a top-level
        ``per_dataset`` dict directly (no per_model nesting required).
    """
    if scope_spec is None:
        return None
    if not metrics:
        return None

    # ScopeSpec is duck-typed via Any in RunContext; access through getattr
    # so this helper does not need a hard import of ScopeSpec.
    is_multi_model = getattr(scope_spec, "is_multi_model", False)
    is_multi_dataset = getattr(scope_spec, "is_multi_dataset", False)
    models = list(getattr(scope_spec, "models", []) or [])
    dataset_ids_fn = getattr(scope_spec, "dataset_ids", None)
    datasets = dataset_ids_fn() if callable(dataset_ids_fn) else []

    if is_multi_model:
        per_model = metrics.get("per_model")
        if not isinstance(per_model, dict) or not per_model:
            return (
                f"per_model_required: scope is multi-model {models}. Write "
                f"metrics.json with a top-level per_model dict keyed by model "
                f"id, e.g. {{'per_model': {{'qwen3-1.7b': {{...}}, "
                f"'qwen2.5-3b': {{...}}}}}}."
            )
        missing = [m for m in models if m not in per_model]
        if missing:
            return (
                f"per_model_incomplete: scope requires entries for {models}; "
                f"missing {missing} in metrics.per_model."
            )
        if is_multi_dataset:
            for model_id, model_metrics in per_model.items():
                pd = (model_metrics or {}).get("per_dataset") if isinstance(model_metrics, dict) else None
                if not isinstance(pd, dict) or not pd:
                    return (
                        f"per_dataset_required: scope is multi-dataset {datasets}. "
                        f"Each per_model entry MUST carry a per_dataset dict; "
                        f"model {model_id!r} has none."
                    )
                missing_ds = [d for d in datasets if d not in pd]
                if missing_ds:
                    return (
                        f"per_dataset_incomplete: model {model_id!r} missing "
                        f"datasets {missing_ds} in per_dataset."
                    )
    elif is_multi_dataset:
        # Single-model + multi-dataset: per_dataset at top level (no per_model nesting).
        pd = metrics.get("per_dataset")
        if not isinstance(pd, dict) or not pd:
            return (
                f"per_dataset_required: scope is multi-dataset {datasets}. "
                f"Write metrics.json with a top-level per_dataset dict keyed by "
                f"dataset id."
            )
        missing_ds = [d for d in datasets if d not in pd]
        if missing_ds:
            return (
                f"per_dataset_incomplete: missing datasets {missing_ds} in "
                f"top-level per_dataset."
            )

    return None


# ---------------------------------------------------------------------------
# A2: GPU escalation count persistence helpers
# ---------------------------------------------------------------------------
# The escalations counter was previously local to a single run_experiment call.
# The RLM repair loop may call run_experiment multiple times in one run; each
# fresh invocation got fresh budget, silently exceeding the per-run cap.
# These helpers persist/restore the counter in rlm_state/gpu_escalation_state.json
# so the cap is honoured across the entire run.

def _load_escalation_count(state_dir: "Path") -> int:
    """Return the persisted escalations_used count, defaulting to 0."""
    import json as _json
    from pathlib import Path as _Path

    path = _Path(state_dir) / "gpu_escalation_state.json"
    if not path.exists():
        return 0
    try:
        return int(_json.loads(path.read_text(encoding="utf-8")).get("escalations_used", 0))
    except Exception:  # noqa: BLE001 — missing / corrupt state is not fatal
        return 0


def _persist_escalation_count(state_dir: "Path", count: int) -> None:
    """Atomically write escalations_used to rlm_state/gpu_escalation_state.json."""
    import json as _json
    from pathlib import Path as _Path

    path = _Path(state_dir) / "gpu_escalation_state.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps({"escalations_used": count}), encoding="utf-8")
    tmp.replace(path)


def run_experiment(
    code_path: str | dict,
    env_id: str,
    *,
    model_id: str = "default",
    eval_env: str = "default",
    ctx: "RunContext",
) -> dict:
    """Execute the baseline in a container from prebuilt image `env_id`; return metrics.

    Args:
        code_path: Path to the code directory containing commands.json.
        env_id: Docker image tag (or empty when a Dockerfile-rebuild path applies).
        model_id: Optional tag identifying which model variant this run executes
            (e.g. "qwen3-1.7b"). Defaults to "default". Persisted to
            experiment_runs.jsonl so the scope cross-check (PR B) can verify
            scope.ran against actual evidence. Use one of the model ids from
            ctx.scope_spec.models when the scope is multi-model.
        eval_env: Optional tag identifying which evaluation environment / dataset
            this run targets (e.g. "ALFWorld"). Defaults to "default". Used the
            same way as model_id; together they form composite "model/eval_env"
            evidence ids for multi-model + multi-env papers.

    Commands are read from `code_path/commands.json` (written by
    `implement_baseline`). Before executing, the image is rebuilt from
    `ctx.project_dir/Dockerfile` — the code agent keeps that file in step with
    the baseline's actual imports, while `detect_environment` (which runs before
    any code exists) routinely under-specifies dependencies. The rebuild is
    content-addressed and Docker-cached, so an unchanged Dockerfile is a no-op;
    `env_id` is the fallback used only when no Dockerfile is on disk.
    Async sandbox work is bridged to sync via a worker thread.

    Hardening (WS-H Batch P):
    - A2-H2: guard empty `env_id` → fail-soft error dict.
    - A2-C1: the whole `_execute_in_sandbox` coroutine (N commands × 3600 s
      each) is now bounded by `_timeout_for(ctx, 7200)`; on timeout the thread
      pool's `.result()` raises `TimeoutError` → fail-soft error dict.  The
      sandbox destroy is `asyncio.shield`-ed in `_execute_in_sandbox` so the
      container is cleaned up even when the coroutine is cancelled.
    """
    import asyncio
    import json
    import uuid
    from pathlib import Path

    if isinstance(code_path, dict) and code_path.get("ok") is True:
        code_path = str(code_path.get("code_path") or "")

    # Lane T — guard against `code_path` being the error dict that
    # ``implement_baseline`` returns on failure. This direct primitive guard is
    # deliberately non-persisting: invalid orchestration must not look like a
    # completed experiment and must not emit experiment_completed.
    if not isinstance(code_path, str) or not code_path.strip():
        return _with_outcome({
            "success": False, "metrics": {},
            "failure_class": "contract_guard",
            "source": "contract_guard",
            "error": (
                f"run_experiment: code_path must be a non-empty string path, "
                f"got {type(code_path).__name__} ({str(code_path)[:200]}). "
                f"This usually means implement_baseline returned its error "
                f"dict instead of a directory — check the previous primitive's "
                f"result before passing it to run_experiment."
            ),
            "contract_violations": [{
                "area": "Experiment execution and reproducibility",
                "detail": f"code_path was {type(code_path).__name__!r}, not a str path",
                "hint": (
                    "After implement_baseline, check `isinstance(code_path, str)` "
                    "(or `if not code_path: ...`) before calling run_experiment. "
                    "When implement_baseline errors, propose_improvements + retry "
                    "instead of forwarding the error dict downstream."
                ),
            }],
        }, PrimitiveOutcome.repairable)

    manifest = Path(code_path) / "commands.json"
    commands = json.loads(manifest.read_text()) if manifest.exists() else []
    if not commands:
        return _persist_experiment_result(ctx, {
            "success": False, "metrics": {},
            "error": f"no commands.json at {manifest}"}, model_id=model_id, eval_env=eval_env)

    # Bug B: the experiment must run against an image matching its own code.
    # detect_environment builds the env spec before any code exists, so it
    # routinely under-specifies dependencies (it missed transformers/datasets
    # for the DPO-toxicity paper). The code agent writes the baseline AND keeps
    # ctx.project_dir/Dockerfile in step with the code's real imports — rebuild
    # from THAT Dockerfile. build_environment is content-addressed and
    # Docker-cached, so an unchanged Dockerfile is a near-instant no-op.
    dockerfile_path = ctx.project_dir / "Dockerfile"
    if dockerfile_path.exists():
        build = build_environment(
            {"dockerfile": dockerfile_path.read_text(encoding="utf-8")}, ctx=ctx)
        if not build.get("ok"):
            return _persist_experiment_result(ctx, {
                "success": False, "metrics": {},
                "error": (
                    f"run_experiment: environment rebuild from {dockerfile_path} "
                    f"failed: {build.get('error')}"
                ),
            }, model_id=model_id, eval_env=eval_env)
        # Local-sandbox build returns image_tag="" (skipped=True) because there
        # is no Docker daemon.  Use the sentinel "__local__" so downstream code
        # has a non-empty value while _execute_in_sandbox (which routes on
        # sandbox_mode, not env_id) ignores it safely.
        env_id = build["image_tag"] or ("__local__" if build.get("skipped") else "")

    # A2-H2: guard empty env_id (reachable only when no Dockerfile was on disk
    # to rebuild from AND the build was not skipped for local-sandbox mode).
    # We exempt the local-sandbox path: build_environment deliberately returns
    # image_tag="" there (see build_environment local-short-circuit above), and
    # _execute_in_sandbox routes to LocalProcessBackend via sandbox_mode, making
    # env_id irrelevant.
    _is_local_sb = (
        str(getattr(getattr(ctx, "sandbox_mode", None), "value",
                    getattr(ctx, "sandbox_mode", None) or "")).lower() == "local"
    )
    if not _is_local_sb and (not env_id or not str(env_id).strip()):
        return _persist_experiment_result(ctx, {
            "success": False,
            "metrics": {},
            "error": "env_id empty and no Dockerfile to rebuild — build_environment must succeed first",
        }, model_id=model_id, eval_env=eval_env)

    # Lane F+I pre-flight: catch scope shortcuts (missing variants, dataset
    # subsetting) and surrogate models BEFORE burning pod time. Hard
    # violations block dispatch entirely; soft violations are attached to
    # the result so the agent's next implement_baseline iteration sees them
    # but the run still proceeds. The post-run rubric_contract validator
    # would otherwise catch these only AFTER ~10 minutes of compute.
    try:
        from backend.agents.rlm.pre_flight_validator import validate_code_pre_flight
        from backend.agents.rlm import rubric_contract as _pf_rc
        _pf_arxiv_id = getattr(ctx, "arxiv_id", None)
        _pf_targets = _pf_rc.load_paper_targets(_pf_arxiv_id)
        # ALWAYS run pre-flight when a base image is set, even with no paper_targets,
        # because the torch-redundancy check is a base-image invariant not a
        # paper-specific check.  This blocks the v10-class torch-download
        # failure regardless of which paper is being reproduced.
        if _pf_targets or env_id:
            _pf_violations = validate_code_pre_flight(
                Path(code_path), _pf_targets, arxiv_id=_pf_arxiv_id,
                base_image=env_id,
            )
            _pf_hard = [v for v in _pf_violations if v.severity == "hard"]
            if _pf_hard:
                return _persist_experiment_result(ctx, {
                    "success": False,
                    "metrics": {},
                    "error": (
                        f"pre_flight: {len(_pf_hard)} hard violation(s) — "
                        f"see contract_violations"
                    ),
                    "contract_violations": [v.to_dict() for v in _pf_violations],
                    "pre_flight_blocked": True,
                }, model_id=model_id, eval_env=eval_env)
    except Exception:  # noqa: BLE001 — pre-flight MUST NOT block on its own bug
        logger.exception("run_experiment: pre_flight_validator raised — skipping")

    # PR-μ Solution B: mode-scaled wall-clock cap.
    # resolve_experiment_timeout_s applies REPROLAB_RUN_EXPERIMENT_TIMEOUT_S >
    # EXPERIMENT_TIMEOUT_BY_MODE[execution_mode] > _DEFAULT_EXPERIMENT_TIMEOUT_S,
    # clamped to ctx.remaining_s() when finite.
    timeout = resolve_experiment_timeout_s(ctx)

    # Load cached gpu_plan if present (written by resolve_gpu_requirements).
    from backend.agents.schemas import GpuPlan as _GpuPlan
    from backend.config import get_settings
    from backend.services.runtime.gpu_catalog import CATALOG as _CATALOG

    gpu_plan: "_GpuPlan | None" = None
    plan_path = ctx.project_dir / "rlm_state" / "gpu_plan.json"
    if plan_path.exists():
        try:
            gpu_plan = _GpuPlan(**json.loads(plan_path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            logger.warning("run_experiment: gpu_plan.json present but unreadable; using legacy default")

    _settings = get_settings()
    max_escalations = _settings.dynamic_gpu_max_escalations
    # A2: load the cross-call persisted counter so the per-run cap is honoured
    # even when the RLM repair loop calls run_experiment multiple times.
    escalations = _load_escalation_count(ctx.project_dir / "rlm_state")
    result: dict = {}

    # Escalation loop (spec 2026-05-23 §OOM + §Capacity): on CUDA OOM OR
    # RunPod capacity exhaustion, pop the next SKU from GpuPlan.ladder_remaining,
    # persist the updated plan atomically, emit gpu_escalated, and retry.
    # Capped by max_escalations. Non-OOM/non-capacity failures and success exit
    # immediately. I12: explicit shutdown(wait=False) per iteration.
    while True:
        run_id = f"{ctx.project_id}-{uuid.uuid4().hex[:8]}"
        infra_error_kind: str | None = None
        # I12: explicit shutdown(wait=False) so a wedged worker cannot block cleanup.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            try:
                result = pool.submit(
                    asyncio.run,
                    _execute_in_sandbox(
                        code_path, env_id, commands,
                        project_id=ctx.project_id, run_id=run_id,
                        sandbox_mode=ctx.sandbox_mode,
                        run_budget=ctx.run_budget,
                        gpu_plan=gpu_plan,
                        gpu_mode=getattr(ctx, "gpu_mode", None),
                        gpu_device_ids=tuple(getattr(ctx, "gpu_device_ids", ()) or ()),
                    ),
                ).result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                # _execute_in_sandbox was cancelled before it could return its
                # _combine_command_output(results) — recover whatever the
                # LocalDocker backend streamed to disk for this run_id so the
                # agent (and a human debugger) still sees what failed.
                recovered_logs = ""
                try:
                    artifact_root = Path(code_path) / "outputs" / run_id
                    log_path = artifact_root / "exec.log"
                    if log_path.exists():
                        recovered_logs = log_path.read_text(encoding="utf-8", errors="replace")[-32000:]
                except OSError:
                    pass
                result = {
                    "success": False,
                    "metrics": {},
                    "logs": recovered_logs,
                    "error": (
                        f"run_experiment: timed out after {timeout:.0f} s"
                        if timeout is not None
                        else "run_experiment: timed out (run-budget deadline reached)"
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                # RunPod infrastructure failures bubble up as SandboxRuntimeError
                # tagged with a sentinel prefix from runpod_backend. Treat them
                # like an OOM — advance the ladder, retry. Any other unexpected
                # exception still produces a fail-soft error dict (consistent
                # with the rest of this function never raising).
                exc_msg = str(exc)
                if "RUNPOD_CAPACITY_EXHAUSTED" in exc_msg:
                    infra_error_kind = "runpod_capacity"
                    result = {
                        "success": False, "metrics": {},
                        "error": f"runpod capacity exhausted on {gpu_plan.short_name if gpu_plan else 'unknown'}",
                    }
                elif "RUNPOD_SSH_TIMEOUT" in exc_msg:
                    infra_error_kind = "runpod_ssh_timeout"
                    result = {
                        "success": False, "metrics": {},
                        "error": f"runpod SSH timeout on {gpu_plan.short_name if gpu_plan else 'unknown'}",
                    }
                elif "RUNPOD_TRANSIENT_500" in exc_msg:
                    # Lane 3: unlabelled 500s from RunPod are typically transient
                    # — advance the ladder so the run doesn't dead-end. Bounded
                    # by dynamic_gpu_max_escalations so a request-shape bug
                    # cannot burn the whole catalog.
                    infra_error_kind = "runpod_transient_500"
                    result = {
                        "success": False, "metrics": {},
                        "error": f"run_experiment: {type(exc).__name__}: {exc_msg[:300]}",
                    }
                else:
                    result = {
                        "success": False, "metrics": {},
                        "error": f"run_experiment: {type(exc).__name__}: {exc_msg[:300]}",
                    }
                # PR-ζ: opt-in sandbox fallback after transient retry exhaustion.
                # When REPROLAB_RUNPOD_AUTO_FALLBACK=true and the exception carries
                # _retry_attempts (set by _execute_in_sandbox after exhausting
                # transient retries), check whether local docker + GPU is viable
                # and if so mutate ctx.sandbox_mode for the rest of this run.
                import os as _os_fallback
                if _os_fallback.environ.get("REPROLAB_RUNPOD_AUTO_FALLBACK", "").lower() == "true":
                    _retry_attempts_on_exc = getattr(exc, "_retry_attempts", None)
                    _mode_str_fb = str(getattr(ctx, "sandbox_mode", "") or "").lower()
                    if (
                        _retry_attempts_on_exc
                        and "runpod" in _mode_str_fb
                    ):
                        try:
                            from backend.agents.execution import SandboxMode as _SandboxMode, _docker_reachable
                            from backend.services.runtime.gpu_resolution import host_supports_nvidia_gpu
                            if host_supports_nvidia_gpu() and _docker_reachable():
                                ctx.sandbox_mode = _SandboxMode.docker
                                _emit_dashboard_event(ctx, event_type="sandbox_fallback", payload={
                                    "from": "runpod",
                                    "to": "local",
                                    "reason": "max_retries_exhausted_after_transient_failures",
                                    "attempts": _retry_attempts_on_exc,
                                })
                                logger.warning(
                                    "run_experiment: RunPod transient retries exhausted — "
                                    "auto-fallback to local docker for the rest of this run."
                                )
                        except Exception:  # noqa: BLE001 — fallback must never crash the run
                            logger.exception("run_experiment: sandbox fallback check failed")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        # ---- Escalation gate ----
        # Break immediately on success, or when no plan is loaded (no gpu_plan
        # means legacy mode — no ladder to advance), or cap reached.
        if result.get("success") or gpu_plan is None or escalations >= max_escalations:
            break
        # Detect escalation trigger: CUDA OOM in logs OR RunPod capacity/SSH-timeout.
        stderr_tail = (result.get("logs") or "")[-4096:]
        exit_code = int(result.get("exit_code", 1))  # _execute_in_sandbox may not surface exit_code; default 1
        is_oom = _detect_cuda_oom(exit_code=exit_code, stderr_tail=stderr_tail)
        is_infra = infra_error_kind is not None
        if not is_oom and not is_infra:
            break
        if not gpu_plan.ladder_remaining:
            result = {
                "success": False,
                "metrics": {},
                "error": (
                    f"CUDA OOM on {gpu_plan.short_name} ({gpu_plan.vram_gb} GB); "
                    f"ladder exhausted. Cumulative SKU cost rate: ${gpu_plan.total_usd_per_hr}/hr."
                ),
                "logs": result.get("logs", ""),
            }
            break

        # Advance ladder: find next SKU by short_name.
        next_short = gpu_plan.ladder_remaining[0]
        next_sku = next((s for s in _CATALOG if s.short_name == next_short), None)
        if next_sku is None:
            result = {
                "success": False,
                "metrics": {},
                "error": f"ladder advance failed: short_name={next_short!r} not in catalog",
                "logs": result.get("logs", ""),
            }
            break

        new_plan = gpu_plan.model_copy(update={
            "runpod_id": next_sku.runpod_id,
            "short_name": next_sku.short_name,
            "vram_gb": next_sku.vram_gb,
            "cloud_type": next_sku.cloud_type,
            "sku_usd_per_hr": next_sku.approx_usd_per_hr,
            "total_usd_per_hr": round(next_sku.approx_usd_per_hr * gpu_plan.gpu_count, 4),
            "container_disk_gb": max(50, next_sku.vram_gb),
            "volume_gb": max(20, next_sku.vram_gb // 4),
            "ladder_remaining": gpu_plan.ladder_remaining[1:],
        })
        # Persist atomically + emit escalation event.
        tmp = plan_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new_plan.model_dump(mode="json"), default=str), encoding="utf-8")
        tmp.replace(plan_path)
        _emit_dashboard_event(ctx, event_type="gpu_escalated", payload={
            "from_sku": gpu_plan.short_name,
            "to_sku": new_plan.short_name,
            "escalation_index": escalations + 1,
            "reason": infra_error_kind if is_infra else "cuda_oom",
        })
        gpu_plan = new_plan
        escalations += 1
        # A2: persist the updated count so subsequent run_experiment calls in
        # the same run start from the correct escalation budget offset.
        _persist_escalation_count(ctx.project_dir / "rlm_state", escalations)

    # Scope-shape validation (PR B): if scope is multi-model / multi-dataset,
    # require metrics.json to carry the expected per_model / per_dataset
    # structure. A successful run with the wrong shape is a fail-soft error
    # so the agent's next implement_baseline gets it as repair_context.
    if result.get("success") and result.get("metrics"):
        hint = _validate_scope_metrics(getattr(ctx, "scope_spec", None), result["metrics"])
        if hint is not None:
            result = {
                **result,
                "success": False,
                "error": hint,
                "scope_shape_violation": True,
            }

    # Rubric-contract validation: post-run diff of metrics + artifacts against
    # the paper's declared docs/papers/<arxiv_id>.yaml paper_targets section.
    # Surfaces concrete, actionable violations the agent can fix on its next
    # implement_baseline iteration.  Covers 4 of 6 rubric areas
    # (Data fidelity, Experiment execution, Eval protocol, Result match,
    # Artifact completeness) deterministically — no LLM call needed.
    try:
        from backend.agents.rlm import rubric_contract as _rc
        _arxiv_id = getattr(ctx, "arxiv_id", None)
        _paper_targets = _rc.load_paper_targets(_arxiv_id)
        if _paper_targets:
            _artifact_root = Path(code_path) / "outputs" / run_id
            _contract_report = _rc.validate(
                metrics=result.get("metrics") or {},
                artifact_root=_artifact_root,
                paper_targets=_paper_targets,
            )
            if _contract_report.violations:
                # Attach violations to result so the root model can pass them
                # back to implement_baseline as repair_context.  Do not flip
                # success to False on its own — a successful experiment with
                # contract violations is still useful; it just gets repaired
                # next iteration.
                result = {
                    **result,
                    "contract_violations": [v.to_dict() for v in _contract_report.violations],
                    "contract_summary": _contract_report.summary,
                }
    except Exception:  # noqa: BLE001 — observability must never block the run
        logger.exception("run_experiment: rubric_contract.validate raised — skipping")

    # θ: metrics_shape post-run check. When the planning contract declared an
    # explicit metrics_shape, validate that every declared json_path actually
    # exists in the emitted metrics.json. This is the authoritative guard that
    # replaces fingerprint guesswork — the agent declared the paths, so any
    # deviation is an unambiguous contract violation, not a shape mismatch.
    # Runs even on failed experiments (partial results are still checked).
    try:
        _contract_obj = getattr(ctx, "reproduction_contract", None)
        _ctx_metrics_shape = []
        if _contract_obj is not None:
            _ctx_metrics_shape = list(getattr(_contract_obj, "metrics_shape", None) or [])
        if _ctx_metrics_shape:
            from backend.agents.rlm.rubric_guard import (
                RubricGuardFailure as _RGF,
                assert_metrics_schema as _assert_ms,
            )
            _shape_dicts = [
                (mp.model_dump() if hasattr(mp, "model_dump") else dict(mp))
                for mp in _ctx_metrics_shape
                if mp is not None
            ]
            try:
                _assert_ms(
                    result.get("metrics") or {},
                    required_keys=[],       # metrics_shape takes priority
                    metrics_shape=_shape_dicts,
                )
            except _RGF as _rg_err:
                import json as _json_ms
                _rg_detail = _json_ms.loads(str(_rg_err))
                _ms_violations = _rg_detail.get("missing_keys", [])
                if _ms_violations:
                    _existing = list(result.get("contract_violations") or [])
                    _existing += [
                        {
                            "area": "Result match and metric key compliance",
                            "detail": f"metrics_shape contract violation: {v}",
                            "hint": (
                                "The plan_reproduction contract declared this metric path. "
                                "Ensure train.py writes metrics.json with exactly this "
                                "dotted path. Use the METRICS CONTRACT section in your "
                                "guidance as the authoritative reference."
                            ),
                        }
                        for v in _ms_violations
                    ]
                    result = {**result, "contract_violations": _existing}
    except Exception:  # noqa: BLE001 — observability must never block the run
        logger.exception("run_experiment: metrics_shape post-run check failed — skipping")

    return _persist_experiment_result(ctx, result, model_id=model_id, eval_env=eval_env)


def _rubric_areas(rubric: dict, leaf_scores_list: list[dict]) -> list[dict]:
    """Derive a flat ``areas`` list from the top-level rubric sub_tasks.

    Each top-level sub_task becomes one area entry:
      {"name": <requirements text, truncated>, "score": <rolled-up float>,
       "weight": <raw weight int/float>}

    This gives the root model named, scored areas it can include verbatim in
    the final report instead of fabricating blank placeholders.  Fail-soft:
    if the rubric has no sub_tasks (e.g. a flat generated rubric) the list
    is empty and the caller continues normally.
    """
    from backend.evals.paperbench.leaf_scorer import roll_up

    sub_tasks = [c for c in (rubric.get("sub_tasks") or []) if isinstance(c, dict)]
    if not sub_tasks:
        return []

    # Build a {leaf_id: score} map from the leaf_scores list for roll_up.
    # Skip entries with score=None — these are data-unavailable leaves (PR-κ)
    # that were explicitly not scored. float(None) raises TypeError.
    leaf_score_map: dict[str, float] = {
        str(e["id"]): float(e["score"])
        for e in leaf_scores_list
        if isinstance(e, dict) and e.get("id") and e.get("score") is not None
    }

    areas: list[dict] = []
    for i, task in enumerate(sub_tasks):
        name = str(task.get("requirements") or "")[:120]
        if not name:
            name = f"Area {i + 1}"
        score = _clamp01(roll_up(task, leaf_score_map))
        weight = task.get("weight")
        areas.append({"area": name, "score": score, "weight": weight})
    return areas


# ---------------------------------------------------------------------------
# β3: Compute-adjusted scoring helpers
# ---------------------------------------------------------------------------


def score_with_floor(
    *,
    actual: float,
    paper_target: float,
    floor: float,
    direction: str,
) -> float:
    """Compute compute-adjusted credit for a result_match leaf.

    Returns 1.0 if the actual metric meets the floor (or beats the paper),
    linearly interpolated partial credit between floor and paper_target,
    or 0.0 when the actual is worse than the floor.

    Direction-aware: "higher" treats larger values as better (accuracy);
    "lower" treats smaller values as better (loss / error / cost).

    When floor == paper_target the scoring degenerates to a step function:
    actual >= floor → 1.0, actual < floor → 0.0.
    """
    if direction == "higher":
        if actual >= paper_target:
            return 1.0
        denom = paper_target - floor
        if denom <= 0:
            # floor >= paper_target (degenerate or floor==target) — step function
            return 1.0 if actual >= floor else 0.0
        if actual < floor:
            return 0.0
        # floor <= actual < paper_target → linear interp (actual==floor gives 0/denom = 0,
        # but the spec says floor == full credit, so return 1.0 when actual == floor exactly)
        if actual == floor:
            return 1.0
        return max(0.0, min(1.0, (actual - floor) / denom))

    if direction == "lower":
        if actual <= paper_target:
            return 1.0
        denom = floor - paper_target
        if denom <= 0:
            # floor <= paper_target (degenerate or floor==target) — step function
            return 1.0 if actual <= floor else 0.0
        if actual > floor:
            return 0.0
        # paper_target < actual <= floor → full credit at floor
        if actual == floor:
            return 1.0
        return max(0.0, min(1.0, (floor - actual) / denom))

    raise ValueError(
        f"score_with_floor: direction must be 'higher' or 'lower', got {direction!r}"
    )


def _apply_compute_adjusted_scoring(
    rubric_result: dict,
    compute_scope: "ComputeScope | None",
    actual_metrics: dict,
) -> dict:
    """Augment rubric_result with compute_adjusted_score per area + overall.

    When compute_scope is None or not clipped, compute_adjusted_score mirrors
    the raw score (always-emit semantic — UI never sees null).

    When compute_scope is present and clipped, result_match leaves whose metric
    is found in compute_scope.metric_floors get re-scored against the floor.
    Other areas are copied through unchanged.

    Mutates and returns rubric_result (a fresh copy is NOT made — callers
    must pass a copy if they want to preserve the original).
    """
    # Always-emit path: no clipping declared or no floors → copy raw values.
    if (
        compute_scope is None
        or not compute_scope.is_clipped
        or not compute_scope.metric_floors
    ):
        for area in rubric_result.get("areas", []):
            area["compute_adjusted_score"] = area.get("score", 0.0)
        rubric_result["compute_adjusted_score"] = rubric_result.get("overall_score", 0.0)
        rubric_result["compute_scope"] = compute_scope.model_dump() if compute_scope else None
        return rubric_result

    # Clipped path: re-score result_match leaves against floors.
    floors_by_metric: dict[str, "MetricFloor"] = {
        mf.metric: mf for mf in compute_scope.metric_floors
    }

    overall_adjusted = 0.0
    total_weight = 0.0

    for area in rubric_result.get("areas", []):
        area_name = area.get("area", "").lower()
        is_result_match = "result match" in area_name or "result_match" in area_name
        area_weight = float(area.get("weight") or 0.0)

        if is_result_match:
            leaf_scores_adj: list[float] = []
            for leaf in area.get("leaves", []):
                metric_name = leaf.get("metric")
                actual = actual_metrics.get(metric_name) if metric_name else None
                floor_def = floors_by_metric.get(metric_name) if metric_name else None
                if actual is not None and floor_def is not None:
                    adj = score_with_floor(
                        actual=float(actual),
                        paper_target=float(floor_def.paper_target),
                        floor=float(floor_def.floor),
                        direction=floor_def.direction,
                    )
                    leaf["compute_adjusted_score"] = adj
                    leaf_scores_adj.append(adj)
                else:
                    # No floor mapped for this leaf → fall back to raw score.
                    raw = float(leaf.get("score", 0.0) or 0.0)
                    leaf["compute_adjusted_score"] = raw
                    leaf_scores_adj.append(raw)
            area_adj = (
                sum(leaf_scores_adj) / len(leaf_scores_adj) if leaf_scores_adj
                else float(area.get("score", 0.0) or 0.0)
            )
            area["compute_adjusted_score"] = area_adj
        else:
            area_adj = float(area.get("score", 0.0) or 0.0)
            area["compute_adjusted_score"] = area_adj

        overall_adjusted += area_adj * area_weight
        total_weight += area_weight

    # Normalize if weights don't sum to 1 (some rubrics use integer weights).
    if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
        overall_adjusted = overall_adjusted / total_weight

    rubric_result["compute_adjusted_score"] = _clamp01(overall_adjusted)
    rubric_result["compute_scope"] = compute_scope.model_dump()
    return rubric_result


# Import type alias for type checkers only — avoids circular import at runtime.
if False:  # TYPE_CHECKING
    from backend.agents.schemas import ComputeScope, MetricFloor  # noqa: F401


def verify_against_rubric(results: dict, rubric: dict, *, ctx: "RunContext") -> dict:
    """Score the run against `rubric` using the authoritative PaperBench leaf scorer.

    Evidence is gathered from the run directory by `score_reproduction`, so the
    in-loop score matches the post-run leaf score exactly. `results` is kept in
    the signature for registry/root call convention compatibility; the leaf scorer
    gathers its own evidence from `ctx.project_dir`.

    Returns:
        overall_score (float), meets_target (bool), target_score (float),
        leaf_count (int), graded (int), rubric_source (str),
        areas (list of {name, score, weight} per top-level sub_task),
        weak_leaves (list of up to 8 lowest-scoring leaf dicts), leaf_scores (list).

    Fail-soft (A2-H3 / D3 pattern): any exception returns an error dict.
    """
    if not rubric or not isinstance(rubric, dict):
        return _with_outcome({
            "success": False,
            "error": "verify_against_rubric: rubric must be a non-empty dict",
        }, PrimitiveOutcome.repairable)

    # Cache key: rubric + metrics + a content hash of the code dir + the
    # paper's metrics.json bytes if present. The leaf scorer reads code +
    # metrics from disk, so the cache MUST invalidate when either changes.
    from backend.agents.rlm import primitive_cache as _cache
    import hashlib as _hashlib
    _evidence_hash_bits: list[str] = []
    try:
        _code_dir = ctx.project_dir / "code"
        if _code_dir.exists():
            for p in sorted(_code_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    try:
                        _evidence_hash_bits.append(_hashlib.sha256(p.read_bytes()).hexdigest()[:16])
                    except OSError:
                        pass
        _metrics_file = ctx.project_dir / "code" / "metrics.json"
        if _metrics_file.exists():
            _evidence_hash_bits.append(_hashlib.sha256(_metrics_file.read_bytes()).hexdigest()[:16])
    except Exception:  # noqa: BLE001
        _evidence_hash_bits = []
    _payload = {
        "rubric_hash": _hashlib.sha256(
            __import__("json").dumps(rubric, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16],
        "results_hash": _hashlib.sha256(
            __import__("json").dumps(results, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16],
        "evidence_hash": _hashlib.sha256(
            "".join(_evidence_hash_bits).encode("utf-8")
        ).hexdigest()[:16],
    }
    _cached = _cache.maybe_get(ctx.project_dir, "verify_against_rubric", payload=_payload)
    if _cached is not None:
        return _with_outcome(_cached, PrimitiveOutcome.ok)

    try:
        from backend.evals.paperbench.leaf_scorer import score_reproduction

        # β1 — tri-state partial-success contract.
        #
        # C2b in-loop wiring: derive `degraded` from the `results` dict we
        # already have. The leaf scorer's auto-detection reads
        # final_report.json, but in-loop (called from the improvement loop
        # before _finalize) that file has not been written yet, so
        # auto-detection returns False and the cap would not fire. Pass it
        # explicitly so the in-loop optimization signal matches what the
        # post-run authoritative score will become.
        #
        # Tri-state logic (β1):
        #   success=True                   → ok      → degraded=False (full grading)
        #   success=False, metrics non-empty → partial → degraded=False (partial evidence
        #       is real evidence; leaf scorer grades against captured metrics)
        #   success=False, metrics empty   → failed  → degraded=True  (cap at ceiling)
        #
        # This ensures a VAE that crashes after capturing 15 real metrics is
        # graded against those metrics, not capped at 0.35 across every leaf.
        has_experiment_result = "success" in results or "metrics" in results
        metrics_present = bool(results.get("metrics") or {})
        degraded = has_experiment_result and (
            (results.get("success") is False) and (not metrics_present)
        )
        scored = score_reproduction(
            rubric_tree=rubric,
            run_dir=ctx.project_dir,
            llm_client=ctx.llm_client,
            rubric_source=str(rubric.get("source") or "paperbench_bundle"),
            degraded=degraded,
            # Paper-hint invariant gate (2026-05-29): thread invariants from
            # RunContext so the deterministic regex gate fires in-loop.
            invariants=list(getattr(ctx, "paper_hint_invariants", None) or []),
        )
        # Honesty guard: if score_reproduction handed back zero successfully-graded
        # leaves for a non-degraded run, the LLM grader's output was unparseable
        # on every batch. That is a real verification failure — never a "scored
        # 0.0" success. Degraded metric-less runs are the exception: they are
        # deterministically scored at zero without calling the LLM grader.
        # Pinned by tests/rlm/test_binding.py::test_verify_against_rubric_emits_nothing_on_failure.
        graded = int(scored.get("graded", 0) or 0)
        leaf_count = int(scored.get("leaf_count", 0) or 0)
        if leaf_count > 0 and graded == 0 and not scored.get("degraded"):
            return _with_outcome({
                "success": False,
                "error": (
                    f"verify_against_rubric: leaf scorer graded 0/{leaf_count} leaves — "
                    f"LLM grader output was unparseable on every batch; no honest score available"
                ),
            }, PrimitiveOutcome.repairable)
        overall_score = _clamp01(scored["overall_score"])
        target = _clamp01(rubric.get("target_score", 0.6))
        meets_target = overall_score >= target

        leaf_scores = scored.get("leaf_scores", [])
        # Up to 8 lowest-scoring leaves (conservative grader — 0.0 means no evidence).
        # Exclude score=None entries (PR-κ data-unavailable leaves — not "weak", just absent).
        weak_leaves = sorted(
            [e for e in leaf_scores if isinstance(e, dict) and e.get("score") is not None],
            key=lambda e: float(e.get("score", 0.0)),
        )[:8]

        result = {
            "overall_score": overall_score,
            "meets_target": meets_target,
            "target_score": target,
            "leaf_count": scored.get("leaf_count", 0),
            "graded": scored.get("graded", 0),
            "rubric_source": scored.get("rubric_source", "paperbench_bundle"),
            "degraded": degraded,
            # β2: coverage_pct from score_reproduction (graded / total leaves).
            # 0.0 on fully-degraded runs; 0.0–1.0 on partial/full runs.
            "coverage_pct": float(scored.get("coverage_pct", 1.0) or 1.0),
            "areas": _rubric_areas(rubric, leaf_scores),
            "weak_leaves": [
                {"id": e.get("id", ""), "score": e.get("score", 0.0),
                 "justification": e.get("justification", "")}
                for e in weak_leaves
            ],
            "leaf_scores": leaf_scores,
            # Paper-hint invariant gate (2026-05-29): surface per-invariant
            # pass/fail so the root model can see which invariants tripped and
            # target repairs on the next iteration.
            "invariant_results": scored.get("invariant_results", []),
            "invariant_gate_applied": bool(scored.get("invariant_gate_applied", False)),
        }
        # Lane P phase B (codex review 2026-05-25): when metrics.json carries
        # an `experiments` dict with per-experiment {status, reason_class},
        # compute the scope-adjusted rubric and attach it alongside the raw
        # score. The UI / leaderboard can then surface both — uncontrollable
        # failures (HF URI deprecation, runpod 500, etc.) don't tank the
        # score, while controllable agent-code bugs still do.
        try:
            experiments = (
                results.get("experiments")
                or (results.get("metrics") or {}).get("experiments")
                or {}
            )
            if isinstance(experiments, dict) and experiments:
                from backend.agents.rlm.scope_classifier import (
                    compute_scope_adjusted_rubric,
                )
                # Map leaf_scores into the shape compute_scope_adjusted_rubric expects.
                # Leaves carry their own experiment binding when the rubric author
                # populated it; otherwise treat the leaf as paper-wide (always-in).
                _leaf_map = {
                    str(e.get("id", f"leaf_{i}")): {
                        "score": float(e.get("score", 0.0) or 0.0),
                        "weight": float(e.get("weight", 1.0) or 1.0),
                        "experiment": e.get("experiment"),
                    }
                    for i, e in enumerate(leaf_scores)
                    if isinstance(e, dict)
                }
                sar = compute_scope_adjusted_rubric(
                    experiments=experiments,
                    leaf_scores=_leaf_map,
                    target_score=target,
                )
                result["scope_adjusted"] = {
                    "overall_score": sar.overall_score,
                    "target_score": sar.target_score,
                    "meets_target": sar.meets_target,
                    "coverage": sar.coverage,
                    "insufficient_coverage": sar.insufficient_coverage,
                    "notes": sar.notes,
                    "judgements": [
                        {
                            "effective_status": j.effective_status,
                            "credit": j.credit,
                            "in_denominator": j.in_denominator,
                            "reason_class": j.reason_class,
                            "notes": j.notes,
                        }
                        for j in sar.judgements
                    ],
                }
        except Exception:  # noqa: BLE001 — scope adjustment is augmenting, not mandatory
            logger.exception("verify_against_rubric: scope_adjusted computation failed")

        # β3: compute-adjusted scoring against per-metric floors.
        # Reads compute_scope from the run context's reproduction contract (if
        # available). Falls back gracefully: when compute_scope is None or
        # not clipped, adjusted == raw (always-emit semantic).
        try:
            from backend.agents.schemas import ComputeScope as _ComputeScope
            _compute_scope: "_ComputeScope | None" = None
            _contract = getattr(ctx, "reproduction_contract", None)
            if _contract is not None:
                _cs = getattr(_contract, "compute_scope", None)
                if isinstance(_cs, _ComputeScope):
                    _compute_scope = _cs
            _actual_metrics = dict(results.get("metrics") or {})
            result = _apply_compute_adjusted_scoring(result, _compute_scope, _actual_metrics)
        except Exception:  # noqa: BLE001 — compute-adjusted is augmenting, not mandatory
            logger.exception("verify_against_rubric: _apply_compute_adjusted_scoring failed")
            # Ensure always-emit fields are present even on failure.
            result.setdefault("compute_adjusted_score", result.get("overall_score", 0.0))
            result.setdefault("compute_scope", None)

        result = _with_outcome(result, PrimitiveOutcome.ok)
        _cache.put(ctx.project_dir, "verify_against_rubric", payload=_payload, result=result)
        return result
    except Exception as exc:  # noqa: BLE001 — fail-soft (A2-H3 / D3 pattern)
        return _with_outcome({
            "success": False,
            "error": f"verify_against_rubric: {type(exc).__name__}: {exc}",
        }, PrimitiveOutcome.repairable)


def propose_improvements(current_results: dict, rubric_scores: dict,
                         k: int | None = None, *, ctx: "RunContext") -> list[dict]:
    """Propose paper-specific improvement hypotheses (variable-length, free-form tags).

    Reuses the `improvement-orchestrator` prompt — no fixed taxonomy. Each item
    is an `ImprovementHypothesis` dict; malformed items are dropped fail-soft.

    Hardening (A2-H1): `k` may arrive as a string (e.g. `"3"`) from
    LLM-generated REPL code; coerce with `int()`, fall back to default 3.
    """
    import json

    from backend.agents.prompts.improvement import IMPROVEMENT_ORCHESTRATOR_PROMPT
    from backend.agents.schemas import ImprovementHypothesis

    # A2-H1: coerce k — LLM REPL code may pass a string like "3".
    if k is not None:
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = None
    # clamp: k <= 0 would empty the slice
    target = max(1, k) if k is not None else 3
    user = (
        "current_results:\n" + json.dumps(current_results, indent=2, default=str)
        + "\n\nrubric_scores (prioritise lifting the weakest areas):\n"
        + json.dumps(rubric_scores, indent=2, default=str)
        + f"\n\nPropose up to {target} improvement hypotheses. Return a JSON "
          'object {"hypotheses": [ImprovementHypothesis, ...]}. Each hypothesis '
          "carries a free-form `category` tag of your choosing."
    )
    try:
        raw = ctx.llm_client.complete(system=IMPROVEMENT_ORCHESTRATOR_PROMPT, user=user)
        items = _extract_json(raw).get("hypotheses", [])
    except Exception as exc:  # noqa: BLE001 — fail-soft (D3 / T11 / review I3)
        return [_with_outcome({
            "success": False,
            "error": f"propose_improvements: {type(exc).__name__}: {exc}",
        }, PrimitiveOutcome.repairable)]

    out: list[dict] = []
    for item in items:
        try:
            if not isinstance(item, dict):
                continue
            required_text = {
                "path_id": item.get("path_id"),
                "hypothesis": item.get("hypothesis"),
                "rationale": item.get("rationale"),
                "category": item.get("category"),
            }
            if any(not isinstance(value, str) or not value.strip() for value in required_text.values()):
                continue
            # Ensure title is never empty — derive a fallback from hypothesis text
            # so candidate_proposed events always carry a human-readable label.
            if not item.get("title"):
                hypothesis_text = (item.get("hypothesis") or "").strip()
                item["title"] = (
                    hypothesis_text.split(".")[0][:80]
                    or hypothesis_text[:80]
                    or item.get("path_id", "candidate")
                )
            out.append(ImprovementHypothesis(**item).model_dump())
        except Exception:
            continue  # fail-soft: skip a malformed hypothesis
    return out[:target]


_VALID_OUTCOMES = {"running", "promoted", "marginal", "failed", "skipped", "declined"}

# 2026-05-23: outcome aliases (case-insensitive). The model often passes
# natural-language variants — "success", "passed", "ok" instead of "promoted";
# "fail", "error", "broken" instead of "failed"; etc. Mapping them here lets
# the validator accept honest signals instead of rejecting them and emitting
# zero outcome events (the C5 regression: 4 record_candidate_outcome calls
# all rejected → 0 outcomes on the wire → UI couldn't see the model's intent).
_OUTCOME_ALIASES = {
    # → promoted
    "success": "promoted", "successful": "promoted", "passed": "promoted",
    "pass": "promoted", "ok": "promoted", "complete": "promoted",
    "completed": "promoted", "improved": "promoted", "better": "promoted",
    "promote": "promoted",
    # → failed
    "fail": "failed", "error": "failed", "broken": "failed", "crashed": "failed",
    "exception": "failed", "regression": "failed", "regressed": "failed",
    "worse": "failed",
    # → marginal
    "partial": "marginal", "mixed": "marginal", "inconclusive": "marginal",
    "neutral": "marginal", "tied": "marginal", "no_change": "marginal",
    "unchanged": "marginal",
    # → declined
    "decline": "declined", "skip": "skipped", "rejected": "declined",
    "abandon": "declined", "abandoned": "declined", "deferred": "skipped",
    # → running
    "run": "running", "in_progress": "running", "started": "running",
    "trying": "running",
}


def _canonicalize_outcome(outcome: object) -> str | None:
    """Map a model-supplied outcome string to one of _VALID_OUTCOMES.

    Returns the canonical value on success, None if no plausible mapping.
    Case-insensitive; tolerates leading/trailing whitespace; accepts aliases.
    """
    if outcome is None:
        return None
    s = str(outcome).strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return None
    if s in _VALID_OUTCOMES:
        return s
    return _OUTCOME_ALIASES.get(s)


def record_candidate_outcome(
    candidate_id: str,
    outcome: str,
    parent_id: str | None = None,
    *,
    ctx: "RunContext",
) -> dict:
    """Record the root model's outcome decision for a candidate (Option B, handoff §5).

    Near-no-op computation — the primitive exists purely so its ``wrap_primitive``
    wrapper can emit a ``candidate_outcome`` SSE event that reflects the root's
    actual decision (not a backend-inferred approximation).  The root calls this
    after evaluating each improvement candidate:

        outcome = "promoted" if score > rubric_target else "failed"
        record_candidate_outcome(candidate_id=cid, outcome=outcome)

    ``candidate_id`` MUST be one of the ``id`` values returned by the most
    recent ``propose_improvements`` call (e.g. ``"path_1"``, ``"path_2"``).
    Passing ``None``, empty string, or the literal ``"None"`` returns an error
    dict — silently coercing those to ``str(None)`` corrupts the SSE stream
    (the 2026-05-23 prj_6b9acbfd8afcd789 bug: every outcome event had
    ``candidate_id="None"`` so the UI could not match outcomes to candidates).

    Valid outcomes: ``"running"``, ``"promoted"``, ``"marginal"``, ``"failed"``,
    ``"skipped"``, ``"declined"``.  ``parent_id`` is the node this candidate
    branches from (passed through to the ``candidate_outcome`` event's
    ``parent_id`` field so the UI can build the exploration tree).

    Returns a plain ``{"success": True, ...}`` dict on the happy path. On bad
    input, returns ``{"success": False, "error": "<message>", ...}`` so
    ``wrap_primitive`` skips the SSE event emission and the UI does not get
    a poisoned candidate_outcome.
    """
    # Defensive input validation — surface model errors as errors instead of
    # propagating None / "None" / "" downstream where it silently corrupts SSE.
    cid = candidate_id
    cid_str = str(cid) if cid is not None else ""
    if cid is None or cid_str.strip() in {"", "None", "null"}:
        return {
            "success": False,
            "outcome": PrimitiveOutcome.repairable.value,
            "error": "candidate_id missing — pass the most recent proposed candidate",
            "candidate_id": cid_str,
            "candidate_outcome": str(outcome) if outcome is not None else "",
            "parent_id": parent_id,
        }
    # 2026-05-23: canonicalize outcome instead of strict-reject. The model
    # often passes natural synonyms ("success", "fail", "partial"); rejecting
    # them silently drops 100% of outcome events for that run (the C5 bug:
    # 4 calls, 4 rejected, 0 outcome events emitted). Accept aliases via the
    # case-insensitive map; only reject if the value is truly empty / None.
    canonical = _canonicalize_outcome(outcome)
    if canonical is None:
        # Fall back gracefully: accept the model's literal value, log it, emit
        # it as-is. Better to surface an unknown outcome string in the UI
        # (which can render it as a gray pill) than to drop the event entirely.
        # The structural data (which candidate, what iteration) is still useful.
        canonical = str(outcome).strip() if outcome is not None else "unknown"
        if not canonical:
            canonical = "unknown"
    return {
        "success": True,
        "candidate_id": cid_str,
        "outcome": canonical,
        "parent_id": parent_id,
    }


def check_user_messages(*, ctx: "RunContext") -> list[dict]:
    """Return new user messages since the last call; advance the read cursor.

    Reads ``runs/<id>/user_messages.jsonl`` and returns only the lines whose
    index is >= the cursor stored in ``runs/<id>/_user_message_cursor.json``.
    The cursor is atomically updated so repeated calls return only new messages.

    Returns a list of ``{role, content, ts}`` dicts (empty when no new messages).
    Emit instrumentation follows the standard `primitive_call` pattern via the
    `wrap_primitive` wrapper in `binding.py`.
    """
    import json as _json
    from datetime import datetime, timezone

    messages_path = ctx.project_dir / "user_messages.jsonl"
    cursor_path = ctx.project_dir / "_user_message_cursor.json"

    # Read cursor (default 0 = read from beginning)
    cursor = 0
    if cursor_path.exists():
        try:
            data = _json.loads(cursor_path.read_text(encoding="utf-8"))
            cursor = int(data.get("offset", 0))
        except Exception:  # noqa: BLE001 — fail-soft; a bad cursor resets to 0
            cursor = 0

    if not messages_path.exists():
        return []

    new_messages: list[dict] = []
    lines = []
    try:
        lines = messages_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in lines[cursor:]:
        line = line.strip()
        if not line:
            continue
        try:
            entry = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if entry.get("role") == "user":
            new_messages.append({
                "role": entry["role"],
                "content": entry.get("content", ""),
                "ts": entry.get("ts", ""),
            })

    # Advance cursor to total line count (includes blank lines, safe)
    new_cursor = len(lines)
    # Atomic write via temp + replace
    import os as _os
    tmp = cursor_path.with_suffix(".json.tmp")
    tmp.write_text(
        _json.dumps({"offset": new_cursor, "updated": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    _os.replace(tmp, cursor_path)

    return new_messages


def respond_to_user(message: str, *, ctx: "RunContext") -> dict:
    """Append an assistant reply to user_messages.jsonl and emit a dashboard event.

    The reply is appended as ``{role:"assistant", content:message, ts:iso8601}``.
    A ``user_message_response`` event is also written to ``dashboard_events.jsonl``
    so the SSE stream surfaces it to the frontend in real time.

    Returns ``{"sent": true}`` on success; ``{"sent": false, "error": ...}`` on
    validation failure (empty message). Never raises — fail-soft (D3 pattern).
    """
    import json as _json
    from datetime import datetime, timezone

    if not message or not str(message).strip():
        return _with_outcome({
            "sent": False,
            "error": "respond_to_user: message must be non-empty",
        }, PrimitiveOutcome.repairable)

    ts = datetime.now(timezone.utc).isoformat()
    assistant_entry = {"role": "assistant", "content": message, "ts": ts}
    dashboard_entry = {
        "event": "user_message_response",
        "timestamp": ts,
        "role": "assistant",
        "content": message,
    }

    messages_path = ctx.project_dir / "user_messages.jsonl"
    dashboard_path = ctx.project_dir / "dashboard_events.jsonl"

    try:
        with messages_path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(assistant_entry, default=str) + "\n")
        with dashboard_path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(dashboard_entry, default=str) + "\n")
    except OSError as exc:
        return _with_outcome({
            "sent": False,
            "error": f"respond_to_user: IO error: {exc}",
        }, PrimitiveOutcome.repairable)

    return _with_outcome({"sent": True}, PrimitiveOutcome.ok)


# Module-level monotonic counter for heartbeat events (thread-safe via GIL for
# int increment; each run process is single-worker so collisions are impossible).
_heartbeat_counter: int = 0


def recommend_next_tool(situation: str, *, ctx: "RunContext") -> dict:
    """Reflexion-lite: get a structured recommendation for the next tool.

    The root calls this when uncertain about how to proceed. The recommendation
    comes from a single llm_query call (no recursion, bounded cost).
    """
    prompt = (
        "You are advising an RLM root model on which tool to use next.\n\n"
        f"CURRENT SITUATION:\n{situation}\n\n"
        "AVAILABLE TOOLS:\n"
        "* rlm_query(slice, question) — spawn a sub-RLM to answer a focused question on a paper slice >8K chars\n"
        "* llm_query(prompt) — single LLM call, no recursion, for simple summarization or transformation\n"
        "* understand_section(text_slice) — extract datasets/metrics/recipe/hardware/ambiguities from a slice (generic schema)\n"
        "* extract_hyperparameters(text_slice) — extract optimizer/lr/batch_size/epochs from a slice\n"
        "* detect_environment(method_spec) — derive Dockerfile + framework + packages from a method spec\n"
        "* build_environment(env_spec) — build the Docker image\n"
        "* plan_reproduction(method_spec, env_spec) — derive smoke-test + eval plan\n"
        "* implement_baseline(plan) — invoke the coding sub-agent to write the baseline\n"
        "* run_experiment(code_path, env_id) — execute the baseline\n"
        "* verify_against_rubric(results, rubric) — score against the rubric\n"
        "* propose_improvements(results, scores, k) — derive k improvement hypotheses\n\n"
        "Reply with JSON:\n"
        "{\"tool\": \"<one of the above>\", \"reason\": \"<one sentence>\", \"alternatives\": [\"<other tool>\"]}\n\n"
        "Be brief. Prefer rlm_query when synthesizing >10K-char passages with a focused question."
    )
    try:
        raw = ctx.llm_client.complete(system="", user=prompt)
        parsed = _extract_json(raw)
        return {
            "tool": str(parsed.get("tool", "")),
            "reason": str(parsed.get("reason", "")),
            "alternatives": [str(a) for a in (parsed.get("alternatives") or [])],
            "outcome": PrimitiveOutcome.ok.value,
        }
    except Exception as exc:  # noqa: BLE001 — advisory; never break the root run
        return _with_outcome({
            "tool": "",
            "reason": f"recommend_next_tool failed: {type(exc).__name__}",
            "alternatives": [],
        }, PrimitiveOutcome.repairable)


def heartbeat(note: str = "", *, ctx: "RunContext") -> dict:
    """Emit a liveness signal so the operator knows the root is still alive.

    Near-no-op computation — the only side effect is appending one
    ``iteration_heartbeat`` JSON line to ``dashboard_events.jsonl`` and
    incrementing a module-level monotonic counter.

    The ``wrap_primitive`` wrapper in ``binding.py`` also emits a standard
    ``primitive_call`` event (primitive="heartbeat") for the primitive trace.
    This dedicated ``iteration_heartbeat`` event lets the UI filter heartbeats
    without walking the full primitive_call stream.

    Returns ``{"alive": True, "counter": <int>, "note": note}``.

    Never raises — fail-soft (D3 pattern): an IO error returns the success dict
    anyway because the caller (root model) must not be interrupted by an
    observability write failure.
    """
    import json as _json
    from datetime import datetime, timezone

    global _heartbeat_counter
    _heartbeat_counter += 1
    counter = _heartbeat_counter
    ts = datetime.now(timezone.utc).isoformat()

    event = {
        "event": "iteration_heartbeat",
        "timestamp": ts,
        "iteration": getattr(ctx, "current_iteration", None),
        "counter": counter,
        "note": note,
    }

    try:
        dashboard_path = ctx.project_dir / "dashboard_events.jsonl"
        with dashboard_path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(event, default=str) + "\n")
    except Exception:  # noqa: BLE001 — observability must never interrupt the run
        logger.exception("heartbeat: failed to write iteration_heartbeat event")

    return _with_outcome({"alive": True, "counter": counter, "note": note}, PrimitiveOutcome.ok)


PRIMITIVE_REGISTRY: dict[str, Callable[..., Any]] = {
    "understand_section": understand_section,
    "extract_hyperparameters": extract_hyperparameters,
    "detect_environment": detect_environment,
    "build_environment": build_environment,
    "plan_reproduction": plan_reproduction,
    "implement_baseline": implement_baseline,
    "run_experiment": run_experiment,
    "verify_against_rubric": verify_against_rubric,
    "propose_improvements": propose_improvements,
    "record_candidate_outcome": record_candidate_outcome,
    "check_user_messages": check_user_messages,
    "respond_to_user": respond_to_user,
    "heartbeat": heartbeat,
    "recommend_next_tool": recommend_next_tool,
    "resolve_gpu_requirements": resolve_gpu_requirements,
}

PRIMITIVE_DESCRIPTIONS: dict[str, str] = {
    "understand_section": "understand_section(text_slice) -> dict — datasets, "
        "metrics, training recipe, hardware clues, ambiguities from a text slice. "
        "A PARTIAL PaperClaimMap (no core_contribution/claims/architecture).",
    "extract_hyperparameters": "extract_hyperparameters(text_slice) -> dict — "
        "optimizer, learning rate, batch size, epochs from a slice.",
    "detect_environment": "detect_environment(method_spec) -> dict — an "
        "EnvironmentSpec (dockerfile, python_version, framework, pip_packages). "
        "`method_spec` is a (partial) PaperClaimMap dict with keys: "
        "core_contribution (str, required), claims (list of dicts — each with "
        "keys like method/dataset/metric/expected_result), metrics (list of "
        "{name, definition} dicts), plus datasets, model_architecture, "
        "training_recipe.",
    "build_environment": "build_environment(env_spec) -> dict — build the Docker "
        "image, repairing the Dockerfile on failure. Returns a BUILD RESULT "
        "{ok, image_tag, error, attempts} — NOT an EnvironmentSpec. Pass "
        "image_tag to run_experiment as env_id.",
    "plan_reproduction": "plan_reproduction(method_spec, env_spec) -> dict — a "
        "ReproductionContract (smoke test, full run, evaluation plan).",
    "implement_baseline": "implement_baseline(plan) -> dict — generate the "
        "baseline code; returns {ok, code_path, files} or {ok:false, error_code, error, repairable}. `plan` is the aggregate "
        "{paper_claim_map (from understand_section), environment_spec (from "
        "detect_environment), reproduction_contract (from plan_reproduction)}. "
        "paper_claim_map must include core_contribution (str), claims (list of "
        "dicts with keys like method/dataset/metric/expected_result), and "
        "metrics (list of {name, definition} dicts). To repair a baseline whose "
        "run_experiment FAILED, also put repair_context (the failed run_experiment "
        "result dict) in plan — the agent then diagnoses the error and fixes the "
        "existing code in place instead of rewriting.",
    "run_experiment": "run_experiment(code_path, env_id) -> dict — run the "
        "baseline in a container from image `env_id` (build_environment's "
        "image_tag); returns {success, metrics, logs}.",
    "verify_against_rubric": "verify_against_rubric(results, rubric) -> dict — "
        "score the run against a PaperBench tree rubric using the authoritative "
        "leaf scorer (flatten→LLM-grade→weighted-rollup). Returns: overall_score "
        "(float), meets_target (bool), target_score (float), leaf_count (int), "
        "graded (int), rubric_source (str), areas (list of {name, score, weight} "
        "per top-level rubric sub_task — use these directly in the final report's "
        "rubric.areas field), weak_leaves (up to 8 lowest-scoring leaf dicts), "
        "leaf_scores (all leaf scores).",
    "propose_improvements": "propose_improvements(current_results, rubric_scores, "
        "k=None) -> list[dict] — paper-specific improvement hypotheses. Each "
        "hypothesis includes a `title` field (short name for the candidate node).",
    "record_candidate_outcome": "record_candidate_outcome(candidate_id, outcome, "
        "parent_id=None) -> dict — record the root's outcome decision for a "
        "candidate. Call this after evaluating each improvement candidate. "
        "outcome is one of: 'running', 'promoted', 'marginal', 'failed', "
        "'skipped', 'declined'. candidate_id must match the id from "
        "candidate_proposed. parent_id is the node this candidate branches from.",
    "check_user_messages": "check_user_messages() -> list[dict] — return any new "
        "user messages posted to this run since the last call. Each item is "
        "{role, content, ts}. Returns an empty list when there are no new messages. "
        "Call at the start of each iteration to check for steering input.",
    "respond_to_user": "respond_to_user(message) -> dict — append an assistant "
        "reply to the conversation and emit it to the live dashboard. Returns "
        "{sent: true} on success. Call after check_user_messages returns messages "
        "you want to acknowledge or answer.",
    "heartbeat": "heartbeat(note='') -> dict — emit a liveness signal so the "
        "operator knows the root is still progressing. Returns {alive: True, "
        "counter: int, note: str}. Call this BEFORE any operation that may take "
        ">30 s: implement_baseline, run_experiment, rlm_query. Example: "
        "heartbeat('about to implement_baseline').",
    "recommend_next_tool": "recommend_next_tool(situation) -> dict — Reflexion-lite: "
        "get a structured recommendation for the next tool to call. Pass a brief "
        "description of the current situation. Returns {tool, reason, alternatives}. "
        "Use sparingly at major branch points (pre-baseline, post-failure, before "
        "sub-RLM spawn) — costs one LLM call.",
    "resolve_gpu_requirements": "resolve_gpu_requirements(requirements) -> dict — "
        "plan-time GPU resolver. `requirements` is a dict (or GpuRequirements) with: "
        "estimated_vram_gb (int|None), paper_gpu_string (str|None), "
        "paper_gpu_count (int|None), reasoning (str), confidence (float 0-1). "
        "Returns a GpuPlan dict: {runpod_id, short_name, vram_gb, gpu_count, "
        "cloud_type, sku_usd_per_hr, total_usd_per_hr, source, ladder_remaining, ...}. "
        "Call ONCE per run after accumulating hardware clues from understand_section. "
        "Idempotent — subsequent calls return the cached plan.",
}
