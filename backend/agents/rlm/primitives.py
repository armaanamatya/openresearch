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
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from backend.agents.rlm.context import RunContext
    from backend.agents.schemas import GpuPlan

logger = logging.getLogger(__name__)

# Module-level alias so tests can monkeypatch RuntimeAppService without
# requiring a live Docker daemon.
from backend.services.runtime.service import RuntimeAppService


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
_EXEC_TIMEOUT_SECONDS = 3600

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
    experiment. Exit code 137 is SIGKILL (OOM killer); substring match covers the
    documented PyTorch/cuBLAS variants. Pattern set is intentionally tight to avoid
    false positives on unrelated CUDA errors.
    """
    if exit_code == 137:
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
    """
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
    return result


def extract_hyperparameters(text_slice: str, *, ctx: "RunContext") -> dict:
    """Extract hyperparameters from a slice (typically the training-recipe section).

    Wraps `paper_understanding._extract_training_recipe`. Returns a flat dict:
    optimizer, learning_rate, batch_size, epochs_or_steps, scheduler,
    other_hparams. The heuristic populates the first four; the root model can
    fill scheduler/other_hparams via `llm_query` if needed.

    `ctx` is required by the primitive-wrapper protocol (design decision D4);
    this heuristic body does not use it.
    """
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
        return {
            "success": False,
            "error": (
                f"detect_environment: method_spec must be a dict, "
                f"got {type(method_spec).__name__!r}"
            ),
        }

    from backend.agents.environment_detective import run_offline
    from backend.agents.schemas import PaperClaimMap

    claim_map = PaperClaimMap(**{"core_contribution": "", **method_spec})
    spec = run_offline(
        ctx.project_id, ctx.runs_root, claim_map, method_spec.get("artifact_index"))
    return spec.model_dump()


def _emit_dashboard_event(ctx: "RunContext", *, event_type: str, payload: dict) -> None:
    """Append a JSON event line to runs/<id>/dashboard_events.jsonl.

    Fail-soft (D3): any IO error is logged but never propagates — observability
    must never interrupt a run.
    """
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path

    events_file = _Path(ctx.project_dir) / "dashboard_events.jsonl"
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
            return cached
        except Exception:  # noqa: BLE001 — corrupt cache → recompute
            logger.warning("resolve_gpu_requirements: cache file unreadable, recomputing")

    # ---- Coerce payload.
    if isinstance(requirements, dict):
        req = _Req(**requirements)
    elif isinstance(requirements, _Req):
        req = requirements
    else:
        raise ValueError(
            f"resolve_gpu_requirements: requirements must be GpuRequirements or dict, "
            f"got {type(requirements).__name__}"
        )

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
    payload = plan.model_dump(mode="json")
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
        return {"ok": False, "image_tag": "", "error": "env_spec.dockerfile is empty",
                "attempts": 0}

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
            return {"ok": True, "image_tag": tag, "attempts": 0, "skipped": True}

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
    except SandboxRuntimeError:
        raise  # infrastructure failure — not a Dockerfile problem; propagate
    except Exception as exc:  # noqa: BLE001 — fail-soft (D3): any other failure
        return {"ok": False, "image_tag": "",
                "error": f"{type(exc).__name__}: {exc}", "attempts": attempts}

    return {"ok": ok, "image_tag": tag if ok else "", "error": error,
            "attempts": attempts}


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
    try:
        raw = ctx.llm_client.complete(system=_PLAN_REPRODUCTION_SYSTEM, user=user)
        data = _extract_json(raw)
        if not any(k in data for k in ReproductionContract.model_fields):
            raise ValueError(
                f"LLM response has no ReproductionContract fields: {list(data)}")
        return ReproductionContract(**data).model_dump()
    except Exception as exc:  # noqa: BLE001 — fail-soft (A2-H3 / D3 pattern)
        return {"success": False, "error": f"plan_reproduction: {type(exc).__name__}: {exc}"}


def _run_baseline_with_sdk(project_id, runs_root, pcm, env, contract, artifact_index, **kw):
    """Indirection over baseline_implementation.run_with_sdk so tests can patch it."""
    from backend.agents.baseline_implementation import run_with_sdk
    return run_with_sdk(project_id, runs_root, pcm, env, contract, artifact_index, **kw)


def implement_baseline(plan: dict, *, ctx: "RunContext") -> str | dict:
    """Generate the baseline code from a reproduction plan; return the code path.

    `plan` is the aggregate dict the root assembles: `{"paper_claim_map":
    <understand_section output>, "environment_spec": <detect_environment
    output>, "reproduction_contract": <plan_reproduction output>}` (plus an
    optional `artifact_index`) — NOT a single producer's output. Wraps
    `baseline_implementation.run_with_sdk` (a code-writing agent) and writes
    `code/commands.json` so `run_experiment` can read the run commands without
    a BaselineResult (design decision D2).

    Hardening (A2-C2): `pool.submit(...).result()` previously blocked the
    worker thread indefinitely; now bounded by `_timeout_for(ctx, 3600)`.
    On timeout returns a fail-soft error dict (never raises).
    """
    import asyncio
    import json

    from backend.agents.schemas import PaperClaimMap, EnvironmentSpec, ReproductionContract

    # core_contribution is PaperClaimMap's one required field; default it so a
    # partial paper_claim_map (e.g. understand_section's output) validates.
    pcm = PaperClaimMap(**{"core_contribution": "", **plan.get("paper_claim_map", {})})
    env = EnvironmentSpec(**plan.get("environment_spec", {}))
    contract = (ReproductionContract(**plan["reproduction_contract"])
                if plan.get("reproduction_contract") else None)
    artifact_index = plan.get("artifact_index")

    # An optional plan["repair_context"] (a failed run_experiment result) puts
    # the code-writing agent into fix-existing-code mode — the root passes it
    # to retry after run_experiment fails.
    repair_context = plan.get("repair_context")

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
            gpu_mode=getattr(ctx, "gpu_mode", None))

    timeout = _timeout_for(ctx, 3600)
    # I12: explicit shutdown(wait=False) so a wedged worker cannot block cleanup.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        try:
            result = pool.submit(asyncio.run, _run()).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {
                "success": False,
                "error": (
                    f"implement_baseline: timed out after {timeout:.0f} s"
                ),
            }
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # run_with_sdk writes the generated code to runs_root/project_id/code;
    # derive commands.json's directory the same way (not ctx.project_dir/code)
    # so the manifest provably lands alongside the code regardless of how
    # RunContext.project_dir was constructed.
    code_dir = ctx.runs_root / ctx.project_id / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    commands = list(getattr(result, "commands_to_run", []) or [])
    (code_dir / "commands.json").write_text(json.dumps(commands), encoding="utf-8")
    return str(code_dir)


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

    if mode is SandboxMode.runpod:
        import backend.services.runtime as _runtime
        from backend.services.runtime.runpod_backend import RunpodBackend

        _runtime.ensure_runpod_available()
        return RunpodBackend(run_budget=run_budget, gpu_plan=gpu_plan)

    # All other modes (local, auto, brev, simulate) are not yet wired
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

    from backend.services.runtime.interface import SandboxConfig
    from backend.services.runtime.service import (
        CreateSandbox, DestroySandbox, ExecuteCommand,
    )

    code_dir = Path(code_path)
    # Per-call artifact dir: deterministic per run_id so retries don't clobber.
    artifact_root = code_dir / "outputs" / run_id
    artifact_root.mkdir(parents=True, exist_ok=True)

    service = RuntimeAppService(_backend_for_sandbox_mode(
        sandbox_mode, run_budget=run_budget, gpu_plan=gpu_plan,
    ))
    config = SandboxConfig(
        project_id=project_id,
        run_id=run_id,
        image=env_id,
        project_root=code_dir,
        artifact_root=artifact_root,
        dockerfile_path=None,   # prebuilt image — no rebuild (design decision D1)
        build_context=None,
        # Bug C: paper reproduction must fetch pretrained weights and datasets
        # (HuggingFace, PyPI, torch hub) — network_disabled defaults to True and
        # blocked every model-download paper. The paper corpus is never mounted
        # into this container (only agent-written code is), so this is not a
        # corpus-leak vector. Scoped here; the global default stays disabled.
        network_disabled=False,
        environment={
            "OUTPUT_DIR": "/artifacts",
            "REPROLAB_ARTIFACT_DIR": "/artifacts",
            "MPLCONFIGDIR": "/artifacts/.matplotlib",
            "PYTHONUNBUFFERED": "1",
        },
    )
    sandbox = await service.create_sandbox(CreateSandbox(config=config))
    results = []
    try:
        for command in commands:
            results.append(await service.execute(
                ExecuteCommand(sandbox=sandbox, command=command,
                               timeout=_EXEC_TIMEOUT_SECONDS)))
    finally:
        # asyncio.shield: destroy completes even if the surrounding
        # wait_for / thread-pool timeout cancels this coroutine (A2-C1).
        await asyncio.shield(service.destroy(DestroySandbox(sandbox=sandbox)))

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

    return {
        "success": all(r.succeeded for r in results),
        "metrics": metrics,
        "logs": _cap_logs(_combine_command_output(results)),
        "artifact_dir": str(artifact_root),
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


def run_experiment(
    code_path: str,
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
        env_id = build["image_tag"]

    # A2-H2: guard empty env_id (reachable only when no Dockerfile was on disk
    # to rebuild from) before attempting any Docker work.
    if not env_id or not str(env_id).strip():
        return _persist_experiment_result(ctx, {
            "success": False,
            "metrics": {},
            "error": "env_id empty and no Dockerfile to rebuild — build_environment must succeed first",
        }, model_id=model_id, eval_env=eval_env)

    # 2026-05-23 (final): NO default per-primitive cap. Only honor explicit
    # caps from either (a) REPROLAB_RUN_EXPERIMENT_TIMEOUT_S env var, or
    # (b) the run-budget deadline via ctx.remaining_s() (the --max-wall-clock
    # CLI flag). Without either set, run_experiment is unbounded — long-running
    # experiments must use the env var or the --max-wall-clock budget if they
    # need a cap. (User mandate 2026-05-23: "no cost cap until set".)
    # The pattern that previously hung B2 — model writes CPU-bound train.py —
    # is now addressed at the agent prompt layer (sandbox-aware
    # implement_baseline picks --smoke-test for CPU sandboxes), not via cap.
    _cap_s = None
    try:
        import os as _os_env
        _override = _os_env.environ.get("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S")
        if _override:
            _cap_s = float(_override)
    except (TypeError, ValueError):
        pass
    if _cap_s is None:
        # No explicit env-var cap: respect only the run-budget. ctx.remaining_s()
        # returns None when no budget is set, which becomes timeout=None below
        # → .result(timeout=None) waits indefinitely.
        timeout = ctx.remaining_s()
    else:
        timeout = _timeout_for(ctx, _cap_s)

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
    escalations = 0
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
                    ),
                ).result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                result = {
                    "success": False,
                    "metrics": {},
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
                else:
                    result = {
                        "success": False, "metrics": {},
                        "error": f"run_experiment: {type(exc).__name__}: {exc_msg[:300]}",
                    }
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
    leaf_score_map: dict[str, float] = {
        str(e["id"]): float(e.get("score", 0.0))
        for e in leaf_scores_list
        if isinstance(e, dict) and e.get("id")
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
        return {
            "success": False,
            "error": "verify_against_rubric: rubric must be a non-empty dict",
        }

    try:
        from backend.evals.paperbench.leaf_scorer import score_reproduction

        # C2b in-loop wiring: derive `degraded` from the `results` dict we
        # already have. The leaf scorer's auto-detection reads
        # final_report.json, but in-loop (called from the improvement loop
        # before _finalize) that file has not been written yet, so
        # auto-detection returns False and the cap would not fire. Pass it
        # explicitly so the in-loop optimization signal matches what the
        # post-run authoritative score will become.
        # Two-layer degraded predicate: post-run path checks verdict+metrics via
        # final_report.json (_is_degraded_run); in-loop path checks success+metrics
        # via the live run_experiment result dict (verdict is a report-level concept
        # not yet written at this point).  Both are correct at their respective layer.
        has_experiment_result = "success" in results or "metrics" in results
        degraded = has_experiment_result and (
            (results.get("success") is False) or (not (results.get("metrics") or {}))
        )
        scored = score_reproduction(
            rubric_tree=rubric,
            run_dir=ctx.project_dir,
            llm_client=ctx.llm_client,
            rubric_source=str(rubric.get("source") or "paperbench_bundle"),
            degraded=degraded,
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
            return {
                "success": False,
                "error": (
                    f"verify_against_rubric: leaf scorer graded 0/{leaf_count} leaves — "
                    f"LLM grader output was unparseable on every batch; no honest score available"
                ),
            }
        overall_score = _clamp01(scored["overall_score"])
        target = _clamp01(rubric.get("target_score", 0.6))
        meets_target = overall_score >= target

        leaf_scores = scored.get("leaf_scores", [])
        # Up to 8 lowest-scoring leaves (conservative grader — 0.0 means no evidence)
        weak_leaves = sorted(
            [e for e in leaf_scores if isinstance(e, dict)],
            key=lambda e: float(e.get("score", 0.0)),
        )[:8]

        return {
            "overall_score": overall_score,
            "meets_target": meets_target,
            "target_score": target,
            "leaf_count": scored.get("leaf_count", 0),
            "graded": scored.get("graded", 0),
            "rubric_source": scored.get("rubric_source", "paperbench_bundle"),
            "degraded": degraded,
            "areas": _rubric_areas(rubric, leaf_scores),
            "weak_leaves": [
                {"id": e.get("id", ""), "score": e.get("score", 0.0),
                 "justification": e.get("justification", "")}
                for e in weak_leaves
            ],
            "leaf_scores": leaf_scores,
        }
    except Exception as exc:  # noqa: BLE001 — fail-soft (A2-H3 / D3 pattern)
        return {
            "success": False,
            "error": f"verify_against_rubric: {type(exc).__name__}: {exc}",
        }


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
        return [{
            "success": False,
            "error": f"propose_improvements: {type(exc).__name__}: {exc}",
        }]

    out: list[dict] = []
    for item in items:
        try:
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
            "error": (
                f"record_candidate_outcome requires a real candidate_id (got {cid!r}). "
                f"Use the 'id' field from the most recent propose_improvements result "
                f"(e.g. 'path_1', 'path_2')."
            ),
            "candidate_id": cid_str,
            "outcome": str(outcome) if outcome is not None else "",
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
        return {"sent": False, "error": "respond_to_user: message must be non-empty"}

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
        return {"sent": False, "error": f"respond_to_user: IO error: {exc}"}

    return {"sent": True}


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
        }
    except Exception as exc:  # noqa: BLE001 — advisory; never break the root run
        return {"tool": "", "reason": f"recommend_next_tool failed: {type(exc).__name__}", "alternatives": []}


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

    return {"alive": True, "counter": counter, "note": note}


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
    "implement_baseline": "implement_baseline(plan) -> str — generate the "
        "baseline code; returns the code dir path. `plan` is the aggregate "
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
