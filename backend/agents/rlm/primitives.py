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
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from backend.agents.rlm.context import RunContext


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
    """
    import json
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")


def _clamp01(val: object) -> float:
    """Coerce an LLM-returned value into [0.0, 1.0]; None / garbage -> 0.0."""
    try:
        return max(0.0, min(1.0, float(val)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


_PLAN_REPRODUCTION_SYSTEM = (
    "You are the Reproduction Planner for ReproLab. Given a paper's method "
    "spec and a target environment spec, produce a ReproductionContract: what "
    "counts as a faithful reproduction, a smoke-test plan, a full-run plan, "
    "the expected output artifacts, a dataset plan, an evaluation plan, and a "
    "verification checklist. Return exactly ONE JSON object with those fields "
    "and nothing else. Do NOT write files; do NOT reference any filesystem path."
)


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
    return {
        "datasets": [d.model_dump() for d in _extract_datasets(sections)],
        "metrics": [m.model_dump() for m in _extract_metrics(sections)],
        "training_recipe": _extract_training_recipe(sections).model_dump(),
        "hardware_clues": _extract_hardware(sections),
        "ambiguities": [a.model_dump() for a in _extract_ambiguities(sections)],
    }


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
    return _extract_training_recipe({"_": text_slice}).model_dump()


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


# Indirection so tests can monkeypatch the async Docker build.
def _build_image(dockerfile_path, context_dir, tag, **kw):
    from backend.services.runtime.local_docker import build_image
    return build_image(dockerfile_path, context_dir, tag, **kw)


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
    import tempfile
    import time
    from pathlib import Path

    dockerfile = str(env_spec.get("dockerfile") or "").strip()
    if not dockerfile:
        return {"ok": False, "image_tag": "", "error": "env_spec.dockerfile is empty",
                "attempts": 0}

    attempts, ok, tag, error = 0, False, "", ""
    try:
        from backend.config import get_settings
        from backend.services.runtime.interface import SandboxRuntimeError

        settings = get_settings()
        max_attempts = max(1, settings.environment_build_max_attempts)
        # Per-attempt budget: 1800 s build + 60 s LLM repair.
        per_attempt_s = getattr(settings, "environment_build_attempt_s", 1800)
        llm_repair_s = getattr(settings, "environment_build_llm_repair_s", 60)
        # Aggregate cap: total time across all repair attempts.
        aggregate_cap_s = _timeout_for(ctx, per_attempt_s * max_attempts)

        tag = f"reprolab/{ctx.project_id}:env-check"
        deadline_abs = time.monotonic() + aggregate_cap_s
        with tempfile.TemporaryDirectory() as tmp:
            context_dir = Path(tmp)
            dockerfile_path = context_dir / "Dockerfile"
            # A2-C3: single executor for all repair iterations.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
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

    user = (
        "method_spec:\n" + json.dumps(method_spec, indent=2, default=str)
        + "\n\nenvironment_spec:\n" + json.dumps(env_spec, indent=2, default=str)
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

    async def _run():
        return await _run_baseline_with_sdk(
            ctx.project_id, ctx.runs_root, pcm, env, contract, artifact_index,
            runtime=ctx.runtime)

    timeout = _timeout_for(ctx, 3600)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            result = pool.submit(asyncio.run, _run()).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {
                "success": False,
                "error": (
                    f"implement_baseline: timed out after {timeout:.0f} s"
                ),
            }

    # run_with_sdk writes the generated code to runs_root/project_id/code;
    # derive commands.json's directory the same way (not ctx.project_dir/code)
    # so the manifest provably lands alongside the code regardless of how
    # RunContext.project_dir was constructed.
    code_dir = ctx.runs_root / ctx.project_id / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    commands = list(getattr(result, "commands_to_run", []) or [])
    (code_dir / "commands.json").write_text(json.dumps(commands), encoding="utf-8")
    return str(code_dir)


async def _execute_in_sandbox(
    code_path: str,
    env_id: str,
    commands: list[str],
    *,
    project_id: str,
    run_id: str,
) -> dict:
    """Run `commands` in a container started from the prebuilt image `env_id`.

    Drives the verified `RuntimeAppService` lifecycle (`service.py`): create a
    sandbox from the existing image (`dockerfile_path=None`, `build_context=None`
    → no rebuild, design decision D1), execute each command, destroy. The
    service methods take `Command` objects. Indirection so tests can patch it.

    Hardening (A2-C1): `asyncio.shield` on destroy so the container is cleaned
    up even when the outer thread's `.result(timeout=...)` fires and the
    coroutine is cancelled.
    """
    import asyncio
    from pathlib import Path

    from backend.services.runtime.interface import SandboxConfig
    from backend.services.runtime.local_docker import LocalDockerBackend
    from backend.services.runtime.service import (
        CreateSandbox, DestroySandbox, ExecuteCommand, RuntimeAppService,
    )

    service = RuntimeAppService(LocalDockerBackend())
    config = SandboxConfig(
        project_id=project_id,
        run_id=run_id,
        image=env_id,
        project_root=Path(code_path),
        dockerfile_path=None,   # prebuilt image — no rebuild (design decision D1)
        build_context=None,
    )
    sandbox = await service.create_sandbox(CreateSandbox(config=config))
    results = []
    try:
        for command in commands:
            results.append(await service.execute(
                ExecuteCommand(sandbox=sandbox, command=command, timeout=3600)))
    finally:
        # asyncio.shield: destroy completes even if the surrounding
        # wait_for / thread-pool timeout cancels this coroutine (A2-C1).
        await asyncio.shield(service.destroy(DestroySandbox(sandbox=sandbox)))
    return {
        "success": all(r.succeeded for r in results),
        "metrics": {},  # real metric extraction from artifacts is Phase 5 (#62)
        "logs": "\n".join(r.stdout for r in results),
    }


def run_experiment(code_path: str, env_id: str, *, ctx: "RunContext") -> dict:
    """Execute the baseline in a container from prebuilt image `env_id`; return metrics.

    Commands are read from `code_path/commands.json` (written by
    `implement_baseline`). `env_id` is a Docker image tag (design decisions
    D1/D2). Async sandbox work is bridged to sync via a worker thread.

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

    # A2-H2: guard empty env_id before attempting any Docker work.
    if not env_id or not str(env_id).strip():
        return {
            "success": False,
            "metrics": {},
            "error": "env_id empty — build_environment must succeed first",
        }

    manifest = Path(code_path) / "commands.json"
    commands = json.loads(manifest.read_text()) if manifest.exists() else []
    if not commands:
        return {"success": False, "metrics": {},
                "error": f"no commands.json at {manifest}"}

    run_id = f"{ctx.project_id}-{uuid.uuid4().hex[:8]}"
    # A2-C1: bound the entire command loop (not just each individual command).
    timeout = _timeout_for(ctx, 7200)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(
                asyncio.run,
                _execute_in_sandbox(code_path, env_id, commands,
                                    project_id=ctx.project_id, run_id=run_id),
            ).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {
                "success": False,
                "metrics": {},
                "error": f"run_experiment: timed out after {timeout:.0f} s",
            }


def verify_against_rubric(results: dict, rubric: dict, *, ctx: "RunContext") -> dict:
    """Score `results` against `rubric` via the rubric-verifier prompt.

    The LLM scores areas only; weights come verbatim from `rubric`. The honesty
    backstop is enforced mechanically: every area score is capped at 0.35 when
    the run did not succeed OR produced no metrics — matching
    `orchestrator._run_rubric_verifier` (which caps on `success`) and extending
    it to the metric-less case (`run_experiment` returns `metrics={}` in
    Phase 2 — see Task 9). `overall_score` / `meets_target` are computed by
    `RubricVerification.from_areas`, never trusted from the model.

    Hardening (WS-H Batch P):
    - A2-H3: `_extract_json` / schema failures return a fail-soft error dict.
    - A2-H4: rubric `weight` and `target_score` coerced via `_clamp01` (handles
      non-numeric LLM output gracefully, consistent with area-score handling).
    """
    import json

    from backend.agents.prompts.rubric_verifier import RUBRIC_VERIFIER_PROMPT
    from backend.agents.schemas import RubricAreaScore, RubricVerification

    user = (
        "results:\n" + json.dumps(results, indent=2, default=str)
        + "\n\nrubric:\n" + json.dumps(rubric, indent=2, default=str)
        + "\n\nScore each rubric area in [0,1]. Return a JSON object: "
          '{"areas": [{"area": str, "score": float, "justification": str, '
          '"weak_points": [str]}], "confidence": float}.'
    )
    try:
        raw = ctx.llm_client.complete(system=RUBRIC_VERIFIER_PROMPT, user=user)
        parsed = _extract_json(raw)
    except Exception as exc:  # noqa: BLE001 — fail-soft (A2-H3 / D3 pattern)
        return {"success": False,
                "error": f"verify_against_rubric: {type(exc).__name__}: {exc}"}

    # A2-H4: use _clamp01 so non-numeric weight values degrade to 0.0 rather
    # than raising TypeError/ValueError from a bare float() call.
    weights = {
        a.get("area", ""): _clamp01(a.get("weight", 0.0))
        for a in rubric.get("areas", [])
    }
    degraded = (not results.get("success")) or (not results.get("metrics"))
    areas: list[RubricAreaScore] = []
    for a in parsed.get("areas", []):
        name = str(a.get("area", ""))
        score = _clamp01(a.get("score"))
        if degraded:
            score = min(score, 0.35)  # honesty backstop
        areas.append(RubricAreaScore(
            area=name,
            weight=weights.get(name, 0.0),
            score=score,
            justification=str(a.get("justification", "")),
            weak_points=[str(w) for w in (a.get("weak_points") or [])],
        ))
    try:
        verification = RubricVerification.from_areas(
            areas,
            rubric_source=rubric.get("source", "generated"),
            target_score=_clamp01(rubric.get("target_score", 0.0)),  # A2-H4
            confidence=_clamp01(parsed.get("confidence")),
        )
    except Exception as exc:  # noqa: BLE001 — fail-soft (A2-H3)
        return {"success": False,
                "error": f"verify_against_rubric: schema error: {type(exc).__name__}: {exc}"}
    return verification.model_dump()


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
    target = k if k is not None else 3
    user = (
        "current_results:\n" + json.dumps(current_results, indent=2, default=str)
        + "\n\nrubric_scores (prioritise lifting the weakest areas):\n"
        + json.dumps(rubric_scores, indent=2, default=str)
        + f"\n\nPropose up to {target} improvement hypotheses. Return a JSON "
          'object {"hypotheses": [ImprovementHypothesis, ...]}. Each hypothesis '
          "carries a free-form `category` tag of your choosing."
    )
    raw = ctx.llm_client.complete(system=IMPROVEMENT_ORCHESTRATOR_PROMPT, user=user)
    items = _extract_json(raw).get("hypotheses", [])

    out: list[dict] = []
    for item in items:
        try:
            out.append(ImprovementHypothesis(**item).model_dump())
        except Exception:
            continue  # fail-soft: skip a malformed hypothesis
    return out[:target]


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
        "metrics (list of {name, definition} dicts).",
    "run_experiment": "run_experiment(code_path, env_id) -> dict — run the "
        "baseline in a container from image `env_id` (build_environment's "
        "image_tag); returns {success, metrics, logs}.",
    "verify_against_rubric": "verify_against_rubric(results, rubric) -> dict — "
        "score the results against a PaperBench-style rubric.",
    "propose_improvements": "propose_improvements(current_results, rubric_scores, "
        "k=None) -> list[dict] — paper-specific improvement hypotheses.",
}
