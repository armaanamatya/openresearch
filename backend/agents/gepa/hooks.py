"""GEPA pre/post call hooks — called by wrap_primitive()."""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

GEPA_TARGETS = frozenset({"plan_reproduction", "implement_baseline", "propose_improvements"})


def _get_config():
    from backend.config import get_settings
    return get_settings()


def _is_enabled(ctx: Any, primitive_name: str) -> bool:
    """Return True if GEPA should run for this primitive call."""
    if ctx.gepa_optimization_active:
        return False  # re-entrancy guard
    try:
        mode = _get_config().gepa_optimization
    except Exception:
        return False
    if mode == "off":
        return False
    if mode == "on":
        return primitive_name in GEPA_TARGETS
    mode_map = {
        "plan-only": "plan_reproduction",
        "baseline-only": "implement_baseline",
        "improve-only": "propose_improvements",
    }
    return mode_map.get(mode) == primitive_name


def _get_seed_prompt(primitive_name: str) -> str:
    if primitive_name == "plan_reproduction":
        from backend.agents.rlm.primitives import _PLAN_REPRODUCTION_SYSTEM
        return _PLAN_REPRODUCTION_SYSTEM
    if primitive_name == "propose_improvements":
        from backend.agents.prompts.improvement import IMPROVEMENT_ORCHESTRATOR_PROMPT
        return IMPROVEMENT_ORCHESTRATOR_PROMPT
    if primitive_name == "implement_baseline":
        from backend.agents.prompts.baseline_implementation import BASELINE_IMPLEMENTATION_PROMPT
        return BASELINE_IMPLEMENTATION_PROMPT
    return ""


def _build_trainset(ctx: Any, primitive_name: str, args: tuple, kwargs: dict) -> list:
    """Combine static paper examples + buffered real examples from previous calls."""
    buffer = ctx.gepa_example_buffer.get(primitive_name, [])

    # Load historical examples from past runs of the same paper
    historical: list = []
    if getattr(ctx, "arxiv_id", None) and getattr(ctx, "runs_root", None):
        try:
            from backend.agents.gepa.trainset.historical_examples import load_historical_examples
            historical = load_historical_examples(
                runs_root=ctx.runs_root,
                arxiv_id=ctx.arxiv_id,
                primitive_name=primitive_name,
                max_examples=10,
            )
        except Exception:
            pass

    if primitive_name == "plan_reproduction":
        from backend.agents.gepa.trainset.paper_examples import PaperExamplesBuilder
        method_spec = kwargs.get("method_spec") or (args[0] if args else {})
        env_spec = kwargs.get("env_spec") or (args[1] if len(args) > 1 else {})
        expected_ids: list[str] = []
        if getattr(ctx, "reproduction_contract", None) is not None:
            try:
                ms = getattr(ctx.reproduction_contract, "metrics_shape", None) or []
                expected_ids = [m.get("metric_id", "") for m in ms if isinstance(m, dict)]
            except Exception:
                pass
        builder = PaperExamplesBuilder(
            understand_output=method_spec,
            extract_output={},
            env_spec=env_spec,
            expected_metric_ids=expected_ids,
        )
        static = builder.build(n=3)
    else:
        static = []

    # Most-recent 5 buffered results for recency bias
    buffered = list(buffer[-5:])
    combined = static + historical + buffered
    return combined if combined else [{}]


def _build_evaluator(ctx: Any, primitive_name: str) -> Any:
    if primitive_name == "plan_reproduction":
        from backend.agents.gepa.adapters.plan_reproduction import PlanReproductionEvaluator
        return PlanReproductionEvaluator(llm_client=ctx.llm_client)
    if primitive_name == "propose_improvements":
        from backend.agents.gepa.adapters.propose_improvements import ProposeImprovementsEvaluator
        return ProposeImprovementsEvaluator(llm_client=ctx.llm_client)
    if primitive_name == "implement_baseline":
        from backend.agents.gepa.adapters.implement_baseline import ImplementBaselineProxyEvaluator
        return ImplementBaselineProxyEvaluator()
    raise ValueError(f"No evaluator for {primitive_name}")


def gepa_pre_call(ctx: Any, primitive_name: str, args: tuple, kwargs: dict) -> None:
    """Run GEPA mini-optimization before a targeted primitive call.

    On return, ctx.gepa_prompt_overrides[primitive_name] holds the best candidate.
    Sets/restores ctx.gepa_optimization_active to prevent recursive calls.
    """
    if not _is_enabled(ctx, primitive_name):
        return

    cfg = _get_config()
    timeout_map = {
        "plan_reproduction": cfg.gepa_timeout_plan_s,
        "implement_baseline": cfg.gepa_timeout_baseline_s,
        "propose_improvements": cfg.gepa_timeout_improve_s,
    }
    calls_map = {
        "plan_reproduction": cfg.gepa_max_metric_calls_plan,
        "implement_baseline": cfg.gepa_max_metric_calls_baseline,
        "propose_improvements": cfg.gepa_max_metric_calls_improve,
    }

    seed = _get_seed_prompt(primitive_name)
    if not seed:
        return

    trainset = _build_trainset(ctx, primitive_name, args, kwargs)
    evaluator = _build_evaluator(ctx, primitive_name)

    from backend.agents.gepa.callback import SSEGepaCallback
    from backend.agents.rlm.sse_bridge import build_gepa_phase_start, build_gepa_phase_complete

    cb = SSEGepaCallback(ctx=ctx, primitive_name=primitive_name)
    max_calls = calls_map[primitive_name]
    timeout_s = timeout_map[primitive_name]

    if ctx.emit:
        ctx.emit(build_gepa_phase_start(
            primitive_name=primitive_name,
            max_metric_calls=max_calls,
        ))

    ctx.gepa_optimization_active = True
    t0 = time.monotonic()
    try:
        from backend.agents.gepa.optimizer import run_gepa_mini
        best_prompt, best_score, total_calls = run_gepa_mini(
            seed_prompt=seed,
            component_name=f"{primitive_name}_system",
            trainset=trainset,
            evaluator=evaluator,
            max_metric_calls=max_calls,
            timeout_s=timeout_s,
            reflection_model=cfg.gepa_reflection_model,
            callbacks=[cb],
        )
    except Exception as exc:
        logger.warning("gepa_pre_call failed for %s: %s", primitive_name, exc)
        best_prompt, best_score, total_calls = seed, 0.0, 0
    finally:
        ctx.gepa_optimization_active = False

    duration_s = time.monotonic() - t0

    if best_prompt != seed:
        ctx.gepa_prompt_overrides[primitive_name] = best_prompt
        logger.info(
            "GEPA improved %s: score=%.3f in %d calls (%.1fs)",
            primitive_name, best_score, total_calls, duration_s,
        )

    if ctx.emit:
        ctx.emit(build_gepa_phase_complete(
            primitive_name=primitive_name,
            final_score=best_score,
            baseline_score=0.0,
            total_metric_calls=total_calls,
            duration_s=duration_s,
        ))


def gepa_post_call(ctx: Any, primitive_name: str, result: Any) -> None:
    """Record real primitive result as a training example for future calls.

    Always accumulates examples for GEPA_TARGETS, regardless of whether
    optimization is currently enabled — this ensures training data is built
    up even during off-mode runs so future enabled runs benefit immediately.
    """
    if primitive_name not in GEPA_TARGETS:
        return
    if not isinstance(result, dict):
        return
    if primitive_name not in ctx.gepa_example_buffer:
        ctx.gepa_example_buffer[primitive_name] = []
    ctx.gepa_example_buffer[primitive_name].append(
        {"output": result, "primitive": primitive_name}
    )
