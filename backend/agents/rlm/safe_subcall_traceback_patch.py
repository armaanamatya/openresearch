"""Surface the real traceback when rlm._subcall's child RLM raises.

Upstream ``RLM._subcall`` catches every exception from ``child.completion(...)``
with ``except Exception as e: error_msg = str(e)`` (rlm/core/rlm.py:789). The
``error_msg`` is then forwarded to ``on_subcall_complete`` and surfaced in the
dashboard ``sub_rlm_complete`` event as ``error``. For a ``RecursionError`` or
similar deep-stack failure, the model and operator see only ``"maximum
recursion depth exceeded"`` — no file, no line, no chain. The 2026-05-29
mechanistic-understanding run lost two sub-RLMs to exactly this; we couldn't
tell whether the recursion was in the LLM client, the REPL parser, the prompt
templater, or a regex on LaTeX, because the traceback was discarded inside
the library.

We re-bind ``RLM._subcall`` to a copy of the upstream method whose
``except Exception`` block formats ``traceback.format_exc()`` into
``error_msg`` (capped at 2000 chars to stay within the SSE egress bound).
Everything else — child construction, budget propagation, cleanup, callback
firing — is unchanged.

Mirror of safe_repl_traceback_patch.py for the sub-call surface.
"""
from __future__ import annotations

import logging
import time
import traceback

from rlm import core as _rlm_core
from rlm.core.rlm import RLM, RLMChatCompletion, UsageSummary

logger = logging.getLogger(__name__)
_PATCHED_ATTR = "_reprolab_subcall_traceback_patched"
_TRACEBACK_CAP = 2000


def _format_error_with_tb(exc: BaseException) -> str:
    tb = traceback.format_exc()
    if len(tb) > _TRACEBACK_CAP:
        tb = "...(truncated)\n" + tb[-(_TRACEBACK_CAP - 18):]
    return f"{exc}\n{tb}"


def _patched_subcall(self, prompt, model=None):  # noqa: C901 — mirrors upstream
    BudgetExceededError = _rlm_core.rlm.BudgetExceededError  # local import to avoid circulars
    get_client = _rlm_core.rlm.get_client
    RLMLogger = _rlm_core.rlm.RLMLogger

    next_depth = self.depth + 1

    if model is not None:
        child_backend_kwargs = (self.backend_kwargs or {}).copy()
        child_backend_kwargs["model_name"] = model
    else:
        child_backend_kwargs = self.backend_kwargs
    resolved_model = model or (child_backend_kwargs or {}).get("model_name", "unknown")

    if next_depth >= self.max_depth:
        if self.other_backends and self.other_backend_kwargs:
            client = get_client(self.other_backends[0], self.other_backend_kwargs[0])
        else:
            client = get_client(self.backend, child_backend_kwargs or {})
        root_model = model or client.model_name
        start_time = time.perf_counter()
        try:
            response = client.completion(prompt)
            end_time = time.perf_counter()
            model_usage = client.get_last_usage()
            usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
            return RLMChatCompletion(
                root_model=root_model,
                prompt=prompt,
                response=response,
                usage_summary=usage_summary,
                execution_time=end_time - start_time,
            )
        except Exception as e:
            end_time = time.perf_counter()
            err_with_tb = _format_error_with_tb(e)
            return RLMChatCompletion(
                root_model=root_model,
                prompt=prompt,
                response=f"Error: LM query failed at max depth - {err_with_tb}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=end_time - start_time,
            )

    remaining_budget = None
    if self.max_budget is not None:
        remaining_budget = self.max_budget - self._cumulative_cost
        if remaining_budget <= 0:
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=(
                    "Error: Budget exhausted "
                    f"(spent ${self._cumulative_cost:.6f} of ${self.max_budget:.6f})"
                ),
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=0.0,
            )

    remaining_timeout = None
    if self.max_timeout is not None and self._completion_start_time is not None:
        elapsed = time.perf_counter() - self._completion_start_time
        remaining_timeout = self.max_timeout - elapsed
        if remaining_timeout <= 0:
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: Timeout exhausted ({elapsed:.1f}s of {self.max_timeout:.1f}s)",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=0.0,
            )

    prompt_preview = prompt[:80] if len(prompt) > 80 else prompt
    if self.on_subcall_start:
        try:
            self.on_subcall_start(next_depth, str(resolved_model), prompt_preview)
        except Exception:
            pass

    subcall_start = time.perf_counter()
    error_msg: str | None = None

    child = RLM(
        backend=self.backend,
        backend_kwargs=child_backend_kwargs,
        environment=self.environment_type,
        environment_kwargs=self.environment_kwargs,
        depth=next_depth,
        max_depth=self.max_depth,
        max_iterations=self.max_iterations,
        max_budget=remaining_budget,
        max_timeout=remaining_timeout,
        max_tokens=self.max_tokens,
        max_errors=self.max_errors,
        custom_system_prompt=self.system_prompt,
        other_backends=self.other_backends,
        other_backend_kwargs=self.other_backend_kwargs,
        logger=RLMLogger() if self.logger else None,
        verbose=False,
        custom_tools=self.custom_sub_tools,
        custom_sub_tools=self.custom_sub_tools,
        on_subcall_start=self.on_subcall_start,
        on_subcall_complete=self.on_subcall_complete,
    )
    try:
        result = child.completion(prompt, root_prompt=None)
        if result.usage_summary and result.usage_summary.total_cost:
            self._cumulative_cost += result.usage_summary.total_cost
        return result
    except BudgetExceededError as e:
        self._cumulative_cost += e.spent
        error_msg = f"Budget exceeded - {e}"
        return RLMChatCompletion(
            root_model=resolved_model,
            prompt=prompt,
            response=f"Error: Child RLM budget exceeded - {e}",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=time.perf_counter() - subcall_start,
        )
    except Exception as e:
        error_msg = _format_error_with_tb(e)
        return RLMChatCompletion(
            root_model=resolved_model,
            prompt=prompt,
            response=f"Error: Child RLM completion failed - {error_msg}",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=time.perf_counter() - subcall_start,
        )
    finally:
        child.close()
        if self.on_subcall_complete:
            try:
                duration = time.perf_counter() - subcall_start
                self.on_subcall_complete(next_depth, str(resolved_model), duration, error_msg)
            except Exception:
                pass


def apply_subcall_traceback_patch() -> None:
    if getattr(RLM, _PATCHED_ATTR, False):
        return
    RLM._subcall = _patched_subcall
    setattr(RLM, _PATCHED_ATTR, True)
    logger.debug("safe_subcall_traceback_patch: RLM._subcall patched")


apply_subcall_traceback_patch()
