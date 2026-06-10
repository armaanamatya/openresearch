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
import re
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
    "cell_execution_error",   # Phase 0C: all run cells errored (non-OOM) → repair
    "degenerate_training",
    "disk_exhausted",
    "incomplete_metrics",
    "contract_violation",
    # Existing classifier labels that require agent-side repair.
    "missing_module",
    "torch_redundancy",
    "cuda_oom",
    "cuda_shlib_load",        # incoherent CUDA stack (libcupti/… won't load) → strip torch re-pin
    "oom_killed",
    "requirements_not_found",
    "missing_dataset",
    "exec_timeout",
    "watchdog_killed",
    "preflight_blocked",
    "permission_denied",
    "syntax_error",
    "scope_shape_violation",
    "dockerfile_invalid",
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


def _local_core_bootstrap_commands(requirements_path: "Path", torch_index: str) -> list[str]:
    """pip-install commands for the local-sandbox bootstrap, hardened by env_pin.

    Installs the harness-owned cu121 core pins (torch/vision/audio) FIRST, then the
    agent's requirements with any conflicting core re-pin stripped (writing
    ``requirements.hardened.txt`` next to ``requirements.txt``). This is the fix for the
    2026-06-07 All-Conv-Net collapse, where the agent's ``torch==2.2.0`` re-pin
    DOWNGRADED the cu121 build and left an incoherent CUDA stack (``libcupti.so.12``
    failed to dlopen → every experiment died at import).

    Fail-soft: env_pin off (``OPENRESEARCH_DISABLE_ENV_PIN``) / no torch index / unknown tag
    / any error → legacy bare-``torch`` install + raw ``requirements.txt``. Returns the
    ordered command list (the caller appends ``accelerate`` afterwards).
    """
    core_install_cmd: str | None = None
    requirements_target = "requirements.txt"
    env_pin_on = bool(torch_index) and os.environ.get(
        "OPENRESEARCH_DISABLE_ENV_PIN", ""
    ).strip().lower() not in ("1", "true", "yes", "on")
    if env_pin_on:
        try:
            from backend.agents.rlm import env_pin

            tag = env_pin.base_tag_for("local", None)  # → "cu121"
            specs = env_pin.pin_install_specs(tag)
            kept, dropped = env_pin.harden_requirements(
                requirements_path.read_text(encoding="utf-8").splitlines(),
                base_tag=tag,
            )
            # Garbage-line guard (2026-06-09): one un-parseable line (agent prose
            # like "(Section 5.2)") aborts the ENTIRE `pip install -r`, silently
            # losing every valid dependency — the failure then surfaces minutes
            # later as missing_module. Strip such lines and keep the rest.
            kept, invalid = env_pin.sanitize_requirements(kept)
            if specs:
                core_install_cmd = (
                    f"python -m pip install {' '.join(specs)} "
                    f"--index-url {torch_index} || true"
                )
            if dropped or invalid:
                hardened = requirements_path.with_name("requirements.hardened.txt")
                header = [
                    f"# pip-invalid line removed by harness: {s}" for s in invalid
                ]
                hardened.write_text(
                    "\n".join(header + kept) + "\n", encoding="utf-8"
                )
                requirements_target = hardened.name
                if dropped:
                    logger.info(
                        "_local_core_bootstrap_commands: env_pin stripped %d core re-pin(s) "
                        "%s; harness installs the pinned cu121 stack", len(dropped), dropped,
                    )
                if invalid:
                    logger.warning(
                        "_local_core_bootstrap_commands: removed %d pip-invalid "
                        "requirements line(s): %s", len(invalid), invalid,
                    )
        except Exception:  # noqa: BLE001 — env_pin must never block the run
            logger.exception(
                "_local_core_bootstrap_commands: env_pin hardening failed; raw requirements.txt"
            )
            core_install_cmd, requirements_target = None, "requirements.txt"

    cmds: list[str] = []
    if core_install_cmd is not None:
        cmds.append(core_install_cmd)
    elif torch_index:
        cmds.append(f"python -m pip install torch --index-url {torch_index} || true")
    cmds.append(f"python -m pip install -r {requirements_target} || true")
    return cmds


# Phase 0C: failure classes whose run_experiment result carries a POPULATED
# metrics dict (aggregate_cell_metrics always returns non-empty) yet represents an
# all-cells-failed, fully-repairable code bug. These must engage the repair floor,
# NOT be typed partial_evidence by the metrics-first short-circuit in
# _classify_run_experiment_outcome. Keep this NARROW: only classes that are set
# exclusively when ZERO cells succeeded, so a genuine some-ok/some-bug partial is
# never reclassified as fully repairable. ``cell_execution_error`` is set only in
# the n_ok==0, n_err>0 branch of the cell matrix (primitives.py ~:3789).
_METRICS_BEARING_REPAIRABLE_FAILURES = {
    "cell_execution_error",
}

_CODEX_HARD_ALLOWED_TASKS = {
    "implementation_repair",
    "test_debugging",
    "dockerfile_repair",
    "requirements_repair",
    "traceback_explanation",
}
_CODEX_REJECTED_TASKS = {
    "paper_summary",
    "paper_navigation",
    "rubric_judgment",
    "final_report",
    "broad_research",
    "open-ended_planning",
    "open_ended_planning",
    "credential_inspection",
    "secret_search",
}
_CODEX_AGENT_CORRECTABLE_FAILURES = {
    "syntax_error",
    "missing_module",
    "requirements_not_found",
    "dockerfile_invalid",
    "contract_violation",
    "scope_shape_violation",
}


def _validate_dockerfile_shape(text: str) -> bool:
    """Deterministic Dockerfile shape guard (BUG-NEW-042).

    Returns True iff the first non-blank, non-comment line is a ``FROM`` or
    ``ARG`` instruction (a leading ``# syntax=`` directive is allowed and
    skipped). Rejects the common sub-agent failure mode of dumping prose /
    markdown in place of a Dockerfile, which would otherwise be handed to
    ``docker build`` and fail with an opaque parser error after wasting a build.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            # blank lines and comments (incl. the `# syntax=` directive) skip
            continue
        head = line.split(None, 1)[0].upper()
        return head in ("FROM", "ARG")
    return False  # empty / all-comment Dockerfile is not buildable

# PR-ζ: transient-error retry policy for _execute_in_sandbox.
# Three retries with exponential backoff: 5s, 10s, 20s.
# Total retry budget is capped so it cannot blow through the primitive
# wall-clock limit (the surrounding run_experiment timeout still bounds).
_MAX_TRANSIENT_RETRIES: int = 3
_BACKOFF_BASE_S: float = 5.0
_RETRY_TIMEOUT_TOTAL_S: float = 90.0

# Pre-emit stall threshold: how long implement_baseline tolerates NO new file in
# code_dir before declaring an SDK hang. The signal is coarse — file mtimes only
# update when the Write tool COMPLETES a file, so a sub-agent that plans for several
# minutes or generates a large file (e.g. a 43 KB train.py) legitimately shows no
# code_dir activity mid-work and looks "stalled".
# Measured 2026-05-29 (SDAR + --paper-hint, sun.cs.txstate.edu, non-WSL): the agent's
# FIRST file landed at ~+593 s and the gap while generating train.py was ~402 s, yet
# it produced a complete, correct implementation. The old 240 s false-killed that
# healthy, productive agent 3× (→ no experiment, verdict=failed). 900 s comfortably
# covers large-file generation + planning on complex papers; a genuine SDK hang still
# surfaces, just later — an acceptable trade for an unbounded reproduction where a
# FALSE stall (which aborts the whole run) is far costlier than slow hang-detection.
# Override with OPENRESEARCH_PRE_EMIT_STALL_S. (Follow-up: make progress SDK-stream/liveness
# aware rather than code_dir-mtime-only, so the threshold matters less.)
_DEFAULT_PRE_EMIT_STALL_S = 900.0


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


def _codex_env_allowed_tasks(raw: str | None) -> set[str]:
    return {
        item.strip().lower().replace("-", "_")
        for item in str(raw or "").split(",")
        if item.strip()
    }


def _codex_repair_error(error_type: str, message: str, **extra: Any) -> dict:
    payload = {
        "ok": False,
        "success": False,
        "disabled": error_type == "disabled",
        "timed_out": False,
        "exit_code": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "changed_files": [],
        "duration_s": 0.0,
        "error_type": error_type,
        "error": message,
    }
    payload.update(extra)
    return _with_outcome(payload, PrimitiveOutcome.repairable)


def _codex_failure_class(
    failure_class: str | None,
    repair_context: dict | None,
) -> str:
    if failure_class:
        return _failure_class_key(failure_class)
    if isinstance(repair_context, dict):
        return _failure_class_key(repair_context.get("failure_class"))
    return ""


def _build_codex_prompt(
    *,
    task_type: str,
    instructions: str,
    test_command: str,
    allowed_paths: list[str],
    timeout_s: int,
    failure_class: str,
) -> str:
    allowed = "\n".join(f"- {p}" for p in allowed_paths) if allowed_paths else "- Entire workspace"
    return (
        "You are a specialized ReproLab repo-editing repair subagent.\n"
        "Do not perform paper navigation, paper summarization, rubric judgment, "
        "final report writing, broad research, credential inspection, or secret search.\n\n"
        f"Task type: {task_type}\n"
        f"Failure class: {failure_class or 'not provided'}\n"
        f"Exact task:\n{instructions.strip()}\n\n"
        "Allowed files or workspace scope:\n"
        f"{allowed}\n\n"
        f"Test command to run:\n{test_command.strip()}\n\n"
        f"Max time budget: {timeout_s} seconds.\n\n"
        "Constraints:\n"
        "- Touch only files needed for the targeted repair and do not touch unrelated files.\n"
        "- Do not print secrets, environment variables, access tokens, credential files, or auth state.\n"
        "- Do not read, print, parse, or copy ~/.codex/auth.json or any credential store.\n"
        "- Run only the targeted test command above unless a smaller diagnostic command is necessary.\n"
        "- Summarize changed files and tests run in your final response.\n"
        "- Stop after the targeted fix; do not continue into open-ended cleanup or planning.\n"
    )


def _classify_run_experiment_outcome(result: dict) -> PrimitiveOutcome:
    """Map a run_experiment result dict to its primitive typestate."""
    if result.get("success") is True:
        return PrimitiveOutcome.ok

    failure_class = _failure_class_key(result.get("failure_class"))

    # Phase 0C: a few failure classes carry a populated metrics dict even though
    # every run cell failed (aggregate_cell_metrics always returns non-empty, so
    # the all-cells-errored ``cell_execution_error`` branch ships metrics). The
    # metrics-first short-circuit below would mis-type those as partial_evidence
    # and skip the repair-iteration floor (which fires only on ``repairable``).
    # Consult the failure_class FIRST for these so a code-bug cell engages repair.
    if failure_class in _METRICS_BEARING_REPAIRABLE_FAILURES:
        return PrimitiveOutcome.repairable

    metrics = result.get("metrics")
    if isinstance(metrics, dict) and bool(metrics):
        return PrimitiveOutcome.partial_evidence

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

    Pre: ``OPENRESEARCH_PRE_EMIT_STALL_S`` may be unset or a positive number.
    Post: returns a positive second threshold, defaulting to 120s.
    Side effects: logs a warning for invalid environment values.
    Exceptions raised: none.
    """
    raw = os.environ.get("OPENRESEARCH_PRE_EMIT_STALL_S", "").strip()
    if not raw:
        return _DEFAULT_PRE_EMIT_STALL_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid OPENRESEARCH_PRE_EMIT_STALL_S=%r; using default", raw)
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
      1. OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S env var (if set and > 0)
      2. EXPERIMENT_TIMEOUT_BY_MODE[ctx.execution_mode]
      3. _DEFAULT_EXPERIMENT_TIMEOUT_S

    Then clamp to ctx.remaining_s() only when finite — infinite remaining
    means no --max-wall-clock was set; honor the mode default unchanged.
    """
    import math as _math
    import os as _os

    _env = _os.environ.get("OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S", "").strip()
    if _env:
        try:
            override = int(_env)
            if override > 0:
                resolved = override
            else:
                resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(
                    getattr(ctx, "execution_mode", None)
                    or _os.environ.get("OPENRESEARCH_EXECUTION_MODE"),
                    _DEFAULT_EXPERIMENT_TIMEOUT_S,
                )
        except ValueError:
            resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(
                getattr(ctx, "execution_mode", None)
                or _os.environ.get("OPENRESEARCH_EXECUTION_MODE"),
                _DEFAULT_EXPERIMENT_TIMEOUT_S,
            )
    else:
        resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(
            getattr(ctx, "execution_mode", None)
            or _os.environ.get("OPENRESEARCH_EXECUTION_MODE"),
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
                                # Override via OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S
                                # in run_experiment if a single run truly needs
                                # a tighter bound.

# Execution-reliability redesign (2026-06-08, local-scoped): the LOCAL inner exec
# OWNS the experiment deadline (we pass the resolved timeout as its per-command cap),
# so its stall/timeout fires FIRST and process-group-kills the train subprocess
# cleanly (no GPU-burning orphan). The outer thread-pool .result() then becomes a
# generous BACKSTOP = resolved + this buffer (covers bootstrap + cleanup). On
# non-local backends the inner cap stays _EXEC_TIMEOUT_SECONDS and the outer stays
# the resolved timeout (runpod/docker byte-for-byte unchanged).
_OUTER_TIMEOUT_BUFFER_S = 1800  # 30 min generous backstop over the inner deadline (local only)

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


def _is_oom_escalation_trigger(result: dict, *, exit_code: int, stderr_tail: str) -> bool:
    """True when a failed experiment should advance the GPU ladder due to OOM (F-04).

    Two cases:
      1. A direct CUDA OOM signal — delegated to ``_detect_cuda_oom`` (exit code
         137/-9 or an OOM marker in the last ~4 KB ``stderr_tail``).
      2. A stall-watchdog kill (``result['watchdog_killed']``) whose OOM marker is
         buried earlier than ``stderr_tail``: the watchdog return dict surfaces no
         ``exit_code`` (the gate defaults it to 1) and the marker can sit thousands
         of lines before the tail, so case 1 alone misses it — scan the FULL
         ``result['logs']`` for a marker.

    The watchdog kills on *staleness* (no signal), NOT memory, so a watchdog kill
    escalates ONLY when the full logs carry an explicit OOM marker; a genuine
    no-signal stall (no marker) breaks the loop as before.
    """
    if _detect_cuda_oom(exit_code=exit_code, stderr_tail=stderr_tail):
        return True
    if result.get("watchdog_killed"):
        full_logs = result.get("logs") or ""
        return any(marker in full_logs for marker in _CUDA_OOM_MARKERS)
    return False


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
    spec_dict = spec.model_dump()
    # Runtime-hardware truth (2026-06-09): papers older than the GPU era never
    # *mention* GPUs, so the paper-derived assumption reads "CPU only (no GPU
    # required)" — and the agent then plans timid CPU-scale experiments on a
    # multi-GPU box (every Adam/All-CNN attempt planned "CPU-only per ENV003"
    # while training actually ran on an RTX A5000). Append a harness-measured
    # assumption naming the real capacity so the agent scales its plan to the
    # hardware it actually has. Fail-soft: any probe error skips the annotation.
    try:
        from backend.services.runtime.gpu_capacity import describe_capacity
        caps = describe_capacity(ctx)
        n_gpus = len(caps.free_gpu_ids or ())
        if n_gpus > 0:
            vram = float(caps.per_gpu_vram_gb or 0.0)
            vram_txt = f"{vram:.0f} GB VRAM each" if vram > 0 else "VRAM unknown"
            assumptions = list(spec_dict.get("assumptions") or [])
            assumptions.append({
                "assumption_id": "ENV-RT1",
                "detail": "Runtime hardware (harness-measured)",
                "chosen_value": (
                    f"{n_gpus}× CUDA GPU available to this run ({vram_txt}). "
                    "Regardless of what the paper mentions, prefer CUDA and "
                    "scale experiments (epochs, model sizes, grids) to this "
                    "hardware rather than planning CPU-only smoke runs."
                ),
                "evidence": [
                    f"harness describe_capacity probe: {n_gpus} free GPU(s), "
                    f"per-GPU budget {vram:.1f} GB"
                ],
                "risk": "low",
                "verified_by": "harness",
            })
            spec_dict["assumptions"] = assumptions
            try:  # keep the on-disk spec consistent with the returned dict
                import json as _json
                (ctx.project_dir / "environment_spec.json").write_text(
                    _json.dumps(spec_dict, indent=2, default=str), encoding="utf-8"
                )
            except OSError:
                logger.debug("detect_environment: environment_spec.json rewrite failed")
    except Exception:  # noqa: BLE001 — capacity annotation must never block detection
        logger.debug("detect_environment: runtime-capacity annotation skipped", exc_info=True)
    result = _with_outcome(spec_dict, PrimitiveOutcome.ok)
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

    # Select the cloud provider from the run's sandbox_mode: azure → azure SKUs
    # (ONDEMAND tier, multi-GPU VM-size aware); everything else → runpod (default,
    # byte-for-byte identical to the pre-azure behaviour).
    from backend.agents.execution import SandboxMode as _SandboxMode
    _sb_mode = getattr(ctx, "sandbox_mode", None)
    try:
        _sb_enum = _SandboxMode(_sb_mode) if _sb_mode is not None else None
    except (ValueError, TypeError):
        _sb_enum = None
    _is_azure = _sb_enum is _SandboxMode.azure

    # ``provisioned_skus`` restricts the azure resolver to the GPU pools that are
    # actually provisioned (Terraform ``var.gpu_skus`` ⇒ ``settings.azure_gpu_skus``),
    # so the primary pick + OOM escalation ladder can never name a pool that does
    # not exist (which would otherwise hang the cell Pending until the
    # capacity-exhausted timeout). ``None`` for non-azure leaves the runpod path
    # byte-for-byte unchanged.
    if _is_azure:
        _provider = "azure"
        cloud_types: tuple[str, ...] = ("ONDEMAND",)
        _provisioned_skus: tuple[str, ...] | None = tuple(
            getattr(settings, "azure_gpu_skus", None) or ()
        ) or None
    else:
        _provider = "runpod"
        cloud_types = (
            ("COMMUNITY", "SECURE")
            if getattr(settings, "runpod_cloud_type", "COMMUNITY") == "SECURE"
            else ("COMMUNITY",)
        )
        _provisioned_skus = None

    from backend.agents.schemas import GpuPlan as _GpuPlan
    plan: "_GpuPlan" = gpu_resolver.resolve(
        req,
        dynamic_gpu_enabled=settings.dynamic_gpu_enabled,
        force_single_gpu=settings.force_single_gpu,
        max_gpu_usd_per_hour=settings.max_gpu_usd_per_hour or None,
        headroom_multiplier=settings.dynamic_gpu_headroom,
        fallback_vram_gb=settings.dynamic_gpu_fallback_vram_gb,
        cloud_types=cloud_types,
        provider=_provider,
        provisioned_skus=_provisioned_skus,
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


def _normalize_runpod_from_line(dockerfile: str) -> str:
    """Replace a hallucinated runpod/ base image tag with the configured one.

    The root model sometimes constructs env_spec dicts with non-existent runpod
    image tags (e.g. ``runpod/pytorch:1.12.1``).  When the FROM line references
    any ``runpod/`` image that doesn't match the settings-configured
    ``OPENRESEARCH_RUNPOD_IMAGE`` (``config.runpod_image``), swap it in.
    Non-runpod FROM lines (e.g.
    ``python:3.11-slim``) are left untouched — they're valid CPU images.
    """
    from backend.config import get_settings

    configured = get_settings().runpod_image
    lines = dockerfile.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.upper().startswith("ARG"):
            continue
        if stripped.upper().startswith("FROM "):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1].startswith("runpod/") and parts[1] != configured:
                logger.warning(
                    "build_environment: replacing hallucinated FROM %s "
                    "with configured %s",
                    parts[1],
                    configured,
                )
                parts[1] = configured
                lines[i] = " ".join(parts)
                return "\n".join(lines)
            break  # first FROM found and is fine (or non-runpod) — stop
    return dockerfile


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
    if _sb_key == "azure":
        return _with_outcome({
            "ok": True,
            "image_tag": "",
            "attempts": 0,
            "skipped": True,
            "note": "azure sandbox: image is pre-baked in ACR; build_environment is a no-op",
        }, PrimitiveOutcome.ok)

    # RunPod sandbox: the pod pulls its base image from Docker Hub directly —
    # a locally-built image is never pushed to a registry, so local Docker is
    # unnecessary.  Short-circuit with the configured RunPod image so
    # run_experiment can pass it to the RunPod backend (which uses
    # self.image_name with higher priority anyway).  Dependencies from
    # requirements.txt are installed on the pod via SSH bootstrap.
    if _sb_key == "runpod":
        from backend.config import get_settings as _get_settings
        _runpod_image = _get_settings().runpod_image
        return _with_outcome({
            "ok": True,
            "image_tag": _runpod_image,
            "attempts": 0,
            "skipped": True,
            "note": f"runpod sandbox: pod pulls {_runpod_image} from Docker Hub; no local build needed",
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

    # Deterministic shape guard (BUG-NEW-042): fail fast — before a wasted
    # `docker build` — when the sub-agent dumped prose instead of a Dockerfile.
    # failure_class=dockerfile_invalid is repairable, so the root re-derives.
    if not _validate_dockerfile_shape(dockerfile):
        return _with_outcome({
            "ok": False,
            "image_tag": "",
            "error": (
                "env_spec.dockerfile does not look like a Dockerfile: its first "
                "non-blank/non-comment line is not FROM/ARG (looks like prose). "
                "Emit a valid Dockerfile beginning with FROM."
            ),
            "error_code": "dockerfile_shape_guard",
            "failure_class": "dockerfile_invalid",
            "attempts": 0,
        }, PrimitiveOutcome.repairable)

    # Normalize runpod/ FROM line (ported from feat/rlm-wedge-hardening
    # 82e9806; gate dropped 2026-06-09). The root model sometimes hallucinates
    # a non-existent runpod image tag (e.g. runpod/pytorch:1.12.1) — a
    # guaranteed manifest-not-found at `docker build`. Unconditional on the
    # build paths that reach here (docker/auto/local-docker; the runpod
    # sandbox returned via its short-circuit above, which made the old
    # `if _sb_key == "runpod"` gate unreachable dead code). The function
    # self-gates: only runpod/ FROM lines that mismatch the configured image
    # are rewritten; python:3.11-slim etc. pass through untouched.
    dockerfile = _normalize_runpod_from_line(dockerfile)

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
        tag = f"openresearch/{ctx.project_id}:env-{digest}"

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
                    # Agent-visible feedback (2026-06-09): this fired on 15/15
                    # Adam+All-CNN attempts because the dashboard warning never
                    # reaches the root. Put the exact expected shape ON the
                    # returned plan so the agent can self-correct, and the
                    # compute-adjusted grading path stops being silently lost.
                    data.setdefault("warnings", []).append(
                        "compute_scope was a string and was DROPPED. If you "
                        "reduced compute vs the paper, resend it as a JSON "
                        "object: {\"is_clipped\": true, \"paper_epochs\": N, "
                        "\"actual_epochs\": M, \"rationale\": \"...\", "
                        "\"metric_floors\": [{...}]} — otherwise omit it."
                    )
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

        contract_out = ReproductionContract(**data).model_dump()
        # ReproductionContract is extra="ignore" — re-attach harness feedback
        # (e.g. the compute_scope shape correction) AFTER the dump so the agent
        # actually sees it on the returned plan instead of it being silently
        # stripped with the other unknown keys.
        if data.get("warnings"):
            contract_out["warnings"] = list(data["warnings"])
        result = _with_outcome(contract_out, PrimitiveOutcome.ok)
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


def _baseline_subprocess_enabled() -> bool:
    """Run the baseline SDK call in an isolated child process (OPT-IN, default off).

    Set ``OPENRESEARCH_BASELINE_SUBPROCESS=1`` to isolate the claude-agent-sdk call in
    a fresh process so its ``aclose()`` async-gen race crashes only that child and
    can't poison the reproduction process. Default OFF: the in-process path is the
    long-standing behavior the unit tests mock (``run_with_sdk`` patched in-process
    is bypassed by the spawn child), and the isolation only covers
    ``implement_baseline`` — the *root* model uses the same SDK, so this is a
    partial mitigation, kept opt-in until a full root-level fix lands.
    """
    return os.environ.get("OPENRESEARCH_BASELINE_SUBPROCESS", "0").strip().lower() in ("1", "true", "yes")


def _drive_baseline_child(
    *,
    heartbeat_path: str,
    project_id: str,
    runs_root: "Path",
    pcm,
    env,
    contract,
    artifact_index,
    kwargs: dict,
    sdk_activity: dict,
    child_holder: dict,
):
    """Run the baseline SDK call in a fresh ``multiprocessing`` (spawn) process.

    Runs on the implement_baseline worker thread. Spawns the child, parks its
    handle in ``child_holder`` (so the caller's stall/timeout/finally path can
    terminate it), and while it runs forwards the child's heartbeat-file mtime
    into ``sdk_activity['last']`` so the EXISTING file+stream stall watchdog works
    unchanged. Returns an object exposing ``commands_to_run`` / ``diff_summary`` /
    ``assumptions_applied`` on success; raises on child failure so the caller's
    ``except`` path harvests artifacts and marks the attempt repairable — and the
    *next* attempt gets a brand-new, un-poisoned process.
    """
    import multiprocessing as _mp
    from pathlib import Path
    from types import SimpleNamespace as _NS

    from backend.agents.rlm.baseline_runner import run_baseline_in_child

    hb = Path(heartbeat_path)
    try:
        hb.touch()
    except Exception:  # noqa: BLE001
        pass
    ctxmp = _mp.get_context("spawn")
    result_q = ctxmp.Queue()
    proc = ctxmp.Process(
        target=run_baseline_in_child,
        args=(result_q, str(hb), project_id, str(runs_root), pcm, env, contract, artifact_index, kwargs),
        daemon=False,
    )
    proc.start()
    child_holder["p"] = proc
    while proc.is_alive():
        proc.join(timeout=2.0)
        try:
            if hb.exists():
                sdk_activity["last"] = max(sdk_activity["last"], hb.stat().st_mtime)
        except Exception:  # noqa: BLE001
            pass
    # Child exited — collect the result (empty queue ⇒ hard crash, e.g. the race).
    payload = None
    try:
        payload = result_q.get_nowait()
    except Exception:  # noqa: BLE001
        payload = None
    if payload is None:
        raise RuntimeError(
            f"baseline child exited (code={proc.exitcode}) without a result — "
            "likely the claude-agent-sdk aclose race; a fresh child will retry"
        )
    if not payload.get("ok"):
        raise RuntimeError(f"baseline child failed: {payload.get('error', 'unknown')}")
    return _NS(
        commands_to_run=payload.get("commands_to_run", []),
        diff_summary=payload.get("diff_summary", ""),
        assumptions_applied=payload.get("assumptions_applied", []),
    )


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

    # SDK-stream liveness signal for robust stall detection. The code-writing sub-agent
    # runs in a worker thread (pool.submit below); the main thread polls code_dir for
    # progress, but file mtimes only change when a Write COMPLETES — so a model that
    # reasons for minutes or streams a large file looks "stalled" to a file-only
    # watchdog (the 2026-05-29 false-stall). collect_agent_text calls _note_sdk_event()
    # on EVERY streamed event, giving the poll loop a precise "SDK is alive and
    # producing" signal. _sdk_activity["last"] is written by the worker thread and read
    # by the main thread; a lone float write/read is atomic under the GIL.
    # comp 3 (2026-05-31 OOM/GPU remediation): hand the code-writing agent its
    # per-GPU budget + the single-cell contract when the backend exposes GPUs
    # (local/docker). describe_capacity is fail-soft; on a failure or a CPU/cloud
    # backend the budget is None and the guidance is byte-identical to before.
    try:
        from backend.services.runtime.gpu_capacity import describe_capacity as _describe_capacity
        _caps = _describe_capacity(ctx)
        _gpu_cell_budget: dict | None = {
            "backend_kind": _caps.backend_kind,
            "num_gpus": _caps.num_gpus,
            "per_gpu_vram_gb": _caps.per_gpu_vram_gb,
        }
    except Exception:  # noqa: BLE001 — a capacity probe must never block code-writing
        logger.debug("implement_baseline: describe_capacity failed; no cell budget", exc_info=True)
        _gpu_cell_budget = None

    import time as _time
    _sdk_activity = {"last": _time.time()}

    def _note_sdk_event() -> None:
        _sdk_activity["last"] = _time.time()

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
            # comp 3: per-GPU budget + single-cell contract for the cell path.
            gpu_cell_budget=_gpu_cell_budget,
            # Liveness hook: bumps _sdk_activity on every streamed SDK event so the
            # stall watchdog distinguishes a working agent from a hung SDK.
            on_event=_note_sdk_event,
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

    _child_holder: dict = {}
    try:
        if _baseline_subprocess_enabled():
            _sub_kwargs = dict(
                model=ctx.agent_model,
                repair_context=repair_context,
                sandbox_mode=ctx.sandbox_mode,
                gpu_mode=getattr(ctx, "gpu_mode", None),
                arxiv_id=getattr(ctx, "arxiv_id", None),
                remaining_s=ctx.remaining_s(),
                minimize_compute=getattr(ctx, "minimize_compute", False),
                metrics_shape=_metrics_shape or None,
                data_recipes=_data_recipes or None,
                gpu_parallelism=getattr(ctx, "gpu_parallelism", None),
                gpu_visible_count=getattr(ctx, "gpu_visible_count", None),
                gpu_cell_budget=_gpu_cell_budget,
            )
            # Isolate the SDK call in a fresh process so its aclose() async-gen
            # race can't poison this reproduction process. The driver runs on the
            # worker thread and feeds the child's heartbeat into the existing
            # stall watchdog; child_holder lets the finally terminate it.
            future = pool.submit(
                _drive_baseline_child,
                heartbeat_path=str(code_dir / ".sdk_heartbeat"),
                project_id=ctx.project_id,
                runs_root=ctx.runs_root,
                pcm=pcm,
                env=env,
                contract=contract,
                artifact_index=artifact_index,
                kwargs=_sub_kwargs,
                sdk_activity=_sdk_activity,
                child_holder=_child_holder,
            )
        else:
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
                # Progress = a file write OR live SDK-stream activity. The latter is the
                # robust signal: a sub-agent reasoning or generating a large file emits
                # stream events continuously even before any file lands, so a healthy
                # agent never trips the stall. Only TRUE silence (no file AND no stream
                # event since the timer start) is treated as a genuine SDK hang.
                latest_progress = max(latest_mtime, _sdk_activity["last"])
                if latest_progress > _pre_emit_stall_start:
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
                # Same robust signal as the pre-emit path: live SDK-stream activity
                # (e.g. the agent still streaming its worker-report) counts as progress,
                # so a genuine aclose hang (stream ended, code on disk) is still caught
                # while a still-streaming agent is not falsely declared hung.
                latest_progress = max(latest_mtime, _sdk_activity["last"])
                if now - latest_progress > _POLL_S:
                    # No file changes AND no SDK activity — start/continue the stall timer
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
        # Terminate any still-running isolated baseline child (stall/timeout path).
        _p = _child_holder.get("p")
        if _p is not None:
            try:
                _p.terminate()
                _p.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
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


def codex_repair(
    task_type: str,
    instructions: str,
    test_command: str,
    allowed_paths: list[str] | tuple[str, ...] | str | None = None,
    repair_context: dict | None = None,
    failure_class: str | None = None,
    readonly: bool = False,
    *,
    ctx: "RunContext",
) -> dict:
    """Deliberately invoke Codex CLI for bounded repo repair.

    This primitive is default-off and gated by task type, failure class, and a
    per-run call budget. It is not used by ``rlm_query`` and is not a provider
    for paper navigation, summarization, rubric judgment, or final reports.
    """
    from pathlib import Path

    from backend.config import get_settings
    from backend.agents.rlm.codex_subagent import run_codex_subagent

    settings = get_settings()
    if not bool(getattr(settings, "codex_subagent", False)):
        return _codex_repair_error(
            "disabled",
            "codex_repair is disabled; set OPENRESEARCH_CODEX_SUBAGENT=1 to enable it.",
        )

    normalized_task = str(task_type or "").strip().lower().replace("-", "_")
    if normalized_task in _CODEX_REJECTED_TASKS:
        return _codex_repair_error(
            "task_type_rejected",
            f"codex_repair rejected task_type={task_type!r}; this route is only for repo repair.",
        )
    env_allowed = _codex_env_allowed_tasks(getattr(settings, "codex_allowed_tasks", ""))
    if normalized_task not in _CODEX_HARD_ALLOWED_TASKS or normalized_task not in env_allowed:
        return _codex_repair_error(
            "task_type_not_allowed",
            f"codex_repair task_type={task_type!r} is not enabled by OPENRESEARCH_CODEX_ALLOWED_TASKS.",
            allowed_tasks=sorted(env_allowed & _CODEX_HARD_ALLOWED_TASKS),
        )

    klass = _codex_failure_class(failure_class, repair_context)
    if klass not in _CODEX_AGENT_CORRECTABLE_FAILURES:
        return _codex_repair_error(
            "failure_class_not_allowed",
            (
                "codex_repair requires a failed run_experiment repair_context or "
                "failure_class in: "
                + ", ".join(sorted(_CODEX_AGENT_CORRECTABLE_FAILURES))
            ),
            failure_class=klass or None,
        )
    if isinstance(repair_context, dict) and repair_context.get("success") is True:
        return _codex_repair_error(
            "experiment_not_failed",
            "codex_repair only runs after a failed experiment result.",
            failure_class=klass,
        )

    call_count = int(getattr(ctx, "_codex_subagent_calls", 0) or 0)
    max_calls = int(getattr(settings, "codex_max_calls_per_run", 3) or 0)
    if call_count >= max_calls:
        return _codex_repair_error(
            "max_calls_exceeded",
            f"codex_repair call budget exhausted ({call_count}/{max_calls}).",
            calls_used=call_count,
            max_calls=max_calls,
        )

    if isinstance(allowed_paths, str):
        allowed = [allowed_paths]
    else:
        allowed = [str(p) for p in (allowed_paths or [])]
    allowed = [p.strip() for p in allowed if p and p.strip()]
    instructions = str(instructions or "").strip()
    test_command = str(test_command or "").strip()
    if not instructions:
        return _codex_repair_error("invalid_request", "codex_repair instructions must be non-empty.")
    if not test_command:
        return _codex_repair_error("invalid_request", "codex_repair test_command must be non-empty.")

    timeout_s = int(getattr(settings, "codex_timeout_s", 900) or 900)
    max_output_chars = int(getattr(settings, "codex_max_output_chars", 12000) or 12000)
    profile = str(getattr(settings, "codex_profile", "") or "").strip() or None
    prompt = _build_codex_prompt(
        task_type=normalized_task,
        instructions=instructions,
        test_command=test_command,
        allowed_paths=allowed,
        timeout_s=timeout_s,
        failure_class=klass,
    )

    def _event_sink(event_type: str, payload: dict) -> None:
        _emit_dashboard_event(ctx, event_type=event_type, payload=payload)

    setattr(ctx, "_codex_subagent_calls", call_count + 1)
    result = run_codex_subagent(
        prompt=prompt,
        workspace=Path(ctx.project_dir),
        timeout_s=timeout_s,
        profile=profile,
        readonly=bool(readonly),
        task_type=normalized_task,
        max_output_chars=max_output_chars,
        event_sink=_event_sink,
    )
    payload = result.as_dict()
    payload["success"] = result.ok
    payload["task_type"] = normalized_task
    payload["failure_class"] = klass
    payload["calls_used"] = call_count + 1
    payload["max_calls"] = max_calls

    if allowed and result.changed_files:
        allowed_prefixes = [p.rstrip("/") + "/" for p in allowed if not p.startswith("..")]
        allowed_exact = {p.rstrip("/") for p in allowed if not p.startswith("..")}
        outside = [
            changed for changed in result.changed_files
            if changed not in allowed_exact
            and not any(changed.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if outside:
            payload["ok"] = False
            payload["success"] = False
            payload["error_type"] = "changed_files_outside_allowed_paths"
            payload["error"] = "codex_repair changed files outside allowed_paths"
            payload["outside_allowed_paths"] = outside

    return _with_outcome(
        payload,
        PrimitiveOutcome.ok if payload.get("ok") else PrimitiveOutcome.repairable,
    )


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

    if mode is SandboxMode.azure:
        import backend.services.runtime as _runtime
        from backend.services.runtime.aks_job_backend import AksJobBackend

        _runtime.ensure_azure_available()
        return AksJobBackend(run_budget=run_budget, gpu_plan=gpu_plan)

    # All other modes (auto, brev, simulate) are not yet wired
    # for the RLM path.  Fall back with a loud WARNING so the operator knows.
    logger.warning(
        "_execute_in_sandbox: sandbox_mode=%r is not supported in the RLM "
        "path — falling back to LocalDockerBackend.  "
        "Set --sandbox docker, --sandbox runpod, or --sandbox azure for a supported backend.",
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


# Scalar fields that unambiguously mean "completed optimizer steps" (NOT a target
# and NOT a per-step list — those are handled separately by the curve extractor).
_STEP_COUNT_KEYS = frozenset({
    "train_steps", "global_step", "optimizer_steps", "steps_completed", "completed_steps",
})


def _max_train_steps(metrics: dict) -> int | None:
    """Largest completed-step count recorded anywhere in a metrics tree (or None).

    Keys on any of :data:`_STEP_COUNT_KEYS` (agents name the field freely) so the
    convergence-floor and degeneracy checks don't silently no-op on a paper that
    emits ``global_step`` instead of ``train_steps``.
    """
    best: int | None = None

    def _walk(obj: object) -> None:
        nonlocal best
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _STEP_COUNT_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    best = int(v) if best is None else max(best, int(v))
                else:
                    _walk(v)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                _walk(x)

    _walk(metrics)
    return best


_OOM_LOG_MARKERS = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "outofmemoryerror",
    "backward oom",
    "loss/backward oom",
)

# A per-model status the agent uses to claim the model trained successfully.
_OK_STATUSES = frozenset({"ok", "success", "completed", "complete", "done"})


def _reward_curve(mv: dict) -> list[float]:
    """The reward time-series from a per-model metrics dict (for variance checks)."""
    tc = mv.get("training_curves") if isinstance(mv, dict) else None
    raw: object = []
    if isinstance(tc, dict):
        raw = tc.get("reward") or tc.get("rewards") or tc.get("mean_reward") or []
    elif isinstance(tc, list):
        raw = [d.get("reward") for d in tc if isinstance(d, dict)]
    if not raw and isinstance(mv, dict) and isinstance(mv.get("reward_history"), list):
        raw = mv["reward_history"]
    if not isinstance(raw, (list, tuple)):
        return []
    return [float(x) for x in raw if isinstance(x, (int, float)) and not isinstance(x, bool)]


def _scalar_rewards(mv: dict) -> list[float]:
    """Scalar OUTCOME reward values in a per-model subtree — keys ENDING in 'reward'
    (searchqa_reward / mean_reward / final_reward …), excluding stat/config fields
    (reward_std / reward_scale / reward_clip / baseline_reward, which don't end in
    'reward' or carry a 'baseline' marker) so config constants can't fake a signal."""
    out: list[float] = []

    def _w(o: object) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and kl.endswith("reward") and "baseline" not in kl):
                    out.append(float(v))
                else:
                    _w(v)
        elif isinstance(o, (list, tuple)):
            for x in o:
                _w(x)

    _w(mv)
    return out


def _degenerate_training_violation(metrics: dict, *, epsilon: float = 1e-6) -> tuple[str, str] | None:
    """Flag a model the agent marked succeeded that shows NO learning signal.

    Conservative (a false-positive wrongly FAILS a healthy run, so only the
    unambiguous cases fire), per-model so a mixed run names the offender:
      (1) status=ok but an EXPLICIT 0 optimizer steps — 'completed' without training;
      (2) status=ok but EVERY recorded reward MAGNITUDE is ~0 (|reward| <= epsilon
          across the whole curve + scalar reward fields) — broken matching / no signal.
    Uses ``abs`` so a legitimately NEGATIVE reward (step/length/KL penalties are
    normal in RL) is NOT mistaken for "no reward"; and does NOT flag a constant but
    non-zero curve (that can be a converged plateau, not degeneracy). Only judges
    models whose status is in :data:`_OK_STATUSES`.
    """
    per_model = metrics.get("per_model")
    if not isinstance(per_model, dict):
        return None
    for m, mv in per_model.items():
        if not isinstance(mv, dict) or str(mv.get("status", "")).lower() not in _OK_STATUSES:
            continue
        # (1) claimed success but explicitly zero optimizer steps.
        if _max_train_steps(mv) == 0:
            return ("degenerate_training",
                    f"degenerate_training: model {m!r} status=ok but ran 0 optimizer steps "
                    "— it 'completed' without training. Ensure the loop runs optimizer.step() "
                    "and records steps/reward.")
        # (2) every recorded reward magnitude ~0 — no signal at all.
        rewards = _reward_curve(mv) + _scalar_rewards(mv)
        if rewards and max(abs(x) for x in rewards) <= epsilon:
            return ("degenerate_training",
                    f"degenerate_training: model {m!r} status=ok but EVERY recorded reward "
                    f"is ~0 ({len(rewards)} value(s)) — training produced no signal. Fix the "
                    "reward/answer-matching so it is non-zero BEFORE the RL loop (extract the "
                    "answer span; token-F1 over the full gold-alias list; print zero-shot "
                    "accuracy first), and confirm optimizer.step() actually runs.")
    return None


_NON_TERMINAL_STATUSES = frozenset(
    {"running", "in_progress", "in-progress", "pending", "started", "queued", "init", "initializing"}
)


def _per_model_has_measured_value(mv: dict) -> bool:
    """True iff a per-model metrics entry carries ANY measured numeric value.

    Reuses the reward value-walkers; also accepts any finite numeric leaf
    (accuracy, loss, steps, return, …) nested anywhere in the entry. An empty
    ``{}`` placeholder → False.
    """
    if not isinstance(mv, dict) or not mv:
        return False
    if _reward_curve(mv) or _scalar_rewards(mv):
        return True

    def _any_number(o) -> bool:
        if isinstance(o, bool):
            return False
        if isinstance(o, (int, float)):
            return True
        if isinstance(o, dict):
            return any(_any_number(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return any(_any_number(v) for v in o)
        return False

    return _any_number(mv)


def _metrics_completeness_violation(result: dict) -> tuple[str, str] | None:
    """Detect a ``success=True`` run whose metrics are a placeholder / unpopulated.

    Every other postflight guard keys on presence/shape/exit-code; this one keys
    on whether measured VALUES exist. A ``train.py`` that writes
    ``{status:"running", per_model:{m:{}}}`` and exits 0 otherwise sails through
    (``success`` is exit-code-only; the placeholder is non-empty so ``degraded``
    stays False) and the rubric grades a half-finished experiment ~0 on
    eval/result/execution. Catching it here flips it to a repairable failure so
    the loop must re-run to REAL measured numbers before it can score or finalize.
    Opt out with ``OPENRESEARCH_METRICS_COMPLETENESS_CHECK=0``. Returns
    ``(failure_class, message)`` or ``None``. See
    docs/superpowers/specs/2026-05-30-rubric-scoring-fidelity-design.md.
    """
    import os as _os

    if _os.environ.get("OPENRESEARCH_METRICS_COMPLETENESS_CHECK", "1").strip().lower() in ("0", "false", "no"):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None  # genuinely-empty metrics → handled by the degraded/empty path

    # (1) Non-terminal top-level status → a placeholder written before results landed.
    status = str(metrics.get("status") or "").strip().lower()
    if status in _NON_TERMINAL_STATUSES:
        return (
            "incomplete_metrics",
            f"incomplete_metrics: metrics.json status={status!r} is non-terminal — the "
            "training script wrote a placeholder and exited before measuring results, so "
            "the rubric scores eval/result/execution ~0. Run training to completion and at "
            "the END set a terminal status and populate per_model[<model>] with the "
            "measured eval metric(s) for every model you ran.",
        )

    # (2) per_model present but EVERY entry is an empty placeholder (no measured value).
    per_model = metrics.get("per_model")
    if isinstance(per_model, dict) and per_model:
        measured = [
            m for m, mv in per_model.items()
            if _per_model_has_measured_value(mv if isinstance(mv, dict) else {})
        ]
        if not measured:
            keys = ", ".join(map(str, list(per_model.keys())[:4]))
            return (
                "incomplete_metrics",
                f"incomplete_metrics: per_model has {len(per_model)} model key(s) ({keys}) "
                "but NONE carry a measured value — the entries are empty placeholders. "
                "Populate per_model[<model>] with the measured eval metric (e.g. accuracy) "
                "and reward/loss for every model that actually ran; an empty {} entry is "
                "treated as 'not measured'.",
            )
    return None


def _training_health_violation(result: dict) -> tuple[str, str] | None:
    """Detect an experiment that exited 0 but did not really train.

    (1) ``silent_oom`` — the script logged a CUDA OOM yet exited 0: it caught the
        backward OOM and skipped the step, so no gradients were applied and the
        metrics are meaningless (the 2026-05-29 SDAR hours-of-grinding failure).
    (2) ``insufficient_train_steps`` — fewer optimizer steps than
        ``OPENRESEARCH_MIN_TRAIN_STEPS`` (opt-in; default 0 = disabled).

    Returns ``(failure_class, message)`` or None. The message becomes repair_context
    so the next implement_baseline reduces memory / trains longer.
    """
    import os as _os

    low = (result.get("logs") or "").lower()
    if any(m in low for m in _OOM_LOG_MARKERS):
        return (
            "silent_oom",
            "silent_oom: the training script logged a CUDA out-of-memory but exited 0 "
            "(it caught the backward OOM and skipped the step), so NO gradient updates "
            "happened and the metrics are meaningless. If multiple GPUs are leased this "
            "means your training did NOT shard — wrap the model with HuggingFace "
            "Accelerate (`accelerator.prepare(model, optimizer)`); the harness launches "
            "you under `accelerate launch` with an FSDP2 config when >1 GPU is leased, so "
            "params/grads/optimizer shard across the cards. Otherwise reduce per-step "
            "memory (smaller batch, fewer rollouts, gradient_checkpointing). Do NOT "
            "catch+skip a backward OOM — let it fail loudly so it can be repaired.",
        )

    try:
        min_steps = int(_os.environ.get("OPENRESEARCH_MIN_TRAIN_STEPS", "0") or "0")
    except ValueError:
        min_steps = 0
    if min_steps > 0:
        steps = _max_train_steps(result.get("metrics") or {})
        if steps is not None and steps < min_steps:
            return (
                "insufficient_train_steps",
                f"insufficient_train_steps: training ran only {steps} optimizer step(s) "
                f"(< OPENRESEARCH_MIN_TRAIN_STEPS={min_steps}). Sparse-reward tasks cannot "
                f"learn in so few steps — increase epochs/steps so total updates "
                f">= {min_steps}.",
            )

    # (2b) insufficient_training (NO-SMOKES) — exited 0 with metrics but ran far too
    # briefly to be REAL training. A seconds-long smoke (CPU stub / surrogate / no real
    # weights) must never be the scored reproduction; loading the paper's real models and
    # running the RL loop takes minutes, not seconds. Opt-in like the step floor above:
    # OPENRESEARCH_MIN_TRAIN_WALL_S (seconds; default 0 = disabled) is the minimum plausible
    # wall-clock for a real training of THIS paper — a wall-time floor is inherently
    # paper-specific (an inference-only paper legitimately finishes in seconds), so it is
    # opt-in per run, never a global default. A run shorter than the floor BUT showing
    # substantial optimizer progress (>= OPENRESEARCH_MIN_REAL_TRAIN_STEPS, default 5) is
    # exempted, so a genuinely fast-but-real run can never be false-flagged. Motivated by
    # the 2026-05-29 SDAR failure that scored a 2 s smoke after real FSDP training crashed.
    try:
        wall_floor = float(_os.environ.get("OPENRESEARCH_MIN_TRAIN_WALL_S", "0") or "0")
    except ValueError:
        wall_floor = 0.0
    wall = result.get("wall_time_s")
    health_metrics = result.get("metrics")
    if (
        wall_floor > 0
        and isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and wall < wall_floor
        and isinstance(health_metrics, dict)
        and health_metrics
    ):
        try:
            step_exempt = int(_os.environ.get("OPENRESEARCH_MIN_REAL_TRAIN_STEPS", "5") or "5")
        except ValueError:
            step_exempt = 5
        wall_steps = _max_train_steps(health_metrics)
        if not (wall_steps is not None and wall_steps >= step_exempt):
            _steps_phrase = (
                f"{wall_steps} optimizer step(s)"
                if wall_steps is not None
                else "no recorded optimizer steps"
            )
            return (
                "insufficient_training",
                f"insufficient_training: the experiment exited 0 with metrics but ran only "
                f"{wall:.1f}s wall-clock ({_steps_phrase}) — below the "
                f"OPENRESEARCH_MIN_TRAIN_WALL_S={wall_floor:.0f}s floor for a REAL training of this "
                f"paper's models. That is a SMOKE / trivial run, not a faithful reproduction, and "
                f"MUST NOT be scored. Run the FULL training — real pretrained weights, real "
                f"episodes, optimizer.step() each iteration — to completion and record the measured "
                f"eval metric for every model before finalizing. (A run with >= {step_exempt} "
                f"optimizer steps is exempt from this floor.)",
            )

    # (3) degenerate_training — exited 0, status=ok, but no learning signal (constant
    # / all-zero reward or 0 steps). Opt-in (default on); disable with =0.
    if _os.environ.get("OPENRESEARCH_DEGENERATE_TRAINING_CHECK", "1").strip().lower() not in ("0", "false", "no"):
        try:
            _deg_eps = float(_os.environ.get("OPENRESEARCH_DEGENERATE_REWARD_EPSILON", "1e-6") or "1e-6")
        except ValueError:
            _deg_eps = 1e-6
        _deg = _degenerate_training_violation(result.get("metrics") or {}, epsilon=_deg_eps)
        if _deg is not None:
            return _deg
    return None


_DISTRIBUTED_MARKERS = (
    "FullyShardedDataParallel",
    "DistributedDataParallel",
    "init_process_group",
    "torch.distributed",
    "fully_shard",       # FSDP2 (torch.distributed.fsdp.fully_shard)
    "from accelerate",   # HuggingFace Accelerate API
    "import accelerate",
    "Accelerator(",
)


def _free_tcp_port() -> int:
    """Return an OS-assigned free TCP port for the accelerate rendezvous.

    Concurrent reproductions on the same host (``batch_reproduce`` over several
    papers) each launch their own ``accelerate launch``; without a distinct
    ``--main_process_port`` they collide on the default 29500 and the second run
    hangs the rendezvous. Binding to port 0 and reading the assigned port back
    guarantees disjoint endpoints.
    """
    import socket as _socket

    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _write_fsdp_accelerate_config(code_dir: "Path", nproc: int) -> "Path":
    """Write a harness-owned FSDP ``accelerate`` config into the run's code dir.

    Defaults to **FSDP1** (FULL_SHARD): it shards params/grads/optimizer
    identically to FSDP2 for our purpose and runs on torch >= 1.12, whereas
    FSDP2 (``fsdp_version: 2``) requires **torch >= 2.6** — this host is pinned to
    torch 2.5.1 by the cu121 wheel index (CUDA-12.2 driver), so FSDP2 errors at
    launch (validated 2026-05-30). Set ``OPENRESEARCH_FSDP_VERSION=2`` on a
    torch>=2.6 environment (e.g. a newer RunPod image) to use the per-parameter
    FSDP2 API. Either way the harness owns the sharding policy (full shard, bf16,
    transformer auto-wrap — Qwen exposes ``_no_split_modules`` so the layer class
    is auto-detected, no CPU offload) so a *correct* shard engages regardless of
    how the agent wired its loop; the agent only calls
    ``accelerator.prepare(model, optimizer)``. Consumed relative to ``code_dir``
    (the execution cwd).
    """
    import os as _os

    version = (_os.environ.get("OPENRESEARCH_FSDP_VERSION", "1") or "1").strip()
    if version not in ("1", "2"):
        version = "1"
    cfg = (
        "compute_environment: LOCAL_MACHINE\n"
        "distributed_type: FSDP\n"
        "mixed_precision: bf16\n"
        "use_cpu: false\n"
        "machine_rank: 0\n"
        "num_machines: 1\n"
        f"num_processes: {nproc}\n"
        "fsdp_config:\n"
        f"  fsdp_version: {version}\n"
        "  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP\n"
        "  fsdp_offload_params: false\n"
        "  fsdp_state_dict_type: SHARDED_STATE_DICT\n"
        "  fsdp_cpu_ram_efficient_loading: true\n"
    )
    if version == "2":
        cfg += "  fsdp_reshard_after_forward: true\n"
    else:
        cfg += (
            "  fsdp_sharding_strategy: FULL_SHARD\n"
            "  fsdp_use_orig_params: true\n"
        )
    path = code_dir / "_reprolab_fsdp.yaml"
    try:
        path.write_text(cfg, encoding="utf-8")
    except Exception:  # noqa: BLE001 — best-effort; caller still launches distributed
        logger.exception("_write_fsdp_accelerate_config: write failed")
    return path


def _nccl_env_prefix() -> str:
    """Inline NCCL env that prevents the first-collective BROADCAST hang on
    older-kernel multi-GPU hosts.

    On this 8xA5000 box (kernel 5.4.0, below torch's recommended 5.5.0) the first
    NCCL collective during FSDP setup hangs for the full 600s timeout at >2 GPUs
    unless P2P is disabled (validated 2026-05-30 — likely the real cause of the
    earlier "multi-GPU runs stall"). Defaults on; override per-var with
    ``OPENRESEARCH_NCCL_P2P_DISABLE=0`` / ``OPENRESEARCH_NCCL_IB_DISABLE=0`` on a
    well-connected box (e.g. NVLink RunPod) where P2P is fast and reliable.
    """
    import os as _os

    def _on(name: str) -> bool:
        return (_os.environ.get(name, "1") or "1").strip().lower() not in ("0", "false", "no")

    parts: list[str] = []
    if _on("OPENRESEARCH_NCCL_P2P_DISABLE"):
        parts.append("NCCL_P2P_DISABLE=1")
    if _on("OPENRESEARCH_NCCL_IB_DISABLE"):
        parts.append("NCCL_IB_DISABLE=1")
    return (" ".join(parts) + " ") if parts else ""


def _resolve_distributed_launch(
    commands: list[str], code_dir: "Path", ngpu: int, run_id: str = ""
) -> list[str]:
    """Dynamically re-launch a plain ``python <script>.py`` under
    ``accelerate launch`` + a harness FSDP2 config when >1 GPU is allocated and
    the script carries distributed/accelerate markers.

    Dynamic by design — the launch strategy is resolved at runtime from the
    *actually-leased* GPU count, never hardcoded:

    * ``ngpu <= 1`` → run verbatim (plain ``python`` / CPU). FSDP on a single
      card is pure all-gather overhead with zero memory benefit, so we never pay
      it; this is also the graceful fallback on a 1-GPU / no-GPU host.
    * ``ngpu >= 2`` + the script uses FSDP/accelerate → rewrite to
      ``accelerate launch --config_file <fsdp2.yaml> --num_processes <ngpu>`` so
      params/grads/optimizer shard across the leased cards (the 3B/7B that OOM a
      single 24 GB card fit comfortably sharded). ``accelerate launch`` is a
      strict superset of ``torchrun`` for launching: a raw ``torch.distributed``
      script still gets a correct process group; an accelerate-API script also
      gets the harness FSDP2 policy.

    No-op when the command is already a distributed launcher, when the script has
    no distributed markers, or when disabled via the escape-hatch toggle. The
    per-model fit choice (run the small model on one card, shard the big ones) is
    expressed in the agent's training code + run guidance; this seam only
    guarantees the *launch* is correct whenever the agent wrote shardable code.
    """
    import os as _os
    import re as _re

    if ngpu <= 1:
        return commands
    # Escape hatch — keep the agent's launch verbatim (operator override).
    if _os.environ.get("OPENRESEARCH_DISABLE_TORCHRUN_WRAP", "").strip().lower() in ("1", "true", "yes"):
        logger.info("_resolve_distributed_launch[%s]: disabled via OPENRESEARCH_DISABLE_TORCHRUN_WRAP", run_id)
        return commands
    # RL-scaffold sentinel — the scaffold owns its own launch orchestration
    # (vLLM server + accelerate trainer partition), so the harness rewriter
    # must NOT wrap the launch command again.  Three detection surfaces:
    #   1. Command-line marker: a command containing '# openresearch:rl-scaffold-owns-launch'
    #   2. Sentinel file:       code_dir/.openresearch_rl_scaffold exists
    #   3. Environment var:     OPENRESEARCH_RL_SCAFFOLD=1
    if _os.environ.get("OPENRESEARCH_RL_SCAFFOLD", "").strip().lower() in ("1", "true", "yes"):
        logger.info(
            "_resolve_distributed_launch[%s]: skipping rewrite — OPENRESEARCH_RL_SCAFFOLD=1 "
            "(scaffold owns launch)",
            run_id,
        )
        return commands
    if (code_dir / ".openresearch_rl_scaffold").exists():
        logger.info(
            "_resolve_distributed_launch[%s]: skipping rewrite — .openresearch_rl_scaffold "
            "sentinel file present (scaffold owns launch)",
            run_id,
        )
        return commands
    if any("# openresearch:rl-scaffold-owns-launch" in cmd for cmd in commands):
        logger.info(
            "_resolve_distributed_launch[%s]: skipping rewrite — "
            "'# openresearch:rl-scaffold-owns-launch' marker in commands (scaffold owns launch)",
            run_id,
        )
        return commands

    out: list[str] = []
    changed = False
    cfg_rel: str | None = None
    for cmd in commands:
        # Already a distributed launcher → leave alone.
        if _re.match(r"^\s*(accelerate\s+launch|torchrun|deepspeed)\b", cmd):
            out.append(cmd)
            continue
        m = _re.match(r"^\s*python3?\s+(\S+\.py)(\s.*)?$", cmd)
        if not m:
            out.append(cmd)
            continue
        script, rest = m.group(1), (m.group(2) or "")
        script_path = code_dir / script
        try:
            text = (
                script_path.read_text(encoding="utf-8", errors="replace")
                if script_path.exists()
                else ""
            )
        except Exception:  # noqa: BLE001
            text = ""
        if any(marker in text for marker in _DISTRIBUTED_MARKERS):
            if cfg_rel is None:
                cfg_rel = _write_fsdp_accelerate_config(code_dir, ngpu).name
            port = _free_tcp_port()
            out.append(
                f"{_nccl_env_prefix()}accelerate launch --config_file {cfg_rel} "
                f"--num_processes {ngpu} --num_machines 1 "
                f"--main_process_port {port} {script}{rest}"
            )
            changed = True
            logger.warning(
                "_resolve_distributed_launch[%s]: %s carries distributed/accelerate "
                "markers but was launched single-process (`%s`); re-launching via "
                "`accelerate launch --num_processes %d` (FSDP2) so params/grads/"
                "optimizer shard across the %d leased GPUs.",
                run_id, script, cmd.strip(), ngpu, ngpu,
            )
        else:
            out.append(cmd)
            # Proactive guard: multi-GPU leased but the script has no FSDP/accelerate
            # markers → it will run on one card and likely OOM a large model. The
            # OOM postflight additionally steers the repair toward accelerate+FSDP.
            logger.warning(
                "_resolve_distributed_launch[%s]: %d GPUs leased but `%s` has no "
                "FSDP/accelerate markers — it will use a single card and may OOM a "
                "large model. The training script should call "
                "`accelerator.prepare(model, optimizer)`.",
                run_id, ngpu, script,
            )
    return out if changed else commands


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
    per_command_timeout: int | None = None,
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
    _venv = (_os.environ.get("OPENRESEARCH_EXPERIMENT_VENV") or "").strip()
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
            "OPENRESEARCH_ARTIFACT_DIR": str(artifact_root) if _mode_str_local == "local" else "/artifacts",
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
        # Local: do NOT inject the dumb heartbeat daemon — it touches .heartbeat every 30 s
        # regardless of progress, which would mask an in-process hang from the watchdog (the
        # 2026-06-08 Adam silence). On local the real liveness comes from the Pillar-1 streaming
        # live-log + experiment_progress SSE + the GPU/CPU-aware inner stall, and the watchdog
        # now does its own GPU/CPU compute check. Keep the daemon for runpod/docker (it detects
        # genuine pod-level wedges where exec.log goes flat but the pod is alive).
        if _watchdog_enabled() and _mode_str_local != "local":
            bootstrap_commands.append(heartbeat_daemon_command("/artifacts"))
    except Exception:  # noqa: BLE001 — instrumentation MUST NOT block the run
        logger.exception("_execute_in_sandbox: heartbeat-daemon injection failed")

    # sandbox_mode may be a SandboxMode enum (str(...) is "SandboxMode.runpod")
    # OR a plain string "runpod". Use substring match to cover both forms.
    _mode_str = str(sandbox_mode).lower() if sandbox_mode else ""
    if "runpod" in _mode_str:
        # Lane 6: when OPENRESEARCH_BOOTSTRAP_MKDIRS is set by the RunPod backend
        # (because a network volume is mounted for persistent pip / HF cache),
        # create those dirs FIRST so pip and HuggingFace can write to them.
        # Pre-pip step — must run before any other bootstrap.
        bootstrap_commands.append(
            'mkdir -p ${OPENRESEARCH_BOOTSTRAP_MKDIRS:-/tmp/.reprolab_noop}'
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
            # The harness launches multi-GPU training via `accelerate launch`
            # (FSDP2); ensure a modern Accelerate is present regardless of the
            # agent's requirements.txt.
            bootstrap_commands.append(
                "python -m pip install -U accelerate"
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
    # Phase 2A (2026-06-07): synthesize requirements.txt on the LOCAL sandbox too.
    # Previously ONLY the runpod block above called ensure_requirements_txt, while the
    # local path gated on the file already existing — so a local run whose agent forgot
    # requirements.txt installed nothing and died at the first third-party import (the
    # matplotlib ModuleNotFoundError class). Mirror the runpod synthesis so the local
    # install block + the commands.json install have a file to install.
    if "local" in _mode_str and not requirements_path.exists():
        try:
            from backend.agents.rlm.requirements_derive import ensure_requirements_txt
            _project_dir_local = code_dir.parent if code_dir.name == "code" else code_dir
            ensure_requirements_txt(
                code_dir,
                dockerfile_path=_project_dir_local / "Dockerfile",
                base_image=env_id,
            )
            if requirements_path.exists():
                logger.info(
                    "_execute_in_sandbox: synthesized requirements.txt for local sandbox "
                    "(%d bytes)", requirements_path.stat().st_size,
                )
        except Exception:  # noqa: BLE001 — synthesis must never block the run
            logger.exception("_execute_in_sandbox: local requirements.txt auto-derive failed")

    if "local" in _mode_str and requirements_path.exists():
        bootstrap_commands.append(
            "python -m pip install --upgrade pip wheel setuptools || true"
        )
        # CUDA-build pin (local sandbox): the host driver caps the usable CUDA toolkit.
        # e.g. driver 535 / CUDA 12.2 CANNOT run torch's DEFAULT cu130 wheels — torch
        # imports but torch.cuda.is_available() is False ("driver too old"), so real
        # training silently falls back to CPU (useless for Qwen). Install a
        # driver-compatible torch FIRST from the matching PyTorch wheel index; the
        # agent's requirements.txt (torch>=…) is then satisfied by it and won't pull an
        # incompatible build. cu121 matches this 8×A5000 host (driver 12.2) and the vLLM
        # stack. Override via OPENRESEARCH_LOCAL_TORCH_INDEX_URL; set it empty to disable.
        _torch_index = _os.environ.get(
            "OPENRESEARCH_LOCAL_TORCH_INDEX_URL",
            "https://download.pytorch.org/whl/cu121",
        ).strip()
        # env_pin (D6a) — the harness OWNS the cu121 core (torch/vision/audio); the
        # agent's conflicting re-pin is stripped before install. This is the fix for the
        # 2026-06-07 All-Conv-Net collapse, where `torch==2.2.0` DOWNGRADED the cu121
        # build and left an incoherent CUDA stack (libcupti.so.12 failed to dlopen →
        # every experiment died at import). See _local_core_bootstrap_commands. Fail-soft;
        # opt out with OPENRESEARCH_DISABLE_ENV_PIN=1 (or OPENRESEARCH_LOCAL_TORCH_INDEX_URL="").
        bootstrap_commands.extend(
            _local_core_bootstrap_commands(requirements_path, _torch_index)
        )
        # Harness owns the multi-GPU launcher (`accelerate launch` + FSDP2) —
        # ensure Accelerate is in the per-run venv regardless of requirements.txt.
        bootstrap_commands.append(
            "python -m pip install -U accelerate || true"
        )

    # Phase 2B — preflight IMPORT smoke (the executing half of preflight "TDD").
    # When OPENRESEARCH_PREFLIGHT_SMOKE is on, emit a stdlib-only probe into code/ and run
    # it as the LAST bootstrap step (after deps install, before the training commands).
    # It imports every third-party dependency on CPU (GPU hidden) — NOT the agent's own
    # modules — so a missing dep (the matplotlib ModuleNotFoundError class) fails in
    # seconds; the command loop then short-circuits the GPU training and the import
    # error becomes the next iteration's repair_context.
    try:
        from backend.agents.rlm import preflight_smoke as _preflight_smoke
        if _preflight_smoke.is_enabled():
            _preflight_smoke.emit(code_dir)
            bootstrap_commands.append(_preflight_smoke.smoke_command(code_dir))
    except Exception:  # noqa: BLE001 — preflight smoke wiring must never block the run
        logger.exception("_execute_in_sandbox: preflight smoke wiring failed")

    # Layer 1 execution smoke: when OPENRESEARCH_EXECUTION_SMOKE is on, run the agent's
    # entry script for 1 step per experiment on tiny data (OPENRESEARCH_SMOKE_STEPS=1) with
    # CUDA_LAUNCH_BLOCKING=1 — AFTER the import smoke, BEFORE the full training. A runtime
    # crash (e.g. a VAE device-side assert from a data/shape bug) surfaces at the real
    # line in seconds, short-circuits the GPU training, and becomes repair_context — the
    # exact class that cost a 25-min run 0.12 of its score. A script that ignores the
    # smoke env is killed by `timeout` (exit 124) and treated as a soft pass (no block).
    try:
        from backend.agents.rlm import execution_smoke as _execution_smoke
        if _execution_smoke.is_enabled():
            _entry = next(
                (e for e in ("train.py", "train_cell.py", "main.py", "run.py")
                 if (code_dir / e).exists()),
                None,
            )
            if _entry is not None:
                bootstrap_commands.append(
                    _execution_smoke.smoke_command(code_dir, entry_script=_entry)
                )
            else:
                logger.info("_execute_in_sandbox: execution smoke skipped — no known entry script")
    except Exception:  # noqa: BLE001 — execution smoke wiring must never block the run
        logger.exception("_execute_in_sandbox: execution smoke wiring failed")

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
    _MAX_SOFT_RECOVERIES = int(_os_env_wd.environ.get("OPENRESEARCH_WATCHDOG_MAX_SOFT_RECOVERIES", "3"))

    _wd_cfg = _WatchdogConfig.from_env()
    # Feed the GPU/CPU compute-liveness signals into the watchdog (2026-06-08, decision #2): the
    # pinned PHYSICAL GPU ids + the Pillar-1 heartbeat sidecar (carries the train pid) so a
    # quiet-but-computing run is never killed and a genuine 0%-util hang is caught even when the
    # dumb daemon is gone. The sidecar lives at project_root (== code_dir) / .exec_heartbeat.json.
    try:
        import dataclasses as _dataclasses
        _wd_cfg = _dataclasses.replace(
            _wd_cfg,
            gpu_device_ids=tuple(gpu_device_ids or ()),
            heartbeat_json_path=str(code_dir / ".exec_heartbeat.json"),
        )
    except Exception:  # noqa: BLE001 — watchdog compute-liveness wiring must never break the run
        logger.debug("_execute_in_sandbox: watchdog GPU/CPU-liveness wiring skipped")

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

            # Distributed-launch safety net: if >1 GPU is allocated and the train
            # script uses FSDP/accelerate but is launched as plain `python`,
            # re-launch it via `accelerate launch` + a harness FSDP2 config so
            # params/grads/optimizer actually shard across the leased cards (else
            # only one card is used → large models OOM). Dynamic + no-op otherwise.
            if len(gpu_device_ids) > 1:
                commands = _resolve_distributed_launch(
                    list(commands), code_dir, len(gpu_device_ids), run_id
                )

            from backend.agents.rlm.preflight_smoke import MARKER as _SMOKE_MARKER
            from backend.agents.rlm import execution_smoke as _exec_smoke
            for command in (*bootstrap_commands, *commands):
                _cmd_res = await service.execute(
                    ExecuteCommand(sandbox=sandbox, command=command,
                                   timeout=per_command_timeout or _EXEC_TIMEOUT_SECONDS))
                results.append(_cmd_res)
                # Phase 2B: a failed preflight IMPORT smoke means the training command
                # would crash on the same missing dep — skip it (and the rest) so the bug
                # surfaces in CPU-seconds, not after the GPU spins up. Only the marked
                # smoke command triggers this; every other command runs exactly as before.
                if _SMOKE_MARKER in command and not _cmd_res.succeeded:
                    logger.warning(
                        "_execute_in_sandbox: preflight import smoke FAILED — skipping "
                        "remaining commands (no GPU training); see preflight_smoke_result.json")
                    break
                # Layer 1: the EXECUTION smoke (1-step dry-run) blocks ONLY on a REAL
                # crash, NOT on a timeout-kill (exit 124 = the script ignored the step
                # cap → not necessarily broken → soft pass). Use the exit code to make
                # that distinction; if it's unavailable, fail-soft (do NOT block — a
                # false block costs more than a missed catch).
                if _exec_smoke.MARKER in command:
                    _code = getattr(_cmd_res, "exit_code", None)
                    if _code is not None:
                        _status, _blocking = _exec_smoke.interpret_exit(int(_code))
                        if _blocking:
                            logger.warning(
                                "_execute_in_sandbox: execution smoke CRASH (%s) — skipping "
                                "remaining commands (no GPU training). The real traceback is "
                                "in the smoke output (CUDA_LAUNCH_BLOCKING=1).", _status)
                            break
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
    # for the remainder of the run. Opt-in via OPENRESEARCH_RUNPOD_AUTO_FALLBACK=true
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
        # Total wall-clock across every command in this experiment (weight download
        # + training + eval). Consumed by the no-smokes postflight guard
        # (_training_health_violation) so a seconds-long smoke cannot be the scored
        # artifact, and surfaced in experiment_runs.jsonl as a diagnostic.
        "wall_time_s": round(
            sum(float(getattr(r, "duration_seconds", 0.0) or 0.0) for r in results), 3
        ),
    }


def _manifest_enrichment(result: dict) -> None:
    """P2 provenance manifest: enrich a run_experiment result IN PLACE with the
    fields that bind a final metric to the artifact that produced it (invariant 2:
    every final metric traces to a persisted artifact via a manifest).

    Best-effort + fail-soft — observability must never break a run, so every
    lookup degrades silently:
      - ``sandbox_backend``: promoted from ``resource_limits`` so the manifest
        names the backend without callers digging into nested limits.
      - ``metrics_sha256``: sha256 of the canonical ``metrics.json`` artifact, so
        a final-report metric can be tied to the exact bytes that produced it
        (the trace that closes invariant 2 for the RLM path).
    """
    try:
        _rl = result.get("resource_limits") or {}
        _backend = _rl.get("sandbox_mode") or _rl.get("sandbox_backend")
        if _backend:
            result.setdefault("sandbox_backend", str(_backend))
    except Exception:  # noqa: BLE001 — manifest enrichment never blocks a run
        pass
    try:
        _artifact_dir = result.get("artifact_dir")
        if _artifact_dir and not result.get("metrics_sha256"):
            import hashlib
            from pathlib import Path as _Path

            _mjson = _Path(str(_artifact_dir)) / "metrics.json"
            if _mjson.exists():
                result["metrics_sha256"] = hashlib.sha256(_mjson.read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001 — manifest enrichment never blocks a run
        pass


def _stamp_manifest_ids(result: dict, *, run_id: str, env_id: str, commands: list) -> None:
    """P2 manifest: record the identifiers that bind a run_experiment result to
    its run — ``experiment_run_id`` (the ``run_id`` used for this attempt's
    artifacts, previously minted in the escalation loop and discarded),
    ``env_id``, and the structured ``commands`` list. ``setdefault`` so a value an
    earlier path already set is never clobbered; a non-dict result is a no-op."""
    if not isinstance(result, dict):
        return
    result.setdefault("experiment_run_id", run_id)
    result.setdefault("env_id", env_id)
    result.setdefault("commands", list(commands) if commands else [])


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

    # P2 manifest: bind metric→artifact (metrics_sha256) + name the backend.
    _manifest_enrichment(result)

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


def _scope_violation_key(hint: str) -> str:
    """Stable signature of a scope-shape violation — the missing element(s).

    The hint names ONE model + the missing datasets, e.g. "per_dataset_incomplete:
    model 'qwen3_1_7b' missing datasets ['WebShop'] …". The model named varies across
    experiments, but the missing piece (the last bracketed list) is the stable
    "what is unobtainable" signature we count toward a tolerated scope reduction.
    """
    import re

    brackets = re.findall(r"\[([^\]]*)\]", hint or "")
    if brackets:
        return brackets[-1].strip().lower()
    return (hint or "").strip().lower()[:120]


# A named Python exception in a recorded error string means the agent CAUGHT a code
# bug and masked it as a data failure. DatasetNotFoundError is deliberately excluded
# (ambiguous: a typo'd id vs a genuinely-removed dataset → default to data-unavailable).
# Only UNAMBIGUOUS exception names — ones that essentially never appear in a genuine
# data-unavailability message. ValueError/RuntimeError/KeyError/IndexError are
# DELIBERATELY excluded (HF datasets raises ValueError/RuntimeError for real
# unavailability — "Unknown split", config-not-found, connection); those are caught
# instead via the specific phrases below so a 404/config error isn't mis-flagged.
_CODE_BUG_RE = re.compile(
    r"\b(TypeError|AttributeError|ImportError|ModuleNotFoundError|NameError|"
    r"UnboundLocalError|AssertionError|FileNotFoundError|HfUriError|HFValidationError)\b"
)
# Phrases that are code/config bugs even without an unambiguous exception class name.
_CODE_BUG_PHRASES = (
    "cannot re-initialize cuda",
    "broken pipe",
    "is not a valid model identifier",
    "invalid hf uri",
    "returned 0 rows",
    "must be a string or a real number",   # float(tuple) family
    "repository id must be",
)

# F-03: a bare OSError-style "no such file" / "errno 2" is a code bug only when a
# config/source co-signal co-occurs (a missing base_config.yaml / *.py). Without
# one, a missing DATA path is a provably-unobtainable dataset, not a code bug.
# ("has no attribute" was dropped from _CODE_BUG_PHRASES — AttributeError /
# FileNotFoundError are already caught by class name in _CODE_BUG_RE.)
_CONFIG_CODE_COSIGNALS = (".yaml", ".yml", ".cfg", ".toml", ".ini", ".py", "config")


def _dir_footprint_gb(root: "Path", cap_gb: float = 8.0) -> float:
    """Approximate on-disk footprint (GB) of ``root``, short-circuiting once it exceeds
    ``cap_gb`` so a big tree can't make the disk check slow. Skips the harness per-run
    ``.venv`` (it is harness-owned, not the agent's download footprint). Fail-soft → 0.0.
    """
    import os as _os

    cap_bytes = cap_gb * 1e9
    total = 0
    try:
        for dirpath, dirnames, filenames in _os.walk(str(root)):
            dirnames[:] = [d for d in dirnames if d not in (".venv", "__pycache__", ".git")]
            for fn in filenames:
                try:
                    total += _os.path.getsize(_os.path.join(dirpath, fn))
                except OSError:
                    continue
            if total >= cap_bytes:
                return round(total / 1e9, 2)
    except Exception:  # noqa: BLE001 — footprint estimation must never break the run
        return round(total / 1e9, 2)
    return round(total / 1e9, 2)


def _disk_floor_violation(paths: list[str]) -> tuple[str, str] | None:
    """Return a repairable ``disk_exhausted`` violation if free disk on ANY of
    ``paths`` is below ``OPENRESEARCH_DISK_FLOOR_GB`` (default 15; 0 disables). Never
    raises. Used as a pre-check (don't start a doomed run) and a post-check.

    Honest attribution (2026-06-08): when the volume is full but THIS run's footprint is
    small, the cause is OTHER runs' caches, not this run's downloads — say so (GC advice)
    instead of telling the agent to slice its (already tiny) dataset. The 2026-06-08 Adam
    failure breached the floor with a 332 KB run footprint while the shared ``/home`` was
    full of other runs' caches; the floor fired correctly but the message blamed Adam.
    ``paths[0]`` is the run dir (``ctx.project_dir``); its ``code/`` subtree is the agent's
    controllable footprint.
    """
    import shutil
    from pathlib import Path as _Path

    try:
        floor_gb = float(os.environ.get("OPENRESEARCH_DISK_FLOOR_GB", "15") or "15")
    except ValueError:
        floor_gb = 15.0
    if floor_gb <= 0:
        return None
    try:
        small_gb = float(os.environ.get("OPENRESEARCH_RUN_SMALL_FOOTPRINT_GB", "5") or "5")
    except ValueError:
        small_gb = 5.0

    run_dir = str(paths[0]) if paths else ""
    seen: set[str] = set()
    for idx, p in enumerate(paths):
        if not p or p in seen:
            continue
        seen.add(p)
        try:
            free_gb = shutil.disk_usage(p).free / 1e9
        except Exception:  # noqa: BLE001 — a bad path must not break the run
            continue
        if free_gb >= floor_gb:
            continue
        # Attribute: how big is THIS run's own footprint? (code/ if present, else run dir.)
        footprint_gb = 0.0
        if run_dir:
            _root = _Path(run_dir) / "code"
            if not _root.exists():
                _root = _Path(run_dir)
            footprint_gb = _dir_footprint_gb(_root, cap_gb=floor_gb)
        if footprint_gb < small_gb:
            # This run is NOT the cause — the shared volume is full of other runs' data.
            return (
                "disk_exhausted",
                f"disk_exhausted: only {free_gb:.1f} GB free on {p} (< floor {floor_gb:.0f} GB), "
                f"but THIS run's footprint is just {footprint_gb:.2f} GB — the shared volume is "
                f"full of OTHER runs' data, not this run's downloads. Reclaim space by GC'ing the "
                f"re-downloadable caches (`rm -rf runs/.cache/data runs/.cache/envs`) or stale run "
                f"outputs, then retry. If this run legitimately needs large data, lower "
                f"OPENRESEARCH_DISK_FLOOR_GB.",
            )
        return (
            "disk_exhausted",
            f"disk_exhausted: only {free_gb:.1f} GB free on {p} (< floor {floor_gb:.0f} GB) and "
            f"this run's footprint is {footprint_gb:.1f}+ GB — a dataset/model download has "
            f"ballooned the disk. Stream + slice datasets, use a lighter variant, or lower "
            f"OPENRESEARCH_DISK_FLOOR_GB if the footprint is legitimately large.",
        )
    return None


def _finalize_timeout_result(
    ctx: "RunContext", code_path: str, run_id: str, result: dict, *, reason: str
) -> dict:
    """Finalize-on-timeout (2026-06-08): score the completed work instead of zeroing it.

    When ``run_experiment`` times out (the outer thread-pool backstop) or the inner exec
    returns ``exec_timeout``/``exec_stalled``, the 4-of-5-families-already-trained case
    (the 2026-06-08 Adam failure) must NOT degrade every rubric leaf to 0. Load the newest
    results-bearing ``metrics.json`` from disk (the same ``(has_results, mtime)`` ranking the
    leaf scorer uses); if ≥ 1 family carries a MEASURED value, attach it and flag
    ``partial_timeout`` (repairable) with a repair_context naming done vs missing families —
    the partial is preserved + scored, and the next iteration is told to bound the long pole.
    A truly empty placeholder is left as the empty-fail (tagged ``exec_stalled``/``exec_timeout``
    so it still classifies). Fail-soft: any error returns ``result`` unchanged.
    """
    try:
        import json as _json

        from backend.evals.paperbench.leaf_scorer import _latest_metrics_path

        loaded: dict = {}
        try:
            mpath = _latest_metrics_path(ctx.project_dir)
        except Exception:  # noqa: BLE001
            mpath = None
        if mpath is not None:
            try:
                # _latest_metrics_path already returns a Path (Path is imported locally in
                # run_experiment, not at module scope — don't re-wrap it here).
                d = _json.loads(mpath.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    loaded = d
            except Exception:  # noqa: BLE001
                loaded = {}

        per_model = loaded.get("per_model") if isinstance(loaded, dict) else None
        measured: list[str] = []
        if isinstance(per_model, dict) and per_model:
            measured = [
                m for m, mv in per_model.items()
                if _per_model_has_measured_value(mv if isinstance(mv, dict) else {})
            ]
        cause = str(result.get("cause_kind") or "")
        empty_class = "exec_stalled" if "stall" in cause.lower() else "exec_timeout"
        if not measured:
            # Nothing recoverable — keep the empty-fail, but make sure it classifies.
            return {**result, "failure_class": result.get("failure_class") or empty_class}

        missing = [m for m in per_model if m not in measured]
        msg = (
            f"partial_timeout: the experiment ended early ({reason}) but "
            f"{len(measured)} model/family result(s) were already written to disk and are "
            f"PRESERVED + scored ({', '.join(map(str, measured[:6]))}). "
            + (
                f"{len(missing)} did not finish ({', '.join(map(str, missing[:6]))}). "
                if missing else ""
            )
            + "To complete the rest without re-burning the finished work: bound the long pole "
            "— emit cells.json + train_cell.py (one cell per config, each independently timed) "
            "OR cap/stream the sweep smallest-config-first — and write metrics.json atomically "
            "after each family, then re-run."
        )
        return {
            **result,
            "success": False,
            "metrics": loaded,
            "failure_class": "partial_timeout",
            "partial_timeout": True,
            "error": msg,
        }
    except Exception:  # noqa: BLE001 — finalize must never break the run
        logger.exception("run_experiment: finalize-on-timeout failed; leaving result unchanged")
        return result


def _emit_experiment_progress_loop(ctx, code_path, stop_event, *, interval_s: float = 30.0) -> None:
    """Background tailer (local only): every ~``interval_s`` read the ``.exec_heartbeat.json``
    sidecar LocalProcessBackend writes and emit a sanitized ``experiment_progress`` dashboard
    event, so a long ``run_experiment`` is observable in the UI/dashboard while it runs (and the
    real-output timestamp feeds the watchdog's SSE liveness). Pure file-read + event-emit; fully
    fail-soft. Exits when ``stop_event`` is set. Emits only when the sidecar advances (new output)
    so it never masks a hang the way the dumb heartbeat daemon did.
    """
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    hb = _Path(code_path) / ".exec_heartbeat.json"
    start = _time.time()
    last_lines = -1
    while not stop_event.wait(interval_s):
        try:
            if not hb.exists():
                continue
            data = _json.loads(hb.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            lines = int(data.get("lines", 0) or 0)
            advanced = lines != last_lines
            last_lines = lines
            # Emit ONLY when output actually advanced (the first poll counts) so
            # experiment_progress is a genuine-liveness signal, not a periodic heartbeat that
            # would mask a hang from the watchdog the way the dumb .heartbeat daemon did. A
            # quiet-but-computing phase is kept alive by the watchdog's GPU/CPU check + the
            # inner stall, not by this event.
            if not advanced:
                continue
            _emit_dashboard_event(ctx, event_type="experiment_progress", payload={
                "last_output_at": str(data.get("last_output_at", "")),
                "last_line": str(data.get("last_line", ""))[:200],
                "lines": lines,
                "pid": data.get("pid"),
                "command": str(data.get("command", ""))[:200],
                "elapsed_s": round(_time.time() - start, 1),
            })
        except Exception:  # noqa: BLE001 — progress emission must never break the run
            continue


def _data_load_failure_is_code_bug(err: str) -> bool:
    """True when a recorded ``data_load_failure`` error is actually a CODE bug (a
    caught Python exception / bad id / parse error) rather than genuine
    data-unavailability (404/403/licence/removed dataset).

    Default for ambiguous cases is data-unavailable (safe: a false-repairable wastes
    one iteration, a false-exclude is the existing behaviour). DatasetNotFoundError is
    NOT in the exception regex, so a bare "dataset doesn't exist on the Hub" stays
    data-unavailable unless a bad-id/URI phrase co-occurs.
    """
    low = (err or "").lower()
    if any(p in low for p in _CODE_BUG_PHRASES):
        return True
    if _CODE_BUG_RE.search(err or ""):
        return True
    # Bare 'no such file'/'errno 2' is a code bug ONLY with a config/source
    # co-signal (a missing base_config.yaml / *.py); a bare missing DATA path
    # stays data-unavailable so it force-reduces on first sight (F-03).
    if ("no such file or directory" in low or "errno 2" in low) and any(
        c in low for c in _CONFIG_CODE_COSIGNALS
    ):
        return True
    return False


def _reclassify_masked_code_bugs(result: dict) -> tuple[str, list[str]] | None:
    """Scan ``metrics.data_load_failures`` for code bugs the agent caught and masked
    as data-unavailability. Returns ``("code_bug", [summaries])`` if any, else None.

    Genuine data-unavailability entries are LEFT in place (the leaf scorer still
    excludes their leaves); only code-bug entries force the experiment back into the
    repair loop, so a real bug can't silently ship a degenerate, leaf-excluded run.
    """
    _metrics = result.get("metrics") or {}
    failures = list(_metrics.get("data_load_failures") or []) + list(_metrics.get("model_load_failures") or [])
    masked: list[str] = []
    for entry in failures:
        if isinstance(entry, dict):
            err = str(entry.get("error") or entry.get("reason") or "")
            name = str(entry.get("dataset") or entry.get("env") or entry.get("model") or entry.get("name") or "?")
        elif isinstance(entry, str):
            err, name = entry, "?"
        else:
            continue
        if err and _data_load_failure_is_code_bug(err):
            masked.append(f"{name}: {err[:160]}")
    return ("code_bug", masked) if masked else None


def _surface_masked_bug_on_failed_run(result: dict) -> dict | None:
    """F-05: for an already-FAILED run with no specific ``failure_class``, surface a
    masked code bug's precise message so the next repair targets the real loader/parse
    bug instead of a vaguer error.

    ``_reclassify_masked_code_bugs`` is metrics-based and success-agnostic, but its
    call site only runs on SUCCESSFUL runs (to flip them to repairable). A run that
    already failed for a vague reason can still carry a masked code bug in
    ``metrics.data_load_failures``; promote that precise message into
    ``error``/``suggested_fix`` and set the precise ``failure_class``.

    Returns the fields to merge into ``result``, or None when not applicable. NEVER
    flips ``success`` (the run is already failed) and never overrides an
    already-specific ``failure_class``.
    """
    if result.get("success"):
        return None
    if _failure_class_key(result.get("failure_class")):
        return None  # a specific class is already set — leave it
    masked = _reclassify_masked_code_bugs(result)
    if masked is None:
        return None
    _cls, _bugs = masked
    msg = (
        "code_bug: a loader/parse error was caught and masked as a data_load_failure "
        "(it would be silently excluded from the rubric). These are CODE bugs to fix, "
        "not missing data — " + "; ".join(_bugs[:5])
    )
    return {
        "failure_class": _cls,
        "error": msg,
        "suggested_fix": result.get("suggested_fix") or msg,
    }


def _gap_in_load_failures(hint: str, metrics: dict) -> bool:
    """True when the scope element named in ``hint`` is covered by the agent's own
    ``metrics.data_load_failures`` — i.e. the agent tried to obtain it and failed.

    That is a PROVABLY-uncontrollable absence (per the soft-failure dataset
    convention in baseline_implementation.py), so we tolerate the reduction on
    first sight rather than waiting for K identical misses. Matching is by token
    subset: every token of the missing-element key must appear among the recorded
    failed-dataset names, so "webshop" matches a failure of dataset "webshop" but
    a two-element gap is only force-reduced when BOTH are recorded failures.
    """
    import re

    key_tokens = {t for t in re.split(r"[^a-z0-9]+", _scope_violation_key(hint)) if t}
    if not key_tokens:
        return False
    failure_tokens: set[str] = set()
    for entry in (metrics or {}).get("data_load_failures") or []:
        if isinstance(entry, dict):
            name = str(entry.get("dataset") or entry.get("name") or "")
            err = str(entry.get("error") or entry.get("reason") or "")
        elif isinstance(entry, str):
            name, err = entry, entry
        else:
            name, err = "", ""
        # A code bug masquerading as a data failure must NOT force-reduce a scope gap
        # (FIX-1 already flips those to repairable, but guard here too so a MISSED code
        # bug can't be laundered into a tolerated 'unobtainable' scope reduction).
        if err and _data_load_failure_is_code_bug(err):
            continue
        failure_tokens |= {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}
    return bool(failure_tokens) and key_tokens <= failure_tokens


def _scope_reduce_or_fail(
    result: dict, hint: str, counts: dict, max_repeats: int, *, force_reduce: bool = False
) -> tuple[dict, bool]:
    """Decide whether a scope-shape violation is a repairable failure (first K-1
    times) or a tolerated SCOPE REDUCTION (Kth+ time the SAME element is missing,
    OR ``force_reduce`` — a provably-uncontrollable absence recorded in
    data_load_failures, tolerated on first sight).

    ``counts`` is the per-run {element_key: miss_count} map; it is mutated. Returns
    ``(updated_result, tolerated)``. Tolerated → keep ``success`` and record the gap
    in metrics.scope_gaps so the rubric downweights it (not 0) and the run converges.
    Otherwise → flip to a repairable ``scope_shape_violation`` so the agent can add it.
    """
    key = _scope_violation_key(hint)
    counts[key] = counts.get(key, 0) + 1
    if force_reduce or (max_repeats > 0 and counts[key] >= max_repeats):
        gaps = sorted({*((result.get("metrics") or {}).get("scope_gaps") or []), key})
        return (
            {**result, "scope_reduced": True,
             "metrics": {**(result.get("metrics") or {}), "scope_gaps": gaps}},
            True,
        )
    return ({**result, "success": False, "error": hint, "scope_shape_violation": True}, False)


def _rubric_plateaued(history: list[float], window: int, epsilon: float) -> bool:
    """True when the rubric score has stopped meaningfully improving.

    Looks at the last ``window`` recorded ``overall_score`` values; returns True
    only when there are at least ``window`` samples AND their spread
    (max − min) is ``<= epsilon`` — i.e. the score has flatlined. A genuinely
    improving run (later scores rising above earlier ones by more than epsilon)
    is never flagged: this detects *stuck*, not *slow*. The rubric score is the
    one true objective, so keying convergence off it — not off experiment shape
    — never false-positives on a run that is making real progress on some other
    axis while a scope element stays permanently unobtainable.
    """
    if window <= 1 or epsilon < 0 or len(history) < window:
        return False
    recent = history[-window:]
    return (max(recent) - min(recent)) <= epsilon


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

    # Compare scope entries to metrics keys on a separator-free canonical key, so a
    # scope display name ("Qwen3-1.7B-Instruct", "Search-QA") matches the agent's
    # sanitized metrics key ("qwen3_1_7b", "searchqa"). Without this a correctly-run
    # model/dataset is falsely flagged per_model_incomplete (2026-05-29 SDAR run).
    from backend.agents.rlm.paper_invariants import canonical_model_key

    def _ck(name: object) -> str:
        return canonical_model_key(str(name)).replace("_", "")

    def _ds_covered(dataset: object, present_keys: set[str]) -> bool:
        """True when ``dataset`` is covered by one of the metrics keys.

        Agents key envs as e.g. ``cifar10_noaug`` / ``mnist_mlp`` — a plain
        equality check against the scope dataset (``CIFAR-10``) falsely flags a
        correctly-run dataset as missing (2026-06-09 All-CNN/Adam). Containment
        is digit-aware so ``cifar100`` can never satisfy ``cifar10``: the
        residual character right after the match must not be a digit.
        """
        cd = _ck(dataset)
        if not cd:
            return False
        for pk in present_keys:
            if not pk:
                continue
            if cd == pk:
                return True
            if pk.startswith(cd) and not pk[len(cd)].isdigit():
                return True
            if cd.startswith(pk) and not cd[len(pk)].isdigit():
                return True
        return False

    if is_multi_model:
        per_model = metrics.get("per_model")
        if not isinstance(per_model, dict) or not per_model:
            return (
                f"per_model_required: scope is multi-model {models}. Write "
                f"metrics.json with a top-level per_model dict keyed by model "
                f"id, e.g. {{'per_model': {{'qwen3-1.7b': {{...}}, "
                f"'qwen2.5-3b': {{...}}}}}}."
            )
        present_keys = {_ck(k) for k in per_model}
        # Models in scope.models_skipped are explicitly excluded (e.g. capacity-gated
        # due to VRAM budget) — treat them as accounted for, not missing.
        skipped_canonical = {
            _ck(k)
            for k in ((metrics.get("scope") or {}).get("models_skipped") or [])
        }
        missing = [
            m for m in models
            if _ck(m) not in present_keys and _ck(m) not in skipped_canonical
        ]
        if missing:
            return (
                f"per_model_incomplete: scope requires entries for {models}; "
                f"missing {missing} in metrics.per_model."
            )
        if is_multi_dataset:
            for model_id, model_metrics in per_model.items():
                pd = (model_metrics or {}).get("per_dataset") if isinstance(model_metrics, dict) else None
                if not isinstance(pd, dict) or not pd:
                    # Accept env-keyed nesting too: agents commonly write
                    # per_model[model][env] directly rather than wrapping it in a
                    # "per_dataset" dict. Treat the model's own keys as the dataset
                    # set so a correctly-structured run isn't flagged per_dataset_required.
                    pd = model_metrics if isinstance(model_metrics, dict) else None
                if not isinstance(pd, dict) or not pd:
                    return (
                        f"per_dataset_required: scope is multi-dataset {datasets}. "
                        f"Each per_model entry MUST carry per-dataset metrics; "
                        f"model {model_id!r} has none."
                    )
                present_ds = {_ck(x) for x in pd}
                missing_ds = [d for d in datasets if not _ds_covered(d, present_ds)]
                if missing_ds:
                    return (
                        f"per_dataset_incomplete: model {model_id!r} missing "
                        f"datasets {missing_ds} in per_dataset."
                    )
    elif is_multi_dataset:
        # Single-model + multi-dataset: per_dataset at top level (no per_model nesting).
        pd = metrics.get("per_dataset")
        if not isinstance(pd, dict) or not pd:
            # Cells-route fallback: the grid route writes per_model[model][env]
            # with no top-level per_dataset; derive dataset coverage from the
            # union of env keys so a correctly-structured cells.json run isn't
            # flagged per_dataset_required.
            per_model = metrics.get("per_model")
            if isinstance(per_model, dict) and per_model:
                env_keys: set[str] = set()
                for model_metrics in per_model.values():
                    if isinstance(model_metrics, dict):
                        env_keys.update(k for k in model_metrics if isinstance(k, str))
                if env_keys:
                    pd = {k: True for k in env_keys}
        if not isinstance(pd, dict) or not pd:
            return (
                f"per_dataset_required: scope is multi-dataset {datasets}. "
                f"Write metrics.json with a top-level per_dataset dict keyed by "
                f"dataset id."
            )
        present_ds = {_ck(x) for x in pd}
        missing_ds = [d for d in datasets if not _ds_covered(d, present_ds)]
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


def _dynamic_gpu_headroom() -> float:
    """OPENRESEARCH_DYNAMIC_GPU_HEADROOM (default 1.25) — the VRAM safety multiplier the
    capacity gate clamps against. Matches the dynamic-GPU resolver's headroom."""
    try:
        h = float(os.environ.get("OPENRESEARCH_DYNAMIC_GPU_HEADROOM", "1.25") or "1.25")
    except ValueError:
        h = 1.25
    return h if h > 0 else 1.25


def _summarize_cell_logs(cells: list, matrix_result: dict, gpus: list) -> str:
    """A clean per-cell status summary for ``result['logs']``.

    Deliberately avoids the raw CUDA-OOM strings (``_OOM_LOG_MARKERS``) so a partial
    cell run is NOT misread by ``_training_health_violation`` as a repairable
    ``silent_oom`` — a shrink-exhausted cell OOM is terminal, surfaced via
    ``stop_reason`` instead. Raw per-cell errors live in ``metrics`` (not scanned).
    """
    lines = [f"cell-matrix: {len(cells)} cell(s) across {len(gpus)} GPU(s)"]
    for c in cells:
        r = matrix_result.get(c.get("id", "")) or {}
        st = r.get("status", "missing")
        st_label = "oom-shrink-exhausted" if st == "oom_failed" else st
        lines.append(
            f"  {c.get('id','?')} -> {st_label} "
            f"(gpu={r.get('gpu', '?')}, retries={r.get('retries', 0)})"
        )
    return "\n".join(lines)


def _operator_scope_exclusions(ctx: "RunContext") -> list:
    """Verified ``operator_scope`` Exclusions from the run's ScopeSpec.

    ``skip_models`` → model-axis exclusions; ``skip_datasets`` → environment-axis
    exclusions. Both are ``verified=True``: the operator's ``--scope-spec`` is the
    evidence that the de-scope was a deliberate human decision (not an agent
    laundering a failure into a free scope reduction), so the rubric EXCLUDES
    their leaves rather than scoring them 0. Empty list when there is no
    scope_spec. Never raises.
    """
    from backend.agents.rlm import exclusion as _excl

    spec = getattr(ctx, "scope_spec", None)
    if spec is None:
        return []
    evidence = "operator ScopeSpec (--scope-spec)"
    out: list = []
    pairs = (
        (getattr(spec, "skip_models", None) or [], _excl.AXIS_MODEL, "skip_models"),
        (getattr(spec, "skip_datasets", None) or [], _excl.AXIS_ENVIRONMENT, "skip_datasets"),
    )
    for items, axis, field in pairs:
        for it in items:
            name = str(it).strip()
            if not name:
                continue
            try:
                out.append(_excl.Exclusion(
                    item=name, axis=axis, kind=_excl.KIND_OPERATOR_SCOPE,
                    reason=f"{name} de-scoped by operator ({field})",
                    verified=True, evidence=evidence,
                ))
            except ValueError:
                logger.debug("operator-scope: skipped malformed exclusion %r", name)
    return out


def _apply_operator_scope(metrics: dict, ctx: "RunContext") -> dict:
    """Fold verified operator_scope + recovered gate exclusions into metrics.scope.

    Rewrites ``metrics['scope']`` so it carries (a) a structured ``exclusions``
    list spanning the capacity/dataset gate gaps AND the operator's
    ``skip_models`` / ``skip_datasets``, and (b) legacy ``environments_skipped`` /
    ``models_skipped`` / ``gaps`` derived from the VERIFIED subset only
    (anti-gaming). Idempotent and fail-soft — any error leaves the original
    metrics untouched so scope enrichment can never break the run.
    """
    if not isinstance(metrics, dict):
        return metrics
    op_excls = _operator_scope_exclusions(ctx)
    # Verified env-setup failures from run-start provisioning (an ALFWorld/WebShop
    # the host could not stand up) are excluded on the SAME fairness footing as an
    # operator de-scope — both are outside the agent's control, both verified.
    env_excls = list(getattr(ctx, "env_setup_exclusions", None) or [])
    all_excls = op_excls + env_excls
    if not all_excls:
        return metrics
    try:
        from backend.agents.rlm import exclusion as _excl

        scope = metrics.get("scope") if isinstance(metrics, dict) else None
        existing = dict(scope) if isinstance(scope, dict) else {}
        # Recover the capacity/dataset gate gaps as structured exclusions so the
        # final ``exclusions`` list is complete; clear ``gaps`` so
        # build_scope_block regenerates ONE deduped list from the structured set.
        recovered = _excl.exclusions_from_gaps(existing.get("gaps") or [])
        existing["gaps"] = []
        metrics["scope"] = _excl.build_scope_block(
            recovered + all_excls,
            models_run=existing.get("models_run"),
            existing=existing,
        )
    except Exception:  # noqa: BLE001 — scope enrichment must never break the run
        logger.warning("cell-matrix: operator-scope exclusion merge failed", exc_info=True)
    return metrics


def _smoke_metrics_violation(smoke_out: "Path", cell_id: str) -> "str | None":
    """U3 — flag a cell that ran but produced no / NaN metrics (the
    ``degraded_no_metrics`` root cause).  Returns a reason string, or None when fine.

    Conservative to avoid false positives: an empty ``{}`` is NOT flagged (a 1-step
    smoke may legitimately write partial metrics); only a MISSING / unparseable /
    NaN-or-inf metrics file blocks.
    """
    import json
    import math as _math
    from pathlib import Path
    cands = list(Path(smoke_out).rglob("metrics.json"))
    if not cands:
        return "ran without writing any metrics.json — the full grid would yield no measurable metrics"
    try:
        data = json.loads(cands[0].read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "wrote an unparseable metrics.json"

    def _bad(v) -> bool:
        if isinstance(v, dict):
            return any(_bad(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return any(_bad(x) for x in v)
        if isinstance(v, float):
            return _math.isnan(v) or _math.isinf(v)
        return False

    return "metrics.json contains NaN/inf values" if _bad(data) else None


def _cell_smoke_repair(failure_class: str, cell_id: str, detail: str, log_tail: str) -> dict:
    """Build a REPAIRABLE ``run_experiment`` failure from the pre-grid cell smoke.

    No ``stop_reason`` → the next iteration repairs train_cell.py and retries
    (distinct from the terminal capacity/oom stops, which end the run).
    """
    return {
        "success": False,
        "metrics": {},
        "logs": f"pre-grid cell smoke [{cell_id}]: {detail}\n{log_tail}",
        "error": f"pre-grid cell smoke ({cell_id}) failed: {detail}",
        "failure_class": failure_class,
        "repair_context": {
            "failure_class": failure_class,
            "detail": (
                f"The cell-aware execution smoke ran the SMALLEST cell '{cell_id}' for a brief "
                f"dry-run BEFORE the full grid and it failed ({detail}). This is a bug in "
                f"train_cell.py that would affect every cell — fix it and re-emit. Log tail:\n{log_tail}"
            ),
        },
    }


def _cell_pregrid_smoke(kept, code, artifact_root, gpus, gpus_per_cell, timeout_s, ctx) -> "dict | None":
    """U2/U3 — run the SMALLEST cell for a brief smoke BEFORE the full grid.

    Catches a non-OOM ``train_cell.py`` code bug (the All-CNN ``cell_execution_error``
    that zeroed 17 cells) on cell 1 in seconds and routes it to repair, instead of
    burning the whole matrix.  Returns a repairable failure dict to block, or None to
    proceed (ok / oom / timeout / soft pass).  Fully fail-soft: any infra error → None
    (a smoke flake must never block a legitimate run).
    """
    from datetime import datetime, timezone
    from pathlib import Path
    from backend.agents.rlm import gpu_cell_runner
    try:
        smoke_cell = min(kept, key=lambda c: float((c or {}).get("est_vram_gb") or 0.0))
        cid = str(smoke_cell.get("id") or "cell0")
        smoke_out = Path(artifact_root) / "_cell_smoke"
        smoke_timeout = int(os.environ.get("OPENRESEARCH_CELL_SMOKE_TIMEOUT_S", "180") or "180")
        if timeout_s:
            smoke_timeout = max(30, min(smoke_timeout, int(timeout_s)))
        # Encourage a cooperating train_cell.py to self-cap to 1 step (run_matrix's
        # child inherits os.environ; this call is sequential so a temporary set is
        # safe).  The short timeout catches a non-honoring-but-working cell as a soft pass.
        _prev = os.environ.get("OPENRESEARCH_SMOKE_STEPS")
        os.environ["OPENRESEARCH_SMOKE_STEPS"] = "1"
        try:
            res = gpu_cell_runner.run_matrix(
                [smoke_cell], str(Path(code) / "train_cell.py"),
                output_root=str(smoke_out),
                gpus=(list(gpus)[:gpus_per_cell] or None),
                per_cell_timeout_s=smoke_timeout,
                overall_timeout_s=smoke_timeout,
                gpus_per_cell=gpus_per_cell,
                now_iso=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            if _prev is None:
                os.environ.pop("OPENRESEARCH_SMOKE_STEPS", None)
            else:
                os.environ["OPENRESEARCH_SMOKE_STEPS"] = _prev
        cell_res = res.get(cid) or {}
        status = cell_res.get("status")
        log_tail = str(cell_res.get("log") or cell_res.get("logs") or "")[-1500:]
        if not log_tail.strip():
            # run_matrix records don't embed log content — the per-cell log lives
            # on disk at <output_root>/<cid>.log. Without this fallback the
            # repair_context said only "status=error" and the agent never saw the
            # traceback (2026-06-10 Adam v6: a PermissionError on a hardcoded
            # /artifacts output dir reached the agent as a bare status).
            try:
                log_tail = (smoke_out / f"{cid}.log").read_text(
                    encoding="utf-8", errors="replace"
                )[-1500:]
            except OSError:
                log_tail = ""
        # OOM has its own shrink-retry ladder; a timeout means the cell ran but did not
        # honor the 1-step cap (soft pass).  Only a genuine crash blocks + repairs.
        if status not in ("ok", "oom_failed", "timeout", None):
            detail = f"status={status}"
            if cell_res.get("error"):
                detail += f"; {str(cell_res.get('error'))[:200]}"
            return _cell_smoke_repair("cell_smoke_failed", cid, detail, log_tail)
        if status == "ok":
            bad = _smoke_metrics_violation(smoke_out, cid)
            if bad is not None:
                return _cell_smoke_repair("incomplete_metrics", cid, bad, log_tail)
        return None
    except Exception:  # noqa: BLE001 — smoke infra must never block a legit run
        logger.warning("run_experiment: pre-grid cell smoke raised (non-blocking)", exc_info=True)
        return None


def _hybrid_route_enabled() -> bool:
    """OPENRESEARCH_HYBRID_EXEC_ROUTE — run cells.json grid AND commands.json in one call.

    Default OFF. The two manifests were mutually exclusive per run_experiment
    call, so a multi-family paper (Adam: VAE sweep grid + train.py families)
    had to burn a full iteration renaming cells.json aside to reach its other
    families — and v6 instead re-ran the whole grid and died at the watchdog.
    """
    val = os.environ.get("OPENRESEARCH_HYBRID_EXEC_ROUTE", "").strip().lower()
    return bool(val) and val not in ("0", "false", "off")


def _merge_hybrid_results(grid_result: dict, cmd_result: dict, code_path: str) -> dict:
    """Fold a successful cells-grid result and the commands-route result into one.

    The agent-written metrics (families, per_dataset) are the base; the grid's
    ``per_model`` aggregate is grafted under keys the agent didn't write, so
    neither route's evidence is lost. The merged blob is persisted to
    ``code/metrics.json`` (what the scorer reads). When the commands route
    failed, the call stays a repairable failure but CARRIES the merged metrics
    so the grid work survives into repair context. Fail-soft throughout.
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        grid_m = grid_result.get("metrics") if isinstance(grid_result.get("metrics"), dict) else {}
        cmd_m = cmd_result.get("metrics") if isinstance(cmd_result.get("metrics"), dict) else {}
        merged = dict(cmd_m) if cmd_m else dict(grid_m)
        if grid_m:
            gpm = grid_m.get("per_model")
            if isinstance(gpm, dict) and gpm:
                target = merged.setdefault("per_model", {})
                if isinstance(target, dict):
                    for model_key, tree in gpm.items():
                        target.setdefault(model_key, tree)
            # Union honest scope gaps from both routes.
            g_gaps = ((grid_m.get("scope") or {}).get("gaps") or []) if isinstance(grid_m.get("scope"), dict) else []
            if g_gaps:
                scope = merged.setdefault("scope", {})
                if isinstance(scope, dict):
                    gaps = scope.setdefault("gaps", [])
                    if isinstance(gaps, list):
                        seen = {_json.dumps(g, sort_keys=True, default=str) for g in gaps}
                        for g in g_gaps:
                            key = _json.dumps(g, sort_keys=True, default=str)
                            if key not in seen:
                                gaps.append(g)
                                seen.add(key)

        out = dict(cmd_result)
        out["metrics"] = merged
        out["logs"] = (
            str(grid_result.get("logs") or "")
            + "\n--- hybrid route: commands.json after cells grid ---\n"
            + str(cmd_result.get("logs") or "")
        )
        warnings = list(grid_result.get("contract_warnings") or []) + list(
            cmd_result.get("contract_warnings") or []
        )
        if warnings:
            out["contract_warnings"] = warnings
        if not cmd_result.get("success") and grid_result.get("success"):
            out["error"] = (
                str(cmd_result.get("error") or "commands route failed")
                + " — NOTE: the cells.json grid SUCCEEDED and its per_model "
                "aggregate is preserved in metrics; repair only the commands/"
                "train.py families."
            )
        try:
            blob = _json.dumps(merged, indent=2, default=str)
            (_Path(code_path) / "metrics.json").write_text(blob, encoding="utf-8")
        except OSError:
            logger.warning("hybrid route: merged metrics.json persist failed")
        return out
    except Exception:  # noqa: BLE001 — merging must never lose the primary result
        logger.exception("hybrid route: merge failed — returning commands result")
        return cmd_result


def _execute_cell_matrix(ctx: "RunContext", code_path: str, caps, *, timeout_s: float | None, run_id: str) -> dict:
    """Run the training matrix one-GPU-per-cell via ``gpu_cell_runner`` (comp 4).

    PREVENT → drop over-budget cells + dead datasets to honest ``scope.gaps``.
    PLACEMENT → ``run_matrix`` pins one cell per GPU, ``min(free, cells)`` parallel,
    per-cell OOM shrink-retry. AGGREGATE → the canonical
    ``per_model[model_key][env][baseline]`` shape the scorer + postflight consume.
    STOP → when every run cell OOM-fails after shrink-exhaustion (or all cells are
    dropped) return a terminal ``stop_reason`` so the run reports instead of looping.

    Returns a ``run_experiment``-shaped result dict. Never raises (fail-soft).
    """
    import json
    import time as _time
    from datetime import datetime, timezone
    from pathlib import Path

    from backend.agents.rlm import cell_fingerprint, cell_matrix, cell_scheduler, gpu_cell_runner, k8s_job_cell_runner

    code = Path(code_path)
    artifact_root = code / "outputs" / run_id

    def _persist_metrics(m: dict) -> None:
        """Write the aggregated metrics where the leaf scorer + final_report read
        them (the per-cell metrics.json files are single-cell leaves; the scorer
        needs the aggregated per_model shape)."""
        try:
            artifact_root.mkdir(parents=True, exist_ok=True)
            blob = json.dumps(m, indent=2, default=str)
            (artifact_root / "metrics.json").write_text(blob, encoding="utf-8")
            (code / "metrics.json").write_text(blob, encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 — persistence failure must not crash the run
            logger.warning("cell-matrix: failed to persist aggregated metrics.json: %s", exc)

    try:
        manifest = json.loads((code / "cells.json").read_text(encoding="utf-8"))
        all_cells = [c for c in (manifest.get("cells") or []) if isinstance(c, dict) and c.get("id")]
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "metrics": {}, "logs": "",
                "error": f"cells.json unreadable: {type(exc).__name__}: {exc}",
                "failure_class": "contract_guard"}
    if not all_cells:
        return {"success": False, "metrics": {}, "logs": "",
                "error": "cells.json present but enumerated no valid cells",
                "failure_class": "contract_guard"}

    # Contract normalization (2026-06-09): every cell must carry the three
    # per_model tree axes (model_key/env/baseline) or the aggregate loses it —
    # the All-CNN run trained 14 cells to paper-grade accuracy and scored as
    # "no measured metrics" because its manifest used its own axis vocabulary.
    # Derive missing axes BEFORE any GPU is spent and tell both the operator
    # (run_warning event) and the agent (logs + contract_warnings on the
    # result) so the next iteration emits explicit axes.
    all_cells, _axis_notes = cell_matrix.normalize_cell_axes(all_cells)
    if _axis_notes:
        try:
            _emit_dashboard_event(ctx, event_type="run_warning", payload={
                "code": "cell_axes_derived",
                "message": " ".join(_axis_notes)[:500],
            })
        except Exception:  # noqa: BLE001 — diagnostics must never break the run
            logger.debug("run_experiment: cell_axes_derived warning emit failed")

    # Multi-GPU cells: a slot of `gpus_per_cell` cards device_map-shards ONE (large)
    # model, so the capacity-gate VRAM budget is the slot's COMBINED VRAM, not one card.
    _gpus_per_cell = max(1, int(os.environ.get("OPENRESEARCH_GPUS_PER_CELL", "1") or "1"))
    # PREVENT — clamp to the per-slot budget, drop confirmed-dead datasets (fail-soft).
    headroom = _dynamic_gpu_headroom()
    kept, cap_gaps, models_skipped = cell_matrix.capacity_gate(
        all_cells, caps.per_gpu_vram_gb * _gpus_per_cell, headroom=headroom)
    kept, ds_gaps, envs_skipped = cell_matrix.dataset_url_preflight(kept)

    gpus = [str(g) for g in (tuple(getattr(ctx, "gpu_device_ids", ()) or ()) or caps.free_gpu_ids)]

    def _terminal(kind: str, error: str, metrics: dict, logs: str) -> dict:
        _persist_metrics(metrics)
        try:
            _emit_dashboard_event(ctx, event_type="run_warning", payload={
                "code": kind, "message": error,
            })
        except Exception:  # noqa: BLE001 — diagnostics must never break the run
            logger.debug("run_experiment: cell-matrix stop event emit failed")
        return {"success": False, "metrics": metrics, "logs": logs, "error": error,
                "failure_class": kind,
                "stop_reason": {"kind": kind, "detail": error,
                                "per_gpu_vram_gb": caps.per_gpu_vram_gb,
                                "models_skipped": models_skipped,
                                "environments_skipped": envs_skipped,
                                "gaps": (cap_gaps or []) + (ds_gaps or [])}}

    # Everything dropped by the gates → nothing fits this backend → terminal.
    if not kept:
        metrics = cell_matrix.aggregate_cell_metrics(
            {}, [], capacity_gaps=cap_gaps, dataset_gaps=ds_gaps,
            models_skipped=models_skipped, environments_skipped=envs_skipped)
        metrics = _apply_operator_scope(metrics, ctx)
        return _terminal(
            "capacity_exhausted",
            f"every cell exceeds the per-GPU budget {caps.per_gpu_vram_gb:.0f} GB "
            f"(headroom {headroom}) or targets a dead dataset; nothing to run on this backend",
            metrics, "cell-matrix: all cells dropped by capacity/dataset gates")

    _t0 = _time.monotonic()
    # Bound the WHOLE matrix as a backstop, but GENEROUSLY: cells run in waves of
    # min(free_gpus, cells), so each sequential wave must get a full per-cell budget
    # — capping the whole matrix at ONE cell's budget would silently drop every cell
    # past the first wave to status=timeout (adversarial-review C1, 2026-06-02). The
    # run-level watchdog is the ultimate hang guard; this just lets the root score a
    # partial sooner than the watchdog's hard-exit if the matrix genuinely overruns.
    _n_slots = max(1, len(gpus) // _gpus_per_cell)
    _waves = (len(kept) + _n_slots - 1) // _n_slots  # ceil(cells / slots)
    # Time-budget gate (2026-06-10): the matrix budget must FIT the run's
    # remaining wall clock (minus a verify+report reserve). Adam v6's 100-epoch
    # VAE re-grid got `per_cell × waves` of budget with 4h of run left, sailed
    # into the 14h watchdog, and was hard-killed mid-cell — when run_matrix's
    # own deadline machinery would have TRIMMED the tail cells to honest
    # `timeout` leaves and returned a scoreable partial. Fail-soft: unknown
    # remaining time leaves the legacy budget untouched.
    _matrix_overall_s: float | None = (timeout_s * max(1, _waves)) if timeout_s else None
    try:
        _rem = ctx.remaining_s() if callable(getattr(ctx, "remaining_s", None)) else None
        _reserve = float(os.environ.get("OPENRESEARCH_MATRIX_FINALIZE_RESERVE_S", "2700") or 2700)
        _capped = cell_scheduler.cap_overall_budget(
            _matrix_overall_s, _rem, reserve_s=_reserve)
        if _capped != _matrix_overall_s:
            logger.info(
                "run_experiment: matrix budget capped to %.0fs by remaining run "
                "wall clock (%.0fs, reserve %.0fs)", _capped or -1, _rem or -1, _reserve)
        _matrix_overall_s = _capped
    except Exception:  # noqa: BLE001 — the gate must never block the grid
        logger.debug("run_experiment: matrix time-budget gate skipped", exc_info=True)
    # Cell-level resume (Track B): compute each kept cell's content fingerprint so
    # run_matrix can (a) record it in the per-cell manifest and (b) — when armed via
    # OPENRESEARCH_RESUME_CELLS — skip a prior ok+unchanged cell. Forced re-runs come from
    # OPENRESEARCH_RESUME_FORCE_CELLS (CSV of cell ids the CLI builds from --rerun-env /
    # --rerun-cell). All no-ops when resume is unset; fingerprints are always recorded.
    _fingerprints = {
        c["id"]: cell_fingerprint.compute_fingerprint(c, str(code))
        for c in kept if isinstance(c, dict) and c.get("id")
    }
    _force_cells = {
        cid.strip()
        for cid in (os.environ.get("OPENRESEARCH_RESUME_FORCE_CELLS", "") or "").split(",")
        if cid.strip()
    }
    _sb_key_ecm = getattr(
        getattr(ctx, "sandbox_mode", None), "value",
        str(getattr(ctx, "sandbox_mode", None) or ""),
    ).lower()
    _run_budget_ecm = getattr(ctx, "run_budget", None)
    _event_sink_ecm = getattr(ctx, "_event_sink", None)

    # Load the cached gpu_plan (written by resolve_gpu_requirements) so the K8s
    # runner can target the resolved SKU/node-pool.  Mirror run_experiment's load;
    # fail-soft to None so a missing plan never blocks the cell-matrix route.
    _ecm_gpu_plan = None
    try:
        _ecm_plan_path = Path(ctx.project_dir) / "rlm_state" / "gpu_plan.json"
        if _ecm_plan_path.exists():
            from backend.agents.schemas import GpuPlan as _ECMGpuPlan
            _ecm_gpu_plan = _ECMGpuPlan(**json.loads(_ecm_plan_path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 — gpu_plan load must never block the cell-matrix route
        logger.debug("_execute_cell_matrix: gpu_plan.json unreadable; proceeding without it")

    # U2/U3 — cell-aware pre-grid execution smoke (OPENRESEARCH_EXECUTION_SMOKE, local/docker
    # only; azure uses the K8s runner).  Run the smallest cell briefly BEFORE the grid so a
    # non-OOM train_cell.py bug (the All-CNN cell_execution_error) is caught on cell 1 and
    # routed to repair.  Skipped for a 1-cell grid (redundant); fully fail-soft.
    try:
        from backend.agents.rlm import execution_smoke as _execution_smoke_cm
        if (
            _execution_smoke_cm.is_enabled()
            and _sb_key_ecm != "azure"
            and gpus
            and len(kept) > 1
        ):
            _smoke_block = _cell_pregrid_smoke(
                kept, code, artifact_root, gpus, _gpus_per_cell, timeout_s, ctx,
            )
            if _smoke_block is not None:
                _persist_metrics(_smoke_block.get("metrics") or {})
                return _smoke_block
    except Exception:  # noqa: BLE001 — the smoke gate must never block a legit run
        logger.debug("run_experiment: cell pre-grid smoke gate raised (non-blocking)", exc_info=True)

    if _sb_key_ecm == "azure":
        with k8s_job_cell_runner.bind_run_context(
            run_budget=_run_budget_ecm, event_sink=_event_sink_ecm, gpu_plan=_ecm_gpu_plan
        ):
            matrix_result = k8s_job_cell_runner.run_matrix(
                kept, str(code / "train_cell.py"),
                output_root=str(artifact_root),
                gpus=gpus or None,
                per_cell_timeout_s=timeout_s,
                overall_timeout_s=_matrix_overall_s,
                gpus_per_cell=_gpus_per_cell,
                fingerprints=_fingerprints,
                force_cells=_force_cells or None,
                now_iso=datetime.now(timezone.utc).isoformat(),
            )
    else:
        matrix_result = gpu_cell_runner.run_matrix(
            kept, str(code / "train_cell.py"),
            output_root=str(artifact_root),
            gpus=gpus or None,
            per_cell_timeout_s=timeout_s,
            overall_timeout_s=_matrix_overall_s,
            gpus_per_cell=_gpus_per_cell,
            fingerprints=_fingerprints,
            force_cells=_force_cells or None,
            now_iso=datetime.now(timezone.utc).isoformat(),
        )
    wall = _time.monotonic() - _t0

    metrics = cell_matrix.aggregate_cell_metrics(
        matrix_result, kept, capacity_gaps=cap_gaps, dataset_gaps=ds_gaps,
        models_skipped=models_skipped, environments_skipped=envs_skipped)
    metrics = _apply_operator_scope(metrics, ctx)
    logs = _summarize_cell_logs(kept, matrix_result, gpus)

    if _axis_notes:
        logs = logs + "\n" + "\n".join(_axis_notes)

    statuses = [(matrix_result.get(c["id"]) or {}).get("status") for c in kept]
    n_ok = sum(s == "ok" for s in statuses)
    n_oom = sum(s == "oom_failed" for s in statuses)
    n_err = sum(s not in ("ok", "oom_failed") for s in statuses)

    # Dead-training early-stop (2026-06-09): cells the guard killed because their loss
    # was pinned (network never learned). These are deterministic architecture/init bugs
    # — NOT transient — so they drive a targeted, repairable ``degenerate_training``
    # signal rather than being silently scored low as fake-``ok`` runs-to-completion.
    n_diverged = sum(s == "training_diverged" for s in statuses)
    _diverged_cells = [
        c["id"] for c, s in zip(kept, statuses) if s == "training_diverged"
    ]

    if n_ok > 0:
        # At least one cell produced real metrics — partial or full success. Honest
        # gaps (dropped/oom/err/diverged cells) are already in metrics.scope; flows to
        # the SAME postflight guards + verify_against_rubric.
        _persist_metrics(metrics)
        result = {"success": True, "metrics": metrics, "logs": logs, "wall_time_s": wall}
        if _axis_notes:
            # Agent-visible: the root sees this on the returned dict and can emit
            # explicit axes in the next cells.json instead of repeating the gap.
            result["contract_warnings"] = list(_axis_notes)
        if n_diverged:
            # Surface the divergence prominently so the root model re-implements the
            # broken architectures on a later iteration (raising the score), instead of
            # the gaps being buried in scope. Advisory — the partial result still ships.
            result["divergence_warning"] = (
                f"{n_diverged} cell(s) early-stopped as dead-training (loss pinned, no "
                f"learning): {', '.join(_diverged_cells)}. These are gaps in the result "
                f"— fix the trainer for these architectures (weight init, normalization, "
                f"or pooling/shape wiring) to recover their contribution to the score."
            )
        return result

    if n_ok == 0 and n_oom == 0 and n_diverged > 0 and n_diverged == n_err:
        # Every run cell early-stopped as dead-training — repairable by fixing the
        # trainer, NOT terminal. Reuse the existing ``degenerate_training`` repairable
        # class (its classifier guidance + repair loop already exist); the error names
        # the exact cells + the concrete fix surface so the agent repairs the right bug.
        _persist_metrics(metrics)
        return {"success": False, "metrics": metrics, "logs": logs,
                "failure_class": "degenerate_training",
                "error": (f"all {n_diverged} run cell(s) early-stopped as dead-training "
                          f"(loss pinned, network not learning): "
                          f"{', '.join(_diverged_cells)} — fix weight init, add a missing "
                          f"normalization layer, or correct the pooling/shape wiring in "
                          f"these architectures, then re-run")}

    if n_err == 0:
        # Every run cell OOM-failed after the shrink ladder — un-repairable by
        # re-running the same config. STOP + report (do NOT reuse silent_oom).
        return _terminal(
            "oom_shrink_exhausted",
            f"all {n_oom} run cell(s) OOM-failed after batch-scale shrink + grad-ckpt "
            f"retries on the per-GPU budget {caps.per_gpu_vram_gb:.0f} GB; the matrix "
            f"cannot fit one card — reduce model size/scope or use a larger GPU",
            metrics, logs)

    # capacity_exhausted promotion: the K8s runner emits "capacity_exhausted:"-prefixed
    # error strings for stuck-Pending cells (pool quota/stock exhausted). When EVERY
    # errored cell carries that prefix (and no OOM, no ok), promote to terminal rather
    # than allowing a pointless repair loop — the cluster, not the code, is the bottleneck.
    # This branch is only reachable for azure runs (non-azure error strings never carry
    # the prefix), so it leaves local/runpod/docker behavior byte-for-byte unchanged.
    if n_ok == 0 and n_oom == 0 and n_err > 0:
        _err_cells = [
            (matrix_result.get(c["id"]) or {}) for c in kept
            if (matrix_result.get(c["id"]) or {}).get("status") not in ("ok", "oom_failed")
        ]
        _all_cap_exhausted = all(
            str((r.get("error") or "")).startswith("capacity_exhausted:")
            for r in _err_cells
        )
        if _all_cap_exhausted:
            _cap_msg = (f"all {n_err} run cell(s) failed: AKS node pool capacity exhausted "
                        f"(stuck-Pending past timeout) — quota or stock unavailable; "
                        f"try reducing azure_max_nodes or requesting a quota increase")
            return _terminal("capacity_exhausted", _cap_msg, metrics, logs)

    # Some non-OOM errors (code bugs) and no ok cell — repairable, not terminal.
    _div_note = (
        f" (of which {n_diverged} early-stopped as dead-training: "
        f"{', '.join(_diverged_cells)})" if n_diverged else ""
    )
    _persist_metrics(metrics)
    return {"success": False, "metrics": metrics, "logs": logs,
            "failure_class": "cell_execution_error",
            "error": (f"{n_err} cell(s) failed with non-OOM errors (likely code bugs)"
                      f"{_div_note}, {n_oom} OOM-failed, 0 succeeded — fix the cell "
                      f"trainer and re-run")}


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
    # comp 4: the harness-owned cell path runs code/train_cell.py via cells.json and
    # needs no commands.json. Only fail when NEITHER manifest is present.
    _cells_present = (Path(code_path) / "cells.json").is_file()
    if not commands and not _cells_present:
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
    # resolve_experiment_timeout_s applies OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S >
    # EXPERIMENT_TIMEOUT_BY_MODE[execution_mode] > _DEFAULT_EXPERIMENT_TIMEOUT_S,
    # clamped to ctx.remaining_s() when finite.
    timeout = resolve_experiment_timeout_s(ctx)

    # Inner-owns-deadline (2026-06-08, local only): give the inner exec the resolved
    # timeout as its per-command cap so its stall/timeout fires FIRST and process-group-
    # kills the train subprocess cleanly (no GPU-burning orphan); the outer thread-pool
    # .result() is a generous backstop = resolved + buffer. Non-local keeps the inner
    # _EXEC_TIMEOUT_SECONDS cap and outer == resolved (runpod/docker byte-for-byte).
    _per_command_timeout = timeout if _is_local_sb else None
    _outer_timeout = (timeout + _OUTER_TIMEOUT_BUFFER_S) if _is_local_sb else timeout

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

    # Disk pre-check (2026-05-30): fail fast if the shared disk is already below the
    # floor — starting a run that then exhausts it starves other users and crashes.
    _disk_pre = _disk_floor_violation([
        str(ctx.project_dir), os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", ""),
    ])
    if _disk_pre is not None:
        return _persist_experiment_result(ctx, {
            "success": False, "error": _disk_pre[1], "failure_class": _disk_pre[0],
        }, model_id=model_id, eval_env=eval_env)

    # Pre-download guard (2026-06-08, Pillar 5): the pre-check passed, but if headroom over the
    # floor is thin a single bulk dataset/model download could still breach it mid-run (otherwise
    # caught only by the post-check, after the wasted download). The harness can't intercept the
    # agent's download, so warn up front to stream/slice. Advisory + fail-soft; 0 disables.
    try:
        import shutil as _shutil_pre
        _headroom_gb = float(os.environ.get("OPENRESEARCH_DISK_PREFLIGHT_HEADROOM_GB", "30") or "30")
        _floor_gb = float(os.environ.get("OPENRESEARCH_DISK_FLOOR_GB", "15") or "15")
        if _headroom_gb > 0:
            _free_gb = _shutil_pre.disk_usage(str(ctx.project_dir)).free / 1e9
            if _free_gb < _floor_gb + _headroom_gb:
                _emit_dashboard_event(ctx, event_type="run_warning", payload={
                    "code": "disk_headroom_thin",
                    "message": (
                        f"disk headroom is thin: {_free_gb:.1f} GB free, only "
                        f"{_free_gb - _floor_gb:.1f} GB above the {_floor_gb:.0f} GB floor. A bulk "
                        f"dataset/model download could breach it mid-run — STREAM + slice datasets "
                        f"(e.g. load_dataset(..., streaming=True), .take(N)), use lighter variants, "
                        f"and avoid full natural_questions-scale downloads. GC runs/.cache/data to free space."
                    ),
                })
    except Exception:  # noqa: BLE001 — the pre-download guard must never break the run
        pass

    result: dict = {}

    # comp 4 (2026-05-31): harness-owned cell-runner route. When the backend exposes
    # GPUs (local/docker) AND the agent emitted code/cells.json + train_cell.py, run
    # the matrix one GPU per cell instead of the monolithic commands.json path — the
    # fix for the cuda:0 matrix-stacking that OOM'd the 2026-05-31 run. Mutually
    # exclusive with the legacy escalation loop below (and thus with
    # _resolve_distributed_launch, which only fires inside _execute_in_sandbox).
    # Fail-soft: any error here, or a missing manifest / no-GPU / cloud backend,
    # falls through to the legacy monolithic path unchanged.
    # Bound before the branch so the post-loop rubric-contract check (which reads
    # outputs/<run_id>) is valid on BOTH paths; the legacy loop reassigns it per
    # iteration, the cell route uses this value as its artifact root.
    run_id = f"{ctx.project_id}-{uuid.uuid4().hex[:8]}"
    _cell_route_taken = False
    _hybrid_grid_result: dict | None = None  # set when the hybrid route stashes a grid result
    # Progress→SSE tailer (local only): emit a sanitized experiment_progress event from the
    # .exec_heartbeat.json sidecar so a long run is observable in the UI while it runs (and the
    # real-output signal feeds the watchdog's SSE liveness). Daemon thread; stopped once the exec
    # work returns. Fail-soft — never blocks or breaks the run.
    import threading as _threading
    _progress_stop = _threading.Event()
    _progress_thread = None
    if _is_local_sb:
        _progress_thread = _threading.Thread(
            target=_emit_experiment_progress_loop,
            args=(ctx, code_path, _progress_stop),
            daemon=True,
        )
        _progress_thread.start()
    try:
        from backend.services.runtime.gpu_capacity import describe_capacity
        _caps = describe_capacity(ctx)
        if (
            _caps.backend_kind in ("local", "docker")
            and not _caps.is_empty
            and (Path(code_path) / "cells.json").is_file()
            and (Path(code_path) / "train_cell.py").is_file()
        ):
            result = _execute_cell_matrix(ctx, code_path, _caps, timeout_s=timeout, run_id=run_id)
            _cell_route_taken = True
            # Hybrid route (2026-06-10, OPENRESEARCH_HYBRID_EXEC_ROUTE, default off):
            # when BOTH manifests exist and the grid produced a successful
            # result, ALSO run the agent's commands.json families in this same
            # call (legacy loop below), then graft the grid aggregate into the
            # agent-written metrics — multi-family papers stop burning an
            # iteration renaming cells.json aside.
            if (
                _hybrid_route_enabled()
                and commands
                and isinstance(result, dict)
                and result.get("success")
            ):
                _hybrid_grid_result = result
                _cell_route_taken = False
    except Exception:  # noqa: BLE001 — the cell route must never crash the run
        logger.exception("run_experiment: cell-matrix route raised; falling back to legacy path")
        _cell_route_taken = False

    # Escalation loop (spec 2026-05-23 §OOM + §Capacity): on CUDA OOM OR
    # RunPod capacity exhaustion, pop the next SKU from GpuPlan.ladder_remaining,
    # persist the updated plan atomically, emit gpu_escalated, and retry.
    # Capped by max_escalations. Non-OOM/non-capacity failures and success exit
    # immediately. I12: explicit shutdown(wait=False) per iteration.
    # comp 4: skipped entirely on the cell-runner route (`_cell_route_taken`).
    while not _cell_route_taken:
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
                        per_command_timeout=_per_command_timeout,
                    ),
                ).result(timeout=_outer_timeout)
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
                    # Local backend streams to <code>/.exec_live.log (not outputs/<id>/exec.log);
                    # fall back to it so a human + the agent still see what was running.
                    if not recovered_logs:
                        live_log = Path(code_path) / ".exec_live.log"
                        if live_log.exists():
                            recovered_logs = live_log.read_text(encoding="utf-8", errors="replace")[-32000:]
                except OSError:
                    pass
                result = {
                    "success": False,
                    "metrics": {},
                    "logs": recovered_logs,
                    # Tag the cause so the post-loop finalize-on-timeout fires (it loads the
                    # on-disk partial metrics and scores them instead of zeroing the run).
                    "cause_kind": "exec_timeout",
                    "error": (
                        f"run_experiment: timed out after {_outer_timeout:.0f} s (outer backstop)"
                        if _outer_timeout is not None
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
                    # infra hiccups — advance the ladder so the run doesn't dead-end.
                    # This is intentionally the same path as CAPACITY_EXHAUSTED
                    # because: (a) the 500 may itself be capacity under a different
                    # marker, and (b) _execute_in_sandbox already exhausted 3 retries
                    # with exponential backoff before bubbling up here, so a genuine
                    # transient would have recovered. BUG-NEW-049: consider adding
                    # a same-tier retry before escalating if TRANSIENT_500 is the
                    # sole failure mode (CAPACITY_EXHAUSTED still escalates
                    # immediately). Bounded by dynamic_gpu_max_escalations so a
                    # request-shape bug cannot burn the whole catalog.
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
                # When OPENRESEARCH_RUNPOD_AUTO_FALLBACK=true and the exception carries
                # _retry_attempts (set by _execute_in_sandbox after exhausting
                # transient retries), check whether local docker + GPU is viable
                # and if so mutate ctx.sandbox_mode for the rest of this run.
                import os as _os_fallback
                if _os_fallback.environ.get("OPENRESEARCH_RUNPOD_AUTO_FALLBACK", "").lower() == "true":
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
        # F-04: also catch a watchdog-killed OOM whose marker is buried earlier
        # than the 4 KB stderr_tail (the watchdog dict carries no exit_code).
        is_oom = _is_oom_escalation_trigger(result, exit_code=exit_code, stderr_tail=stderr_tail)
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

    # Stop the progress tailer — the exec work is done, no more progress to stream.
    _progress_stop.set()
    if _progress_thread is not None:
        _progress_thread.join(timeout=2)

    # P2 manifest: the escalation loop has produced its final result — stamp the
    # identifiers the persist chokepoint records. run_id/env_id/commands are in
    # scope (the while-True ran ≥1 time, so run_id is bound to the last attempt).
    _stamp_manifest_ids(result, run_id=run_id, env_id=env_id, commands=commands)

    # Finalize-on-timeout (2026-06-08): a timed-out / stalled experiment must SCORE its
    # completed work, not zero it (the Adam failure: 4/5 families trained, the timeout fired
    # mid-VAE, every rubric leaf scored 0). Both the inner exec_timeout/exec_stalled return
    # and the outer pool-timeout handler tag cause_kind; load the newest on-disk partial
    # metrics and preserve them as a repairable partial_timeout. Runs BEFORE the
    # success-gated guards (which are skipped while success is False) so the partial survives.
    if not result.get("success") and str(result.get("cause_kind") or "") in (
        "exec_timeout", "exec_stalled",
    ):
        result = _finalize_timeout_result(
            ctx, code_path, run_id, result, reason=str(result.get("cause_kind"))
        )

    # Masked-code-bug reclassification (2026-05-30): the agent frequently CATCHES a
    # Python exception (TypeError/AttributeError/HfUriError/bad model id/'returned 0
    # rows') and records it in metrics.data_load_failures, which the leaf scorer would
    # then EXCLUDE from the rubric — silently shipping a degenerate run as success. A
    # code bug is repairable, not data-unavailability: flip it back into the repair
    # loop. Runs BEFORE scope-reduce/training-health (which gate on success) so a code
    # bug can't be tolerated as a scope gap. Genuine 404/licence entries are untouched.
    if result.get("success"):
        _masked = _reclassify_masked_code_bugs(result)
        if _masked is not None:
            _cls, _bugs = _masked
            result = {
                **result, "success": False, "failure_class": _cls,
                "error": (
                    "code_bug: a loader/parse error was caught and masked as a "
                    "data_load_failure (it would be silently excluded from the rubric). "
                    "These are CODE bugs to fix, not missing data — " + "; ".join(_bugs[:5])
                ),
            }
            logger.warning(
                "run_experiment[%s]: reclassified %d masked code bug(s) as repairable: %s",
                getattr(ctx, "run_id", "?"), len(_bugs), _bugs[:3],
            )
    else:
        # F-05: the run already failed for a vague reason — still surface a masked
        # code bug's precise message (never flipping success) so the next repair
        # targets the real loader/parse bug instead of a vague error.
        _surfaced = _surface_masked_bug_on_failed_run(result)
        if _surfaced is not None:
            result = {**result, **_surfaced}
            logger.warning(
                "run_experiment[%s]: surfaced masked code bug on a failed run (%s)",
                getattr(ctx, "run_id", "?"), _surfaced.get("failure_class"),
            )

    # Training-health postflight (2026-05-29): a run that exited 0 but logged a
    # caught CUDA OOM (backward skipped → no gradient updates) or trained far below
    # the convergence floor produced metrics yet learned nothing. Flip it to a
    # repairable failure so the next implement_baseline reduces memory / trains
    # longer instead of the loop accepting 0-reward metrics as success.
    if result.get("success"):
        _health = _training_health_violation(result)
        if _health is not None:
            _hcls, _hmsg = _health
            result = {**result, "success": False, "error": _hmsg, "failure_class": _hcls}

    # Metrics-completeness postflight (2026-05-30): a run that exited 0 but wrote a
    # placeholder / unpopulated metrics.json (status:"running", empty per_model)
    # measured NOTHING, yet every other guard keys on presence/shape/exit-code and
    # lets it through — so the rubric grades a half-finished experiment and scores
    # eval/result/execution ~0. Flip it to a repairable failure so the loop re-runs
    # to REAL measured metrics before it can score/finalize, and emit a descriptive
    # warning so the failure is never silent.
    if result.get("success"):
        _complete = _metrics_completeness_violation(result)
        if _complete is not None:
            _ccls, _cmsg = _complete
            result = {**result, "success": False, "error": _cmsg, "failure_class": _ccls}
            try:
                _emit_dashboard_event(ctx, event_type="run_warning", payload={
                    "code": "metrics_incomplete", "message": _cmsg,
                })
            except Exception:  # noqa: BLE001 — diagnostics must never break the run
                logger.debug("run_experiment: metrics_incomplete event emit failed")
            logger.warning("run_experiment[%s]: %s", getattr(ctx, "run_id", "?"), _cmsg)

    # Scope-shape validation (PR B): if scope is multi-model / multi-dataset,
    # require metrics.json to carry the expected per_model / per_dataset
    # structure. A successful run with the wrong shape is a fail-soft error
    # so the agent's next implement_baseline gets it as repair_context.
    if result.get("success") and result.get("metrics"):
        hint = _validate_scope_metrics(getattr(ctx, "scope_spec", None), result["metrics"])
        if hint is not None:
            # Self-healing scope reduction (2026-05-30): the first few times the
            # metrics are shape-incomplete, treat it as a repairable failure so the
            # agent can add the missing dataset/model. But if the SAME piece is
            # missing K times running, it is demonstrably unobtainable (e.g. WebShop
            # needs an external server) — TOLERATE the reduction: keep the partial,
            # record the gap so the rubric downweights it (not 0), and let the run
            # CONVERGE to its best achievable result instead of looping forever.
            import os as _os

            _counts = getattr(ctx, "_scope_violation_counts", None)
            if not isinstance(_counts, dict):
                _counts = {}
                try:
                    ctx._scope_violation_counts = _counts
                except Exception:  # noqa: BLE001
                    pass
            try:
                _maxr = int(_os.environ.get("OPENRESEARCH_MAX_SCOPE_FAILURE_REPEATS", "2") or "2")
            except ValueError:
                _maxr = 2
            # A gap the agent recorded in data_load_failures is provably
            # uncontrollable — tolerate on first sight, don't make it loop K times.
            _forced = _gap_in_load_failures(hint, result.get("metrics") or {})
            result, _tolerated = _scope_reduce_or_fail(
                result, hint, _counts, _maxr, force_reduce=_forced
            )
            if _tolerated:
                _k = _scope_violation_key(hint)
                logger.warning(
                    "run_experiment[%s]: scope element %r unobtainable (%s) — "
                    "tolerating a SCOPE REDUCTION (partial credit) instead of looping.",
                    getattr(ctx, "run_id", "?"), _k,
                    "recorded data_load_failure" if _forced else f"{_counts.get(_k, 0)} misses",
                )

    # Disk post-check (2026-05-30): the experiment may have ballooned the HF cache and
    # left the shared disk below the floor even though it "succeeded". Surface it as
    # repairable so the next iteration streams/slices instead of downloading again.
    if result.get("success"):
        _disk_post = _disk_floor_violation([
            str(ctx.project_dir), os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", ""),
        ])
        if _disk_post is not None:
            result = {**result, "success": False, "error": _disk_post[1], "failure_class": _disk_post[0]}
            logger.warning("run_experiment[%s]: disk floor breached post-run — %s",
                           getattr(ctx, "run_id", "?"), _disk_post[1][:120])

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

    # Hybrid route merge (2026-06-10): both manifests ran in this one call —
    # graft the stashed grid aggregate into the agent-written family metrics so
    # the scorer sees one combined metrics.json and neither route's work is lost.
    if _hybrid_grid_result is not None:
        result = _merge_hybrid_results(_hybrid_grid_result, result, code_path)

    return _persist_experiment_result(ctx, result, model_id=model_id, eval_env=eval_env)


def _leaf_status(score: object, state: object) -> str:
    """Map a leaf (score, state) pair to a UI status string.

    ``"unavailable"`` when the leaf was skipped (state contains ``skipped``) or
    its score is ``None`` (PR-κ data-unavailable). Otherwise threshold the score:
    ``pass`` (>=0.75) / ``partial`` (>=0.4) / ``fail``. Fail-soft: a malformed
    score coerces to 0.0 → ``fail``.
    """
    if score is None or (isinstance(state, str) and "skipped" in state):
        return "unavailable"
    try:
        s = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "fail"
    if s >= 0.75:
        return "pass"
    if s >= 0.4:
        return "partial"
    return "fail"


def _enrich_area_leaves(area_node: dict, leaf_detail: dict[str, dict]) -> list[dict]:
    """Build the per-area ``leaves`` list for one top-level rubric sub_task.

    Walks the area's sub-tree via ``flatten_leaves`` to get the leaf nodes (which
    carry the human-readable ``requirements`` label + ``id``), then joins each to
    its scored record in ``leaf_detail`` (id -> {score, justification, state}).

    Each emitted leaf: ``{id, label, score, status, why}`` where ``label`` is the
    requirements text (≤140 chars), ``score`` is the leaf score (``None`` when
    unavailable), ``status`` from :func:`_leaf_status`, and ``why`` the leaf
    justification (≤280 chars, single line). Fail-soft per leaf: a malformed leaf
    node is skipped rather than crashing the whole area.
    """
    from backend.evals.paperbench.leaf_scorer import flatten_leaves

    out: list[dict] = []
    try:
        leaf_nodes = flatten_leaves(area_node)
    except Exception:  # noqa: BLE001 — a malformed tree must not crash the area
        return out
    for node in leaf_nodes:
        try:
            if not isinstance(node, dict):
                continue
            lid = str(node.get("id", "") or "")
            if not lid:
                continue
            label = " ".join(str(node.get("requirements") or "").split())[:140]
            detail = leaf_detail.get(lid) or {}
            raw_score = detail.get("score") if "score" in detail else None
            state = detail.get("state")
            status = _leaf_status(raw_score, state)
            score_out: float | None = None
            if status != "unavailable":
                try:
                    score_out = max(0.0, min(1.0, float(raw_score)))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    score_out = None
            why = " ".join(str(detail.get("justification") or "").split())[:280]
            out.append({
                "id": lid,
                "label": label,
                "score": score_out,
                "status": status,
                "why": why,
            })
        except Exception:  # noqa: BLE001 — skip the one bad leaf, keep the rest
            continue
    return out


def _rubric_areas(rubric: dict, leaf_scores_list: list[dict]) -> list[dict]:
    """Derive a flat ``areas`` list from the top-level rubric sub_tasks.

    Each top-level sub_task becomes one area entry:
      {"area": <requirements text, truncated>, "score": <rolled-up float>,
       "weight": <raw weight int/float>,
       "leaves": [{id, label, score, status, why}, ...]}

    The ``leaves`` list carries leaf-level detail (which criteria fail + why),
    mapped to the area via the same rubric-tree structure that produces the
    rollup (``flatten_leaves`` over the area sub-tree).

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

    # Build a {leaf_id: full-record} map so per-area leaves can attach
    # score + justification + state (the label/requirements come from the tree).
    leaf_detail: dict[str, dict] = {
        str(e["id"]): e
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
        areas.append({
            "area": name,
            "score": score,
            "weight": weight,
            "leaves": _enrich_area_leaves(task, leaf_detail),
        })
    return areas


def _recent_experiment_errors(project_dir: "Path", limit: int = 3) -> list[dict]:
    """Return the last ``limit`` failed ``run_experiment`` entries as UI rows.

    Reads ``experiment_runs.jsonl`` in ``project_dir`` and selects the most
    recent lines with ``success=False``. Each row: ``{kind, message, iteration}``
    where ``kind`` is the classified ``failure_class`` (fallback ``cause_kind`` /
    ``"unknown"``), ``message`` is the error string truncated to 200 chars
    (falling back to a logs tail when ``error`` is absent — the real-world common
    case), and ``iteration`` is the entry's iteration if recorded else ``None``.

    Fail-soft: a missing/unreadable file or any malformed line yields ``[]`` (or
    the rows parsed so far) — observability must never break the run.
    """
    import json as _json_re

    path = project_dir / "experiment_runs.jsonl"
    try:
        if not path.exists():
            return []
        # Read the tail only — the file can grow large over a long run.
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:  # noqa: BLE001 — best-effort
        return []

    out: list[dict] = []
    # Walk newest-first so we collect the most recent failures.
    for raw in reversed(lines):
        if len(out) >= max(0, int(limit)):
            break
        raw = raw.strip()
        if not raw or '"success"' not in raw:
            continue
        try:
            entry = _json_re.loads(raw)
        except Exception:  # noqa: BLE001 — skip a corrupt line
            continue
        if not isinstance(entry, dict) or entry.get("success") is not False:
            continue
        kind = str(
            entry.get("failure_class")
            or entry.get("cause_kind")
            or "unknown"
        )
        # `error` is frequently None on real failures — the diagnostic lives in
        # the logs tail (e.g. a ModuleNotFoundError traceback). Fall back so the
        # UI row is actually informative.
        message = entry.get("error")
        if not message:
            logs = str(entry.get("logs") or "")
            message = logs[-200:] if logs else (entry.get("suggested_fix") or "")
        message = " ".join(str(message).split())[:200]
        iteration = entry.get("iteration")
        if not isinstance(iteration, int):
            iteration = None
        out.append({"kind": kind, "message": message, "iteration": iteration})
    return out


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
            ((results.get("success") is False) and (not metrics_present))
            # Defense-in-depth (2026-05-30): placeholder/unpopulated metrics (a
            # non-terminal status, or per_model entries all empty) measured nothing
            # too — engage the honesty ceiling so a half-finished experiment can't
            # score unbounded even if the run_experiment completeness guard was
            # bypassed (e.g. near the wall-clock).
            or (_metrics_completeness_violation(results) is not None)
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
            # Model-load-bug fix (2026-05-31): pass the operator's skip list so
            # requested models whose load failed are not silently excluded from
            # the rubric (only truly operator-skipped models are excluded).
            operator_skip_models=list(
                getattr(getattr(ctx, "scope_spec", None), "skip_models", None) or []
            ),
            # 2026-06-01: the operator's de-scoped environments (skip_datasets,
            # e.g. ALFWorld/WebShop on a Search-QA-only run) gate the environment
            # axis the same way skip_models gates the model axis — verified
            # operator scope is excluded; an agent-laundered env skip stays scored.
            operator_skip_environments=list(
                getattr(getattr(ctx, "scope_spec", None), "skip_datasets", None) or []
            ),
        )
        # Phase 0B: persist the rubric tree so the deterministic finalize re-roll-up
        # (report → leaf_scorer.finalize_rescore) can recompute the score under a
        # late-declared scope WITHOUT re-grading. Best-effort — finalize falls back
        # to today's recorded score if this file is absent.
        try:
            (ctx.project_dir / "rubric_tree.json").write_text(
                __import__("json").dumps(rubric, default=str), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass
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
        # leaf_id -> top-level area name, so the UI can group each weak leaf under
        # its area. Built from the SAME top-level sub_task structure as the area
        # rollup. Fail-soft: a malformed tree yields an empty map (area="").
        _leaf_area: dict[str, str] = {}
        try:
            from backend.evals.paperbench.leaf_scorer import flatten_leaves as _flat
            for _i, _task in enumerate(
                c for c in (rubric.get("sub_tasks") or []) if isinstance(c, dict)
            ):
                _aname = str(_task.get("requirements") or "")[:120] or f"Area {_i + 1}"
                for _ln in _flat(_task):
                    if isinstance(_ln, dict) and _ln.get("id"):
                        _leaf_area[str(_ln["id"])] = _aname
        except Exception:  # noqa: BLE001 — area attribution is best-effort
            _leaf_area = {}

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
                 "justification": e.get("justification", ""),
                 "area": _leaf_area.get(str(e.get("id", "")), "")}
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

        # No-progress convergence detector. The rubric score is the one true
        # objective, so we measure convergence off IT — never off experiment
        # shape (which false-positives a run that is improving on some axis while
        # a scope element stays permanently unobtainable). When the score has
        # flatlined across the last `window` verifications AND the iteration
        # floor is satisfied AND we are still below target, attach a
        # `convergence_note` directing the root to ship its best partial instead
        # of looping. Purely advisory — the hard ceiling is the iteration cap +
        # wall-clock watchdog; this just lets a stuck run converge sooner.
        try:
            import os as _os
            _hist = getattr(ctx, "_rubric_score_history", None)
            if not isinstance(_hist, list):
                _hist = []
                try:
                    ctx._rubric_score_history = _hist
                except Exception:  # noqa: BLE001
                    pass
            _hist.append(float(overall_score))
            try:
                _win = int(_os.environ.get("OPENRESEARCH_RUBRIC_PLATEAU_WINDOW", "3") or "3")
            except ValueError:
                _win = 3
            try:
                _eps = float(_os.environ.get("OPENRESEARCH_RUBRIC_PLATEAU_EPSILON", "0.005") or "0.005")
            except ValueError:
                _eps = 0.005
            try:
                _floor = int(_os.environ.get("OPENRESEARCH_MIN_RUBRIC_ITERATIONS", "2") or "2")
            except ValueError:
                _floor = 2
            _cur_iter = int(getattr(ctx, "current_iteration", 0) or 0)
            if (
                not meets_target
                and _cur_iter >= _floor
                and _rubric_plateaued(_hist, _win, _eps)
            ):
                result["convergence_note"] = (
                    f"NO-PROGRESS: rubric overall_score has held at ~{overall_score:.3f} "
                    f"(< target {target:.3f}) across the last {_win} verifications. You have "
                    "plateaued — re-running the same configuration will not move the score. "
                    "If the remaining gap is unobtainable scope (see scope_gaps / "
                    "data_load_failures), record it in the final report's scope.gaps and call "
                    "FINAL_VAR now with this best partial; otherwise change the APPROACH "
                    "(a materially different hypothesis), not the same experiment again."
                )
                logger.info(
                    "verify_against_rubric[%s]: rubric plateaued at %.3f over %d iters — "
                    "attached convergence_note.",
                    getattr(ctx, "run_id", getattr(ctx, "project_id", "?")), overall_score, _win,
                )
        except Exception:  # noqa: BLE001 — convergence hint is augmenting, never fatal
            logger.exception("verify_against_rubric: plateau detection failed")

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


def read_context_map(*, ctx: "RunContext") -> dict:
    """Return the PEEK-lite intra-run context map (OPENRESEARCH_CONTEXT_MAP).

    A free, deterministic orientation cache unioning the structured outputs of
    understand_section / extract_hyperparameters / detect_environment into
    rlm_state/context_map.json. Returns ``{}`` when the flag is off or nothing
    has been recorded yet. NAVIGATION AID ONLY — never cite it as report
    evidence (the evidence gate remains the backstop).
    """
    from backend.agents.rlm.context_map import read_context_map as _read
    return _read(ctx.project_dir)


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
    "codex_repair": codex_repair,
    "read_context_map": read_context_map,  # PEEK-lite, OPENRESEARCH_CONTEXT_MAP
}

PRIMITIVE_DESCRIPTIONS: dict[str, str] = {
    "understand_section": "understand_section(text_slice) -> dict — datasets, "
        "metrics, training recipe, hardware clues, ambiguities from a text slice. "
        "A PARTIAL PaperClaimMap (no core_contribution/claims/architecture).",
    "extract_hyperparameters": "extract_hyperparameters(text_slice) -> dict — "
        "optimizer, learning rate, batch size, epochs from a slice.",
    "detect_environment": "detect_environment(method_spec) -> dict — an "
        "EnvironmentSpec (dockerfile, python_version, framework, pip_packages). "
        "The returned dockerfile already uses the correct base image for the "
        "active sandbox (runpod/docker/local) — pass the result through to "
        "build_environment unchanged; do NOT construct your own Dockerfile. "
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
    "codex_repair": "codex_repair(task_type, instructions, test_command, "
        "allowed_paths=None, repair_context=None, failure_class=None, readonly=False) "
        "-> dict — optional, default-off Codex CLI repo-editing subagent for "
        "bounded software-engineering repairs only. Use only after a failed "
        "run_experiment with an agent-correctable failure_class such as "
        "syntax_error, missing_module, requirements_not_found, dockerfile_invalid, "
        "contract_violation, or scope_shape_violation. Do NOT use for paper "
        "navigation, paper summaries, rubric judgment, final reports, broad "
        "research, credential inspection, secret search, or high-frequency "
        "rlm_query calls.",
    "read_context_map": "read_context_map() -> dict — PEEK-lite orientation "
        "cache (enabled by OPENRESEARCH_CONTEXT_MAP). Returns already-derived "
        "datasets/metrics/hardware/env facts unioned from understand_section, "
        "extract_hyperparameters, and detect_environment so you can avoid "
        "re-deriving them. Returns {} when disabled or empty. NAVIGATION ONLY — "
        "never cite as report evidence.",
}
