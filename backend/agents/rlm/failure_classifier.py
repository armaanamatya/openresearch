"""Auto root-cause classifier for failed run_experiment results.

Every recent failure mode this session falls into a small set of
recognisable shapes — ``ModuleNotFoundError``, ``CUDA out of memory``,
RunPod 500 ``RUNPOD_CAPACITY_EXHAUSTED``, the 755 MB torch wheel that
truncated mid-stream, requirements.txt missing, attempt_isolation
PermissionError, etc.  Classifying them programmatically and surfacing
the class + a concrete suggested fix in the ``experiment_completed``
event makes the next iteration's ``repair_context`` actionable instead
of an opaque traceback the agent has to re-diagnose every time.

The classifier is called from ``_persist_experiment_result`` AFTER the
result dict is finalised but BEFORE the SSE event fires.  Adds two
keys to the payload:

  ``failure_class``  one of the literals in :data:`FAILURE_CLASSES`
  ``suggested_fix``  a one-line operator/agent-actionable hint

Fail-soft: any exception inside classify_failure returns
``("unknown", "")`` — observability MUST NEVER break the run.
"""

from __future__ import annotations

from typing import Final


# Canonical failure classes — keep stable so dashboards / monitoring
# can filter on these strings.  When you add a new class, also add a
# matching detector and suggested-fix below.
FAILURE_CLASSES: Final[tuple[str, ...]] = (
    "missing_module",            # ModuleNotFoundError: <pkg>
    "torch_redundancy",          # PyPI torch download failed; base image has it
    "network_flake",             # pip / hf / urllib download truncated
    "cuda_oom",                  # torch.cuda.OutOfMemoryError or similar
    "runpod_capacity",           # RUNPOD_CAPACITY_EXHAUSTED / no instances
    "runpod_transient_500",      # Bare RunPod 500 (treated as escalation trigger)
    "runpod_ssh_timeout",        # Pod created but never reachable via SSH
    "runpod_balance_too_low",    # Funding exhausted
    "requirements_not_found",    # pip CWD vs requirements.txt path mismatch
    "missing_dataset",           # HuggingFace datasets URI failure / dataset 404
    "exec_timeout",              # Per-command 4h cap hit
    "watchdog_killed",           # Lane E watchdog tripped
    "preflight_blocked",         # Lane F+I caught a scope / surrogate violation
    "permission_denied",         # File ownership / FS perm
    "syntax_error",              # train.py wouldn't parse
    "scope_shape_violation",     # metrics.json missing per_model when scope demands
    "contract_violation",        # RubricContract violations only (no other class fired)
    "unknown",                   # falls-through
)


def _suggest(klass: str, *, extra: str = "") -> str:
    """Map class → one-line suggested-fix string."""
    base = {
        "missing_module":
            "ensure the package is in requirements.txt or pre-installed in the base image; "
            "if it's matplotlib / numpy / tqdm, the auto-derive should have included it — "
            "verify the Dockerfile parsed cleanly",
        "torch_redundancy":
            "remove torch / torchvision / torchaudio from requirements.txt — the "
            "runpod/pytorch base image already provides them",
        "network_flake":
            "transient — the next attempt should succeed; consider mounting a "
            "persistent pip cache via REPROLAB_RUNPOD_NETWORK_VOLUME_ID",
        "cuda_oom":
            "reduce batch size in train.py or escalate to a larger-VRAM SKU "
            "(Lane gpu_escalation handles the latter automatically)",
        "runpod_capacity":
            "RunPod has no available instances of the requested SKU — escalator "
            "advances the ladder automatically; ensure REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS "
            "is high enough or switch tier (SECURE has better availability than COMMUNITY)",
        "runpod_transient_500":
            "bare 500 from RunPod — automatic ladder advance; if the next SKU also "
            "500s, RunPod itself may be experiencing an outage",
        "runpod_ssh_timeout":
            "pod created but SSH never reached READY — the ladder advances automatically",
        "runpod_balance_too_low":
            "add funds at https://runpod.io/console/user/billing — non-retryable until fixed",
        "requirements_not_found":
            "auto-derive should have written requirements.txt to code/; check that "
            "the Dockerfile exists at runs/<id>/Dockerfile and parses cleanly",
        "missing_dataset":
            "the HuggingFace datasets URL changed shape — pin datasets==2.18.0 OR "
            "switch to the torchvision direct-fetch path",
        "exec_timeout":
            "the per-command 4h cap fired; reduce train.py epochs OR set "
            "REPROLAB_RUN_EXPERIMENT_TIMEOUT_S higher",
        "watchdog_killed":
            "no exec.log / heartbeat / SSE-event activity for 10 min — agent or pod was "
            "wedged; next iteration starts fresh against the persistent pod",
        "preflight_blocked":
            "pre-flight validator found a scope shortcut or surrogate model in train.py; "
            "see contract_violations for the exact lines to fix",
        "permission_denied":
            "file owned by a container's root user; attempt_isolation now chowns to host "
            "uid before archive — confirm Docker is reachable",
        "syntax_error":
            "train.py wouldn't parse; pre-flight emits the exact line number",
        "scope_shape_violation":
            "metrics.json was emitted but missing per_model when the scope is multi-model; "
            "agent's next iteration gets the precise hint",
        "contract_violation":
            "RubricContract found gaps vs paper_targets — see contract_violations field",
        "unknown":
            "classifier didn't recognise the failure shape; logs_tail will have the trace",
    }
    msg = base.get(klass, "")
    if extra:
        msg = f"{msg} — {extra}" if msg else extra
    return msg


def classify_failure(result: dict) -> tuple[str, str]:
    """Return ``(failure_class, suggested_fix)`` for a failed result dict.

    ``result`` is the run_experiment return shape:
        {"success": False, "error": str, "logs": str, "metrics": dict,
         "contract_violations": list[dict], "watchdog_killed": bool, ...}

    On a successful result returns ``("ok", "")``.  Fail-soft on every
    parse path — returns ``("unknown", "")`` on internal exceptions.
    """
    try:
        if result.get("success"):
            return ("ok", "")

        err = str(result.get("error") or "")
        logs = str(result.get("logs") or "")
        haystack = (err + " " + logs).lower()

        # Pre-flight + watchdog flag fast-paths
        if result.get("pre_flight_blocked"):
            return ("preflight_blocked", _suggest("preflight_blocked"))
        if result.get("watchdog_killed"):
            return ("watchdog_killed", _suggest("watchdog_killed"))
        if result.get("scope_shape_violation"):
            return ("scope_shape_violation", _suggest("scope_shape_violation"))

        # RunPod-specific sentinels (from runpod_backend exceptions)
        if "runpod_capacity_exhausted" in haystack:
            return ("runpod_capacity", _suggest("runpod_capacity"))
        if "runpod_transient_500" in haystack:
            return ("runpod_transient_500", _suggest("runpod_transient_500"))
        if "runpod_ssh_timeout" in haystack:
            return ("runpod_ssh_timeout", _suggest("runpod_ssh_timeout"))
        if "runpod_balance_too_low" in haystack or "balance is too low" in haystack:
            return ("runpod_balance_too_low", _suggest("runpod_balance_too_low"))

        # Timeout sentinels
        if "timed out after" in haystack and "run_experiment" in haystack:
            return ("exec_timeout", _suggest("exec_timeout"))

        # CUDA OOM
        if (
            "cuda out of memory" in haystack
            or "torch.cuda.outofmemoryerror" in haystack
            or "cuda error: out of memory" in haystack
            or "cublas_status_alloc_failed" in haystack
        ):
            return ("cuda_oom", _suggest("cuda_oom"))

        # Network flake during a torch wheel download — distinguish from generic
        # network because the fix is "strip torch from requirements" not "retry"
        if (
            "not enough bytes were received" in haystack
            and ("torch" in haystack or "pytorch" in haystack)
        ):
            return ("torch_redundancy", _suggest("torch_redundancy"))

        # Generic network flake
        if (
            "not enough bytes were received" in haystack
            or "download failed after" in haystack
            or "connection reset" in haystack
            or "connection timed out" in haystack
            or "temporary failure in name resolution" in haystack
        ):
            return ("network_flake", _suggest("network_flake"))

        # Missing requirements.txt at pip's CWD
        if (
            "could not open requirements file" in haystack
            and "requirements.txt" in haystack
        ):
            return ("requirements_not_found", _suggest("requirements_not_found"))

        # Permission denied (typical attempt_isolation case)
        if "permission denied" in haystack and (
            "_rmtree_safe_fd" in haystack or "shutil" in haystack or "errno 13" in haystack
        ):
            return ("permission_denied", _suggest("permission_denied"))

        # Python SyntaxError
        if "syntaxerror" in haystack or "indentationerror" in haystack:
            return ("syntax_error", _suggest("syntax_error"))

        # ModuleNotFoundError — try to surface which module
        if "modulenotfounderror" in haystack:
            import re as _re
            m = _re.search(r"no module named ['\"]([\w\.\-]+)['\"]", err + " " + logs, _re.IGNORECASE)
            module = m.group(1) if m else ""
            if module.lower() in {"torch", "torchvision", "torchaudio"}:
                return ("torch_redundancy", _suggest("torch_redundancy"))
            extra = f"module: {module}" if module else ""
            return ("missing_module", _suggest("missing_module", extra=extra))

        # HuggingFace datasets URI failure
        if (
            "invalid hf uri" in haystack
            or "datasetnotfounderror" in haystack
            or ("huggingface" in haystack and "404" in haystack)
        ):
            return ("missing_dataset", _suggest("missing_dataset"))

        # Contract violations only (set by rubric_contract.validate post-run)
        if result.get("contract_violations"):
            return ("contract_violation", _suggest("contract_violation"))

        return ("unknown", _suggest("unknown"))
    except Exception:  # noqa: BLE001 — observability MUST NEVER break the run
        return ("unknown", "")


__all__ = ["FAILURE_CLASSES", "classify_failure"]
