"""Tests for backend.agents.rlm.failure_classifier — auto root-cause tagging.

Every recent failure mode this session has a recognisable shape.  These
tests pin the classifier's recognition of each one PLUS the fail-soft
``unknown`` fallback so an unrecognised failure never crashes the run.
"""

from __future__ import annotations

from backend.agents.rlm.failure_classifier import FAILURE_CLASSES, classify_failure


def test_success_returns_ok() -> None:
    klass, fix = classify_failure({"success": True})
    assert klass == "ok"
    assert fix == ""


def test_unknown_when_no_signal() -> None:
    klass, _ = classify_failure({"success": False})
    assert klass == "unknown"


def test_module_not_found_matplotlib() -> None:
    result = {
        "success": False, "error": "",
        "logs": "Traceback ...\nModuleNotFoundError: No module named 'matplotlib'\n",
    }
    klass, fix = classify_failure(result)
    assert klass == "missing_module"
    assert "matplotlib" in fix or "requirements.txt" in fix


def test_module_not_found_torch_routed_to_redundancy() -> None:
    result = {
        "success": False, "error": "",
        "logs": "ModuleNotFoundError: No module named 'torch'\n",
    }
    klass, fix = classify_failure(result)
    # torch is special — fix is "remove from requirements", not "add to requirements"
    assert klass == "torch_redundancy"
    assert "remove" in fix.lower()


def test_torch_redundancy_via_network_truncation() -> None:
    """The Adam v10 #2 signature — pip retried torch wheel 6× then aborted."""
    result = {
        "success": False, "error": "",
        "logs": (
            "Download failed after 6 attempts because not enough bytes were "
            "received (400.6 MB/755.5 MB) URL: torch-2.2.0-cp310-...\n"
            "ModuleNotFoundError: No module named 'matplotlib'\n"
        ),
    }
    klass, _ = classify_failure(result)
    assert klass == "torch_redundancy"


def test_network_flake_generic() -> None:
    result = {
        "success": False, "error": "",
        "logs": "Download failed after 6 attempts because not enough bytes were received (foo)\n",
    }
    klass, _ = classify_failure(result)
    assert klass == "network_flake"


def test_cuda_oom() -> None:
    result = {
        "success": False, "error": "",
        "logs": "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...\n",
    }
    klass, fix = classify_failure(result)
    assert klass == "cuda_oom"
    assert "batch" in fix.lower() or "vram" in fix.lower()


def test_runpod_capacity() -> None:
    result = {
        "success": False,
        "error": "RUNPOD_CAPACITY_EXHAUSTED: no instances of NVIDIA L40S available",
    }
    klass, _ = classify_failure(result)
    assert klass == "runpod_capacity"


def test_runpod_transient_500() -> None:
    result = {
        "success": False,
        "error": "RUNPOD_TRANSIENT_500: Runpod API request failed (HTTP 500)",
    }
    klass, _ = classify_failure(result)
    assert klass == "runpod_transient_500"


def test_runpod_ssh_timeout() -> None:
    result = {
        "success": False, "error": "RUNPOD_SSH_TIMEOUT: pod never reached READY on a6000",
    }
    klass, _ = classify_failure(result)
    assert klass == "runpod_ssh_timeout"


def test_runpod_balance_too_low() -> None:
    result = {
        "success": False, "error": "RUNPOD_BALANCE_TOO_LOW: account balance is too low",
    }
    klass, _ = classify_failure(result)
    assert klass == "runpod_balance_too_low"


def test_exec_timeout() -> None:
    result = {"success": False, "error": "run_experiment: timed out after 14400 s"}
    klass, _ = classify_failure(result)
    assert klass == "exec_timeout"


def test_watchdog_killed_flag() -> None:
    result = {"success": False, "error": "...", "watchdog_killed": True}
    klass, _ = classify_failure(result)
    assert klass == "watchdog_killed"


def test_preflight_blocked_flag() -> None:
    result = {"success": False, "pre_flight_blocked": True}
    klass, _ = classify_failure(result)
    assert klass == "preflight_blocked"


def test_scope_shape_violation_flag() -> None:
    result = {"success": False, "scope_shape_violation": True}
    klass, _ = classify_failure(result)
    assert klass == "scope_shape_violation"


def test_requirements_not_found() -> None:
    result = {
        "success": False, "error": "",
        "logs": "ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'\n",
    }
    klass, _ = classify_failure(result)
    assert klass == "requirements_not_found"


def test_permission_denied_attempt_isolation() -> None:
    result = {
        "success": False, "error": "",
        "logs": (
            "File '/usr/lib/python/shutil.py', line 717, in _rmtree_safe_fd\n"
            "PermissionError: [Errno 13] Permission denied\n"
        ),
    }
    klass, _ = classify_failure(result)
    assert klass == "permission_denied"


def test_syntax_error_in_train_py() -> None:
    result = {"success": False, "error": "", "logs": "SyntaxError: invalid syntax\n"}
    klass, _ = classify_failure(result)
    assert klass == "syntax_error"


def test_missing_dataset_hf_uri() -> None:
    result = {
        "success": False, "error": "",
        "logs": "Invalid HF URI 'hf://datasets/imdb@...'\n",
    }
    klass, _ = classify_failure(result)
    assert klass == "missing_dataset"


def test_contract_violation_only() -> None:
    result = {
        "success": False, "error": None,
        "contract_violations": [{"area": "Result match", "detail": "off by 19%"}],
    }
    klass, fix = classify_failure(result)
    assert klass == "contract_violation"
    assert "contract" in fix.lower() or "paper_targets" in fix.lower()


def test_failsoft_on_internal_error(monkeypatch) -> None:
    """If something crashes inside classify_failure (we throw a non-dict in),
    return unknown rather than raising."""
    klass, fix = classify_failure(None)  # type: ignore[arg-type]
    assert klass == "unknown"
    assert fix == ""


def test_all_known_classes_have_suggested_fix() -> None:
    """Every entry in FAILURE_CLASSES must produce a non-empty suggested fix."""
    from backend.agents.rlm.failure_classifier import _suggest
    for klass in FAILURE_CLASSES:
        if klass == "unknown":
            continue  # unknown is allowed to have a short hint
        fix = _suggest(klass)
        assert fix, f"missing suggested fix for class {klass!r}"
