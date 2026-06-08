from __future__ import annotations

from backend.agents.rlm.primitives import _detect_cuda_oom, _is_oom_escalation_trigger


def test_detects_exit_code_137():
    assert _detect_cuda_oom(exit_code=137, stderr_tail="") is True


def test_detects_exit_code_minus_9():
    assert _detect_cuda_oom(exit_code=-9, stderr_tail="") is True


def test_detects_pytorch_oom_substring():
    msg = "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB ..."
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is True


def test_detects_torch_outofmemoryerror():
    msg = "torch.cuda.OutOfMemoryError: CUDA out of memory."
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is True


def test_detects_cublas_alloc_failed():
    msg = "RuntimeError: cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED"
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is True


def test_normal_failure_is_not_oom():
    msg = "ImportError: No module named 'transformers'"
    assert _detect_cuda_oom(exit_code=1, stderr_tail=msg) is False


def test_clean_exit_is_not_oom():
    assert _detect_cuda_oom(exit_code=0, stderr_tail="") is False


# --- F-04: watchdog-killed OOM escalation trigger ---------------------------
# The stall-watchdog return dict carries watchdog_killed=True but no exit_code,
# and the OOM marker can be buried earlier than the 4 KB stderr_tail the
# escalation gate inspects. Such a run must still advance the GPU ladder; a
# genuine no-signal staleness kill (no OOM marker) must NOT.

def test_escalation_trigger_delegates_to_detect_cuda_oom():
    # Direct OOM signals still escalate, regardless of the watchdog flag.
    assert _is_oom_escalation_trigger({"logs": ""}, exit_code=137, stderr_tail="") is True
    assert _is_oom_escalation_trigger(
        {"logs": ""},
        exit_code=1,
        stderr_tail="torch.cuda.OutOfMemoryError: CUDA out of memory.",
    ) is True


def test_watchdog_killed_with_oom_marker_only_in_full_logs_escalates():
    buried = "CUDA out of memory. Tried to allocate 2.50 GiB\n" + ("filler line\n" * 5000)
    tail = buried[-4096:]
    # Preconditions: the marker is NOT in the 4 KB tail, so the old gate missed it.
    assert "CUDA out of memory" not in tail
    assert _detect_cuda_oom(exit_code=1, stderr_tail=tail) is False
    result = {"watchdog_killed": True, "logs": buried}
    assert _is_oom_escalation_trigger(result, exit_code=1, stderr_tail=tail) is True


def test_watchdog_killed_without_oom_marker_does_not_escalate():
    # Genuine no-signal staleness — no OOM marker anywhere — must not escalate.
    result = {"watchdog_killed": True, "logs": "epoch 1 step 10\n(no progress for 20 min)\n"}
    assert _is_oom_escalation_trigger(result, exit_code=1, stderr_tail="(no progress)") is False


def test_non_watchdog_non_oom_failure_does_not_escalate():
    result = {"logs": "ImportError: No module named 'transformers'"}
    assert _is_oom_escalation_trigger(
        result, exit_code=1, stderr_tail="ImportError: No module named 'transformers'"
    ) is False
