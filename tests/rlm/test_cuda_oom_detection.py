from __future__ import annotations

from backend.agents.rlm.primitives import _detect_cuda_oom


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
