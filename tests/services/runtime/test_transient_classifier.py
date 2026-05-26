"""Tests for the transient error classifier (PR-ζ piece ζ.1).

Each marker in each classification bucket must resolve to the correct class.
Unknown/unrecognised messages must resolve to `unknown` (conservative default).
"""

import pytest

from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError
from backend.services.runtime.transient_classifier import (
    TransientClass,
    classify_exception,
)


# ---------------------------------------------------------------------------
# Fatal markers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marker", [
    "RUNPOD_BALANCE_TOO_LOW",
    "balance_too_low",
    "RUNPOD_AUTH_FAILED",
    "auth_failed",
    "Unauthorized",
    "quota_exceeded",
])
def test_fatal_markers_classify_as_fatal(marker):
    exc = RuntimeError(f"API call failed: {marker}")
    assert classify_exception(exc) == TransientClass.fatal


# ---------------------------------------------------------------------------
# Transient markers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marker", [
    "Connection closed",
    "Connection refused",
    "Connection reset",
    "Server error '5",
    "500 Internal Server Error",
    "502 Bad Gateway",
    "503 Service Unavailable",
    "NO_CAPACITY_AVAILABLE",
    "RUNPOD_CAPACITY_EXHAUSTED",
    "RUNPOD_SSH_TIMEOUT",
    "RUNPOD_TRANSIENT_500",
    "network is unreachable",
    "Operation timed out",
])
def test_transient_markers_classify_as_transient(marker):
    exc = RuntimeError(f"Infra failure: {marker}")
    assert classify_exception(exc) == TransientClass.transient


# ---------------------------------------------------------------------------
# Code-bug markers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marker", [
    "preflight_blocked",
    "contract_violation",
    "AttributeError",
    "NameError",
    "ImportError",
    "TypeError",
    "SyntaxError",
])
def test_code_bug_markers_classify_as_code_bug(marker):
    exc = RuntimeError(f"Agent code crashed: {marker}")
    assert classify_exception(exc) == TransientClass.code_bug


# ---------------------------------------------------------------------------
# Unknown / unrecognised
# ---------------------------------------------------------------------------

def test_unknown_exception_string_returns_unknown():
    exc = RuntimeError("Some completely unknown error with no known markers")
    assert classify_exception(exc) == TransientClass.unknown


def test_empty_exception_message_returns_unknown():
    exc = RuntimeError("")
    assert classify_exception(exc) == TransientClass.unknown


# ---------------------------------------------------------------------------
# SandboxRuntimeError wrapping
# ---------------------------------------------------------------------------

def test_sandbox_runtime_error_with_connection_closed():
    exc = SandboxRuntimeError(
        RuntimeCauseKind.backend_unavailable,
        "Connection closed by remote host during artifact sync",
    )
    assert classify_exception(exc) == TransientClass.transient


def test_sandbox_runtime_error_with_balance_too_low():
    exc = SandboxRuntimeError(
        RuntimeCauseKind.backend_unavailable,
        "RUNPOD_BALANCE_TOO_LOW: RunPod account has insufficient funds",
    )
    assert classify_exception(exc) == TransientClass.fatal


def test_sandbox_runtime_error_with_attribute_error():
    exc = SandboxRuntimeError(
        RuntimeCauseKind.command_failed,
        "AttributeError: 'WakeSleepVAE' object has no attribute 'reparameterize'",
    )
    assert classify_exception(exc) == TransientClass.code_bug


# ---------------------------------------------------------------------------
# Priority: fatal wins over transient when both substrings appear
# ---------------------------------------------------------------------------

def test_fatal_takes_priority_over_transient_when_both_appear():
    # Contrived: a message containing both a fatal and a transient marker.
    # Fatal markers are checked first, so fatal wins.
    exc = RuntimeError("RUNPOD_BALANCE_TOO_LOW: Connection closed during deduction check")
    assert classify_exception(exc) == TransientClass.fatal
