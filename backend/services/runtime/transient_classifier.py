"""Classify a SandboxRuntimeError as fatal / transient / code_bug / unknown.

Conservative defaults: anything unrecognised returns ``unknown``, which the
caller maps to ``repairable`` (not retried forever).  The three positive
classes drive the retry policy in ``_execute_in_sandbox``:

* ``fatal``     — user must act (credit balance, auth).  No retry.
* ``transient`` — infra flake (SSH drop, 500, capacity).  Retry with backoff.
* ``code_bug``  — agent code error (AttributeError, etc.).  Feed repair loop,
                  do NOT retry — same code will fail again.
* ``unknown``   — conservative: treat as repairable upstream, no auto-retry.
"""

from __future__ import annotations

from enum import Enum


class TransientClass(str, Enum):
    fatal = "fatal"
    transient = "transient"
    code_bug = "code_bug"
    unknown = "unknown"


# RunPod-specific funding / auth failures — user must add credit or fix key.
_FATAL_MARKERS: tuple[str, ...] = (
    "RUNPOD_BALANCE_TOO_LOW",
    "balance_too_low",
    "RUNPOD_AUTH_FAILED",
    "auth_failed",
    "Unauthorized",
    "quota_exceeded",
)

# Network / infra transients — safe to retry with backoff.
_TRANSIENT_MARKERS: tuple[str, ...] = (
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
)

# Errors produced by broken agent code — retrying the same code is futile.
_CODE_BUG_MARKERS: tuple[str, ...] = (
    "preflight_blocked",
    "contract_violation",
    "AttributeError",
    "NameError",
    "ImportError",
    "TypeError",
    "SyntaxError",
)


def classify_exception(exc: BaseException) -> TransientClass:
    """Inspect exception type + message to classify transient-ness.

    Scans fatal markers first (highest priority — do not retry even if a
    transient marker also matches), then transient, then code_bug.  Falls
    back to ``unknown`` for unrecognised messages.
    """
    text = str(exc)
    for marker in _FATAL_MARKERS:
        if marker in text:
            return TransientClass.fatal
    for marker in _TRANSIENT_MARKERS:
        if marker in text:
            return TransientClass.transient
    for marker in _CODE_BUG_MARKERS:
        if marker in text:
            return TransientClass.code_bug
    return TransientClass.unknown


__all__ = [
    "TransientClass",
    "classify_exception",
]
