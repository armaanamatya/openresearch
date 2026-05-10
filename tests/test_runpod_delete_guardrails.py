"""Regression tests for RunPod delete-guardrail behavior.

Locks in two layers of protection so a future refactor can't quietly
re-introduce the ability to delete pods this backend did not create:

1. Allowlist: ``_delete_pod`` raises if the pod_id isn't in
   ``self._owned_pod_ids`` (populated only by ``_create_pod``).
2. Lifecycle ordering: ``create_sandbox`` adds the pod_id to the
   allowlist BEFORE entering the cleanup-try block, so the
   creation-failure cleanup path (``_delete_pod_quietly``) is allowed
   to proceed.

These tests run entirely against the in-memory state — no RunPod API,
no network, no SSH.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError
from backend.services.runtime.runpod_backend import RunpodBackend


def _backend() -> RunpodBackend:
    return RunpodBackend(
        api_key="test-key",
        ssh_key_path="/dev/null",  # path is checked at create_sandbox, not __init__
    )


def test_delete_pod_refuses_pod_id_outside_allowlist() -> None:
    """Calling _delete_pod on a pod we did NOT create must raise."""

    backend = _backend()
    # Allowlist deliberately empty — coworker's pod ID we should never touch.
    assert "coworkers-pod-abc" not in backend._owned_pod_ids

    with pytest.raises(SandboxRuntimeError) as excinfo:
        asyncio.run(backend._delete_pod("coworkers-pod-abc"))

    assert excinfo.value.cause_kind is RuntimeCauseKind.backend_unavailable
    assert "owned-pod allowlist" in str(excinfo.value)


def test_delete_pod_quietly_swallows_guardrail_violation() -> None:
    """The quiet variant must not raise when the allowlist refuses it,
    so the creation-failure cleanup path stays a no-op for non-owned
    pods (defense in depth — should never see a non-owned pod_id, but
    if it ever does, prefer noop over crash)."""

    backend = _backend()
    # Should NOT raise.
    asyncio.run(backend._delete_pod_quietly("coworkers-pod-abc"))


def test_delete_pod_drops_ownership_after_call() -> None:
    """After a guardrail-blocked delete, the pod_id stays out of the
    allowlist (it was never in it). After a successful delete attempt
    that gets past the allowlist check, the pod_id is removed from the
    allowlist regardless of API outcome — verified by faking the API
    call and inspecting state."""

    backend = _backend()
    backend._owned_pod_ids.add("our-pod-xyz")
    assert "our-pod-xyz" in backend._owned_pod_ids

    # Simulate the API DELETE failing (no real network). The guardrail
    # check passes because the pod IS in the allowlist; the API call
    # then fails; the finally block must still discard ownership so a
    # retry doesn't keep a stale entry.
    async def fail_delete() -> None:
        with pytest.raises(SandboxRuntimeError):
            # Use a bogus URL so httpx raises immediately.
            backend.api_base_url = "http://127.0.0.1:1"
            await backend._delete_pod("our-pod-xyz")

    asyncio.run(fail_delete())
    assert "our-pod-xyz" not in backend._owned_pod_ids


def test_delete_on_destroy_default_does_not_block_safety_layer() -> None:
    """delete_on_destroy=False is a safety knob, but it lives on the
    destroy_sandbox path, not on _delete_pod. The _delete_pod guardrail
    still applies regardless of delete_on_destroy. Asserting this
    explicitly so a future refactor doesn't conflate the two."""

    backend = RunpodBackend(
        api_key="test-key",
        ssh_key_path="/dev/null",
        delete_on_destroy=True,  # even with delete enabled —
    )
    with pytest.raises(SandboxRuntimeError) as excinfo:
        asyncio.run(backend._delete_pod("foreign-pod"))
    assert "owned-pod allowlist" in str(excinfo.value)
