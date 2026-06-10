"""RunpodBackend registers an atexit cleanup after create_sandbox acquires a pod;
destroy() is idempotent (two calls result in at most one delete API call)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend():
    from backend.services.runtime.runpod_backend import RunpodBackend
    return RunpodBackend(api_key="dummy", ssh_key_path="/dev/null")


# ---------------------------------------------------------------------------
# D1 / D3: atexit registration
# ---------------------------------------------------------------------------


def test_create_sandbox_registers_atexit_cleanup():
    """After _owned_pod_ids is populated, atexit.register records the cleanup."""
    backend = _make_backend()
    # Simulate what create_sandbox does: records ownership + registers atexit.
    backend._owned_pod_ids.add("test-pod-id")
    with patch("atexit.register") as mock_register:
        backend._register_atexit_cleanup("test-pod-id")
        mock_register.assert_called_once()
        args, _ = mock_register.call_args
        assert callable(args[0])


# ---------------------------------------------------------------------------
# D3: idempotent destroy
# ---------------------------------------------------------------------------


def test_destroy_sync_is_idempotent():
    """Calling _delete_pod_sync twice sends at most one HTTP DELETE."""
    backend = _make_backend()
    backend._owned_pod_ids.add("test-pod")

    delete_calls: list[str] = []

    def _fake_delete(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        delete_calls.append(url)
        return resp

    def _fake_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"name": "reprolab-test"}
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.Client") as mock_client_cls:
        instance = MagicMock()
        instance.__enter__ = MagicMock(return_value=instance)
        instance.__exit__ = MagicMock(return_value=False)
        instance.delete = _fake_delete
        instance.get = _fake_get
        mock_client_cls.return_value = instance

        # First call: pod is owned, should delete.
        backend._delete_pod_sync("test-pod")
        # After first call, pod is removed from owned set.
        assert "test-pod" not in backend._owned_pod_ids

        # Second call: pod no longer owned, must be a no-op.
        backend._delete_pod_sync("test-pod")

    # Only one DELETE should have been issued.
    assert len(delete_calls) == 1, f"expected 1 delete, got {len(delete_calls)}: {delete_calls}"
