"""Unit guardrails for scripts/serve_local_llm.py (local vLLM accelerator launcher)
and the batch_reproduce._probe_accelerator helper.

Loaded via importlib because scripts/ is not a package. Pins the 2026-05-29
bring-up fixes: the auth-aware readiness probe, offline-when-cached env, and the
process-group kill that stops orphaned tensor-parallel workers from surviving.
"""
from __future__ import annotations

import importlib.util
import pathlib
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "serve_local_llm.py"
_spec = importlib.util.spec_from_file_location("serve_local_llm", _PATH)
serve = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(serve)

_BATCH_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "batch_reproduce.py"
_bspec = importlib.util.spec_from_file_location("batch_reproduce", _BATCH_PATH)
batch = importlib.util.module_from_spec(_bspec)
assert _bspec and _bspec.loader
_bspec.loader.exec_module(batch)


def _ctx(resp):
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


# --- readiness probe must speak the server's auth ----------------------------

def test_probe_200_is_ready():
    with patch.object(serve.urllib.request, "urlopen", return_value=_ctx(SimpleNamespace(status=200))):
        assert serve._probe_server("127.0.0.1", 8001, api_key="local") is True


def test_probe_401_counts_as_up():
    # vLLM runs with --api-key; an unauth (or mismatched) probe gets 401, but the
    # server IS up — the port binds only after the model loads.
    err = urllib.error.HTTPError("http://h", 401, "Unauthorized", {}, None)
    with patch.object(serve.urllib.request, "urlopen", side_effect=err):
        assert serve._probe_server("127.0.0.1", 8001, api_key="local") is True


def test_probe_connection_refused_is_not_ready():
    with patch.object(serve.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")):
        assert serve._probe_server("127.0.0.1", 8001, api_key="local") is False


# --- offline-when-cached (no huggingface.co round-trip on boot) ---------------

def test_cached_model_forces_offline_env():
    lease = SimpleNamespace(gpu_indices=[0, 1])
    with patch.object(serve, "_model_is_cached", return_value=True):
        env = serve._build_child_env(lease, model="Qwen/Qwen2.5-Coder-14B-Instruct")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"  # indices, not UUIDs (vLLM int-parses)


def test_uncached_model_stays_online():
    lease = SimpleNamespace(gpu_indices=[3])
    with patch.object(serve, "_model_is_cached", return_value=False):
        env = serve._build_child_env(lease, model="some/uncached-model")
    assert "HF_HUB_OFFLINE" not in env


# --- process-group kill: no orphaned tensor-parallel workers -----------------

def test_terminate_child_signals_process_group_not_just_pid():
    proc = MagicMock()
    proc.poll.return_value = None   # alive
    proc.pid = 4242
    proc.wait.return_value = 0      # exits cleanly on SIGTERM (no escalation)
    with patch.object(serve.os, "getpgid", return_value=4242), \
         patch.object(serve.os, "killpg") as killpg:
        serve._terminate_child(proc, "vLLM")
    # The fix signals the whole group; it must NOT fall back to proc.terminate().
    proc.terminate.assert_not_called()
    assert killpg.call_args_list[0].args == (4242, serve.signal.SIGTERM)


# --- batch_reproduce._probe_accelerator auth-awareness (FIX 1) ---------------
# urllib is imported inside _probe_accelerator, so patch the stdlib target directly.

def test_batch_probe_200_is_reachable(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_ACCELERATOR_API_KEY", raising=False)
    with patch("urllib.request.urlopen", return_value=_ctx(SimpleNamespace(status=200))):
        assert batch._probe_accelerator("127.0.0.1", 8001) is True


def test_batch_probe_401_counts_as_reachable(monkeypatch):
    """vLLM with --api-key returns 401 to an unauthenticated probe; server IS up."""
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_API_KEY", "local")
    err = urllib.error.HTTPError("http://h", 401, "Unauthorized", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        assert batch._probe_accelerator("127.0.0.1", 8001) is True


def test_batch_probe_403_counts_as_reachable(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_API_KEY", "local")
    err = urllib.error.HTTPError("http://h", 403, "Forbidden", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        assert batch._probe_accelerator("127.0.0.1", 8001) is True


def test_batch_probe_connection_refused_is_not_reachable(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_ACCELERATOR_API_KEY", raising=False)
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        assert batch._probe_accelerator("127.0.0.1", 8001) is False


def test_batch_probe_sends_auth_header(monkeypatch):
    """Verify the Authorization header is sent so an api-key-protected vLLM responds."""
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_API_KEY", "tok-abc")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["auth"] = req.get_header("Authorization")
        cm = MagicMock()
        cm.__enter__.return_value = SimpleNamespace(status=200)
        cm.__exit__.return_value = False
        return cm

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        batch._probe_accelerator("127.0.0.1", 8001)

    assert captured.get("auth") == "Bearer tok-abc"
