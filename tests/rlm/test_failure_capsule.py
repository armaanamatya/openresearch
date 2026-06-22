"""Tests for durable failure capsules (feature #11).

Capsules persist bounded/redacted real evidence of a run_experiment failure and
thread it into the next repair iteration. Everything is gated behind
``OPENRESEARCH_FAILURE_CAPSULES`` (default OFF → byte-for-byte unchanged).

Coverage:
  1. OFF → no capsule file written; result carries no capsule text.
  2. ON  → capsule persisted with bounded fields + redacted secrets.
  3. capsule build error → fail-soft (no raise).
  4. capsule text appears in repair-context-bound result only when ON.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm import failure_capsule as fcap
from backend.agents.rlm.primitives import _persist_experiment_result

_FLAG = "OPENRESEARCH_FAILURE_CAPSULES"
_CAPSULE_FILE = "rlm_state/failure_capsules.jsonl"


def _failed_result() -> dict:
    return {
        "success": False,
        "failure_class": "cuda_oom",
        "error": "RuntimeError: CUDA out of memory\nlast line of trace",
        "logs": "epoch 1\nepoch 2\nTraceback (most recent call last)\nCUDA OOM",
        "stderr": "boom\n",
        "command": "python train.py --epochs 3",
        "exit_code": 1,
        "metrics": {"artifact_paths": ["code/metrics.json", "code/train.py"]},
    }


def _stub_ctx(project_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(project_dir=project_dir, project_id="prj_test")


# ---------------------------------------------------------------------------
# (1) OFF → no capsule file, no capsule text on the result
# ---------------------------------------------------------------------------
def test_flag_off_writes_no_capsule(tmp_path, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    ctx = _stub_ctx(tmp_path)
    out = _persist_experiment_result(ctx, _failed_result())
    assert not (tmp_path / _CAPSULE_FILE).exists()
    assert "failure_capsule_text" not in out


def test_capsules_enabled_default_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert fcap.capsules_enabled() is False
    monkeypatch.setenv(_FLAG, "1")
    assert fcap.capsules_enabled() is True


# ---------------------------------------------------------------------------
# (2) ON → capsule persisted, bounded, secrets redacted
# ---------------------------------------------------------------------------
def test_flag_on_persists_capsule_with_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("FOO_API_KEY", "sk-supersecret-value")
    monkeypatch.setenv("BENIGN_VAR", "hello-world")
    ctx = _stub_ctx(tmp_path)

    out = _persist_experiment_result(ctx, _failed_result())

    cap_path = tmp_path / _CAPSULE_FILE
    assert cap_path.exists(), "capsule file must be written when flag ON"
    lines = cap_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    capsule = json.loads(lines[0])

    # Bounded / structured fields present.
    assert capsule["failure_class"] == "cuda_oom"
    assert capsule["error_signature"].startswith("cuda_oom")
    assert "traceback_tail" in capsule and "log_tail" in capsule
    assert capsule["artifacts"], "artifact paths recorded"

    # Redaction: the secret env var name + value must NOT appear anywhere.
    blob = lines[0]
    assert "FOO_API_KEY" not in blob
    assert "sk-supersecret-value" not in blob
    # ...but a benign env var survives the fingerprint.
    assert capsule["env_fingerprint"].get("BENIGN_VAR") == "hello-world"

    # And the result is stamped so repair_context carries it.
    assert "failure_capsule_text" in out


def test_build_capsule_bounds_tails():
    huge = "x\n" * 50_000  # ~100 KB
    capsule = fcap.build_capsule(
        {"failure_class": "unknown", "logs": huge, "stderr": huge},
    )
    assert len(capsule["traceback_tail"]) <= fcap._TAIL_CHARS
    assert len(capsule["log_tail"]) <= fcap._TAIL_CHARS


def test_redaction_drops_secret_lines():
    text = "normal line\nOPENAI_API_KEY=sk-abc123\nanother normal line"
    out = fcap._redact_text(text)
    assert "sk-abc123" not in out
    assert "OPENAI_API_KEY" not in out
    assert "normal line" in out


def test_redact_env_drops_secret_keys():
    env = {
        "MY_TOKEN": "t",
        "DB_PASSWORD": "p",
        "SOME_SECRET": "s",
        "API_KEY": "k",
        "PLAIN": "v",
    }
    out = fcap._redact_env(env)
    assert out == {"PLAIN": "v"}


# ---------------------------------------------------------------------------
# (3) capsule build error → fail-soft (no raise)
# ---------------------------------------------------------------------------
def test_build_capsule_failsoft_on_bad_input():
    # A non-dict-ish result whose .get raises must not propagate.
    class Boom:
        def get(self, *_a, **_k):
            raise RuntimeError("nope")

    capsule = fcap.build_capsule(Boom())  # type: ignore[arg-type]
    assert isinstance(capsule, dict)
    assert capsule.get("failure_class") == "unknown"


def test_persist_capsule_failsoft_on_unwritable(tmp_path, monkeypatch):
    # Point os.replace at a raise to simulate an atomic-swap failure.
    monkeypatch.setattr(
        "backend.agents.rlm.failure_capsule.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("ro")),
    )
    ok = fcap.persist_capsule(tmp_path, {"schema": 1, "failure_class": "x"})
    assert ok is False  # returns False, does not raise


def test_persist_hook_failsoft_does_not_break_result(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setattr(
        "backend.agents.rlm.failure_capsule.build_capsule",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    ctx = _stub_ctx(tmp_path)
    # The run must complete; the failed result is still returned intact.
    out = _persist_experiment_result(ctx, _failed_result())
    assert out["success"] is False
    assert out["failure_class"] == "cuda_oom"


# ---------------------------------------------------------------------------
# (4) capsule text in repair-context-bound result only when ON
# ---------------------------------------------------------------------------
def test_capsule_text_only_when_on(tmp_path, monkeypatch):
    # OFF
    monkeypatch.delenv(_FLAG, raising=False)
    off = _persist_experiment_result(_stub_ctx(tmp_path / "off"), _failed_result())
    assert "failure_capsule_text" not in off

    # ON
    monkeypatch.setenv(_FLAG, "1")
    on = _persist_experiment_result(_stub_ctx(tmp_path / "on"), _failed_result())
    text = on.get("failure_capsule_text")
    assert text and "FAILURE CAPSULE" in text
    assert "cuda_oom" in text


def test_render_capsule_text_empty_on_empty():
    assert fcap.render_capsule_text({}) == ""


def test_success_result_writes_no_capsule(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    ctx = _stub_ctx(tmp_path)
    out = _persist_experiment_result(ctx, {"success": True, "metrics": {}})
    assert not (tmp_path / _CAPSULE_FILE).exists()
    assert "failure_capsule_text" not in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
