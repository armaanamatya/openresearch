"""Unit tests for the hermetic secret resolver + redaction (Phase 1d, Unit A).

Verbatim from ``docs/history/plans/2026-07-01-phase-1d-credentials-assets-cpu-tier.md``
(Unit A). Hermetic: ``env``/``settings_getter`` are injected fakes, never real
``os.environ`` or real Settings.
"""

from backend.services.runtime.credential_broker import CredentialBroker


def test_get_prefers_env_then_settings():
    b = CredentialBroker(env={"HF_TOKEN": "hf_xxx"})
    assert b.get("hf_token") == "hf_xxx" and b.available("hf_token") is True


def test_get_settings_fallback():
    class _S:  # fake settings
        anthropic_api_key = "sk-ant"

    b = CredentialBroker(env={}, settings_getter=lambda: _S())
    assert b.get("anthropic_api_key") == "sk-ant"


def test_absent_secret_is_none_and_unavailable():
    b = CredentialBroker(env={}, settings_getter=lambda: object())
    assert b.get("hf_token") is None and b.available("hf_token") is False


def test_redact_env_drops_secret_keys():
    out = CredentialBroker.redact_env(
        {"ANTHROPIC_API_KEY": "x", "HF_TOKEN": "y", "SEED": "0", "MODEL": "qwen"}
    )
    assert out == {"SEED": "0", "MODEL": "qwen"}


def test_redact_text_drops_secret_lines():
    text = "line ok\nANTHROPIC_API_KEY=sk-secret\nother fine"
    red = CredentialBroker.redact_text(text)
    assert "sk-secret" not in red and "line ok" in red and "other fine" in red


def test_gated_exclusion_when_absent_else_none():
    from backend.agents.rlm import exclusion as X

    b = CredentialBroker(env={})
    exc = b.gated_exclusion(item="gated-ds", secret_name="hf_token", axis=X.AXIS_DATASET)
    assert exc is not None and exc.verified and exc.kind == X.KIND_ENV_SETUP_FAILED and "gated" in exc.reason.lower()
    b2 = CredentialBroker(env={"HF_TOKEN": "hf_xxx"})
    assert b2.gated_exclusion(item="gated-ds", secret_name="hf_token", axis=X.AXIS_DATASET) is None


def test_never_logs_or_returns_secret_in_redaction(caplog):
    # redact_env must not leak the value anywhere it returns.
    out = CredentialBroker.redact_env({"OPENAI_API_KEY": "sk-leak"})
    assert "sk-leak" not in repr(out)
