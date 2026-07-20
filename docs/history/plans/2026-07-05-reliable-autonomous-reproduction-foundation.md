# Reliable Autonomous Reproduction — Foundation (WS-A/B/C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make paper reproduction reliably autonomous by routing the RLM root to Claude Opus 4.8 + executor/grader to Sonnet 5 (via the Azure Foundry Anthropic endpoint) and promoting the harness-owned deterministic driver (`run_lifecycle_primary`) to the reproduction backbone — validated by a passing SDAR Search-3B Phase-1 run.

**Architecture:** A new `anthropic-foundry` LLM provider reaches Opus 4.8 / Sonnet 5 through the Anthropic Messages API on Azure Foundry, scoped **per-client** (root `get_client` patch + `AnthropicMessagesClient` `base_url` param + executor `ClaudeAgentOptions.env`) so it never leaks into the `claude-oauth` path, guarded by a hard co-residency invariant. The deterministic driver (already present, bypasses the flaky root loop) becomes the default backbone via a run-spec toggle; its `_synth_result_from_summary` is hardened to project an honest report. The execute-mode wiring bug (`REPRODUCTION_MODE` defaulting to `adapt`) is fixed and made fail-loud.

**Tech Stack:** Python 3.12 (dev venv 3.14), FastAPI, the `rlm`/`rlms` library, `claude-agent-sdk`, `anthropic` SDK, pytest, ruff 0.15.16.

**Spec:** `docs/history/specs/2026-07-05-reliable-autonomous-reproduction-design.md` (commit `a9cbb32b`).

## Global Constraints

- **Default-OFF / byte-identical:** every change is a no-op when its new flag/field/token is absent; unset ⇒ byte-identical to today. Each task ships a hermetic ON **and** OFF test.
- **TDD:** write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- **No OAuth leak:** NEVER set `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY` in the process-global `os.environ`. Scope per-client/per-subprocess only. `anthropic-foundry` and `claude-oauth` must never be co-resident in one run (enforced + tested).
- **Verified facts:** endpoint `https://appradhann-4738-resource.services.ai.azure.com/anthropic/v1/messages`; auth header `x-api-key: $AZURE_FOUNDRY_API_KEY` (already in `.env`); `anthropic-version: 2023-06-01`; models `claude-opus-4-8` + `claude-sonnet-5` (both live, HTTP 200).
- **Test runner:** `.venv/bin/python -m pytest <path> -v`. Lint: `uvx ruff@0.15.16 check <files>`.
- **Git:** commit at each task; descriptive present-tense headline (what+symptom+resolution); identity `lolout1` / `appradhann@gmail.com`; **no Co-Authored-By trailer**; push `deepinvent` only (and only when asked). Use `git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit`.
- **Money:** no GPU spend inside this plan except the pre-authorized ~$30 SDAR Phase-1 slice at the end (Task 15). The $400 grid stays checkpointed.
- **Isolate from the external-monitor feature:** do NOT stage `backend/services/external_monitor/`, `backend/routes/external_runs.py`, `configs/external_runs.json`, `frontend/.../external-runs/`, or the `app.py`/`lab-sidebar.tsx` external-monitor hunks in any foundation commit. `git add` exact paths only.

---

## File Structure

**Create:**
- `backend/agents/runtime/foundry_anthropic.py` — canonical resolver `(base_url=…/anthropic/v1, api_key, {opus,sonnet})`; mirror of `foundry_endpoint.py`.
- `backend/agents/rlm/_anthropic_foundry_patch.py` — `apply_anthropic_foundry_backend_patch()` registering the rlm `get_client` for the `anthropic-foundry` backend literal (constructs `anthropic.Anthropic(api_key, base_url)`).
- `scripts/foundry_anthropic_smoke.py` — opt-in live reachability smoke (opus + sonnet ping).
- Tests: `tests/agents/runtime/test_foundry_anthropic.py`, `tests/agents/rlm/test_anthropic_foundry_patch.py`, `tests/rlm/test_anthropic_foundry_roles.py`, `tests/rlm/test_coresidency_guard.py`, plus extensions to existing suites.

**Modify:**
- `backend/services/context/workspace/tools/anthropic_messages_client.py:85-102` — add `base_url: str | None = None` param.
- `backend/agents/rlm/grader_transport.py:195-204` — add `"anthropic-foundry"` transport branch.
- `backend/agents/rlm/models.py` — `_VALID_RLM_BACKENDS`, `_BACKEND_ENV_KEY`, `_MODEL_ALIASES`, two new `RootModel` entries, `resolve_root_model` base_url branch.
- `backend/agents/rlm/role_models.py` — `PROVIDER_ANTHROPIC_FOUNDRY`, `SUBROLE_PROVIDERS`, `_VALIDATED_SUBROLE_PROVIDERS`, `_ROLE_VOCAB`, `_classify_model_family`.
- `backend/agents/rlm/run.py` — import+apply the foundry patch (`~123-124`); executor provider map (`429-441`); co-residency guard + subrole-backend handling (`~2884-2911`); `_synth_result_from_summary` (`1072-1121`); primary-path input assertions (`~3862-3882`).
- `backend/agents/rlm/lifecycle_driver.py` — edge-case hardening (implement_baseline repairable re-drive; pre-committed ordered plan).
- `configs/sdar_execute_run_spec.json` — extend with WS-A/B keys.
- `scripts/sdar_gcp_run.sh` / `scripts/gcp_sdar_preflight.sh` — honor `--run-spec` / `REPRODUCTION_MODE` override.

---

## Task 1: Anthropic-Foundry credential resolver

**Files:**
- Create: `backend/agents/runtime/foundry_anthropic.py`
- Test: `tests/agents/runtime/test_foundry_anthropic.py`

**Interfaces:**
- Consumes: env vars `AZURE_FOUNDRY_ENDPOINT` / `AZURE_FOUNDRY_API_KEY` / optional `AZURE_FOUNDRY_ANTHROPIC_ENDPOINT` / `AZURE_FOUNDRY_ANTHROPIC_OPUS` / `AZURE_FOUNDRY_ANTHROPIC_SONNET`; Settings `azure_foundry_endpoint`/`azure_foundry_api_key` (via `foundry_endpoint._env_or_settings`).
- Produces: `resolve_foundry_anthropic_credentials() -> tuple[str, str, dict[str,str]]` returning `(base_url, api_key, {"opus": "claude-opus-4-8", "sonnet": "claude-sonnet-5"})`; `has_foundry_anthropic_credentials() -> bool`; module constant `ANTHROPIC_FOUNDRY_MODES: frozenset[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/runtime/test_foundry_anthropic.py
import pytest
from backend.agents.runtime import foundry_anthropic as fa


def test_derives_anthropic_base_from_openai_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://res-x.services.ai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-123")
    monkeypatch.delenv("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT", raising=False)
    base_url, api_key, models = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "https://res-x.services.ai.azure.com/anthropic/v1"
    assert api_key == "k-123"
    assert models == {"opus": "claude-opus-4-8", "sonnet": "claude-sonnet-5"}
    assert fa.has_foundry_anthropic_credentials() is True


def test_explicit_anthropic_endpoint_wins_and_strips_messages(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
                       "https://res-x.services.ai.azure.com/anthropic/v1/messages")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-9")
    base_url, _, _ = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "https://res-x.services.ai.azure.com/anthropic/v1"


def test_model_overrides(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k")
    monkeypatch.setenv("AZURE_FOUNDRY_ANTHROPIC_OPUS", "claude-opus-4-9")
    _, _, models = fa.resolve_foundry_anthropic_credentials()
    assert models["opus"] == "claude-opus-4-9"


def test_unset_is_empty_and_absent(monkeypatch):
    for v in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
              "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(fa, "_settings_endpoint", lambda: "")  # neutralize .env fallback
    base_url, api_key, _ = fa.resolve_foundry_anthropic_credentials()
    assert base_url == "" and api_key == ""
    assert fa.has_foundry_anthropic_credentials() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agents/runtime/test_foundry_anthropic.py -v`
Expected: FAIL (ModuleNotFoundError: `backend.agents.runtime.foundry_anthropic`).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/agents/runtime/foundry_anthropic.py
"""Anthropic-compatible Azure Foundry endpoint resolver.

Mirror of ``foundry_endpoint.py`` (the OpenAI-compat resolver) but for the
``…/anthropic/v1`` Messages-API surface that serves Claude Opus 4.8 + Sonnet 5.
Same resource + key as the OpenAI-compat Foundry; only the base-url path differs.
os.environ → Settings-backed .env, fail-soft; unset ⇒ "" (callers decide).
"""
from __future__ import annotations

from urllib.parse import urlparse

from backend.agents.runtime.foundry_endpoint import _env_or_settings

ANTHROPIC_FOUNDRY_MODES: frozenset[str] = frozenset(
    {"anthropic-foundry", "opus-foundry", "sonnet-foundry"}
)
_OPUS_DEFAULT = "claude-opus-4-8"
_SONNET_DEFAULT = "claude-sonnet-5"


def normalize_anthropic_base_url(raw: str) -> str:
    """Return the canonical ``…/anthropic/v1`` base (strip trailing ``/messages`` / slash)."""
    url = (raw or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/messages"):
        url = url[: -len("/messages")].rstrip("/")
    if url.endswith("/anthropic/v1"):
        return url
    if url.endswith("/anthropic"):
        return url + "/v1"
    return url  # already a base (explicit override) — trust it


def _settings_endpoint() -> str:
    """The OpenAI-compat endpoint (host source when no explicit anthropic endpoint)."""
    return _env_or_settings("AZURE_FOUNDRY_ENDPOINT", "azure_foundry_endpoint")


def _derive_base_url() -> str:
    explicit = _env_or_settings("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
                                "azure_foundry_anthropic_endpoint")
    if explicit:
        return normalize_anthropic_base_url(explicit)
    oai = _settings_endpoint()
    if not oai:
        return ""
    parsed = urlparse(oai)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/anthropic/v1"


def resolve_foundry_anthropic_credentials() -> tuple[str, str, dict[str, str]]:
    """Return ``(base_url, api_key, {"opus":…, "sonnet":…})``. Any unset field is ``""``."""
    base_url = _derive_base_url()
    api_key = _env_or_settings("AZURE_FOUNDRY_API_KEY", "azure_foundry_api_key")
    models = {
        "opus": _env_or_settings("AZURE_FOUNDRY_ANTHROPIC_OPUS",
                                 "azure_foundry_anthropic_opus") or _OPUS_DEFAULT,
        "sonnet": _env_or_settings("AZURE_FOUNDRY_ANTHROPIC_SONNET",
                                   "azure_foundry_anthropic_sonnet") or _SONNET_DEFAULT,
    }
    return base_url, api_key, models


def has_foundry_anthropic_credentials() -> bool:
    base_url, api_key, _ = resolve_foundry_anthropic_credentials()
    return bool(base_url and api_key)
```

Note: `_env_or_settings` on a non-existent Settings attr (`azure_foundry_anthropic_*`) fail-softs to `""` (getattr default) — no new Settings fields needed (confirmed: `azure_foundry_api_key`/`_endpoint` exist at `config.py:202/195`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/agents/runtime/test_foundry_anthropic.py -v`
Expected: PASS (4 passed). Then `uvx ruff@0.15.16 check backend/agents/runtime/foundry_anthropic.py tests/agents/runtime/test_foundry_anthropic.py`.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/runtime/foundry_anthropic.py tests/agents/runtime/test_foundry_anthropic.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Add the Azure Foundry Anthropic endpoint resolver (Opus 4.8 / Sonnet 5)"
```

---

## Task 2: `AnthropicMessagesClient` gains a `base_url` param

**Files:**
- Modify: `backend/services/context/workspace/tools/anthropic_messages_client.py:85-102`
- Test: `tests/services/context/workspace/tools/test_anthropic_messages_client_base_url.py`

**Interfaces:**
- Produces: `AnthropicMessagesClient(model, *, api_key=None, base_url=None, max_tokens=4096, timeout=300.0)` — `base_url` forwarded to `anthropic.Anthropic(base_url=…)`; `None` ⇒ SDK default (byte-identical).

- [ ] **Step 1: Write the failing test**

```python
# tests/services/context/workspace/tools/test_anthropic_messages_client_base_url.py
from unittest.mock import patch
from backend.services.context.workspace.tools.anthropic_messages_client import (
    AnthropicMessagesClient,
)


def test_base_url_is_forwarded_to_sdk():
    with patch("anthropic.Anthropic") as mock_anthropic:
        AnthropicMessagesClient(model="claude-sonnet-5", api_key="k",
                                base_url="https://x/anthropic/v1")
    _, kwargs = mock_anthropic.call_args
    assert kwargs["base_url"] == "https://x/anthropic/v1"
    assert kwargs["api_key"] == "k"


def test_base_url_none_is_omitted_byte_identical():
    with patch("anthropic.Anthropic") as mock_anthropic:
        AnthropicMessagesClient(model="claude-sonnet-5")
    _, kwargs = mock_anthropic.call_args
    # None must NOT be passed (or must be None) so the SDK resolves its default.
    assert kwargs.get("base_url", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/services/context/workspace/tools/test_anthropic_messages_client_base_url.py -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'base_url'`).

- [ ] **Step 3: Write minimal implementation**

In `anthropic_messages_client.py`, change the signature (line ~85) and the client construction (line ~100):

```python
    def __init__(
        self,
        model: str = DEFAULT_GRADER_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ) -> None:
        import anthropic

        _client_kwargs: dict = {"api_key": api_key, "timeout": timeout, "max_retries": 6}
        if base_url:
            _client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**_client_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/services/context/workspace/tools/test_anthropic_messages_client_base_url.py -v`
Expected: PASS (2 passed). Ruff-check the two files.

- [ ] **Step 5: Commit**

```bash
git add backend/services/context/workspace/tools/anthropic_messages_client.py tests/services/context/workspace/tools/test_anthropic_messages_client_base_url.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Let AnthropicMessagesClient target a custom base_url (Foundry Anthropic scoping)"
```

---

## Task 3: `grader_transport` `anthropic-foundry` branch (grader + verifier → Sonnet 5)

**Files:**
- Modify: `backend/agents/rlm/grader_transport.py:139-289` (add a branch); reuse Task 1 + Task 2.
- Test: `tests/rlm/test_grader_transport_anthropic_foundry.py`

**Interfaces:**
- Consumes: `resolve_foundry_anthropic_credentials()` (Task 1); `AnthropicMessagesClient(base_url=…)` (Task 2).
- Produces: `build_transport_client(backend="anthropic-foundry", model=…)` returns an `AnthropicMessagesClient` pointed at the Foundry base_url + `AZURE_FOUNDRY_API_KEY` + default model `claude-sonnet-5`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_grader_transport_anthropic_foundry.py
from unittest.mock import patch
from backend.agents.rlm import grader_transport


def test_anthropic_foundry_backend_builds_scoped_client(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-42")
    with patch.object(grader_transport, "AnthropicMessagesClient") as mock_cls:
        grader_transport.build_transport_client(
            backend="anthropic-foundry", model="claude-sonnet-5", role_label="grader"
        )
    _, kwargs = mock_cls.call_args
    assert kwargs["base_url"] == "https://r.services.ai.azure.com/anthropic/v1"
    assert kwargs["api_key"] == "k-42"
    assert kwargs["model"] == "claude-sonnet-5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_grader_transport_anthropic_foundry.py -v`
Expected: FAIL (the unknown-backend warning path returns a non-foundry client; `base_url` assertion fails).

- [ ] **Step 3: Write minimal implementation**

In `grader_transport.py`, immediately after the existing `anthropic` branch (`195-204`), add:

```python
        if backend == "anthropic-foundry":
            from backend.agents.runtime.foundry_anthropic import (
                resolve_foundry_anthropic_credentials,
            )

            base_url, api_key, models = resolve_foundry_anthropic_credentials()
            resolved_model = model or models["sonnet"]
            logger.info("grader_transport[%s]: anthropic-foundry model=%s",
                        role_label, resolved_model)
            return AnthropicMessagesClient(
                model=resolved_model, api_key=api_key, base_url=base_url
            )
```

(Ensure `AnthropicMessagesClient` is imported at module top; it already is for the `anthropic` branch.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_grader_transport_anthropic_foundry.py -v`
Expected: PASS. Ruff-check.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/grader_transport.py tests/rlm/test_grader_transport_anthropic_foundry.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Route the grader/verifier transport to Sonnet 5 via the Foundry Anthropic endpoint"
```

---

## Task 4: Root `get_client` patch + registry entries (`opus-foundry` / `sonnet-foundry`)

**Files:**
- Create: `backend/agents/rlm/_anthropic_foundry_patch.py`
- Modify: `backend/agents/rlm/models.py` (`_VALID_RLM_BACKENDS 334-347`, `_BACKEND_ENV_KEY 356-361`, two `RootModel` entries near the `claude` entry `219-227`, `_MODEL_ALIASES 631-642`, `resolve_root_model` base_url branch `770-774`)
- Test: `tests/agents/rlm/test_anthropic_foundry_patch.py`, extend `tests/rlm/test_registry.py` if present.

**Interfaces:**
- Consumes: Task 1 resolver; the rlm library's `get_client` registry.
- Produces: `apply_anthropic_foundry_backend_patch()` — idempotent; makes rlm build `anthropic.Anthropic(api_key=$AZURE_FOUNDRY_API_KEY, base_url=…/anthropic/v1)` for `backend="anthropic-foundry"`. Root models `opus-foundry` (model `claude-opus-4-8`) and `sonnet-foundry` (`claude-sonnet-5`), `rlm_backend="anthropic-foundry"`, `api_key_env="AZURE_FOUNDRY_API_KEY"`, `paper_validated=True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rlm/test_anthropic_foundry_patch.py
from unittest.mock import patch
from backend.agents.rlm import models
from backend.agents.rlm._anthropic_foundry_patch import (
    apply_anthropic_foundry_backend_patch,
    build_anthropic_foundry_client,
)


def test_registry_has_foundry_root_entries():
    assert "opus-foundry" in models.ROOT_MODELS
    assert models.ROOT_MODELS["opus-foundry"].backend_kwargs["model_name"] == "claude-opus-4-8"
    assert models.ROOT_MODELS["opus-foundry"].rlm_backend == "anthropic-foundry"
    assert models.ROOT_MODELS["sonnet-foundry"].backend_kwargs["model_name"] == "claude-sonnet-5"


def test_resolve_opus_foundry_alias():
    entry = models.resolve_root_model("opus-4-8")
    assert entry.key == "opus-foundry"


def test_client_builder_targets_foundry(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-7")
    with patch("anthropic.Anthropic") as mock_anthropic:
        build_anthropic_foundry_client({"model_name": "claude-opus-4-8"})
    _, kwargs = mock_anthropic.call_args
    assert kwargs["base_url"] == "https://r.services.ai.azure.com/anthropic/v1"
    assert kwargs["api_key"] == "k-7"


def test_patch_is_idempotent():
    apply_anthropic_foundry_backend_patch()
    apply_anthropic_foundry_backend_patch()  # second call must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_anthropic_foundry_patch.py -v`
Expected: FAIL (module + registry entries absent).

- [ ] **Step 3a: Write the patch module**

```python
# backend/agents/rlm/_anthropic_foundry_patch.py
"""Register the rlm ``get_client`` for the ``anthropic-foundry`` backend literal.

The rlm ``AnthropicClient`` constructor takes no ``base_url``, so a foundry-anthropic
ROOT is reached only by teaching rlm's client factory to build an
``anthropic.Anthropic(api_key, base_url)`` for this backend — WITHOUT touching the
process-global ``ANTHROPIC_BASE_URL`` (which would hijack ``claude-oauth``).
Mirror of ``apply_oauth_backend_patch``; idempotent.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_APPLIED = False


def build_anthropic_foundry_client(backend_kwargs: dict[str, Any]):
    """Construct an rlm-compatible Anthropic client pinned to the Foundry endpoint."""
    from backend.agents.runtime.foundry_anthropic import (
        resolve_foundry_anthropic_credentials,
    )
    from rlm.clients.anthropic import AnthropicClient

    base_url, api_key, _models = resolve_foundry_anthropic_credentials()
    model_name = backend_kwargs.get("model_name", "claude-opus-4-8")
    max_tokens = backend_kwargs.get("max_tokens", 8192)
    client = AnthropicClient(api_key=api_key, model_name=model_name, max_tokens=max_tokens)
    # AnthropicClient built anthropic.Anthropic(api_key=…) with no base_url; re-point
    # its underlying sync+async SDK clients at the Foundry base_url in-place.
    import anthropic

    client.client = anthropic.Anthropic(api_key=api_key, base_url=base_url,
                                        timeout=getattr(client, "timeout", 300.0))
    client.async_client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url,
                                                   timeout=getattr(client, "timeout", 300.0))
    return client


def apply_anthropic_foundry_backend_patch() -> None:
    """Teach rlm's client factory about the ``anthropic-foundry`` backend. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    from rlm import clients as _rlm_clients

    _orig_get_client = _rlm_clients.get_client

    def _patched_get_client(backend: str, backend_kwargs: dict[str, Any] | None = None):
        if backend == "anthropic-foundry":
            return build_anthropic_foundry_client(backend_kwargs or {})
        return _orig_get_client(backend, backend_kwargs)

    _rlm_clients.get_client = _patched_get_client
    # Also patch the name the RLM class imported, if it bound it directly.
    try:
        import rlm.core.rlm as _rlm_core
        if hasattr(_rlm_core, "get_client"):
            _rlm_core.get_client = _patched_get_client
    except Exception:  # noqa: BLE001
        logger.debug("anthropic-foundry patch: rlm.core.rlm.get_client not rebound")
    _APPLIED = True
    logger.info("anthropic-foundry backend patch applied")
```

Note for the implementer: verify the rlm client factory symbol name (`rlm.clients.get_client` vs `rlm.clients.__init__.get_client`) against `.venv/.../rlm/clients/__init__.py` (the mapper saw the backend dispatch there at lines 26-48) and bind whichever the `RLM` class actually calls. If rlm dispatches by a dict, register the key instead. Adjust the two lines above to match; the test `test_client_builder_targets_foundry` covers the builder regardless.

- [ ] **Step 3b: Add the registry entries + backend literal + aliases (models.py)**

Add to `_VALID_RLM_BACKENDS` (334-347): `"anthropic-foundry"`. Add to `_BACKEND_ENV_KEY` (356-361): `"anthropic-foundry": "AZURE_FOUNDRY_API_KEY"`. Clone the `claude` entry (219-227) twice:

```python
        "opus-foundry": RootModel(
            key="opus-foundry",
            rlm_backend="anthropic-foundry",
            backend_kwargs={"model_name": "claude-opus-4-8"},
            sub_backend="anthropic-foundry",
            sub_backend_kwargs={"model_name": "claude-sonnet-5"},
            prompt_addendum="",
            paper_validated=True,
            api_key_env="AZURE_FOUNDRY_API_KEY",
        ),
        "sonnet-foundry": RootModel(
            key="sonnet-foundry",
            rlm_backend="anthropic-foundry",
            backend_kwargs={"model_name": "claude-sonnet-5"},
            sub_backend="anthropic-foundry",
            sub_backend_kwargs={"model_name": "claude-sonnet-5"},
            prompt_addendum="",
            paper_validated=True,
            api_key_env="AZURE_FOUNDRY_API_KEY",
        ),
```

Add aliases in the `_MODEL_ALIASES` foundry block (631-642) — keep DISTINCT from `opus`/`sonnet` (which map to `claude-oauth` at 601/603):

```python
    "opus-foundry": "opus-foundry",
    "opus-4-8": "opus-foundry",
    "claude-opus-4-8": "opus-foundry",
    "sonnet-foundry": "sonnet-foundry",
    "claude-sonnet-5": "sonnet-foundry",
```

If `cred_provider` (90-111) is used by the preflight, add an `opus-foundry`/`sonnet-foundry` → `"azure-foundry"` case.

- [ ] **Step 3c: base_url injection is handled by the patch, not `_inject_api_key`**

Because the client is built by the `get_client` patch, no `_inject_foundry_kwargs`-style base_url injection is needed in `resolve_root_model`. Confirm `resolve_root_model` still injects the api_key via `api_key_env` (745-761): the `anthropic-foundry` backend must validate `AZURE_FOUNDRY_API_KEY` (not `ANTHROPIC_API_KEY`) — `api_key_env` on the entry handles this. Ensure it does NOT fall into the `anthropic-oauth` early-return (707-734).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_anthropic_foundry_patch.py tests/rlm/test_registry.py -v`
Expected: PASS. Update `tests/rlm/test_registry.py` `EXPECTED` set if it enumerates ROOT_MODELS keys. Ruff-check.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/_anthropic_foundry_patch.py backend/agents/rlm/models.py tests/agents/rlm/test_anthropic_foundry_patch.py tests/rlm/test_registry.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Add opus-foundry/sonnet-foundry roots reaching Opus 4.8 via a scoped rlm get_client patch"
```

---

## Task 5: Role-model tokens `opus-foundry` / `sonnet-foundry`

**Files:**
- Modify: `backend/agents/rlm/role_models.py` (`PROVIDER_* 58-71`, `SUBROLE_PROVIDERS 76-84`, `_VALIDATED_SUBROLE_PROVIDERS 88-90`, `_ROLE_VOCAB 103-142`, `_classify_model_family 145-195`)
- Test: `tests/rlm/test_anthropic_foundry_roles.py`

**Interfaces:**
- Produces: `resolve_role_models(...)` accepts tokens `opus-foundry`/`sonnet-foundry` for executor/verifier/grader; `RoleSpec.provider == "anthropic-foundry"`, `.family == "claude"`, `.model ∈ {claude-opus-4-8, claude-sonnet-5}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_anthropic_foundry_roles.py
from backend.agents.rlm.role_models import resolve_role_models, PROVIDER_ANTHROPIC_FOUNDRY


def test_sonnet_foundry_executor_and_grader():
    sel = resolve_role_models({"executor": "sonnet-foundry", "grader": "sonnet-foundry"})
    assert sel.executor.provider == PROVIDER_ANTHROPIC_FOUNDRY
    assert sel.executor.model == "claude-sonnet-5"
    assert sel.executor.family == "claude"
    assert sel.grader.model == "claude-sonnet-5"


def test_unset_roles_are_byte_identical_none():
    sel = resolve_role_models({})
    assert sel.executor is None or sel.executor.provider != PROVIDER_ANTHROPIC_FOUNDRY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_anthropic_foundry_roles.py -v`
Expected: FAIL (`ImportError: PROVIDER_ANTHROPIC_FOUNDRY`; token unknown).

- [ ] **Step 3: Write minimal implementation**

- Add `PROVIDER_ANTHROPIC_FOUNDRY = "anthropic-foundry"` near line 66.
- Add it to `SUBROLE_PROVIDERS` (76-84) and `_VALIDATED_SUBROLE_PROVIDERS` (88-90) (it IS Claude — suppresses the fidelity warning).
- Add to `_ROLE_VOCAB` (103-142):
  ```python
      "sonnet-foundry": (PROVIDER_ANTHROPIC_FOUNDRY, "claude-sonnet-5"),
      "opus-foundry": (PROVIDER_ANTHROPIC_FOUNDRY, "claude-opus-4-8"),
  ```
- In `_classify_model_family` (145-195), map `PROVIDER_ANTHROPIC_FOUNDRY → "claude"` (near the anthropic case ~173).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_anthropic_foundry_roles.py tests/agents/rlm/test_role_models.py -v`
Expected: PASS (new + existing role-model suite green). Ruff-check.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/role_models.py tests/rlm/test_anthropic_foundry_roles.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Add opus-foundry/sonnet-foundry role tokens (Claude family, validated sub-role)"
```

---

## Task 6: `run.py` wiring — apply the patch, executor subprocess env, co-residency guard

**Files:**
- Modify: `backend/agents/rlm/run.py` (imports+apply `~79-124`; `_resolve_agent_runtime` `429-441`; subrole handling + guard `~2884-2911`)
- Modify: `backend/agents/runtime/claude_runtime.py:86` (thread a per-run `env` dict into `ClaudeAgentOptions`)
- Create: `tests/rlm/test_coresidency_guard.py`

**Interfaces:**
- Consumes: Tasks 4 + 5.
- Produces: `assert_no_foundry_oauth_coresidency(root_key, role_selection) -> None` (raises `ValueError` on a mixed `anthropic-foundry` + `claude-oauth` run); the executor runtime for an `anthropic-foundry` spec passes `ClaudeAgentOptions(env={"ANTHROPIC_BASE_URL":…, "ANTHROPIC_API_KEY":…})`; the root patch is applied at import.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_coresidency_guard.py
import pytest
from backend.agents.rlm.run import assert_no_foundry_oauth_coresidency


class _Spec:
    def __init__(self, provider): self.provider = provider


class _Sel:
    def __init__(self, **roles): self.__dict__.update(roles)
    def specs(self): return [v for v in self.__dict__.values() if v is not None]


def test_guard_raises_on_mixed_foundry_and_oauth():
    sel = _Sel(executor=_Spec("anthropic-foundry"), grader=_Spec("anthropic-oauth"))
    with pytest.raises(ValueError, match="co-resident"):
        assert_no_foundry_oauth_coresidency("opus-foundry", sel)


def test_guard_allows_all_foundry():
    sel = _Sel(executor=_Spec("anthropic-foundry"), grader=_Spec("anthropic-foundry"))
    assert_no_foundry_oauth_coresidency("opus-foundry", sel) is None


def test_guard_allows_all_oauth_no_foundry():
    sel = _Sel(executor=_Spec("anthropic-oauth"))
    assert_no_foundry_oauth_coresidency("claude-oauth", sel) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_coresidency_guard.py -v`
Expected: FAIL (`ImportError: assert_no_foundry_oauth_coresidency`).

- [ ] **Step 3a: Add the guard + apply the patch (run.py)**

Near the other patch applications (`run.py:123-124`), add:

```python
from backend.agents.rlm._anthropic_foundry_patch import apply_anthropic_foundry_backend_patch
apply_anthropic_foundry_backend_patch()
```

Add the guard function (module level):

```python
def assert_no_foundry_oauth_coresidency(root_key: str, role_selection) -> None:
    """A run must not mix anthropic-foundry and claude-oauth (a global ANTHROPIC_BASE_URL
    would hijack the OAuth path). Raise if both families appear across root + roles."""
    providers: set[str] = set()
    rk = (root_key or "").lower()
    if "foundry" in rk and ("opus" in rk or "sonnet" in rk):
        providers.add("anthropic-foundry")
    if rk in ("claude-oauth", "opus", "sonnet"):
        providers.add("anthropic-oauth")
    specs = role_selection.specs() if hasattr(role_selection, "specs") else []
    for sp in specs:
        p = getattr(sp, "provider", None)
        if p in ("anthropic-foundry", "anthropic-oauth"):
            providers.add(p)
    if "anthropic-foundry" in providers and "anthropic-oauth" in providers:
        raise ValueError(
            "anthropic-foundry and claude-oauth cannot be co-resident in one run "
            "(a global ANTHROPIC_BASE_URL would hijack the OAuth path)."
        )
```

Call it right after `resolve_role_models(...)` (`~2884-2895`), before building clients.

- [ ] **Step 3b: Executor subprocess env (claude_runtime.py + run.py)**

In `_resolve_agent_runtime` (429-441), map `"anthropic-foundry" → "anthropic"` and, when the executor spec provider is `anthropic-foundry`, resolve `(base_url, api_key, _)` and pass a `subprocess_env={"ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_API_KEY": api_key}` into the runtime, which forwards it to `ClaudeAgentOptions(env=subprocess_env)` at `claude_runtime.py:86`. Do NOT set these in `os.environ`.

```python
# run.py, in _resolve_agent_runtime, executor override branch (~429-441):
if spec is not None and spec.provider == "anthropic-foundry":
    from backend.agents.runtime.foundry_anthropic import resolve_foundry_anthropic_credentials
    _b, _k, _ = resolve_foundry_anthropic_credentials()
    runtime = make_runtime("anthropic")
    runtime.subprocess_env = {"ANTHROPIC_BASE_URL": _b, "ANTHROPIC_API_KEY": _k}
    runtime.agent_model = spec.model  # "claude-sonnet-5"
    return runtime
```

In `claude_runtime.py` `run_agent` (~86), merge `getattr(self, "subprocess_env", None)` into `ClaudeAgentOptions(env=...)`.

- [ ] **Step 3c: Verifier/grader env bridge**

At the verifier client build (`2913-2922`) and the grader env bridge (`2983-2986`), when the sub-role provider is `anthropic-foundry`, set `OPENRESEARCH_GRADER_BACKEND=anthropic-foundry` (and the verifier `build_transport_client(backend="anthropic-foundry", …)`) so Task 3's branch fires. Ensure `_subrole_backend`/`resolve_anthropic_subrole_backend` does NOT remap a foundry sub-role to `oauth`/`anthropic` (special-case: if `spec.provider == "anthropic-foundry"`, return `"anthropic-foundry"`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_coresidency_guard.py -v`
Expected: PASS. Then a broad regression: `.venv/bin/python -m pytest tests/rlm/ tests/agents/rlm/ -q` — no new failures vs baseline (record the pre-existing failures from the handoff: `test_accelerator`, `test_external_validator`, `test_report_validation_stamp`, `test_gcp_orchestrator_settings::test_claude_code_oauth_token_prefixed_env_override`). Ruff-check touched files.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/run.py backend/agents/runtime/claude_runtime.py tests/rlm/test_coresidency_guard.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Wire the Opus-root/Sonnet-role Foundry routing with a per-subprocess env and a co-residency guard"
```

---

## Task 7: Live reachability smoke script (opt-in, not CI)

**Files:**
- Create: `scripts/foundry_anthropic_smoke.py`

**Interfaces:** none (operator tool).

- [ ] **Step 1: Write the script**

```python
# scripts/foundry_anthropic_smoke.py
"""Opt-in live smoke of the Foundry Anthropic endpoint (opus + sonnet). Costs a few tokens.
Run: .venv/bin/python scripts/foundry_anthropic_smoke.py"""
from __future__ import annotations

import sys

from backend.agents.runtime.foundry_anthropic import resolve_foundry_anthropic_credentials


def main() -> int:
    import anthropic

    base_url, api_key, models = resolve_foundry_anthropic_credentials()
    if not (base_url and api_key):
        print("FAIL: missing base_url/api_key (set AZURE_FOUNDRY_ENDPOINT + _API_KEY)")
        return 1
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    ok = True
    for label, model in models.items():
        try:
            msg = client.messages.create(
                model=model, max_tokens=16,
                messages=[{"role": "user", "content": "reply with one word: pong"}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            print(f"[{label}:{model}] OK stop={msg.stop_reason} text={text!r}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[{label}:{model}] FAIL {type(exc).__name__}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it (operator, live)**

Run: `.venv/bin/python scripts/foundry_anthropic_smoke.py`
Expected: `[opus:claude-opus-4-8] OK …` and `[sonnet:claude-sonnet-5] OK …`.

- [ ] **Step 3: Commit**

```bash
git add scripts/foundry_anthropic_smoke.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Add an opt-in live smoke for the Foundry Anthropic Opus/Sonnet endpoint"
```

---

## Task 8: Harden `_synth_result_from_summary` (the load-bearing driver fix)

**Files:**
- Modify: `backend/agents/rlm/run.py:1072-1121`
- Test: `tests/rlm/test_synth_result_projection.py`

**Interfaces:**
- Consumes: the `run_lifecycle_primary` summary dict (`{rubric_score, verify_result, driven, stopped_reason, …}`) + `ctx`.
- Produces: `_synth_result_from_summary(summary, ctx)` returns an `RLMChatCompletion`-shaped object whose `.response` is a JSON report projecting `verdict`/`reproduction_summary`/`baseline_metrics` from the driven `verify_result` + evidence, even when `rubric_score is None` but gradeable evidence exists; returns a `failed`-shaped result ONLY when there is genuinely no evidence.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_synth_result_projection.py
import json
from backend.agents.rlm.run import _synth_result_from_summary


class _Ctx:
    project_id = "p1"
    cost_ledger = []
    def remaining_s(self): return 999


def test_projects_report_from_scored_summary():
    summary = {"rubric_score": 0.46,
               "verify_result": {"overall_score": 0.46, "target_score": 0.456,
                                 "meets_target": True},
               "driven": ["implement_baseline", "run_experiment", "verify_against_rubric"]}
    res = _synth_result_from_summary(summary, _Ctx())
    payload = json.loads(res.response)
    assert payload["verdict"] in ("reproduced", "partial")
    assert payload.get("rubric") or payload.get("overall_score") is not None


def test_completed_but_unscored_with_evidence_is_not_failed():
    summary = {"rubric_score": None,
               "verify_result": {"overall_score": None},
               "driven": ["implement_baseline", "run_experiment"],
               "has_evidence": True}
    res = _synth_result_from_summary(summary, _Ctx())
    payload = json.loads(res.response)
    assert payload["verdict"] != "failed"  # honest partial, not a scoreless failure


def test_genuinely_empty_run_fails_honestly():
    summary = {"rubric_score": None, "verify_result": None, "driven": [], "has_evidence": False}
    res = _synth_result_from_summary(summary, _Ctx())
    payload = json.loads(res.response)
    assert payload["verdict"] == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_synth_result_projection.py -v`
Expected: FAIL (current impl returns `None`/failed when `rubric_score is None`; `test_completed_but_unscored…` fails).

- [ ] **Step 3: Write minimal implementation**

Replace the body of `_synth_result_from_summary` (1072-1121) so it always emits a JSON `response`. Project from `verify_result`; derive `verdict` via the existing `_reconcile_verdict`-compatible rules (score ≥ target ⇒ `reproduced`; evidence-but-below ⇒ `partial`; no evidence ⇒ `failed`). Keep the `RLMChatCompletion` shape the report writer consumes (`.response` = `json.dumps(report_dict)`, `.metadata` carrying iterations, `.usage_summary`). Read the real class shape from the current 1072-1121 body and the `_synth_*` return type; preserve it. Base `has_evidence` on `summary.get("has_evidence")` OR a non-empty `verify_result` OR `driven` containing `run_experiment`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_synth_result_projection.py tests/rlm/test_run_lifecycle_primary.py -v`
Expected: PASS (new + existing primary suite). Ruff-check.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/run.py tests/rlm/test_synth_result_projection.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Project an honest report from the lifecycle-driver summary instead of failing a scoreless-but-evidenced run"
```

---

## Task 9: Primary-path inputs + run-spec engagement

**Files:**
- Modify: `backend/agents/rlm/run.py:3862-3882` (assert wrapped tools / paper_text / rubric_spec on the primary branch)
- Test: `tests/rlm/test_lifecycle_primary_inputs.py`

**Interfaces:**
- Consumes: `_lifecycle_primary_enabled()` (1052-1063, already reads `OPENRESEARCH_LIFECYCLE_PRIMARY`); `run_lifecycle_primary`.
- Produces: on the primary branch, `run_lifecycle_primary` receives the WRAPPED `custom_tools` dict (3635) + non-empty `paper_text` + `rubric_spec`; a missing input logs a loud `run_warning` and falls back to the normal loop (never silently no-ops).

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_lifecycle_primary_inputs.py
import backend.agents.rlm.run as run_mod


def test_primary_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_PRIMARY", "1")
    assert run_mod._lifecycle_primary_enabled() is True
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_PRIMARY", "0")
    assert run_mod._lifecycle_primary_enabled() is False
    monkeypatch.delenv("OPENRESEARCH_LIFECYCLE_PRIMARY", raising=False)
    assert run_mod._lifecycle_primary_enabled() is False  # default OFF (byte-identical)


def test_primary_requires_inputs(monkeypatch):
    # A helper that validates inputs and returns a reason on missing ones.
    ok, reason = run_mod._primary_inputs_ready(tools={"implement_baseline": lambda: None},
                                               paper_text="abc", rubric_spec={"leaves": []})
    assert ok and reason is None
    ok, reason = run_mod._primary_inputs_ready(tools={}, paper_text="", rubric_spec=None)
    assert not ok and reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_lifecycle_primary_inputs.py -v`
Expected: FAIL (`_primary_inputs_ready` absent).

- [ ] **Step 3: Write minimal implementation**

Add `_primary_inputs_ready(tools, paper_text, rubric_spec) -> tuple[bool, str | None]` (returns `(False, "<what is missing>")` if tools empty / paper_text falsy / rubric_spec falsy). In the primary branch (3862-3882), call it before `run_lifecycle_primary`; on `not ok` emit `run_warning(code="lifecycle_primary_skipped", message=reason)` and fall through to `_run_completion_on_worker()` (the normal loop) so the run still proceeds. Default-OFF path (`_lifecycle_primary_enabled()` False) is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_lifecycle_primary_inputs.py tests/rlm/test_run_lifecycle_primary.py tests/rlm/test_run_lifecycle_drive.py -v`
Expected: PASS. Ruff-check.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/run.py tests/rlm/test_lifecycle_primary_inputs.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Gate the lifecycle-primary backbone on ready inputs, falling back loudly when unpopulated"
```

---

## Task 10: Driver edge-case hardening (repairable implement + fingerprint repair + pre-committed plan)

**Files:**
- Modify: `backend/agents/rlm/lifecycle_driver.py` (`drive_lifecycle_chain` 145-415: implement re-drive; `run_lifecycle_primary` 418-574: ordered-plan persistence)
- Test: `tests/rlm/test_lifecycle_driver_hardening.py`

**Interfaces:**
- Consumes: the driver's `_is_repairable`/`_has_gradeable_evidence` helpers (65-113).
- Produces: (a) a bounded re-drive when `implement_baseline` returns a repairable result (not only `run_experiment`); (b) `plan_reproduction`'s result persisted as an ordered `rlm_state/reproduction_plan.json` step list the driver reads back (E2); (c) an honest `stopped_reason="repair_exhausted"` when the evidence fingerprint stops changing across repairs.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_lifecycle_driver_hardening.py
from backend.agents.rlm import lifecycle_driver as ld


def _tool(seq):
    calls = {"n": 0}
    def fn(*a, **k):
        r = seq[min(calls["n"], len(seq) - 1)]; calls["n"] += 1; return r
    return fn, calls


def test_repairable_implement_baseline_triggers_redrive(tmp_path):
    # implement_baseline first returns repairable, then ok; driver should retry it.
    impl, impl_calls = _tool([{"ok": False, "outcome": "repairable"},
                              {"ok": True, "code_path": str(tmp_path)}])
    tools = {
        "understand_section": _tool([{"ok": True}])[0],
        "detect_environment": _tool([{"ok": True}])[0],
        "plan_reproduction": _tool([{"ok": True, "steps": ["s1"]}])[0],
        "implement_baseline": impl,
        "run_experiment": _tool([{"ok": True, "metrics": {"val/success_rate": 0.46}}])[0],
        "verify_against_rubric": _tool([{"overall_score": 0.46}])[0],
    }
    summary = ld.drive_lifecycle_chain(
        tools=tools, ctx=None, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=2)
    assert impl_calls["n"] >= 2  # implement was re-driven on the repairable result
    assert summary["rubric_score"] == 0.46
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_lifecycle_driver_hardening.py -v`
Expected: FAIL (today a repairable `implement_baseline` stops the chain; `impl_calls["n"] == 1`).

- [ ] **Step 3: Write minimal implementation**

In `drive_lifecycle_chain`, after the `implement_baseline` step (319-331), if `_is_repairable(result)` and repair budget remains, stuff the failure into `plan["repair_context"]` and re-call `implement_baseline` (bounded by `max_repair_iterations`) before proceeding to `run_experiment`. In `run_lifecycle_primary`, after `plan_reproduction`, persist the ordered steps to `Path(ctx.project_dir)/"rlm_state"/"reproduction_plan.json"` (fail-soft) and read them back for dispatch (E2). Add `stopped_reason="repair_exhausted"` when the evidence fingerprint (reuse the evidence-bundle key if available, else `str(sorted(metrics.items()))`) is unchanged across two repairs.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_lifecycle_driver_hardening.py tests/rlm/test_run_lifecycle_primary.py -v`
Expected: PASS. Ruff-check.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/lifecycle_driver.py tests/rlm/test_lifecycle_driver_hardening.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Harden the lifecycle driver: re-drive a repairable implement, persist an ordered plan, stop honestly on repair-exhaustion"
```

---

## Task 11: Startup fail-loud when execute requested but mode≠execute (B2 guard)

**Files:**
- Modify: `backend/agents/rlm/run.py` (after `_resolve_and_clone_repo` persists `repo_spec.json`, ~664-675)
- Test: `tests/rlm/test_execute_mode_wiring.py`

**Interfaces:**
- Consumes: `OPENRESEARCH_REPRODUCTION_MODE`, `OPENRESEARCH_USE_AUTHOR_REPO`, the persisted `rlm_state/repo_spec.json`.
- Produces: `assert_execute_mode_stamped(repro_mode_env, repo_spec) -> None` — raises a loud `RuntimeError` (or emits a fatal `run_warning` + aborts) when `REPRODUCTION_MODE=execute` + `USE_AUTHOR_REPO` truthy but `repo_spec["mode"] != "execute"` (e.g. clone failed / resolver downgraded).

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_execute_mode_wiring.py
import pytest
from backend.agents.rlm.run import assert_execute_mode_stamped


def test_raises_when_execute_requested_but_adapt_stamped():
    with pytest.raises(RuntimeError, match="execute"):
        assert_execute_mode_stamped("execute", {"mode": "adapt", "clone_succeeded": True})


def test_ok_when_execute_stamped():
    assert_execute_mode_stamped("execute", {"mode": "execute", "clone_succeeded": True}) is None


def test_noop_when_not_execute_requested():
    assert_execute_mode_stamped("adapt", {"mode": "adapt"}) is None
    assert_execute_mode_stamped("", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_execute_mode_wiring.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
def assert_execute_mode_stamped(repro_mode_env: str, repo_spec: dict | None) -> None:
    """Fail-loud if execute mode was requested but not stamped (B2 backstop)."""
    if (repro_mode_env or "").strip().lower() != "execute":
        return
    stamped = (repo_spec or {}).get("mode")
    if stamped != "execute":
        raise RuntimeError(
            f"REPRODUCTION_MODE=execute was requested but repo_spec stamped mode={stamped!r} "
            "— the repo did not clone/seed in execute mode; refusing to run in adapt disguise."
        )
```

Call it right after `_resolve_and_clone_repo` writes `repo_spec.json` (guarded by `OPENRESEARCH_USE_AUTHOR_REPO` truthy).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_execute_mode_wiring.py -v`
Expected: PASS. Ruff-check.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/run.py tests/rlm/test_execute_mode_wiring.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Fail loudly when execute mode is requested but repo_spec stamps adapt (B2 backstop)"
```

---

## Task 12: Extend the SDAR Phase-1 run-spec + a round-trip test

**Files:**
- Modify: `configs/sdar_execute_run_spec.json`
- Test: `tests/config/test_sdar_execute_run_spec.py`

**Interfaces:**
- Produces: a run-spec that sets root=`opus-foundry`, roles executor/grader/verifier=`sonnet-foundry`, `OPENRESEARCH_LIFECYCLE_PRIMARY=1`, `OPENRESEARCH_REPRODUCTION_MODE=execute`, `OPENRESEARCH_USE_AUTHOR_REPO=1`, guards ON.

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_sdar_execute_run_spec.py
import json, pathlib


def test_run_spec_selects_foundry_and_execute_and_primary():
    spec = json.loads(pathlib.Path("configs/sdar_execute_run_spec.json").read_text())
    assert spec["OPENRESEARCH_REPRODUCTION_MODE"] == "execute"
    assert spec["OPENRESEARCH_USE_AUTHOR_REPO"] == "1"
    assert spec["OPENRESEARCH_LIFECYCLE_PRIMARY"] == "1"
    # foundry routing (root + roles); accept either dedicated keys or ROLE_MODELS json
    assert "opus-foundry" in json.dumps(spec)
    assert "sonnet-foundry" in json.dumps(spec)
    for guard in ("OPENRESEARCH_ZERO_METRICS_GUARD", "OPENRESEARCH_EVAL_PROVENANCE_GUARD",
                  "OPENRESEARCH_ENV_LIVENESS_GATE", "OPENRESEARCH_EXTERNAL_VALIDATOR"):
        assert spec[guard] == "1"
    # driver-owned per-attempt keys must NOT be present (run_spec contract)
    for banned in ("OPENRESEARCH_SEED_BEST_ATTEMPT", "OPENRESEARCH_TARGET_BEST_FLOOR"):
        assert banned not in spec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/config/test_sdar_execute_run_spec.py -v`
Expected: FAIL (current spec lacks LIFECYCLE_PRIMARY + foundry routing).

- [ ] **Step 3: Write minimal implementation**

Extend `configs/sdar_execute_run_spec.json`:

```json
{
  "OPENRESEARCH_USE_AUTHOR_REPO": "1",
  "OPENRESEARCH_REPRODUCTION_MODE": "execute",
  "OPENRESEARCH_REPO_LOCAL_PATH": "/mnt/sdar-cache/SDAR",
  "OPENRESEARCH_REPO_COMMIT": "f6d0d318",
  "OPENRESEARCH_EXECUTE_OWNS_DEPS": "1",
  "HF_HOME": "/mnt/sdar-cache/hf",
  "OPENRESEARCH_CELL_ENV_PASSTHROUGH": "HF_HOME,HF_DATASETS_CACHE,ALFWORLD_DATA,WEBSHOP_URL,SEARCH_QA_INDEX_DIR",
  "OPENRESEARCH_RLM_ROOT_MODEL": "opus-foundry",
  "OPENRESEARCH_ROLE_MODELS": "{\"executor\":\"sonnet-foundry\",\"grader\":\"sonnet-foundry\",\"verifier\":\"sonnet-foundry\"}",
  "OPENRESEARCH_LIFECYCLE_PRIMARY": "1",
  "OPENRESEARCH_ENV_LIVENESS_GATE": "1",
  "OPENRESEARCH_EVAL_PROVENANCE_GUARD": "1",
  "OPENRESEARCH_ZERO_METRICS_GUARD": "1",
  "OPENRESEARCH_NO_LEARNING_SIGNAL_GATE": "1",
  "OPENRESEARCH_EXTERNAL_VALIDATOR": "1",
  "OPENRESEARCH_MAX_RUN_GPU_USD": "400"
}
```

(Confirm `OPENRESEARCH_REPO_COMMIT` matches the VM's STEP-2 SHA — handoff says `f6d0d318`; verify on the VM before the run. The external validator panel should be a NON-Claude family, e.g. grok, for separation — set `OPENRESEARCH_VALIDATOR_BACKEND=azure-foundry` if a distinct family is wanted; otherwise the Sonnet grader + evidence guards remain the fitness signal.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/config/test_sdar_execute_run_spec.py -v`
Expected: PASS. Validate the JSON parses: `.venv/bin/python -c "import json;json.load(open('configs/sdar_execute_run_spec.json'))"`.

- [ ] **Step 5: Commit**

```bash
git add configs/sdar_execute_run_spec.json tests/config/test_sdar_execute_run_spec.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Point the SDAR Phase-1 run-spec at the Opus-root/Sonnet-role Foundry routing + lifecycle-primary + execute mode"
```

---

## Task 13: Driver script honors the run-spec (drop the hardcoded `adapt`)

**Files:**
- Modify: `scripts/gcp_sdar_preflight.sh:618` (and `scripts/sdar_gcp_run.sh:136-137` if it re-defaults)
- Test: `tests/scripts/test_preflight_mode_override.py` (or a bash assertion in a shellcheck-style test)

**Interfaces:**
- Produces: the preflight/run scripts honor `OPENRESEARCH_REPRODUCTION_MODE=execute` from the environment / `--run-spec` instead of hardcoding `adapt`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_preflight_mode_override.py
import pathlib, re


def test_preflight_does_not_hardcode_adapt_default_only():
    text = pathlib.Path("scripts/gcp_sdar_preflight.sh").read_text()
    # The mode line must respect an env override, not force adapt.
    # Accept `${OPENRESEARCH_REPRODUCTION_MODE:-adapt}` (env wins) but NOT a bare `=adapt`.
    assert not re.search(r'OPENRESEARCH_REPRODUCTION_MODE\s*=\s*["\']?adapt["\']?\s*$',
                         text, re.MULTILINE), "preflight still hardcodes adapt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/test_preflight_mode_override.py -v`
Expected: FAIL if a bare `=adapt` assignment exists. (If line 618 already uses `${…:-adapt}`, the test passes — then the fix is only to EXPORT `REPRODUCTION_MODE=execute` from the run-spec before the preflight builds its `.cache` spec; verify by reading `_spec_add OPENRESEARCH_REPRODUCTION_MODE` at 618.)

- [ ] **Step 3: Write minimal implementation**

Change `gcp_sdar_preflight.sh:618` so the mode comes from the environment (which the run-spec/kickoff exports), e.g. `_spec_add OPENRESEARCH_REPRODUCTION_MODE "${OPENRESEARCH_REPRODUCTION_MODE:-adapt}"` (env override wins). In the Phase-1 kickoff wrapper (Task 14), `export OPENRESEARCH_REPRODUCTION_MODE=execute` (or source the run-spec) before invoking the preflight. If `sdar_gcp_run.sh:136-137` re-defaults, apply the same `${…:-}` pattern.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/scripts/test_preflight_mode_override.py -v` and `bash -n scripts/gcp_sdar_preflight.sh` (syntax check).
Expected: PASS + no bash syntax error.

- [ ] **Step 5: Commit**

```bash
git add scripts/gcp_sdar_preflight.sh scripts/sdar_gcp_run.sh tests/scripts/test_preflight_mode_override.py
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Make the SDAR driver honor REPRODUCTION_MODE=execute instead of hardcoding adapt (B2 wiring)"
```

---

## Task 14: Phase-1 kickoff wrapper (deterministic, autostop-ON)

**Files:**
- Create: `scripts/sdar_phase1_foundry.sh`
- Test: `bash -n` syntax check only (operator-run; no unit test).

**Interfaces:**
- Produces: a wrapper that (on the VM) exports the run-spec env, runs the Phase-1 cell config with root=opus-foundry, uploads `runs/<pid>/` to GCS, and `shutdown`s (autostop). Mirrors the existing `runs/phase1_autonomous.sh` but sources `configs/sdar_execute_run_spec.json` and points `OPENRESEARCH_CELLS_SEED_PATH=configs/sdar_execute_cells_phase1.json`, `--scope-spec '{"models":["Qwen2.5-3B-Instruct"],"datasets":["Search-QA"]}'`.

- [ ] **Step 1: Write the wrapper**

```bash
#!/usr/bin/env bash
# scripts/sdar_phase1_foundry.sh — deterministic SDAR Search-3B Phase-1, autostop ON.
# Run ON the VM (sdar-2model-a). Requires the staged /mnt/sdar-cache disk + .env with AZURE_FOUNDRY_*.
set -euo pipefail
cd "$(dirname "$0")/.."
PID="sdar_phase1_foundry_$(date +%s)"
echo "$PID" > runs/.phase1_project_id

# Load the run-spec keys into the environment (foundry routing + execute + guards + primary).
python3 - "$PID" <<'PY' > runs/.cache/phase1_env.sh
import json, sys, shlex
spec = json.load(open("configs/sdar_execute_run_spec.json"))
for k, v in spec.items():
    print(f"export {k}={shlex.quote(str(v))}")
PY
# shellcheck disable=SC1091
source runs/.cache/phase1_env.sh
export OPENRESEARCH_CELLS_SEED_PATH="configs/sdar_execute_cells_phase1.json"
export HF_HOME="/mnt/sdar-cache/hf"

nohup bash -c '
  set -x
  env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
    .venv/bin/python -m backend.cli reproduce 2605.15155 \
      --project-id '"$PID"' --sandbox local --execution-mode max \
      --model opus-foundry \
      --scope-spec '"'"'{"models":["Qwen2.5-3B-Instruct"],"datasets":["Search-QA"]}'"'"' \
      --run-spec configs/sdar_execute_run_spec.json \
      --paper-hint 2605.15155
  rc=$?
  gsutil -m cp -r runs/'"$PID"' gs://deepinvent-ext-ut-sdar-runs/'"$PID"'/ || true
  cp runs/phase1_run.out gs://... 2>/dev/null || true
  sudo shutdown -h now
' > runs/phase1_run.out 2>&1 &
echo "[phase1] launched $PID (autostop ON). tail -f runs/phase1_run.out"
```

(The implementer MUST reconcile this against the existing `runs/phase1_autonomous.sh` on the VM — reuse its proven GCS-upload + shutdown trap; this is the shape, not a blind replacement. Keep `--run-spec` AND `--model opus-foundry` both explicit; `--model` wins for the root stamp.)

- [ ] **Step 2: Syntax check**

Run: `bash -n scripts/sdar_phase1_foundry.sh`
Expected: no output (valid).

- [ ] **Step 3: Commit**

```bash
git add scripts/sdar_phase1_foundry.sh
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Add the deterministic SDAR Phase-1 Foundry kickoff wrapper (autostop ON)"
```

---

## Task 15: SDAR Phase-1 validation run (operator, ~$30 pre-authorized)

**Files:** none (operator step). This is the plan's acceptance gate.

- [ ] **Step 1: Pre-run reconciliation (no spend)**
  - Confirm `OPENRESEARCH_REPO_COMMIT` in the run-spec matches the VM's STEP-2 SHA (handoff: `f6d0d318`).
  - Decide the verl `eval_provenance.json` schema edge (spec §6 / R4): either exempt verl-sourced cells from `EVAL_PROVENANCE_GUARD` or have the verl adapter write a `records`-shaped sidecar. Land the decision as a tiny guarded change BEFORE the run.
  - Run the full OFF-state regression locally: `.venv/bin/python -m pytest tests/rlm/ tests/agents/rlm/ tests/agents/runtime/ tests/config/ -q` — no new failures vs the known pre-existing set.

- [ ] **Step 2: Restart the VM + smoke the endpoint (~$0)**

```bash
export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud
gcloud compute instances start sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut
# on the VM: sync changed backend/ + configs/ + scripts/ (tar the changed files only), then:
.venv/bin/python scripts/foundry_anthropic_smoke.py   # expect opus + sonnet OK
```

- [ ] **Step 3: Launch Phase-1 (~$30, autostop ON)**

```bash
# on the VM:
bash scripts/sdar_phase1_foundry.sh
# monitor: tail -f runs/phase1_run.out ; the VM self-stops on completion.
```

- [ ] **Step 4: Evaluate the PASS gate (from GCS after autostop)**

```bash
gsutil -m cp -r gs://deepinvent-ext-ut-sdar-runs/<PID> /tmp/phase1 && cd /tmp/phase1/<PID>
```
PASS ⇔ ALL of: harness-driven `val/success_rate` ≥ 0.40 (target 0.456); guards clean (zero-metrics / eval-provenance / env-liveness / no-learning, no veto); `code/metrics.json` has a real measured value + an `eval_provenance.json` (`provenance_kind:"aggregate"`); external-validator no-veto; `final_report.reproduction.mode=="execute"` AND `execution.ran==true`; `final_report.verdict` not `failed`.

- [ ] **Step 5: On PASS → checkpoint the operator for the $400 grid.** On MISS → debug on the ~$30 evidence (driver/seam/adapter), do NOT spend on the grid. Record the outcome in a runbook + update the memory `project_reliable_autonomous_reproduction`.

---

## Task 16: Document the foundation in CLAUDE.md (G7) — after Tasks 1-13 land, before Task 15

**Files:**
- Modify: `CLAUDE.md` (a concise rule block under the RLM-auth / feature-flags sections)
- Test: none (docs); a reviewer reads the diff.

**Interfaces:** none.

- [ ] **Step 1: Add the rule block** — three tight rules, the *rule* not the incident narrative (which lives in the spec):
  1. **Anthropic-Foundry provider** — `opus-foundry`(root)/`sonnet-foundry`(exec+grader+verifier) reach Claude Opus 4.8 / Sonnet 5 via `…/anthropic/v1`, `x-api-key=$AZURE_FOUNDRY_API_KEY` (same resource+key as the OpenAI-compat `azure-foundry`/grok endpoint). Scoped **per-client** (root `get_client` patch + `AnthropicMessagesClient` `base_url` + executor `ClaudeAgentOptions.env`); **`anthropic-foundry` and `claude-oauth` must never be co-resident** (asserted). Default-OFF/byte-identical when unselected. Code: `foundry_anthropic.py`, `_anthropic_foundry_patch.py`.
  2. **`OPENRESEARCH_LIFECYCLE_PRIMARY`** — the harness-owned deterministic reproduction backbone (`run_lifecycle_primary` bypasses the root loop, driving plan→implement→run→verify→repair); the reliable default for reproduction runs (run-spec toggle first; global default-ON flip gated on the ≥3-A/B+σ rule). Opt-out `=0`.
  3. **Execute-mode wiring rule** — `OPENRESEARCH_REPRODUCTION_MODE` is authoritative and set at SETUP in `repo_spec.json`; a run **fail-loud**s when `execute` is requested but `adapt` is stamped (`assert_execute_mode_stamped`).

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit -m "Document the Anthropic-Foundry provider, lifecycle-primary backbone, and execute-mode wiring rule in CLAUDE.md"
```

---

## Self-Review (checklist — completed by the plan author)

**Spec coverage:** WS-A → Tasks 1-7; WS-B → Tasks 8-10 (incl. SOTA E1 forced-action = driver-direct-calls + E2 pre-committed plan in Task 10); WS-C → Tasks 11-14; CLAUDE.md/G7 → Task 16; Phase-1 gate → Task 15. WS-D / WS-E(E3/E5/E6/E7) / WS-F are explicitly deferred to follow-on plans (spec §5 / rollout §8). Memory update = Task 15 Step 5.

**Placeholder scan:** no "TBD/TODO/handle edge cases" — every code step shows real code; the two operator-judgment spots (rlm `get_client` symbol name in Task 4; reconcile vs `phase1_autonomous.sh` in Task 14) are flagged with the exact file to check, not left vague.

**Type consistency:** `resolve_foundry_anthropic_credentials() -> (base_url, api_key, models)` used identically in Tasks 1/3/4/6; `AnthropicMessagesClient(base_url=…)` defined in Task 2, consumed in Task 3; `assert_no_foundry_oauth_coresidency` / `assert_execute_mode_stamped` / `_primary_inputs_ready` / `_synth_result_from_summary` names consistent across their tasks; backend literal `"anthropic-foundry"` consistent in models.py / role_models.py / grader_transport.py / run.py.
