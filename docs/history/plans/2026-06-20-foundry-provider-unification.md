# Foundry Provider Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task; every task is RED → GREEN → REFACTOR → COMMIT. Do not write implementation before its failing test exists and has been observed failing. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Audit (ground truth):** `docs/audits/2026-06-20-workstream-a-provider-unification-audit.md` (this branch) — the G1–G5 gap list with file:line.

**Goal:** Make a single env/CLI vocabulary (`azure-foundry` / `foundry` / `grok`) select an Azure AI Foundry OpenAI-compatible model for **any** of the five RLM LLM tiers (root, executor, verifier, grader, navigation accelerator) while keeping every existing claude/openai/oauth/openrouter/azure-openai path byte-for-byte identical when Foundry env/flags are unset.

**Architecture:** Four of the five tiers already reach Foundry: the **root** registry (`models.py`'s `azure-foundry` entry), the **executor/verifier/grader** sub-roles (`role_models.py` + `grader_transport.py`, both already tested). The single net-new capability is **G1: the navigation accelerator** (`accelerator.py`), whose mode vocabulary stops at `off/auto/local/runpod/azure/endpoint` and never reads `AZURE_FOUNDRY_*`. The remaining work is **extend/harden/de-dup**: add foundry to the legacy `OPENRESEARCH_EXECUTOR` flag (G2, graceful-fallback-preserving), let `selected_provider()` name `azure`/`azure-foundry` for generic sub-agents (G3), add the `AZURE_FOUNDRY_*` keys + missing `AZURE_OPENAI_*` keys to the shell-shadow validator (G4, warn-only), and route `models.py`'s duplicated Foundry endpoint normalization through the canonical `foundry_endpoint.resolve_foundry_credentials()` (G5, behavior-preserving). All Foundry tiers ride the OpenAI SDK (`OpenAILlmClient` / `OpenAiAgentRuntime` with a custom `base_url`); **none touch `ClaudeAgentOptions`**, so BUG-NEW-038 SDK-isolation is non-applicable by construction.

**Tech Stack:** Python 3.12 (CI/Docker; dev venv 3.14), FastAPI backend. `pytest` (config in `pyproject.toml`, `pythonpath=["."]`); socket-hermetic suite (`pytest-socket` blocks non-loopback) — **all tests monkeypatch/fake; zero network**. Canonical Foundry resolver: `backend/agents/runtime/foundry_endpoint.py::resolve_foundry_credentials() -> (base_url, deployment, api_key)`. Settings: `backend/config.py` exposes `azure_foundry_endpoint` / `azure_foundry_api_key` / `azure_foundry_deployment`.

---

## File Structure

| File | Created / Modified | Responsibility |
|---|---|---|
| `backend/agents/rlm/accelerator.py` | Modified | **G1** — add `_resolve_foundry()`; wire `"azure-foundry"`/`"foundry"`/`"grok"` modes into `resolve_accelerator`; add foundry to the `auto` chain; new `AcceleratorEndpoint(kind="foundry", is_azure=False)`. |
| `backend/agents/rlm/executor.py` | Modified | **G2** — add `_FOUNDRY_MODES` to `resolve_executor()` with graceful fallback (never fail-fast). |
| `backend/agents/runtime/factory.py` | Modified | **G3** — `selected_provider()` maps `azure`/`azure-foundry`/`grok`/`foundry` → `"openai"` (matching the existing `validate_provider_credentials` precedent). |
| `backend/cli.py` | Modified | **G4** — add `AZURE_FOUNDRY_API_KEY` / `AZURE_FOUNDRY_ENDPOINT` / `AZURE_FOUNDRY_DEPLOYMENT` + `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` to `_SUSPECT_KEYS`. |
| `backend/agents/rlm/models.py` | Modified | **G5** — delete local `_normalize_foundry_base_url`; `_inject_foundry_kwargs` reads base_url+deployment from `resolve_foundry_credentials()` (keeps its own fail-fast; discards the resolver's api_key). |
| `tests/rlm/test_accelerator_foundry.py` | Created | G1 tests. |
| `tests/rlm/test_executor_foundry.py` | Created | G2 tests. |
| `tests/agents/runtime/test_selected_provider_azure_foundry.py` | Created | G3 tests. |
| `tests/cli/test_env_override_warning.py` | Modified | G4 tests. |
| `tests/rlm/test_models_foundry_dedup.py` | Created | G5 tests. |
| `tests/rlm/test_foundry_five_tier_acceptance.py` | Created | Acceptance: one vocabulary selects foundry for all five tiers; unset == inherit. |
| `CLAUDE.md`, `system_overview.md` | Modified | Doc task: accelerator's new foundry mode + the new shadow-validator keys. |

> **Executor note:** the line numbers below are from the audit snapshot; verify with a quick `grep -n` before editing — match the **function/anchor**, not the literal line.

---

## Task 1 — G1: Foundry navigation accelerator

**Files:**
- `backend/agents/rlm/accelerator.py` (module docstring; `AcceleratorEndpoint.kind` doc; new `_resolve_foundry` after `_resolve_azure`; dispatch in `resolve_accelerator`; `_resolve_auto` chain; the trailing `ValueError` message)
- `tests/rlm/test_accelerator_foundry.py` (new)

- [ ] **Step 1: Write the failing test** — `tests/rlm/test_accelerator_foundry.py`:

```python
"""G1: Azure AI Foundry navigation-accelerator tier. No network — all creds
come from env (monkeypatched) and no probe is performed for Foundry."""
from __future__ import annotations

import types
import pytest

from backend.agents.rlm.accelerator import (
    AcceleratorEndpoint,
    AcceleratorError,
    build_accelerator_client,
    resolve_accelerator,
)


def _set_foundry(monkeypatch, endpoint="https://x.services.ai.azure.com",
                 deployment="grok-4.3", key="foundry-key"):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", deployment)
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", key)


def _clear_foundry(monkeypatch):
    for k in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENT", "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    # Neutralise Settings/.env — the real .env may carry Foundry creds.
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())


@pytest.mark.parametrize("mode", ["azure-foundry", "foundry", "grok"])
def test_explicit_foundry_returns_endpoint_no_probe(monkeypatch, mode):
    _set_foundry(monkeypatch)
    ep = resolve_accelerator(mode)
    assert isinstance(ep, AcceleratorEndpoint)
    assert ep.base_url == "https://x.services.ai.azure.com/openai/v1"
    assert ep.model == "grok-4.3"
    assert ep.api_key == "foundry-key"
    assert ep.kind == "foundry"
    assert ep.is_azure is False


def test_explicit_foundry_raises_on_missing_creds(monkeypatch):
    _clear_foundry(monkeypatch)
    with pytest.raises(AcceleratorError, match="AZURE_FOUNDRY"):
        resolve_accelerator("foundry")


def test_explicit_foundry_raises_on_missing_deployment(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "foundry-key")
    monkeypatch.delenv("AZURE_FOUNDRY_DEPLOYMENT", raising=False)
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())
    with pytest.raises(AcceleratorError, match="AZURE_FOUNDRY_DEPLOYMENT"):
        resolve_accelerator("foundry")


def test_auto_picks_foundry_when_creds_present_and_no_gpu(monkeypatch):
    monkeypatch.setattr(
        "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
        lambda: False,
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    _set_foundry(monkeypatch)
    ep = resolve_accelerator("auto")
    assert ep is not None
    assert ep.kind == "foundry"


def test_unset_foundry_off_is_unchanged(monkeypatch):
    _clear_foundry(monkeypatch)
    monkeypatch.setattr(
        "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
        lambda: False,
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    assert resolve_accelerator("off") is None
    assert resolve_accelerator("auto") is None


def test_build_client_for_foundry_uses_openai_client(monkeypatch):
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    ep = AcceleratorEndpoint(
        base_url="https://x.services.ai.azure.com/openai/v1",
        model="grok-4.3",
        api_key="foundry-key",
        kind="foundry",
        is_azure=False,
    )
    client = build_accelerator_client(ep)
    assert isinstance(client, OpenAILlmClient)
    assert hasattr(client, "complete")
```

> **Pre-impl verification:** confirm the real names with `grep -n "class AcceleratorEndpoint\|class AcceleratorError\|def resolve_accelerator\|def build_accelerator_client\|def _resolve_azure\|def _resolve_auto\|Unknown accelerator mode" backend/agents/rlm/accelerator.py`. If `AcceleratorEndpoint`'s constructor does not accept `kind`/`is_azure`, add those fields first (dataclass) — the test pins them.

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/rlm/test_accelerator_foundry.py -q`
  Expected: FAIL — `ValueError: Unknown accelerator mode 'azure-foundry'…` (no foundry branch) + `kind == "foundry"` assertion errors.

- [ ] **Step 3: Write minimal implementation** — in `accelerator.py`, add a sub-resolver after `_resolve_azure`:

```python
def _resolve_foundry(*, explicit: bool) -> AcceleratorEndpoint | None:
    """Resolve the Azure AI Foundry accelerator provider (OpenAI-compatible, e.g. Grok).

    Reads the canonical (base_url, deployment, api_key) triple from
    ``foundry_endpoint.resolve_foundry_credentials`` (env then Settings/.env).
    No probe — like ``_resolve_azure``, Foundry is a managed endpoint and
    credential presence is the gate. base_url is already normalised to
    ``…/openai/v1``; the deployment is the model id, so it must be non-empty.

    Raises :class:`AcceleratorError` in explicit mode when base_url+api_key are
    absent OR the deployment (model id) is missing; returns ``None`` in
    ``"auto"`` mode (graceful fallback).
    """
    from backend.agents.runtime.foundry_endpoint import resolve_foundry_credentials

    base_url, deployment, api_key = resolve_foundry_credentials()
    if not (base_url and api_key):
        if explicit:
            raise AcceleratorError(
                "Azure Foundry accelerator requires AZURE_FOUNDRY_ENDPOINT and "
                "AZURE_FOUNDRY_API_KEY to be set. Set these environment variables "
                "(and AZURE_FOUNDRY_DEPLOYMENT) and retry."
            )
        _log.info(
            "accelerator[foundry]: credentials not present (AZURE_FOUNDRY_ENDPOINT / "
            "AZURE_FOUNDRY_API_KEY); skipping in auto mode"
        )
        return None
    if not deployment:
        if explicit:
            raise AcceleratorError(
                "Azure Foundry accelerator requires AZURE_FOUNDRY_DEPLOYMENT (the "
                "deployed model name, e.g. grok-4.3) — it is the OpenAI-compatible "
                "model id and cannot be empty."
            )
        return None
    return AcceleratorEndpoint(
        base_url=base_url,
        model=deployment,
        api_key=api_key,
        kind="foundry",
        is_azure=False,
    )
```

  Add the dispatch branch in `resolve_accelerator` (after the `"azure"` branch, before `"endpoint"`):

```python
    if mode_lower in {"azure-foundry", "foundry", "grok"}:
        return _resolve_foundry(explicit=True)
```

  Update the trailing `ValueError` message to list the new modes:

```python
    raise ValueError(
        f"Unknown accelerator mode {mode!r}. "
        "Valid values: off, auto, local, runpod, azure, azure-foundry, endpoint."
    )
```

  Add foundry to the `_resolve_auto` chain — insert **after** the Azure-credentials block, before the final `return None`:

```python
    # --- Azure AI Foundry credentials (OpenAI-compatible, e.g. Grok) ---
    try:
        ep = _resolve_foundry(explicit=False)
        if ep is not None:
            _log.info("accelerator[auto]: selected foundry (credentials present)")
            return ep
    except Exception as exc:  # noqa: BLE001
        _log.debug("accelerator[auto]: foundry check failed: %s", exc)
```

  Update the module docstring (add the `"azure-foundry"` bullet) and the `AcceleratorEndpoint.kind` doc to add `"foundry"`.

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/rlm/test_accelerator_foundry.py tests/rlm/test_accelerator.py -q`
  Expected: PASS (new foundry tests pass; existing accelerator tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/accelerator.py tests/rlm/test_accelerator_foundry.py
git commit -m "feat(accelerator): add Azure Foundry navigation tier (G1)

Adds _resolve_foundry + azure-foundry/foundry/grok modes and an auto-chain
pickup; reads the canonical resolve_foundry_credentials triple, no probe
(managed endpoint), is_azure=False so it rides OpenAILlmClient. Default-OFF:
unset foundry env leaves off/auto byte-for-byte identical.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — G2: Legacy `OPENRESEARCH_EXECUTOR` foundry support (graceful fallback)

**Files:**
- `backend/agents/rlm/executor.py` (docstring; mode sets; `resolve_executor` azure block as the pattern; insert a foundry block after it)
- `tests/rlm/test_executor_foundry.py` (new)

- [ ] **Step 1: Write the failing test** — `tests/rlm/test_executor_foundry.py`:

```python
"""G2: OPENRESEARCH_EXECUTOR=azure-foundry/grok routes implement_baseline onto a
Foundry deployment; missing creds GRACEFULLY fall back to None (NOT fail-fast).
No network — the foundry runtime is faked."""
from __future__ import annotations

import types
import pytest

from backend.agents.rlm.executor import ExecutorPlan, resolve_executor


def _clear_foundry(monkeypatch):
    for k in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENT", "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())


@pytest.mark.parametrize("mode", ["azure-foundry", "foundry", "grok"])
def test_foundry_mode_returns_plan(monkeypatch, mode):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", mode)
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", "grok-4.3")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "foundry-key")

    import backend.agents.runtime.azure_foundry_runtime as afr

    class _FakeRT:
        def __init__(self):
            self.built = True

    monkeypatch.setattr(afr, "AzureFoundryAgentRuntime", _FakeRT)

    plan = resolve_executor()
    assert isinstance(plan, ExecutorPlan)
    assert plan.model == "grok-4.3"
    assert plan.label == "azure-foundry:grok-4.3"
    assert isinstance(plan.runtime, _FakeRT)


def test_foundry_mode_missing_creds_falls_back_gracefully(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "grok")
    _clear_foundry(monkeypatch)
    assert resolve_executor() is None  # default Sonnet executor


def test_executor_unset_is_unchanged(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTOR", raising=False)
    assert resolve_executor() is None
```

> **Pre-impl verification:** `grep -n "class ExecutorPlan\|def resolve_executor\|_AZURE_MODES\|_VLLM_MODES\|OPENRESEARCH_EXECUTOR" backend/agents/rlm/executor.py` — confirm `ExecutorPlan` fields (`runtime`, `model`, `label`) and the azure-block shape you mirror.

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/rlm/test_executor_foundry.py -q`
  Expected: FAIL — `grok`/`azure-foundry`/`foundry` hit the unknown-mode warn branch and return `None`, so `test_foundry_mode_returns_plan` fails.

- [ ] **Step 3: Write minimal implementation** — in `executor.py`, add the mode set:

```python
_FOUNDRY_MODES = {"azure-foundry", "foundry", "grok", "grok-4.3", "azure-foundry-openai"}
```

  Add a foundry block in `resolve_executor` **after** the `_AZURE_MODES` block, before the vLLM check:

```python
    if mode in _FOUNDRY_MODES:
        # Foundry (OpenAI-compatible custom endpoint, e.g. Grok). Graceful
        # fallback — a missing/incomplete cred set must NEVER fail-fast here
        # (preserves the legacy OPENRESEARCH_EXECUTOR contract); the run keeps
        # the default Sonnet executor instead.
        from backend.agents.runtime.foundry_endpoint import resolve_foundry_credentials

        base_url, deployment, api_key = resolve_foundry_credentials()
        if not (base_url and api_key and deployment):
            logger.warning(
                "OPENRESEARCH_EXECUTOR=%r but AZURE_FOUNDRY_ENDPOINT / "
                "AZURE_FOUNDRY_DEPLOYMENT / AZURE_FOUNDRY_API_KEY incomplete"
                " — falling back to the default Sonnet executor",
                mode,
            )
            return None
        from backend.agents.runtime.azure_foundry_runtime import AzureFoundryAgentRuntime

        runtime = AzureFoundryAgentRuntime()
        logger.info("executor tier active: azure-foundry → deployment=%s", deployment)
        return ExecutorPlan(runtime=runtime, model=deployment, label=f"azure-foundry:{deployment}")
```

  Update the docstring to add the foundry mode row.

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/rlm/test_executor_foundry.py -q`

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/executor.py tests/rlm/test_executor_foundry.py
git commit -m "feat(executor): add azure-foundry/grok to legacy EXECUTOR flag (G2)

Routes implement_baseline onto AzureFoundryAgentRuntime when
OPENRESEARCH_EXECUTOR=azure-foundry/foundry/grok; preserves the flag's
graceful-fallback contract — incomplete Foundry creds degrade to the default
Sonnet executor, never fail-fast. Unset == unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — G3: `selected_provider()` accepts azure / azure-foundry

**Files:**
- `backend/agents/runtime/factory.py` (`selected_provider`)
- `tests/agents/runtime/test_selected_provider_azure_foundry.py` (new)

> `ProviderName = Literal["anthropic", "openai"]` (`base.py`). `make_runtime` already branches on azure/foundry **before** calling `selected_provider`, and `has_provider_credentials` handles azure separately. So azure/azure-foundry/grok/foundry → `"openai"` (the **existing precedent**: `validate_provider_credentials` already returns `"openai"` for azure). **No Literal widening.**

- [ ] **Step 1: Write the failing test** — `tests/agents/runtime/test_selected_provider_azure_foundry.py`:

```python
"""G3: selected_provider() lets generic sub-agents NAME azure / azure-foundry.
Both resolve to the "openai" ProviderName literal (Foundry rides the OpenAI
SDK), matching the validate_provider_credentials precedent. Existing
anthropic/openai resolution is unchanged."""
from __future__ import annotations

import pytest

from backend.agents.runtime.base import ProviderConfigurationError
from backend.agents.runtime.factory import selected_provider


@pytest.mark.parametrize("name", ["azure", "azure-openai", "azure_openai"])
def test_azure_resolves_to_openai(name):
    assert selected_provider(name) == "openai"


@pytest.mark.parametrize("name", ["azure-foundry", "foundry", "grok", "grok-4.3"])
def test_azure_foundry_resolves_to_openai(name):
    assert selected_provider(name) == "openai"


def test_anthropic_and_openai_unchanged():
    assert selected_provider("anthropic") == "anthropic"
    assert selected_provider("claude") == "anthropic"
    assert selected_provider("openai") == "openai"
    assert selected_provider("oai") == "openai"


def test_unknown_provider_still_raises():
    with pytest.raises(ProviderConfigurationError):
        selected_provider("bananas")


def test_unset_defaults_unchanged(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda *a, **k: type("S", (), {"llm_provider": ""})(),
    )
    assert selected_provider() == "anthropic"
```

> **Pre-impl verification:** `grep -n "def selected_provider\|def validate_provider_credentials\|ProviderConfigurationError\|normalized" backend/agents/runtime/factory.py` — confirm the normalize variable name + the alias set the anthropic/openai branches use (e.g. `claude`/`oai`).

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/agents/runtime/test_selected_provider_azure_foundry.py -q`
  Expected: FAIL — `selected_provider("azure")` raises `ProviderConfigurationError`.

- [ ] **Step 3: Write minimal implementation** — in `selected_provider` (after the `openai`/`oai` branch):

```python
    # Azure OpenAI and Azure AI Foundry are not ProviderName literals; both ride
    # the OpenAI SDK surface, so generic sub-agents may name them here and get
    # the "openai" literal back (matching validate_provider_credentials, which
    # returns "openai" for azure as the "closest ProviderName"). The dedicated
    # Azure/Foundry runtimes are selected upstream in make_runtime() before this
    # call; this branch only lets a caller that passed the name through resolve.
    if normalized in {
        "azure", "azure-openai", "azure_openai",
        "azure-foundry", "foundry", "grok", "grok-4.3", "azure-foundry-openai",
    }:
        return "openai"
```

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/agents/runtime/test_selected_provider_azure_foundry.py tests/agents/runtime/ -q`
  Expected: PASS; existing factory tests (esp. `make_runtime` azure/foundry branches) unchanged.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/runtime/factory.py tests/agents/runtime/test_selected_provider_azure_foundry.py
git commit -m "feat(factory): selected_provider accepts azure/azure-foundry (G3)

Generic sub-agents can now name azure / azure-foundry / grok; all map to the
\"openai\" ProviderName literal (Foundry rides the OpenAI SDK), matching the
existing validate_provider_credentials precedent — no Literal widening. The
dedicated runtimes are still selected upstream in make_runtime. anthropic/openai
resolution + unknown-provider raise unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — G4: Shell-shadow validator covers `AZURE_FOUNDRY_*` + `AZURE_OPENAI_*`

**Files:**
- `backend/cli.py` (`_SUSPECT_KEYS`)
- `tests/cli/test_env_override_warning.py` (extend; mirror the existing `_run` helper)

- [ ] **Step 1: Write the failing test** — append to `tests/cli/test_env_override_warning.py`:

```python
# ---------------------------------------------------------------------------
# G4: Azure / Azure Foundry keys are suspect too — a stale shell value silently
# shadowing .env causes 401s exactly like the OpenAI/Anthropic keys.
# ---------------------------------------------------------------------------
import pytest


@pytest.mark.parametrize(
    "key",
    [
        "AZURE_FOUNDRY_API_KEY",
        "AZURE_FOUNDRY_ENDPOINT",
        "AZURE_FOUNDRY_DEPLOYMENT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
    ],
)
def test_warns_when_azure_foundry_keys_differ(key) -> None:
    out = _run(key, "shell-value-AAAAAAAAAAAA", "dotenv-value-BBBBBBBBBBBB")
    assert key in out
    assert "warn" in out.lower()
    assert f"env -u {key}" in out


@pytest.mark.parametrize("key", ["AZURE_FOUNDRY_API_KEY", "AZURE_OPENAI_ENDPOINT"])
def test_no_warning_when_azure_keys_match(key) -> None:
    same = "same-value-CCCCCCCCCCCCCC"
    out = _run(key, same, same)
    assert "warn" not in out.lower()
    assert key not in out
```

> **Pre-impl verification:** open `tests/cli/test_env_override_warning.py` and confirm the existing `_run(key, shell_value, dotenv_value)` helper signature + the exact warning text (`env -u <KEY>`); adapt the asserts to the real output shape.

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/cli/test_env_override_warning.py -q`
  Expected: FAIL — the foundry/azure keys are not in `_SUSPECT_KEYS`, so no warning is printed.

- [ ] **Step 3: Write minimal implementation** — extend `_SUSPECT_KEYS` in `cli.py`:

```python
    _SUSPECT_KEYS = (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "FEATHERLESS_API_KEY", "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENT",
        "OPENRESEARCH_RUNPOD_API_KEY",
    )
```

(Preserve whatever keys are already present — this is a superset; do not drop any.)

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/cli/test_env_override_warning.py -q`

- [ ] **Step 5: Commit**

```bash
git add backend/cli.py tests/cli/test_env_override_warning.py
git commit -m "feat(cli): shadow-validator covers AZURE_FOUNDRY_* + AZURE_OPENAI endpoint/deployment (G4)

A stale shell AZURE_FOUNDRY_API_KEY/ENDPOINT/DEPLOYMENT (or AZURE_OPENAI_ENDPOINT/
DEPLOYMENT) silently shadows .env and 401s a foundry run; add them to
_SUSPECT_KEYS. Warn-only, fires only when shell != .env.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — G5: De-dup `models.py` Foundry normalization through the canonical resolver

**Files:**
- `backend/agents/rlm/models.py` (`_normalize_foundry_base_url` → delete; `_inject_foundry_kwargs` → rewrite; keep `_env_or_settings` if still used by `_model_missing_credentials`)
- `tests/rlm/test_models_foundry_dedup.py` (new)

> **Behavior-preserving guardrails:** (1) `_inject_foundry_kwargs` MUST keep its own fail-fast `ValueError` on missing endpoint/deployment — `resolve_foundry_credentials()` returns `("","","")` without raising. (2) It must NOT inject `api_key` — that's already done upstream by `_inject_api_key(AZURE_FOUNDRY_API_KEY)`; **discard the resolver's api_key return**. (3) The resolved `backend_kwargs`/`sub_backend_kwargs` for a configured foundry run must be **byte-identical** before/after.

- [ ] **Step 1: Write the failing test** — `tests/rlm/test_models_foundry_dedup.py`:

```python
"""G5: models.py routes Foundry endpoint normalization through the SINGLE
canonical foundry_endpoint.resolve_foundry_credentials — no local re-impl.
Behaviour-preserving: kwargs byte-identical, fail-fast on missing
endpoint/deployment kept, api_key injected ONCE (upstream)."""
from __future__ import annotations

import pytest

import backend.agents.rlm.models as models
from backend.agents.rlm.models import resolve_root_model


def _set_foundry(monkeypatch,
                 endpoint="https://x.services.ai.azure.com/openai/v1/chat/completions",
                 deployment="grok-4.3", key="foundry-key"):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", deployment)
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", key)


def test_resolve_foundry_root_kwargs_exact(monkeypatch):
    _set_foundry(monkeypatch)
    entry = resolve_root_model("azure-foundry")
    assert entry.backend_kwargs == {
        "base_url": "https://x.services.ai.azure.com/openai/v1",
        "model_name": "grok-4.3",
        "api_key": "foundry-key",
    }
    assert entry.sub_backend_kwargs == {
        "base_url": "https://x.services.ai.azure.com/openai/v1",
        "model_name": "grok-4.3",
        "api_key": "foundry-key",
    }


def test_inject_foundry_kwargs_routes_through_canonical_resolver(monkeypatch):
    called = {"n": 0}

    def _fake_resolve():
        called["n"] += 1
        return ("https://canon.services.ai.azure.com/openai/v1", "canon-dep", "canon-key")

    monkeypatch.setattr(
        "backend.agents.runtime.foundry_endpoint.resolve_foundry_credentials",
        _fake_resolve,
    )
    out = models._inject_foundry_kwargs({"api_key": "preinjected"}, model_key="azure-foundry")
    assert called["n"] >= 1
    assert out["base_url"] == "https://canon.services.ai.azure.com/openai/v1"
    assert out["model_name"] == "canon-dep"
    assert out["api_key"] == "preinjected"  # NOT overwritten by resolver's key


def test_inject_foundry_fail_fast_on_missing(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.runtime.foundry_endpoint.resolve_foundry_credentials",
        lambda: ("", "", ""),
    )
    with pytest.raises(ValueError, match="AZURE_FOUNDRY"):
        models._inject_foundry_kwargs({}, model_key="azure-foundry")


def test_local_normalize_helper_removed():
    assert not hasattr(models, "_normalize_foundry_base_url")
```

> **Pre-impl verification:** `grep -n "_normalize_foundry_base_url\|_inject_foundry_kwargs\|_env_or_settings\|_inject_api_key\|_model_missing_credentials" backend/agents/rlm/models.py` — confirm the current `_inject_foundry_kwargs` signature (`(kwargs, *, model_key)`) and that the api_key is injected separately upstream. Adjust the test's call signature to match reality if it differs.

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/rlm/test_models_foundry_dedup.py -q`
  Expected: FAIL — current impl reads env directly (never calls `resolve_foundry_credentials`) and `_normalize_foundry_base_url` still present.

- [ ] **Step 3: Write minimal implementation** — in `models.py`, delete `_normalize_foundry_base_url` and rewrite `_inject_foundry_kwargs`:

```python
def _inject_foundry_kwargs(kwargs: dict, *, model_key: str) -> dict:
    """Return a copy of *kwargs* with the env-driven Foundry base_url + model_name.

    Single source of truth: delegates normalization + env/Settings resolution to
    the canonical ``foundry_endpoint.resolve_foundry_credentials`` (no local
    re-impl). Keeps its OWN fail-fast — the resolver returns ``("","","")``
    without raising, but a missing base_url/deployment is fatal here. The
    resolver's api_key is DISCARDED: the key is injected once, upstream, via
    ``_inject_api_key(AZURE_FOUNDRY_API_KEY)``.
    """
    from backend.agents.runtime.foundry_endpoint import resolve_foundry_credentials

    out = dict(kwargs)
    base_url, deployment, _api_key = resolve_foundry_credentials()
    missing = [
        name
        for name, val in (
            ("AZURE_FOUNDRY_ENDPOINT", base_url),
            ("AZURE_FOUNDRY_DEPLOYMENT", deployment),
        )
        if not val
    ]
    if missing:
        raise ValueError(
            f"Root model {model_key!r} uses the Azure Foundry endpoint but "
            f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set. "
            "Set AZURE_FOUNDRY_ENDPOINT (e.g. "
            "https://<resource>.services.ai.azure.com/openai/v1) and "
            "AZURE_FOUNDRY_DEPLOYMENT (the deployed model name, e.g. grok-4.3)."
        )
    out["base_url"] = base_url
    out["model_name"] = deployment
    return out
```

  Keep `_env_or_settings` if `_model_missing_credentials` still references it.

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/rlm/test_models_foundry_dedup.py "tests/rlm" -k "model or registry or foundry" -q`

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/models.py tests/rlm/test_models_foundry_dedup.py
git commit -m "refactor(models): route Foundry endpoint normalization through canonical resolver (G5)

Deletes the duplicated _normalize_foundry_base_url; _inject_foundry_kwargs now
delegates to foundry_endpoint.resolve_foundry_credentials (single source of
truth). Behaviour-preserving: keeps its own fail-fast on missing endpoint/
deployment and discards the resolver's api_key (injected once upstream).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6 — Acceptance: one vocabulary selects Foundry for all five tiers; unset == inherit

**Files:**
- `tests/rlm/test_foundry_five_tier_acceptance.py` (new)

- [ ] **Step 1: Write the test** — `tests/rlm/test_foundry_five_tier_acceptance.py`:

```python
"""ACCEPTANCE: a single Foundry vocabulary (azure-foundry/foundry/grok) selects
a Foundry model for ANY of the five tiers; with the vocabulary UNSET, every tier
is byte-for-byte today's path. No network — runtimes/clients faked or only
descriptor-resolved."""
from __future__ import annotations

import types
import pytest


@pytest.fixture()
def foundry_env(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", "grok-4.3")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "foundry-key")
    yield


@pytest.fixture()
def no_foundry_env(monkeypatch):
    for k in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENT", "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())
    yield


def test_tier_root_foundry(foundry_env):
    from backend.agents.rlm.models import resolve_root_model
    entry = resolve_root_model("grok")
    assert entry.key == "azure-foundry"
    assert entry.backend_kwargs["model_name"] == "grok-4.3"


def test_tier_executor_foundry(foundry_env, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "grok")
    import backend.agents.runtime.azure_foundry_runtime as afr
    monkeypatch.setattr(afr, "AzureFoundryAgentRuntime", lambda: object())
    from backend.agents.rlm.executor import resolve_executor
    plan = resolve_executor()
    assert plan is not None
    assert plan.label == "azure-foundry:grok-4.3"


@pytest.mark.parametrize("role", ["verifier", "grader"])
def test_tier_verifier_grader_foundry(foundry_env, monkeypatch, role):
    import backend.services.context.workspace.tools.openai_client as oac

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(oac, "OpenAILlmClient", _FakeOpenAI)
    from backend.agents.rlm.grader_transport import build_transport_client
    client, label = build_transport_client(
        backend="grok", model=None,
        fallback_client=object(), fallback_label="fb", role_label=role,
    )
    assert isinstance(client, _FakeOpenAI)
    assert label == f"{role}:azure-foundry:grok-4.3"


def test_tier_accelerator_foundry(foundry_env):
    from backend.agents.rlm.accelerator import resolve_accelerator
    ep = resolve_accelerator("foundry")
    assert ep is not None
    assert ep.kind == "foundry"
    assert ep.model == "grok-4.3"


def test_all_tiers_unset_inherit(no_foundry_env, monkeypatch):
    from backend.agents.rlm.executor import resolve_executor
    from backend.agents.rlm.accelerator import resolve_accelerator
    from backend.agents.rlm.grader_transport import build_grader_client

    monkeypatch.delenv("OPENRESEARCH_EXECUTOR", raising=False)
    assert resolve_executor() is None
    assert resolve_accelerator("off") is None

    monkeypatch.delenv("OPENRESEARCH_GRADER_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_MODEL", raising=False)
    sentinel = object()
    client, label = build_grader_client(sentinel, "root-label")
    assert client is sentinel and label == "root-label"
```

> **Pre-impl verification:** confirm the real signatures of `build_transport_client(...)` and `build_grader_client(...)` in `backend/agents/rlm/grader_transport.py` (param names/order) and adjust the calls if they differ.

- [ ] **Step 2: Run test to verify it passes** (after Tasks 1–5) — `.venv/bin/python -m pytest tests/rlm/test_foundry_five_tier_acceptance.py -q`

- [ ] **Step 3: Run the full touched-module sweep** —
  `.venv/bin/python -m pytest tests/rlm/ tests/agents/runtime/ tests/cli/ tests/agents/rlm/test_role_models.py -q`
  Expected: all green; `tests/rlm/test_accelerator.py`, `tests/rlm/test_grader_transport.py`, `tests/agents/rlm/test_role_models.py` unchanged.

- [ ] **Step 4: Commit**

```bash
git add tests/rlm/test_foundry_five_tier_acceptance.py
git commit -m "test(foundry): acceptance — one vocabulary selects Foundry across all five tiers

azure-foundry/foundry/grok selects Foundry for root/executor/verifier/grader/
accelerator; unset == inherit today's path for every tier. Zero network.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7 — Documentation (repo doc policy)

**Files:** `CLAUDE.md`, `system_overview.md`

- [ ] **Step 1: Find the anchors** — `grep -n "off/auto/local/runpod/azure/endpoint\|_warn_on_shell_env_override\|suspect keys\|azure-foundry\|ACCELERATOR" CLAUDE.md`
- [ ] **Step 2: Edit CLAUDE.md** — update the accelerator mode vocabulary to include `azure-foundry` (aliases `foundry`/`grok`), the shell-shadow suspect-key list/count (now includes `AZURE_FOUNDRY_*` + `AZURE_OPENAI_ENDPOINT/DEPLOYMENT`), and the Azure AI Foundry section (selected_provider azure/foundry → openai; legacy EXECUTOR foundry mode; models.py de-dup).
- [ ] **Step 3: Edit system_overview.md** — mirror the accelerator-provider-list + shadow-validator coverage ("the navigation accelerator was the last tier missing Foundry; one canonical resolver now feeds all five tiers").
- [ ] **Step 4: Doc-check** — `grep -rn "docs-check\|doc-check" Makefile scripts/ 2>/dev/null` then run it if present.
- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md system_overview.md
git commit -m "docs: Foundry across all five tiers + shadow-validator key coverage

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review — spec coverage

### Gap coverage (G1–G5)

| Gap | Requirement | Task |
|---|---|---|
| **G1** | Foundry path in the navigation accelerator | **Task 1** |
| **G2** | Legacy `OPENRESEARCH_EXECUTOR` accepts azure-foundry/grok, graceful fallback preserved | **Task 2** |
| **G3** | `selected_provider()` accepts azure / azure-foundry (→ `"openai"`, no Literal widening) | **Task 3** |
| **G4** | Shadow validator adds `AZURE_FOUNDRY_*` + `AZURE_OPENAI_ENDPOINT/DEPLOYMENT`, warn-only | **Task 4** |
| **G5** | `models.py` de-dup → canonical `resolve_foundry_credentials` | **Task 5** |

### Hard-constraint / acceptance coverage

| Constraint | Where enforced |
|---|---|
| Default-OFF + fail-soft; unset == byte-for-byte | `test_unset_foundry_off_is_unchanged` (T1), `test_executor_unset_is_unchanged` (T2), `test_unset_defaults_unchanged` (T3), `test_no_warning_when_azure_keys_match` (T4), `test_resolve_foundry_root_kwargs_exact` byte-identical (T5), `test_all_tiers_unset_inherit` (T6) |
| SDK isolation (BUG-NEW-038) | Non-applicable: all Foundry tiers ride the OpenAI SDK, never `ClaudeAgentOptions`. No `ClaudeAgentOptions` path is touched. |
| No live API; zero network | Every test monkeypatches the runtime/client or only resolves descriptors; no probe for Foundry; suite is socket-hermetic. |
| paper_validated gating unchanged; Foundry/grok → `role_model_fidelity` advisory | No task touches `paper_validated` (the `azure-foundry` entry keeps `paper_validated=False`) nor `role_models.py` fidelity logic; existing `test_role_models.py` foundry coverage stays green (T6 sweep). |
| Acceptance: one flag selects Foundry for ANY of 5 tiers; existing paths unchanged when unset | **Task 6** |
| Docs updated | **Task 7** |

### Notes for the executing agent
- Only **G1 is net-new**; G2–G5 are extend/harden/de-dup. Tasks 1–5 are independent and TDD-ordered.
- G3 return value (`"openai"`) is grounded: `make_runtime` branches azure/foundry **before** `selected_provider`; `validate_provider_credentials` already returns `"openai"` for azure.
- G5 invariants: keep the local fail-fast (resolver returns empties, never raises); discard the resolver's `api_key` (injected once upstream); assert byte-identical kwargs.
- Each task's first action is the `grep -n` **Pre-impl verification** — the audit's line numbers are a snapshot; bind to the function/anchor, not the literal line.
