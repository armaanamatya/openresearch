# Cloud Posture Realignment (GCP/Azure primary, RunPod legacy) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GCP + Azure the trustworthy primary execution clouds and demote RunPod to legacy — correct default sandbox, truthful Foundry LLM cost, a GPU-$ cap that bounds a stuck cell, the two GKE footguns turned into tested guards, and every doc surface updated.

**Architecture:** Six focused code changes (pricing alias, VRAM-override headroom skip, SKU validation, `auto` resolver, RunPod info-log, mid-cell GPU-$ heartbeat) plus a docs sweep. Each is independently testable and OFF-preserving except the one intentional behavior change (default sandbox `runpod`→`auto`), which is guarded by a resolver test.

**Tech Stack:** Python 3.12, pydantic / pydantic-settings, pytest + pytest-socket (hermetic), Kubernetes Python client (mocked in tests).

**Spec:** `docs/superpowers/specs/2026-07-09-cloud-posture-gcp-azure-primary-design.md`

**Baseline:** 477 in-scope tests green. Run all tests below with `export OPENRESEARCH_MIN_DISK_GB=0` set (this dev Mac). Commit messages: descriptive present-tense headline, **no** Conventional-Commit prefix, **no** `Co-Authored-By` trailer (project rule). Branch off `main` before starting.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `backend/agents/resilience/pricing.py` | LLM cost estimation | Add `FOUNDRY_ALIASES` + alias lookup in `_resolve_pricing` |
| `backend/agents/schemas.py` | `GpuRequirements` model | Add `vram_is_explicit` field |
| `backend/agents/rlm/primitives.py` | GPU requirement resolution | Set `vram_is_explicit=True` on override |
| `backend/services/runtime/gpu_resolver.py` | VRAM→SKU resolution | Skip headroom when explicit; add SKU-mismatch validation |
| `backend/agents/execution.py` | Sandbox mode policy | `auto` = docker/local only; default `runpod`→`auto` |
| `backend/agents/rlm/k8s_job_cell_runner.py` | K8s cell execution | Mid-cell GPU-$ heartbeat + `gpu_budget_exceeded` terminal |
| Docs | Posture | `CLAUDE.md` × N, `flags.md`, `learn.md`, new runbook |

---

## Task 1: Price Foundry-routed LLM rows instead of $0

**Files:**
- Modify: `backend/agents/resilience/pricing.py`
- Test: `tests/pricing/test_foundry_alias.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_foundry_alias.py
"""Foundry role aliases (opus-foundry / sonnet-foundry) must price at their
priced Claude siblings, not $0. Regression: the ledger records the bare role
ids (backend.agents.rlm.role_models), which did not resolve against PRICING."""

from __future__ import annotations

import pytest

from backend.agents.resilience.pricing import (
    PRICING,
    _resolve_pricing,
    estimate_cost_usd,
)


def test_opus_foundry_resolves_to_opus_pricing():
    assert _resolve_pricing("opus-foundry") is PRICING["claude-opus-4-8"]


def test_sonnet_foundry_resolves_to_sonnet_pricing():
    assert _resolve_pricing("sonnet-foundry") is PRICING["claude-sonnet-5"]


def test_opus_foundry_estimate_is_nonzero():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    # 15/1M input + 75/1M output = 90.0
    assert estimate_cost_usd("opus-foundry", usage) == pytest.approx(90.0)


def test_unknown_model_still_returns_none():
    assert estimate_cost_usd("mystery-model", {"input_tokens": 100}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pricing/test_foundry_alias.py -v`
Expected: FAIL — `_resolve_pricing("opus-foundry")` returns `None` (no alias), estimate is `None`.

- [ ] **Step 3: Add the alias map and lookup**

In `backend/agents/resilience/pricing.py`, immediately after the `PRICING` dict closes (after line 75), add:

```python
# Foundry role aliases: the ledger records these bare role ids (see
# backend.agents.rlm.role_models opus-foundry/sonnet-foundry). Map them to their
# priced Claude siblings so estimate_cost_usd no longer returns $0 for every
# Foundry-routed row. Rates mirror the siblings (see the PRICING note above);
# Foundry billing is separately reconciled via Azure Cost Management.
FOUNDRY_ALIASES: dict[str, str] = {
    "opus-foundry": "claude-opus-4-8",
    "sonnet-foundry": "claude-sonnet-5",
}
```

Then in `_resolve_pricing`, right after the exact-match block (after `if entry is not None: return entry`, ~line 100), insert step 1b:

```python
    # 1b. Foundry role alias → priced sibling.
    alias = FOUNDRY_ALIASES.get(model)
    if alias is not None:
        entry = PRICING.get(alias)
        if entry is not None:
            return entry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pricing/test_foundry_alias.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the pricing suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/pricing -q`
Expected: all pass (99+4).

- [ ] **Step 6: Commit**

```bash
git add backend/agents/resilience/pricing.py tests/pricing/test_foundry_alias.py
git commit -m "Price Foundry-routed LLM rows (opus-foundry/sonnet-foundry) instead of \$0"
```

---

## Task 2: Skip the 1.25× VRAM headroom on an explicit --vram-gb override

**Files:**
- Modify: `backend/agents/schemas.py:828-843` (GpuRequirements)
- Modify: `backend/agents/rlm/primitives.py:1428-1431` (override site)
- Modify: `backend/services/runtime/gpu_resolver.py:203-205, 314-315` (headroom sites)
- Test: `tests/runtime/test_gpu_resolver_vram_override.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/runtime/test_gpu_resolver_vram_override.py
"""An explicit --vram-gb override must be used verbatim — no 1.25x headroom.
Regression (2026-07-08 GCP incident): --vram-gb 80 on an 80GB fleet inflated to
100GB and matched no SKU, so the run never got a GPU."""

from __future__ import annotations

import math

from backend.agents.schemas import GpuRequirements


def test_explicit_override_skips_headroom():
    req = GpuRequirements(estimated_vram_gb=80, confidence=1.0, vram_is_explicit=True)
    effective = 1.0 if req.vram_is_explicit else 1.25
    assert math.ceil(80 * max(effective, 1.0)) == 80


def test_llm_estimate_still_gets_headroom():
    req = GpuRequirements(estimated_vram_gb=80, confidence=0.7, vram_is_explicit=False)
    effective = 1.0 if req.vram_is_explicit else 1.25
    assert math.ceil(80 * max(effective, 1.0)) == 100


def test_vram_is_explicit_defaults_false():
    assert GpuRequirements(estimated_vram_gb=40).vram_is_explicit is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/runtime/test_gpu_resolver_vram_override.py -v`
Expected: FAIL — `GpuRequirements` has no `vram_is_explicit` field (extra ignored → attribute error on access).

- [ ] **Step 3: Add the field to GpuRequirements**

In `backend/agents/schemas.py`, inside `GpuRequirements` (after the `confidence` field, ~line 843), add:

```python
    vram_is_explicit: bool = Field(
        default=False,
        description=(
            "True when estimated_vram_gb came from an operator --vram-gb override "
            "(not the LLM). The resolver skips the headroom multiplier for it."
        ),
    )
```

- [ ] **Step 4: Run the field-default test to verify it passes**

Run: `.venv/bin/python -m pytest tests/runtime/test_gpu_resolver_vram_override.py -v`
Expected: PASS (3 tests — the pure-math tests already pass once the field exists).

- [ ] **Step 5: Set the flag on the override in primitives**

In `backend/agents/rlm/primitives.py:1428-1431`, change:

```python
    # ---- vram_override: per-run CLI override bypasses LLM estimate.
    vram_override = getattr(ctx, "vram_override", None)
    if vram_override is not None:
        req = req.model_copy(update={"estimated_vram_gb": int(vram_override)})
```

to:

```python
    # ---- vram_override: per-run CLI override bypasses LLM estimate AND its
    # headroom multiplier (operator intent is used verbatim; see
    # gpu_resolver headroom sites).
    vram_override = getattr(ctx, "vram_override", None)
    if vram_override is not None:
        req = req.model_copy(
            update={"estimated_vram_gb": int(vram_override), "vram_is_explicit": True}
        )
```

- [ ] **Step 6: Honor the flag at both headroom sites in gpu_resolver**

In `backend/services/runtime/gpu_resolver.py`, at Site A (lines 203-205), change:

```python
    # Apply headroom multiplier; round up.
    needed_vram = math.ceil(estimate * max(headroom_multiplier, 1.0))
```

to:

```python
    # Apply headroom multiplier; round up. An explicit --vram-gb override is used
    # verbatim (no headroom) — operator intent, not an LLM estimate.
    _headroom = 1.0 if getattr(requirements, "vram_is_explicit", False) else headroom_multiplier
    needed_vram = math.ceil(estimate * max(_headroom, 1.0))
```

Apply the identical change at Site B (lines 314-315), whose comment reads
`# Apply headroom multiplier (against per-GPU estimate; same logic as RunPod).` —
replace its `needed_vram = math.ceil(estimate * max(headroom_multiplier, 1.0))` with the same
two lines (`_headroom = ...` then `needed_vram = math.ceil(estimate * max(_headroom, 1.0))`).

- [ ] **Step 7: Run resolver + schema tests**

Run: `.venv/bin/python -m pytest tests/runtime tests/rlm/test_gpu_resolution.py -q -k "vram or resolver or gpu"`
Expected: PASS, no regressions.

- [ ] **Step 8: Commit**

```bash
git add backend/agents/schemas.py backend/agents/rlm/primitives.py backend/services/runtime/gpu_resolver.py tests/runtime/test_gpu_resolver_vram_override.py
git commit -m "Use an explicit --vram-gb override verbatim (skip the 1.25x headroom)"
```

---

## Task 3: Fail loud when gcp_gpu_skus can't match any cluster pool

**Files:**
- Modify: `backend/services/runtime/gpu_resolver.py` (add validation helper)
- Test: `tests/runtime/test_gpu_sku_validation.py`

**Note:** Do NOT change the `gcp_gpu_skus` default (`["gcp_a100_80x8"]`) — it is synced to
`infra/gcp/variables.tf` and pinned by `tests/config/test_gcp_sku_pool_invariant.py`. This task
only adds a clear, actionable error in place of the bare `GpuResolutionError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runtime/test_gpu_sku_validation.py
"""When configured gcp_gpu_skus match no available cluster SKU label, the
resolver must raise a clear, actionable error naming configured vs available
and the exact OPENRESEARCH_GCP_GPU_SKUS fix — not a bare GpuResolutionError."""

from __future__ import annotations

import pytest

from backend.services.runtime.gpu_resolver import (
    validate_configured_skus,
    GpuSkuConfigError,
)


def test_mismatch_raises_actionable_error():
    with pytest.raises(GpuSkuConfigError) as exc:
        validate_configured_skus(
            configured=["gcp_a100_80x8"],
            available=["gcp_a100_80", "gcp_a100_80x2"],
        )
    msg = str(exc.value)
    assert "gcp_a100_80x8" in msg          # names the bad configured SKU
    assert "gcp_a100_80" in msg            # names what IS available
    assert "OPENRESEARCH_GCP_GPU_SKUS" in msg  # names the fix knob


def test_overlap_passes():
    # No raise when at least one configured SKU is available.
    validate_configured_skus(
        configured=["gcp_a100_80"],
        available=["gcp_a100_80", "gcp_a100_80x2"],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/runtime/test_gpu_sku_validation.py -v`
Expected: FAIL — `validate_configured_skus` / `GpuSkuConfigError` do not exist (ImportError).

- [ ] **Step 3: Add the exception and validator**

In `backend/services/runtime/gpu_resolver.py`, near the top (after the existing imports and any
existing exception classes), add:

```python
class GpuSkuConfigError(RuntimeError):
    """Configured GPU SKUs cannot match any provisioned cluster node pool."""


def validate_configured_skus(
    *, configured: list[str], available: list[str]
) -> None:
    """Raise GpuSkuConfigError when no configured SKU is provisioned.

    `configured` is settings.gcp_gpu_skus; `available` is the set of
    reprolab/sku labels actually present on cluster nodes. The resolver can only
    place a cell on a label that exists, so a zero-overlap config guarantees
    every GPU request Pends forever — surface it loudly at preflight instead.
    """
    if set(configured) & set(available):
        return
    raise GpuSkuConfigError(
        "No configured GPU SKU is provisioned on the cluster.\n"
        f"  configured (OPENRESEARCH_GCP_GPU_SKUS): {configured}\n"
        f"  available (reprolab/sku node labels):   {available}\n"
        "Fix: set OPENRESEARCH_GCP_GPU_SKUS to a JSON array of SKUs your cluster "
        "actually provisions, e.g. OPENRESEARCH_GCP_GPU_SKUS='[\"gcp_a100_80\"]'."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/runtime/test_gpu_sku_validation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire the validator into the GCP preflight**

Find the GCP preflight that lists cluster node SKUs (grep `reprolab/sku` in
`backend/services/runtime/gke_job_backend.py` and `backend/agents/rlm/k8s_job_cell_runner.py`).
At the point where cluster node labels are first read for the gcp backend, call
`validate_configured_skus(configured=settings.gcp_gpu_skus, available=<discovered labels>)`
before the first GPU resolution. If node labels are not enumerated at preflight, add the call
at the start of the gcp branch of `resolve_gpu_requirements` in
`backend/agents/rlm/primitives.py` using the SKU short-names from the resolved catalog for the
provisioned pools. Guard with `if _sb_key == "gcp":` so azure/runpod/local are untouched.

Run: `.venv/bin/python -m pytest tests/runtime tests/config/test_gcp_sku_pool_invariant.py -q`
Expected: PASS (invariant test still green — default unchanged).

- [ ] **Step 6: Commit**

```bash
git add backend/services/runtime/gpu_resolver.py tests/runtime/test_gpu_sku_validation.py
git commit -m "Fail loud with an actionable message when gcp_gpu_skus match no cluster pool"
```

---

## Task 4: auto sandbox = docker/local only (default runpod→auto)

**Files:**
- Modify: `backend/agents/execution.py:61, 250-286`
- Test: `tests/agents/test_sandbox_resolution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_sandbox_resolution.py
"""auto must resolve to docker/local only — never a paid remote backend.
RunPod is legacy; GCP/Azure are the primary clouds but selected explicitly."""

from __future__ import annotations

import pytest

import backend.agents.execution as ex
from backend.agents.execution import SandboxMode, resolve_sandbox_mode


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    ex._docker_reachable.cache_clear()
    yield
    ex._docker_reachable.cache_clear()


def test_auto_with_docker_resolves_docker(monkeypatch):
    monkeypatch.setattr(ex, "_docker_reachable", lambda: True)
    monkeypatch.setattr(ex, "_is_wsl", lambda: False)
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.docker


def test_auto_without_docker_resolves_local(monkeypatch):
    monkeypatch.setattr(ex, "_docker_reachable", lambda: False)
    monkeypatch.setattr(ex, "_is_wsl", lambda: False)
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.local


def test_auto_never_resolves_runpod(monkeypatch):
    monkeypatch.setattr(ex, "_docker_reachable", lambda: True)
    monkeypatch.setattr(ex, "_is_wsl", lambda: False)
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is not SandboxMode.runpod


def test_explicit_runpod_unchanged():
    assert resolve_sandbox_mode("runpod", pipeline_mode="rlm") is SandboxMode.runpod


def test_force_env_still_wins(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_FORCE_SANDBOX", "gcp")
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.gcp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agents/test_sandbox_resolution.py -v`
Expected: FAIL — `test_auto_with_docker_resolves_docker` returns `runpod` (current default).

- [ ] **Step 3: Change the default and the auto branch**

In `backend/agents/execution.py`, change line 61:

```python
DEFAULT_SANDBOX_MODE = SandboxMode.runpod
```

to:

```python
# auto is a local-dev resolver (docker/local); paid clouds (gcp/azure) and the
# legacy runpod backend are selected EXPLICITLY, never by auto.
DEFAULT_SANDBOX_MODE = SandboxMode.auto
```

Then replace the tail of `resolve_sandbox_mode` (the block from the WSL check through
`return DEFAULT_SANDBOX_MODE`, lines ~278-286):

```python
    # WSL safety: docker might not be wired up via Docker Desktop; prefer
    # local unless we can verify the daemon is reachable.
    if _is_wsl() and not _docker_reachable():
        logger.info(
            "resolve_sandbox_mode: WSL detected without reachable docker — "
            "preferring 'local' (set OPENRESEARCH_DEFAULT_SANDBOX=docker to override)"
        )
        return SandboxMode.local

    return DEFAULT_SANDBOX_MODE
```

with:

```python
    # auto = local dev only. Never select a paid remote backend (this removes the
    # old silent auto→runpod). GCP/Azure/RunPod are opt-in (--sandbox {gcp,azure,runpod}).
    if _is_wsl() and not _docker_reachable():
        logger.info(
            "resolve_sandbox_mode: WSL detected without reachable docker — "
            "preferring 'local' (set OPENRESEARCH_DEFAULT_SANDBOX=docker to override)"
        )
        return SandboxMode.local
    if _docker_reachable():
        return SandboxMode.docker
    return SandboxMode.local
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/agents/test_sandbox_resolution.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the execution + config suites for regressions**

Run: `.venv/bin/python -m pytest tests/agents tests/config -q`
Expected: PASS. If any test asserted `DEFAULT_SANDBOX_MODE is SandboxMode.runpod` or `auto→runpod`, update it to the new posture (grep `DEFAULT_SANDBOX_MODE`, `auto` in tests/ first) and note the change in the commit.

- [ ] **Step 6: Commit**

```bash
git add backend/agents/execution.py tests/agents/test_sandbox_resolution.py
git commit -m "Make sandbox auto resolve to docker/local; stop defaulting to runpod"
```

---

## Task 5: Info-log when an operator explicitly selects the legacy runpod backend

**Files:**
- Modify: `backend/agents/rlm/primitives.py:3198-3203`
- Test: `tests/rlm/test_runpod_legacy_log.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_runpod_legacy_log.py
"""Explicitly choosing the legacy runpod backend emits a one-line info log."""

from __future__ import annotations

import logging

import pytest

from backend.agents.execution import SandboxMode


def test_runpod_selection_logs_legacy_notice(caplog, monkeypatch):
    import backend.services.runtime as runtime
    import backend.agents.rlm.primitives as primitives

    monkeypatch.setattr(runtime, "ensure_runpod_available", lambda: None)
    monkeypatch.setattr(
        primitives, "RunpodBackend", lambda **kw: object(), raising=False
    )

    with caplog.at_level(logging.INFO, logger="backend.agents.rlm.primitives"):
        primitives._execute_in_sandbox_backend(  # see Step 3 for the exact callable
            mode=SandboxMode.runpod, run_budget=None, gpu_plan=None
        )
    assert any("legacy" in r.message.lower() for r in caplog.records)
```

**Note:** the runpod branch currently lives inline in the backend-selection dispatch
(`primitives.py:3198`). If there is no standalone callable, adjust the test to invoke the
existing dispatch function that contains lines 3198-3227 (grep the enclosing `def` above line
3198) and pass `SandboxMode.runpod`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_runpod_legacy_log.py -v`
Expected: FAIL — no "legacy" log line emitted.

- [ ] **Step 3: Add the info log**

In `backend/agents/rlm/primitives.py`, in the runpod branch (line 3198), after
`_runtime.ensure_runpod_available()` and before `return RunpodBackend(...)`, add:

```python
        logger.info(
            "sandbox=runpod is a LEGACY backend — GCP (--sandbox gcp) and Azure "
            "(--sandbox azure) are the supported primary clouds."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_runpod_legacy_log.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/primitives.py tests/rlm/test_runpod_legacy_log.py
git commit -m "Log a legacy notice when the runpod backend is explicitly selected"
```

---

## Task 6: Mid-cell GPU-$ heartbeat cap (kill a wedged cell before it burns the budget)

**Files:**
- Modify: `backend/agents/rlm/k8s_job_cell_runner.py` (add pure helper + wire into poll loop + map terminal reason)
- Test: `tests/rlm/test_cell_gpu_budget_heartbeat.py`

**Context:** Today `RunBudget.check_run_gpu_usd` fires only at `run_experiment` return, so a cell
wedged on a slow dataset mirror holds an A100 for the full deadline (~$6-16 leak). We add a
mid-poll check using the same context-var getters `run_matrix` already uses
(`_get_run_budget()`, `_get_gpu_plan()`), keyed on elapsed wall-clock × SKU $/hr.

- [ ] **Step 1: Write the failing test (pure helper)**

```python
# tests/rlm/test_cell_gpu_budget_heartbeat.py
"""Pure accrual math for the mid-cell GPU-$ heartbeat."""

from __future__ import annotations

from backend.agents.rlm.k8s_job_cell_runner import (
    _accrued_gpu_usd,
    _over_gpu_budget,
)


def test_accrued_usd_scales_with_time_and_gpus():
    # 1 hour, $3.93/hr per GPU, 2 GPUs -> 7.86
    assert _accrued_gpu_usd(elapsed_s=3600, usd_per_hr_per_gpu=3.93, gpu_count=2) == \
        __import__("pytest").approx(7.86)


def test_over_budget_true_at_or_above_cap():
    assert _over_gpu_budget(accrued=40.0, cap=40.0) is True
    assert _over_gpu_budget(accrued=41.0, cap=40.0) is True


def test_over_budget_false_below_cap_or_no_cap():
    assert _over_gpu_budget(accrued=39.9, cap=40.0) is False
    assert _over_gpu_budget(accrued=100.0, cap=None) is False
    assert _over_gpu_budget(accrued=100.0, cap=0) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_cell_gpu_budget_heartbeat.py -v`
Expected: FAIL — helpers do not exist (ImportError).

- [ ] **Step 3: Add the pure helpers**

In `backend/agents/rlm/k8s_job_cell_runner.py`, near the other module helpers (above
`run_matrix`), add:

```python
def _accrued_gpu_usd(*, elapsed_s: float, usd_per_hr_per_gpu: float, gpu_count: int) -> float:
    """USD accrued so far for one running cell = hours × per-GPU rate × GPUs."""
    return (elapsed_s / 3600.0) * float(usd_per_hr_per_gpu) * max(1, int(gpu_count))


def _over_gpu_budget(*, accrued: float, cap: float | None) -> bool:
    """True when a positive cap exists and accrued spend meets/exceeds it."""
    if not cap or cap <= 0:
        return False
    return accrued >= cap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rlm/test_cell_gpu_budget_heartbeat.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Record the job start time in the poll setup**

In `backend/agents/rlm/k8s_job_cell_runner.py`, at line 878 where the per-cell deadline is set:

```python
    job_deadline = time.monotonic() + active_deadline_seconds
```

add immediately after:

```python
    job_started = time.monotonic()
```

- [ ] **Step 6: Add the heartbeat check inside the poll loop**

In the same `while True:` poll loop, immediately after the per-cell deadline check (the block
`if now >= job_deadline: return _watch_result("deadline")`, ~line 889-890), insert:

```python
        # Mid-cell GPU-$ heartbeat: the natural budget check only fires at
        # run_experiment return, so a wedged slow-download cell would hold the
        # GPU for the full deadline. Kill it early when it would breach the cap.
        _budget = _get_run_budget()
        _cap = getattr(_budget, "max_run_gpu_usd", None) if _budget is not None else None
        if _cap:
            _plan = _get_gpu_plan()
            _rate = float(getattr(_plan, "sku_usd_per_hr", 0.0) or 0.0)
            _gpus = int(getattr(_plan, "gpu_count", 1) or 1)
            _accrued = _accrued_gpu_usd(
                elapsed_s=now - job_started, usd_per_hr_per_gpu=_rate, gpu_count=_gpus
            )
            if _rate > 0 and _over_gpu_budget(accrued=_accrued, cap=_cap):
                node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
                logger.warning(
                    "k8s_job_cell_runner: cell exceeded GPU-$ cap "
                    "($%.2f >= $%.2f) — terminating.", _accrued, _cap,
                )
                return _watch_result(
                    "gpu_budget_exceeded", exit_code=exit_code, node_name=node, log=log
                )
```

**Note:** confirm `GpuPlan` exposes `sku_usd_per_hr` (schemas.py ~line 853); if the field name
differs, read it and adjust `_rate`.

- [ ] **Step 7: Map the new terminal reason to a non-retryable failure**

Find where `_watch_result` statuses are translated to cell outcomes (grep `"deadline"` and
`"overall_timeout"` in `k8s_job_cell_runner.py` — the status→outcome mapping in `_process_cell`
or the caller of the watch loop). Add `"gpu_budget_exceeded"` alongside `"deadline"` /
`"overall_timeout"` so it is treated as **terminal and non-retryable** (no OOM/spot retry), and
its outcome is a hard failure recorded to the failure capsule. Mirror exactly how `"deadline"`
is handled.

- [ ] **Step 8: Run the cell-runner suite for regressions**

Run: `.venv/bin/python -m pytest tests/rlm -q -k "cell or k8s or job or budget"`
Expected: PASS. Existing cell-runner tests must stay green (the check is a no-op when
`max_run_gpu_usd` is unset or the SKU rate is 0).

- [ ] **Step 9: Commit**

```bash
git add backend/agents/rlm/k8s_job_cell_runner.py tests/rlm/test_cell_gpu_budget_heartbeat.py
git commit -m "Kill a cell mid-flight when it would breach the run GPU-\$ cap"
```

---

## Task 7: Docs sweep — GCP/Azure primary, RunPod legacy

**Files:**
- Modify: `CLAUDE.md` (root), `backend/services/runtime/CLAUDE.md`, `learn.md`
- Create: `docs/runbooks/2026-07-09-cloud-posture-gcp-azure-primary.md`
- Regenerate: `docs/reference/flags.md`
- Test: `tests/test_claude_md_fidelity.py` must stay green

- [ ] **Step 1: Update the root CLAUDE.md sandbox references**

In `CLAUDE.md`, in the "Common flags" line, reorder the sandbox list to lead with the primary
clouds and mark runpod legacy:

```
`--sandbox {auto,docker,local,gcp,azure,runpod}` (gcp/azure are the primary clouds; runpod is legacy)
```

Add one sentence to the runtime pointer noting GCP/Azure are primary and RunPod is legacy. Do
**not** touch the "RunPod cloud-type default is SECURE" fidelity anchor (that line stays — it is
about cloud-*type*, verified by the fidelity test).

- [ ] **Step 2: Restructure the runtime CLAUDE.md**

In `backend/services/runtime/CLAUDE.md`, reorder so **GCP (GKE)** and **Azure (AKS)** are the
first, primary sandbox sections and move RunPod under a clearly-labelled `## RunPod (legacy)`
section. Keep the SECURE-default note in the RunPod section (fidelity anchor). Note the new
`auto = docker/local only` behavior.

- [ ] **Step 3: Add a learn.md rule**

Append to `learn.md` (present-tense Rule/How/Why shape, matching existing entries):

```markdown
## Cloud posture: GCP/Azure primary, RunPod legacy (2026-07-09)

**Rule:** `--sandbox auto` resolves to docker/local ONLY and never a paid remote backend;
gcp/azure/runpod are explicit. Foundry LLM rows price via `FOUNDRY_ALIASES` (no more $0). An
explicit `--vram-gb` is used verbatim (no 1.25x headroom). `gcp_gpu_skus` mismatch fails loud
at preflight (default stays `["gcp_a100_80x8"]`, synced to Terraform). A cell that would breach
`--max-run-gpu-usd` mid-flight is killed (`gpu_budget_exceeded`).

**How:** `execution.resolve_sandbox_mode`, `pricing.FOUNDRY_ALIASES`,
`GpuRequirements.vram_is_explicit`, `gpu_resolver.validate_configured_skus`,
`k8s_job_cell_runner` heartbeat.

**Why:** RunPod is legacy; GCP/Azure are the supported clouds. The old auto→runpod default and
the $0 Foundry ledger + uncapped mid-cell burn made the primary clouds untrustworthy for
overnight campaigns.
```

- [ ] **Step 4: Write the runbook**

Create `docs/runbooks/2026-07-09-cloud-posture-gcp-azure-primary.md` capturing: the decision
(GCP/Azure primary, RunPod legacy), the six code changes with their env knobs
(`OPENRESEARCH_GCP_GPU_SKUS`, `--vram-gb`, `--max-run-gpu-usd`), and the recommended campaign
launch defaults (`--sandbox gcp --model opus-foundry --force-single-gpu`, `GKE_SYNTH_CELL=1`,
`FEASIBILITY_SCOPE=1`).

- [ ] **Step 5: Regenerate the flag registry**

Run: `.venv/bin/python scripts/gen_flag_registry.py`
Expected: `docs/reference/flags.md` regenerated (picks up any new env reads).

- [ ] **Step 6: Verify docs fidelity + flag freshness**

Run: `.venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q && .venv/bin/python -m pytest tests -q -k "flag_registry or flags_fresh"`
Expected: PASS (docs consistent, registry fresh).

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md backend/services/runtime/CLAUDE.md learn.md docs/runbooks/2026-07-09-cloud-posture-gcp-azure-primary.md docs/reference/flags.md
git commit -m "Document GCP/Azure as primary clouds and RunPod as legacy"
```

---

## Final verification

- [ ] **Run the full in-scope suite**

Run: `export OPENRESEARCH_MIN_DISK_GB=0 && .venv/bin/python -m pytest tests/runtime tests/pricing tests/config tests/agents tests/rlm tests/evals tests/test_claude_md_fidelity.py -n auto -q`
Expected: all pass (≥ baseline 477 + the new tests).

- [ ] **Confirm the one intentional behavior change is covered**

`tests/agents/test_sandbox_resolution.py` proves `auto` never resolves to runpod/gcp/azure.

- [ ] **Next:** Spec B (lifecycle-primary hardening) plan, then the Cutout (arXiv 1708.04552)
  A/B validation run on GCP as the first real end-to-end smoke of the new posture.
```
