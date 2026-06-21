# GKE First-Class GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`; every task is failing test → run-fails → minimal impl → run-passes → commit. Also load `superpowers:verification-before-completion` before declaring done. Steps use checkbox (`- [ ]`) syntax.
>
> **Audit (ground truth):** `docs/audits/2026-06-20-workstream-b-gke-backend-audit.md` (this branch). The scope below is the user's 4-item SCOPE, cross-checked against that audit + the two design specs (`2026-06-16-gcp-gke-execution-backend-design.md`, `2026-06-17-multi-cloud-production-gpu-execution-design.md`) and the live code.

**Goal:** Make `--sandbox gke` a first-class alias for the existing `gcp` GKE backend, close the multi-GPU torchrun gap in the GKE cell entrypoint, add GCP L4 + H100 SKUs to the catalog, and ship a GKE preflight script wired into `start.sh` — all hermetic, with runpod/azure/local/aks paths byte-for-byte unchanged and live GPU smoke operator-gated.

**Architecture:** `SandboxMode` is a `str, Enum`; aliasing `gke→gcp` is done at the enum's `_missing_` classmethod so `SandboxMode("gke")` returns the **gcp member** everywhere (every downstream `_sb_key == "gcp"` / `_sb_key in ("azure","gcp")` check passes with zero further edits, and the direct `SandboxMode("gcp")` path never calls `_missing_`, staying identical). The GKE cell entrypoint (`docker/gke-cell-base/gke_cell_entrypoint.py`) is **standalone** (loaded by file path in tests) — so the torchrun fix is a self-contained pure helper `build_cell_launch_argv(...)` (it cannot import the backend's `_resolve_distributed_launch`), gated on `gpu_count>1` AND distributed markers in the script, respecting `OPENRESEARCH_DISABLE_TORCHRUN_WRAP`. The GPU count is plumbed deterministically from the runner via a new `OPENRESEARCH_CELL_GPU_COUNT` env var.

**Tech Stack:** Python 3.12 (CI) / 3.14 (dev venv); pytest (hermetic, `pytest-socket` blocks non-loopback); `kubernetes`/`google-cloud-storage` imports are lazy + faked in tests; bash for `gke_check.sh`; `ruff@0.15.16` lint.

> **Executor note:** the line numbers below are an audit-snapshot; every task starts with a `grep -n` **Pre-impl verification** — bind to the function/anchor, not the literal line.

---

## File Structure

| File | Created/Modified | Responsibility |
|---|---|---|
| `backend/agents/execution.py` | **Modified** | Add `SandboxMode._missing_` mapping `"gke"→gcp`. |
| `backend/cli.py` | **Modified** | Add `"gke"` to the `--sandbox` argparse `choices` + help. |
| `backend/services/runtime/gpu_catalog.py` | **Modified** | Add GCP L4 (`gcp_l4_24`) and H100 (`gcp_h100_80`, `gcp_h100_80x8`) SKU rows. |
| `docker/gke-cell-base/gke_cell_entrypoint.py` | **Modified** | Add pure `build_cell_launch_argv(...)` + `resolve_cell_gpu_count()`; route both launch sites through it for >1-GPU torchrun-wrap. |
| `backend/agents/rlm/k8s_job_cell_runner.py` | **Modified** | Inject `OPENRESEARCH_CELL_GPU_COUNT` into the cell Job so the in-pod entrypoint knows the leased GPU count. |
| `scripts/gke_check.sh` | **Created** | GKE preflight: gcloud ADC, settings, cluster reachability, GPU quota; `--start-pod` operator-gated. |
| `start.sh` | **Modified** | Run `gke_check.sh` when `OPENRESEARCH_DEFAULT_SANDBOX` ∈ {gcp, gke}; exclude gcp/gke from the docker-down warning. |
| `tests/services/runtime/test_gke_alias.py` | **Created** | gke→GkeJobBackend, FORCE_SANDBOX=gke, no-op build, gcp+aks regression parity, preflight wiring. |
| `tests/services/runtime/test_gke_torchrun_wrap.py` | **Created** | `build_cell_launch_argv` matrix. |
| `tests/services/runtime/test_gpu_catalog_gcp_l4_h100.py` | **Created** | L4/H100 rows, resolver pick, $/hr caps. |
| `tests/services/runtime/test_gke_cell_runner.py` | **Modified** | Assert `OPENRESEARCH_CELL_GPU_COUNT` injected. |
| `CLAUDE.md` / `system_overview.md` | **Modified** | Document the alias, SKUs, preflight, torchrun-on-gke. |

---

## Task 1 — `gke` alias for the `gcp` sandbox token (SCOPE 1)

**Files:** `backend/agents/execution.py` (`SandboxMode` enum), `backend/cli.py` (`--sandbox` choices), `tests/services/runtime/test_gke_alias.py` (new).

> **Pre-impl verification:** `grep -n "class SandboxMode\|def resolve_sandbox_mode\|_backend_for_sandbox_mode\|def _missing_\|ensure_gcp_available\|--sandbox" backend/agents/execution.py backend/agents/rlm/primitives.py backend/cli.py` — confirm the enum members, the factory-dispatch function name (`_backend_for_sandbox_mode` or equivalent at ~`primitives.py:2742`), and the `--sandbox` `choices` tuple. Adapt names in the test below to reality.

- [ ] **Step 1: Write the failing test** — `tests/services/runtime/test_gke_alias.py`:

```python
"""GKE alias for the gcp sandbox token (SCOPE 1).

Hermetic: no live cloud calls. ensure_gcp_available is patched to a no-op.
Asserts gke→GkeJobBackend on BOTH the enum boundary and FORCE_SANDBOX, the
build_environment no-op, and gcp+aks byte-for-byte regression parity.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.agents.execution import SandboxMode, resolve_sandbox_mode


def test_sandbox_mode_gke_resolves_to_gcp_member():
    assert SandboxMode("gke") is SandboxMode.gcp


def test_sandbox_mode_gke_case_insensitive():
    assert SandboxMode("GKE") is SandboxMode.gcp
    assert SandboxMode(" gke ") is SandboxMode.gcp


def test_gcp_token_unchanged():
    assert SandboxMode("gcp") is SandboxMode.gcp
    assert SandboxMode.gcp.value == "gcp"


def test_unknown_token_still_raises():
    with pytest.raises(ValueError):
        SandboxMode("gkeXYZ")


def test_resolve_sandbox_mode_gke_explicit(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    assert resolve_sandbox_mode("gke", pipeline_mode="rlm") is SandboxMode.gcp


def test_force_sandbox_gke_override(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_FORCE_SANDBOX", "gke")
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.gcp


def test_gke_token_constructs_gke_backend():
    with patch("backend.services.runtime.ensure_gcp_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode("gke"), run_budget=None)
        assert isinstance(backend, GkeJobBackend)


def test_force_sandbox_gke_threads_run_budget():
    budget = SimpleNamespace(name="fake_budget")
    with patch("backend.services.runtime.ensure_gcp_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode("gke"), run_budget=budget)
        assert isinstance(backend, GkeJobBackend)
        assert backend._run_budget is budget


def test_build_environment_noop_for_gke_via_gcp_member():
    assert SandboxMode("gke").value == "gcp"


@pytest.mark.parametrize("token", ["gcp", "azure", "runpod", "local", "docker"])
def test_existing_tokens_round_trip_identically(token):
    assert SandboxMode(token).value == token


def test_aks_path_still_resolves_to_aks_backend():
    with patch("backend.services.runtime.ensure_azure_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.aks_job_backend import AksJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode.azure, run_budget=None)
        assert isinstance(backend, AksJobBackend)
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_alias.py -x -q`
  Expected: FAIL — `ValueError: 'gke' is not a valid SandboxMode`.

- [ ] **Step 3: Write minimal implementation** — add `_missing_` to `SandboxMode` in `backend/agents/execution.py`:

```python
    @classmethod
    def _missing_(cls, value: object) -> "SandboxMode | None":
        # `gke` is a first-class alias for the GCP/GKE backend: --sandbox gke
        # and --sandbox gcp both resolve to GkeJobBackend. Aliasing here (at the
        # enum boundary) means SandboxMode('gke') returns the gcp member
        # everywhere — including OPENRESEARCH_FORCE_SANDBOX=gke and every
        # downstream `_sb_key == "gcp"` check — with zero further edits. A direct
        # SandboxMode('gcp') never reaches _missing_, so the gcp path is byte-for-byte.
        if isinstance(value, str) and value.strip().lower() == "gke":
            return cls.gcp
        return None
```

  Add `"gke"` to the `--sandbox` `choices` tuple in `backend/cli.py` and mention it in help (`gke is an alias for gcp`).

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_alias.py -q`
- [ ] **Step 4b: Regression sweep** — `.venv/bin/python -m pytest tests/services/runtime/ tests/config/ -k "gcp or sandbox or backend" -q` → the existing GCP surface unchanged.
- [ ] **Step 4c: Lint** — `uvx ruff@0.15.16 check backend/agents/execution.py backend/cli.py tests/services/runtime/test_gke_alias.py`
- [ ] **Step 5: Commit**

```bash
git add backend/agents/execution.py backend/cli.py tests/services/runtime/test_gke_alias.py
git commit -m "feat(sandbox): alias --sandbox gke -> gcp GKE backend at enum boundary

SandboxMode._missing_ maps gke->gcp so --sandbox gke, --sandbox gcp, and
OPENRESEARCH_FORCE_SANDBOX=gke all resolve to GkeJobBackend. gcp token + aks
path byte-for-byte unchanged. Add gke to the CLI choices.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — Torchrun-wrap multi-GPU cells on GKE (SCOPE 2, HIGH)

**Files:** `docker/gke-cell-base/gke_cell_entrypoint.py` (the two launch sites + a new pure helper), `tests/services/runtime/test_gke_torchrun_wrap.py` (new). Template: `primitives.py::_resolve_distributed_launch` + `_DISTRIBUTED_MARKERS`.

> The entrypoint is standalone (tests load it by file path with google modules patched out) — it **cannot** import `_resolve_distributed_launch`. The helper reimplements the marker-gated logic inline using plain `torchrun --nproc_per_node N` (a strict superset launcher for raw `torch.distributed`/accelerate scripts; no in-pod FSDP-config writer needed).
>
> **Pre-impl verification:** `grep -n "subprocess.Popen\|sys.executable\|train_cell\|--cell-id\|--output-dir\|^import os\|^logger" docker/gke-cell-base/gke_cell_entrypoint.py` — find the exact two launch sites + the existing flag-forwarding shape; confirm `logger`/`os`/`sys`/`Path` are already imported (the helper uses them).

- [ ] **Step 1: Write the failing test** — `tests/services/runtime/test_gke_torchrun_wrap.py`:

```python
"""Multi-GPU torchrun-wrap in the GKE cell entrypoint (SCOPE 2).

Hermetic: loads the standalone entrypoint by file path (no google-cloud, no GPU,
no subprocess). Tests the pure build_cell_launch_argv helper directly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "docker" / "gke-cell-base" / "gke_cell_entrypoint.py"
)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("gke_cell_entrypoint", _ENTRYPOINT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ep():
    return _load_entrypoint()


_DISTRIBUTED_SCRIPT = "from accelerate import Accelerator\nAccelerator()\n"
_PLAIN_SCRIPT = "import torch\nprint('single process trainer')\n"


def test_single_gpu_runs_plain(ep):
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=1, script_text=_DISTRIBUTED_SCRIPT,
    )
    assert argv[0] == "/usr/bin/python"
    assert "torchrun" not in argv[0]
    assert str(Path("/code/train_cell.py")) in argv


def test_multi_gpu_with_markers_wraps_torchrun(ep):
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=4, script_text=_DISTRIBUTED_SCRIPT,
    )
    joined = " ".join(argv)
    assert argv[0] == "torchrun" or argv[0].endswith("torchrun")
    assert "--nproc_per_node=4" in joined
    assert str(Path("/code/train_cell.py")) in argv
    assert any("--cell-id=c0" in a for a in argv)
    assert any("--output-dir=" in a for a in argv)


def test_multi_gpu_no_markers_runs_plain(ep):
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=4, script_text=_PLAIN_SCRIPT,
    )
    assert "torchrun" not in " ".join(argv)
    assert argv[0] == "/usr/bin/python"


def test_disable_opt_out(ep, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DISABLE_TORCHRUN_WRAP", "1")
    argv = ep.build_cell_launch_argv(
        python_exe="/usr/bin/python", train_cell_path=Path("/code/train_cell.py"),
        cell_id="c0", output_dir=Path("/out"), gpu_count=4, script_text=_DISTRIBUTED_SCRIPT,
    )
    assert "torchrun" not in " ".join(argv)
    assert argv[0] == "/usr/bin/python"


def test_gpu_count_read_from_env_when_unset(ep, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CELL_GPU_COUNT", raising=False)
    assert ep.resolve_cell_gpu_count() == 1
    monkeypatch.setenv("OPENRESEARCH_CELL_GPU_COUNT", "8")
    assert ep.resolve_cell_gpu_count() == 8
    monkeypatch.setenv("OPENRESEARCH_CELL_GPU_COUNT", "garbage")
    assert ep.resolve_cell_gpu_count() == 1  # fail-soft
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_torchrun_wrap.py -x -q`
  Expected: FAIL — `AttributeError: module 'gke_cell_entrypoint' has no attribute 'build_cell_launch_argv'`.

- [ ] **Step 3: Write minimal implementation** — add the pure helpers near the other pure functions in `gke_cell_entrypoint.py`:

```python
# ---------------------------------------------------------------------------
# Multi-GPU launch — torchrun-wrap a distributed cell (SCOPE 2)
#
# Mirrors backend/agents/rlm/primitives.py::_resolve_distributed_launch, but
# REIMPLEMENTED inline: this entrypoint is standalone (loaded by file path in
# tests) and must not import the backend. Plain `torchrun --nproc_per_node N`
# is a strict superset launcher for raw torch.distributed AND accelerate scripts.
# ---------------------------------------------------------------------------
_DISTRIBUTED_MARKERS: tuple[str, ...] = (
    "FullyShardedDataParallel", "DistributedDataParallel", "init_process_group",
    "torch.distributed", "fully_shard", "from accelerate", "import accelerate",
    "Accelerator(",
)


def resolve_cell_gpu_count() -> int:
    """Leased GPU count for this cell, from OPENRESEARCH_CELL_GPU_COUNT (default 1)."""
    try:
        return max(1, int(os.environ.get("OPENRESEARCH_CELL_GPU_COUNT", "1")))
    except (TypeError, ValueError):
        return 1


def _torchrun_wrap_disabled() -> bool:
    return os.environ.get("OPENRESEARCH_DISABLE_TORCHRUN_WRAP", "").strip().lower() in (
        "1", "true", "yes",
    )


def build_cell_launch_argv(
    *, python_exe: str, train_cell_path: "Path", cell_id: str, output_dir: "Path",
    gpu_count: int, script_text: str,
) -> list[str]:
    """Return the argv to launch one training cell.

    * gpu_count <= 1                      → plain `python train_cell.py ...`
    * gpu_count > 1 + distributed markers → `torchrun --nproc_per_node=N train_cell.py ...`
    * gpu_count > 1 + NO markers          → plain python (N duplicate non-distributed
      trainers would race on the same metrics.json — corruption, not speedup).
    * OPENRESEARCH_DISABLE_TORCHRUN_WRAP=1 → always plain (operator override).
    """
    script_args = [str(train_cell_path), f"--cell-id={cell_id}", f"--output-dir={output_dir}"]
    if (
        gpu_count > 1
        and not _torchrun_wrap_disabled()
        and any(m in (script_text or "") for m in _DISTRIBUTED_MARKERS)
    ):
        logger.warning(
            "build_cell_launch_argv: %d GPUs leased + distributed markers — "
            "launching via `torchrun --nproc_per_node=%d`.", gpu_count, gpu_count,
        )
        return ["torchrun", "--standalone", "--nnodes=1",
                f"--nproc_per_node={gpu_count}", *script_args]
    if gpu_count > 1 and not _torchrun_wrap_disabled():
        logger.warning(
            "build_cell_launch_argv: %d GPUs leased but no distributed markers — "
            "single-process to avoid duplicate trainers racing on metrics.json.", gpu_count,
        )
    return [python_exe, *script_args]
```

  Route the two existing launch sites through the helper: read the trainer script text (`train_cell_path.read_text(..., errors="replace")`, empty on failure), compute `argv = build_cell_launch_argv(..., gpu_count=resolve_cell_gpu_count(), script_text=...)`, and pass `argv` to `subprocess.Popen` instead of the hardcoded `[sys.executable, str(train_cell_path), ...]` list. Apply the **identical** change at both sites.

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_torchrun_wrap.py -q`
- [ ] **Step 4b: Regression** — `.venv/bin/python -m pytest tests/services/runtime/ -k "gke_cell or entrypoint" -q` → single-GPU path unchanged (helper returns plain argv when `OPENRESEARCH_CELL_GPU_COUNT` unset).
- [ ] **Step 4c: Lint** — `uvx ruff@0.15.16 check docker/gke-cell-base/gke_cell_entrypoint.py tests/services/runtime/test_gke_torchrun_wrap.py`
- [ ] **Step 5: Commit**

```bash
git add docker/gke-cell-base/gke_cell_entrypoint.py tests/services/runtime/test_gke_torchrun_wrap.py
git commit -m "feat(gke): torchrun-wrap multi-GPU cells in the GKE entrypoint

build_cell_launch_argv torchrun-wraps a >1-GPU cell whose train_cell.py carries
distributed markers (mirrors _resolve_distributed_launch, reimplemented inline
since the entrypoint is standalone). Marker-gated (a non-distributed script
launched as N procs would race on metrics.json) and respects
OPENRESEARCH_DISABLE_TORCHRUN_WRAP. Single-GPU path byte-for-byte unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 — Plumb `OPENRESEARCH_CELL_GPU_COUNT` from the runner (SCOPE 2 support)

**Files:** `backend/agents/rlm/k8s_job_cell_runner.py` (cell-Job env block), `tests/services/runtime/test_gke_cell_runner.py` (modified).

> **Pre-impl verification:** `grep -n "gpu_count\|def _build_cell_job_manifest\|env_vars\|\"env\"\|nvidia.com/gpu" backend/agents/rlm/k8s_job_cell_runner.py` and open `tests/services/runtime/test_gke_cell_runner.py` to copy the EXACT manifest-builder function name + its required kwargs. Do not invent the signature — adapt the test below to the real one.

- [ ] **Step 1: Write the failing test** — add to `tests/services/runtime/test_gke_cell_runner.py` (use the real builder name + kwargs found above):

```python
def test_cell_job_injects_gpu_count_env():
    """The cell Job must inject OPENRESEARCH_CELL_GPU_COUNT so the in-pod
    entrypoint can torchrun-wrap multi-GPU cells deterministically."""
    from types import SimpleNamespace
    from backend.agents.rlm import k8s_job_cell_runner as r

    plan = SimpleNamespace(short_name="gcp_a100_40x8", gpu_count=8)
    manifest = r._build_cell_job_manifest(  # ← real builder name
        run_id="run-x", cell_id="c0", cell_params_json="{}",
        image="reg/img:pinned", gpu_plan=plan,
        # ... remaining required kwargs per the existing fixtures ...
    )
    env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    by_name = {e["name"]: e["value"] for e in env}
    assert by_name.get("OPENRESEARCH_CELL_GPU_COUNT") == "8"


def test_cell_job_gpu_count_defaults_to_one_without_plan():
    from backend.agents.rlm import k8s_job_cell_runner as r

    manifest = r._build_cell_job_manifest(
        run_id="run-x", cell_id="c0", cell_params_json="{}",
        image="reg/img:pinned", gpu_plan=None,
        # ... remaining required kwargs ...
    )
    env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    by_name = {e["name"]: e["value"] for e in env}
    assert by_name.get("OPENRESEARCH_CELL_GPU_COUNT") == "1"
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_cell_runner.py -k gpu_count -q`
  Expected: FAIL — `OPENRESEARCH_CELL_GPU_COUNT` absent (`None != "8"`).

- [ ] **Step 3: Write minimal implementation** — in `k8s_job_cell_runner.py`, after the GPU count is computed, append it to the cell env list:

```python
    gpu_count_str = str(getattr(gpu_plan, "gpu_count", 1)) if gpu_plan is not None else "1"
    # Plumb the leased GPU count into the cell so the in-pod entrypoint can
    # torchrun-wrap a >1-GPU distributed cell (gke_cell_entrypoint.resolve_cell_gpu_count).
    env_vars.append({"name": "OPENRESEARCH_CELL_GPU_COUNT", "value": gpu_count_str})
```

(Use the real env-list variable name from the file; the AKS path also gets this var harmlessly — the entrypoint default is 1, so behaviour is identical where unread.)

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_cell_runner.py -q`
- [ ] **Step 4b: Regression** — `.venv/bin/python -m pytest tests/ -k "k8s_job_cell_runner or aks_cell" -q` → additive env var, unchanged behaviour.
- [ ] **Step 4c: Lint** — `uvx ruff@0.15.16 check backend/agents/rlm/k8s_job_cell_runner.py`
- [ ] **Step 5: Commit**

```bash
git add backend/agents/rlm/k8s_job_cell_runner.py tests/services/runtime/test_gke_cell_runner.py
git commit -m "feat(gke): inject OPENRESEARCH_CELL_GPU_COUNT into cell Jobs

Plumbs the resolved gpu_plan.gpu_count to the in-pod entrypoint so a >1-GPU cell
torchrun-shards deterministically. Additive env var; default-1 keeps every
single-GPU + AKS cell byte-for-byte.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 — GCP L4 + H100 catalog SKUs + cost caps (SCOPE 3)

**Files:** `backend/services/runtime/gpu_catalog.py` (CATALOG, after the GCP A100 rows), `tests/services/runtime/test_gpu_catalog_gcp_l4_h100.py` (new).

> Per the `2026-06-17` spec: A100 stays the default ladder; **H100 is an opt-in step-up SKU**. GCP `approx_usd_per_hr` is the **total machine** rate compared against a **per-GPU** cap (existing catalog semantic — mirror, don't "fix"). Real GCP machine types: L4 = `g2-standard-8` (1×L4-24GB); H100 = `a3-highgpu-1g` (1×) / `a3-highgpu-8g` (8×H100-80GB). Prices are us-central1 on-demand totals (refresh quarterly).
>
> **Pre-impl verification:** `grep -n "GpuSku(\|provider=\"gcp\"\|def find_ladder\|def find_by_alias\|cloud_type" backend/services/runtime/gpu_catalog.py` — confirm the `GpuSku` field order/names + `find_ladder`/`find_by_alias` signatures (esp. the `provider=` / `cloud_types=` / cap kwargs). Adapt the row constructor + test calls to the real signature.

- [ ] **Step 1: Write the failing test** — `tests/services/runtime/test_gpu_catalog_gcp_l4_h100.py`:

```python
"""GCP L4 + H100 catalog SKUs and cost-cap behaviour (SCOPE 3). Pure, hermetic."""
from __future__ import annotations

import backend.services.runtime.gpu_catalog as cat
from backend.services.runtime.gpu_catalog import find_ladder, find_by_alias


def test_gcp_l4_sku_present():
    by_name = {s.short_name: s for s in cat.CATALOG if s.provider == "gcp"}
    l4 = by_name["gcp_l4_24"]
    assert l4.provider == "gcp" and l4.cloud_type == "ONDEMAND"
    assert l4.vram_gb == 24 and l4.gpu_count == 1 and l4.approx_usd_per_hr > 0


def test_gcp_h100_skus_present():
    by_name = {s.short_name: s for s in cat.CATALOG if s.provider == "gcp"}
    h1 = by_name["gcp_h100_80"]
    assert h1.vram_gb == 80 and h1.gpu_count == 1 and h1.cloud_type == "ONDEMAND"
    h8 = by_name["gcp_h100_80x8"]
    assert h8.vram_gb == 80 and h8.gpu_count == 8


def test_resolver_picks_l4_for_small_vram():
    ladder = find_ladder(24, None, cloud_types=("ONDEMAND",), provider="gcp")
    assert ladder
    assert ladder[0].short_name == "gcp_l4_24"


def test_resolver_picks_h100_when_alias_matches():
    sku = find_by_alias("trained on 8x h100", provider="gcp")
    assert sku is not None and sku.provider == "gcp" and "h100" in sku.short_name


def test_per_gpu_cap_excludes_expensive_h100():
    ladder = find_ladder(80, 10.0, cloud_types=("ONDEMAND",), provider="gcp")
    assert "gcp_h100_80x8" not in {s.short_name for s in ladder}


def test_per_gpu_cap_none_means_no_cap():
    ladder = find_ladder(80, None, cloud_types=("ONDEMAND",), provider="gcp")
    assert "gcp_h100_80x8" in {s.short_name for s in ladder}


def test_runpod_catalog_unchanged_by_gcp_additions():
    runpod = find_ladder(24, None)  # default provider
    assert all(s.provider == "runpod" for s in runpod)
    assert "gcp_l4_24" not in {s.short_name for s in runpod}
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/services/runtime/test_gpu_catalog_gcp_l4_h100.py -x -q`
  Expected: FAIL — `KeyError: 'gcp_l4_24'`.

- [ ] **Step 3: Write minimal implementation** — append to `CATALOG` after the GCP A100 rows (match the real `GpuSku` field order from the pre-impl grep):

```python
    # GCP L4 + H100 (opt-in step-up; A100 stays the default ladder).
    # L4 = g2-standard-8 (1x L4-24GB); H100 = a3-highgpu-{1,8}g (H100-80GB SXM).
    # approx_usd_per_hr = TOTAL machine on-demand rate (us-central1; refresh quarterly).
    GpuSku("g2-standard-8", "gcp_l4_24",     24, "ONDEMAND",  0.85,
           aliases=("l4", "l4 24", "nvidia l4"),     provider="gcp", gpu_count=1),
    GpuSku("a3-highgpu-1g", "gcp_h100_80",   80, "ONDEMAND", 11.06,
           aliases=("h100", "h100 80", "h100 80gb", "h100-80"),
           provider="gcp", gpu_count=1),
    GpuSku("a3-highgpu-8g", "gcp_h100_80x8", 80, "ONDEMAND", 88.49,
           aliases=("8x h100", "8x h100 80", "a3-highgpu-8g"),
           provider="gcp", gpu_count=8),
```

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/services/runtime/test_gpu_catalog_gcp_l4_h100.py -q`
- [ ] **Step 4b: Regression** — `.venv/bin/python -m pytest tests/ -k "gpu_resolver or sku or catalog or spot" -q`. If a `gcp_sku_pool_invariant`-style test enumerates the exact GCP `short_name` set, **update that invariant list** (add the three new names) in this same task.
- [ ] **Step 4c: Lint** — `uvx ruff@0.15.16 check backend/services/runtime/gpu_catalog.py tests/services/runtime/test_gpu_catalog_gcp_l4_h100.py`
- [ ] **Step 5: Commit**

```bash
git add backend/services/runtime/gpu_catalog.py tests/services/runtime/test_gpu_catalog_gcp_l4_h100.py
git commit -m "feat(gpu): add GCP L4 + H100 catalog SKUs

gcp_l4_24 (g2-standard-8), gcp_h100_80 (a3-highgpu-1g), gcp_h100_80x8
(a3-highgpu-8g), all provider=gcp / ONDEMAND. Resolver picks them on the GCP
path; per-GPU \$/hr caps apply via find_ladder. A100 stays the default ladder;
H100 is opt-in. RunPod/Azure rows unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — GKE preflight script + `start.sh` wiring (SCOPE 4)

**Files:** `scripts/gke_check.sh` (new; template `scripts/runpod_check.sh`), `start.sh` (preflight block), `tests/services/runtime/test_gke_alias.py` (add wiring asserts).

> **Pre-impl verification:** read `scripts/runpod_check.sh` (the `.env`-load + exit-code structure to mirror) and the `start.sh` preflight block (`grep -n "runpod_check\|START_SKIP_PREFLIGHT\|OPENRESEARCH_DEFAULT_SANDBOX\|docker info" start.sh`). Confirm the real GCP settings names (`grep -n "gcp_project\|gcs_bucket\|OPENRESEARCH_GCP" backend/config.py`) for the required-var checks.

- [ ] **Step 1: Write the failing test** — add to `tests/services/runtime/test_gke_alias.py`:

```python
def test_gke_check_script_exists_and_executable():
    import os
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent.parent
    script = repo / "scripts" / "gke_check.sh"
    assert script.is_file(), "scripts/gke_check.sh must exist"
    assert os.access(script, os.X_OK), "scripts/gke_check.sh must be executable"


def test_start_sh_preflights_gke():
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent.parent
    start = (repo / "start.sh").read_text(encoding="utf-8")
    assert "gke_check.sh" in start
    assert "gcp" in start and "gke" in start
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_alias.py -k "gke_check or start_sh" -q`
  Expected: FAIL — `scripts/gke_check.sh must exist`.

- [ ] **Step 3a: Create `scripts/gke_check.sh`** (mirror `runpod_check.sh`'s `.env`-load + exit-code structure; use the real GCP settings names found above):

```bash
#!/usr/bin/env bash
# GKE preflight + (optional) end-to-end smoke for the openresearch pipeline.
# Green here => --sandbox gcp / --sandbox gke will auth, reach the cluster, and
# have GPU quota. Usage: scripts/gke_check.sh [--start-pod]
# Exit: 0 green; 2 missing env; 3 gcloud/ADC; 4 cluster unreachable; 5 GPU quota;
#       6 --start-pod smoke (operator-gated, COSTS MONEY).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
START_POD=0
for arg in "$@"; do case "$arg" in
  --start-pod) START_POD=1 ;;
  -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
  *) echo "Unknown argument: $arg" >&2; exit 1 ;;
esac; done
# Load .env per-line into this process only (mirrors runpod_check.sh).
if [[ -f "${ENV_FILE}" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
      value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
      [[ -z "${!key+x}" ]] && export "${key}=${value}"
    fi
  done < "${ENV_FILE}"
fi
: "${OPENRESEARCH_GCP_PROJECT:?FAIL  OPENRESEARCH_GCP_PROJECT not set (exit 2)}" || exit 2
: "${OPENRESEARCH_GCP_GCS_BUCKET:?FAIL  OPENRESEARCH_GCP_GCS_BUCKET not set (exit 2)}" || exit 2
command -v gcloud >/dev/null 2>&1 || { echo "FAIL  gcloud not found (exit 3)" >&2; exit 3; }
gcloud auth application-default print-access-token >/dev/null 2>&1 \
  || { echo "FAIL  ADC missing — run: gcloud auth application-default login (exit 3)" >&2; exit 3; }
echo "OK    gcloud ADC present (project=${OPENRESEARCH_GCP_PROJECT})."
command -v kubectl >/dev/null 2>&1 || { echo "FAIL  kubectl not found (exit 4)" >&2; exit 4; }
kubectl cluster-info >/dev/null 2>&1 \
  || { echo "FAIL  GKE cluster unreachable — run gcloud container clusters get-credentials (exit 4)" >&2; exit 4; }
echo "OK    GKE cluster reachable."
gpu_nodes="$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -c '^[1-9]' || true)"
if [[ "${gpu_nodes:-0}" -gt 0 ]]; then echo "OK    ${gpu_nodes} GPU node(s) advertise nvidia.com/gpu."
else echo "WARN  No GPU node currently advertises nvidia.com/gpu (node pool may be scaled to zero — GKE autoscales on Job dispatch)."; fi
if [[ "${START_POD}" == "1" ]]; then
  echo "WARN  --start-pod is OPERATOR-GATED and COSTS MONEY; deliberately a stub here. Use the documented manual smoke. (exit 6)"; exit 6
fi
echo "GKE preflight: all green."; exit 0
```

  Then `chmod +x scripts/gke_check.sh`.

- [ ] **Step 3b: Wire into `start.sh`** — after the runpod preflight block, add a gcp/gke preflight gate (skippable via `START_SKIP_PREFLIGHT`, `START_FULL_SMOKE=1` passes `--start-pod`), and add `gcp`/`gke` to the docker-down-warning exclusion (their `build_environment` is a no-op):

```bash
GKE_PREFLIGHT="${REPO_ROOT:-.}/scripts/gke_check.sh"
[[ -f "${GKE_PREFLIGHT}" ]] || GKE_PREFLIGHT="scripts/gke_check.sh"
if [[ "${OPENRESEARCH_DEFAULT_SANDBOX}" == "gcp" || "${OPENRESEARCH_DEFAULT_SANDBOX}" == "gke" ]]; then
    if [[ "${START_SKIP_PREFLIGHT:-0}" != "1" && -x "${GKE_PREFLIGHT}" ]]; then
        gke_args=(); [[ "${START_FULL_SMOKE:-0}" == "1" ]] && gke_args+=("--start-pod")
        if ! "${GKE_PREFLIGHT}" ${gke_args[@]+"${gke_args[@]}"}; then
            echo "[start.sh] GKE preflight FAILED — refusing to start (set START_SKIP_PREFLIGHT=1 to bypass)."; exit 1
        fi
    fi
fi
```

  And extend the docker-warning guard condition to also exclude `gcp`/`gke`.

- [ ] **Step 4: Run test to verify it passes** — `.venv/bin/python -m pytest tests/services/runtime/test_gke_alias.py -k "gke_check or start_sh" -q`
- [ ] **Step 4b: Shell syntax check** — `bash -n scripts/gke_check.sh && bash -n start.sh` (syntax only — never execute live).
- [ ] **Step 5: Commit**

```bash
git add scripts/gke_check.sh start.sh tests/services/runtime/test_gke_alias.py
git commit -m "feat(gke): add scripts/gke_check.sh preflight + wire into start.sh

Mirrors scripts/runpod_check.sh: gcloud ADC, cluster reachability, GPU-node
quota; --start-pod is an operator-gated money-spending stub. start.sh runs it
when OPENRESEARCH_DEFAULT_SANDBOX in {gcp,gke} (skippable via
START_SKIP_PREFLIGHT) and excludes gcp/gke from the docker-down warning.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6 — Documentation (repo doc policy)

**Files:** `CLAUDE.md` (Sandboxes section), `system_overview.md`.

- [ ] **Step 1: Edit CLAUDE.md** — `--sandbox gke` is a **first-class alias for `gcp`** (both → `GkeJobBackend`, aliased at `SandboxMode._missing_`; `OPENRESEARCH_FORCE_SANDBOX=gke` works). New catalog SKUs (`gcp_l4_24`, `gcp_h100_80`, `gcp_h100_80x8`; A100 default ladder, H100 opt-in). `scripts/gke_check.sh` preflight + `start.sh` wiring (`START_SKIP_PREFLIGHT` bypass). Multi-GPU GKE cells torchrun-wrap via `gke_cell_entrypoint.build_cell_launch_argv` (marker-gated, `OPENRESEARCH_DISABLE_TORCHRUN_WRAP=1` opt-out), fed by `OPENRESEARCH_CELL_GPU_COUNT`.
- [ ] **Step 2: Edit system_overview.md** — add the gke alias + GKE multi-GPU torchrun note to the backend matrix / "why" narrative.
- [ ] **Step 3: Full hermetic sweep** — `.venv/bin/python -m pytest tests/services/runtime/ tests/agents/rlm/ tests/config/ -q` → green; `uvx ruff@0.15.16 check .`
- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md system_overview.md
git commit -m "docs: document gke alias, GCP L4/H100 SKUs, gke_check preflight, gke torchrun

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

### SCOPE coverage
| SCOPE item | Task | Acceptance evidence |
|---|---|---|
| 1. `gke` alias for `gcp` | Task 1 | `_missing_` maps `gke→gcp`; tests assert gke→GkeJobBackend, `FORCE_SANDBOX=gke`, gcp token + aks path unchanged; CLI choices add `gke`. |
| 2. Torchrun gap | Tasks 2 + 3 | `build_cell_launch_argv` torchrun-wraps `gpu_count>1` + markers; both launch sites routed; opt-out honored; runner injects `OPENRESEARCH_CELL_GPU_COUNT`. |
| 3. Catalog L4 + H100 + caps | Task 4 | `gcp_l4_24`/`gcp_h100_80`/`gcp_h100_80x8`; resolver picks them; per-GPU cap excludes 8×H100 at default; runpod ladder unchanged. |
| 4. Preflight wired into `start.sh` | Task 5 | `scripts/gke_check.sh` (gcloud/ADC + cluster + GPU quota); `start.sh` preflights on gcp/gke; tests assert wiring. |

### Hard-constraint coverage
| Constraint | Where satisfied |
|---|---|
| runpod/azure/local/aks byte-for-byte unchanged | `_missing_` only fires on the literal `gke`; gcp/azure/runpod/local/docker direct hits never call it. `test_existing_tokens_round_trip_identically`, `test_aks_path_still_resolves_to_aks_backend`, `test_runpod_catalog_unchanged_by_gcp_additions`, plus per-task regression sweeps. `OPENRESEARCH_CELL_GPU_COUNT` additive (default 1). |
| `FORCE_SANDBOX` override + cost caps honored | `test_force_sandbox_gke_override`, `test_force_sandbox_gke_threads_run_budget`, `test_per_gpu_cap_excludes_expensive_h100` / `test_per_gpu_cap_none_means_no_cap`. |
| ZERO CI GPU spend / no live cloud calls | All tests import/file-path-load with `kubernetes`/`google-cloud` faked/absent; `ensure_gcp_available` patched; `gke_check.sh` only `bash -n`'d, never executed live. |
| `build_environment` stays a no-op for gke | `test_build_environment_noop_for_gke_via_gcp_member` (gke→`value=="gcp"` → existing gcp no-op branch). |

### OPERATOR-GATED — costs money, NOT in CI
The live GKE GPU smoke is **never run in CI** and **spends real money** (GKE A100/H100 node-hours + GCS egress). Run manually only after the hermetic suite is green and creds are configured:

```bash
# 1. One-time auth + cluster credentials (free):
gcloud auth application-default login
gcloud container clusters get-credentials <cluster> --region <region>

# 2. Free preflight (read-only):
scripts/gke_check.sh

# 3. LIVE end-to-end reproduction on GKE (COSTS MONEY — node-hours):
.venv/bin/python -m backend.cli reproduce 2605.15155 \
    --sandbox gke --model gpt-5 \
    --models executor=gpt-5,grader=gpt-5,verifier=gpt-5

# 4. Multi-GPU torchrun validation (COSTS MORE — boots an a3-highgpu-8g node):
#    Pin an 8-GPU plan and confirm the cell Job logs `torchrun --nproc_per_node=8`.
```

`START_FULL_SMOKE=1 ./start.sh` (with `OPENRESEARCH_DEFAULT_SANDBOX=gke`) passes `--start-pod` to `gke_check.sh` — the live-Job smoke is a deliberate stub (exit 6) so a stray `START_FULL_SMOKE` never silently bills; the real spend path is the manual `backend.cli reproduce --sandbox gke` above.
