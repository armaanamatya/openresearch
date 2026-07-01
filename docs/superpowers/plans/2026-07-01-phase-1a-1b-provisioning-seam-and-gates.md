# Phase 1a + 1b — Provisioning Seam + Deterministic Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift SDAR's three environment-provisioning code paths behind a paper-agnostic `EnvironmentAdapter` interface + a generalized `AssetCache` (Phase 1a), and add the deterministic pre-GPU-lease feasibility gates — `RunBudget.max_gpu_hours`, `RunPlan.required_assets` extraction, `FeasibilityTriage`, `estimate_scope_cost` (Phase 1b) — without changing the behavior of the live SDAR run.

**Architecture:** Phase 1a is a **strangler-fig refactor**: `EnvCacheManager` / `provision_scope` stay as a byte-for-byte **facade** that delegates to a new `env_adapters/` registry over an `AssetCache`. Phase 1b is **new, unwired, deterministic infrastructure**: pure modules the (Phase-1c) `ReproductionRun` will call, tested hermetically, touching no live path. Every behavior claim is proven by tests — the 45 existing `env_cache` tests stay green **unchanged**, plus new characterization + unit tests.

**Tech Stack:** Python 3.12 (Docker) / floor 3.11; `pytest` (socket-hermetic — `pytest-socket` blocks non-loopback); stdlib + `pydantic` (existing schemas). No new third-party deps.

## Global Constraints

- **Never break the live SDAR run.** The public import surface of `env_cache.py` is frozen: `EnvCacheManager`, `provision_scope`, `EnvSetupResult`, `ProvisionResult`, `default_cache_dir`, `FULL_SCOPE_ENV_GUIDANCE` must remain importable from `backend.services.runtime.env_cache` with identical signatures + behavior. Callers to leave untouched: `backend/agents/rlm/run.py:3059`, `backend/cli.py:1729`, `scripts/sdar_gcp_assets.py:106`, `scripts/batch_reproduce.py:490`.
- **Zero test churn on the existing suite.** `tests/services/runtime/test_env_cache.py` (30 tests), `test_env_cache_dense.py`, `test_asset_provisioning.py` (45 total) must pass **without edits**. If a refactor step would require editing an existing test, the refactor is wrong — rework the facade instead. (The monkeypatch seams `EC.os.kill` / `EC.time.sleep` are global singleton-module patches that apply everywhere; `EC._pid_alive` is threaded into `WebShopAdapter` via constructor DI from the facade so the pre-construction monkeypatch is honored — see Task 5.)
- **All new feature-flags default-OFF ⇒ byte-identical.** Phase 1b adds no flag-gated live behavior (the modules are unwired). Phase 1a is a refactor (not flag-gated) proven behavior-preserving by tests, per spec §8/§9/§10.
- **Env-var naming is canonically `OPENRESEARCH_*`.** (Legacy `REPROLAB_*` is bridged at import by `config._apply_legacy_env_aliases`; use `OPENRESEARCH_*` in new code.)
- **Fail-soft everywhere.** Provisioning failure → a verified `Exclusion` (never an exception, never a fake-0). Triage/estimate helpers never raise into a caller — bad input → a conservative decision.
- **Commits are milestone-level**, not per-task: one commit at the end of Phase 1a, one at the end of Phase 1b (the branch `reconcile/grounded-self-improvement-on-main` is already a feature branch). No Conventional-Commits prefix; descriptive present-tense headline; **no `Co-Authored-By` trailer**; author `lolout1 / appradhann@gmail.com`. Confirm with the operator before the first commit.
- **Run tests with** `.venv/bin/python -m pytest <path> -q`. Lint with `uvx ruff@0.15.16 check <path>`.

---

## PHASE 1a — Provisioning-seam refactor

### File structure (1a)

| File | New/Modify | Responsibility |
|---|---|---|
| `backend/services/runtime/asset_cache.py` | **Create** | The generalized crash-safe store: `AssetCache` (fcntl-locked keyed state dict, lifted from `EnvCacheManager._locked_state`), `default_cache_dir()`, `_pid_alive()`. Stdlib-only. |
| `backend/services/runtime/env_adapters/__init__.py` | **Create** | Package exports: `EnvironmentAdapter`, `EnvSetupResult`, `SmokeResult`, `HealthReport`, `ProvisionCtx`, the 3 adapters, `resolve_adapter`. |
| `backend/services/runtime/env_adapters/base.py` | **Create** | `EnvironmentAdapter` ABC + result/ctx types + the shared `_fail()` helper. Imports `exclusion` only. |
| `backend/services/runtime/env_adapters/alfworld.py` | **Create** | `AlfworldAdapter` — `ensure_alfworld` logic lifted verbatim. |
| `backend/services/runtime/env_adapters/webshop.py` | **Create** | `WebShopAdapter` — `acquire_webshop`/`release_webshop` logic lifted verbatim. |
| `backend/services/runtime/env_adapters/search_qa.py` | **Create** | `SearchQaAdapter` — `ensure_search_qa_index` logic lifted verbatim. |
| `backend/services/runtime/env_adapters/registry.py` | **Create** | `resolve_adapter(env_name, adapters)` — name-routing (replaces the `setup()` if-ladder). |
| `backend/services/runtime/env_cache.py` | **Modify (rewrite as facade)** | `EnvCacheManager`/`provision_scope`/`ProvisionResult`/`FULL_SCOPE_ENV_GUIDANCE` kept; methods delegate to adapters; re-exports `EnvSetupResult`/`default_cache_dir`/`_pid_alive`. |
| `tests/services/runtime/test_env_provisioning_characterization.py` | **Create** | Public-API, DI-only behavior contract (refactor-robust). |
| `tests/services/runtime/env_adapters/test_*.py` | **Create** | Per-adapter + AssetCache + registry unit tests. |

### Interfaces (1a) — the contracts every task shares

```python
# backend/services/runtime/env_adapters/base.py
@dataclass
class EnvSetupResult:                      # MOVED verbatim from env_cache.py (fields + as_env_vars unchanged)
    env: str
    ok: bool
    data_path: str | None = None
    base_url: str | None = None
    exclusion: Exclusion | None = None
    detail: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    def as_env_vars(self) -> dict[str, str]: ...   # identical body

@dataclass(frozen=True)
class SmokeResult:                         # NEW — cheap liveness (the served>0 signal)
    ok: bool
    served: int | None = None
    detail: str = ""

@dataclass(frozen=True)
class HealthReport:                        # NEW — runtime env_health.jsonl view
    env: str
    served: int = 0
    unavailable: int = 0
    detail: str = ""

@dataclass(frozen=True)
class ProvisionCtx:                        # NEW — minimal; 1c enriches it
    display_name: str = ""
    code_dir: str | None = None            # for health() to read env_health.jsonl

class EnvironmentAdapter(ABC):
    key: str                               # class attribute, e.g. "alfworld"
    def applies(self, env_name: str) -> bool: ...          # name/alias match
    @abstractmethod
    def provision(self, ctx: ProvisionCtx) -> EnvSetupResult: ...
    def smoke(self, ctx: ProvisionCtx) -> SmokeResult: ...  # default: SmokeResult(ok=True)
    def health(self, ctx: ProvisionCtx) -> HealthReport: ... # default: reads env_liveness.read_env_health

def _fail(env: str, reason: str, evidence: str = "") -> EnvSetupResult: ...  # MOVED from env_cache

# backend/services/runtime/asset_cache.py
def default_cache_dir() -> Path: ...        # MOVED verbatim
def _pid_alive(pid: int) -> bool: ...       # MOVED verbatim
class AssetCache:
    def __init__(self, cache_dir: Path | str | None = None) -> None: ...
    cache_dir: Path
    @contextlib.contextmanager
    def locked_state(self) -> Iterator[dict[str, Any]]: ...   # == EnvCacheManager._locked_state

# backend/services/runtime/env_adapters/registry.py
def resolve_adapter(env_name: str, adapters: Sequence[EnvironmentAdapter]) -> EnvironmentAdapter | None: ...
```

Adapter constructors (side-effect callables injected, defaults = the lifted `_default_*`):

```python
AlfworldAdapter(cache: AssetCache, *, downloader=None, clock=time.monotonic)
WebShopAdapter(cache: AssetCache, *, server_launcher=None, probe=None, inprocess_smoke=None,
               pid_alive=None, webshop_port=3000, server_ready_timeout_s=60.0, clock=time.monotonic)
SearchQaAdapter(cache: AssetCache, *, index_builder=None, clock=time.monotonic)
```

---

### Task 1: Public-API characterization contract (write FIRST, against current code)

**Files:**
- Test: `tests/services/runtime/test_env_provisioning_characterization.py` (Create)

**Interfaces:**
- Consumes: the CURRENT `env_cache.EnvCacheManager` / `provision_scope` public API.
- Produces: a DI-only behavior contract that must stay green across the whole refactor.

**Why first:** These tests use ONLY constructor injection (`downloader`/`server_launcher`/`probe`/`inprocess_smoke`/`clock` + a new `pid_alive` kwarg added in Task 5) — no module monkeypatching — so they are robust to where the logic physically lives. They pin the exact env-var output shape, the exclusion routing, and the WebShop lifecycle. They pass against the current code too (except the two that use the `pid_alive=` kwarg, which are added in Task 5's step).

- [ ] **Step 1: Write the characterization tests**

```python
"""Refactor-robust behavior contract for env provisioning (Phase 1a).

Injection is via the public constructor ONLY (no module monkeypatching), so these
tests pin BEHAVIOR independent of where the provisioning logic physically lives.
They must stay green byte-for-byte across the env_adapters refactor.
"""
from __future__ import annotations
from pathlib import Path

from backend.agents.rlm import exclusion as X
from backend.services.runtime.env_cache import EnvCacheManager, provision_scope


def _dl_ok(cache_dir: Path) -> None:
    d = Path(cache_dir) / "json_2.1.1" / "train" / "g0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "traj_data.json").write_text("{}", encoding="utf-8")


def _dl_fail(cache_dir: Path) -> None:
    raise RuntimeError("alfworld-download exit 1")


def test_alfworld_env_var_shape(tmp_path: Path):
    m = EnvCacheManager(tmp_path, downloader=_dl_ok)
    r = m.ensure_alfworld()
    assert r.ok and r.as_env_vars() == {"ALFWORLD_DATA": str((tmp_path / "alfworld").resolve())}


def test_alfworld_failure_is_verified_exclusion_only(tmp_path: Path):
    m = EnvCacheManager(tmp_path, downloader=_dl_fail)
    r = m.ensure_alfworld()
    assert not r.ok and r.as_env_vars() == {}
    assert r.exclusion and r.exclusion.verified and r.exclusion.kind == X.KIND_ENV_SETUP_FAILED


def test_search_qa_bm25_env_var_shape(tmp_path: Path):
    # index_builder returns None → BM25, never an exclusion.
    m = EnvCacheManager(tmp_path, index_builder=lambda c: None)
    r = m.ensure_search_qa_index()
    assert r.ok and r.as_env_vars() == {"SEARCH_QA_RETRIEVER": "bm25"}


def test_provision_scope_env_vars_and_exclusions_contract(tmp_path: Path):
    # ALFWorld fails (→ exclusion), Search-QA runs BM25 (→ env var, no exclusion).
    m = EnvCacheManager(tmp_path, downloader=_dl_fail, index_builder=lambda c: None)
    res = provision_scope(["ALFWorld", "Search-QA"], m)
    assert res.env_vars == {"SEARCH_QA_RETRIEVER": "bm25"}
    assert [e.item for e in res.exclusions] == ["ALFWorld"]
    assert X.build_scope_block(res.exclusions)["environments_skipped"] == ["ALFWorld"]
    res.release()  # no webshop lease → safe no-op


def test_webshop_inprocess_env_var_shape(tmp_path: Path, monkeypatch):
    data = tmp_path / "ws"; data.mkdir()
    monkeypatch.setenv("WEBSHOP_DATA_DIR", str(data))
    monkeypatch.delenv("WEBSHOP_PACKAGE_DIR", raising=False)
    m = EnvCacheManager(tmp_path, inprocess_smoke=lambda d: True)
    r = m.acquire_webshop()
    assert r.ok and r.as_env_vars() == {"WEBSHOP_DATA_DIR": str(data)}
    assert r.base_url is None
```

- [ ] **Step 2: Run against current code**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_env_provisioning_characterization.py -q`
Expected: **5 passed** (they characterize the current behavior).

*(No commit yet — part of the Phase 1a milestone commit.)*

---

### Task 2: `AssetCache` (generalized crash-safe store)

**Files:**
- Create: `backend/services/runtime/asset_cache.py`
- Test: `tests/services/runtime/env_adapters/test_asset_cache.py` (+ `tests/services/runtime/env_adapters/__init__.py` if needed)

**Interfaces:**
- Produces: `AssetCache`, `default_cache_dir`, `_pid_alive` (signatures above). `env_cache.py` will re-export these in Task 7.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from backend.services.runtime.asset_cache import AssetCache, default_cache_dir, _pid_alive


def test_locked_state_persists_across_instances(tmp_path: Path):
    with AssetCache(tmp_path).locked_state() as st:
        st["alfworld"] = {"ready": True, "data_path": "/x"}
    # A fresh instance (later run/cell) sees the persisted record.
    with AssetCache(tmp_path).locked_state() as st2:
        assert st2["alfworld"] == {"ready": True, "data_path": "/x"}


def test_locked_state_rolls_back_nothing_on_read(tmp_path: Path):
    with AssetCache(tmp_path).locked_state() as st:
        assert st == {}                       # empty on cold cache


def test_default_cache_dir_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ENV_CACHE_DIR", str(tmp_path / "e"))
    assert default_cache_dir() == (tmp_path / "e").resolve()


def test_pid_alive_self_true():
    import os
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(-1) is False
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/services/runtime/env_adapters/test_asset_cache.py -q` → FAIL (module not found).

- [ ] **Step 3: Implement `asset_cache.py`**

Move **verbatim** from `env_cache.py`: `default_cache_dir` (lines 146–152), `_pid_alive` (187–199), and the `_locked_state`/`_read_state`/`_write_state` bodies (408–436) onto `AssetCache`. `AssetCache.__init__` sets `self.cache_dir` (resolve/mkdir), `self._state_path = cache_dir/"env_cache_state.json"`, `self._lock_path = cache_dir/".env_cache.lock"` — **identical filenames** so a warm SDAR cache disk is byte-compatible. Rename `_locked_state` → public `locked_state`. Stdlib-only (`contextlib`, `fcntl`, `json`, `os`, `tempfile`, `pathlib`).

- [ ] **Step 4: Run to verify it passes** — Expected: **4 passed**.

---

### Task 3: `EnvironmentAdapter` base + result types

**Files:**
- Create: `backend/services/runtime/env_adapters/base.py`, `backend/services/runtime/env_adapters/__init__.py`
- Test: `tests/services/runtime/env_adapters/test_base.py`

**Interfaces:**
- Consumes: `backend.agents.rlm.exclusion` (`Exclusion`, `AXIS_ENVIRONMENT`, `KIND_ENV_SETUP_FAILED`).
- Produces: `EnvironmentAdapter`, `EnvSetupResult`, `SmokeResult`, `HealthReport`, `ProvisionCtx`, `_fail` (signatures above).

- [ ] **Step 1: Write the failing test**

```python
from backend.services.runtime.env_adapters.base import (
    EnvironmentAdapter, EnvSetupResult, SmokeResult, HealthReport, ProvisionCtx, _fail,
)
from backend.agents.rlm import exclusion as X


def test_env_setup_result_env_vars_roundtrip():
    r = EnvSetupResult(env="ALFWorld", ok=True, data_path="/d")
    assert r.as_env_vars() == {"ALFWORLD_DATA": "/d"}
    assert EnvSetupResult(env="X", ok=False).as_env_vars() == {}


def test_fail_builds_verified_exclusion():
    r = _fail("WebShop", "boom", evidence="e")
    assert not r.ok and r.exclusion.verified and r.exclusion.kind == X.KIND_ENV_SETUP_FAILED
    assert r.exclusion.axis == X.AXIS_ENVIRONMENT and r.exclusion.item == "WebShop"


def test_default_smoke_and_health_are_safe():
    class _A(EnvironmentAdapter):
        key = "x"
        def applies(self, env_name): return env_name == "x"
        def provision(self, ctx): return EnvSetupResult(env="x", ok=True)
    a = _A()
    assert a.smoke(ProvisionCtx()).ok is True
    assert isinstance(a.health(ProvisionCtx(display_name="x")), HealthReport)
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `base.py`**

Move `EnvSetupResult` (env_cache 155–184) + `_fail` (438–447) verbatim. Add `SmokeResult`/`HealthReport`/`ProvisionCtx` (frozen dataclasses, signatures above). `EnvironmentAdapter` ABC: `applies` default `return self.key in _canon(env_name)` (each adapter overrides with its alias set); `smoke` default `SmokeResult(ok=True)`; `health` default reads `backend.agents.rlm.env_liveness.read_env_health(ctx.code_dir)` (lazy import, fail-soft → `HealthReport(env=ctx.display_name)`), summing `served`/`unavailable` for the matching env; `provision` abstract. `__init__.py` re-exports the base symbols (adapters added in later tasks).

- [ ] **Step 4: Run to verify it passes** — Expected: **3 passed**.

---

### Task 4: `AlfworldAdapter`

**Files:**
- Create: `backend/services/runtime/env_adapters/alfworld.py`
- Test: `tests/services/runtime/env_adapters/test_alfworld_adapter.py`

**Interfaces:**
- Consumes: `AssetCache`, `EnvSetupResult`, `_fail`, `ProvisionCtx`.
- Produces: `AlfworldAdapter(cache, *, downloader=None, clock=time.monotonic)`; `key="alfworld"`; `provision(ctx)`; `smoke(ctx)`.

- [ ] **Step 1: Write the failing test** (mirror the ALFWorld cases from `test_env_cache.py`, but drive the adapter directly)

```python
from pathlib import Path
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.alfworld import AlfworldAdapter
from backend.services.runtime.env_adapters.base import ProvisionCtx
from backend.agents.rlm import exclusion as X


class _DL:
    def __init__(self, fail=False, games=True):
        self.calls = 0; self.fail = fail; self.games = games
    def __call__(self, cache_dir: Path) -> None:
        self.calls += 1
        if self.fail: raise RuntimeError("boom")
        if self.games:
            g = Path(cache_dir) / "json_2.1.1" / "train" / "g0"
            g.mkdir(parents=True, exist_ok=True); (g / "traj_data.json").write_text("{}")


def test_downloads_once_then_cache_hit(tmp_path):
    dl = _DL()
    a = AlfworldAdapter(AssetCache(tmp_path), downloader=dl)
    r1 = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert r1.ok and dl.calls == 1 and r1.detail == "downloaded"
    r2 = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert r2.ok and dl.calls == 1 and r2.detail == "cache hit"


def test_empty_download_is_verified_exclusion(tmp_path):
    a = AlfworldAdapter(AssetCache(tmp_path), downloader=_DL(games=False))
    r = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert not r.ok and r.exclusion.verified and r.exclusion.kind == X.KIND_ENV_SETUP_FAILED


def test_smoke_reflects_games_present(tmp_path):
    dl = _DL()
    a = AlfworldAdapter(AssetCache(tmp_path), downloader=dl)
    r = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert a.smoke(ProvisionCtx(code_dir=None, display_name=r.data_path)).ok is True
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `alfworld.py`** — move `_alfworld_has_games` (65–83), `_resolve_console_script` (202–214), `_default_alfworld_downloader` (217–233) into this module. `provision(ctx)` = the `ensure_alfworld` body (464–517) with `display_name = ctx.display_name or "ALFWorld"`, `self._cache.locked_state()` for the lock, `self._downloader` (default `_default_alfworld_downloader`), `self._clock`. `smoke(ctx)` = `SmokeResult(ok=_alfworld_has_games(<data_path>))` (data_path resolved from the cached state or `ctx`). `applies(name)` matches `{"alfworld","alf world","alf-world"}`.

- [ ] **Step 4: Run to verify it passes** — Expected: **3 passed**.

---

### Task 5: `WebShopAdapter` (the lifecycle-critical one)

**Files:**
- Create: `backend/services/runtime/env_adapters/webshop.py`
- Test: `tests/services/runtime/env_adapters/test_webshop_adapter.py`

**Interfaces:**
- Consumes: `AssetCache`, `EnvSetupResult`, `_fail`, `ProvisionCtx`, `SmokeResult`.
- Produces: `WebShopAdapter(cache, *, server_launcher=None, probe=None, inprocess_smoke=None, pid_alive=None, webshop_port=3000, server_ready_timeout_s=60.0, clock=time.monotonic)`; `provision(ctx)`, `release()`, `smoke(ctx)`; `key="webshop"`.

**CRITICAL — pid-liveness injection.** `pid_alive` is a constructor kwarg defaulting to `asset_cache._pid_alive` (via `from ..asset_cache import _pid_alive`). The adapter calls `self._pid_alive(pid)` (never a module global) so the facade can thread the monkeypatched `env_cache._pid_alive` (Task 7). The adapter uses plain `import os` / `import signal` / `import time` for `os.kill` / `SIGTERM` / `time.sleep` — those are global singleton-module patches and are honored automatically.

- [ ] **Step 1: Write the failing test** (drive the adapter with DI — the same behaviors the existing suite pins, but injected not monkeypatched)

```python
import signal
from pathlib import Path
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.webshop import WebShopAdapter
from backend.services.runtime.env_adapters.base import ProvisionCtx
from backend.agents.rlm import exclusion as X


def test_one_server_many_leases_then_torn_down(tmp_path, monkeypatch):
    launches, kills = [], []
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.os.kill",
                        lambda pid, sig: kills.append((pid, sig)))
    a = WebShopAdapter(AssetCache(tmp_path), server_launcher=lambda c, p: (launches.append(p) or 42),
                       probe=lambda u: True, pid_alive=lambda pid: True)
    r1 = a.provision(ProvisionCtx(display_name="WebShop")); assert r1.ok and r1.base_url and len(launches) == 1
    a.provision(ProvisionCtx(display_name="WebShop")); assert len(launches) == 1   # reuse
    a.release(); assert kills == []                                                # 1 lease left
    a.release(); assert kills and kills[-1][1] == signal.SIGTERM                   # torn down


def test_not_ready_fails_and_kills(tmp_path, monkeypatch):
    kills = []
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.os.kill",
                        lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.time.sleep", lambda *_: None)
    a = WebShopAdapter(AssetCache(tmp_path), server_launcher=lambda c, p: 9, probe=lambda u: False,
                       pid_alive=lambda pid: True, server_ready_timeout_s=0.01)
    r = a.provision(ProvisionCtx(display_name="WebShop"))
    assert not r.ok and r.exclusion.verified and kills and kills[-1][1] == signal.SIGTERM


def test_stale_pid_relaunches(tmp_path, monkeypatch):
    alive = {42: True}; launches = []
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.os.kill", lambda p, s: None)
    a = WebShopAdapter(AssetCache(tmp_path), server_launcher=lambda c, p: (launches.append(1) or 42),
                       probe=lambda u: True, pid_alive=lambda pid: alive.get(pid, False))
    a.provision(ProvisionCtx(display_name="WebShop")); assert len(launches) == 1
    alive[42] = False; a.release(); alive[42] = True
    a.provision(ProvisionCtx(display_name="WebShop")); assert len(launches) == 2


def test_inprocess_path(tmp_path, monkeypatch):
    data = tmp_path / "d"; data.mkdir()
    monkeypatch.setenv("WEBSHOP_DATA_DIR", str(data))
    monkeypatch.delenv("WEBSHOP_PACKAGE_DIR", raising=False)
    a = WebShopAdapter(AssetCache(tmp_path), inprocess_smoke=lambda d: True)
    r = a.provision(ProvisionCtx(display_name="WebShop"))
    assert r.ok and r.as_env_vars() == {"WEBSHOP_DATA_DIR": str(data)} and r.base_url is None
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `webshop.py`** — move `_default_webshop_launcher` (236–256), `_default_inprocess_smoke` (259–281), `_default_probe` (284–292), `_find_sdar_webshop_pkg` (519–537), `_await_ready` (684–691). `provision(ctx)` = the `acquire_webshop` body (539–619) with `self._cache.locked_state()`, `self._pid_alive`, `self._launcher`/`self._probe`/`self._inprocess_smoke`, `self._webshop_port`, `self._ready_timeout_s`, `self._clock`. `release()` = the `release_webshop` body (621–638). `smoke(ctx)` = in-process smoke or `self._probe(base_url)`. `applies(name)` matches `{"webshop","web shop","web-shop"}`.

- [ ] **Step 4: Run to verify it passes** — Expected: **4 passed**.

---

### Task 6: `SearchQaAdapter` + registry

**Files:**
- Create: `backend/services/runtime/env_adapters/search_qa.py`, `backend/services/runtime/env_adapters/registry.py`
- Test: `tests/services/runtime/env_adapters/test_search_qa_adapter.py`, `tests/services/runtime/env_adapters/test_registry.py`

**Interfaces:**
- Produces: `SearchQaAdapter(cache, *, index_builder=None, clock=time.monotonic)`, `key="search_qa"`; `resolve_adapter(env_name, adapters) -> EnvironmentAdapter | None`.

- [ ] **Step 1: Write the failing tests**

```python
# test_search_qa_adapter.py
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.search_qa import SearchQaAdapter
from backend.services.runtime.env_adapters.base import ProvisionCtx


def test_bm25_when_no_index(tmp_path):
    a = SearchQaAdapter(AssetCache(tmp_path), index_builder=lambda c: None)
    r = a.provision(ProvisionCtx(display_name="Search-QA"))
    assert r.ok and r.as_env_vars() == {"SEARCH_QA_RETRIEVER": "bm25"}


def test_dense_when_index_built(tmp_path):
    idx = tmp_path / "idx"; idx.mkdir(); (idx / "x.faiss").write_text("")
    a = SearchQaAdapter(AssetCache(tmp_path), index_builder=lambda c: idx)
    r = a.provision(ProvisionCtx(display_name="Search-QA"))
    assert r.ok and r.env_vars["SEARCH_QA_RETRIEVER"] == "e5"
    assert r.env_vars["SEARCH_QA_INDEX_DIR"] == str(idx)


def test_never_excludes_on_builder_raise(tmp_path):
    def _boom(c): raise RuntimeError("x")
    a = SearchQaAdapter(AssetCache(tmp_path), index_builder=_boom)
    r = a.provision(ProvisionCtx(display_name="Search-QA"))
    assert r.ok and r.env_vars == {"SEARCH_QA_RETRIEVER": "bm25"}   # degrade, never exclude
```

```python
# test_registry.py
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.registry import resolve_adapter
from backend.services.runtime.env_adapters.alfworld import AlfworldAdapter
from backend.services.runtime.env_adapters.webshop import WebShopAdapter
from backend.services.runtime.env_adapters.search_qa import SearchQaAdapter


def test_routes_by_name_and_alias(tmp_path):
    c = AssetCache(tmp_path)
    adapters = [AlfworldAdapter(c), WebShopAdapter(c), SearchQaAdapter(c)]
    assert isinstance(resolve_adapter("alf-world", adapters), AlfworldAdapter)
    assert isinstance(resolve_adapter("WebShop", adapters), WebShopAdapter)
    assert isinstance(resolve_adapter("searchqa", adapters), SearchQaAdapter)
    assert resolve_adapter("mnist", adapters) is None          # unknown → None
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** — `search_qa.py`: move `_search_qa_encoder` (295–300) + `_default_search_qa_index_builder` (303–364); `provision(ctx)` = `ensure_search_qa_index` body (640–682); `applies(name)` matches the Search-QA alias set (458–459). `registry.py`: `resolve_adapter` returns the first adapter whose `applies(name)` is True (case-insensitive), else `None`.

- [ ] **Step 4: Run to verify they pass** — Expected: **3 + 1 passed**.

---

### Task 7: Rewrite `env_cache.py` as the facade

**Files:**
- Modify: `backend/services/runtime/env_cache.py`

**Interfaces:**
- Consumes: `AssetCache`, the 3 adapters, `resolve_adapter`, `EnvSetupResult`, `_pid_alive`.
- Produces: the FROZEN public surface (unchanged): `EnvCacheManager`, `provision_scope`, `ProvisionResult`, `EnvSetupResult`, `default_cache_dir`, `FULL_SCOPE_ENV_GUIDANCE`, `_pid_alive`.

- [ ] **Step 1: Rewrite the module as a facade**

Keep verbatim: `FULL_SCOPE_ENV_GUIDANCE`, `_ALFWORLD/_WEBSHOP/_SEARCH_QA`, `ProvisionResult` + `provision_scope`. Re-export for back-compat:
```python
from backend.services.runtime.asset_cache import AssetCache, default_cache_dir, _pid_alive
from backend.services.runtime.env_adapters import (
    EnvSetupResult, EnvironmentAdapter, resolve_adapter,
    AlfworldAdapter, WebShopAdapter, SearchQaAdapter,
)
```
Rewrite `EnvCacheManager`:
```python
class EnvCacheManager:
    def __init__(self, cache_dir=None, *, downloader=None, server_launcher=None, probe=None,
                 index_builder=None, inprocess_smoke=None, webshop_port=3000,
                 server_ready_timeout_s=60.0, clock=time.monotonic) -> None:
        self._cache = AssetCache(cache_dir)
        self.cache_dir = self._cache.cache_dir                       # back-compat attr
        self._alfworld = AlfworldAdapter(self._cache, downloader=downloader, clock=clock)
        self._webshop = WebShopAdapter(
            self._cache, server_launcher=server_launcher, probe=probe,
            inprocess_smoke=inprocess_smoke, pid_alive=_pid_alive,    # resolved from module global → monkeypatch-honored
            webshop_port=webshop_port, server_ready_timeout_s=server_ready_timeout_s, clock=clock)
        self._search_qa = SearchQaAdapter(self._cache, index_builder=index_builder, clock=clock)
        self._adapters = (self._alfworld, self._webshop, self._search_qa)

    def setup(self, env: str) -> EnvSetupResult:
        a = resolve_adapter(env, self._adapters)
        if a is None:
            return EnvSetupResult(env=env or "", ok=True, detail="no environment to provision")
        return a.provision(ProvisionCtx(display_name=env or a.key))

    def ensure_alfworld(self, *, display_name=_ALFWORLD):
        return self._alfworld.provision(ProvisionCtx(display_name=display_name))
    def acquire_webshop(self, *, display_name=_WEBSHOP):
        return self._webshop.provision(ProvisionCtx(display_name=display_name))
    def release_webshop(self) -> None:
        self._webshop.release()
    def ensure_search_qa_index(self, *, display_name=_SEARCH_QA):
        return self._search_qa.provision(ProvisionCtx(display_name=display_name))
```
**Note on `pid_alive=_pid_alive`:** `_pid_alive` is looked up in `env_cache`'s module namespace when `__init__` runs. Every existing test patches `EC._pid_alive` BEFORE constructing the manager, so the adapter receives the patched callable (which closes over the test's mutable `alive` dict where relevant). This is why the 7 lifecycle tests pass unchanged.

- [ ] **Step 2: Run the FULL existing suite unchanged**

Run: `.venv/bin/python -m pytest tests/services/runtime/test_env_cache.py tests/services/runtime/test_env_cache_dense.py tests/services/runtime/test_asset_provisioning.py tests/services/runtime/test_env_provisioning_characterization.py tests/services/runtime/env_adapters/ -q`
Expected: **all pass** (45 existing + 5 characterization + adapter/cache/registry units), **zero edits** to the three existing files.

- [ ] **Step 3: Grep-verify the SDAR caller imports still resolve**

Run: `.venv/bin/python -c "from backend.services.runtime.env_cache import EnvCacheManager, provision_scope, EnvSetupResult, ProvisionResult, default_cache_dir, FULL_SCOPE_ENV_GUIDANCE; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Lint** — `uvx ruff@0.15.16 check backend/services/runtime/asset_cache.py backend/services/runtime/env_adapters/ backend/services/runtime/env_cache.py` → clean.

- [ ] **Step 5: Phase 1a milestone commit** (after operator review of the diff)

```bash
git add backend/services/runtime/asset_cache.py backend/services/runtime/env_adapters/ \
        backend/services/runtime/env_cache.py tests/services/runtime/
git commit -m "Lift SDAR env provisioning behind an EnvironmentAdapter seam + generalized AssetCache (Phase 1a)"
```

---

## PHASE 1b — Deterministic gates (unwired, hermetic)

### File structure (1b)

| File | New/Modify | Responsibility |
|---|---|---|
| `backend/agents/resilience/budget.py` | **Modify** | Add `RunBudget.max_gpu_hours` field + `check_gpu_hours()`. |
| `backend/services/runtime/run_plan.py` | **Create** | `RequiredAsset`, `RunPlan`, `extract_required_assets(...)` (semantic-contract path + claim-map/rubric fallback). |
| `backend/services/runtime/feasibility_triage.py` | **Create** | `est_train_seconds`, `estimate_scope_cost`, `TriageDecision`, `FeasibilityTriage`. |
| `tests/agents/resilience/test_budget_gpu_hours.py` | **Create** | `max_gpu_hours` gate tests. |
| `tests/services/runtime/test_run_plan.py` | **Create** | required-assets extraction (3 sources). |
| `tests/services/runtime/test_feasibility_triage.py` | **Create** | cost model + 3-axis triage decisions. |

### Interfaces (1b)

```python
# backend/agents/resilience/budget.py  (extend RunBudget)
max_gpu_hours: float | None = None
def check_gpu_hours(self, *, gpu_hours_used: float, agent_id: str) -> None: ...   # raises BudgetExhausted

# backend/services/runtime/run_plan.py
@dataclass(frozen=True)
class RequiredAsset:
    kind: str            # "dataset" | "weights" | "image" | "service" | "framework"
    identifier: str
    gated: bool = False  # known to need a credential (best-effort hint)
    size_hint_gb: float | None = None

@dataclass(frozen=True)
class RunPlan:
    paper_id: str = ""
    scope: "ScopeSpec | None" = None
    budget: "RunBudget | None" = None
    required_assets: tuple[RequiredAsset, ...] = ()

def extract_required_assets(*, contract=None, claim_map=None, rubric=None) -> list[RequiredAsset]: ...

# backend/services/runtime/feasibility_triage.py
def est_train_seconds(model_key: str, steps: int) -> float: ...          # conservative, deterministic
def estimate_scope_cost(scope, sku, *, steps: int, overhead: float = 2.0) -> tuple[float, float]: ...  # (gpu_hours, usd)

@dataclass(frozen=True)
class TriageDecision:
    decision: str                      # "PROCEED" | "DOWN_SCOPE" | "PLAN_ONLY"
    scope: "ScopeSpec | None"
    reasons: tuple[str, ...]
    est_gpu_hours: float
    est_usd: float

class FeasibilityTriage:
    def __init__(self, *, reachability_probe=None, adapters=None) -> None: ...
    def triage(self, plan: RunPlan, sku) -> TriageDecision: ...
```

---

### Task 8: `RunBudget.max_gpu_hours`

**Files:**
- Modify: `backend/agents/resilience/budget.py` (add field after `max_run_gpu_usd`, line ~19; add method after `check_run_gpu_usd`, line ~97)
- Test: `tests/agents/resilience/test_budget_gpu_hours.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from backend.agents.resilience.budget import RunBudget, BudgetExhausted


def test_gpu_hours_within_budget_passes():
    RunBudget(max_gpu_hours=10.0).check_gpu_hours(gpu_hours_used=4.0, agent_id="root")   # no raise


def test_gpu_hours_over_budget_raises():
    with pytest.raises(BudgetExhausted):
        RunBudget(max_gpu_hours=2.0).check_gpu_hours(gpu_hours_used=2.5, agent_id="root")


def test_gpu_hours_none_is_unbounded():
    RunBudget().check_gpu_hours(gpu_hours_used=1_000.0, agent_id="root")                 # no raise


def test_gpu_hours_zero_disables():
    RunBudget(max_gpu_hours=0.0).check_gpu_hours(gpu_hours_used=1_000.0, agent_id="root")  # 0 == disabled
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — add `max_gpu_hours: float | None = None` to the frozen dataclass (after `max_run_gpu_usd`, line ~19); add the method **matching the exact idiom of `check_run_gpu_usd`** (disable-guard `is None or <= 0`, `>=` comparison, `BudgetExhausted(msg, provider=None, agent_id=agent_id)`):
```python
def check_gpu_hours(self, *, gpu_hours_used: float, agent_id: str) -> None:
    """Raise BudgetExhausted when cumulative GPU-hours >= max_gpu_hours.

    Cap honored only when set and > 0; None or 0 disables (mirrors check_run_gpu_usd).
    """
    if self.max_gpu_hours is None or self.max_gpu_hours <= 0:
        return
    if gpu_hours_used >= self.max_gpu_hours:
        raise BudgetExhausted(
            f"Run GPU-hours budget exhausted before invoking {agent_id}: "
            f"{gpu_hours_used:.2f}h >= {self.max_gpu_hours:.2f}h",
            provider=None,
            agent_id=agent_id,
        )
```

- [ ] **Step 4: Run to verify it passes** — Expected: **4 passed**.

---

### Task 9: `RunPlan.required_assets` extraction

**Files:**
- Create: `backend/services/runtime/run_plan.py`
- Test: `tests/services/runtime/test_run_plan.py`

**Interfaces:**
- Consumes: `SemanticReproductionContract` (`resource_identities`, `capability_profile`), `PaperClaimMap` (`datasets`, `model_architecture`, `hardware_clues`, `training_recipe`), the rubric dict (leaf `requirements` text — dataset/model mention scan, best-effort).
- Produces: `RequiredAsset`, `RunPlan`, `extract_required_assets`.

**Behavior:** contract wins when present; else fall back to claim-map; else scan rubric leaf text. Dedupe by `(kind, identifier.lower())`. Never raise — bad input → `[]`.

- [ ] **Step 1: Write the failing test**

```python
from backend.services.runtime.run_plan import RequiredAsset, extract_required_assets
from backend.agents.rlm.semantic_contract import (
    SemanticReproductionContract, ResourceIdentity, CapabilityProfile,
)
from backend.agents.schemas import PaperClaimMap, DatasetRequirement


def test_from_contract_resource_identities_and_capabilities():
    c = SemanticReproductionContract(
        resource_identities=[ResourceIdentity(kind="dataset", identifier="alfworld"),
                             ResourceIdentity(kind="weights", identifier="Qwen/Qwen3-1.7B")],
        capability_profile=CapabilityProfile(datasets=["search-qa"], frameworks=["pytorch"],
                                             external_services=["webshop-server"]))
    got = extract_required_assets(contract=c)
    kinds = {(a.kind, a.identifier) for a in got}
    assert ("dataset", "alfworld") in kinds
    assert ("weights", "Qwen/Qwen3-1.7B") in kinds
    assert ("dataset", "search-qa") in kinds
    assert ("framework", "pytorch") in kinds
    assert ("service", "webshop-server") in kinds


def test_fallback_to_claim_map_when_no_contract():
    cm = PaperClaimMap(core_contribution="x",
                       datasets=[DatasetRequirement(name="CIFAR-10")],
                       model_architecture="ResNet-18", hardware_clues=["1x A100"])
    got = extract_required_assets(claim_map=cm)
    ids = {(a.kind, a.identifier) for a in got}
    assert ("dataset", "CIFAR-10") in ids
    assert ("weights", "ResNet-18") in ids


def test_rubric_fallback_scans_leaf_text():
    rubric = {"children": [{"requirements": "Train on the IMDB dataset with PyTorch", "weight": 1.0}]}
    got = extract_required_assets(rubric=rubric)
    assert any(a.kind == "dataset" and a.identifier.lower() == "imdb" for a in got)


def test_dedupe_and_never_raises():
    assert extract_required_assets() == []
    c = SemanticReproductionContract(
        resource_identities=[ResourceIdentity(kind="dataset", identifier="alfworld"),
                             ResourceIdentity(kind="dataset", identifier="ALFWorld")])
    assert len(extract_required_assets(contract=c)) == 1     # case-insensitive dedupe
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `run_plan.py`** — `extract_required_assets`:
  1. If `contract` is not None: map `resource_identities[*]` → `RequiredAsset(kind=<kind>, identifier)`; `capability_profile.datasets` → `dataset`, `.frameworks` → `framework`, `.external_services` → `service`. (Map contract `kind` `"service"` verbatim; `"image"`→`image`.)
  2. Elif `claim_map` is not None: `datasets[*].name` → `dataset` (carry `gated = bool(download_method or source)` heuristic = False by default; leave `gated=False` unless `source` looks gated — keep simple: `False`); `model_architecture` → `weights` when non-empty; reuse `dataset_recipes.find_recipes_in_text` over `hardware_clues`? No — keep to declared fields only.
  3. Elif `rubric` is not None: flatten leaves (walk `children`/`requirements`), run `backend.agents.dataset_recipes.find_recipes_in_text(leaf_text)` → each recipe's `canonical_name` as a `dataset` asset; scan for framework keywords (`pytorch`/`tensorflow`/`jax`) → `framework`. Best-effort.
  4. Dedupe by `(kind, identifier.casefold())`, preserving first-seen order. Wrap the whole body in a `try/except Exception: return []` guard (fail-soft).

- [ ] **Step 4: Run to verify it passes** — Expected: **4 passed**.

---

### Task 10: cost model — `est_train_seconds` + `estimate_scope_cost`

**Files:**
- Create: `backend/services/runtime/feasibility_triage.py` (part 1)
- Test: `tests/services/runtime/test_feasibility_triage.py` (part 1)

**Interfaces:**
- Consumes: `GpuSku` (`approx_usd_per_hr`, `gpu_count`), `ScopeSpec` (`models`, `datasets`, `seeds`), `backend.services.pricing.paper_features` (model-size classification — reference, do not duplicate).
- Produces: `est_train_seconds`, `estimate_scope_cost`.

**DRY note:** Do NOT duplicate `pricing/estimator.py::estimate_paper_budget` (async, LLM). This is a *deterministic, no-LLM, per-scope-cell* estimate for the pre-lease gate; the live `WATCH` budget check is the true backstop, so it need only be **conservative (over-estimate, never under)**. Reuse the model-size → seconds/step intuition from a small explicit table; classify model size from the model key via a tiny helper (mirror `paper_features` size buckets: tiny/small/medium/large).

- [ ] **Step 1: Write the failing test**

```python
from backend.services.runtime.feasibility_triage import est_train_seconds, estimate_scope_cost
from backend.services.runtime.gpu_catalog import find_by_alias
from backend.agents.schemas import ScopeSpec


def test_est_train_seconds_monotonic_in_steps_and_size():
    assert est_train_seconds("qwen3-1.7b", 400) < est_train_seconds("qwen3-1.7b", 800)
    assert est_train_seconds("qwen3-1.7b", 400) < est_train_seconds("qwen2.5-7b", 400)
    assert est_train_seconds("unknown-model", 100) > 0.0          # conservative default, never 0


def test_estimate_scope_cost_scales_with_cells():
    sku = find_by_alias("rtx4090")                                # approx_usd_per_hr known
    small = ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0])
    big = ScopeSpec(models=["qwen3-1.7b", "qwen2.5-3b"],
                    datasets=[{"name": "alfworld"}, {"name": "webshop"}], seeds=[0, 1])
    gh_s, usd_s = estimate_scope_cost(small, sku, steps=400)
    gh_b, usd_b = estimate_scope_cost(big, sku, steps=400)
    assert gh_b > gh_s and usd_b > usd_s
    assert usd_s == round(gh_s * sku.approx_usd_per_hr, 4)         # usd derives from gpu_hours × $/hr
    # 8 cells (2×2×2) vs 1 cell → ~8× the compute
    assert gh_b > 6 * gh_s


def test_estimate_scope_cost_empty_scope_is_zero():
    sku = find_by_alias("rtx4090")
    assert estimate_scope_cost(ScopeSpec(), sku, steps=400) == (0.0, 0.0)
```

*(If `find_by_alias` is not exported, use `gpu_catalog.CATALOG[0]` — confirm the alias helper name at implementation time; `gpu_catalog.__all__` lists `find_by_alias`.)*

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement part 1** —
```python
_SECONDS_PER_STEP = {"tiny": 0.5, "small": 1.5, "medium": 3.0, "large": 6.0}   # conservative

def _model_size_bucket(model_key: str) -> str:
    k = model_key.lower()
    # parse a "<n>b" parameter hint; fall back to 'medium' (conservative) when unknown
    ...

def est_train_seconds(model_key: str, steps: int) -> float:
    return max(1, int(steps)) * _SECONDS_PER_STEP.get(_model_size_bucket(model_key), 3.0)

def estimate_scope_cost(scope, sku, *, steps: int, overhead: float = 2.0):
    models = scope.models or []
    datasets = scope.dataset_ids() if scope.datasets else []
    seeds = scope.seeds or [0]
    if not models or not datasets:
        return (0.0, 0.0)
    total_s = 0.0
    for m in models:
        total_s += est_train_seconds(m, steps) * len(datasets) * len(seeds)
    gpu_hours = round(total_s * overhead / 3600.0, 4)
    usd = round(gpu_hours * float(sku.approx_usd_per_hr), 4)
    return (gpu_hours, usd)
```
The unknown-model bucket defaults to `medium` (conservative). `overhead` mirrors `OPENRESEARCH_ESTIMATE_OVERHEAD_MULTIPLIER` default 2.0.

- [ ] **Step 4: Run to verify it passes** — Expected: **3 passed**.

---

### Task 11: `FeasibilityTriage` (3-axis decision)

**Files:**
- Modify: `backend/services/runtime/feasibility_triage.py` (part 2)
- Test: `tests/services/runtime/test_feasibility_triage.py` (part 2)

**Interfaces:**
- Consumes: `RunPlan` (`required_assets`, `scope`, `budget`), `GpuSku`, an injected `reachability_probe: Callable[[RequiredAsset], str]` returning `"reachable"|"gated"|"missing"`, the adapter list for env stand-ability.
- Produces: `TriageDecision`, `FeasibilityTriage`.

**Three axes → combined decision (deterministic, no LLM, no network in tests):**
1. **Data reachability** — `reachability_probe(asset)` per dataset/weights asset. `missing` (non-excludable) blocks; `gated` is a blocking gap in 1b (CredentialBroker is 1d) → contributes to PLAN_ONLY; `reachable` ok.
2. **Compute feasibility** — `estimate_scope_cost(scope, sku, steps)` vs `budget.max_gpu_hours`/`max_run_gpu_usd`. Within → PROCEED; over-but-trimmable → DOWN_SCOPE (drop to the single smallest model / fewest seeds that fits); infeasible even minimal → PLAN_ONLY.
3. **Env stand-ability** — for each dataset asset, `resolve_adapter(name, adapters)` present → adapter; else generic-resolvable (default assume yes in 1b) → generic.

Combine: any blocking gap (a `missing` asset that can't be excluded, or minimal-scope compute ≫ budget, or a `gated` asset with no cred) → **PLAN_ONLY**. Else if compute over budget but trimmable → **DOWN_SCOPE** (return the trimmed `ScopeSpec`). Else **PROCEED**.

- [ ] **Step 1: Write the failing test**

```python
from backend.services.runtime.feasibility_triage import FeasibilityTriage, TriageDecision
from backend.services.runtime.run_plan import RunPlan, RequiredAsset
from backend.services.runtime.gpu_catalog import find_by_alias
from backend.agents.resilience.budget import RunBudget
from backend.agents.schemas import ScopeSpec

_SKU = find_by_alias("rtx4090")


def test_proceed_when_reachable_and_within_budget():
    plan = RunPlan(scope=ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   budget=RunBudget(max_gpu_hours=100.0),
                   required_assets=(RequiredAsset("dataset", "alfworld"),))
    t = FeasibilityTriage(reachability_probe=lambda a: "reachable")
    d = t.triage(plan, _SKU)
    assert d.decision == "PROCEED"


def test_plan_only_when_asset_gated_no_cred():
    plan = RunPlan(scope=ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "gated-ds"}], seeds=[0]),
                   budget=RunBudget(max_gpu_hours=100.0),
                   required_assets=(RequiredAsset("dataset", "gated-ds", gated=True),))
    t = FeasibilityTriage(reachability_probe=lambda a: "gated")
    d = t.triage(plan, _SKU)
    assert d.decision == "PLAN_ONLY" and any("gated" in r for r in d.reasons)


def test_down_scope_when_over_budget_but_trimmable():
    plan = RunPlan(
        scope=ScopeSpec(models=["qwen2.5-7b", "qwen3-1.7b"],
                        datasets=[{"name": "alfworld"}, {"name": "webshop"}], seeds=[0, 1, 2]),
        budget=RunBudget(max_gpu_hours=0.5),          # tight → must trim
        required_assets=(RequiredAsset("dataset", "alfworld"), RequiredAsset("dataset", "webshop")))
    t = FeasibilityTriage(reachability_probe=lambda a: "reachable")
    d = t.triage(plan, _SKU)
    assert d.decision == "DOWN_SCOPE"
    assert len(d.scope.models) < 2 or len(d.scope.seeds) < 3       # trimmed to fit


def test_plan_only_when_infeasible_even_minimal():
    plan = RunPlan(scope=ScopeSpec(models=["qwen2.5-7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   budget=RunBudget(max_gpu_hours=0.0001),          # nothing fits
                   required_assets=(RequiredAsset("dataset", "alfworld"),))
    t = FeasibilityTriage(reachability_probe=lambda a: "reachable")
    d = t.triage(plan, _SKU)
    assert d.decision == "PLAN_ONLY"
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement part 2** — `FeasibilityTriage.triage`:
  - Probe each `dataset`/`weights` asset; collect `gated`/`missing` blockers.
  - Compute `(gh, usd) = estimate_scope_cost(scope, sku, steps=<default 400>)`.
  - `within = budget is None or ((not budget.max_gpu_hours or gh <= budget.max_gpu_hours) and (not budget.max_run_gpu_usd or usd <= budget.max_run_gpu_usd))`.
  - If any hard blocker → `PLAN_ONLY`.
  - Elif `within` → `PROCEED`.
  - Else try `_trim_scope(scope)` (single smallest model by `_model_size_bucket`, one seed, reachable datasets) and re-estimate; if the trimmed scope fits → `DOWN_SCOPE` with the trimmed `ScopeSpec`; else `PLAN_ONLY`.
  - Always populate `reasons`, `est_gpu_hours`, `est_usd`. Fail-soft: any exception → `PLAN_ONLY` with reason `"triage_error"`.

- [ ] **Step 4: Run to verify it passes** — Expected: **4 passed**.

- [ ] **Step 5: Run the full 1b suite + lint**

Run: `.venv/bin/python -m pytest tests/agents/resilience/test_budget_gpu_hours.py tests/services/runtime/test_run_plan.py tests/services/runtime/test_feasibility_triage.py -q`
Then: `uvx ruff@0.15.16 check backend/agents/resilience/budget.py backend/services/runtime/run_plan.py backend/services/runtime/feasibility_triage.py`
Expected: all pass, lint clean.

- [ ] **Step 6: Phase 1b milestone commit** (after operator review)

```bash
git add backend/agents/resilience/budget.py backend/services/runtime/run_plan.py \
        backend/services/runtime/feasibility_triage.py tests/
git commit -m "Add deterministic pre-lease feasibility gates: RunBudget.max_gpu_hours, RunPlan.required_assets, FeasibilityTriage + estimate_scope_cost (Phase 1b)"
```

---

## Validation (both phases)

- [ ] **Regression sweep** — run the broader runtime + resilience + rlm suites to catch any import-time breakage from the env_cache rewrite:

Run: `.venv/bin/python -m pytest tests/services/runtime/ tests/agents/resilience/ -q -x`
Expected: all pass (no collateral).

- [ ] **Import smoke of the live callers** — the modules that import `env_cache` must still import cleanly:

Run: `.venv/bin/python -c "import backend.agents.rlm.run, backend.cli; print('callers import ok')"`
Expected: `callers import ok`.

- [ ] **Docs** — update `CLAUDE.md` (the "Where to look first" + a one-line rule for the new provisioning seam + the 1b gate modules) and the `multicloud-reproduction-refactor` memory (mark Phase 1a + 1b DONE with the module map). Keep incident narratives out of `CLAUDE.md` — only the resulting rule.

---

## Self-Review (checked against the spec)

**Spec coverage:**
- §6.2 `EnvironmentAdapter` (applies/provision/smoke/health) + SDAR's 3 adapters behavior-preserving → Tasks 3–6. ✓
- §6.2 `AssetCache` = `EnvCacheManager` generalized, host-shared, fcntl-locked, keyed by identity → Task 2. ✓
- §9/§10 characterization-first, zero-regression, "not merely re-point the tests" → Task 1 (DI contract) + the 45 existing tests kept **unchanged** (Task 7). ✓
- §5.4/§13 `RunBudget.max_gpu_hours` → Task 8. ✓
- §6.1/§13 `RunPlan.required_assets` from `SemanticReproductionContract`, fallback rubric+claim-map, "never assume the contract is present" → Task 9. ✓
- §6.1 `FeasibilityTriage` (3 axes) + PROCEED/DOWN_SCOPE/PLAN_ONLY, no LLM, CredentialBroker-uses-deferred-to-1d (injected probe) → Task 11. ✓
- §6.1 `estimate_scope_cost`/`est_train_seconds` (new, conservative) → Task 10. ✓
- §8 fail-soft two-regimes; new work unwired ⇒ byte-identical live path → 1b modules are standalone; Task 7 preserves the facade. ✓

**Deferred to later phases (correctly out of scope here):** `smoke`/`health` are defined + unit-tested but NOT wired into a live GREEN_GATE (that is 1c's `ReproductionRun`); the reachability probe is injected/faked (real HEAD/HF-metadata + `CredentialBroker` are 1d); `FeasibilityTriage` is not called from `run.py` (wiring is 1c). This is intentional per the spec's phase ordering (gates before GPU, but the state machine that calls them is 1c).

**Type consistency:** `EnvSetupResult` defined once (base.py), re-exported by env_cache. `RequiredAsset`/`RunPlan` names match between Task 9 (produce) and Task 11 (consume). `estimate_scope_cost` returns `(gpu_hours, usd)` in Tasks 10 + 11. `TriageDecision.decision` string literals identical across test + impl.

**Placeholder scan:** none — every step has runnable test code + concrete move-source line ranges.
