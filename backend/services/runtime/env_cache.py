"""EnvCacheManager — host-shared, crash-safe cache for heavy RL environments.

Part B of full-scope-envs (2026-06-01). The SDAR full scope needs two expensive
environments that a Search-QA-only run skips: **ALFWorld** (a multi-GB one-time
``alfworld-download``) and **WebShop** (a single indexed server process). Doing
either per-run — or worse, per-cell — wastes minutes and disk and is flaky. This
manager makes both:

* **idempotent + host-shared** — ALFWorld is downloaded ONCE into a shared cache
  dir (``REPROLAB_ENV_CACHE_DIR``, default ``<runs_root>/.cache/envs``) and reused
  by every later run/cell;
* **ref-counted** — ONE WebShop server backs N concurrent leases and is torn down
  only when the last lease releases;
* **crash-safe** — an ``fcntl``-locked state file with stale-server reclaim by PID
  liveness, mirroring ``backend/services/runtime/local_gpu_allocator.py``;
* **fail-soft into the rubric** — a setup that cannot complete on this host returns
  a VERIFIED ``env_setup_failed`` :class:`~backend.agents.rlm.exclusion.Exclusion`
  (NOT an exception), so the grid runs the environments that work and the rubric
  EXCLUDES (numerator AND denominator) the rest. This is the fairness principle
  (2026-06-01): never dock the rubric for an environment the harness could not
  stand up. The verified Exclusion flows through ``exclusion.build_scope_block``
  into ``metrics.json::scope`` and is honoured by the leaf scorer (self-sufficient
  since the Part A review follow-up — a verified ``scope.exclusions`` record
  excludes its leaves even without a co-populated ``environments_skipped``).

INTEGRATION STATUS (design doc §5): the cache / lock / ref-count / Exclusion logic
is unit-tested with the downloader subprocess and the health probe INJECTED. The
real ``alfworld-download`` invocation and the WebShop server bring-up are NOT
end-to-end verified in this session (the shared box has no spare GPU and a live
run holds it). The injected ``downloader`` / ``server_launcher`` / ``probe`` seams
are exactly where a clean-host integration test plugs real commands in.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from backend.agents.rlm.exclusion import (
    AXIS_ENVIRONMENT,
    KIND_ENV_SETUP_FAILED,
    Exclusion,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EnvSetupResult",
    "EnvCacheManager",
    "ProvisionResult",
    "provision_scope",
    "default_cache_dir",
    "FULL_SCOPE_ENV_GUIDANCE",
]

# Guidance appended to REPROLAB_BASELINE_EXTRA_GUIDANCE (by backend/cli.py) ONLY
# when the effective scope keeps ALFWorld/WebShop active — tells the agent to
# write a BaseEnv subclass per environment + add their cells, and to consume the
# cache locations the EnvCacheManager exports rather than re-provisioning.
FULL_SCOPE_ENV_GUIDANCE = (
    "[full-scope envs] This run's scope includes {envs}. For EACH of those "
    "environments: write a concrete `*Env` class (e.g. `ALFWorldEnv`, `WebShopEnv`) "
    "in code/sdar/ that subclasses `sdar_env_base.BaseEnv` and implements both "
    "`build_student_prompt` and `build_teacher_prompt`, and add its cells to "
    "code/cells.json (one cell per model x baseline x seed, with the env's name in "
    "the `env` field). Load ALFWorld episodes from the directory in the "
    "`ALFWORLD_DATA` environment variable and reach the WebShop server at the URL "
    "in the `WEBSHOP_URL` environment variable — both are provided by the "
    "host-shared environment cache, so do NOT re-download ALFWorld or start your "
    "own WebShop server. If an environment's data or server is genuinely "
    "unavailable at runtime, record it as a scope gap (do not crash the grid)."
)

# Environments this manager knows how to stand up. "Search-QA" needs no special
# environment (it is a dataset loaded by the cell trainer), so it is always "ok"
# with nothing to provision — listed so the caller can treat all axes uniformly.
_ALFWORLD = "ALFWorld"
_WEBSHOP = "WebShop"


def default_cache_dir() -> Path:
    """Resolve the shared env-cache dir from ``REPROLAB_ENV_CACHE_DIR`` or default."""
    override = os.environ.get("REPROLAB_ENV_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    runs_root = os.environ.get("REPROLAB_RUNS_ROOT", "").strip() or "runs"
    return (Path(runs_root) / ".cache" / "envs").resolve()


@dataclass
class EnvSetupResult:
    """Outcome of provisioning one environment.

    Exactly one of (``ok=True`` with a path/url) or (``ok=False`` with
    ``exclusion``) holds. ``data_path`` is set for ALFWorld, ``base_url`` for
    WebShop; ``Search-QA`` returns ``ok=True`` with neither (nothing to provision).
    """

    env: str
    ok: bool
    data_path: str | None = None
    base_url: str | None = None
    exclusion: Exclusion | None = None
    detail: str = ""

    def as_env_vars(self) -> dict[str, str]:
        """Cache locations to splice into a child run's environment (empty on fail)."""
        out: dict[str, str] = {}
        if self.ok and self.data_path:
            out["ALFWORLD_DATA"] = self.data_path
        if self.ok and self.base_url:
            out["WEBSHOP_URL"] = self.base_url
        return out


def _pid_alive(pid: int) -> bool:
    """True iff ``pid`` is a live process (signal 0 probe). Mirrors the GPU allocator."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return True


def _default_alfworld_downloader(cache_dir: Path) -> None:
    """Run ``alfworld-download`` into ``cache_dir`` (real path; injected in tests)."""
    import subprocess  # local import: only the real path needs it

    env = {**os.environ, "ALFWORLD_DATA": str(cache_dir)}
    subprocess.run(["alfworld-download"], check=True, env=env, timeout=3600)


def _default_webshop_launcher(cache_dir: Path, port: int) -> int:
    """Start the WebShop server, return its PID (real path; injected in tests)."""
    import subprocess

    log = open(cache_dir / "webshop_server.log", "ab")  # noqa: SIM115 — child owns it
    proc = subprocess.Popen(
        ["python", "-m", "web_agent_site.app", "--port", str(port)],
        cwd=str(cache_dir), stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    return proc.pid


def _default_probe(url: str, *, timeout_s: float = 2.0) -> bool:
    """HTTP liveness probe (real path; injected in tests)."""
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            return 200 <= int(getattr(resp, "status", 0) or 0) < 500
    except Exception:  # noqa: BLE001
        return False


class EnvCacheManager:
    """Idempotent, fcntl-locked, ref-counted cache for ALFWorld + WebShop.

    All side-effecting operations (download, server launch, health probe) are
    injected callables, so the entire lifecycle is unit-testable without touching
    the network, a multi-GB download, or a real server. Every public method is
    fail-soft: a provisioning error becomes an :class:`EnvSetupResult` carrying a
    verified ``env_setup_failed`` :class:`Exclusion`, never a raised exception.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        downloader: Callable[[Path], None] | None = None,
        server_launcher: Callable[[Path, int], int] | None = None,
        probe: Callable[[str], bool] | None = None,
        webshop_port: int = 3000,
        server_ready_timeout_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.cache_dir / "env_cache_state.json"
        self._lock_path = self.cache_dir / ".env_cache.lock"
        self._downloader = downloader or _default_alfworld_downloader
        self._launcher = server_launcher or _default_webshop_launcher
        self._probe = probe or _default_probe
        self._webshop_port = int(webshop_port)
        self._ready_timeout_s = float(server_ready_timeout_s)
        self._clock = clock

    # --- locked state I/O (mirrors local_gpu_allocator's fcntl discipline) ----

    @contextlib.contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        """Yield the mutable state dict under an exclusive lock; persist on exit."""
        with open(self._lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state()
                yield state
                self._write_state(state)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _read_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, prefix=".env_cache_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, self._state_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @staticmethod
    def _fail(env: str, reason: str, evidence: str = "") -> EnvSetupResult:
        """Build a fail result carrying a VERIFIED env_setup_failed Exclusion."""
        return EnvSetupResult(
            env=env, ok=False, detail=reason,
            exclusion=Exclusion(
                item=env, axis=AXIS_ENVIRONMENT, kind=KIND_ENV_SETUP_FAILED,
                reason=reason, verified=True, evidence=evidence,
            ),
        )

    # --- public provisioning -------------------------------------------------

    def setup(self, env: str) -> EnvSetupResult:
        """Provision one environment by name (case-insensitive). Never raises."""
        key = (env or "").strip().lower()
        if key in ("alfworld", "alf world", "alf-world"):
            return self.ensure_alfworld(display_name=env or _ALFWORLD)
        if key in ("webshop", "web shop", "web-shop"):
            return self.acquire_webshop(display_name=env or _WEBSHOP)
        # Search-QA (and any dataset-only env): nothing to provision.
        return EnvSetupResult(env=env or "", ok=True, detail="no environment to provision")

    def ensure_alfworld(self, *, display_name: str = _ALFWORLD) -> EnvSetupResult:
        """Download ALFWorld once into the shared cache; reuse on later calls."""
        data_dir = self.cache_dir / "alfworld"
        try:
            with self._locked_state() as state:
                rec = state.get("alfworld") or {}
                if rec.get("ready") and Path(rec.get("data_path", "")).exists():
                    return EnvSetupResult(env=display_name, ok=True,
                                          data_path=rec["data_path"], detail="cache hit")
                data_dir.mkdir(parents=True, exist_ok=True)
                self._downloader(data_dir)   # injected; real path runs alfworld-download
                state["alfworld"] = {"ready": True, "data_path": str(data_dir),
                                     "downloaded_at": self._clock()}
                return EnvSetupResult(env=display_name, ok=True,
                                      data_path=str(data_dir), detail="downloaded")
        except Exception as exc:  # noqa: BLE001 — fail-soft into a verified Exclusion
            logger.warning("env_cache: ALFWorld setup failed: %s", exc)
            return self._fail(display_name, f"alfworld-download failed: {type(exc).__name__}: {exc}",
                              evidence=str(exc)[:200])

    def acquire_webshop(self, *, display_name: str = _WEBSHOP) -> EnvSetupResult:
        """Acquire a lease on the shared WebShop server (start it if needed)."""
        base_url = f"http://127.0.0.1:{self._webshop_port}"
        try:
            with self._locked_state() as state:
                rec = state.get("webshop") or {}
                pid = rec.get("pid")
                running = bool(rec.get("running")) and _pid_alive(int(pid)) if pid else False
                if not running:
                    # Stale/never-started → (re)launch and health-probe.
                    new_pid = self._launcher(self.cache_dir, self._webshop_port)
                    if not self._await_ready(base_url):
                        with contextlib.suppress(Exception):
                            os.kill(int(new_pid), signal.SIGTERM)
                        return self._fail(display_name,
                                          f"WebShop server did not become ready at {base_url}",
                                          evidence=base_url)
                    rec = {"pid": int(new_pid), "running": True, "url": base_url, "refcount": 0}
                rec["refcount"] = int(rec.get("refcount", 0)) + 1
                state["webshop"] = rec
                return EnvSetupResult(env=display_name, ok=True, base_url=base_url,
                                      detail=f"lease #{rec['refcount']}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("env_cache: WebShop setup failed: %s", exc)
            return self._fail(display_name, f"WebShop setup failed: {type(exc).__name__}: {exc}",
                              evidence=str(exc)[:200])

    def release_webshop(self) -> None:
        """Drop one WebShop lease; stop the server when the last lease releases."""
        try:
            with self._locked_state() as state:
                rec = state.get("webshop") or {}
                if not rec:
                    return
                rec["refcount"] = max(0, int(rec.get("refcount", 0)) - 1)
                if rec["refcount"] <= 0:
                    pid = rec.get("pid")
                    if pid and _pid_alive(int(pid)):
                        with contextlib.suppress(Exception):
                            os.kill(int(pid), signal.SIGTERM)
                    rec["running"] = False
                    rec["pid"] = None
                state["webshop"] = rec
        except Exception as exc:  # noqa: BLE001 — release must never raise
            logger.warning("env_cache: WebShop release failed (non-fatal): %s", exc)

    def _await_ready(self, url: str) -> bool:
        """Poll the health probe until ready or the ready-timeout elapses."""
        deadline = self._clock() + self._ready_timeout_s
        while self._clock() < deadline:
            if self._probe(url):
                return True
            time.sleep(0.5)
        return self._probe(url)  # one last chance


@dataclass
class ProvisionResult:
    """Outcome of provisioning a whole scope's worth of environments.

    ``env_vars`` are the cache locations to splice into the child run's
    environment (ALFWORLD_DATA / WEBSHOP_URL). ``exclusions`` are the VERIFIED
    ``env_setup_failed`` records for any env that could not be stood up on this
    host — feed them to ``exclusion.build_scope_block`` so the rubric excludes
    (not zeroes) those leaves. ``release()`` drops every WebShop lease acquired
    (a no-op for ALFWorld / Search-QA); call it in the run's ``finally``.
    """

    env_vars: dict[str, str] = field(default_factory=dict)
    exclusions: list[Exclusion] = field(default_factory=list)
    _release: Callable[[], None] = lambda: None

    def release(self) -> None:
        self._release()


def provision_scope(env_names: list[str], manager: EnvCacheManager) -> ProvisionResult:
    """Provision every environment in a scope; collect env-vars + failures.

    Each env is set up via :meth:`EnvCacheManager.setup`. Successes contribute
    their cache env-vars; failures contribute a verified ``env_setup_failed``
    Exclusion (never raise). WebShop leases are counted so ``release()`` drops
    exactly as many as were acquired. The caller injects ``env_vars`` into the
    child run and merges ``exclusions`` into ``metrics.json::scope`` via
    ``build_scope_block``.
    """
    env_vars: dict[str, str] = {}
    exclusions: list[Exclusion] = []
    webshop_leases = 0
    for name in env_names or []:
        res = manager.setup(name)
        if res.ok:
            env_vars.update(res.as_env_vars())
            if res.base_url:
                webshop_leases += 1
        elif res.exclusion is not None:
            exclusions.append(res.exclusion)

    def _release() -> None:
        for _ in range(webshop_leases):
            manager.release_webshop()

    return ProvisionResult(env_vars=env_vars, exclusions=exclusions, _release=_release)
