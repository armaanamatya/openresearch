"""WebShopAdapter — WebShop server / in-process provisioning (Phase 1a).

Part of the provisioning-seam refactor (see
``docs/history/plans/2026-07-01-phase-1a-1b-provisioning-seam-and-gates.md``):
lifts ``EnvCacheManager.acquire_webshop``/``release_webshop`` and their helper
functions out of ``env_cache.py`` verbatim, behind the
:class:`~backend.services.runtime.env_adapters.base.EnvironmentAdapter`
contract. ``env_cache.py`` itself is untouched by this unit; a later task
rewrites it as a facade delegating to this adapter.

The on-disk state key stays ``"webshop"`` (== :attr:`WebShopAdapter.key`) so a
warm SDAR cache disk is byte-compatible across the refactor.

**pid-liveness injection:** ``pid_alive`` defaults to
:func:`backend.services.runtime.asset_cache._pid_alive` and is ALWAYS called as
``self._pid_alive(pid)`` — never a module global — so a facade (a later task)
can thread a monkeypatched ``env_cache._pid_alive`` through construction. This
module uses plain ``import os`` / ``import signal`` / ``import time`` for
``os.kill`` / ``SIGTERM`` / ``time.sleep`` — those ARE module globals, and
tests patch ``env_adapters.webshop.os.kill`` / ``.time.sleep`` directly.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable

from backend.services.runtime.asset_cache import AssetCache, _pid_alive
from backend.services.runtime.env_adapters.base import (
    EnvironmentAdapter,
    EnvSetupResult,
    ProvisionCtx,
    SmokeResult,
    _fail,
)

logger = logging.getLogger(__name__)

__all__ = ["WebShopAdapter"]

_ALIASES = {"webshop", "web shop", "web-shop"}


def _default_webshop_launcher(cache_dir: Path, port: int) -> int:
    """Start the WebShop server, return its PID (real path; injected in tests)."""
    import subprocess

    log = open(cache_dir / "webshop_server.log", "ab")  # noqa: SIM115 — child owns it
    # Use the running interpreter, not a bare ``python`` (which need not be on the
    # cell's PATH — a venv invoked by path doesn't put its bin/ there). With the
    # correct interpreter, an un-installed ``web_agent_site`` surfaces as a clear
    # ModuleNotFoundError that ``provision`` turns into a verified
    # ``env_setup_failed`` Exclusion, instead of a misleading ``No such file: python``.
    #
    # OPENRESEARCH_WEBSHOP_PYTHON allows a dedicated Python 3.10 venv to be used for
    # WebShop's server, keeping its frozen old torch/transformers stack isolated from
    # the run venv's modern stack (set by install_webshop_dedicated in asset_provisioning).
    webshop_python = os.environ.get("OPENRESEARCH_WEBSHOP_PYTHON") or sys.executable
    proc = subprocess.Popen(
        [webshop_python, "-m", "web_agent_site.app", "--port", str(port)],
        cwd=str(cache_dir), stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    return proc.pid


def _default_inprocess_smoke(data_dir: str) -> bool:
    """Minimal in-process smoke: verify data files exist + web_agent_site is importable.

    This is the real path injected in production.  Tests inject a fast lambda
    so CI never needs web_agent_site installed.  Returns False on any failure
    (never raises) — the caller converts False into a verified Exclusion.
    """
    items = os.path.join(data_dir, "items_shuffle.json")
    attrs = os.path.join(data_dir, "items_ins_v2.json")
    if not (os.path.exists(items) and os.path.exists(attrs)):
        logger.debug("env_adapters.webshop: in-process smoke: missing data files under %s", data_dir)
        return False
    pkg_dir = os.environ.get("WEBSHOP_PACKAGE_DIR", "").strip()
    if pkg_dir:
        import sys as _sys  # noqa: PLC0415 — lazy
        if pkg_dir not in _sys.path:
            _sys.path.insert(0, pkg_dir)
    try:
        import web_agent_site  # noqa: F401, PLC0415 — intentional lazy import
        return True
    except ImportError:
        logger.debug("env_adapters.webshop: in-process smoke: web_agent_site not importable")
        return False


def _default_probe(url: str, *, timeout_s: float = 2.0) -> bool:
    """HTTP liveness probe (real path; injected in tests)."""
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            return 200 <= int(getattr(resp, "status", 0) or 0) < 500
    except Exception:  # noqa: BLE001
        return False


class WebShopAdapter(EnvironmentAdapter):
    """Ref-counted WebShop provisioning: in-process (preferred) or HTTP server.

    **In-process path (2026-06-27):** activated when ``WEBSHOP_DATA_DIR`` is set
    in the environment (operator pre-staged product corpus). On success sets
    ``WEBSHOP_DATA_DIR`` (NOT ``WEBSHOP_URL``) in ``env_vars``; ``WebShopEnv``
    detects this and constructs an in-process backend (zero sockets, no
    Java/Lucene).

    **Legacy HTTP path:** used when ``WEBSHOP_DATA_DIR`` is unset. Starts (or
    reuses) a shared ``web_agent_site.app`` server process; ref-counted
    (:meth:`release` tears it down when the last lease drops). Sets
    ``WEBSHOP_URL``.
    """

    key = "webshop"

    def __init__(
        self,
        cache: AssetCache,
        *,
        server_launcher: Callable[[Path, int], int] | None = None,
        probe: Callable[[str], bool] | None = None,
        inprocess_smoke: "Callable[[str], bool] | None" = None,
        pid_alive: Callable[[int], bool] | None = None,
        webshop_port: int = 3000,
        server_ready_timeout_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = cache
        self._launcher = server_launcher or _default_webshop_launcher
        self._probe = probe or _default_probe
        self._inprocess_smoke: Callable[[str], bool] = inprocess_smoke or _default_inprocess_smoke
        self._pid_alive = pid_alive or _pid_alive
        self._webshop_port = int(webshop_port)
        self._ready_timeout_s = float(server_ready_timeout_s)
        self._clock = clock

    def applies(self, env_name: str) -> bool:
        return (env_name or "").strip().lower() in _ALIASES

    def provision(self, ctx: ProvisionCtx) -> EnvSetupResult:
        """Acquire WebShop: in-process (preferred) or HTTP server (legacy).

        **In-process path:** activated when ``WEBSHOP_DATA_DIR`` is set in the
        environment (operator pre-staged product corpus) OR the SDAR ref
        package can be detected at the standard cache location AND
        ``WEBSHOP_DATA_DIR`` is set. On success sets ``WEBSHOP_DATA_DIR`` (NOT
        ``WEBSHOP_URL``) in ``env_vars``.

        **Legacy HTTP path:** used when ``WEBSHOP_DATA_DIR`` is unset. Starts
        (or reuses) a shared ``web_agent_site.app`` server process;
        ref-counted (:meth:`release` tears it down when the last lease
        drops). Sets ``WEBSHOP_URL`` as before.
        """
        display_name = ctx.display_name or "WebShop"
        # ---- In-process path ------------------------------------------------
        data_dir_env = os.environ.get("WEBSHOP_DATA_DIR", "").strip()
        if data_dir_env:
            # Resolve (or auto-detect) the package root alongside the data dir.
            pkg_dir = os.environ.get("WEBSHOP_PACKAGE_DIR", "").strip() or self._find_sdar_webshop_pkg()

            # Smoke test: data files present + web_agent_site importable.
            try:
                smoke_ok = self._inprocess_smoke(data_dir_env)
            except Exception as exc:  # noqa: BLE001 — smoke must never propagate
                smoke_ok = False
                logger.warning("env_adapters.webshop: in-process smoke raised: %s", exc)

            if not smoke_ok:
                return _fail(
                    display_name,
                    f"WebShop in-process smoke failed for WEBSHOP_DATA_DIR={data_dir_env!r}",
                    evidence=data_dir_env,
                )

            # Record state (best-effort — failure here is non-fatal).
            with contextlib.suppress(Exception):
                with self._cache.locked_state() as state:
                    state[self.key] = {
                        "ready": True, "inprocess": True,
                        "data_dir": data_dir_env,
                        **({"pkg_dir": pkg_dir} if pkg_dir else {}),
                    }

            env_vars: dict[str, str] = {"WEBSHOP_DATA_DIR": data_dir_env}
            if pkg_dir:
                env_vars["WEBSHOP_PACKAGE_DIR"] = pkg_dir
            num_prods = os.environ.get("WEBSHOP_NUM_PRODUCTS", "").strip()
            if num_prods:
                env_vars["WEBSHOP_NUM_PRODUCTS"] = num_prods
            return EnvSetupResult(
                env=display_name, ok=True,
                env_vars=env_vars,
                detail=f"in-process (data_dir={data_dir_env})",
            )

        # ---- Legacy HTTP server path (unchanged from original) ---------------
        base_url = f"http://127.0.0.1:{self._webshop_port}"
        try:
            with self._cache.locked_state() as state:
                rec = state.get(self.key) or {}
                pid = rec.get("pid")
                running = bool(rec.get("running")) and self._pid_alive(int(pid)) if pid else False
                if not running:
                    # Stale/never-started → (re)launch and health-probe.
                    new_pid = self._launcher(self._cache.cache_dir, self._webshop_port)
                    if not self._await_ready(base_url):
                        with contextlib.suppress(Exception):
                            os.kill(int(new_pid), signal.SIGTERM)
                        return _fail(display_name,
                                    f"WebShop server did not become ready at {base_url}",
                                    evidence=base_url)
                    rec = {"pid": int(new_pid), "running": True, "url": base_url, "refcount": 0}
                rec["refcount"] = int(rec.get("refcount", 0)) + 1
                state[self.key] = rec
                return EnvSetupResult(env=display_name, ok=True, base_url=base_url,
                                      detail=f"lease #{rec['refcount']}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("env_adapters.webshop: setup failed: %s", exc)
            return _fail(display_name, f"WebShop setup failed: {type(exc).__name__}: {exc}",
                        evidence=str(exc)[:200])

    def release(self) -> None:
        """Drop one WebShop lease; stop the server when the last lease releases."""
        try:
            with self._cache.locked_state() as state:
                rec = state.get(self.key) or {}
                if not rec:
                    return
                rec["refcount"] = max(0, int(rec.get("refcount", 0)) - 1)
                if rec["refcount"] <= 0:
                    pid = rec.get("pid")
                    if pid and self._pid_alive(int(pid)):
                        with contextlib.suppress(Exception):
                            os.kill(int(pid), signal.SIGTERM)
                    rec["running"] = False
                    rec["pid"] = None
                state[self.key] = rec
        except Exception as exc:  # noqa: BLE001 — release must never raise
            logger.warning("env_adapters.webshop: release failed (non-fatal): %s", exc)

    def smoke(self, ctx: ProvisionCtx) -> SmokeResult:
        """Cheap post-provision liveness: in-process smoke or an HTTP probe."""
        data_dir_env = os.environ.get("WEBSHOP_DATA_DIR", "").strip()
        if data_dir_env:
            try:
                ok = self._inprocess_smoke(data_dir_env)
            except Exception:  # noqa: BLE001
                ok = False
            return SmokeResult(ok=ok, detail="in-process" if ok else "in-process smoke failed")
        base_url = f"http://127.0.0.1:{self._webshop_port}"
        ok = self._probe(base_url)
        return SmokeResult(ok=ok, detail=base_url)

    def _find_sdar_webshop_pkg(self) -> str | None:
        """Return the ``web_agent_site`` package root from the SDAR ref, or ``None``.

        The SDAR ref is at ``<cache_dir>/../SDAR_ref`` (i.e. ``runs/.cache/SDAR_ref``).
        The package root — the directory whose child is ``web_agent_site/`` — is at
        ``SDAR_ref/agent_system/environments/env_package/webshop/webshop/``.
        """
        candidate = (
            self._cache.cache_dir.parent
            / "SDAR_ref"
            / "agent_system"
            / "environments"
            / "env_package"
            / "webshop"
            / "webshop"
        )
        if (candidate / "web_agent_site").is_dir():
            return str(candidate)
        return None

    def _await_ready(self, url: str) -> bool:
        """Poll the health probe until ready or the ready-timeout elapses."""
        deadline = self._clock() + self._ready_timeout_s
        while self._clock() < deadline:
            if self._probe(url):
                return True
            time.sleep(0.5)
        return self._probe(url)  # one last chance
