"""EnvCacheManager — host-shared, crash-safe cache for heavy RL environments.

Part B of full-scope-envs (2026-06-01), extended 2026-06-01 for the agentic
re-enablement of the SDAR full scope. The SDAR paper needs three environments a
Search-QA-only run skips: **ALFWorld** (a multi-GB one-time ``alfworld-download``),
**WebShop** (a single indexed server process), and a **dense Search-QA retriever**
(an E5 index over the wiki-18 corpus — large to build, pointless to rebuild). This
manager makes all three:

* **idempotent + host-shared** — ALFWorld data and the dense Search-QA index are
  built/downloaded ONCE into a shared cache dir (``REPROLAB_ENV_CACHE_DIR``,
  default ``<runs_root>/.cache/envs``) and reused by every later run/cell;
* **ref-counted** — ONE WebShop server backs N concurrent leases and is torn down
  only when the last lease releases;
* **crash-safe** — an ``fcntl``-locked state file with stale-server reclaim by PID
  liveness, mirroring ``backend/services/runtime/local_gpu_allocator.py``;
* **fail-soft into the rubric** — a setup that cannot complete on this host returns
  a VERIFIED ``env_setup_failed`` :class:`~backend.agents.rlm.exclusion.Exclusion`
  (NOT an exception) for ALFWorld/WebShop, so the grid runs the environments that
  work and the rubric EXCLUDES (numerator AND denominator) the rest. Search-QA
  never excludes: a cold/unavailable dense index degrades to BM25 (still real
  retrieval), so the environment always runs. This is the fairness principle
  (2026-06-01): never dock the rubric for an environment the harness could not
  stand up. The verified Exclusion flows through ``exclusion.build_scope_block``
  into ``metrics.json::scope`` and is honoured by the leaf scorer.

DENSE RETRIEVER (2026-06-01): the dense E5/wiki-18 path is **opt-in + configurable**
so a cold or offline host degrades to BM25 rather than blocking the grid on a
multi-GB build. ``REPROLAB_SEARCH_QA_DENSE`` must be truthy to attempt anything;
``REPROLAB_SEARCH_QA_INDEX_REPO`` names a HF repo holding a prebuilt FAISS index +
passage store (snapshot-downloaded, cached, reused). Absent either, Search-QA
provisions ``SEARCH_QA_RETRIEVER=bm25`` and the env's BM25/overlap retriever runs.

INTEGRATION STATUS: the cache / lock / ref-count / Exclusion logic is unit-tested
with the downloader subprocess, the health probe, and the index builder INJECTED.
The real ``alfworld-download`` invocation (now resolved by abs path next to the
interpreter), the WebShop server bring-up, and the dense index download are the
seams a clean-host integration test / a live run plug real commands into.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import signal
import sys
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

# Guidance appended to REPROLAB_BASELINE_EXTRA_GUIDANCE (by backend/cli.py) when the
# effective scope keeps the SDAR paper environments active. Tells the agent to use
# the SHIPPED concrete agentic env modules (copied into code/ as harness helpers)
# rather than re-implementing ALFWorld / WebShop / retrieval by hand, to consume the
# cache locations the EnvCacheManager exports, and to train at full depth.
FULL_SCOPE_ENV_GUIDANCE = (
    "[full-scope envs] This run's scope includes {envs}. These are REAL multi-turn "
    "agentic environments — do NOT fake them (no closed-book QA, no scripted stubs). "
    "The harness has copied ready-made, tested env modules into your code/ dir; "
    "import and use them rather than re-implementing:\n"
    "  • `from sdar_env_base import AgenticEnv, StepResult` — the multi-turn contract "
    "(reset()/step()/episode_reward() + transcript-rendering prompt builders).\n"
    "  • `from search_qa_env import SearchQAEnv, load_search_qa_tasks` — real retrieval "
    "QA (the model issues search(<q>) then answer(<a>)); it reads the cached dense E5 "
    "index from SEARCH_QA_INDEX_DIR when SEARCH_QA_RETRIEVER=e5, else BM25. It KEEPS "
    "HotpotQA contexts. Reward = token-F1.\n"
    "  • `from alfworld_env import ALFWorldEnv` — real ALFWorld TextWorld episodes "
    "loaded from the directory in the ALFWORLD_DATA env var.\n"
    "  • `from webshop_env import WebShopEnv` — real WebShop via the server at the "
    "WEBSHOP_URL env var.\n"
    "  • `from agentic_rollout import rollout_episode` — drives ONE multi-turn episode "
    "and returns a flat Trajectory(sequence_ids, response_mask, reward, info). Compute "
    "the GRPO advantage over a group of G such rollouts and the OPSD gate token-wise "
    "over the response_mask positions — do NOT hand-roll the turn→token-mask "
    "conversion.\n"
    "ALFWORLD_DATA / WEBSHOP_URL / SEARCH_QA_INDEX_DIR / SEARCH_QA_RETRIEVER are "
    "provided by the host-shared environment cache — consume them. The ALFWorld game "
    "data is ALREADY downloaded under $ALFWORLD_DATA; load games from there and do NOT "
    "run `alfworld-download` yourself (it is unnecessary and may not be on PATH). Do "
    "NOT start your own WebShop server or rebuild the index. Add one cell per "
    "(model × baseline × seed × env) to code/cells.json for EVERY environment in "
    "scope — you MUST include Search-QA AND ALFWorld (and WebShop when WEBSHOP_URL is "
    "set); a run that trains only one environment is incomplete. Put the env name in "
    "each cell's `env` field, and add the env deps your modules import to requirements.txt "
    "(rank_bm25, sentence-transformers, faiss-cpu, datasets, alfworld; requests is "
    "optional — webshop_env prefers stdlib urllib). Train at PAPER DEPTH, not "
    "smoke-test depth: STEPS >= 400, GROUP_SIZE = 8, and a token budget large enough "
    "for multi-turn rollouts (agentic episodes need many turns × tokens). If an "
    "environment's data or server is genuinely unavailable at runtime, record it as a "
    "scope gap (do NOT crash the grid) — the harness converts a verified-unavailable "
    "env into a rubric exclusion."
)

# Environments this manager knows how to stand up.
_ALFWORLD = "ALFWorld"
_WEBSHOP = "WebShop"
_SEARCH_QA = "Search-QA"


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

    Exactly one of (``ok=True`` with a path/url/env-vars) or (``ok=False`` with
    ``exclusion``) holds. ``data_path`` is set for ALFWorld, ``base_url`` for
    WebShop; ``Search-QA`` returns ``ok=True`` with ``env_vars`` carrying the
    retriever selection (``SEARCH_QA_INDEX_DIR`` + ``SEARCH_QA_RETRIEVER``) and no
    path/url. ``env_vars`` is a generic bag merged into the child environment by
    :meth:`as_env_vars` (alongside the ALFWorld/WebShop legacy keys).
    """

    env: str
    ok: bool
    data_path: str | None = None
    base_url: str | None = None
    exclusion: Exclusion | None = None
    detail: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)

    def as_env_vars(self) -> dict[str, str]:
        """Cache locations to splice into a child run's environment (empty on fail)."""
        if not self.ok:
            return {}
        out: dict[str, str] = dict(self.env_vars)
        if self.data_path:
            out["ALFWORLD_DATA"] = self.data_path
        if self.base_url:
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


def _resolve_console_script(name: str) -> str | None:
    """Resolve a venv console script (e.g. ``alfworld-download``) to an abs path.

    Console scripts install next to the interpreter (``<venv>/bin/<name>``) but
    that dir is not necessarily on a child process's PATH, so resolve by abs path
    first and fall back to a PATH lookup. Returns ``None`` if not found.
    """
    import shutil

    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


def _default_alfworld_downloader(cache_dir: Path) -> None:
    """Run ``alfworld-download`` into ``cache_dir`` (real path; injected in tests).

    ``ALFWORLD_DATA`` controls the download target. The console script is resolved
    by abs path (it may not be on the child's PATH); a missing script raises, which
    ``ensure_alfworld`` converts into a verified Exclusion.
    """
    import subprocess  # local import: only the real path needs it

    exe = _resolve_console_script("alfworld-download")
    if not exe:
        raise FileNotFoundError(
            "alfworld-download console script not found next to the interpreter "
            f"({Path(sys.executable).parent}) or on PATH"
        )
    env = {**os.environ, "ALFWORLD_DATA": str(cache_dir)}
    subprocess.run([exe], check=True, env=env, timeout=3600)


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


def _default_search_qa_index_builder(cache_dir: Path) -> Path | None:
    """Build/download a dense E5 wiki-18 retrieval index (real path; injected in tests).

    Returns the index dir on success, ``None`` to fall back to BM25 — NEVER raises.
    Opt-in + configurable so a cold/offline host degrades gracefully:

      * ``REPROLAB_SEARCH_QA_DENSE`` must be truthy to attempt anything;
      * ``REPROLAB_SEARCH_QA_INDEX_REPO`` — a HF repo holding a prebuilt FAISS index
        + passage store; snapshot-downloaded into ``cache_dir`` when set (fastest,
        no local embedding). ``REPROLAB_SEARCH_QA_INDEX_REPO_TYPE`` selects the HF
        repo type (default ``dataset``).

    The downloaded artifact is cached under ``cache_dir`` and reused by
    :meth:`EnvCacheManager.ensure_search_qa_index`. A local-embed path (download the
    corpus + embed with e5 on GPU) is intentionally left to a follow-up — the repo
    download keeps the common case fast and the BM25 fallback keeps every host live.
    """
    flag = os.environ.get("REPROLAB_SEARCH_QA_DENSE", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None
    repo = os.environ.get("REPROLAB_SEARCH_QA_INDEX_REPO", "").strip()
    if not repo:
        logger.info(
            "env_cache: REPROLAB_SEARCH_QA_DENSE set but REPROLAB_SEARCH_QA_INDEX_REPO "
            "is empty — using BM25 (set the repo to enable dense E5 retrieval)."
        )
        return None
    try:
        from huggingface_hub import snapshot_download  # lazy: only the real path needs it

        repo_type = os.environ.get("REPROLAB_SEARCH_QA_INDEX_REPO_TYPE", "dataset").strip() or "dataset"
        dest = cache_dir / "search_qa_index"
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo, repo_type=repo_type, local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        # Minimal sanity: a FAISS index file must exist for the env to load it.
        if any(dest.rglob("*.index")) or any(dest.rglob("*.faiss")):
            return dest
        logger.warning(
            "env_cache: search-qa index repo %s downloaded but no .index/.faiss file "
            "found — BM25 fallback.", repo,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — dense is best-effort; BM25 always works
        logger.warning(
            "env_cache: dense search-qa index build failed (%s: %s); BM25 fallback.",
            type(exc).__name__, str(exc)[:160],
        )
        return None


class EnvCacheManager:
    """Idempotent, fcntl-locked, ref-counted cache for ALFWorld + WebShop + Search-QA.

    All side-effecting operations (download, server launch, health probe, dense
    index build) are injected callables, so the entire lifecycle is unit-testable
    without touching the network, a multi-GB download, or a real server. Every
    public method is fail-soft: an ALFWorld/WebShop provisioning error becomes an
    :class:`EnvSetupResult` carrying a verified ``env_setup_failed``
    :class:`Exclusion`; a Search-QA dense-index failure degrades to BM25 (never an
    exclusion — the env always runs). Nothing raises.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        downloader: Callable[[Path], None] | None = None,
        server_launcher: Callable[[Path, int], int] | None = None,
        probe: Callable[[str], bool] | None = None,
        index_builder: Callable[[Path], "Path | None"] | None = None,
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
        self._index_builder = index_builder or _default_search_qa_index_builder
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
        if key in ("search-qa", "searchqa", "search_qa", "search qa",
                   "nq", "nq-open", "nq_open", "hotpotqa", "hotpot_qa"):
            return self.ensure_search_qa_index(display_name=env or _SEARCH_QA)
        # Any other dataset-only env: nothing to provision.
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

    def ensure_search_qa_index(self, *, display_name: str = _SEARCH_QA) -> EnvSetupResult:
        """Provide a Search-QA retriever: dense E5 index when buildable + cached,
        else BM25 (always works).

        Unlike ALFWorld/WebShop, Search-QA NEVER returns an exclusion — a cold or
        unavailable dense index degrades to ``SEARCH_QA_RETRIEVER=bm25`` and the
        env's BM25/overlap retriever runs. The dense build is idempotent + shared
        (cached under ``<cache>/search_qa_index``).
        """
        try:
            with self._locked_state() as state:
                rec = state.get("search_qa") or {}
                if (rec.get("ready") and rec.get("retriever") == "e5"
                        and Path(rec.get("index_dir", "")).exists()):
                    return EnvSetupResult(
                        env=display_name, ok=True, detail="cache hit (e5)",
                        env_vars={"SEARCH_QA_INDEX_DIR": rec["index_dir"],
                                  "SEARCH_QA_RETRIEVER": "e5"},
                    )
                built = self._index_builder(self.cache_dir)  # injected; None → BM25
                if built is not None and Path(built).exists():
                    state["search_qa"] = {"ready": True, "retriever": "e5",
                                          "index_dir": str(built), "built_at": self._clock()}
                    return EnvSetupResult(
                        env=display_name, ok=True, detail="dense index ready",
                        env_vars={"SEARCH_QA_INDEX_DIR": str(built),
                                  "SEARCH_QA_RETRIEVER": "e5"},
                    )
                state["search_qa"] = {"ready": True, "retriever": "bm25",
                                      "built_at": self._clock()}
                return EnvSetupResult(
                    env=display_name, ok=True, detail="bm25 (no dense index)",
                    env_vars={"SEARCH_QA_RETRIEVER": "bm25"},
                )
        except Exception as exc:  # noqa: BLE001 — Search-QA must always run
            logger.warning("env_cache: search-qa provisioning issue (%s); BM25",
                           type(exc).__name__)
            return EnvSetupResult(
                env=display_name, ok=True, detail="bm25 (provisioning fell back)",
                env_vars={"SEARCH_QA_RETRIEVER": "bm25"},
            )

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
    environment (ALFWORLD_DATA / WEBSHOP_URL / SEARCH_QA_INDEX_DIR /
    SEARCH_QA_RETRIEVER). ``exclusions`` are the VERIFIED ``env_setup_failed``
    records for any env that could not be stood up on this host — feed them to
    ``exclusion.build_scope_block`` so the rubric excludes (not zeroes) those
    leaves. ``release()`` drops every WebShop lease acquired (a no-op for ALFWorld
    / Search-QA); call it in the run's ``finally``.
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
