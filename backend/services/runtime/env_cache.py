"""EnvCacheManager — facade over the env_adapters registry (Phase 1a).

Part B of full-scope-envs (2026-06-01), extended 2026-06-01 for the agentic
re-enablement of the SDAR full scope, and refactored 2026-07-01 into a thin
facade (see
``docs/history/plans/2026-07-01-phase-1a-1b-provisioning-seam-and-gates.md``).
The SDAR paper needs three environments a Search-QA-only run skips:
**ALFWorld** (a multi-GB one-time ``alfworld-download``), **WebShop** (a single
indexed server process, or an in-process backend), and a **dense Search-QA
retriever** (an E5 index over the wiki-18 corpus). This module used to own all
three environments' setup/lifecycle logic directly; that logic has moved
verbatim into three :class:`~backend.services.runtime.env_adapters.EnvironmentAdapter`
implementations (:class:`~backend.services.runtime.env_adapters.AlfworldAdapter` /
:class:`~backend.services.runtime.env_adapters.WebShopAdapter` /
:class:`~backend.services.runtime.env_adapters.SearchQaAdapter`) sharing one
:class:`~backend.services.runtime.asset_cache.AssetCache`. ``EnvCacheManager``
now just builds those three adapters and delegates every public method to
them — a behavior-preserving refactor, not a new API. The on-disk state
filenames (``env_cache_state.json`` / ``.env_cache.lock``) and JSON keys
(``alfworld`` / ``webshop`` / ``search_qa``) are unchanged, so a warm cache
disk from before this refactor stays valid.

Behavior preserved from the pre-refactor implementation:

* **idempotent + host-shared** — ALFWorld data and the dense Search-QA index are
  built/downloaded ONCE into a shared cache dir (``OPENRESEARCH_ENV_CACHE_DIR``,
  default ``<runs_root>/.cache/envs``) and reused by every later run/cell;
* **ref-counted** — ONE WebShop server backs N concurrent leases and is torn
  down only when the last lease releases;
* **crash-safe** — an ``fcntl``-locked state file with stale-server reclaim by
  PID liveness (now :class:`~backend.services.runtime.asset_cache.AssetCache`),
  mirroring ``backend/services/runtime/local_gpu_allocator.py``;
* **fail-soft into the rubric** — a setup that cannot complete on this host
  returns a VERIFIED ``env_setup_failed``
  :class:`~backend.agents.rlm.exclusion.Exclusion` (NOT an exception) for
  ALFWorld/WebShop, so the grid runs the environments that work and the
  rubric EXCLUDES (numerator AND denominator) the rest. Search-QA never
  excludes: a cold/unavailable dense index degrades to BM25 (still real
  retrieval), so the environment always runs. This is the fairness principle
  (2026-06-01): never dock the rubric for an environment the harness could not
  stand up. The verified Exclusion flows through ``exclusion.build_scope_block``
  into ``metrics.json::scope`` and is honoured by the leaf scorer.

DENSE RETRIEVER (2026-06-01): the dense E5/wiki-18 path is **opt-in + configurable**
so a cold or offline host degrades to BM25 rather than blocking the grid on a
multi-GB build. ``OPENRESEARCH_SEARCH_QA_DENSE`` must be truthy to attempt anything;
``OPENRESEARCH_SEARCH_QA_INDEX_REPO`` names a HF repo holding a prebuilt FAISS index +
passage store (snapshot-downloaded, cached, reused). Absent either, Search-QA
provisions ``SEARCH_QA_RETRIEVER=bm25`` and the env's BM25/overlap retriever runs.

The public import surface is FROZEN (callers: ``backend/agents/rlm/run.py``,
``backend/cli.py``, ``scripts/sdar_gcp_assets.py``, ``scripts/batch_reproduce.py``):
``EnvCacheManager``, ``provision_scope``, ``EnvSetupResult``, ``ProvisionResult``,
``default_cache_dir``, ``FULL_SCOPE_ENV_GUIDANCE``. ``_pid_alive`` and a handful
of per-env default-callable helpers (``_search_qa_encoder`` /
``_default_search_qa_index_builder`` / ``_default_webshop_launcher``) are also
re-exported here — not because this module uses them, but because existing
tests call them directly as ``env_cache.<name>`` module attributes, and
``EnvCacheManager.__init__`` threads a pre-construction ``monkeypatch.setattr(
env_cache, "_pid_alive", ...)`` into :class:`WebShopAdapter` by referencing the
bare ``_pid_alive`` name (resolved from this module's namespace at call time,
never captured early).
"""

from __future__ import annotations

# `os` is imported bare (not `from os import kill`) purely so tests can do
# `monkeypatch.setattr(EC.os, "kill", ...)`. `os` is a singleton module, so
# patching it via this module's `os` attribute also patches
# `env_adapters.webshop`'s own `import os`, which is what actually calls
# `os.kill` when tearing down the WebShop server. This module has no direct
# `os.*` call of its own.
import os  # noqa: F401 — monkeypatch surface only, see comment above
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from backend.agents.rlm.exclusion import Exclusion
from backend.services.runtime.asset_cache import AssetCache, _pid_alive, default_cache_dir
from backend.services.runtime.env_adapters import (
    AlfworldAdapter,
    EnvironmentAdapter,
    EnvSetupResult,
    ProvisionCtx,
    SearchQaAdapter,
    WebShopAdapter,
    resolve_adapter,
)

# Back-compat re-exports: existing tests call these directly as module
# attributes (``env_cache._search_qa_encoder()``,
# ``env_cache._default_search_qa_index_builder(...)``,
# ``env_cache._default_webshop_launcher(...)``) rather than going through an
# adapter instance. Each is defined ONCE in its owning adapter module and
# imported here, never redefined — no dead duplicates.
from backend.services.runtime.env_adapters.search_qa import (  # noqa: F401
    _default_search_qa_index_builder,
    _search_qa_encoder,
)
from backend.services.runtime.env_adapters.webshop import _default_webshop_launcher  # noqa: F401

__all__ = [
    "EnvSetupResult",
    "EnvCacheManager",
    "ProvisionResult",
    "provision_scope",
    "default_cache_dir",
    "FULL_SCOPE_ENV_GUIDANCE",
]

# Guidance appended to OPENRESEARCH_BASELINE_EXTRA_GUIDANCE (by backend/cli.py) when the
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
    "  • `from webshop_env import WebShopEnv` — real WebShop (in-process when "
    "WEBSHOP_DATA_DIR is set, or via HTTP server when WEBSHOP_URL is set). For "
    "WebShop cells add flask, gym, beautifulsoup4, rank_bm25, cleantext to "
    "requirements.txt (installed alongside the web_agent_site package).\n"
    "  • `from agentic_rollout import rollout_episode` — drives ONE multi-turn episode "
    "and returns a flat Trajectory(sequence_ids, response_mask, reward, info). Compute "
    "the GRPO advantage over a group of G such rollouts and the OPSD gate token-wise "
    "over the response_mask positions — do NOT hand-roll the turn→token-mask "
    "conversion.\n"
    "ALFWORLD_DATA / WEBSHOP_DATA_DIR + WEBSHOP_PACKAGE_DIR (in-process) or "
    "WEBSHOP_URL (HTTP server) / SEARCH_QA_INDEX_DIR / SEARCH_QA_RETRIEVER are "
    "provided by the host-shared environment cache — consume them. The ALFWorld game "
    "data is ALREADY downloaded under $ALFWORLD_DATA; load games from there and do NOT "
    "run `alfworld-download` yourself (it is unnecessary and may not be on PATH). Do "
    "NOT start your own WebShop server or rebuild the index. Add one cell per "
    "(model × baseline × seed × env) to code/cells.json for EVERY environment in "
    "scope — you MUST include Search-QA AND ALFWorld (and WebShop when WEBSHOP_DATA_DIR "
    "or WEBSHOP_URL is set); a run that trains only one environment is incomplete. Put "
    "the env name in each cell's `env` field, and add the env deps your modules import "
    "to requirements.txt (rank_bm25, sentence-transformers, faiss-cpu, datasets, "
    "alfworld). Train at PAPER DEPTH, not smoke-test depth: STEPS >= 400, "
    "GROUP_SIZE = 8, and a token budget large enough for multi-turn rollouts (agentic "
    "episodes need many turns × tokens). If an environment's data or server is genuinely "
    "unavailable at runtime, record it as a scope gap (do NOT crash the grid) — the "
    "harness converts a verified-unavailable env into a rubric exclusion."
)

# Environments this manager knows how to stand up.
_ALFWORLD = "ALFWorld"
_WEBSHOP = "WebShop"
_SEARCH_QA = "Search-QA"


class EnvCacheManager:
    """Facade over the env_adapters registry (ALFWorld / WebShop / Search-QA).

    Builds one shared :class:`~backend.services.runtime.asset_cache.AssetCache`
    plus the three concrete adapters, and delegates every public method to
    them. All side-effecting operations (download, server launch, health
    probe, dense index build) are injected callables — passed straight
    through to the owning adapter — so the entire lifecycle is unit-testable
    without touching the network, a multi-GB download, or a real server.
    Every public method is fail-soft: an ALFWorld/WebShop provisioning error
    becomes an :class:`EnvSetupResult` carrying a verified ``env_setup_failed``
    :class:`~backend.agents.rlm.exclusion.Exclusion`; a Search-QA dense-index
    failure degrades to BM25 (never an exclusion — the env always runs).
    Nothing raises.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        downloader: Callable[[Path], None] | None = None,
        server_launcher: Callable[[Path, int], int] | None = None,
        probe: Callable[[str], bool] | None = None,
        index_builder: Callable[[Path], "Path | None"] | None = None,
        inprocess_smoke: "Callable[[str], bool] | None" = None,
        webshop_port: int = 3000,
        server_ready_timeout_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = AssetCache(cache_dir)
        self.cache_dir = self._cache.cache_dir  # back-compat attribute
        self._alfworld = AlfworldAdapter(self._cache, downloader=downloader, clock=clock)
        self._webshop = WebShopAdapter(
            self._cache,
            server_launcher=server_launcher,
            probe=probe,
            inprocess_smoke=inprocess_smoke,
            # Bare module-global name, resolved from THIS module's namespace at
            # call time (not `asset_cache._pid_alive`, not captured into a local
            # early) — so a pre-construction `monkeypatch.setattr(env_cache,
            # "_pid_alive", ...)` is honoured by the adapter.
            pid_alive=_pid_alive,
            webshop_port=webshop_port,
            server_ready_timeout_s=server_ready_timeout_s,
            clock=clock,
        )
        self._search_qa = SearchQaAdapter(self._cache, index_builder=index_builder, clock=clock)
        self._adapters: tuple[EnvironmentAdapter, ...] = (
            self._alfworld,
            self._webshop,
            self._search_qa,
        )

    # --- public provisioning -------------------------------------------------

    def setup(self, env: str) -> EnvSetupResult:
        """Provision one environment by name (case-insensitive). Never raises."""
        adapter = resolve_adapter(env, self._adapters)
        if adapter is None:
            # Any other dataset-only env: nothing to provision.
            return EnvSetupResult(env=env or "", ok=True, detail="no environment to provision")
        return adapter.provision(ProvisionCtx(display_name=env or adapter.key))

    def ensure_alfworld(self, *, display_name: str = _ALFWORLD) -> EnvSetupResult:
        """Download ALFWorld once into the shared cache; reuse on later calls.

        Delegates to :meth:`AlfworldAdapter.provision`.
        """
        return self._alfworld.provision(ProvisionCtx(display_name=display_name))

    def acquire_webshop(self, *, display_name: str = _WEBSHOP) -> EnvSetupResult:
        """Acquire WebShop: in-process (preferred) or HTTP server (legacy).

        Delegates to :meth:`WebShopAdapter.provision`.
        """
        return self._webshop.provision(ProvisionCtx(display_name=display_name))

    def release_webshop(self) -> None:
        """Drop one WebShop lease; stop the server when the last lease releases.

        Delegates to :meth:`WebShopAdapter.release`.
        """
        self._webshop.release()

    def ensure_search_qa_index(self, *, display_name: str = _SEARCH_QA) -> EnvSetupResult:
        """Provide a Search-QA retriever: dense E5 index when buildable, else BM25.

        Delegates to :meth:`SearchQaAdapter.provision`.
        """
        return self._search_qa.provision(ProvisionCtx(display_name=display_name))


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

    Asset pre-staging (OPENRESEARCH_ASSET_RESOLVER_V2, default OFF):
    ``asset_prestage.build_default_resolver()`` returns ``None`` when the flag
    is unset — the ``if resolver is not None`` guard is never entered, so the
    loop body below is **byte-identical to today** when the flag is OFF.
    """
    from backend.services.runtime import asset_prestage  # lazy: avoids import cycle

    # build_default_resolver() → None when OPENRESEARCH_ASSET_RESOLVER_V2 is
    # off (default).  When None, the inner guard is never entered — flag-off
    # path is provably byte-identical to the pre-wiring code above.
    resolver = asset_prestage.build_default_resolver()

    env_vars: dict[str, str] = {}
    exclusions: list[Exclusion] = []
    webshop_leases = 0
    for name in env_names or []:
        # Pre-stage corpus assets (no-op when resolver is None / flag OFF).
        if resolver is not None:
            staged = asset_prestage.prestage_env_assets(name, resolver, manager._cache)
            if staged:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "provision_scope: pre-staged %d file(s) for %r: %s",
                    len(staged), name, staged,
                )
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
