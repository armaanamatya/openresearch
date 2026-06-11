"""Application configuration via Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _apply_legacy_env_aliases() -> None:
    """Backward-compat shim for the 2026-06 env-var rename
    ``REPROLAB_*`` -> ``OPENRESEARCH_*``.

    For every still-set legacy or new variable, fill in the missing counterpart
    (never overwriting an explicitly-set value) so existing deployments, CI, and
    shells that still export the old ``REPROLAB_*`` names keep working unchanged.
    Runs once at import, before any ``Settings()`` is constructed.

    NOTE: this mirrors *process* environment variables only. A pre-existing
    ``.env`` file that still uses ``REPROLAB_*`` keys should be migrated to
    ``OPENRESEARCH_*`` (the committed ``.env.example`` already is).
    """
    for key, val in list(os.environ.items()):
        if key.startswith("REPROLAB_"):
            os.environ.setdefault("OPENRESEARCH_" + key[len("REPROLAB_") :], val)
        elif key.startswith("OPENRESEARCH_"):
            os.environ.setdefault("REPROLAB_" + key[len("OPENRESEARCH_") :], val)


_apply_legacy_env_aliases()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENRESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # pydantic-settings does NOT mutate os.environ, but it does read the
        # .env file with full precedence rules: shell env > .env file >
        # default. That's exactly what we want for API keys — see the
        # ``anthropic_api_key`` / ``openai_api_key`` fields below.
        populate_by_name=True,
    )

    environment: Literal["development", "testing", "production"] = "development"
    database_url: str = "sqlite:///openresearch.db"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    # Per-project blob directory root. Bound to OPENRESEARCH_RUNS_ROOT via the
    # OPENRESEARCH_ env_prefix above. None = use the call-site default (usually
    # ``<repo>/runs``). The dev launchers (scripts/dev.ps1, scripts/dev.sh)
    # export OPENRESEARCH_RUNS_ROOT to colocate pipeline workspaces with each
    # launch's server logs; without this field, that export was cosmetic.
    runs_root: Path | None = None
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_default_model: str = "claude-sonnet-4-6"
    anthropic_reasoning_model: str = "claude-opus-4-7"
    openai_default_model: str = "gpt-4o"
    openai_reasoning_model: str = "o4-mini"
    agent_provider_overrides: dict[str, str] = Field(default_factory=dict)
    # Per-agent wall-clock cap overrides (seconds). Bumps the
    # ExecutionProfile.agent_wall_clock_seconds for a specific agent so
    # heavy stages like baseline-implementation on complex papers don't
    # die at the profile's blanket 1200s. Example .env:
    #   OPENRESEARCH_AGENT_WALL_CLOCK_OVERRIDES='{"baseline-implementation": 2400}'
    # Unset agents continue to use the profile default. Avoids forcing the
    # whole run to executionMode=max when only one agent needs more time.
    agent_wall_clock_overrides: dict[str, float] = Field(default_factory=dict)
    # Force the orchestrator's provider chain to [primary] only — no
    # cross-provider fallback. Useful when the operator only has working
    # credentials for one provider and the other key in env is invalid
    # (common: a leftover sk-svcacct-* in the shell that 401s on chat
    # completions). Without this, a transient anthropic blip can trigger
    # an openai fallback attempt that surfaces a misleading 401 and kills
    # the run. Default keeps existing behaviour for users with two valid
    # keys.
    provider_fallback_disabled: bool = False

    # External provider API keys. We read both the unprefixed names that
    # the upstream SDKs (anthropic, openai) and most CI conventions use,
    # AND the OPENRESEARCH_-prefixed forms, because some deployments reserve
    # the unprefixed names for a different scope. First match wins.
    #
    # WHY THIS LIVES IN SETTINGS, NOT os.environ:
    # The Hermes audit providers used to read these directly from
    # ``os.environ.get(...)`` and were skipped whenever the spawning
    # process (Lab UI's Next.js dev server, docker entrypoint without
    # env_file, pytest from a fresh shell) hadn't loaded the .env. The
    # values were always in .env, but never in os.environ. Funnelling
    # through Settings makes pydantic-settings the single source of
    # truth: it reads .env from disk on every ``Settings()`` construction
    # regardless of what os.environ contains. Providers pass these
    # values explicitly to ``anthropic.Anthropic(api_key=...)`` /
    # ``openai.OpenAI(api_key=...)`` so the SDKs don't fall back to
    # their own os.environ lookup either.
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ANTHROPIC_API_KEY",
            "OPENRESEARCH_ANTHROPIC_API_KEY",
        ),
    )
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "OPENRESEARCH_OPENAI_API_KEY",
        ),
    )
    openai_admin_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "OPENAI_ADMIN_KEY",
            "OPENRESEARCH_OPENAI_ADMIN_KEY",
        ),
    )
    codex_cli_path: str = ""
    codex_auth_path: str = ""

    # Optional Codex repo-editing subagent route. This is deliberately NOT a
    # general LLM provider and is default-off; the RLM root can only reach it
    # through the gated codex_repair primitive.
    codex_subagent: bool = False
    codex_timeout_s: int = Field(default=900, ge=1)
    codex_max_calls_per_run: int = Field(default=3, ge=0)
    codex_max_output_chars: int = Field(default=12000, ge=100)
    codex_profile: str = "openresearch-readwrite"
    codex_allowed_tasks: str = (
        "implementation_repair,test_debugging,dockerfile_repair,"
        "requirements_repair"
    )

    # Paper extraction mode. "hybrid" uses vision (Claude) to enrich scanned
    # pages and figure descriptions; falls back to text-only when no API key
    # is set, so the default is safe. "text" forces the text-only path.
    paper_extraction_mode: Literal["text", "hybrid"] = "hybrid"
    paper_extraction_vision_model: str = "claude-sonnet-4-6"

    # Track 3 — rubric verifier + self-improvement loop. Opt-in surface: with
    # these defaults the verifier runs for new runs and degrades cleanly on
    # error; existing runs under existing configs are otherwise unaffected.
    rubric_verifier_enabled: bool = True
    rubric_verifier_model: str = ""  # empty -> inherit the run's model
    # Heuristic target on the verifier's own 0-1 rubric scale — NOT calibrated
    # against PaperBench's judge (a different scale). Per-version calibration is
    # future work.
    rubric_target_score: float = 0.70
    rubric_max_improvement_iterations: int = 2

    # Track 4 — environment build-and-repair loop. Opt-in surface: with these
    # defaults the Dockerfile is built (and repaired on failure) at
    # ENVIRONMENT_BUILT instead of failing ~30 min later at BASELINE_RUN. With
    # validation disabled the run behaves exactly as it did before Track 4.
    environment_build_validation_enabled: bool = True
    environment_build_max_attempts: int = 3

    # Default sandbox mode for dashboard requests that omit a sandbox. CLI
    # defaults remain controlled separately by argparse flags. Local launchers
    # set this to runpod for GPU-backed dev runs; deployments can set it to
    # docker or local in env.
    default_sandbox: Literal["auto", "local", "docker", "runpod", "azure"] = "runpod"

    # Optional hard override for every run's sandbox mode, regardless of what
    # the client requested. Empty means "honor the request/default_sandbox".
    # Deployments that must forbid RunPod should set OPENRESEARCH_FORCE_SANDBOX to
    # "docker" or "local" explicitly; the code default must stay empty so a
    # missing/commented .env line does not silently rewrite sandbox=runpod.
    force_sandbox: Literal["", "auto", "local", "docker", "runpod", "azure"] = ""

    # Force the LLM provider for every run regardless of what the client
    # requested — analogous to force_sandbox. The UI hard-codes provider=
    # "anthropic" in the start-run request; on deployments where the operator
    # only has OpenAI credentials, OPENRESEARCH_FORCE_LLM_PROVIDER=openai rewrites
    # the request server-side so a stale UI default doesn't trigger an
    # unconfigured-provider error mid-pipeline. Empty disables the override.
    force_llm_provider: Literal["", "anthropic", "openai"] = ""

    # Shared secret gating the run-start endpoints on public deployments.
    # Empty = gate disabled (local dev). When set, POST /runs and
    # POST /runs/upload require a matching X-Demo-Secret header.
    demo_secret: str = ""

    runpod_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "RUNPOD_API_KEY",
            "OPENRESEARCH_RUNPOD_API_KEY",
        ),
    )
    runpod_api_base_url: str = "https://rest.runpod.io/v1"
    # Reverted from -runtime- back to -devel-: runtime variant lacks CUDA dev
    # headers, which breaks bitsandbytes / flash-attn / deepspeed at pip-install
    # time (no precompiled wheel → tries to JIT, fails). SDAR run hit this:
    # bitsandbytes silently failed under chained `pip install -q ... && python`,
    # train.py then ModuleNotFoundError'd on transformers. The 14GB cold-start
    # savings aren't worth the breakage. Override via OPENRESEARCH_RUNPOD_IMAGE
    # if you have a paper that genuinely doesn't need dev headers.
    runpod_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    runpod_gpu_type: str = "NVIDIA GeForce RTX 4090"
    runpod_gpu_count: int = 1
    runpod_cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    runpod_container_disk_gb: int = 50
    runpod_volume_gb: int = 20
    runpod_volume_mount_path: str = "/workspace"
    runpod_network_volume_id: str = ""
    runpod_data_center_ids: str = ""
    runpod_ssh_key_path: str = ""
    runpod_ssh_public_key: str = ""
    runpod_ssh_user: str = "root"
    runpod_boot_timeout_seconds: int = 900
    runpod_delete_on_destroy: bool = True
    runpod_bootstrap_command: str = ""
    # When set, the Runpod backend attaches to this existing pod ID
    # instead of creating a fresh pod per run. The pod is NEVER deleted
    # by the backend (the _owned_pod_ids allowlist enforces this even
    # if delete_on_destroy=true). Useful for persistent shared workers.
    runpod_pod_id: str = ""

    # --- Azure AKS GPU backend (spec 2026-06-03, --sandbox azure) ---
    # All fields default to empty/sensible stubs so importing Settings never
    # requires Azure credentials to be present — the backend is lazy-imported
    # and only instantiated when --sandbox azure is actually selected.
    azure_resource_group: str = Field(default="", description="Azure resource group for the AKS cluster")
    azure_region: str = Field(default="eastus", description="Azure region (e.g. eastus, westus2)")
    azure_storage_account: str = Field(default="", description="Azure storage account name (Blob + Files)")
    azure_blob_container: str = Field(default="reprolab-artifacts", description="Blob container for run artifacts")
    azure_files_share: str = Field(default="reprolab-cache", description="Azure Files share for HF_HOME + pip cache")
    azure_acr_login_server: str = Field(default="", description="ACR login server (e.g. myregistry.azurecr.io)")
    azure_aks_cluster: str = Field(default="", description="AKS cluster name")
    azure_namespace: str = Field(default="reprolab", description="Kubernetes namespace for Job submission")
    azure_service_account: str = Field(default="reprolab-sa", description="K8s ServiceAccount annotated for workload identity (must match the federated-credential subject: system:serviceaccount:<ns>:reprolab-sa)")
    azure_node_pool_name: str = Field(default="gpua100", description="GPU node pool name (scale-to-zero)")
    azure_per_gpu_vram_gb: float = Field(default=80.0, ge=1.0, description="VRAM per GPU in the node pool (A100=80)")
    azure_max_nodes: int = Field(default=4, ge=1, description="Node pool max-nodes (orchestrator-side concurrency cap)")
    # Empty means the operator MUST set REPROLAB_AZURE_BASE_IMAGE to a PINNED
    # ACR tag (e.g. myregistry.azurecr.io/reprolab:20260603-abc1234). The runner
    # errors clearly on empty rather than defaulting to a floating :latest tag.
    azure_base_image: str = Field(default="", description="Pre-baked ACR base image (build_environment no-op); operator must set to a PINNED ACR tag — never :latest")
    azure_gpu_usd_per_hour: float = Field(default=3.67, ge=0.0, description="Per-GPU $/hr for budget tracking (default = Standard_NC24ads_A100_v4 on-demand list price; set your negotiated rate). 0 disables the run-USD cost cap.")
    azure_boot_timeout_seconds: int = Field(default=900, ge=1, description="Seconds to wait for a Job pod to leave Pending")
    azure_pending_timeout_seconds: int = Field(default=1500, ge=1, description="Seconds before a stuck-Pending cell is failed as capacity_exhausted (AKS GPU cold-start from zero can take 10-12 min; 900s killed legitimate scale-up)")
    # Catalog short_names of the Azure GPU SKUs that are actually provisioned
    # as AKS node pools (mirrors Terraform var.gpu_skus). The SKU resolver
    # only selects from this list; the OOM escalation ladder (reused from
    # dynamic_gpu_max_escalations — no new field) only advances within it.
    # pydantic-settings 2.x parses this from a JSON array env var:
    #   REPROLAB_AZURE_GPU_SKUS='["azure_a100_80","azure_a100_80x2"]'
    # or from a comma-separated string via the built-in list coercion when
    # a plain string is supplied (e.g. REPROLAB_AZURE_GPU_SKUS=azure_a100_80,azure_a100_80x2).
    # Default = single A100-80 pool = one quota ask at cluster bootstrap.
    azure_gpu_skus: list[str] = Field(
        default_factory=lambda: ["azure_a100_80"],
        description=(
            "Catalog short_names of the Azure GPU SKUs that are actually provisioned "
            "as node pools (Terraform var.gpu_skus). The resolver only selects from "
            "these; the OOM ladder only escalates within these. "
            "Default = single A100-80 pool = one quota ask."
        ),
    )
    # TTL added to the Job spec's ttlSecondsAfterFinished; Kubernetes deletes
    # the Job + Pod objects this many seconds after they reach a terminal state
    # (Succeeded or Failed). Keeps the namespace tidy without operator cron jobs.
    azure_ttl_seconds_after_finished: int = Field(
        default=3600,
        ge=1,
        description="Job.spec.ttlSecondsAfterFinished — Kubernetes auto-deletes finished Jobs after this many seconds",
    )
    # Number of times Kubernetes will restart the Job's Pod on failure.
    # Set to 0 because OOM retry + podFailurePolicy is handled in-process
    # by the OOM wrapper; letting Kubernetes restart the Pod would bypass
    # that logic and double-count failures.
    azure_job_backoff_limit: int = Field(
        default=0,
        ge=0,
        description="Job.spec.backoffLimit (Pod-level retries); keep at 0 — OOM retry is delegated to the in-Job wrapper + podFailurePolicy",
    )
    # Path inside the Job Pod where the Azure Files share (HF_HOME + pip cache)
    # is mounted. Must match the volume mount in the Job template and the
    # HF_HOME / pip cache env vars injected by the runner.
    azure_cache_mount_path: str = Field(
        default="/mnt/reprolab-cache",
        description="Mount path for the Azure Files share (HF_HOME + pip cache) inside Job Pods",
    )
    # How often the Job watcher polls the Kubernetes API for Pod phase changes.
    # Lower values reduce latency between Job completion and result ingestion;
    # higher values reduce API server load on large clusters.
    azure_watch_poll_interval_s: float = Field(
        default=5.0,
        gt=0,
        description="Polling interval (seconds) for the Job/Pod phase watcher",
    )
    # Batch-size scale factors for the two-step OOM shrink retry.
    # Step 1: multiply the cell's batch size by this factor before the first
    # OOM retry. Step 2 (floor): never go below this factor regardless of
    # further OOMs. Must satisfy 0 < floor <= step1 <= 1.
    azure_oom_batch_scale_step1: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Batch-size scale factor applied on the first OOM retry (step 1 of 2)",
    )
    azure_oom_batch_scale_floor: float = Field(
        default=0.25,
        gt=0,
        le=1,
        description="Minimum batch-size scale factor for OOM shrink retries (floor; never go below this)",
    )
    # Timeout (seconds) for `pip install -r requirements.txt` inside the Job
    # bootstrap script. Some SDAR dependency trees are large; 600s avoids
    # spurious bootstrap timeouts on slow ACR pulls or large sdists.
    azure_bootstrap_pip_timeout_s: int = Field(
        default=600,
        ge=1,
        description="Timeout (seconds) for pip install in the Job bootstrap script",
    )

    # --- Forced-iteration policy (Lane H, spec 2026-05-24) ---
    # When the root model calls FINAL_VAR but the latest rubric overall_score
    # is below target_score AND the run has not yet attempted at least this
    # many iterations, the orchestrator refuses the FINAL_VAR, emits a
    # `run_warning` SSE event, and forces the loop to continue so the root
    # has a real chance to call propose_improvements + implement_baseline
    # again with repair_context. Wall-clock takes precedence: when the
    # remaining budget is below the floor (≤60s), the policy is bypassed
    # and a partial report is shipped honestly.
    #   0 — disables the policy (any FINAL_VAR is accepted).
    #   2 (default) — at least two rubric-aware attempts before bailing out.
    min_rubric_iterations: int = Field(
        default=2,
        ge=0,
        le=10,
        description=(
            "Force the root model to attempt at least this many iterations "
            "before FINAL_VAR is accepted when the rubric score is below "
            "target_score. 0 disables. Bypassed when wall-clock <= 60s."
        ),
    )

    # --- Dynamic GPU selection (spec 2026-05-23) ---
    # Accepts both spellings: OPENRESEARCH_DYNAMIC_GPU is what CLAUDE.md
    # documents and what the CLI --dynamic-gpu/--no-dynamic-gpu flag writes;
    # without the alias both were silent no-ops (only ..._ENABLED was read).
    dynamic_gpu_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "OPENRESEARCH_DYNAMIC_GPU_ENABLED",
            "OPENRESEARCH_DYNAMIC_GPU",
        ),
        description="Wire paper hardware clues to RunPod SKU choice",
    )
    force_single_gpu: bool = Field(default=True, description="Cap RunPod GPU count at 1 regardless of paper")
    max_gpu_usd_per_hour: float = Field(default=10.0, ge=0.0, description="Per-GPU $/hr cap; 0 disables")
    max_run_gpu_usd: float = Field(default=10.0, ge=0.0, description="Total RunPod $ per run cap; 0 disables")
    dynamic_gpu_headroom: float = Field(default=1.25, ge=1.0, description="Multiplier on LLM VRAM estimate before tier-up")
    dynamic_gpu_fallback_vram_gb: int = Field(default=24, ge=1, description="Substitute VRAM when LLM cannot estimate")
    dynamic_gpu_max_escalations: int = Field(default=2, ge=0, description="Max OOM-driven ladder advances per run")

    # --- BES on RDR (spec 2026-06-07, default OFF) ---
    # Competing candidates extend the RDR controller behind a MASTER gate. When
    # bes_enabled is False every child flag below is inert and run_rdr behaves
    # bit-for-bit as today. See docs/superpowers/specs/2026-06-07-bes-integration/.
    bes_enabled: bool = Field(default=False, description="MASTER gate for BES-on-RDR; off => today's RDR path")
    bes_candidates_per_cluster: int = Field(default=1, ge=1, le=8, description="N competing candidates per cluster; 1 = parity")
    bes_select_metric: str = Field(default="cluster_score", description="Candidate SELECT metric (cluster_score | failed_leaves); an unknown value falls back to cluster_score at use-site, so it never aborts the default RDR path")
    bes_splice_enabled: bool = Field(default=False, description="Evolve/splice (v2, deferred) — no-op in v1")
    # Adaptive gating (2026-06-11, RLM path): compete only where selection has
    # variance to remove. The allcnn-ab-20260611 pool discriminated weakly
    # (0.549 vs 0.557) BECAUSE the seeded best-attempt + champion rails already
    # anchor implementation quality on papers with history — the pool's value
    # concentrates on FIRST attempts and weak-history papers. When bes_adaptive
    # is on, the pool engages only if the project has no prior attempt or its
    # best score is below bes_adaptive_skip_score; the decision is persisted to
    # rlm_state/bes_adaptive.json and stamped into experiment_arm. Keep OFF for
    # A/B arms (they need deterministic pool behaviour).
    bes_adaptive: bool = Field(default=False, description="Engage the RLM candidate pool only on first-attempt / weak-history papers")
    bes_adaptive_skip_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Best prior attempt score at/above which adaptive mode skips the pool")

    # --- Mode-agnostic RDR pre-run gate (Phase 2, default OFF) ---
    rdr_preflight_gate: bool = Field(default=False, description="Run scan_code_dir before run_experiment on the RDR path")
    rdr_preflight_max_regens: int = Field(default=1, ge=0, le=3, description="Max code regenerations on a pre-run-gate violation")

    # Budget-awareness prompt for implement_baseline. Tells the baseline-writing
    # agent to scale train.py to fit remaining_s wall-clock.
    #   "auto"   — inject only on cost-bearing sandboxes (runpod / brev)
    #   "always" — inject regardless of sandbox
    #   "never"  — skip regardless (paper-faithful epoch counts)
    budget_awareness_mode: str = Field(
        default="auto",
        pattern=r"^(auto|always|never)$",
        description="When to inject the EXECUTION-BUDGET AWARENESS block into the baseline agent prompt",
    )

    # Multi-tenant / production LLM auth strategy.
    #   "auto" (default) — pick whichever credential is available, preferring
    #     a funded API key (separate rate-limit pool, billable) over the
    #     local OAuth subscription (shared, single-user rate limit).  This
    #     is the right default for solo dev where OAuth is free.
    #   "api_only" — refuse to start unless a paid API key is present.  Use
    #     for production / multi-user deployments where OAuth's per-account
    #     rate limit would throttle every concurrent agent.
    #   "oauth_only" — force the OAuth subscription path even when an API
    #     key is set.  Useful for cost-bounded local iteration.
    # The strategy is enforced at runtime resolution; an unsatisfiable
    # strategy fails fast at startup rather than silently degrading.
    llm_auth_strategy: str = Field(
        default="auto",
        pattern=r"^(auto|api_only|oauth_only)$",
        description=(
            "LLM credential preference. 'api_only' requires a paid API key and "
            "is the recommended production setting (separate rate-limit pool, "
            "no single-user OAuth contention)."
        ),
    )

    # --- PR-π Module E — parsed paper precondition gate (spec 2026-05-26) ---
    # When True (default), the RLM loop proceeds even when parsed_full_text.txt
    # is missing or smaller than 1 KB, logging a warning. When False, the run
    # fails fast with a descriptive RuntimeError before the RLM loop starts.
    # Default is True for backwards compatibility; flip to False in PR-ρ after
    # observing production for a week.
    allow_lossy_paper_text: bool = Field(
        default=True,
        description=(
            "Allow the RLM loop to proceed when parsed_full_text.txt is missing "
            "or <1 KB (lossy workspace fallback). When False, missing/small "
            "parsed_full_text.txt raises RuntimeError before the loop starts."
        ),
    )

    # Apify ArXiv MCP server (https://github.com/apify/actor-arxiv-mcp-server).
    # When apify_api_token is set, the Claude agent runtime registers the
    # SSE endpoint as an MCP server named ``apify-arxiv`` and exposes its
    # tools to the agents listed in apify_arxiv_enabled_agents. When the
    # token is empty, MCP wiring is skipped entirely (no extra latency,
    # no failed handshake on cold start). The token is read via the
    # ``APIFY_API_TOKEN`` env var (matching Apify's own SDK convention)
    # OR the ``OPENRESEARCH_APIFY_API_TOKEN`` form.
    apify_api_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "APIFY_API_TOKEN",
            "OPENRESEARCH_APIFY_API_TOKEN",
        ),
    )
    apify_arxiv_mcp_url: str = "https://jakub-kopecky--arxiv-mcp-server.apify.actor/sse"
    # Comma-separated agent ids that should see the apify-arxiv MCP tools.
    # Defaults to the builder agents that already do paper / artifact
    # research. Override via .env if a custom agent should also use it.
    apify_arxiv_enabled_agents: str = "artifact-discovery,paper-understanding"

    @model_validator(mode="after")
    def _fall_back_to_legacy_sqlite_db(self) -> "Settings":
        """Backward-compat for the 2026-06 DB-file rename ``reprolab.db`` ->
        ``openresearch.db``.

        If ``database_url`` is still the default and the new ``openresearch.db``
        does not exist on disk but a legacy ``reprolab.db`` does, keep using the
        legacy file so existing local installs and mounted volumes don't silently
        start from an empty database. Explicit ``OPENRESEARCH_DATABASE_URL``
        overrides are untouched.
        """
        if self.database_url == "sqlite:///openresearch.db":
            if not Path("openresearch.db").exists() and Path("reprolab.db").exists():
                self.database_url = "sqlite:///reprolab.db"
        return self


_settings_cache: Settings | None = None


def get_settings(_force_reload: bool = False) -> Settings:
    """Return application settings, cached after first call."""
    global _settings_cache
    if _force_reload or _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


# Marker so tests can check: hasattr(get_settings, '_force_reload')
get_settings._force_reload = True
