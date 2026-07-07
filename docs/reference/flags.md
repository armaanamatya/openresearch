# OPENRESEARCH_* Feature-Flag Registry

> **Generated — do not hand-edit.** Regenerate with `.venv/bin/python scripts/gen_flag_registry.py`. Source of truth is the code, not this file.

## Summary

- **Total distinct flags:** 398
- **Managed by `config.py` Settings (typed, default known):** 137
- **Ad-hoc `os.environ` reads (no central default):** 261
- **Mentioned in `CLAUDE.md`:** 92 (23%)

Legend: **cfg** = typed in `config.py`; **doc** = appears in `CLAUDE.md`; **sites** = ad-hoc read count; **default** = literal default at first read site (best effort).

## Config fields bypassed by ad-hoc reads

15 flags are typed in `config.py` **and** still read directly via `os.environ`, so the typed default is dead at those call sites. Review before consolidating — some are call-time reads on purpose (test monkeypatch / per-run toggle).

| Flag | cfg default | ad-hoc reads |
|---|---|:--:|
| `OPENRESEARCH_AZURE_BLOB_CONTAINER` | `Field(...)` | 1 |
| `OPENRESEARCH_AZURE_REGION` | `Field(...)` | 2 |
| `OPENRESEARCH_AZURE_STORAGE_ACCOUNT` | `Field(...)` | 1 |
| `OPENRESEARCH_CODEX_CLI_PATH` | `""` | 1 |
| `OPENRESEARCH_DYNAMIC_GPU_HEADROOM` | `Field(...)` | 1 |
| `OPENRESEARCH_MIN_RUBRIC_ITERATIONS` | `Field(...)` | 1 |
| `OPENRESEARCH_REPRODUCTION_MODE` | `"adapt"` | 1 |
| `OPENRESEARCH_RUBRIC_VERIFIER_MODEL` | `""` | 1 |
| `OPENRESEARCH_RUNPOD_API_KEY` | `Field(...)` | 5 |
| `OPENRESEARCH_RUNPOD_CONTAINER_DISK_GB` | `50` | 1 |
| `OPENRESEARCH_RUNPOD_GPU_COUNT` | `1` | 1 |
| `OPENRESEARCH_RUNPOD_GPU_TYPE` | `"NVIDIA GeForce RTX 4090"` | 1 |
| `OPENRESEARCH_RUNPOD_IMAGE` | `"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-` | 1 |
| `OPENRESEARCH_RUNPOD_VOLUME_GB` | `20` | 1 |
| `OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH` | `"/workspace"` | 5 |

### `OPENRESEARCH_AB_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_AB_ARM` |  | ✅ |  | |
| `OPENRESEARCH_AB_PAIR_ID` |  | ✅ |  | |

### `OPENRESEARCH_ACCELERATOR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ACCELERATOR` |  | ✅ | 2 | `""` |
| `OPENRESEARCH_ACCELERATOR_API_KEY` |  | ✅ | 3 | `"local"` |
| `OPENRESEARCH_ACCELERATOR_BASE_URL` |  | ✅ | 3 | `_DEFAULT_LOCAL_BASE_URL` |
| `OPENRESEARCH_ACCELERATOR_MODEL` |  | ✅ | 4 | `_DEFAULT_LOCAL_MODEL` |
| `OPENRESEARCH_ACCELERATOR_SCOPE` |  | ✅ | 2 | `""` |

### `OPENRESEARCH_AGENT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_AGENT_WALL_CLOCK_OVERRIDES` |  |  |  | |

### `OPENRESEARCH_ALFWORLD_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ALFWORLD_` |  |  |  | |
| `OPENRESEARCH_ALFWORLD_ENV_REUSE` |  |  |  | |
| `OPENRESEARCH_ALFWORLD_MAX_TURNS` |  |  |  | |
| `OPENRESEARCH_ALFWORLD_SHAPED_REWARD` |  |  |  | |
| `OPENRESEARCH_ALFWORLD_SHAPING` |  |  |  | |

### `OPENRESEARCH_ALLOW_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ALLOW_LOSSY_PAPER_TEXT` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_ANTHROPIC_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ANTHROPIC_API_KEY` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_ANTHROPIC_DEFAULT_MODEL` | ✅ |  |  | `"claude-sonnet-4-6"` |
| `OPENRESEARCH_ANTHROPIC_REASONING_MODEL` | ✅ |  |  | `"claude-opus-4-7"` |

### `OPENRESEARCH_ANTIFAB_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ANTIFAB_GUARD` |  |  | 2 | `"1"` |
| `OPENRESEARCH_ANTIFAB_MIN_VRAM_GB` |  |  | 1 | `"1.5"` |

### `OPENRESEARCH_APIFY_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_APIFY_API_TOKEN` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_APIFY_ARXIV_ENABLED_AGENTS` | ✅ |  |  | `"artifact-discovery,paper-understanding"` |
| `OPENRESEARCH_APIFY_ARXIV_MCP_URL` | ✅ |  |  | `"https://jakub-kopecky--arxiv-mcp-server` |

### `OPENRESEARCH_ARG_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ARG_CONTRACTS` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_ARTIFACT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ARTIFACT_DIR` |  |  |  | |

### `OPENRESEARCH_AZURE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_AZURE_` |  | ✅ |  | |
| `OPENRESEARCH_AZURE_ACR_LOGIN_SERVER` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_AKS_CLUSTER` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_BASE_IMAGE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_BLOB_CONTAINER` | ✅ |  | 1 | `Field(...)` |
| `OPENRESEARCH_AZURE_BOOTSTRAP_PIP_TIMEOUT_S` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_BOOT_TIMEOUT_SECONDS` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_CACHE_MOUNT_PATH` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_CELL_ROUTE` |  | ✅ | 1 | `"1"` |
| `OPENRESEARCH_AZURE_DATASTORE_GB` |  |  | 1 | `"0"` |
| `OPENRESEARCH_AZURE_DATASTORE_MOUNT` |  |  | 1 | `"/mnt/azureml"` |
| `OPENRESEARCH_AZURE_DATA_DISK_GB` |  |  | 1 | `"100"` |
| `OPENRESEARCH_AZURE_FILES_CACHE_ENABLED` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_FILES_SHARE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_FOUNDRY_API_KEY` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_FOUNDRY_DEPLOYMENT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_FOUNDRY_ENDPOINT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_GPUS_PER_NODE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_GPU_SKUS` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_GPU_USD_PER_HOUR` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_IMAGE` |  |  | 1 | `"mcr.microsoft.com/azureml/curated/acpt-` |
| `OPENRESEARCH_AZURE_JOB_BACKOFF_LIMIT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_MAX_NODES` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_NAMESPACE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_NODE_POOL_NAME` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_OOM_BATCH_SCALE_FLOOR` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_OOM_BATCH_SCALE_STEP1` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_OPENAI_API_KEY` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_OPENAI_API_VERSION` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_OPENAI_DEPLOYMENT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_OPENAI_ENDPOINT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_PENDING_TIMEOUT_SECONDS` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_PER_GPU_VRAM_GB` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_REGION` | ✅ |  | 2 | `Field(...)` |
| `OPENRESEARCH_AZURE_RESOURCE_GROUP` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_SERVICE_ACCOUNT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_SPOT_BACKOFF_LIMIT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_STORAGE_ACCOUNT` | ✅ |  | 1 | `Field(...)` |
| `OPENRESEARCH_AZURE_TTL_SECONDS_AFTER_FINISHED` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_USE_SPOT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_AZURE_VM_SIZE` |  |  | 1 | `""` |
| `OPENRESEARCH_AZURE_WATCH_POLL_INTERVAL_S` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_BASELINE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` |  | ✅ | 3 | `""` |
| `OPENRESEARCH_BASELINE_SUBPROCESS` |  |  | 1 | `"0"` |

### `OPENRESEARCH_BES_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_BES_ADAPTIVE` | ✅ | ✅ |  | `Field(...)` |
| `OPENRESEARCH_BES_ADAPTIVE_SKIP_SCORE` | ✅ | ✅ |  | `Field(...)` |
| `OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER` | ✅ | ✅ |  | `Field(...)` |
| `OPENRESEARCH_BES_CONTINUE_MIN_S` |  |  |  | |
| `OPENRESEARCH_BES_ENABLED` | ✅ | ✅ |  | `Field(...)` |
| `OPENRESEARCH_BES_MIN_REMAINING_S` |  | ✅ |  | |
| `OPENRESEARCH_BES_SELECT_METRIC` | ✅ | ✅ |  | `Field(...)` |
| `OPENRESEARCH_BES_SELECT_MIN_SPREAD` |  |  |  | |
| `OPENRESEARCH_BES_SMOKE_SELECT` |  |  |  | |
| `OPENRESEARCH_BES_SPLICE_ENABLED` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_BLOB_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_BLOB_CODE_PREFIX` |  |  | 1 | |
| `OPENRESEARCH_BLOB_OUTPUT_PREFIX` |  |  | 1 | |

### `OPENRESEARCH_BLOCKED_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_BLOCKED_TERMS_JSON` |  |  |  | |

### `OPENRESEARCH_BOOTSTRAP_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_BOOTSTRAP_MKDIRS` |  |  |  | |
| `OPENRESEARCH_BOOTSTRAP_PIP_TIMEOUT_S` |  |  |  | |

### `OPENRESEARCH_BREV_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_BREV_` |  |  |  | |
| `OPENRESEARCH_BREV_API_KEY` |  |  | 2 | |
| `OPENRESEARCH_BREV_CONTAINER_DISK_GB` |  |  | 1 | `"50"` |
| `OPENRESEARCH_BREV_GPU_COUNT` |  |  | 1 | `"1"` |
| `OPENRESEARCH_BREV_GPU_TYPE` |  |  | 1 | `""` |
| `OPENRESEARCH_BREV_IMAGE` |  |  | 1 | `""` |
| `OPENRESEARCH_BREV_INSTANCE_ID` |  |  |  | |
| `OPENRESEARCH_BREV_REGION` |  |  | 1 | `""` |
| `OPENRESEARCH_BREV_SSH_KEY_PATH` |  |  | 1 | |

### `OPENRESEARCH_BUDGET_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_BUDGET_AWARENESS_MODE` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_CACHE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CACHE_MOUNT` |  |  | 1 | |

### `OPENRESEARCH_CELL_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CELL_BATCH_SCALE` |  | ✅ |  | |
| `OPENRESEARCH_CELL_CHECKPOINT_DIR` |  |  |  | |
| `OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S` |  |  | 1 | `"600"` |
| `OPENRESEARCH_CELL_FINGERPRINT` |  |  |  | |
| `OPENRESEARCH_CELL_GPU_COUNT` |  | ✅ |  | |
| `OPENRESEARCH_CELL_GRAD_CHECKPOINT` |  | ✅ |  | |
| `OPENRESEARCH_CELL_ID` |  |  | 1 | |
| `OPENRESEARCH_CELL_MAX_OOM_RETRIES` |  |  | 1 | |
| `OPENRESEARCH_CELL_MAX_STEPS` |  |  |  | |
| `OPENRESEARCH_CELL_MEM_FRACTION` |  |  | 1 | |
| `OPENRESEARCH_CELL_NOW_ISO` |  |  |  | |
| `OPENRESEARCH_CELL_OOM_BATCH_SCALE_FLOOR` |  |  |  | |
| `OPENRESEARCH_CELL_OOM_BATCH_SCALE_STEP1` |  |  |  | |
| `OPENRESEARCH_CELL_OUTPUT_DIR` |  | ✅ |  | |
| `OPENRESEARCH_CELL_PARAMS` |  | ✅ | 1 | |
| `OPENRESEARCH_CELL_PREEMPT_GRACE_S` |  |  |  | |
| `OPENRESEARCH_CELL_SMOKE_TIMEOUT_S` |  |  | 1 | `"180"` |
| `OPENRESEARCH_CELL_TINY_SLICE` |  |  |  | |

### `OPENRESEARCH_CELLS_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CELLS_ROUTE_RETENTION` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_CHAMPION_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CHAMPION_ARTIFACT` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_CLAUDE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CLAUDE_CLI_BIN` |  |  | 1 | `""` |
| `OPENRESEARCH_CLAUDE_CODE_OAUTH_TOKEN` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_CODE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CODE_REVIEW_GATE` |  |  | 2 | `""` |

### `OPENRESEARCH_CODEX_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CODEX_ALLOWED_TASKS` | ✅ |  |  | `(` |
| `OPENRESEARCH_CODEX_AUTH_PATH` | ✅ |  |  | `""` |
| `OPENRESEARCH_CODEX_CLI_PATH` | ✅ |  | 1 | `""` |
| `OPENRESEARCH_CODEX_MAX_CALLS_PER_RUN` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_CODEX_MAX_OUTPUT_CHARS` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_CODEX_PROFILE` | ✅ |  |  | `"openresearch-readwrite"` |
| `OPENRESEARCH_CODEX_SUBAGENT` | ✅ |  |  | `False` |
| `OPENRESEARCH_CODEX_TIMEOUT_S` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_CONTEXT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CONTEXT_MAP` |  | ✅ | 2 | `""` |

### `OPENRESEARCH_CUSTOM_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_CUSTOM_TOOLS_SECTION` |  |  |  | |

### `OPENRESEARCH_DATABASE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DATABASE_URL` | ✅ | ✅ |  | `"sqlite:///openresearch.db"` |

### `OPENRESEARCH_DEAD_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DEAD_LOSS_DESCENT` |  |  |  | |
| `OPENRESEARCH_DEAD_LOSS_EARLYSTOP` |  |  | 1 | `""` |
| `OPENRESEARCH_DEAD_LOSS_EPS` |  |  |  | |
| `OPENRESEARCH_DEAD_LOSS_MIN` |  |  |  | |
| `OPENRESEARCH_DEAD_LOSS_WINDOW` |  |  |  | |

### `OPENRESEARCH_DEBUG_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DEBUG` | ✅ |  |  | `False` |
| `OPENRESEARCH_DEBUG_RUNS_ROOT` |  |  | 1 | |

### `OPENRESEARCH_DEFAULT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DEFAULT_SANDBOX` |  | ✅ |  | |

### `OPENRESEARCH_DEGENERATE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_DEGENERATE_REWARD_EPSILON` |  |  | 1 | `"1e-6"` |
| `OPENRESEARCH_DEGENERATE_TRAINING_CHECK` |  |  | 1 | `"1"` |

### `OPENRESEARCH_DEMO_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DEMO_SECRET` | ✅ | ✅ |  | `""` |

### `OPENRESEARCH_DETERMINISTIC_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DETERMINISTIC_LEAVES` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_DISABLE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DISABLE_ENV_PIN` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_DISABLE_TORCHRUN_WRAP` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_DISK_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DISK_FLOOR_GB` |  |  | 2 | `"15"` |
| `OPENRESEARCH_DISK_PREFLIGHT_HEADROOM_GB` |  |  | 1 | `"30"` |

### `OPENRESEARCH_DYNAMIC_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_DYNAMIC_GPU` |  | ✅ |  | |
| `OPENRESEARCH_DYNAMIC_GPU_ENABLED` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_DYNAMIC_GPU_FALLBACK_VRAM_GB` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_DYNAMIC_GPU_HEADROOM` | ✅ | ✅ | 1 | `Field(...)` |
| `OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS` | ✅ | ✅ |  | `Field(...)` |

### `OPENRESEARCH_ENV_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ENV_CACHE_DIR` |  |  | 1 | `""` |

### `OPENRESEARCH_ENVIRONMENT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ENVIRONMENT_BUILD_MAX_ATTEMPTS` | ✅ |  |  | `3` |
| `OPENRESEARCH_ENVIRONMENT_BUILD_VALIDATION_ENABLED` | ✅ |  |  | `True` |

### `OPENRESEARCH_ESTIMATE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ESTIMATE_OVERHEAD_MULTIPLIER` |  |  | 1 | `"2.0"` |

### `OPENRESEARCH_EVIDENCE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_EVIDENCE_AUDIT` |  |  | 1 | `""` |
| `OPENRESEARCH_EVIDENCE_FINGERPRINT` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_EVIDENCE_GATE` |  | ✅ | 1 | `"1"` |

### `OPENRESEARCH_EXCLUDE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_EXCLUDE_THEORY_LEAVES` |  |  | 1 | `""` |

### `OPENRESEARCH_EXEC_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_EXEC_COMMAND` |  |  |  | |
| `OPENRESEARCH_EXEC_MODE` |  |  |  | |

### `OPENRESEARCH_EXECUTION_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_EXECUTION_MODE` |  |  | 4 | `""` |
| `OPENRESEARCH_EXECUTION_SMOKE` |  |  | 1 | `""` |

### `OPENRESEARCH_EXECUTOR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_EXECUTOR` |  | ✅ | 1 | |
| `OPENRESEARCH_EXECUTOR_API_KEY` |  |  | 1 | |
| `OPENRESEARCH_EXECUTOR_BASE_URL` |  |  | 1 | |
| `OPENRESEARCH_EXECUTOR_MODEL` |  |  | 1 | |

### `OPENRESEARCH_EXPERIMENT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_EXPERIMENT_GPU_LIVENESS` |  | ✅ |  | |
| `OPENRESEARCH_EXPERIMENT_STALL_S` |  | ✅ |  | |
| `OPENRESEARCH_EXPERIMENT_VENV` |  |  | 1 | |

### `OPENRESEARCH_EXTERNAL_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_EXTERNAL_VALIDATOR` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_FIDELITY_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_FIDELITY_EVIDENCE` |  |  | 1 | `""` |
| `OPENRESEARCH_FIDELITY_MUTATION_TIMEOUT_S` |  |  | 1 | `"60"` |
| `OPENRESEARCH_FIDELITY_TEST_TIMEOUT_S` |  |  | 1 | `"120"` |

### `OPENRESEARCH_FINALIZE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_FINALIZE_REGRADE` |  | ✅ |  | |
| `OPENRESEARCH_FINALIZE_RESCORE` |  |  | 1 | `"1"` |

### `OPENRESEARCH_FLOOR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_FLOOR_HARD` |  |  | 2 | `""` |

### `OPENRESEARCH_FORCE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_FORCE_LLM_PROVIDER` |  |  |  | |
| `OPENRESEARCH_FORCE_SANDBOX` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_FORCE_SINGLE_GPU` | ✅ | ✅ |  | `Field(...)` |

### `OPENRESEARCH_FSDP_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_FSDP_VERSION` |  |  | 1 | `"1"` |

### `OPENRESEARCH_GCP_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_GCP_` |  |  |  | |
| `OPENRESEARCH_GCP_ARTIFACT_REGISTRY` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_BASE_IMAGE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_BOOTSTRAP_PIP_TIMEOUT_S` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_BOOT_TIMEOUT_SECONDS` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_CACHE_MOUNT_PATH` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_CELL_ROUTE` |  |  | 1 | `"1"` |
| `OPENRESEARCH_GCP_CSI_MOUNT_PATH` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_FILESTORE_SHARE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_FILES_CACHE_ENABLED` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_GCS_BUCKET` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_GKE_CLUSTER` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_GPUS_PER_NODE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_GPU_SKUS` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_GPU_USD_PER_HOUR` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_JOB_BACKOFF_LIMIT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_MAX_NODES` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_NAMESPACE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_NODE_POOL_NAME` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_OOM_BATCH_SCALE_FLOOR` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_OOM_BATCH_SCALE_STEP1` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_ORCHESTRATOR_IMAGE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_PENDING_TIMEOUT_SECONDS` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_PER_GPU_VRAM_GB` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_PROJECT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_REGION` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_SERVICE_ACCOUNT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_SPOT_BACKOFF_LIMIT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_TTL_SECONDS_AFTER_FINISHED` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_USE_SPOT` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_GCP_WATCH_POLL_INTERVAL_S` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_GPU_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_GPU_DEVICE_IDS` |  | ✅ | 1 | |
| `OPENRESEARCH_GPU_MODE` |  |  |  | |
| `OPENRESEARCH_GPU_PARALLELISM` |  |  | 1 | |

### `OPENRESEARCH_GPUS_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_GPUS_PER_CELL` |  |  | 1 | `"1"` |

### `OPENRESEARCH_GRADER_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_GRADER_` |  | ✅ |  | |
| `OPENRESEARCH_GRADER_BACKEND` |  | ✅ | 3 | `""` |
| `OPENRESEARCH_GRADER_DIGEST` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_GRADER_MODEL` |  | ✅ | 3 | `""` |
| `OPENRESEARCH_GRADER_SAMPLES` |  | ✅ | 2 | `"1"` |

### `OPENRESEARCH_HF_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_HF_CACHE_CAP_GB` |  |  |  | |

### `OPENRESEARCH_HOST_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_HOST` | ✅ |  |  | `"127.0.0.1"` |

### `OPENRESEARCH_HYBRID_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_HYBRID_EXEC_ROUTE` |  |  | 1 | `""` |

### `OPENRESEARCH_INJECT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_INJECT_STEERING` |  |  | 1 | `""` |

### `OPENRESEARCH_LEAF_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_LEAF_ACTUATE` |  | ✅ |  | |
| `OPENRESEARCH_LEAF_ACTUATE_MAX_COST` |  | ✅ |  | |
| `OPENRESEARCH_LEAF_ACTUATE_SEEDS` |  | ✅ |  | |
| `OPENRESEARCH_LEAF_EVIDENCE_GATE` |  |  | 2 | `""` |
| `OPENRESEARCH_LEAF_SEED_MAX` |  | ✅ |  | |
| `OPENRESEARCH_LEAF_TRIAGE` |  | ✅ |  | |

### `OPENRESEARCH_LIFECYCLE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_LIFECYCLE_DRIVE` |  |  | 1 | `""` |
| `OPENRESEARCH_LIFECYCLE_LEDGER` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_LIFECYCLE_MAX_IMPROVE` |  |  | 1 | `""` |
| `OPENRESEARCH_LIFECYCLE_PRIMARY` |  |  | 1 | `""` |

### `OPENRESEARCH_LLM_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_LLM_AUTH_STRATEGY` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_LLM_PROVIDER` |  |  | 1 | |

### `OPENRESEARCH_LOCAL_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_LOCAL_TORCH_INDEX_URL` |  | ✅ | 1 | `"https://download.pytorch.org/whl/cu121"` |

### `OPENRESEARCH_LOG_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_LOG_DIR` |  |  | 1 | |

### `OPENRESEARCH_MATRIX_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_MATRIX_FINALIZE_RESERVE_S` |  |  | 2 | `"2700"` |

### `OPENRESEARCH_MAX_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_MAX_GPU_USD_PER_HOUR` | ✅ | ✅ |  | `Field(...)` |
| `OPENRESEARCH_MAX_POD_SECONDS` |  |  | 1 | |
| `OPENRESEARCH_MAX_RLM_ITERATIONS` |  |  | 2 | `""` |
| `OPENRESEARCH_MAX_RUN_GPU_USD` | ✅ | ✅ |  | `Field(...)` |
| `OPENRESEARCH_MAX_SCOPE_FAILURE_REPEATS` |  |  | 1 | `"2"` |
| `OPENRESEARCH_MAX_WALL_CLOCK_S` |  |  |  | |

### `OPENRESEARCH_METRIC_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_METRIC_PROVENANCE` |  |  | 1 | `"true"` |
| `OPENRESEARCH_METRIC_REALITY_SMOKE` |  |  | 2 | `""` |
| `OPENRESEARCH_METRIC_SEMANTICS_GUARD` |  |  | 1 | `""` |

### `OPENRESEARCH_METRICS_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_METRICS_COMPLETENESS_CHECK` |  |  | 1 | `"1"` |

### `OPENRESEARCH_MIN_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_MIN_DISK_GB` |  |  | 1 | `"10"` |
| `OPENRESEARCH_MIN_REAL_TRAIN_STEPS` |  |  | 1 | `"5"` |
| `OPENRESEARCH_MIN_REPAIR_ITERATIONS` |  | ✅ | 2 | `"2"` |
| `OPENRESEARCH_MIN_RUBRIC_ITERATIONS` | ✅ | ✅ | 1 | `Field(...)` |
| `OPENRESEARCH_MIN_SEEDS_FOR_CONTRADICTION` |  |  | 1 | |
| `OPENRESEARCH_MIN_TRAIN_STEPS` |  |  | 1 | `"0"` |
| `OPENRESEARCH_MIN_TRAIN_WALL_S` |  |  | 1 | `"0"` |

### `OPENRESEARCH_MODEL_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_MODEL_NAME` |  |  | 1 | `""` |
| `OPENRESEARCH_MODEL_PREFLIGHT` |  |  | 1 | `"1"` |

### `OPENRESEARCH_NCCL_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_NCCL_IB_DISABLE` |  |  |  | |
| `OPENRESEARCH_NCCL_P2P_DISABLE` |  |  |  | |

### `OPENRESEARCH_NEGATIVE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_NEGATIVE_LESSONS` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_NOTIFY_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_NOTIFY_WEBHOOK_URL` |  |  |  | |

### `OPENRESEARCH_OAUTH_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_OAUTH_AUTODRIVE` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_OAUTH_FALLBACK_MODEL` |  |  | 1 | `""` |

### `OPENRESEARCH_OOM_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_OOM_ENFORCE` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_OPENAI_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_OPENAI_ADMIN_KEY` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_OPENAI_API_KEY` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_OPENAI_DEFAULT_MODEL` | ✅ |  |  | `"gpt-4o"` |
| `OPENRESEARCH_OPENAI_REASONING_MODEL` | ✅ |  |  | `"o4-mini"` |

### `OPENRESEARCH_ORPHAN_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ORPHAN_GUARD` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_PAPER_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PAPER_EXTRACTION_VISION_MODEL` | ✅ |  |  | `"claude-sonnet-4-6"` |
| `OPENRESEARCH_PAPER_HINT_INVARIANTS_JSON` |  |  | 1 | `""` |
| `OPENRESEARCH_PAPER_TEXT_PATH` |  |  | 2 | `""` |

### `OPENRESEARCH_PER_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PER_MODEL_STATUS_GATE` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_POD_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_POD_SWEEP_ENABLED` |  |  | 2 | `"true"` |
| `OPENRESEARCH_POD_SWEEP_INTERVAL_S` |  |  | 1 | `"1800"` |
| `OPENRESEARCH_POD_SWEEP_MAX_AGE_S` |  |  | 2 | `"7200"` |

### `OPENRESEARCH_PORT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PORT` | ✅ |  |  | `8000` |

### `OPENRESEARCH_POSITIVE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_POSITIVE_RECIPES` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_PRE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PRE_EMIT_STALL_S` |  |  | 1 | `""` |

### `OPENRESEARCH_PREFLIGHT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PREFLIGHT_SMOKE` |  |  | 1 | `"on"` |

### `OPENRESEARCH_PRELOAD_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PRELOAD_ASSETS` |  |  | 1 | `"1"` |

### `OPENRESEARCH_PRIMITIVE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PRIMITIVE_CACHE` |  |  |  | |
| `OPENRESEARCH_PRIMITIVE_LLM_MODEL` |  |  | 1 | `""` |

### `OPENRESEARCH_PRIOR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PRIOR_ATTEMPT_EVIDENCE` |  |  |  | |

### `OPENRESEARCH_PROVIDER_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_PROVIDER_FALLBACK_DISABLED` | ✅ |  |  | `False` |

### `OPENRESEARCH_RDR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RDR_PREFLIGHT_GATE` | ✅ |  |  | `Field(...)` |
| `OPENRESEARCH_RDR_PREFLIGHT_MAX_REGENS` | ✅ |  |  | `Field(...)` |

### `OPENRESEARCH_REPAIR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_REPAIR_MAX_ITERATIONS` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_REPO_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_REPO_CLONE_LFS` | ✅ | ✅ |  | `False` |
| `OPENRESEARCH_REPO_CLONE_MAX_MB` | ✅ | ✅ |  | `2048` |
| `OPENRESEARCH_REPO_CLONE_TIMEOUT_S` | ✅ | ✅ |  | `300` |
| `OPENRESEARCH_REPO_URL` |  |  | 1 | `""` |

### `OPENRESEARCH_REPORT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_REPORT_CLAIM_GATE` |  |  | 1 | `""` |

### `OPENRESEARCH_REPRODUCTION_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_REPRODUCTION_MODE` | ✅ | ✅ | 1 | `"adapt"` |

### `OPENRESEARCH_REQUIRE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_REQUIRE_VALIDATED_ROOT` |  |  | 1 | `""` |

### `OPENRESEARCH_RESUME_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RESUME_CELLS` |  |  | 2 | `""` |
| `OPENRESEARCH_RESUME_FORCE_CELLS` |  |  | 1 | `""` |

### `OPENRESEARCH_REUSE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_REUSE_RUBRIC` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_RL_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RL_SCAFFOLD` |  |  | 1 | `""` |

### `OPENRESEARCH_RLM_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RLM_CLI_TIMEOUT_S` |  |  | 1 | `""` |
| `OPENRESEARCH_RLM_EMPTY_TURN_FALLBACK` |  |  | 1 | `"1"` |
| `OPENRESEARCH_RLM_ROOT_MODEL` |  | ✅ |  | |
| `OPENRESEARCH_RLM_ROOT_MODEL_NAME` |  |  | 1 | `""` |
| `OPENRESEARCH_RLM_ROOT_SDK_MAX_RETRIES` |  |  | 1 | `""` |
| `OPENRESEARCH_RLM_ROOT_SLUG_KIMI` |  |  |  | |
| `OPENRESEARCH_RLM_ROOT_SLUG_QWEN` |  |  |  | |
| `OPENRESEARCH_RLM_ROOT_TRANSPORT` |  |  | 1 | `"cli"` |
| `OPENRESEARCH_RLM_STUB_PRIMITIVES` |  |  | 2 | |
| `OPENRESEARCH_RLM_SUB_SLUG_KIMI` |  |  |  | |
| `OPENRESEARCH_RLM_SUB_SLUG_QWEN` |  |  |  | |

### `OPENRESEARCH_ROLE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ROLE_MODELS` |  | ✅ | 2 | `""` |

### `OPENRESEARCH_ROOT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ROOT_EFFORT` |  |  | 1 | `"high"` |

### `OPENRESEARCH_RUBRIC_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RUBRIC_DECLINE_ADVISORY` |  |  | 1 | `""` |
| `OPENRESEARCH_RUBRIC_MAX_IMPROVEMENT_ITERATIONS` | ✅ |  |  | `2` |
| `OPENRESEARCH_RUBRIC_PLATEAU_EPSILON` |  |  | 1 | `"0.005"` |
| `OPENRESEARCH_RUBRIC_PLATEAU_WINDOW` |  |  | 1 | `"3"` |
| `OPENRESEARCH_RUBRIC_TARGET_SCORE` | ✅ |  |  | `0.70` |
| `OPENRESEARCH_RUBRIC_VERIFIER_ENABLED` | ✅ |  |  | `True` |
| `OPENRESEARCH_RUBRIC_VERIFIER_MODEL` | ✅ |  | 1 | `""` |

### `OPENRESEARCH_RUN_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S` |  |  | 1 | `""` |
| `OPENRESEARCH_RUN_SMALL_FOOTPRINT_GB` |  |  | 1 | `"5"` |
| `OPENRESEARCH_RUN_TITLE` |  |  | 1 | |

### `OPENRESEARCH_RUNPOD_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RUNPOD_` |  | ✅ |  | |
| `OPENRESEARCH_RUNPOD_API_BASE_URL` | ✅ |  |  | `"https://rest.runpod.io/v1"` |
| `OPENRESEARCH_RUNPOD_API_KEY` | ✅ | ✅ | 5 | `Field(...)` |
| `OPENRESEARCH_RUNPOD_AUTO_FALLBACK` |  |  |  | |
| `OPENRESEARCH_RUNPOD_BOOTSTRAP_COMMAND` | ✅ |  |  | `""` |
| `OPENRESEARCH_RUNPOD_BOOT_TIMEOUT_SECONDS` | ✅ |  |  | `900` |
| `OPENRESEARCH_RUNPOD_CLOUD_TYPE` |  | ✅ | 1 | `"SECURE"` |
| `OPENRESEARCH_RUNPOD_CONTAINER_DISK_GB` | ✅ |  | 1 | `50` |
| `OPENRESEARCH_RUNPOD_DATA_CENTER_IDS` | ✅ |  |  | `""` |
| `OPENRESEARCH_RUNPOD_DELETE_ON_DESTROY` | ✅ |  |  | `True` |
| `OPENRESEARCH_RUNPOD_GPU_COUNT` | ✅ |  | 1 | `1` |
| `OPENRESEARCH_RUNPOD_GPU_TYPE` | ✅ |  | 1 | `"NVIDIA GeForce RTX 4090"` |
| `OPENRESEARCH_RUNPOD_IMAGE` | ✅ | ✅ | 1 | `"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-` |
| `OPENRESEARCH_RUNPOD_NETWORK_VOLUME_ID` | ✅ |  |  | `""` |
| `OPENRESEARCH_RUNPOD_POD_ID` | ✅ |  |  | `""` |
| `OPENRESEARCH_RUNPOD_SKIP_BUILD` |  | ✅ | 1 | `"1"` |
| `OPENRESEARCH_RUNPOD_SSH_KEY_PATH` | ✅ | ✅ |  | `""` |
| `OPENRESEARCH_RUNPOD_SSH_PUBLIC_KEY` | ✅ |  |  | `""` |
| `OPENRESEARCH_RUNPOD_SSH_USER` | ✅ |  |  | `"root"` |
| `OPENRESEARCH_RUNPOD_STALL_WARN` |  | ✅ | 1 | `"1"` |
| `OPENRESEARCH_RUNPOD_VOLUME_GB` | ✅ |  | 1 | `20` |
| `OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH` | ✅ |  | 5 | `"/workspace"` |

### `OPENRESEARCH_RUNS_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_RUNS_DIR` |  |  |  | |
| `OPENRESEARCH_RUNS_RETENTION_DAYS` |  | ✅ |  | |
| `OPENRESEARCH_RUNS_ROOT` |  |  | 7 | `""` |

### `OPENRESEARCH_SCOPE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SCOPE_INCLUSION` |  | ✅ |  | |
| `OPENRESEARCH_SCOPE_INCLUSION_EXCLUDE` |  | ✅ | 1 | `""` |
| `OPENRESEARCH_SCOPE_SPEC_JSON` |  |  | 2 | `""` |

### `OPENRESEARCH_SDAR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SDAR_BASELINES` |  |  |  | |

### `OPENRESEARCH_SDK_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SDK_HERMETIC` |  |  | 1 | `"true"` |
| `OPENRESEARCH_SDK_ISOLATION_DISABLED` |  |  | 1 | `""` |
| `OPENRESEARCH_SDK_MAX_RETRIES` |  |  | 1 | `""` |

### `OPENRESEARCH_SEARCH_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SEARCH_QA_` |  |  |  | |
| `OPENRESEARCH_SEARCH_QA_DENSE` |  |  | 1 | `""` |
| `OPENRESEARCH_SEARCH_QA_ENCODER` |  |  | 1 | `""` |
| `OPENRESEARCH_SEARCH_QA_INDEX_DIR` |  |  | 1 | `""` |
| `OPENRESEARCH_SEARCH_QA_INDEX_REPO` |  |  | 1 | `""` |
| `OPENRESEARCH_SEARCH_QA_INDEX_REPO_TYPE` |  |  | 1 | `"dataset"` |

### `OPENRESEARCH_SEED_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SEED_ALL_MODELS` |  |  | 1 | `""` |
| `OPENRESEARCH_SEED_BEST_ATTEMPT` |  |  |  | |
| `OPENRESEARCH_SEED_MODELS_MAX` |  |  | 1 | `"1"` |
| `OPENRESEARCH_SEED_REPLICATION` |  |  | 1 | `""` |

### `OPENRESEARCH_SKIP_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SKIP_CRED_PREFLIGHT` |  |  | 1 | `""` |

### `OPENRESEARCH_SMOKE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SMOKE_MAX_CELLS` |  |  | 1 | `""` |
| `OPENRESEARCH_SMOKE_STEPS` |  |  | 2 | `'0'` |
| `OPENRESEARCH_SMOKE_TIMEOUT_S` |  |  | 1 | `""` |

### `OPENRESEARCH_STABLE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_STABLE_RUN_ID` |  |  | 1 | `""` |

### `OPENRESEARCH_STUB_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_STUB_METRICS_GUARD` |  | ✅ | 1 | `""` |

### `OPENRESEARCH_SUBAGENT_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SUBAGENT_TRANSPORT_BACKOFF_S` |  |  | 1 | `"8"` |
| `OPENRESEARCH_SUBAGENT_TRANSPORT_RETRIES` |  |  | 1 | `""` |

### `OPENRESEARCH_SUBRLM_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_SUBRLM_OPENAI_TIMEOUT_S` |  | ✅ | 1 | |

### `OPENRESEARCH_TARGET_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_TARGET_BEST_FLOOR` |  |  |  | |

### `OPENRESEARCH_TRAINER_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_TRAINER_GPUS` |  |  | 1 | `'1'` |
| `OPENRESEARCH_TRAINER_VERSION` |  |  |  | |

### `OPENRESEARCH_TWO_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_TWO_AXIS_VERDICT` |  | ✅ | 3 | `""` |

### `OPENRESEARCH_UPDATE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_UPDATE_CALIBRATION` |  |  | 1 | `""` |

### `OPENRESEARCH_USE_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_USE_AUTHOR_REPO` | ✅ | ✅ |  | `False` |

### `OPENRESEARCH_VALIDATOR_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_VALIDATOR_BACKEND` |  | ✅ | 3 | `""` |
| `OPENRESEARCH_VALIDATOR_CHECK_REPORT` |  |  | 1 | `""` |
| `OPENRESEARCH_VALIDATOR_MODEL` |  |  | 5 | `""` |
| `OPENRESEARCH_VALIDATOR_PANEL_N` |  |  | 1 | `""` |

### `OPENRESEARCH_VERIFICATION_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_VERIFICATION_PROVIDER` |  |  |  | |

### `OPENRESEARCH_VLLM_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_VLLM_HOST` |  |  | 1 | `'localhost'` |
| `OPENRESEARCH_VLLM_PORT` |  |  | 2 | `'8000'` |

### `OPENRESEARCH_VRAM_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_VRAM_OVERRIDE_GB` |  | ✅ | 4 | `""` |

### `OPENRESEARCH_WATCHDOG_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_WATCHDOG_DISABLED` |  |  |  | |
| `OPENRESEARCH_WATCHDOG_HARD_CEILING_S` |  |  | 1 | `""` |
| `OPENRESEARCH_WATCHDOG_KILL_SECONDS` |  |  |  | |
| `OPENRESEARCH_WATCHDOG_MAX_SOFT_RECOVERIES` |  |  |  | |
| `OPENRESEARCH_WATCHDOG_POLL_INTERVAL_SECONDS` |  |  |  | |
| `OPENRESEARCH_WATCHDOG_WARN_SECONDS` |  |  |  | |

### `OPENRESEARCH_WEBSHOP_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_WEBSHOP_` |  |  |  | |
| `OPENRESEARCH_WEBSHOP_PYTHON` |  |  | 2 | |
| `OPENRESEARCH_WEBSHOP_REPO_URL` |  |  | 2 | `"https://github.com/princeton-nlp/WebSho` |

### `OPENRESEARCH_ZERO_*`

| Flag | cfg | doc | sites | default |
|---|:--:|:--:|:--:|---|
| `OPENRESEARCH_ZERO_METRICS_GUARD` |  | ✅ | 1 | `""` |

