# GEPA Phase 0 — Pre-implementation Audit

> Status: locked (2026-05-26)
> Branch: `feat/gepa-phase0`
> Pairs with: `docs/superpowers/specs/2026-05-25-gepa-prompt-optimization-design.md`
> Purpose: resolve every spec ambiguity BEFORE the adapter is written. Phase 1
> may not start until this doc is signed off.

## 0.1 — `gepa` library audit

- Pinned version: **`gepa==0.1.1`** (`backend/requirements.txt`).
- PyPI metadata (verified 2026-05-26 against `pypi.org/pypi/gepa/json`):
  - Core deps: minimal — `numpy`, `pydantic`. No transport client by default.
  - `extras_require["full"]` pulls `litellm>=1.81.0` on py≥3.14, `datasets`, `mlflow-skinny`, `wandb`. ReproLab does NOT need `[full]` — we own our task LM transport via the existing root-model factory.
- **Local install blocker (not a project blocker):** brew-built Python 3.14.5 on this dev machine has a broken `pyexpat` ABI (`Symbol not found: _XML_SetAllocTrackerActivationThreshold`). Both `pip` and `uv` fail to install `gepa` here. Real install env (Docker image + RunPod sidecar) is unaffected. Phase 1 first action: `docker compose build` and verify `python -c "import gepa; print(gepa.__version__)"` succeeds in-container.
- **Open question #4 (per-example vs per-batch reflection)** — to be answered by reading `gepa.core.adapter.GEPAAdapter.make_reflective_dataset` source after install. Until then assume per-batch and budget GPT-5 context accordingly (`minibatch ≤ 50` should fit < 100k tokens after `TraceMinimizer` redaction).

## 0.3 — AgentSpec prompt-snapshot hazard audit

The spec calls out `AGENT_REGISTRY["improvement-orchestrator"]` (registry.py:104) as a cache hazard. Audit of `backend/agents/registry.py` confirms the hazard exists on **two distinct snapshot points**, affecting **four agents**:

### Snapshot point 1 — `AgentSpec.prompt` field (import-time)

`AGENT_REGISTRY` dict literal copies the module global string into `AgentSpec.prompt` at import time. Monkey-patching the source module global will NOT update the registry entry.

| Agent ID | `AgentSpec.prompt` source | registry.py line |
|---|---|---|
| `baseline-implementation` | `BASELINE_IMPLEMENTATION_PROMPT` | 89 |
| `rubric-verifier` | `RUBRIC_VERIFIER_PROMPT` | 99 |
| `improvement-orchestrator` | `IMPROVEMENT_ORCHESTRATOR_PROMPT` | 108 |
| `improvement-path` | `IMPROVEMENT_PATH_PROMPT` | 118 |

### Snapshot point 2 — `get_agent_definitions()` (import-time of consumer)

`get_agent_definitions()` (registry.py:125-140) reads `spec.prompt` and copies into `AgentDefinition(prompt=spec.prompt)`. Callers cache the returned dict; subsequent mutation has no effect.

### Required Phase 1 fix

`AgentSpec.to_runtime_spec(provider, *, prompt_override: str | None = None)` — when `prompt_override` is non-None, use it instead of `self.prompt` for `AgentRuntimeSpec.instructions`. `get_agent_definitions()` must be re-derived per-invocation when a `PromptOverrideContext` is active, OR `invoke_agent` must bypass it and call `to_runtime_spec` directly with the override.

### Spec gap (filed for ratification)

Spec §3 Lane B names three prompts: `IMPROVEMENT_ORCHESTRATOR_PROMPT`, `ADAPTIVE_POOL_GENERATION_PROMPT`, `ADAPTIVE_RERANK_PROMPT`. `backend/agents/prompts/improvement.py` ACTUALLY defines **six** prompt constants:

| Constant | improvement.py line | In spec Lane B? | Decision |
|---|---:|---|---|
| `IMPROVEMENT_ORCHESTRATOR_PROMPT` | 4 | ✓ | Lane B v1 |
| `ADAPTIVE_POOL_GENERATION_PROMPT` | 46 | ✓ | Lane B v1 |
| `ADAPTIVE_RERANK_PROMPT` | 97 | ✓ | Lane B v1 |
| `IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT` | 127 | ✗ | **Add to Lane B v1.** Load-bearing for multi-round adaptive mode. Spec omission. |
| `IMPROVEMENT_PATH_PROMPT` | 174 | ✗ | **Lane B v1.1** (defer). Drives the path-execution sub-agent; mutating it without the orchestrator landing first is high-risk. |
| `COMPOSITION_AGENT_PROMPT` | 210 | ✗ | **Out of scope v1.** Composition path is not on the default lab flow. |

Lane B v1 component count: **4** (the original 3 plus `improvement.orchestrator_round_n.body`).

### Other call sites that must be checked before Phase 1

`get_agent_definitions()` is invoked at startup by the `ClaudeAgentOptions` builder. Search for `AGENT_REGISTRY[` and `get_agent_definitions(` in `backend/agents/runtime/`, `backend/agents/rdr/`, `backend/agents/hybrid/`. Any second snapshot point not listed above is a Phase 1 work-item.

## 0.4 — `primitive_cache` audit

File: `backend/agents/rlm/primitive_cache.py`.

Cache key (`make_key`, line 180): `sha256(json.dumps(payload, sort_keys=True, default=str))[:32]` — namespaced by primitive name and `_CACHE_VERSION`. **No prompt text, no candidate hash, no surface id in the key today.**

Validators present (line 160-167) — these are the primitives that **are cached today**:

| Primitive | Cached? | Prompt-dependent? | GEPA Phase 1 cache rule |
|---|---|---|---|
| `understand_section` | ✓ | ✗ — payload is raw text slice; primitive is pure LLM extraction | **No change.** Cache as today. |
| `extract_hyperparameters` | ✓ | ✗ — same | **No change.** |
| `detect_environment` | ✓ | ✗ — input is method-spec dict, not prompt-derived | **No change.** |
| `plan_reproduction` | ✓ | ◐ — output is contract; influenced by environment but not by any GEPA-target prompt | **No change v1.** Re-check when Lane C lands. |
| `verify_against_rubric` | ✓ | ◐ — depends on `RUBRIC_VERIFIER_PROMPT` (immutable in v1; Lane D out of scope) | **No change v1.** Re-check if Lane D ever enters scope. |
| `implement_baseline` | ✓ | ✓ — depends on `BASELINE_IMPLEMENTATION_PROMPT` (**Lane C target**) | **Must include `candidate_hash + surface_id` in cache key, OR disable cache for Lane C evals.** Validation runs (§4.6) always disable. |

`propose_improvements` is **not currently cached** (no validator in `_CACHE_VALIDATORS`). Lane B may or may not need to add caching; v1 leaves it uncached so candidate evaluations always re-execute.

### Phase 1 required change to `primitive_cache.make_key`

Add an optional `surface_salt: str | None = None` argument. When passed, it's appended to the payload before hashing. The Lane C adapter (Phase 4) wraps `implement_baseline` calls in a `PrimitiveCacheSurfaceContext` that injects `surface_salt = f"{candidate_hash}:{surface_id}"`. Outside an active context, behavior is identical to today (byte-for-byte cache key compatibility).

Validation runs use `REPROLAB_PRIMITIVE_CACHE=disabled` for the whole subprocess.

### Race / append-safety note

`primitive_cache.jsonl` is append-only; `ProcessPoolExecutor` Lane B concurrency (spec §4.5) is safe IF each parallel eval writes to a **different** `project_id` directory. The cache file is per-run, not global, so this is satisfied by candidate-scoped run dirs (`runs/_gepa/<ts>/candidates/<hash>/evals/<paper_id>/`).

## 0.5 — Paper archetype labels (open question #5)

Decision: `tests/fixtures/papers/<arxiv_id>/archetype.txt` per spec §9 Q5. Seed labels written this phase:

| arxiv_id | archetype | Why |
|---|---|---|
| `2605.15155` (SDAR) | `rl-agent` | Trains an RL agent across ALFWorld/WebShop/Search-QA |
| `1502.04623` (VAE on Frey Face) | `cv-ablation` | Variational autoencoder image-reconstruction ablation |
| `1412.6980` (Adam) | `optimization` | Optimizer algorithm with synthetic-curve benchmarks |

Schema: file contains exactly one of `rl-agent | nlp-eval | cv-ablation | optimization | other`, newline-terminated, no other content. Trainset loader reads via `Path(...).read_text().strip()`; archetype is a **trace field**, not a candidate input (per spec §9 Q5).

## 0.6 — Driver CLI flag spec

`scripts/optimize_prompts_gepa.py` (Phase 2 deliverable):

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--lane` | enum {A,B,C} | required | Which GEPA lane to optimize |
| `--task-lm` | str | `openai/gpt-5` | Model used for in-loop evaluation runs |
| `--reflection-lm` | str | `openai/gpt-5` | Model used by GEPA reflection (G4 — may differ from task LM) |
| `--trainset` | path | required | JSON file listing arxiv IDs |
| `--valset` | path | required | JSON file listing arxiv IDs; must be ≥30% of total (G2) |
| `--held-out` | path | required | Single arxiv ID used for §4.6 validation run |
| `--max-metric-calls` | int | `100` | Passed to `gepa.optimize` |
| `--max-merge-invocations` | int | `5` | Passed to `gepa.optimize` |
| `--max-usd-per-eval` | float | `0.50` | Passed to `EvalBudgetEnforcer` → `RunBudget` |
| `--max-parallel-pods` | int | env `REPROLAB_GEPA_MAX_PARALLEL_PODS` else `2` | Lane A/C only |
| `--cache-strategy` | enum {scoped,disabled} | `scoped` for Lane B, `disabled` for validation | Whether to use `surface_salt` or `REPROLAB_PRIMITIVE_CACHE=disabled` |
| `--run-dir` | path | `runs/_gepa/<UTC ISO timestamp>` | Output dir (spec §4.5 artifact tree) |
| `--seed` | int | `0` | Passed to `gepa.optimize` |

Refused at parse time:
- `len(valset) / (len(trainset)+len(valset)) < 0.30` → G2 violation.
- Any arxiv_id appearing in both `trainset` and `valset` → leakage.
- `--held-out` arxiv_id appearing in `trainset` or `valset` → §4.6 contamination.
- `--lane A` or `--lane C` with `--max-parallel-pods > REPROLAB_GEPA_MAX_PARALLEL_PODS` → enforced cap.

## Sign-off checklist (gates Phase 1)

- [ ] `gepa==0.1.1` pin lands in `backend/requirements.txt` ✓ (this PR)
- [ ] `import gepa` succeeds inside the Docker image (Phase 1.0 — manual verify)
- [ ] Hazard table (0.3) reviewed; no additional snapshot points found
- [ ] Cache audit table (0.4) reviewed; `surface_salt` API approved
- [ ] Archetype labels reviewed; seed set agreed
- [ ] CLI flag spec (0.6) reviewed

Once all six are checked, branch `feat/gepa-phase1-foundation` can open against this doc.
