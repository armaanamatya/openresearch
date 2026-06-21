# Workstream A — Provider-Unification Audit (Azure AI Foundry first)

> **Type:** READ-ONLY audit. No production code changed. This doc gates design work for unifying all LLM tiers behind one "Azure AI Foundry first" provider-resolution surface.
> **Date:** 2026-06-20 · **Branch:** `feat/bes-conversion-correctness` · **Scope:** the five LLM tiers (root, executor, sub-agents, verifier+grader, navigation accelerator).

---

## 1. Executive summary

This is **NOT** a pure per-tier patchwork — there is already a real partial shared layer, but it is asymmetric and three knobs bypass it.

- `backend/agents/rlm/role_models.py` (`resolve_role_models` → `RoleSelection`) is a **pure, unified per-role provider picker** that already covers four roles (planner / executor / verifier / grader) from one env vocabulary (`--models` / `OPENRESEARCH_ROLE_MODELS`), and **already accepts `azure-foundry`/`grok` tokens** for the three sub-roles (`role_models.py:65`, `:75-83`, `:127-134`).
- `backend/agents/runtime/factory.py::make_runtime` (`factory.py:511-569`) is the matching **runtime builder** and **already has first-class `azure` and `azure-foundry` branches** (`factory.py:525-552`).
- `backend/agents/rlm/grader_transport.py::build_transport_client` (`grader_transport.py:116-261`) is the **client builder for verifier + grader** and **already has an `azure-foundry`/`grok` branch** routing through the canonical `foundry_endpoint.resolve_foundry_credentials` (`grader_transport.py:219-243`).

So Foundry is already wired for **root, executor (unified-surface only), verifier, and grader**. The residual gaps are concentrated in: (a) the **navigation accelerator** (`accelerator.py` — no Foundry mode at all), (b) the **legacy `OPENRESEARCH_EXECUTOR` knob** (`executor.py` — azure but no foundry), and (c) the **generic sub-agent provider gate** `selected_provider()` (`factory.py:227-243` — accepts only anthropic/openai, so azure AND foundry are unselectable through it). Section 4 names the single reusable seam; Section 5 ranks the gaps.

---

## 2. Per-tier provider-selection table

A tier can be **buildable** (a runtime/client exists) yet **not selectable** (no env knob routes to it). The table separates the two.

| Tier | Selection env var(s) | Resolver fn : line | Default | azure_openai wired? | azure_foundry wired? | Routes via factory.make_runtime / shared layer? |
|---|---|---|---|---|---|---|
| **1. Root** (rlm lib, raw HTTP) | `OPENRESEARCH_RLM_ROOT_MODEL` / `--model` / `--models planner=` | `models.py::resolve_root_model` `models.py:586` | `gpt-5` if `OPENAI_API_KEY` else featherless → claude-oauth → qwen3-coder (`models.py:603-617`) | **YES** — `azure-gpt-4o` entry, `rlm_backend="azure_openai"`, `_inject_azure_kwargs` (`models.py:225-238`, `:357-399`) | **YES** — `azure-foundry` entry + `_inject_foundry_kwargs` (`models.py:239-252`, `:438-467`) | **No** — own registry path; builds `rlm.RLM(...)` not a runtime. Does NOT use make_runtime. |
| **2. Executor** (`implement_baseline`) | `--models executor=` (unified) **or** legacy `OPENRESEARCH_EXECUTOR` | unified: `run.py::_resolve_agent_runtime` `run.py:382-394`; legacy: `executor.py::resolve_executor` `executor.py:66` | None ⇒ Sonnet via `_resolve_agent_runtime` claude path (`run.py:413-419`) | **YES (both paths)** — unified→`make_runtime("azure")`; legacy `_AZURE_MODES` (`executor.py:39`, `:77-97`) | **PARTIAL** — unified→`make_runtime("azure-foundry")` (`run.py:388-391`); legacy `OPENRESEARCH_EXECUTOR` has **NO foundry mode** (`executor.py:39` only `_AZURE_MODES`) | **Unified path: YES** (`make_runtime`). **Legacy path: NO** (own `AzureOpenAiAgentRuntime()` / vLLM `OpenAiAgentRuntime`). |
| **3. Generic sub-agents** (claude-agent-sdk, non-executor) | `OPENRESEARCH_LLM_PROVIDER` / `settings.llm_provider` | `factory.py::selected_provider` `factory.py:227-243` | `anthropic` (`factory.py:233`) | **NO via this gate** — `selected_provider` rejects anything but anthropic/openai (`factory.py:236-243`). Azure runtime exists but is unreachable through `selected_provider`. | **NO** — same gate rejects it. Foundry runtime exists but only the executor-role path passes the string to `make_runtime`. | Partially — calls `make_runtime`, but the *provider string* is gated by `selected_provider` to anthropic/openai. Confirmed no non-executor caller passes "azure"/"azure-foundry" (grep §below). |
| **4a. Verifier** (`verify_against_rubric` judge) | `OPENRESEARCH_RUBRIC_VERIFIER_MODEL` / `rubric_verifier_model` + `--models verifier=` | `run.py:2102-2133` → `grader_transport.build_transport_client` | None ⇒ inherits planner client (`run.py:2124` guard) | **YES** — `build_transport_client` `azure` branch (`grader_transport.py:196-217`) | **YES** — `build_transport_client` `azure-foundry`/`grok` branch via `resolve_foundry_credentials` (`grader_transport.py:219-243`) | **YES** — `build_transport_client` is the shared client layer; provider chosen by `_subrole_backend` (`run.py:2119-2122`). |
| **4b. Grader** (leaf scorer) | `OPENRESEARCH_GRADER_BACKEND` / `OPENRESEARCH_GRADER_MODEL` + `--models grader=` | `grader_transport.py::build_grader_client` `grader_transport.py:264-294` → `build_transport_client` | Both unset ⇒ rides root/planner client unchanged (`grader_transport.py:150-151`) | **YES** — same `azure` branch (`grader_transport.py:196-217`) | **YES** — same `azure-foundry`/`grok` branch (`grader_transport.py:219-243`) | **YES** — shared `build_transport_client`. `run.py:2139-2142` folds a `--models grader=` pick into the `OPENRESEARCH_GRADER_*` env this reads. |
| **5. Navigation accelerator** (`rlm_query`/`llm_query`) | `OPENRESEARCH_ACCELERATOR` (+ `_BASE_URL`/`_MODEL`/`_API_KEY`) | `accelerator.py::resolve_accelerator` `accelerator.py:410-471` | `off` ⇒ None ⇒ default Sonnet/OAuth path | **YES** — `_resolve_azure` mode reads `AZURE_OPENAI_*` (`accelerator.py:303-346`, dispatched `:458-459`) | **NO** — no Foundry mode; modes are off/auto/local/runpod/azure/endpoint (`accelerator.py:445-471`). `_resolve_azure` never calls `resolve_foundry_credentials`. `endpoint` mode could be *manually* pointed at a Foundry URL but bypasses the canonical resolver. | **No** — own `AcceleratorEndpoint` + `build_accelerator_client` (`accelerator.py:524-580`), entirely separate from factory/role_models. |

**make_runtime caller grep (backs the tier-3 "unselectable" cell):**
```
backend/agents/runtime/invoke.py:135     make_runtime(provider)          # provider gated by selected_provider upstream
backend/agents/rlm/run.py:391            _make_runtime(_exec_provider, ...)  # ONLY site passing "azure"/"azure-foundry"
backend/agents/rlm/run.py:416            make_runtime("anthropic")
backend/agents/rlm/run.py:424            make_runtime(provider, require_api_key=True)
backend/agents/rdr/run.py:230/233        make_runtime("anthropic" | "openai")
backend/agents/rdr/agent.py:475          make_runtime(provider)
```
Only `run.py:391` (the executor-role branch) ever passes an Azure/Foundry provider string. Every other caller is anthropic/openai/env-default.

---

## 3. Credential model + shadow validator findings

### Two distinct Azure credential families
| Family | Env vars | Settings fields | Resolver | Auth shape |
|---|---|---|---|---|
| **Azure OpenAI** (classic) | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` | `config.py:133-156` (also bridges portal `KEY1`/`KEY2`) | `models.py::_inject_azure_kwargs` (`:357-399`); `factory.configure_azure_openai_credentials` bridges `.env`→`os.environ` (`factory.py:384-411`); `_has_azure_openai_credentials` (`factory.py:176-184`) | `AsyncAzureOpenAI`, `/openai/deployments/{name}?api-version=`, **API key only** |
| **Azure AI Foundry** (OpenAI-compatible v1) | `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_DEPLOYMENT`, `AZURE_FOUNDRY_API_KEY` | `config.py:169-183` | **Single canonical** `foundry_endpoint.resolve_foundry_credentials()` → normalized `(base_url, deployment, api_key)` triple (`foundry_endpoint.py:56-65`); `has_foundry_credentials` (`:68-71`) | plain OpenAI SDK, Bearer auth, `base_url=…/openai/v1`, model=deployment, **API key only** |

`foundry_endpoint.resolve_foundry_credentials` reads `os.environ` first then Settings/.env, normalizes the URL via `normalize_foundry_base_url` (`foundry_endpoint.py:24-40`), and is stdlib + `backend.config` only (no `backend.agents.*` import) so any layer can call it without a cycle. It is consumed identically by the root registry (`models.py:438` has its own near-duplicate `_inject_foundry_kwargs`), the executor/Foundry runtime (`azure_foundry_runtime.py:26,45`), and the grader/verifier transport (`grader_transport.py:225-232`). **Minor redundancy:** `models.py` re-implements the same normalize/env-or-settings logic (`models.py:402-467`) instead of calling `foundry_endpoint`; functionally equivalent but a second source of truth.

### shell-wins-over-.env precedence
`config.py::_apply_legacy_env_aliases` (`config.py:13-33`, runs at import) only bridges the `REPROLAB_*` ⇄ `OPENRESEARCH_*` prefix via `setdefault` (never overwrites). It is **not** an Azure-specific bridge. The actual shell>.env precedence comes from pydantic-settings reading `os.environ` first (`config.py:42-46`), and from every Azure/Foundry consumer reading `os.environ` directly with `_env_or_settings` falling back to Settings second (`models.py:402-418`, `foundry_endpoint.py:43-53`).

### Shadow validator — `cli.py::_warn_on_shell_env_override` (`cli.py:138-167`)
Warn-only, advisory, fires on the CLI reproduce path (`cli.py:1519-1521`). Covered keys (`cli.py:151-155`):
```
OPENAI_API_KEY, ANTHROPIC_API_KEY, FEATHERLESS_API_KEY,
OPENROUTER_API_KEY, AZURE_OPENAI_API_KEY, OPENRESEARCH_RUNPOD_API_KEY
```
**Azure/Foundry coverage gaps:** `AZURE_OPENAI_API_KEY` is covered; **`AZURE_FOUNDRY_API_KEY` is NOT**, and neither endpoint (`AZURE_OPENAI_ENDPOINT`, `AZURE_FOUNDRY_ENDPOINT`) nor deployment is covered. A stale shell `AZURE_FOUNDRY_*` shadowing `.env` is silent. Also note: the validator runs **only on the CLI path**, not on server boot (BUG-LR-014 partially remediated, per CLAUDE.md).

---

## 4. The single biggest existing reusable seam

**`role_models.resolve_role_models` (pure picker) + `factory.make_runtime` (runtime builder) + `grader_transport.build_transport_client` (client builder) together already ARE the "one provider layer" for 4 of the 5 tiers, and all three already accept `azure-foundry`.**

Evidence the seam is real and Foundry-aware:
- `role_models.SUBROLE_PROVIDERS` includes `PROVIDER_AZURE_FOUNDRY` (`role_models.py:75-83`); vocab maps `azure-foundry`/`foundry`/`grok`/`grok-4.3` (`role_models.py:127-134`); `parse_model_spec` accepts them for executor/verifier/grader.
- `make_runtime` has explicit `azure` and `azure-foundry`/`grok` branches with `require_api_key` fail-fast (`factory.py:525-552`).
- `build_transport_client` has `azure` and `azure-foundry`/`grok` branches (`grader_transport.py:196-243`).
- run.py already threads all of it: planner→root, executor→`_resolve_agent_runtime`→make_runtime (`run.py:382-394`), verifier→build_transport_client (`run.py:2124-2133`), grader→`OPENRESEARCH_GRADER_*` env→build_grader_client (`run.py:2139-2142`).

**Implication for design:** the work is NOT "build a provider layer." It is "(a) route the navigation accelerator through the existing seam (add a `_resolve_foundry`/foundry mode), and (b) close the two legacy bypass knobs — `OPENRESEARCH_EXECUTOR` and `selected_provider` — so they too can name azure-foundry." The seam already exists; ~80% of the unification is wiring the two bypassers + the accelerator into it.

---

## 5. GAP LIST (ranked)

| # | Tier | Gap | Exact site to route through the shared layer | file:line |
|---|---|---|---|---|
| **G1** | Navigation accelerator | No Foundry mode at all; `_resolve_azure` reads only `AZURE_OPENAI_*` | Add a `_resolve_foundry(explicit)` calling `foundry_endpoint.resolve_foundry_credentials`, dispatch a new `"foundry"`/`"grok"` mode in `resolve_accelerator`; `build_accelerator_client` already handles a non-azure OpenAI-compatible endpoint (`accelerator.py:559-580`) so a Foundry `AcceleratorEndpoint(is_azure=False)` works as-is. | `accelerator.py::resolve_accelerator` `accelerator.py:445-471`; `_resolve_azure` `accelerator.py:303-346` |
| **G2** | Executor (legacy knob) | `OPENRESEARCH_EXECUTOR` has `azure` modes but **no foundry mode** | Add a `_FOUNDRY_MODES` branch in `resolve_executor` returning an `ExecutorPlan` over `AzureFoundryAgentRuntime` (mirror the `_AZURE_MODES` block). Runtime already exists. | `executor.py::resolve_executor` `executor.py:39`, `:77-97` |
| **G3** | Generic sub-agents (tier 3) | `selected_provider()` accepts only anthropic/openai → azure AND foundry are **unselectable** for any non-executor SDK sub-agent | Either widen `selected_provider` to recognize `azure`/`azure-openai`/`azure-foundry`/`grok` and return them (then `make_runtime`'s early branches already build them), or have the non-executor callers (`invoke.py:135`, `rdr/run.py:230-233`, `rdr/agent.py:475`) consult `role_models`. | `factory.py::selected_provider` `factory.py:227-243` |
| **G4** | Shadow validator | `AZURE_FOUNDRY_API_KEY` + both endpoints not covered | Extend `_SUSPECT_KEYS` (advisory only; cheap). | `cli.py:151-155` |
| **G5** | Root (cleanup, not a functional gap) | `models.py` re-implements Foundry endpoint normalization instead of calling the canonical `foundry_endpoint` resolver | Replace `models.py::_inject_foundry_kwargs` / `_normalize_foundry_base_url` / `_env_or_settings` (`models.py:402-467`) with calls into `foundry_endpoint`. Functionally equivalent today; removes a second source of truth. | `models.py:402-467` |

---

## 6. Open questions for the 3 DECIDE points

### (a) Which tiers need Foundry day-1?
Current state (what's already done vs. cheap to finish):
- **Root, verifier, grader** — Foundry **already fully wired** (`models.py:239`, `grader_transport.py:219`). Day-1 ready, no work.
- **Executor** — Foundry wired via the **unified `--models executor=grok`** surface (`run.py:388`), so day-1 reachable; only the *legacy* `OPENRESEARCH_EXECUTOR` knob lacks it (G2 — small).
- **Navigation accelerator** — **the only tier with zero Foundry support** (G1). If the day-1 goal is "a fully OAuth-free, all-Foundry run including hot-volume navigation," this is the gating item. Note CLAUDE.md guidance keeps the quality-critical grader/verifier off the accelerator and on Sonnet (`ACCELERATOR_SCOPE=navigation`), so accelerator-Foundry mainly matters for the cheap navigation calls.
- **Generic sub-agents (tier 3)** — needs G3 only if a non-executor SDK sub-agent must run on Foundry; currently they all run Claude.

**Recommendation to surface for DECIDE:** root/verifier/grader/executor(unified) are done; decide whether navigation-accelerator Foundry (G1) and the legacy-knob/sub-agent parity (G2/G3) are in day-1 scope or fast-follow.

### (b) Foundry auth = API-key-only, or also Entra / managed identity?
**Evidence-based answer: every Azure LLM path today is API-key-only.**
- `resolve_foundry_credentials` returns only `(base_url, deployment, api_key)` — no credential object (`foundry_endpoint.py:56-65`).
- `_inject_azure_kwargs` / `AzureOpenAiAgentRuntime` use `api_key` only (`models.py:357-399`, `azure_openai_runtime.py:64-80`).
- `DefaultAzureCredential` **does** appear in the codebase — but **only** in the K8s cell-runner's Azure Blob I/O path (`k8s_job_cell_runner.py:432`, `:1616`), via AKS Workload Identity (`k8s_job_cell_runner.py:746-749`), **never for any LLM completion tier**. `azure-identity>=1.16` is a declared dependency (`requirements.txt:35`) but scoped to blob/storage.

So Entra/managed-identity for LLM auth is **not implemented anywhere** — it would be net-new. The blob path proves the dependency and the Workload-Identity pattern already exist in-repo, which lowers the cost if DECIDE chooses to add it (e.g. a `DefaultAzureCredential`-backed token provider for the Foundry/Azure-OpenAI clients).

### (c) Keep `azure_openai_runtime` separate from `azure_foundry_runtime`, or collapse?
They are genuinely different transports, not redundant:
- `AzureOpenAiAgentRuntime` overrides `_configure_sdk_client` to build `AsyncAzureOpenAI` (`/openai/deployments/{name}?api-version=`, `azure_openai_runtime.py:64-80`).
- `AzureFoundryAgentRuntime` is a thin subclass of `OpenAiAgentRuntime` using the plain OpenAI SDK with `base_url=…/openai/v1` + Bearer (`azure_foundry_runtime.py:30-52`) — it does NOT override `_configure_sdk_client`.

They share only `_model_override` (return the deployment as the model id). **Recommendation to surface:** keep them separate (different SDK clients = different auth/URL shape); the meaningful unification is at the *selection* layer (role_models tokens + make_runtime branches), which is already done, not at the runtime-class layer. Collapsing would force one class to branch on URL shape — strictly worse than the current two-class split.

---

## Appendix — SDK-isolation sites (BUG-NEW-038)

Three `ClaudeAgentOptions(...)` construction sites; the invariant is `setting_sources=[]` + explicit `mcp_servers` + a non-plan `permission_mode` so the SDK never loads the developer's `~/.claude/settings.json` / MCP / plan-mode.

| Site | setting_sources | mcp_servers | permission_mode |
|---|---|---|---|
| Root completions — `rlm_query.py:725-732` | `[]` (`:731`) | `{}` (`:732`) | `"default"` (`:725`) |
| Sub-agents — `claude_runtime.py::_agent_options_kwargs` | `[]` when hermetic (default on, `:276-277`) + `strict_mcp_config=True` (`:278`) | explicit `mcp_servers` always (`:269`) | `agent.permission_mode`, default `bypassPermissions` (`base.py:107`, `:263`) |
| Hermes audit — `hermes_audit/providers.py:388-393` | `[]` (`:392`) | `{}` (`:393`) | `"bypassPermissions"` (`:388`) |

Note: `claude_runtime`'s `setting_sources=[]` is gated by `OPENRESEARCH_SDK_HERMETIC` (default true, `claude_runtime.py:230-244`); the other two are unconditional. The `allowed_tools` restriction in `claude_runtime` is always on regardless of the hermetic flag (`:271-274`).
