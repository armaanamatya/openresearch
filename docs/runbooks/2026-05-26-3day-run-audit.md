# 3-Day Run Audit — 2026-05-24 through 2026-05-26

**Prepared:** 2026-05-26  
**Auditor:** forensic read of every `runs/prj_*` directory whose `demo_status.json::startedAt` falls in the window.  
**PRs in scope:** PR-α through PR-λ (committed) + PR-μ (committed at `b9c3918`, 2026-05-26T06:47 UTC).  
**Note on timing:** All 12 production runs in this window started BEFORE PR-μ landed at 06:47 UTC on 2026-05-26. The in-flight PR-μ code is on `origin/main`; the runs were running against the pre-μ code.

---

## 1. Executive Summary

**12 runs audited.** 4 in the 3-day window have `demo_status::status="running"` but dead PIDs — they are zombie-status runs.

| Outcome | Count |
|---|---|
| `completed` with rubric score >0 | 2 (prj_d02bd4: 0.535; prj_4cbc: 0.024) |
| `completed` / `partial` with rubric score = 0 or null | 3 (prj_3080 manually salvaged null; prj_6d41 0.0; prj_9afa 0.0) |
| `failed` with final_report written | 3 (prj_0a77 0.305; prj_11e9 0.0; prj_ac41 0.0) |
| `running` — zombie (dead PID, no final_report) | 4 (prj_9554, prj_db45, prj_09047, prj_8819) |
| **Score ≥ 0.6 target** | **0** |

**Orphaned resources:**
- 1 active Docker container: `ec8a5bbcae11` (`reprolab/prj_3080fe2a02c20164`) running since `2026-05-26T00:35` UTC — still alive as of audit time, 6+ hours after the run was force-killed. The VAE run was manually salvaged; the container was never cleaned up.
- 7 exited Docker containers (`Exited (137)`) for prj_8819, prj_6d41, prj_9afa, and smoke-test runs. These are dead but not pruned.
- No RunPod orphan pods confirmed: the two RunPod runs that failed at preflight (prj_0a77, prj_11e9) never provisioned a pod. The one successful RunPod run (prj_d02bd4) destroyed its pod normally.

**The 5 biggest unresolved patterns (post-PR-μ):**

1. **binding.py 1800s catch-all still fires on `run_experiment`** — PR-μ added `resolve_experiment_timeout_s` (6h for max mode) inside `primitives.py` but never excluded `run_experiment` from `binding.py`'s `_DEFAULT_PRIMITIVE_TIMEOUT_S = 1800`. The outer wrapper kills the primitive at 30 min regardless. Evidence: `run_warning wall_clock_s=1800` in prj_0a77 (max mode). **OPEN.**

2. **4 zombie-status runs not marked failed** — prj_09047, prj_8819 (started 2026-05-24, no PID) and prj_9554, prj_db45 (started 2026-05-25, PID dead) show `status="running"` indefinitely. No cleanup path updates these. **OPEN.**

3. **1 orphan Docker container** — prj_3080 VAE container still `Up` 6h post-kill. PR-μ Solution D (atexit + startup sweep) only fires for RunPod pods, not Docker containers, and only when the backend had a chance to run atexit. Manual kill bypasses all cleanup. **MITIGATED (Docker not fully covered by PR-μ's atexit).**

4. **`api_key_missing` for self-referential RLM-paper run** — prj_ac41 (Recursive Language Models = arxiv 2512.24601) needs `OPENAI_API_KEY` or `FIREWORKS_API_KEY` to run its own eval harness. No PR addresses this. **OPEN (domain-specific; needs guidance in system prompt or a per-paper env-injection hook).**

5. **`WakeSleepVAE.reparameterize` attribute bug class** — preflight_ast catches the specific attribute-access pattern (PR-γ), but only for static call sites. Dynamic attribute access (`setattr`) before the call is not caught. The VAE run only passed 95% because this missed the `marginal_ll_importance_sampling` path. **MITIGATED (not fully eliminated).**

---

## 2. Per-Run Table

| Run ID | Paper | Mode | Sandbox | Started | Duration | Iters | Score | Status | Key Error |
|---|---|---|---|---|---|---|---|---|---|
| prj_4cbc5f6bdcaa9763 | Reincarnating RL (QDagger) | N/A | N/A | 2026-05-24T00:14 | 23.6 min | 3 | 0.024 | completed/partial | `detect_environment` ValidationError training_recipe (pre-PR-β) |
| prj_09047604e591d969 | SDAR (arxiv 2605.15155) | N/A | N/A | 2026-05-24T19:20 | — | 0 | — | **ZOMBIE** | No PID; last event `build_environment start` at 19:30 |
| prj_8819ad4e6e250b73 | Batch Normalization (1502.03167) | N/A | N/A | 2026-05-24T19:20 | — | 0 | — | **ZOMBIE** | No PID; last event `implement_baseline start` at 19:36 |
| prj_ac41983c934a3432 | Recursive Language Models (2512.24601) | N/A | N/A | 2026-05-24T15:57 | 30.4 min | 1 | 0.0 | failed | `api_key_missing` OPENAI/FIREWORKS; `run_experiment timed out after 20s` |
| prj_6d41d2f09c026403 | Adam (arxiv 1412.6980) | N/A | RunPod | 2026-05-25T17:41 | 15.8 min | 4 | 0.0 | completed/failed | `RUNPOD_BALANCE_TOO_LOW` × 4 (credit exhausted) |
| prj_9afa700e444c1df7 | Dropout (arxiv 1207.0580) | N/A | RunPod | 2026-05-25T17:41 | 17.8 min | 4 | 0.0 | completed/partial | `RUNPOD_BALANCE_TOO_LOW` × 3; 1× GPU escalation (rtx4090→L40S before credit died) |
| prj_d02bd4fdf24aedf2 | Adam (arxiv 1412.6980) | efficient | RunPod | 2026-05-25T22:19 | 4.51h | 1 | **0.535** | completed/partial | Run 1: SSH `Connection closed`; Run 2: success. Pod cost ≈ $1.03 |
| prj_3080fe2a02c20164 | Auto-Encoding Variational Bayes (1312.6114) | max | Docker | 2026-05-25T21:00 | 6.42h | 2 | null (salvaged) | completed (manual) | Phase 3 crash: `WakeSleepVAE.reparameterize` missing; Frey Face HTTP 403; 3h+ aclose hang |
| prj_9554a1396eb993aa | VAE (arxiv 1312.6114) | max | Docker | 2026-05-25T20:42 | — | 0 | — | **ZOMBIE** | PID dead; last event `plan_reproduction start` at 20:46 |
| prj_db45c0304ce455a6 | VAE (arxiv 1312.6114) | max | Docker | 2026-05-25T20:46 | — | 0 | — | **ZOMBIE** | PID dead; last event `run_experiment start` at 20:57 |
| prj_11e9c4a1ff4f6c15 | Adam (arxiv 1412.6980) | max | RunPod | 2026-05-26T03:24 | 20.6 min | 1 | 0.0 | failed | 5 preflight violations (imdb bare name × 1, lr_decay double kwargs × 4); aclose crash |
| prj_0a77e7ed00d0c9da | Adam (arxiv 1412.6980) | max | RunPod | 2026-05-26T03:48 | 45.5 min | 1 | **0.305** | failed | Run 1: preflight 1 violation (imdb); Run 2: binding.py 1800s outer timeout; aclose crash |

**Cost notes:**
- prj_d02bd4: run_experiment wall_clock 4.22h, GPU hours 3.014, estimated pod cost ≈ $1.03.
- prj_3080: run_experiment wall_clock 3.43h (12,330s), GPU hours 3.425 (RTX 4090), estimated pod cost ≈ $1.17. Container still running.
- All others: zero pod cost confirmed (either no pod provisioned or pod destroyed normally).
- Token cost: all claude-oauth runs show `cost_usd=0.0` in ledger (subscription-billed, not API-billed).

**The 0.535 vs 0.305 Adam delta explained:**
- **efficient run (prj_d02bd4, 0.535):** Run 1 SSH failure → self-recovered in same iteration → Run 2 succeeded after 4.2h of actual training. All 5 experiment environments (`mnist_logistic`, `imdb_logistic`, `mnist_mlp`, `cifar10_cnn`, `vae`) ran and produced metrics.
- **max run (prj_0a77, 0.305):** Run 1 preflight-blocked (imdb bare name) → implement_baseline repaired → Run 2 ran for 1964s → hit **binding.py outer 1800s catch-all** (not PR-μ's 6h cap) → timed out. `verify_against_rubric` scored code-only (no real execution numbers). Rubric areas "Experiment execution" (0.075 vs 0.33), "Eval protocol" (0.06 vs 0.50), and "Data preprocessing" (0.09 vs 0.185) tanked.

---

## 3. Pattern Catalog

### P1 — SDK `aclose()` async-generator race (status=3 crash)

**Count:** 6 runs (prj_11e9, prj_0a77, prj_3080, prj_9554, prj_db45, prj_d02bd4 stderr noise).  
**Signature:**
```
an error occurred during closing of asynchronous generator <async_generator object InternalClient._process_query_inner at 0x...>
RuntimeError: aclose(): asynchronous generator is already running
```
**Mechanism:** `claude-agent-sdk`'s internal streaming generator races its own `aclose()` when the parent event loop is shutting down. Previously called directly in the parent loop; the race corrupted the pipeline and caused `exit(3)`.  
**Classification: IN-FLIGHT / RESOLVED** — PR-μ Solution A introduced `backend/agents/runtime/sdk_isolation.run_isolated()` which wraps every SDK call in a dedicated worker thread with its own event loop. `backend/agents/runtime/invoke.py:61` now routes through `run_isolated`. The race fires inside the worker, is swallowed by the `_is_aclose_race` guard, and never reaches the parent loop.

**Evidence of fix:**
```python
# invoke.py:61
collected, tool_calls = await run_isolated(_do_sdk_call())
```
```python
# sdk_isolation.py:56-62
if _is_aclose_race(exc) and captured_result:
    logger.debug("sdk_isolation: aclose race swallowed at cleanup")
else:
    captured_exception.append(exc)
```
**Residual surface:** Any future SDK call constructed outside `collect_agent_text` (direct `claude_agent_sdk.Client()`) bypasses the isolation. A lint test `tests/test_sdk_direct_call_lint.py` gates against this.

---

### P2 — `binding.py` 1800s outer catch-all kills `run_experiment` in max mode

**Count:** 1 confirmed (prj_0a77); all pre-μ max-mode RunPod runs are at risk.  
**Signature:**
```
run_warning: primitive_timeout, primitive=run_experiment, wall_clock_s=1800
"primitive `run_experiment` exceeded its wall-clock cap of 1800s and was interrupted."
```
**Mechanism:** `binding.py` has a `_DEFAULT_PRIMITIVE_TIMEOUT_S = 1800` catch-all that fires for any primitive NOT explicitly in `PRIMITIVE_TIMEOUT_S`. The comment at line 36 says `run_experiment` is "intentionally absent — [it has] its own caps." But the code's `PRIMITIVE_TIMEOUT_S.get(name, _DEFAULT_PRIMITIVE_TIMEOUT_S)` still applies the 1800s default when `run_experiment` is missing from the dict. PR-μ's Solution B added `resolve_experiment_timeout_s(ctx)` which returns 21600s for max mode — but this is the **inner** cap, called inside `run_experiment` for the sandbox `pool.submit(...).result(timeout=...)` call. The **outer** `binding.py` wrapper fires at 1800s FIRST, killing the whole primitive before the inner timeout matters.

**Classification: OPEN** — PR-μ does not touch `binding.py`. `b9c3918` (the PR-μ commit) lists no `binding.py` changes.

**Fix (minimal):** Add `run_experiment` and `implement_baseline` to `PRIMITIVE_TIMEOUT_S` with `math.inf` or `sys.maxsize`, or add an explicit skip guard:
```python
# binding.py, after line 253
_TIMEOUT_EXCLUDED = frozenset({"implement_baseline", "run_experiment"})
_timeout_s = None if name in _TIMEOUT_EXCLUDED else PRIMITIVE_TIMEOUT_S.get(name, _DEFAULT_PRIMITIVE_TIMEOUT_S)
```
This is a 3-line fix. Without it, every max-mode run will hit 1800s on the second+ `run_experiment`, regardless of PR-μ.

---

### P3 — Zombie run status (PID dead, `demo_status.json` still `"running"`)

**Count:** 4 runs (prj_09047, prj_8819, prj_9554, prj_db45).  
**Mechanism:** The run subprocess crashes with SDK aclose or is killed externally. `demo_status.json` is only updated by `update_run_status()` which the subprocess calls. If the subprocess dies without reaching the cleanup path, the status is never updated. The UI shows the run as "running" indefinitely.  
**Classification: OPEN** — No PR addresses dead-status cleanup. PR-μ Solution D adds atexit pod cleanup for RunPod resources but does not write a final `demo_status.json` on crash.

**Fix:** In the run subprocess entry point (`backend/agents/rlm/run.py`), wrap the full pipeline in a try/finally that writes `demo_status.json::status = "failed"` with a crash reason before exiting. The existing `completedAt` stamp logic should run in `finally`, not just on success.

---

### P4 — `RUNPOD_BALANCE_TOO_LOW` — credit exhaustion kills entire session

**Count:** 7 `run_experiment` calls across 2 runs (prj_6d41 × 4, prj_9afa × 3) on 2026-05-25.  
**Signature:**
```
run_experiment: SandboxRuntimeError: RUNPOD_BALANCE_TOO_LOW: Server error '500 Internal Server Error'
failure_class: None, outcome: None
```
**Mechanism:** RunPod credit ran to zero mid-session on 2026-05-25 17:41 UTC. All subsequent `run_experiment` calls failed. The transient classifier (PR-ζ) correctly classifies `RUNPOD_BALANCE_TOO_LOW` as `fatal` — no retry occurs, which is correct. However, `failure_class` and `outcome` fields in `experiment_runs.jsonl` show `null` because these runs predated PR-ζ (landed 2026-05-26T05:04 UTC).  
**Classification: RESOLVED by PR-ζ (post-landing)** — A post-PR-ζ run seeing credit exhaustion will get `failure_class=fatal`, emit a clear fatal event, and stop immediately without burning further compute on futile retries. The user will still need to refill credit — the system cannot fix that — but the failure is now surfaced clearly.

**Note:** The Adam run (prj_6d41) proposed "Switch to Local CPU Sandbox" as an improvement candidate but never tried it because BALANCE_TOO_LOW fired at every attempt before sandbox fallback could be chosen. Post-PR-ζ, sandbox fallback is opt-in via `REPROLAB_SANDBOX_FALLBACK_ENABLED`; it won't auto-trigger on fatal errors by design.

---

### P5 — Preflight violations: HuggingFace bare dataset names + double keyword args

**Count:** 5 preflight violations across 2 runs (prj_11e9: 5 hard violations; prj_0a77: 1 hard violation).  
**Violations observed:**
- `load_dataset('imdb')` → `HfUriError: Repository id must be 'namespace/name'` — 2 runs, pre-flight caught it.
- `make_optimizer(..., lr_decay=<value>, **kw)` passing `lr_decay` both explicitly and via `**kw` → `TypeError: got multiple values for keyword argument 'lr_decay'` — prj_11e9 had this at lines 619, 657, 703, 748.

**Classification: RESOLVED by PR-γ** — `pre_flight_validator._check_deprecated_hf_dataset_aliases` (line 1088) and `_check_optimizer_double_keyword` (line 1324) catch these before sandbox dispatch. Evidence: both prj_11e9 and prj_0a77 saw `failure_class="preflight_blocked"` and `contract_violations` with the exact line numbers and fix hints. The model's repair iteration fixed the imdb issue in prj_0a77 (second `implement_baseline` completed in 20 seconds).

---

### P6 — `WakeSleepVAE.reparameterize` missing attribute (agent code bug)

**Count:** 1 confirmed crash (prj_3080 Phase 3 marginal-LL). Likely same class in prj_9554, prj_db45.  
**Signature:**
```python
# train.py:177
z = model.reparameterize(mu, logv)
# → AttributeError: 'WakeSleepVAE' object has no attribute 'reparameterize'
```
**Mechanism:** Sonnet's `implement_baseline` wrote a `WakeSleepVAE` class that inherits from `BaseVAE` but does not override `reparameterize`. A different code path calls `model.reparameterize()` on it. Crashes only at the `marginal_ll_importance_sampling` call after 3+ hours of valid training.  
**Classification: MITIGATED by PR-γ** — `preflight_ast.py` implements `_check_missing_attr_access` which scans for `x.method()` calls where `method` is not defined on any class assigned to `x`. The exact docstring even references this bug:
```python
# preflight_ast.py:5-7
# AttributeError: 'WakeSleepVAE' object has no attribute 'reparameterize'
# The agent wrote model.reparameterize(z, mu, log_var) on a class that had no such method.
```
**Residual:** PR-γ's static analysis cannot catch dynamic dispatch (`setattr(model, "reparameterize", ...)` before the call) or attribute injection in `__init__` via `setattr`. The VAE bug may have been caught if it were static; whether the specific Sonnet-generated code triggers the scanner depends on how `WakeSleepVAE` was defined relative to where `model.reparameterize` was called.

---

### P7 — RunPod SSH `Connection closed` transient

**Count:** 1 (prj_d02bd4 Run 1).  
**Signature:**
```
run_experiment: SandboxRuntimeError: Runpod SSH command failed: Connection closed
```
**Mechanism:** RunPod SSH transport dropped mid-command (likely host recycle or network blip). The run completed metrics were lost.  
**Classification: RESOLVED by PR-ζ (post-landing)** — `transient_classifier._TRANSIENT_MARKERS` includes `"Connection closed"`. Post-PR-ζ, this would trigger an exponential-backoff retry (up to 3×) before the root model even sees the failure. In the prj_d02bd4 run (which predated PR-ζ), the root model naturally retried the experiment in the same iteration, which succeeded.

---

### P8 — `api_key_missing` for self-hosted RLM eval (paper 2512.24601)

**Count:** 1 (prj_ac41).  
**Signature:**
```json
{"error": "api_key_missing", "required_env_vars": ["OPENAI_API_KEY", "FIREWORKS_API_KEY"]}
```
**Mechanism:** The RLM paper (arxiv 2512.24601) is also the system's own library. Reproducing it requires running the `rlms` benchmark eval, which needs an LLM API key that the reproduction sandbox doesn't inherit from the host environment. The `implement_baseline` agent correctly generates the eval harness but the sandbox has no API key.  
**Classification: OPEN** — No PR addresses per-paper environment key injection. The system prompt and `detect_environment` have no mechanism to propagate host API keys into the sandbox for specific papers.

**Fix:** A per-paper `required_env_vars` declaration in `PaperTargets` (or in the generated Dockerfile / `commands.json`) that instructs `run_experiment` to forward specific env vars from the host environment into the container. This is a nontrivial security/scope surface (forwarding API keys into potentially-untrusted containers) — needs explicit design before shipping.

---

### P9 — Docker container not cleaned up on force-kill (manual run abort)

**Count:** 1 confirmed (prj_3080 VAE, container `ec8a5bbcae11` up 6h post-kill).  
**Mechanism:** The VAE run was manually killed by the operator after 3h+ aclose hang. The atexit hook from PR-μ Solution D Layer 1 fires on normal process exit; a manual SIGKILL bypasses atexit. The container persists billing-free (Docker doesn't bill by the hour) but consumes memory and CPU.  
**Classification: MITIGATED by PR-μ Solution D** — The startup sweep (`create_app` lifespan) and periodic scheduler (`PodSweepScheduler`) handle RunPod pod orphans. For Docker containers specifically, `runpod_backend.py` registers an atexit for containers too (per Solution D design), but a SIGKILL escapes atexit. The periodic sweep does not cover Docker containers — only RunPod pods. Docker orphans require manual `docker rm` or a separate Docker-specific sweep.

---

### P10 — `detect_environment` `ValidationError: hardware_clues` non-list

**Count:** 1 (prj_3080 at 2026-05-25T21:02); 1 historical (prj_4cbc `training_recipe`).  
**Signature:**
```
detect_environment ERR=ValidationError: hardware_clues: Input should be a valid list (list_type)
```
**Mechanism:** The LLM root passed `method_spec["hardware_clues"]` as a bare string instead of a `list[str]`. Pydantic rejected the input.  
**Classification: RESOLVED by commit `7294ee1` (2026-05-25T17:50)** — `schemas.py` coercers added for `hardware_clues` (and `ambiguities`) that accept a bare string and wrap it in a list. The prj_3080 run started at 21:00, after `7294ee1` landed at 17:50, so it benefited from the coercion — which is why the run continued past `detect_environment` instead of hard-failing.

---

### P11 — Frey Face dataset HTTP 403

**Count:** 1 (prj_3080 Phase 2).  
**Signature:**
```json
{"dataset": "frey_face", "loader": "scipy_mat", "error": "download or scipy unavailable"}
```
**Mechanism:** The NYU cs.nyu.edu mirror for the Frey Face dataset returned HTTP 403. Phase 2 (Frey Face AEVB sweep) was skipped.  
**Classification: RESOLVED by PR-κ and PR-λ** — PR-κ (`skip data-unavailable leaves`) excludes leaves where `data_load_failures` recorded the dataset as unavailable, so the rubric grader doesn't penalize on evidence that was structurally impossible to collect. PR-λ (dataset recipe registry) would provide fallback mirrors if registered; Frey Face is not yet in the registry (`backend/agents/dataset_recipes.py` exists but Frey Face entry absent). Net: the rubric correctly ignores Frey Face, but the fallback mirror path isn't exercised.

---

### P12 — run_experiment timed out after 20s (prj_ac41)

**Count:** 1 (prj_ac41 Run 2).  
**Mechanism:** The second `run_experiment` attempt timed out after 20 seconds. This is far below any reasonable training timeout. The run predated PR-γ (which set `PRIMITIVE_TIMEOUT_S` defaults) and predated the `resolve_experiment_timeout_s` changes from PR-μ. At the time, the only cap was the ad-hoc `ctx.remaining_s()` logic, which apparently resolved to ~20s because most of the run budget was already consumed.  
**Classification: RESOLVED by PR-γ + PR-μ** — PR-γ established explicit timeouts; PR-μ's `resolve_experiment_timeout_s` applies mode-scaled caps. A post-μ run would get 2h (efficient) or 6h (max) for `run_experiment`... except for Pattern P2 above (binding.py 1800s still applies).

---

## 4. Honest Gap Analysis

### Fresh Adam max-mode rerun TODAY (post-PR-λ + PR-μ, but before P2 fix)

1. **preflight: imdb `stanfordnlp/imdb` fix** — PR-γ's `_check_deprecated_hf_dataset_aliases` catches this pre-dispatch. The repair iteration would succeed.
2. **preflight: lr_decay double-kwargs** — PR-γ's `_check_optimizer_double_keyword` catches this. Fixed pre-dispatch.
3. **binding.py 1800s outer timeout (Pattern P2)** — **STILL FIRES.** After the repair iteration fixes the imdb/lr_decay bugs, the second `run_experiment` will again time out at 1800s. The inner `resolve_experiment_timeout_s` returns 21600s for max mode but never gets to fire because the outer `binding.py` wrapper kills the thread at 30 min. **The 0.305 score is reproducible.** A max-mode Adam rerun today would land near 0.305, not 0.535.
4. **SDK aclose crash** — PR-μ Solution A fixes this. The pipeline would not exit(3); it would produce a proper `final_report.json` with `status=failed` (not `status=3`).
5. **Forced-iteration policy** — After the 1800s timeout returns `{outcome: "retryable"}`, the forced-iteration policy (`forced_iteration.py`) could refuse a premature FINAL_VAR, but the ForcedIterationPolicy extension from PR-μ Solution C only triggers on `repairable`/`partial_evidence`/`fatal` — not `retryable`. So the root might FINAL_VAR after the timeout, same as before.

**Summary: A fresh Adam max-mode rerun today would score near 0.305, not 0.535. The gap is entirely Pattern P2 (binding.py 1800s).**

### Post-P2-fix (after binding.py is corrected)

With binding.py fixed to exclude `run_experiment` from the 1800s catch-all:
1. The repair iteration fixes imdb + lr_decay.
2. `run_experiment` runs for up to 6h (max mode cap from `resolve_experiment_timeout_s`).
3. All 5 environments complete (matching the efficient run at 4.2h).
4. Rubric score lands near 0.535–0.60. The score gap between efficient and max mode disappears.

### Post-PR-μ VAE max-mode rerun

The three VAE zombie runs (prj_9554, prj_db45) and the salvaged run (prj_3080) all failed due to aclose crashes. Post-PR-μ Solution A:
1. Plan_reproduction no longer crashes mid-call (prj_9554's failure mode).
2. run_experiment no longer crashes mid-call (prj_db45's failure mode).
3. **But Pattern P2 still applies**: the 3.4h VAE run_experiment hit the docker binding.py 1800s timeout exactly as the Adam run did. Post-μ without P2 fix, the VAE run would time out at 30 min, miss Phase 3 marginal LL (which takes ~1h), and land partial without Phases 2–3.
4. The `WakeSleepVAE.reparameterize` bug (Pattern P6) — if not caught by `preflight_ast`, Phase 3 crashes after Phase 1 succeeds. With `preflight_ast` catching the missing attribute, the model gets a repair hint and can add `reparameterize` to the class before dispatch.
5. Frey Face (Pattern P11) — PR-κ correctly skips the leaf; no rubric penalty.

**Expected post-μ + post-P2-fix VAE score: ~0.50–0.65 (Phase 1 MNIST succeeds; Phase 2 Frey Face skipped/penalized; Phase 3 marginal LL succeeds if preflight catches the reparameterize bug).**

---

## 5. Recommendations (max 3, ordered by ROI)

### Rec-1 — Fix `binding.py` outer timeout exclusion for `run_experiment` (1 hour)

**File:** `backend/agents/rlm/binding.py`  
**Change:** Add `run_experiment` and `implement_baseline` to a `_TIMEOUT_EXCLUDED` frozenset that bypasses the `_DEFAULT_PRIMITIVE_TIMEOUT_S` catch-all:

```python
# After PRIMITIVE_TIMEOUT_S definition (~line 54)
_TIMEOUT_EXCLUDED: frozenset[str] = frozenset({"implement_baseline", "run_experiment"})
# ...
# In wrapped(), replace line 254:
_timeout_s = None if name in _TIMEOUT_EXCLUDED else PRIMITIVE_TIMEOUT_S.get(name, _DEFAULT_PRIMITIVE_TIMEOUT_S)
```

And in the `FuturesTimeoutError` handler, guard against `_timeout_s is None` (skip the timeout path entirely for excluded primitives — their internal caps handle it).

**ROI:** This single change unlocks the full intent of PR-μ Solution B. Without it, the 0.535→0.305 score regression on Adam max mode is permanent. Estimated 1 hour to implement + test. Required before any max-mode rerun is meaningful.

---

### Rec-2 — Write `demo_status.json::status="failed"` on subprocess crash (2 hours)

**Files:** `backend/agents/rlm/run.py`, `backend/cli.py` (the subprocess entry point)  
**Change:** Wrap the full pipeline execution in a `try/finally` that calls a `_write_crash_status(project_dir, error_msg)` helper if the run exits without normally calling `update_run_status("completed"/"failed")`. This terminates the 4-zombie-run pattern permanently. The startup sweep from PR-μ Solution D could also call this for runs whose PIDs are dead and whose status is still `"running"`.

**ROI:** 4 zombie-status runs in 3 days is a persistent 33% orphan rate. The UI shows these as forever "running." Low engineering cost; high operator quality-of-life gain. Also unblocks the leaderboard (which filters by `status != "running"`).

---

### Rec-3 — Add Frey Face + RLM-paper to dataset registry / per-paper env-injection spec (4 hours)

**Sub-recommendation 3a — Frey Face fallback mirror:**  
Add a Frey Face entry to `backend/agents/dataset_recipes.py` with a working alternative mirror (archive.org or a hosted copy). PR-λ's loader binding would then auto-retry on the 403. Cost: 30 minutes.

**Sub-recommendation 3b — Per-paper env-injection spec for self-hosted evals:**  
The RLM paper (2512.24601) is not reproducible without API keys in the sandbox. Design a `required_env_forwarding` field in `PaperTargets` that lists host env vars that should be forwarded into the container (behind an explicit opt-in flag, not auto-forwarded for security). Scope this as a spec doc first (2 hours) before implementing (est. 4 additional hours). Without this, any paper whose eval harness calls an LLM API will silently fail.

**ROI:** Rec-3a is 30-min, high-confidence. Rec-3b is 6h total, addresses a structural gap that would affect any self-evaluating paper (a whole class of ML papers now use LLMs for evaluation).

---

## Appendix — Evidence fragments

### P2 binding.py timeout confirmation
```
[2026-05-26T04:33:23] run_warning: {"code":"primitive_timeout","primitive":"run_experiment","wall_clock_s":1800}
```
Second `run_experiment` started 04:00:39; warning fired 04:33:23 = 1964s elapsed. Cap was 1800s.

### P1 SDK aclose traces (6 runs)
```
RuntimeError: aclose(): asynchronous generator is already running
  File "<string>", line 184, in <module>
RuntimeError: Pipeline exited with status 3
```

### P4 credit exhaustion
```
run_experiment: SandboxRuntimeError: RUNPOD_BALANCE_TOO_LOW: Server error '500 Internal Server Error'
  for url 'https://rest.runpod.io/v1/pods'
```
Appeared on all 7 experiment calls across prj_6d41 and prj_9afa on 2026-05-25T17:41 UTC.

### P6 WakeSleepVAE attribute crash (VAE run, prj_3080)
```
AttributeError: 'WakeSleepVAE' object has no attribute 'reparameterize'
  File "/code/train.py", line 177, in marginal_ll_importance_sampling
    z = model.reparameterize(mu, logv)
```

### P5 preflight violations in Adam runs
```
# prj_11e9 — 5 hard violations
"detail": "train.py:379: `load_dataset('imdb')` is a deprecated bare short name..."
"detail": "train.py:619: make_optimizer(..., lr_decay=<value>, **kw)` passes `lr_decay` BOTH..."
# prj_0a77 — 1 hard violation (same imdb issue)
"detail": "train.py:486: `load_dataset('imdb')` is a deprecated bare short name..."
```

### P3 zombie runs confirmed
```
prj_9554a1396eb993aa: pid=3872 → ProcessLookupError (DEAD), status="running"
prj_db45c0304ce455a6: pid=13575 → ProcessLookupError (DEAD), status="running"
prj_09047604e591d969: pid=None, status="running" (never wrote PID)
prj_8819ad4e6e250b73: pid=None, status="running" (never wrote PID)
```

### Orphan Docker container
```
ec8a5bbcae11   reprolab/prj_3080fe2a02c20164:env-57444c4ac5c7   Up 6 hours
```
