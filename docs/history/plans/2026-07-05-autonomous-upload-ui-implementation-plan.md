# Autonomous Upload → Live Reproduction UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`)
> syntax for tracking. Fresh Sonnet-max implementer per task; **Opus reviews every diff** (two-stage:
> the implementer self-verifies, then a fresh reviewer with mutation-armed skepticism).

**Goal:** Wire the already-built Anthropic-Foundry reproduction engine to the web UI so a PDF upload (or
arXiv id) triggers an autonomous GCP/GKE reproduction driven by an Opus-4.8 root, shows a visible
spec-generation + external-validation phase, and seamlessly redirects to a live agentic-reasoning session
view — in the alphaXiv maroon / neo-brutalist aesthetic.

**Architecture:** Backend seam is additive + opt-in (`autonomous: bool = False`). When set, a new
`apply_autonomous_profile_override` forces `sandbox=gke` + `model=opus-foundry` + a server-fixed
`run_spec` (the canonical autonomous profile). A new `spec_validator.py` (structural sibling of
`external_validator.py`) machine-checks the generated rubric against the full paper before any GPU spend
and emits 4 new SSE events. The frontend adds a scoped neo-brutalist design kit + new pages (landing,
repo-confirm, spec-validation stepper) + a new `SessionReasoningView` re-skinning the existing SSE stream.
The existing dark lab is **not** repainted — the new flow is a separate, light-maroon surface.

**Tech Stack:** Python 3.12 / FastAPI / pydantic v2 (backend); `rlm` (`rlms`) orchestrator; Next.js 16
App Router / React 19 / Tailwind 3 + CSS Modules / Vitest / Playwright (frontend). Foundry Anthropic
endpoint for `opus-foundry`/`sonnet-foundry`.

## Global Constraints

Every task's requirements implicitly include this section.

- **Base commit:** `ed086edb` (foundation: Anthropic-Foundry provider + lifecycle-primary + execute-mode)
  on `feat/autonomous-upload-ui`. Do NOT rebase/rewrite below it.
- **Opt-in / default-OFF is byte-identical.** `autonomous` defaults `False`; `OPENRESEARCH_SPEC_VALIDATOR`
  master flag defaults OFF. Unset ⇒ every existing run path is byte-for-byte unchanged. This is a hard,
  test-enforced invariant (each backend task ships an OFF-state test proving no behavior change).
- **Corpus isolation (hard invariant).** The 4 new SSE spec-phase events + the session view carry ONLY
  ids/counts/verdict-enums/leaf-ids — NEVER paper text. Route through the existing egress discipline
  (`sse_bridge`). `spec_validator` reads the full paper server-side; its verdict persists leaf-ids only.
- **Secrets server-side.** The Foundry key + cloud creds stay in `CredentialBroker`/`.env`; never sent to
  the browser, never embedded in a run-spec on the wire. `configs/autonomous_reproduction_run_spec.json`
  contains ONLY `OPENRESEARCH_*` flags (no secrets).
- **Auth before spend (D4).** The GPU-spending launch is gated by a confirm dialog (estimated cap) + an
  auth gate + a hard per-run `OPENRESEARCH_MAX_RUN_GPU_USD` from the autonomous profile.
- **Model tokens (from the committed foundation):** root = `opus-foundry`; executor/grader/verifier =
  `sonnet-foundry`; spec-validator = funded `azure-foundry`/`grok` (cross-family vs the Opus/Sonnet stack),
  pluggable via `OPENRESEARCH_SPEC_VALIDATOR_BACKEND`/`_MODEL`.
- **TDD, commit per task.** Test-first; run to see it fail; minimal impl; run to see it pass; commit. No
  AI-attribution trailers; descriptive present-tense commit headlines (what+symptom+resolution).
- **Backend test env:** `.venv/bin/python -m pytest …` (socket-hermetic; `pytest-socket` installed).
  **Frontend test env:** `source ~/.nvm/nvm.sh && nvm use v22.14.0`; `npm test` (vitest); `npx playwright
  test` (no npm alias); lab suite `--no-file-parallelism`.
- **ruff clean** on touched Python (`uvx ruff@0.15.16 check .`); `npx tsc --noEmit` clean on touched TS.
- **Baseline known-failing tests** (env-dependent; exit bar = NO NEW failures beyond these):
  `test_accelerator`, `test_external_validator`, `test_report_validation_stamp`,
  `test_gcp_orchestrator_settings::test_claude_code_oauth_token_prefixed_env_override`. All should pass in
  this worktree since `.env` is present; if any fails, it is pre-existing, not yours.
- **WS-F external-monitor stream** (`backend/services/external_monitor/`, `backend/routes/external_runs.py`,
  `frontend/src/**/external-runs/`, `app.py`/`config.py` poller wiring) lives UNCOMMITTED in the *primary*
  worktree only. It is file-disjoint from this feature. Do NOT touch or depend on it.

## Dependency graph (build order)

```
Backend (Python, hermetic — can run in parallel with Frontend Phase 0):
  T1 advanced-field fix ─┐
  T2 autonomous+run_spec fields ─┬─► T3 autonomous profile override + run-spec config
  T4 spec_validator sub-role ────┼─► T5 build_spec_validator_client ──► T6 spec_validator.py ──┐
  T7 SSE spec-phase builders ────┘                                                              ├─► T8 run.py hook + context + report stamp + emit
  T3b (P1) pre-run repo-resolve endpoint (best-effort)                                          ┘

Frontend:
  T9a tokens + base kit (Card/PrimaryButton/Eyebrow) ─► T9b flow kit (RepoField/Stepper/ReasoningChip/SessionRail/PaperTabs)
  T10 thread autonomous+repoUrl through launch chain  (needs T2/T3 backend contract)
  T11 landing (_3) + repo-confirm (_4) + cost-guard    (needs T9b, T10, T3b)
  T12 spec-validation stepper (screen D)               (needs T9b, T8 SSE events)
  T13 SessionReasoningView (_5) + session rail          (needs T9b; rides existing SSE)
  T14 e2e + integration (Playwright + vitest flow)      (needs T10-T13)
```

## File Structure

**Backend — modify:**
- `backend/app.py` — read advanced + `autonomous` form/JSON fields; declare them on `StartArxivRunRequest`.
- `backend/services/events/live_runs.py` — `StartRunRequest` (+`autonomous`,+`run_spec`); `_subprocess_env`
  (thread 4 budget fields); `apply_autonomous_profile_override`; `common` dict + uploaded `Namespace`
  (thread `run_spec`).
- `backend/agents/rlm/role_models.py` — `spec_validator` sub-role (mirror `validator`).
- `backend/agents/rlm/grader_transport.py` — `build_spec_validator_client` (mirror `build_validator_client`).
- `backend/agents/rlm/context.py` — `RunContext.spec_validator_client` field.
- `backend/agents/rlm/run.py` — spec-validator client build + separation tier + hook + 4 event emits.
- `backend/agents/rlm/sse_bridge.py` — 4 event builders + `__all__`.
- `backend/agents/rlm/report.py` — `RLMFinalReport.spec_validation` field + stamp.
- `backend/agents/rlm/rubric_gen.py` — none (targets already module-level/importable).

**Backend — create:**
- `backend/agents/rlm/spec_validator.py` — the rubric-vs-paper validator module.
- `configs/autonomous_reproduction_run_spec.json` — the canonical autonomous profile.
- `backend/routes/papers_resolve.py` (T3b, P1) — best-effort pre-run repo-resolve endpoint.

**Frontend — create (new, scoped — does NOT touch the dark lab):**
- `frontend/src/styles/autoresearch-tokens.css` — light/maroon/neo-brutalist scoped tokens.
- `frontend/src/components/autoresearch/ui/{Card,PrimaryButton,Eyebrow,RepoField,Stepper,ReasoningChip,SessionRail,PaperTabs}.tsx` (+ `.module.css` + `.test.tsx`).
- `frontend/src/components/autoresearch/{PaperLanding,RepoConfirm,SpecValidationStepper,SessionReasoningView}.tsx`.
- `frontend/src/app/abs/[arxivId]/page.tsx` — landing route (`_3`).
- `frontend/src/app/orgs/[org]/abs/[arxivId]/page.tsx` — repo-confirm route (`_4`) (or a query-state on `/abs`).
- `frontend/src/app/sessions/[runId]/page.tsx` — live session route (`_5`).
- `frontend/src/lib/autoresearch/session-events.ts` — reducer handlers for the 4 spec-phase events.
- `frontend/e2e/autoresearch-flow.spec.ts` — Playwright e2e.

**Frontend — modify:**
- `frontend/src/hooks/use-run.ts` — `ProviderRunOptions` (+`autonomous`,+`repoUrl`); 3 launchers send them.
- `frontend/src/lib/user-prefs.ts` — `ProviderPrefs` (+`autonomous`); persist.
- `frontend/src/components/lab/lab-shell.tsx` — state + `useRun` options + prop pass + `on*Change`.
- `frontend/src/components/lab/upload-view.tsx` — prop contract + autonomous toggle + repoUrl field.
- `frontend/src/lib/events/rlm-events.ts` — type the 4 spec-phase event shapes (extend the union).
- `frontend/src/hooks/use-rlm-run.ts` — `fold()` handlers for the 4 spec-phase events.

---

## Task 1: §3.1 Advanced-field forwarding (backend-only)

The frontend already SENDS all advanced fields (camelCase FormData for upload, snake_case JSON for arxiv);
the backend drops them. `gpu_parallelism`/`accelerator` are already threaded in `_subprocess_env`;
`dynamic_gpu`/`force_single_gpu`/`max_gpu_usd_per_hour`/`vram_gb` are dropped there. `root_provider`/
`subagent_auth` have NO backend consumer (verified) — forward for model parity but they remain inert.

**Files:**
- Modify: `backend/app.py` — `/runs/upload` form reads (~`:675-690`); `StartArxivRunRequest` (`:1131-1158`)
  + arxiv reconstruction (`:637-649`).
- Modify: `backend/services/events/live_runs.py` — `_subprocess_env` (after `:540`).
- Test: `tests/routes/test_advanced_field_forwarding.py` (new).

**Interfaces:**
- Consumes: `StartRunRequest` fields (`live_runs.py:175-218`), `_optional_form_value`/`_optional_form_bool`
  (`app.py:1219`,`:1226`).
- Produces: end-to-end forwarding so `_subprocess_env` sets `OPENRESEARCH_DYNAMIC_GPU`,
  `OPENRESEARCH_FORCE_SINGLE_GPU`, `OPENRESEARCH_MAX_GPU_USD_PER_HOUR`, `OPENRESEARCH_VRAM_OVERRIDE_GB`.

- [ ] **Step 1: Write the failing test**

```python
# tests/routes/test_advanced_field_forwarding.py
from backend.services.events.live_runs import StartRunRequest, FileLiveRunService

def _svc():
    return FileLiveRunService.__new__(FileLiveRunService)  # env-only method under test

def test_subprocess_env_threads_budget_fields():
    req = StartRunRequest(dynamic_gpu=True, force_single_gpu=True,
                          max_gpu_usd_per_hour=7.5, vram_gb=24)
    env = FileLiveRunService._subprocess_env(_svc(), req)
    assert env["OPENRESEARCH_DYNAMIC_GPU"] == "true"
    assert env["OPENRESEARCH_FORCE_SINGLE_GPU"] == "true"
    assert env["OPENRESEARCH_MAX_GPU_USD_PER_HOUR"] == "7.5"
    assert env["OPENRESEARCH_VRAM_OVERRIDE_GB"] == "24"

def test_subprocess_env_omits_unset_budget_fields():
    env = FileLiveRunService._subprocess_env(_svc(), StartRunRequest())
    for k in ("OPENRESEARCH_DYNAMIC_GPU", "OPENRESEARCH_FORCE_SINGLE_GPU",
              "OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "OPENRESEARCH_VRAM_OVERRIDE_GB"):
        assert k not in env

def test_arxiv_request_declares_advanced_fields():
    from backend.app import StartArxivRunRequest
    r = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155",
                             dynamic_gpu=True, vram_gb=48)
    assert r.dynamic_gpu is True and r.vram_gb == 48
```

- [ ] **Step 2: Run — expect FAIL** (`KeyError`/`ValidationError`: fields unset/undeclared).
  `.venv/bin/python -m pytest tests/routes/test_advanced_field_forwarding.py -q`

- [ ] **Step 3: Implement.** In `live_runs.py::_subprocess_env`, after the existing accelerator block
  (`:537`), mirror the pattern (booleans → `"true"/"false"`, numbers → `str`):

```python
        if request.dynamic_gpu is not None:
            env["OPENRESEARCH_DYNAMIC_GPU"] = "true" if request.dynamic_gpu else "false"
        if request.force_single_gpu is not None:
            env["OPENRESEARCH_FORCE_SINGLE_GPU"] = "true" if request.force_single_gpu else "false"
        if request.max_gpu_usd_per_hour is not None:
            env["OPENRESEARCH_MAX_GPU_USD_PER_HOUR"] = str(request.max_gpu_usd_per_hour)
        if request.vram_gb is not None:
            env["OPENRESEARCH_VRAM_OVERRIDE_GB"] = str(request.vram_gb)
```

  In `app.py`: add the 8 fields to `StartArxivRunRequest` (mirror the existing optional-field style), forward
  them in the arxiv reconstruction (`:637-649`), and in `/runs/upload` read them from the form:

```python
            dynamic_gpu=_optional_form_bool(form, "dynamicGpu"),
            force_single_gpu=_optional_form_bool(form, "forceSingleGpu"),
            gpu_parallelism=_optional_form_value(form, "gpuParallelism"),
            accelerator=_optional_form_value(form, "accelerator"),
            max_gpu_usd_per_hour=_optional_form_float(form, "maxGpuUsdPerHour"),
            vram_gb=_optional_form_int(form, "vramGb"),
            root_provider=_optional_form_value(form, "rootProvider"),
            subagent_auth=_optional_form_value(form, "subagentAuth"),
```

  Add `_optional_form_float`/`_optional_form_int` beside `_optional_form_value` (None/"" → None; else
  `float(...)`/`int(...)`; on `ValueError` → None). The arxiv path is JSON → pydantic coerces natively.

- [ ] **Step 4: Run — expect PASS.** Full: `.venv/bin/python -m pytest tests/routes/ -q`

- [ ] **Step 5: Commit.**
  `git add backend/app.py backend/services/events/live_runs.py tests/routes/test_advanced_field_forwarding.py`
  `git commit -m "Forward the advanced GPU fields on the upload + arxiv run paths (were silent no-ops)"`

---

## Task 2: `autonomous` + `run_spec` request fields + run_spec threading

`autonomous` only needs to reach `_start_python_run` (where T3's override consumes it). `run_spec` must
reach the child's `cmd_reproduce` Namespace (`args.run_spec`) via `common` → uploaded `Namespace`.

**Files:**
- Modify: `live_runs.py` — `StartRunRequest` (+`autonomous: bool = False` `:219`, +`run_spec: str | None = None`);
  `common` dict (`:1777`, +`run_spec`); uploaded `Namespace` (`:1979`, +`run_spec`).
- Modify: `app.py` — `StartArxivRunRequest` (+`autonomous`); `/runs/upload` form read
  `_optional_form_bool(form,"autonomous")`; arxiv reconstruction forwards `autonomous`.
- Test: `tests/routes/test_autonomous_request_threading.py` (new).

**Interfaces:**
- Produces: `StartRunRequest.autonomous: bool`, `StartRunRequest.run_spec: str | None`; `run_spec` present in
  the generated `Namespace` so `cmd_reproduce` picks it up via `getattr(args, "run_spec", None)`
  (`cli.py:1621`). Consumed by T3.

- [ ] **Step 1: Write the failing test**

```python
# tests/routes/test_autonomous_request_threading.py
from backend.services.events.live_runs import StartRunRequest

def test_request_has_autonomous_and_run_spec_defaults():
    r = StartRunRequest()
    assert r.autonomous is False and r.run_spec is None

def test_run_spec_flows_into_uploaded_namespace(tmp_path, monkeypatch):
    # Build the `common` dict + Namespace the way _python_script does and assert run_spec rides it.
    from backend.services.events import live_runs as lr
    req = StartRunRequest(run_spec="configs/autonomous_reproduction_run_spec.json")
    common = lr.FileLiveRunService._build_common_for_test(req, project_id="p1")  # thin helper (Step 3)
    assert common["run_spec"] == "configs/autonomous_reproduction_run_spec.json"
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError`/`ValidationError`).

- [ ] **Step 3: Implement.** Add the two fields to `StartRunRequest`; add `autonomous` to
  `StartArxivRunRequest` + forward it; add `"run_spec": request.run_spec` to the `common` dict (`:1777`);
  add `run_spec=config.get("run_spec")` to the uploaded `Namespace` (`:1979`). Extract the `common`-dict
  construction into a tiny `_build_common_for_test` classmethod-or-staticmethod OR test the real
  `_python_script` output — pick the lowest-surface option that lets the test assert `common["run_spec"]`
  without spawning a subprocess. `_REPRODUCE_DEFAULTS` already has `run_spec: None` (`cli.py:816`), so the
  Namespace key is consumed for free.

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit.**
  `git commit -m "Thread an opt-in autonomous flag + run_spec through the run-launch request chain"`

---

## Task 3: `apply_autonomous_profile_override` + the autonomous run-spec config

**Files:**
- Modify: `live_runs.py` — new `apply_autonomous_profile_override(request) -> StartRunRequest` after
  `apply_provider_override` (`:474`); call it in `_start_python_run` right after the sandbox/provider
  override chain (`:786`).
- Create: `configs/autonomous_reproduction_run_spec.json`.
- Test: `tests/routes/test_autonomous_profile_override.py` (new).

**Interfaces:**
- Consumes: `StartRunRequest.autonomous` (T2). Produces: when `autonomous`, a request with `sandbox="gke"`,
  `model="opus-foundry"`, `run_spec="configs/autonomous_reproduction_run_spec.json"`. OFF ⇒ request unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/routes/test_autonomous_profile_override.py
import json, pathlib
from backend.services.events.live_runs import StartRunRequest, apply_autonomous_profile_override
from backend.agents.rlm.run_spec_contract import run_spec_key_applies

def test_override_off_is_identity():
    r = StartRunRequest(sandbox="runpod", model="sonnet")
    assert apply_autonomous_profile_override(r) is r or apply_autonomous_profile_override(r) == r

def test_override_forces_gke_opus_runspec():
    r = apply_autonomous_profile_override(StartRunRequest(autonomous=True))
    assert r.sandbox == "gke" and r.model == "opus-foundry"
    assert r.run_spec == "configs/autonomous_reproduction_run_spec.json"

def test_autonomous_run_spec_keys_all_apply():
    spec = json.loads(pathlib.Path("configs/autonomous_reproduction_run_spec.json").read_text())
    bad = [k for k in spec if not run_spec_key_applies(k) and k not in ("models","baseline_extra_guidance")]
    assert bad == [], f"run-spec keys the contract rejects: {bad}"

def test_autonomous_run_spec_has_no_sdar_pins():
    spec = json.loads(pathlib.Path("configs/autonomous_reproduction_run_spec.json").read_text())
    for forbidden in ("OPENRESEARCH_REPO_LOCAL_PATH","OPENRESEARCH_REPO_COMMIT","HF_HOME"):
        assert forbidden not in spec
```

- [ ] **Step 2: Run — expect FAIL** (function missing / config missing).

- [ ] **Step 3: Implement.**

```python
# live_runs.py, after apply_provider_override
_AUTONOMOUS_RUN_SPEC = "configs/autonomous_reproduction_run_spec.json"

def apply_autonomous_profile_override(request: "StartRunRequest") -> "StartRunRequest":
    """Opt-in autonomous profile: GKE dispatch + Opus-4.8-Foundry root + the canonical run-spec.
    model + sandbox MUST be explicit request overrides (the uploaded Namespace passes an explicit
    model/sandbox that wins over env). OFF ⇒ identity."""
    if not getattr(request, "autonomous", False):
        return request
    return request.model_copy(update={
        "sandbox": "gke",
        "model": "opus-foundry",
        "run_spec": request.run_spec or _AUTONOMOUS_RUN_SPEC,
    })
```

  Wire in `_start_python_run` after `:786`: `request = apply_autonomous_profile_override(request)`.

  Create `configs/autonomous_reproduction_run_spec.json` (paper-agnostic; every key must pass
  `run_spec_key_applies`):

```json
{
  "OPENRESEARCH_LIFECYCLE_PRIMARY": "1",
  "OPENRESEARCH_CELL_RESUME_AUTO": "1",
  "OPENRESEARCH_USE_AUTHOR_REPO": "1",
  "OPENRESEARCH_REPRODUCTION_MODE": "execute",
  "OPENRESEARCH_ZERO_METRICS_GUARD": "1",
  "OPENRESEARCH_EXTERNAL_VALIDATOR": "1",
  "OPENRESEARCH_EVIDENCE_GATE": "1",
  "OPENRESEARCH_EVIDENCE_AUDIT": "1",
  "OPENRESEARCH_ENV_LIVENESS_GATE": "1",
  "OPENRESEARCH_NO_LEARNING_SIGNAL_GATE": "1",
  "OPENRESEARCH_EVAL_PROVENANCE_GUARD": "1",
  "OPENRESEARCH_STUB_METRICS_GUARD": "1",
  "OPENRESEARCH_TWO_AXIS_VERDICT": "1",
  "OPENRESEARCH_SPEC_VALIDATOR": "1",
  "OPENRESEARCH_MAX_RUN_GPU_USD": "10.0",
  "models": "executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry,spec_validator=grok"
}
```

  NOTE: verify each `OPENRESEARCH_*` key is real (`grep -rn "<KEY>" backend/`). Drop any that
  `run_spec_key_applies` rejects (the test enforces this). `models` + `baseline_extra_guidance` are the two
  special non-prefixed keys the contract accepts.

- [ ] **Step 4: Run — expect PASS.** `.venv/bin/python -m pytest tests/routes/test_autonomous_profile_override.py -q`
- [ ] **Step 5: Commit.**
  `git commit -m "Add the autonomous-profile override (GKE + Opus-Foundry root + canonical run-spec)"`

---

## Task 3b (P1): Best-effort pre-run repo-resolve endpoint

The `_4` repo-confirm screen shows an auto-found repo (e.g. `ZJU-REAL/SDAR`) BEFORE launch. The in-run
`RepoResolver` clones inside the run; this endpoint does a cheap resolve-without-clone for the UI. Fail-soft
(empty suggestion ⇒ the field is editable/blank; the run's own resolver still finds it).

**Files:**
- Create: `backend/routes/papers_resolve.py` — `GET /papers/{arxiv_id}/repo` → `{repo_url, provider, confidence}`
  or `{repo_url: null}`. Reuse `backend/services/ingestion/repo/` `RepoResolver.resolve(...)` WITHOUT provision.
- Modify: `backend/app.py` — `app.include_router(papers_resolve_router)`.
- Test: `tests/routes/test_papers_resolve.py` — mock the resolver; assert fail-soft `{repo_url: null}` on
  resolve failure, and a well-formed body on success.

- [ ] Steps 1-5 mirror Task 1's TDD cycle. **Verify** `RepoResolver`'s resolve-only API first
  (`grep -n "class RepoResolver" backend/services/ingestion/repo/*.py`; find a method that resolves without
  cloning, or add a thin `resolve_only`). Keep it a 5-second bounded, fail-soft GET.

---

## Task 4: `spec_validator` sub-role in `role_models.py`

Mirror the `validator` sub-role EXACTLY.

**Files:** Modify `backend/agents/rlm/role_models.py`. Test: `tests/rlm/test_role_models_spec_validator.py`.

**Interfaces:**
- Produces: `RoleSelection.spec_validator: RoleSpec | None`; `resolve_role_models(..., spec_validator_model_setting=None)`;
  `.stamp()` includes `spec_validator`; `separation_strength(planner_spec, spec_validator_spec)` reusable.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_role_models_spec_validator.py
from backend.agents.rlm.role_models import resolve_role_models, RoleSelection

def test_spec_validator_unset_is_none():
    sel = resolve_role_models(planner_token="opus-foundry")
    assert sel.spec_validator is None

def test_spec_validator_override_resolves_and_stamps():
    sel = resolve_role_models(planner_token="opus-foundry", cli_models="spec_validator=grok")
    assert sel.spec_validator is not None
    assert "spec_validator" in sel.stamp()
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement.** Add `"spec_validator"` to `ROLES` (`:99`) + `_SUBROLES` (`:104`); add
  `spec_validator: RoleSpec | None = None` to `RoleSelection` (`:291`); include it in `.stamp()` (`:305-319`),
  `explicit_subroles` (`:296-303`), and `fidelity_warnings` if applicable; resolve it in
  `resolve_role_models` via `_resolve_subrole("spec_validator", ...)` mirroring the `validator` block
  (`:556-563`). Keep any role-count assertions in the module + tests in sync.
- [ ] **Step 4: Run — expect PASS** (+ `tests/rlm/test_role_models.py` still green).
- [ ] **Step 5: Commit.** `git commit -m "Add the spec_validator sub-role to the role-model resolver"`

---

## Task 5: `build_spec_validator_client` in `grader_transport.py`

Mirror `build_validator_client` (`:358-420`); read `OPENRESEARCH_SPEC_VALIDATOR_BACKEND`/`_MODEL`, fall back
to `OPENRESEARCH_VALIDATOR_BACKEND`/`_MODEL`; FAIL-CLOSED (raise if the resolved client IS the planner
fallback — never silently judge the rubric with the same lineage that wrote it).

**Files:** Modify `backend/agents/rlm/grader_transport.py`. Test: `tests/rlm/test_grader_transport_spec_validator.py`.

**Interfaces:** Produces `build_spec_validator_client(*, fallback_client, fallback_label="") -> tuple[client, label]`.

- [ ] **Step 1: failing test** — assert it reads the SPEC_VALIDATOR_* env, falls back to VALIDATOR_*, and
  raises `ValueError` when the resolved client is identity-equal to `fallback_client` (mirror
  `test_grader_transport`'s validator fail-closed test). **Step 2** FAIL. **Step 3** implement via the shared
  `build_transport_client(...)` dispatch (`:142-322`) with `role_label="spec_validator"`; add to `__all__`
  (`:423`). **Step 4** PASS. **Step 5** `git commit -m "Add build_spec_validator_client (fail-closed, VALIDATOR_* fallback)"`.

---

## Task 6: `spec_validator.py` — the rubric-vs-paper validator module

Structural sibling of `external_validator.py`. Checks RUBRIC-vs-PAPER (not metrics-vs-disk), ONCE pre-loop.
LLM only points at suspicions; every veto is a harness machine-check (min-aggregation).

**Files:** Create `backend/agents/rlm/spec_validator.py`. Test: `tests/rlm/test_spec_validator.py`.

**Interfaces (mirror `external_validator`'s dataclasses/functions):**
- `@dataclass(frozen=True) SpecPredicateVerdict(predicate, leaf_id, violated, detail)`
- `@dataclass(frozen=True) SpecValidatorVerdict(status, flagged_leaves, predicates, panel_models, separation, rubric_fingerprint)`
- `spec_validator_enabled() -> bool` (`OPENRESEARCH_SPEC_VALIDATOR`)
- `spec_validator_panel_n() -> int` (`OPENRESEARCH_SPEC_VALIDATOR_PANEL_N`, default 2)
- `spec_validator_block_enabled() -> bool` (`OPENRESEARCH_SPEC_VALIDATOR_BLOCK`)
- `rubric_fingerprint(rubric: dict) -> str` (sha256 of canonical-JSON rubric)
- `run_spec_validation_panel(*, spec_validator_client, panel_models, rubric, paper_text, separation) -> SpecValidatorVerdict`
- `persist_spec_verdict(project_dir, verdict)` / `load_spec_verdict(project_dir, *, expect_fingerprint=None)`
- `apply_block(rubric, verdict) -> dict` (drop machine-CONFIRMED `hallucinated_leaf`/`wrong_target` leaves +
  renormalize sibling weights via `rubric_gen._normalize_weights`; NEVER `missing_key_claim`; never hard-abort)

**Predicates (machine-checked; reuse `paper_grounding._normalize`/`_token_overlap`,
`rubric_gen._is_placeholder_requirement`):**
- `hallucinated_leaf` — leaf cites a dataset/method/number with <0.5 max token-overlap vs the FULL paper text.
- `wrong_target` — leaf's numeric target contradicts the paper's number for that metric (only when BOTH
  extract cleanly; else fail-soft `violated=False`).
- `placeholder_leaf` — `_is_placeholder_requirement(leaf)` re-check (vendored-rubric belt-and-suspenders).
- `missing_key_claim` — ADVISORY only, never auto-remediated (open-world absence).

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_spec_validator.py
from backend.agents.rlm import spec_validator as sv

_RUBRIC = {"leaves": [
    {"id": "L1", "requirement": "Report ALFWorld success rate near 84.4"},   # grounded
    {"id": "L2", "requirement": "Report ImageNet top-1 accuracy of 99.9"},   # hallucinated (absent)
]}
_PAPER = "SDAR improves over GRPO (+9.4% on ALFWorld ... 84.4 ...). Search-QA, WebShop."

class _FakeClient:  # sample_completions returns a JSON array of suspicions
    def __init__(self, arr): self._arr = arr

def test_hallucinated_leaf_flagged(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L2"}]'])
    v = sv.run_spec_validation_panel(spec_validator_client=_FakeClient(None),
        panel_models=["grok"], rubric=_RUBRIC, paper_text=_PAPER, separation="independent")
    assert v.status == "flagged" and "L2" in v.flagged_leaves and "L1" not in v.flagged_leaves

def test_grounded_leaf_not_flagged_even_if_llm_points_at_it(monkeypatch):
    # LLM opinion is never dispositive: machine-check clears L1 (grounded), so no veto.
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L1"}]'])
    v = sv.run_spec_validation_panel(spec_validator_client=_FakeClient(None),
        panel_models=["grok"], rubric=_RUBRIC, paper_text=_PAPER, separation="independent")
    assert "L1" not in v.flagged_leaves

def test_apply_block_drops_confirmed_and_renormalizes():
    v = sv.SpecValidatorVerdict(status="flagged", flagged_leaves=["L2"],
        predicates=[sv.SpecPredicateVerdict("hallucinated_leaf","L2",True,"absent")],
        panel_models=["grok"], separation="independent", rubric_fingerprint="x")
    out = sv.apply_block(_RUBRIC, v)
    assert [l["id"] for l in out["leaves"]] == ["L1"]

def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR", raising=False)
    assert sv.spec_validator_enabled() is False
```

- [ ] **Step 2: Run — expect FAIL** (module missing). **Step 3: Implement** by copying `external_validator.py`'s
  shape and swapping the metrics-vs-disk machine-checks for the rubric-vs-paper predicates above; reuse
  `_parse_suspicions` (fence-tolerant JSON-array parse), min-aggregation (`violated` iff ANY panelist +
  the machine-check confirms), and the atomic `persist`/`load` with fingerprint staleness. **Step 4: PASS.**
  **Step 5:** `git commit -m "Add spec_validator: machine-checked rubric-vs-paper pre-loop validation"`

---

## Task 7: The 4 SSE spec-phase event builders

**Files:** Modify `backend/agents/rlm/sse_bridge.py` (builders near `:554` after `build_repo_resolved_event`;
add all 4 to `__all__` `:965`). Test: `tests/rlm/test_spec_phase_events.py`.

**Interfaces (corpus-free — ids/counts/enums only):**
- `build_spec_generation_started_event() -> dict`
- `build_spec_generated_event(*, leaf_count: int) -> dict`
- `build_spec_validation_started_event(*, validator_model: str) -> dict`
- `build_spec_validated_event(*, verdict: str, flagged_leaves: list[str]) -> dict`
- Each returns `{"event": "<name>", "timestamp": _now_iso(), ...}`.

- [ ] **Step 1: failing test** — assert each builder returns the right `event` name + fields, and that NO
  builder accepts/emits paper text (only ints/enums/leaf-id lists). **Step 2** FAIL. **Step 3** implement
  mirroring `build_run_warning_event`. **Step 4** PASS. **Step 5**
  `git commit -m "Add the 4 spec-phase SSE event builders (corpus-free)"`.

---

## Task 8: run.py spec-validator hook + context field + report stamp + event emits (integration)

Ties Tasks 4/5/6/7 into `run_pipeline_rlm`, strictly between the rubric cascade (`run.py:3564`) and the
`RLM(...)` construction (`:3670`) — before any of the 18 primitives can run (a true "before any GPU spend"
gate). All fail-soft; OFF ⇒ byte-identical.

**Files:**
- Modify: `backend/agents/rlm/context.py` (+`spec_validator_client: Any = None`, `:58`).
- Modify: `backend/agents/rlm/run.py` — build `spec_validator_client` (mirror the validator block `:3178-3191`
  + unified-surface bridge `:3166-3169`); `_spec_validator_separation_tier(role_selection)` (mirror
  `_validator_separation_tier` `:2735`); thread into `RunContext(...)` (`:3298`); the hook
  `_run_spec_validator(ctx, rubric, paper_text, project_dir, emit)` after `:3564`; emit
  `spec_generation_started` before `:3551`, `spec_generated{leaf_count}` after `:3564`,
  `spec_validation_started{validator_model}`/`spec_validated{verdict,flagged_leaves}` bracketing the hook.
- Modify: `backend/agents/rlm/report.py` (+`spec_validation: dict` field `:207`; parallel stamp beside
  `:1957-2006`, loading `load_spec_verdict(project_dir, expect_fingerprint=rubric_fingerprint(rubric))`).
- Test: `tests/rlm/test_spec_validator_hook.py` — ON: hook fires, verdict persisted, report stamped, 4 events
  emitted (capture via a fake `emit`); OFF (flag unset): no hook, no events, no `spec_validation` on report
  (byte-identical). BLOCK ON: confirmed-bad leaf dropped before `RLM(...)`.

- [ ] **Step 1: failing test** (fake `llm_client` + fake `spec_validator_client`; assert the 4 events + a
  persisted `rlm_state/spec_validation_verdict.json` + `report.spec_validation.status`). **Step 2** FAIL.
  **Step 3** implement the hook (guarded `if spec_validator_enabled() and rubric and paper_text:`,
  try/except fail-soft), the client build, the separation tier, the emits, the report stamp. **Step 4** PASS
  + run `tests/rlm/test_report_validation_stamp.py` and a broad `tests/rlm/ -q` to confirm no regression.
  **Step 5** `git commit -m "Wire spec_validator into run_pipeline_rlm (pre-loop gate + SSE + report stamp)"`.

---

## Task 9a: Scoped neo-brutalist tokens + base component kit

Do NOT repaint `src/styles/tokens.css` (that flips the whole dark lab). Add a SCOPED light/maroon token set
applied via a wrapper class, plus base primitives for the new flow only.

**Files (all new):**
- `frontend/src/styles/autoresearch-tokens.css` — a `.autoresearch` scope defining the palette.
- `frontend/src/components/autoresearch/ui/{Card,PrimaryButton,Eyebrow}.tsx` (+ `.module.css` + `.test.tsx`).

**Design tokens (from the reference screenshots):**
```css
.autoresearch {
  --or-maroon: #8B1E2D;  --or-maroon-press: #6F1723;  --or-maroon-tint: #F7EBED;
  --or-ink: #141414;     --or-paper: #FDFCFB;         --or-muted: #6B6B6B;
  --or-border: 1.5px solid var(--or-ink);
  --or-shadow: 6px 6px 0 var(--or-ink);   /* hard, un-blurred offset */
  --or-radius: 2px;
  --or-eyebrow-ls: 0.14em;
}
```

**Components:**
- `Card` — white bg, `--or-border`, `--or-shadow`, `--or-radius`; `variant="panel"` = softer (subtle
  border + `2px 2px 0`) for secondary content.
- `PrimaryButton` — maroon bg, white text, `--or-shadow`; press = `translate(3px,3px)` + shadow→`3px 3px 0`.
- `Eyebrow` — uppercase, `letter-spacing: var(--or-eyebrow-ls)`, `--or-muted`, `text-xs`.

- [ ] **Step 1: failing test** — `Card.test.tsx` renders children + carries the neo-brutalist class;
  `PrimaryButton.test.tsx` renders label + fires `onClick`; `Eyebrow.test.tsx` uppercases. Run:
  `nvm use v22.14.0 && npm test -- src/components/autoresearch`. **Step 2** FAIL (components missing).
  **Step 3** implement + apply `frontend-design` skill for polish against `alphaxiv_3.png`/`_4.png`.
  **Step 4** PASS + `npx tsc --noEmit`. **Step 5** `git commit -m "Add scoped neo-brutalist tokens + base UI kit (Card/PrimaryButton/Eyebrow)"`.

---

## Task 9b: Flow-specific component kit

**Files (all new under `frontend/src/components/autoresearch/ui/`):** `RepoField.tsx` (label + bordered input
+ github-link affordance, `_4`), `Stepper.tsx` (ordered steps with pending/active/done + a live dot, screen
D), `ReasoningChip.tsx` (gray tool-call chip row: dot + label, `_5`), `SessionRail.tsx` (project header +
"New agent" + session list with live dot, `_5` left rail), `PaperTabs.tsx` (Paper/Blog/Autoresearch/Audio,
maroon active underline). Each + `.module.css` + `.test.tsx`.

- [ ] TDD per component (render + interaction test each). Acceptance = matches `alphaxiv_4.png` (RepoField),
  `alphaxiv_5.png` (ReasoningChip/SessionRail), `alphaxiv_autoresearc.png` (PaperTabs/Stepper). Apply the
  `frontend-design` skill. Commit: `git commit -m "Add the autoresearch flow component kit"`.

---

## Task 10: Thread `autonomous` + `repoUrl` through the launch chain

**Files:**
- `frontend/src/lib/user-prefs.ts` — `ProviderPrefs` (+`autonomous?: boolean`); `read/writeProviderPrefs`.
- `frontend/src/hooks/use-run.ts` — `ProviderRunOptions` (+`autonomous?: boolean`, +`repoUrl?: string`)
  (`:145-166`); all three launchers send them (`startUploadedRun` camelCase FormData `:511`; `startArxivRun`
  snake_case JSON `:568`; `startFixtureRun` query `:465`).
- `frontend/src/components/lab/lab-shell.tsx` — state (`:142-193`), `useRun` options (`:205-218`), prop pass,
  `on*Change` persist (`:303-354`).
- `frontend/src/components/lab/upload-view.tsx` — prop contract (`:56-152`) + an `autonomous` toggle
  (top-level fieldset, mirror minimize-compute `:564-583`, labelled experimental/opt-in, OFF by default) +
  a `repoUrl` field (in Advanced).

- [ ] **Step 1: failing test** — extend `use-run.test.ts`: assert `startUploadedRun` FormData contains
  `autonomous`; `startArxivRun` JSON contains `autonomous` + `repo_url`. Assert prefs round-trip `autonomous`.
  **Step 2** FAIL. **Step 3** thread the field (camelCase for upload FormData + query; snake_case
  `root_provider`-style for arxiv JSON — match the existing casing per launcher). **Step 4** PASS +
  `npx tsc --noEmit`. **Step 5** `git commit -m "Thread the autonomous toggle + repoUrl through the run-launch UI chain"`.

---

## Task 11: Paper landing (`_3`) + repo-confirm (`_4`) + cost-guard

**Files (new):** `frontend/src/app/abs/[arxivId]/page.tsx` (landing server-component: fetch paper
metadata title+id; render `PaperLanding`), `frontend/src/components/autoresearch/{PaperLanding,RepoConfirm}.tsx`,
`frontend/src/app/orgs/[org]/abs/[arxivId]/page.tsx` (or a `?confirm=1` state on `/abs`).

- **PaperLanding** — `Card` + `Eyebrow "WELCOME TO OPENRESEARCH"` + title + `arXiv <id>` + the two body
  paragraphs + a `PrimaryButton "→ Sign in to start"` (stubbed auth: sets a dev session flag, advances to
  repo-confirm). Matches `alphaxiv_3.png`.
- **RepoConfirm** — `Eyebrow "ARXIV <id>"` + title + "We found the code for this paper…" + a `RepoField`
  pre-filled from `GET /papers/{id}/repo` (T3b; fail-soft empty) + `PrimaryButton "🚀 Start autoresearch"`.
- **Cost-guard (D4)** — the Start button opens a confirm dialog showing the estimated cap
  (`OPENRESEARCH_MAX_RUN_GPU_USD` = $10 from the profile) + requires the (stubbed) auth flag; on confirm it
  calls `startArxivRun`/`startUploadedRun` with `autonomous: true` + `repoUrl: <confirmed>`, then routes to
  `/sessions/<runId>`.

- [ ] TDD: `PaperLanding.test.tsx` (renders metadata + CTA), `RepoConfirm.test.tsx` (pre-fills repo, the
  cost-guard dialog gates launch, confirm calls the launcher with `autonomous:true`). Playwright coverage in
  T14. Commit: `git commit -m "Add the paper-landing + repo-confirm screens with the launch cost-guard"`.

---

## Task 12: Spec-validation stepper (screen D)

The visible "Analyzing → Generating spec → Validating with `<model>` → ✓" phase between launch and the live
view. Consumes the 4 spec-phase SSE events (T7/T8).

**Files (new):** `frontend/src/components/autoresearch/SpecValidationStepper.tsx`;
`frontend/src/lib/autoresearch/session-events.ts` (typed reducers for the 4 events); modify
`frontend/src/lib/events/rlm-events.ts` (extend the event union with the 4 shapes) + `use-rlm-run.ts`
`fold()` (`:1128`) handlers.

- Stepper stages driven by events: `spec_generation_started` → step "Generating reproduction spec" active;
  `spec_generated{leaf_count}` → done (show leaf count); `spec_validation_started{validator_model}` → step
  "Validating spec with `<model>`" active; `spec_validated{verdict, flagged_leaves}` → done (✓ or a
  non-blocking "N flagged" note). On `spec_validated` (advisory, no blocking veto) → auto-navigate to
  `/sessions/<runId>` (the "seamless redirect").

- [ ] TDD: `SpecValidationStepper.test.tsx` drives a fake event sequence → asserts stage transitions + the
  auto-redirect fires on `spec_validated`. `session-events.test.ts` folds each event. Commit:
  `git commit -m "Add the spec-generation + external-validation stepper (screen D) + spec-event reducers"`.

---

## Task 13: `SessionReasoningView` (`_5`) + session rail

The centerpiece: the existing SSE stream rendered as a clean light-maroon session log. Build NEW (do not edit
the dark `rlm-lab`). Rides the SAME `/api/demo/events` proxy + the existing `useRlmRun` reducer state
(`iterations`, `primitiveCalls`, `rubric`).

**Files (new):** `frontend/src/app/sessions/[runId]/page.tsx`,
`frontend/src/components/autoresearch/SessionReasoningView.tsx` (consume `use-run.ts`'s EventSource /
`dashboardEvents` via `useRlmRunBatched`; render interleaved `repl_iteration` sanitized reasoning text +
`ReasoningChip` panels for `primitive_call` + mono id pills + maroon links + a `RubricStrip`-equivalent).
Left rail = `SessionRail` (project name + "New agent" + session list from `localStorage`/leaderboard).
Reuse `use-steering-chat.ts` for the docked steering input.

- **Corpus invariant:** render ONLY sanitized SSE fields (the egress sanitizer already strips corpus). No raw
  `context`/`locals`/prompt fields.

- [ ] TDD: `SessionReasoningView.test.tsx` feeds a fixture event stream
  (`src/components/lab/rlm/__fixtures__/rlm-run.fixture.ts`) → asserts reasoning text + tool-call chips +
  id pills render, and no corpus field leaks. `SessionRail.test.tsx` (session list + live dot). Commit:
  `git commit -m "Add the live agentic-reasoning session view (_5) + session rail"`.

---

## Task 14: End-to-end + integration

**Files (new):** `frontend/e2e/autoresearch-flow.spec.ts`.

- Playwright e2e: upload/arxiv entry → landing → repo-confirm → (cost-guard confirm, `autonomous:true`) →
  a MOCKED run (backend `/runs/*` + `/runs/<id>/events` stubbed to emit the 4 spec-phase events then
  `repl_iteration`/`primitive_call`) → stepper D advances → auto-redirect → session view renders the reasoning
  + tool chips. Assert the corpus never appears in the DOM.
- Backend integration (hermetic): a test that `autonomous=True` end-to-end forces `sandbox=gke` +
  `model=opus-foundry` + `run_spec=<path>` on the constructed run request (compose T1-T3), and `autonomous=False`
  is byte-identical.

- [ ] Run `npx playwright test e2e/autoresearch-flow.spec.ts` + the full `npm test -- --no-file-parallelism`
  + `.venv/bin/python -m pytest tests/ -q` (no NEW failures beyond the baseline set). Commit:
  `git commit -m "Add the autonomous-flow e2e + upload→autonomous integration coverage"`.

---

## Self-review (author checklist — run before execution)

**Spec coverage:** §3.1 → T1. §3.2 (autonomous profile + run-spec) → T2/T3. §3.3 spec_validator → T4/T5/T6/T8;
its SSE events → T7/T8; stepper → T12. §3.4 live session view → T13. §3.5 D2 (backend-orchestrated + GKE) →
the profile forces `sandbox=gke` (T3); GKE cluster readiness is an INFRA precondition (`infra/gcp/` + the
IaC-gap audit) tracked separately, not a code task here. Phase 0 design system → T9a/T9b. D4 cost-guard →
T11. Auth stub → T11 (dev session flag). Testing (§7) → each task's tests + T14.

**Known open / operator-owned (NOT code tasks in this plan):**
- GKE cluster production-readiness (`infra/gcp/`) — the live-spend precondition for D2. Until it's ready, the
  autonomous launch is exercisable against the mocked/local SSE path (T14) but a real GPU run needs the cluster.
- The default-flip (`autonomous` becomes the default for every upload) is gated on the SDAR Phase-1 PASS +
  canary (§8) — this plan ships `autonomous` default-OFF only.
- Full org/billing/compute console (`_4` sidebar) — Phase 4, deferred (a minimal session shell suffices here).

**Placeholder scan:** none — every task has a concrete test + impl anchor. Mirror tasks (T4/T5/T6) name the
exact reference (`external_validator.py`, `build_validator_client`, the `validator` sub-role) + the deltas.

**Type consistency:** `SpecValidatorVerdict`/`SpecPredicateVerdict` field names are consistent across T6/T8/
report stamp. The 4 event names (`spec_generation_started`/`spec_generated`/`spec_validation_started`/
`spec_validated`) are identical across T7 (builders), T8 (emits), T12 (reducers). `autonomous`/`run_spec`
field names consistent across T2/T3/T10.
