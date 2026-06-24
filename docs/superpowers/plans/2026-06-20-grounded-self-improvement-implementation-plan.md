# Grounded Self-Improvement + Harness Reliability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the plausible-but-fake-result hallucination class and make *any* RLM root usable across *all* transports, by adding a three-tier trust model (deterministic floor → external adversarial validator → grounded self-improvement) whose fitness signal is the deterministic evidence layer, never the LLM grade.

**Architecture:** Implements `docs/superpowers/specs/2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md` (rev 2, post-Codex). New pure/stdlib modules (`zero_metrics_detection`, `lifecycle_ledger`, `external_validator`, `recipe_library`) plus surgical wiring into existing seams (`primitives.py`, `binding.py`, `role_models.py`, `grader_transport.py`, `forced_iteration.py`, `run.py`, `report.py`, `leaf_triage.py`). Phased P0→P5; every new flag default-OFF; the existing default-ON rails are the baseline.

**Tech Stack:** Python 3.12 (floor 3.11), stdlib-only for the floor/ledger/recipe modules, pytest (hermetic — pytest-socket blocks non-loopback), existing `grader_transport.sample_completions` for the panel, existing `leaf_scorer`/`evidence_gate`/`provenance` evidence-gather.

## Global Constraints

- **Default-OFF == baseline (scoped).** Every NEW flag is default-OFF. With all new flags unset, the harness is byte-identical to its *current baseline*, which already includes default-ON rails: `REPROLAB_FINALIZE_REGRADE`, `REPROLAB_LEAF_TRIAGE`, `OPENRESEARCH_METRIC_PROVENANCE`, report-level `_apply_evidence_gate`. "Byte-identical" means *to that baseline*.
- **Flag naming.** New flags use canonical `OPENRESEARCH_*`. `config.py::_apply_legacy_env_aliases` bridges `REPROLAB_*`→`OPENRESEARCH_*` at import (a var set *after* import is not bridged — use `OPENRESEARCH_*` in code/tests).
- **The red line.** New Tier-3 decisions (what to repair, whether a repair is accepted, whether a recipe is admitted) use Tier-1 predicates (+ the Tier-2 veto), **never** a grade-derived score. `recipe_library` + the repair-acceptance path must not import or read grade-derived score fields except to copy them into the report. Enforced by a static-import guard test.
- **Funded transports = `{azure, oauth}` only.** The operator's deployment funds exactly two transports: **Azure OpenAI** (`AZURE_OPENAI_*`, GPT family) and **Claude OAuth** (`CLAUDE_CODE_OAUTH_TOKEN`, Sonnet family). `OPENAI_API_KEY` is dead, `ANTHROPIC_API_KEY` is empty. Every transport-touching piece MUST construct with only `{azure, oauth}` present. **Two validator panels are first-class supported configs** (a separation-strength ladder, not a binary):
  - `independent` (strongest) — executor=`oauth`(Sonnet, claude family) × validator=`azure`(gpt-4o, gpt family). Cross-family.
  - `weak` (supported, soft notice) — executor=`azure`(deployment A, e.g. gpt-4o) × validator=`azure`(**different deployment/model** B, e.g. gpt-4.1 / o-series). Same provider+family, **different model**. The validator's model is `OPENRESEARCH_VALIDATOR_MODEL` (for azure = an alternate **deployment** overriding `AZURE_OPENAI_DEPLOYMENT`), so two Azure deployments are distinguishable.
  - `degraded` (loud warning) — executor=`azure`(A) × validator=`azure`(**same** A). Same model, different seed only.

  The veto rests on machine-verified deterministic predicates, so even `weak`/`degraded` panels' vetoes are grounded and actionable; the tier only modulates trust in the LLM's suspicion-pointing. No new key dependency anywhere.
- **Fail-soft floor / fail-closed validator.** Tier-1 deterministic guards never raise (fail-soft → no veto on error). The Tier-2 validator transport is fail-CLOSED: a missing/unconstructable validator client yields a `degraded`/`unavailable` verdict, never a silent fall-through to the executor-family client.
- **No paid runs in this plan.** Code + hermetic tests only. The ~$2 `THRESHOLD=16` precondition experiment (spec §5.2) and the ≥3 paired SDAR GPU runs gating any default-flip (spec §13) are operator next-steps, documented not executed. No flag is flipped to default-ON here.
- **Push policy.** Commit at phase milestones; push only on request, to deepinvent only.

---

## Parallelization graph

**Wave 1 — disjoint files, fully parallel (7 agents).** New modules + their hermetic unit tests, plus the role surface, the CLI run-spec, and the cache-resume verification. No two Wave-1 agents edit the same file.

| Agent | Phase | Owns (exclusive) | New files |
|---|---|---|---|
| W1-A | P0 | — | `zero_metrics_detection.py` + test |
| W1-B | P1 | — | `lifecycle_ledger.py` + test |
| W1-C | P2 | — | `external_validator.py` + test |
| W1-D | P2 | `role_models.py`, `grader_transport.py`, `context.py` | role tests |
| W1-E | P4 | — | `recipe_library.py` + test |
| W1-F | P0 | `cli.py`, `scripts/gcp_sdar_preflight.sh` | cli run-spec test |
| W1-G | P1 | `primitive_cache.py` | resume test |

**Wave 2 — integration wiring, dependency-ordered, serialized on shared files (Opus-driven).** `run.py` is the convergence point (P2/P3/P4/P5) and is edited sequentially. P3 (the control-loop in `forced_iteration.py` + `run.py`) is owned by Opus directly.

| Step | Phase | Files (serialized where shared) | Depends on |
|---|---|---|---|
| W2-1 | P0 | `primitives.py` (zero-metrics wire @6465, fail-honest) | W1-A |
| W2-2 | P1 | `binding.py` (ledger record-only @~640) | W1-B |
| W2-3 | P2 | `run.py` (validator build + offline invoke), `report.py` (`validation` field + consume verdict) | W1-C, W1-D, W2-2 |
| W2-4 | P3 | `forced_iteration.py`, `run.py`, `root_progress.py`, `leaf_triage.py`, `primitives.py`(verify hook), `baseline_implementation.py`(inject) | W2-1, W2-3 |
| W2-5 | P4 | `run.py` (admission hooks ×3 finalize paths), `baseline_implementation.py` (recipe inject) | W1-E, W2-3 |
| W2-6 | P5 | `root_progress.py`, `primitive_cache.py`, docs | W2-4, W1-G |

**Wave 3 — full hermetic suite + lint + type-check + integration dry-run, then Codex review.**

---

## P0 — cheap + fail-honest (no repair loop yet)

### Task P0.1 (W1-A): `zero_metrics_detection.py` — the flat+nested zero/constant veto

**Files:**
- Create: `backend/agents/rlm/zero_metrics_detection.py`
- Test: `tests/rlm/test_zero_metrics_detection.py`
- Reference (mirror its shape exactly): `backend/agents/rlm/stub_detection.py`

**Interfaces — Produces:**
```python
def zero_metrics_guard_enabled() -> bool: ...
    # gate on OPENRESEARCH_ZERO_METRICS_GUARD in {"1","true","yes","on"}; default OFF.

def normalize_metric_values(metrics: object) -> list[float]:
    # Flatten BOTH shapes to the list of RESULT-CLAIMING numeric leaf values:
    #  - flat scalar:  {"loss":0.0,"return":31.1,"return_std":4.4}  -> [0.0, 31.1, 4.4]
    #  - nested:       {"status":..,"per_model":{m:{e:{b:{"metric":0.086,"reward_mean":..}}}}, "scope":..}
    # Rules: recurse dicts/lists; collect only int/float (and numeric-coercible str) leaf values;
    # EXCLUDE structural/count keys ("status","scope","cell_id","n","*_n","count","steps","steps_run",
    # "epochs","seed","batch*","wall_time_s","elapsed*","len","retries") and any key whose name marks it
    # as a denominator/size rather than a result. Fail-soft: non-dict / error -> [].

def looks_like_zero_metrics(metrics: object) -> bool:
    # True iff normalize_metric_values(metrics) is non-empty AND
    #   (all values == 0.0) OR (all values bit-identical to each other, i.e. constant across cells).
    # This is the metric-shape signal ONLY; the provenance/GPU discriminators are applied by the caller
    # (P0.2 wire) so this stays pure + unit-testable. Fail-soft -> False.

def zero_metrics_repair_message(metrics: object) -> str:
    # Actionable, names the offending pattern (all-zero vs constant) and the first ≤6 result keys.
    # Mirrors stub_detection.stub_repair_message tone: tells the implementer the training/reward/eval
    # is not wired to real model outputs and must be reconnected.
```

**What NOT to touch:** no edits to `primitives.py` (that is the W2-1 wire). Pure module.

- [ ] **Step 1 — fixtures.** Copy a real all-zero metrics file into the test as a literal fixture. Use the SDAR v6 shape: `{"loss":0.0,"l_grpo":0.0,"mean_reward":0.0,"accuracy_avg":0.0,"f1_avg":0.0,"teacher_gap_mean":0.0,"gate_activation_ratio":0.0}` (all result keys 0.0). Also a flat legit-mixed fixture `{"loss":1.2,"return":31.1}` and a nested aggregated fixture (one `per_model[m][e][b]` cell with `{"status":"ok","metric":0.086,"reward_mean":0.086}`).
- [ ] **Step 2 — failing tests.** Assert: v6 all-zero → `looks_like_zero_metrics` True; legit-mixed → False; nested-with-real-0.086 → False; nested all-cells-`metric:0.0` → True; constant-across-cells (`metric:0.5` in every cell) → True; `{}` → False; `normalize_metric_values` excludes `*_n`/`steps_run`/`wall_time_s`/`status`/`scope`/`cell_id` (e.g. the all-zero-with-nonzero-count file `{"x_pct_correct":0.0,"x_n":50}` → True, because `x_n` is excluded and `x_pct_correct` is the only result key and it's 0).
- [ ] **Step 3 — run, verify fail.** `pytest tests/rlm/test_zero_metrics_detection.py -v` → FAIL (module missing).
- [ ] **Step 4 — implement** the module to the contract.
- [ ] **Step 5 — run, verify pass.** All green. Run `uvx ruff@0.15.16 check backend/agents/rlm/zero_metrics_detection.py`.
- [ ] **Step 6 — commit** (`git add` the new module + test; descriptive message).

**Acceptance:** the fixture suite in spec §11.1 row "zero/constant-metrics veto" — v6 flat-all-zero fires; flat-0-with-provenance does NOT fire *at the wire* (the provenance discriminator is W2-1, but the module test proves the shape signal); converged-near-0 with real values does not fire.

### Task P0.2 (W2-1): wire the veto into `run_experiment` — fail-honest, report-only

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (sibling block at ~6465, immediately after the `STUB_METRICS_GUARD` block ~6442-6464, before the finalize-on-timeout block)
- Test: `tests/rlm/test_zero_metrics_guard_integration.py`

**Interfaces — Consumes:** `zero_metrics_detection.{zero_metrics_guard_enabled,looks_like_zero_metrics,zero_metrics_repair_message}` (P0.1); `gpu_cell_runner.metrics_claim_gpu_training` (existing); a provenance presence check.

**Wire logic (the three-part discriminator from spec §6.1):** inside `if result.get("success"):`, in a `try/except` fail-soft block mirroring the stub guard:
```
if zero_metrics_guard_enabled() and looks_like_zero_metrics(result.get("metrics")):
    gpu_claim = metrics_claim_gpu_training(result.get("metrics"))
    prov_present = _provenance_links_metric(ctx, code_path)   # see below
    if gpu_claim and not prov_present:
        # P0: FAIL-HONEST report-only degrade (NOT a repair expectation yet — that is P3)
        result = {**result, "success": False,
                  "failure_class": "fabrication_suspected",
                  "error": zero_metrics_repair_message(result.get("metrics"))}
        emit run_warning code="fabrication_suspected"
    # else: all-zero WITH provenance (legit failing baseline) OR no GPU claim -> NOT vetoed
```
- `_provenance_links_metric(ctx, code_path)`: True iff a `provenance.json` exists under `code_path` (or `code_path/outputs/**`) — i.e. the harness/agent emitted a manifest linking the metric to a real output. Fail-soft: missing/unparseable → treat as ABSENT (conservative, the spec's risk #4 — but only matters when GPU-claim is also present). If `provenance.json` is absent on a route that legitimately has no manifest, the veto would fire on a GPU-claim+all-zero result, which is exactly the v6 case we want to catch; a legit failing baseline that ran on GPU should still emit provenance, so this is acceptable for P0 fail-honest (P2 escalates ambiguous cases to the validator).

**What NOT to touch:** do not change the stub-guard block; do not alter the monolithic/cells routing.

- [ ] **Step 1 — failing integration test.** Build a tmp run dir with `code/metrics.json` all-zero + a metrics dict carrying a `device:"cuda"` claim and NO `provenance.json`; call the guard helper path with `OPENRESEARCH_ZERO_METRICS_GUARD=1`; assert `result["success"] is False` and `failure_class=="fabrication_suspected"`. Second test: identical but WITH a `provenance.json` → `success` stays True (not vetoed). Third: flag unset → never vetoed (byte-identical).
- [ ] **Step 2-5:** run-fail → wire → run-pass → ruff → commit.

**Acceptance:** spec §13 P0 "would have flagged v6"; spec §11.1 false-positive guard (flat-0 with provenance does not fire).

### Task P0.3 (W1-F): `--run-spec <path.json>` config-file run spec

**Files:**
- Modify: `backend/cli.py` (the `reproduce` subparser + the early env-sink load ~1566-1697, explicit flags overriding)
- Modify: `scripts/gcp_sdar_preflight.sh` (replace the 12-var launch whitelist + staged guidance file with one `scp`'d spec; the `launch` path ~line 304)
- Test: `tests/test_cli_run_spec.py`

**Interfaces — Produces:** `--run-spec PATH` on `reproduce`. Loader reads a JSON object of `{"OPENRESEARCH_*": "value", ...}` (+ optional `models`, `baseline_extra_guidance` multi-line) into `os.environ` **before** flag resolution; an explicit CLI flag overrides a spec value (same precedence the env sink already has).

- [ ] **Step 1 — failing test.** Write a tmp spec JSON with `{"OPENRESEARCH_ZERO_METRICS_GUARD":"1","OPENRESEARCH_STUB_METRICS_GUARD":"1"}`; invoke the loader; assert both land in `os.environ`; assert an explicit `--max-usd 5` still wins over a spec `OPENRESEARCH_MAX_USD`. Assert a multi-line `baseline_extra_guidance` lands in `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` intact (the env-word-split footgun the spec §10 calls out).
- [ ] **Step 2-5:** run-fail → implement loader + script change → run-pass → commit. For the shell script, the acceptance is: `launch` `scp`s one `run_spec.json` and the remote `reproduce` consumes it via `--run-spec`; document the new shape in the script header comment. (No paid run — the script change is verified by shellcheck + a dry `--help`/grep assertion, not by launching a VM.)

**Acceptance:** spec §10; the GCP launcher ships one spec instead of the whitelist; multi-line guidance survives.

---

## P1 — evidence ledger (record/provenance) + verify resume

### Task P1.1 (W1-B): `lifecycle_ledger.py` — append-only, redacted, record-only

**Files:**
- Create: `backend/agents/rlm/lifecycle_ledger.py`
- Test: `tests/rlm/test_lifecycle_ledger.py`
- Reference patterns: `primitive_cache.put` (append-only JSONL fail-soft), `lesson_distiller` (atomic temp+replace — for the per-primitive projection registry only if needed).

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class LedgerRecord:
    primitive: str; seq: int; inputs_projection: dict; outputs_pointer: dict
    evidence_keys: list[str]; outcome: str; iteration: int   # outcome in {ok,failed,raised,timeout}

def lifecycle_ledger_enabled() -> bool: ...   # OPENRESEARCH_LIFECYCLE_LEDGER, default OFF

def project_inputs(primitive: str, kwargs: dict) -> dict:
    # per-primitive REDACTED projection — bounded, non-corpus fields ONLY.
    # plan_reproduction -> {"section_ids":[...], "hparam_keys":[...]}  (NEVER paper prose)
    # implement_baseline -> {"plan_present":bool,"repair_context_present":bool,"sandbox_mode":str,"gpu_mode":str}
    # run_experiment -> {"env_id":str,"code_present":bool}
    # default -> {} (unknown primitives project nothing)

def append_record(project_dir: Path, record: LedgerRecord) -> None:
    # append one JSON line to runs/<id>/rlm_state/lifecycle/ledger.jsonl; fail-soft; the single-line
    # append IS the atomic unit. Creates the dir. Never raises.

def read_records(project_dir: Path) -> list[LedgerRecord]: ...   # fail-soft -> []
```

**Redaction guard (spec §4.3):** a sentinel test writes a record for a `plan_reproduction` whose raw inputs contain a known canary string (`"CANARY_PAPER_TEXT_DO_NOT_LEAK"`); asserts the canary appears in NO file under `rlm_state/`. The projection functions must only emit the bounded fields above.

**What NOT to touch:** no edits to `binding.py` (that is W2-2). Pure module.

- [ ] **Step 1 — failing tests:** `append_record`+`read_records` round-trips a record; the sentinel canary test; `project_inputs("plan_reproduction", {...paper prose...})` returns only section ids + hparam keys; `lifecycle_ledger_enabled()` False by default; outcome field accepts only the 4 values.
- [ ] **Step 2-5:** run-fail → implement → run-pass → ruff → commit.

**Acceptance:** spec §11.1 `LedgerRecord` row; sentinel canary absent from all ledger files.

### Task P1.2 (W2-2): wire record-only ledger into `wrap_primitive`

**Files:**
- Modify: `backend/agents/rlm/binding.py` (after the three-way `_ledger(...)` cost-stamp ~line 640; derive a single `_outcome` local and reuse it)
- Test: `tests/rlm/test_binding_lifecycle_ledger.py`

**Interfaces — Consumes:** `lifecycle_ledger.{lifecycle_ledger_enabled,project_inputs,append_record,LedgerRecord}`.

**Wire logic (spec §4.2 — never record `ok` on timeout/raised):**
- The `except Exception` path re-raises (line 615) so the post-validation region is structurally unreachable for `raised` — no `ok` record can be written for a raised primitive. ✓
- The timeout envelope `{"outcome":"retryable","error":"primitive_hung"}` has `error` truthy → `failed=True` → it is NOT in the `ok` branch. ✓
- Compute the same `_outcome` string the cost-ledger derives (`ok`/`failed`/`partial_timeout`); write the lifecycle record in ALL non-raised cases, with `outcome` set to that string (so `failed`/`partial_timeout`/`timeout` are recorded as themselves, and only a genuine `ok` is recorded `ok`). Map the retryable-timeout envelope to `outcome="timeout"`.
- Place the append AFTER `_emit_supplemental(...)` returns (so `evidence_keys` can be captured from the just-emitted state; re-derive via `evidence_key.evidence_key(...)` if needed). Gate on `lifecycle_ledger_enabled()`. Fail-soft.

- [ ] **Step 1 — failing tests:** a fake primitive returning `{"success":True,"metrics":{...}}` writes one `ok` ledger record (flag on); a primitive returning `{"success":False,"error":"x"}` writes a `failed` record, never `ok`; the timeout envelope writes a `timeout` record, never `ok`; flag off → no `lifecycle/` dir created (byte-identical). Use a stubbed `ctx`/`wrap_primitive` harness — keep hermetic.
- [ ] **Step 2-5:** run-fail → wire → run-pass → ruff → commit.

**Acceptance:** spec §11.1 "ledger never records `ok` on timeout/raised".

### Task P1.3 (W1-G): verify `primitive_cache` cross-restart resume (spec §5.4)

**Files:**
- Modify (only if a gap is found): `backend/agents/rlm/primitive_cache.py`
- Test: `tests/rlm/test_primitive_cache_resume.py`

**Goal:** prove (not assume) that on a same-`--project-id` relaunch, a fresh root's `implement_baseline`/`plan_reproduction` cache-HITS, so spot-preemption rework is bounded to the root's re-reasoning, not the expensive primitive bodies.

- [ ] **Step 1 — failing test:** write a record via `put(project_dir, "implement_baseline", payload=P, result=R)`; construct a NEW in-process context pointed at the same `project_dir`; assert `maybe_get(project_dir, "implement_baseline", payload=P) == R` (key is `version:primitive:sha256[:32](payload)`, no run-id component → stable). Second test: a payload differing only in a volatile field that the implement_baseline payload-builder is supposed to EXCLUDE (e.g. `remaining_s`) still hits — i.e. assert the payload excludes volatile fields (read the builder; if it does NOT exclude them, that is the gap → fix the builder to drop them and document). Third: `run_experiment` + `build_environment` are NOT cacheable (`maybe_get` returns None).
- [ ] **Step 2 — run, observe.** If the resume is already stable: the test passes after writing it (verification task). If a volatile field leaks into the key: fix the payload builder (smallest change) + document in the test.
- [ ] **Step 3 — commit** the test (+ any fix). Record the finding (resume stable / gap fixed) in the P1 section of the plan + a one-line `learn.md` rule if a gap was fixed.

**Acceptance:** spec §15 risk #3 — resume confirmed enabled + stable for SDAR, or the gap is fixed.

---

## P2 — deterministic floor consolidation + validator as OFFLINE report-stamping

### Task P2.1 (W1-D): `validator` role + `RoleSpec.family` + fail-closed transport

**Files:**
- Modify: `backend/agents/rlm/role_models.py` (ROLES@91, _SUBROLES@94, RoleSpec@146-188 +`family`, `_classify_model_family` new, RoleSelection@191 +`validator`, `resolve_role_models`@340, `_resolve_subrole`@381, `parse_model_spec`@255-287)
- Modify: `backend/agents/rlm/grader_transport.py` (new `build_validator_client` fail-closed sibling, after `build_grader_client`)
- Modify: `backend/agents/rlm/context.py` (add `validator_client: Any = None` field)
- Test: `tests/agents/rlm/test_role_models_validator.py`, `tests/rlm/test_validator_transport.py`

**Interfaces — Produces:**
```python
# role_models.py
def _classify_model_family(provider: str, model: str | None) -> str | None:
    # lineage, NOT transport. claude/claude-oauth/anthropic-oauth/sonnet/opus/haiku -> "claude"
    #   (sub-family "sonnet"/"opus"/"haiku" retained for stamping but family separation keys on "claude")
    # azure/gpt-4o-azure/openai/gpt-5/gpt-4o/o4-mini -> "gpt"
    # azure-foundry+grok/grok-4.3 -> "grok"; azure-foundry+kimi -> "kimi"; qwen* -> "qwen"
    # The KEY invariant: family("azure")=="gpt" != family("claude-oauth")=="claude".

def separation_strength(executor: RoleSpec | None, validator: RoleSpec | None) -> str:
    # the THREE-TIER ladder (replaces the binary same-family check):
    #   "independent"  -> executor.family != validator.family            (e.g. oauth-Sonnet × azure-gpt)
    #   "weak"         -> same family, DIFFERENT model/deployment         (e.g. azure gpt-4o × azure gpt-4.1)
    #   "degraded"     -> same family AND same model (seed-only)          (e.g. azure gpt-4o × azure gpt-4o)
    #   "unavailable"  -> validator is None
    # For azure×azure, "model" is the resolved deployment, so OPENRESEARCH_VALIDATOR_MODEL=<deploymentB>
    # vs AZURE_OPENAI_DEPLOYMENT=<deploymentA> yields "weak", which is SUPPORTED (the operator's ask).
@dataclass(frozen=True)
class RoleSpec:  # + family: str | None = None
ROLES = (..., "validator"); _SUBROLES = frozenset({..., "validator"})
class RoleSelection:  # + validator: RoleSpec | None; stamp() gains "validator"

# grader_transport.py
def build_validator_client(*, fallback_client, fallback_label="") -> tuple[Any, str]:
    # reads OPENRESEARCH_VALIDATOR_BACKEND / OPENRESEARCH_VALIDATOR_MODEL.
    # backend unset -> return (fallback_client, fallback_label) (validator role not overridden).
    # backend set -> build via build_transport_client(backend, model=OPENRESEARCH_VALIDATOR_MODEL, ...);
    #   if it returns the fallback (= a credential was missing / unknown backend / construction error)
    #   RAISE ValueError. FAIL-CLOSED — never silently fall through to the executor-family client.
    #   Supports backend in {azure, oauth, anthropic, openai, azure-foundry, grok}.
    # AZURE DEPLOYMENT OVERRIDE: for backend=azure the `model` arg MUST override AZURE_OPENAI_DEPLOYMENT
    #   so executor(azure, deploymentA) and validator(azure, deploymentB) hit DIFFERENT deployments.
    #   Verify the build_transport_client azure branch honors a passed `model` as the deployment; if it
    #   pins AZURE_OPENAI_DEPLOYMENT regardless, that is the gap to fix here (smallest change: pass the
    #   deployment override through to AzureOpenAILlmClient).
```

**Acceptance tests (load-bearing — both panels):**
```python
def test_validator_independent_oauth_exec_azure_val():       # tier: independent
    sel = resolve_role_models("claude-oauth", cli_models="executor=sonnet,validator=gpt-4o-azure")
    assert sel.executor.family == "claude" and sel.validator.family == "gpt"
    assert separation_strength(sel.executor, sel.validator) == "independent"

def test_validator_weak_azure_exec_azure_val_diff_model():   # tier: weak (the operator's ask)
    sel = resolve_role_models("azure-gpt-4o", cli_models="executor=gpt-4o-azure,validator=azure")
    # validator deployment differs from executor deployment via OPENRESEARCH_VALIDATOR_MODEL
    # (set the two RoleSpec.model values to distinct deployments in the test)
    assert sel.executor.family == sel.validator.family == "gpt"
    assert sel.executor.model != sel.validator.model
    assert separation_strength(sel.executor, sel.validator) == "weak"      # SUPPORTED, not blocked

def test_validator_degraded_same_model():                    # tier: degraded (loud warning downstream)
    # executor and validator resolve to the SAME azure deployment
    assert separation_strength(spec_a, spec_a) == "degraded"

def test_validator_build_fail_closed_missing_creds(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    # no AZURE_OPENAI_* set -> build_transport_client returns fallback -> build_validator_client RAISES
    with pytest.raises(ValueError):
        build_validator_client(fallback_client=object(), fallback_label="exec")

def test_azure_validator_uses_distinct_deployment(monkeypatch):
    # AZURE_OPENAI_DEPLOYMENT=execA, OPENRESEARCH_VALIDATOR_MODEL=valB ->
    # the constructed validator client targets deployment valB, NOT execA.
```

**What NOT to touch:** do not route the executor through this (executor keeps its legacy graceful-fallback path); do not touch `run.py` (W2-3).

- [ ] **Step 1 — failing tests** (the five above + family classifier unit tests for every token in the vocab + `RoleSelection.stamp()["validator"]` present + de-collapse opus≠sonnet still holds + grader/verifier unchanged).
- [ ] **Step 2-5:** run-fail → implement → run-pass → ruff → commit.

**Acceptance:** spec §7.2/§7.4/§11.1 `RoleSpec.family` row, generalized to the three-tier ladder; BOTH the cross-family (`independent`) and same-provider-different-model (`weak`) panels construct; the azure validator hits a distinct deployment; fail-closed on missing creds.

### Task P2.2 (W1-C): `external_validator.py` — typed-predicate panel + min-aggregation + verdict store

**Files:**
- Create: `backend/agents/rlm/external_validator.py`
- Test: `tests/rlm/test_external_validator.py`
- Reuse (import, do not edit): `grader_transport.sample_completions`; `leaf_scorer._result_leaf_substantiated` / evidence-gather; `evidence_gate.leaf_claims_measured_result`; `zero_metrics_detection.normalize_metric_values`; `gpu_cell_runner.metrics_claim_gpu_training`; `provenance` presence.

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class PredicateVerdict:
    predicate: str        # provenance_present | not_all_constant | gpu_claim_plausible | rerun_agrees
    metric_ref: str       # the cited metric/leaf
    violated: bool        # machine-verified, NOT the LLM's opinion
    detail: str
@dataclass(frozen=True)
class ValidatorVerdict:
    status: str           # clean | vetoed | unavailable
    veto_set: list[str]   # metric_refs vetoed (min-aggregation: ANY violated predicate -> veto)
    predicates: list[PredicateVerdict]
    panel_models: list[str]
    separation: str       # independent | weak | degraded | unavailable  (from role_models.separation_strength)
    evidence_fingerprint: str

def external_validator_enabled() -> bool: ...   # OPENRESEARCH_EXTERNAL_VALIDATOR, default OFF
def validator_panel_n() -> int: ...             # OPENRESEARCH_VALIDATOR_PANEL_N, small default (2)

# The four MACHINE-CHECKS (deterministic; the LLM only chooses which/where to point):
def check_provenance_present(metrics, project_dir) -> bool: ...
def check_not_all_constant(metrics) -> bool: ...      # uses normalize_metric_values; True == healthy (not constant)
def check_gpu_claim_plausible(metrics, evidence) -> bool: ...
def check_rerun_agrees(...) -> bool: ...               # P2: stub returns None/skipped; real impl in P3 re-run

def run_validation_panel(*, validator_client, panel_models, metrics, project_dir, leaf_records,
                          separation: str) -> ValidatorVerdict:
    # 1) sample_completions(validator_client, ..., n=panel_n) with an ADVERSARIAL "find the fakery"
    #    prompt that must emit a structured list of {predicate, metric_ref} suspicions.
    # 2) for each suspicion the HARNESS machine-checks the named predicate against the artifact.
    # 3) min-aggregation: a metric is vetoed iff ANY panelist's predicate machine-verifies as violated.
    #    (no majority vote.) Returns ValidatorVerdict.
    # If validator_client is None/unavailable -> status="unavailable" (Tier-1 floor is the backstop).

def persist_verdict(project_dir: Path, verdict: ValidatorVerdict) -> None:
    # atomic temp+os.replace -> rlm_state/validation_verdict.json, keyed by evidence_fingerprint.
def load_verdict(project_dir: Path, *, expect_fingerprint: str | None = None) -> ValidatorVerdict | None:
    # returns None if absent OR (expect_fingerprint given AND mismatched) — a stale verdict is ignored.
```

**Min-aggregation is REMEDIAL (spec §7.7):** the veto set feeds the P3 repair loop, not an immediate fail. In P2 the panel is invoked OFFLINE (report-stamping only) — see P2.3.

**What NOT to touch:** no `run.py` edits; no continue-policy dependency (P2 is offline). For unit tests, STUB `sample_completions` (return canned suspicion JSON) and feed canned metrics/provenance — keep hermetic, no network.

- [ ] **Step 1 — failing tests:** stubbed panel returns a `provenance_present` suspicion on the v6 all-zero+no-provenance fixture → machine-check confirms violated → `veto_set` non-empty, `status=="vetoed"`; same fixture WITH provenance → not violated → `status=="clean"`; `validator_client=None` → `status=="unavailable"`; `persist_verdict`/`load_verdict` round-trip; `load_verdict(expect_fingerprint="X")` returns None on mismatch (stale ignored); min-aggregation (one panelist violated, one clean → vetoed).
- [ ] **Step 2-5:** run-fail → implement → run-pass → ruff → commit.

**Acceptance:** spec §7.3 typed-predicate machine-check (not citation-existence); §7.6 fingerprint-keyed store; §11.1 `ValidatorVerdict` row.

### Task P2.3 (W2-3): wire validator OFFLINE (report-stamping) + `report.validation`

**Files:**
- Modify: `backend/agents/rlm/run.py` (role wiring ~2188-2283: add `validator_model_setting`, build `validator_client` via `build_validator_client` fail-closed, thread `ctx.validator_client`; invoke the panel at the `FINAL_VAR`-attempt seam, persist the verdict — but in P2 do NOT let it drive continue-policy)
- Modify: `backend/agents/rlm/report.py` (add typed `validation: dict = Field(default_factory=dict)` to `RLMFinalReport` ~line 119; in `write_final_report_rlm` after the `rubric_evaluation.json` merge (~after 1778) read `validation_verdict.json` fail-soft, stamp `report.validation`, and — fingerprint-matched only — optionally cap verdict if the panel says fabricated)
- Test: `tests/rlm/test_report_validation_stamp.py`

**Interfaces — Consumes:** `external_validator.{run_validation_panel,persist_verdict,load_verdict}`, `grader_transport.build_validator_client`, `role_models.RoleSelection.validator`.

**Wire logic:**
- Build `validator_client` fail-closed; if `OPENRESEARCH_VALIDATOR_BACKEND` unset, validator is inherited/None → panel `unavailable` (Tier-1 floor backs it). Compute `separation = role_models.separation_strength(executor, validator)`: `independent`/`weak` run normally (a `weak` panel emits a one-line `validator_separation_weak` notice, NOT a blocker — same-provider-different-model is supported); `degraded` (same model) emits the loud `validator_separation_degraded` run_warning and downstream treats the LLM-suspicion portion as non-independent (the machine-checked veto still stands).
- `write_final_report_rlm` consumes ONLY a verdict whose `evidence_fingerprint` matches the shipped evidence (stale verdict ignored). `final_report.json.validation` stamps panel model(s), per-leaf predicate verdicts, the min-aggregated veto set, separation-degraded.
- P2 = offline: the verdict is recorded + stamped; it does NOT yet refuse `FINAL_VAR` (that upgrade is P3).

- [ ] **Step 1 — failing tests:** a planted `validation_verdict.json` with matching fingerprint → `final_report.json.validation` carries the veto set; a fingerprint-MISMATCHED verdict → `validation` empty (ignored); `OPENRESEARCH_EXTERNAL_VALIDATOR` unset → `validation` empty + byte-identical report; `validator` set but azure creds missing → run aborts at build (fail-closed) OR (if validator unset) degrades — assert the loud warning.
- [ ] **Step 2-5:** run-fail → wire → run-pass → ruff → commit. **Opus reviews the run.py diff line-by-line.**

**Acceptance:** spec §7.1/§7.6; §11.1 `ValidatorVerdict` `_finalize` ignores fingerprint-mismatch; offline stamping covers all 3 finalize paths via the `write_final_report_rlm` chokepoint.

---

## P3 — unified continue-policy + the fix-first loop + minimal degenerate-recovery

> **Opus-owned.** This is the riskiest change — `forced_iteration.py` gates the `FINAL_VAR` of every run, and `run.py` orchestrates the repair loop. Every diff reviewed line-by-line. The migration-safety contract: `OPENRESEARCH_EXTERNAL_VALIDATOR` OFF ⇒ byte-identical to the hardened 2026-06-17 policy.

### Task P3.1: distinct repair-refusal class in `ForcedIterationPolicy`

**Files:**
- Modify: `backend/agents/rlm/forced_iteration.py`
- Test: `tests/rlm/test_forced_iteration_repair_class.py`

**Interfaces — Produces (extends the existing policy):**
- New fields: `_repair_max_iterations: int` (from `OPENRESEARCH_REPAIR_MAX_ITERATIONS`, default 4), `_last_repair_fingerprint: str | None`, `evidence_fingerprint: Callable[[], str] | None`.
- New terminal class `"repair_exhausted"` added to `_TERMINAL_FAILURE_CLASSES`.
- A repair refusal's "progress" is keyed to the **evidence fingerprint changing** between attempts (NOT a root no-progress signature). The repair-refusal signature is EXCLUDED from the `register_refusal` degenerate-loop counter (it never increments `_noprogress_refusals`).
- Stop condition: same `failure_class` AND unchanged evidence fingerprint across attempts, OR `_repair_iter_count >= _repair_max_iterations` → stop with `failure_class="repair_exhausted"` (honest), never the confusing `root_degenerate_loop`.

**Key existing machinery to extend (from recon):** `record_repair_attempt(failure_class)` (already increments `_repair_iter_count` + stamps `_last_repair_failure_class`), `_build_repair_refusal(min_repair)` (already exists), `record_state_change()` (resets degenerate counter — a real repair re-runs `implement_baseline`/`run_experiment` which already calls this, so the healthy fix-first path is byte-safe).

- [ ] **Step 1 — failing tests:** a repair refusal does NOT increment `_noprogress_refusals` (assert the degenerate counter stays 0 across N repair refusals while evidence fingerprint CHANGES each time — progress); a stuck repair (same failure_class + unchanged fingerprint) after `_repair_max_iterations` → `note_terminal_failure("repair_exhausted")` and the next `FINAL_VAR` is accepted; a healthy run with the validator flag OFF is byte-identical (the new fields are inert).
- [ ] **Step 2-5:** run-fail → implement → run-pass → ruff → commit.

**Acceptance:** spec §5.3/§11.1 `RepairState` row — a floor/validator refusal never increments the `root_degenerate_loop` counter; stuck-repair ⇒ `repair_exhausted`.

### Task P3.2: the unified error → repair → re-validate loop (fix-first, fail-honest)

**Files:**
- Modify: `backend/agents/rlm/run.py` (the tool-wrapper repair accounting ~1111-1208; the `FINAL_VAR`-attempt seam; upgrade the P0 zero-metrics veto + P2 validator from fail-honest → fix-first)
- Modify: `backend/agents/rlm/leaf_triage.py` (accept validator-sourced + zero-metrics-sourced structured directives into the SAME plan-list shape `{leaf_id, score, repair_class, cost, directive, justification}`)
- Modify: `backend/agents/rlm/primitives.py` (`verify_against_rubric` ~7274-7282 — also fold validator/zero-metrics directives into the persisted `leaf_triage.json`)
- Modify: `backend/agents/baseline_implementation.py` (the directive is already injected via `guidance_block` hook 6.7 @2487-2492 — confirm the new directives flow through)
- Test: `tests/rlm/test_repair_loop.py`, `tests/rlm/test_repair_loop_integration.py`

**Pipeline (spec §8):** classify `failure_class` → build a cited, structured repair directive → inject via the `leaf_triage → next-implementer-prompt` channel → re-implement (cache-bust) → re-run → re-validate → the continue-policy decides accept / repeat / honest-stop.
- **Repairable classes enter the loop:** existing `_RUN_EXPERIMENT_REPAIRABLE_FAILURES` + new `fabrication_suspected` + validator predicate vetoes.
- **Terminal classes fail fast:** `capacity_exhausted`, `oom_shrink_exhausted`, dead-credit/auth, `root_degenerate_loop`, `repair_exhausted`.
- **Fix-first never relaxes the bar** — "fix" = real re-implementation so the evidence becomes real; the validator/floor standard is fixed, only the code changes.
- **Honest failure is the floor:** budget/wall-clock/no-progress exhausted with evidence still fake ⇒ ship `failed`/`degraded` with the cited unfixed reason (extends `_hard_stop_with_report`).

- [ ] **Step 1 — failing integration test (`--sandbox local` dry-run, hermetic — no GPU):** plant a flat-all-zero, no-provenance cell; assert the §8 loop (a) refuses the first `FINAL_VAR`, (b) injects a `fabrication_suspected` cited directive through `leaf_triage.json`, (c) after `_repair_max_iterations` with unchanged evidence ships `degraded`/`failed` with the cited reason — within budget, never a shipped fake, never a silent pass.
- [ ] **Step 2 — failing unit tests:** validator `vetoed` verdict (flag ON) refuses `FINAL_VAR` and routes a directive; same verdict with flag OFF is inert; a successful repair (evidence fingerprint changes, predicate now clean) is accepted.
- [ ] **Step 3-6:** run-fail → implement → run-pass → ruff → commit. **Opus reviews every diff.**

**Acceptance:** spec §8/§14 integration test "repairs-then-honest-fails within budget".

### Task P3.3: minimal recovery-aware degenerate nudge (cited, points at ledger)

**Files:**
- Modify: `backend/agents/rlm/root_progress.py` (the stage directive now cites the ledger record path, e.g. *"plan at `rlm_state/lifecycle/ledger.jsonl` seq N; call `implement_baseline(plan=…)`"*)
- Modify: `backend/agents/rlm/run.py` (`_make_degenerate_loop_callback` ~980-1108 — on a no-progress refusal emit the stage-specific cited nudge; the harness NEVER synthesizes the root's calls — nudge only)
- Test: `tests/rlm/test_degenerate_nudge.py`

- [ ] **Step 1 — failing test:** with the ledger flag ON and a `need_baseline` stage, the nudge message names the missing primitive AND cites the ledger path; the harness issues a directive, not a synthesized primitive call (assert no primitive is auto-executed); transport-agnostic (same for oauth + azure roots).
- [ ] **Step 2-5:** run-fail → implement → run-pass → commit.

**Acceptance:** spec §5.1 detect→nudge→(rope)→clean-abort, no harness synthesis; minimal integration lands in P3 (not P5) so no veto's recovery is absent.

---

## P4 — Tier B cross-run positive recipes

### Task P4.1 (W1-E): `recipe_library.py` — principle-level, gated on Tier-1 + validator (NOT grade)

**Files:**
- Create: `backend/agents/rlm/recipe_library.py`
- Test: `tests/rlm/test_recipe_library.py`
- Reference (mirror the pattern, NOT the keying): `lesson_distiller.mine_lessons` (atomic temp+replace store, `_finalize` hook, `guidance_block` injection).

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class Recipe:
    problem_sig: dict      # model/dataset/task axes
    solution_sig: dict     # key hyperparameters + cells.json shape + ≤200-char technique summary + code-path pointer
    evidence_key: str
    paper_class: str

def positive_recipes_enabled() -> bool: ...   # OPENRESEARCH_POSITIVE_RECIPES, default OFF
def derive_paper_class(*, arxiv_id, paper_hints, rubric) -> str:
    # NO paper_class exists on ctx — derive from PAPER_HINTS class / rubric shape. Keyed by paper-CLASS
    # (transfer), not arxiv_id. Falls back to a coarse rubric-shape bucket.
def admit_recipe(project_dir, runs_root, *, report, validator_verdict, paper_class) -> None:
    # ADMISSION GATE (red line) — admit ONLY if ALL hold:
    #   (1) report-level evidence_gate passed
    #   (2) a success=True run_experiment ledger row exists
    #   (3) validator verdict is clean (or unavailable+floor-passed per policy)
    #   (4) deterministic meets_target (measured metric vs paper's claimed target) holds
    # NEVER the champion median_score, NEVER agent prose. Writes runs/_recipes/<paper_class>.json
    # (atomic temp+replace), capped, novelty-deduped, staleness-retired.
def recipe_guidance_block(runs_root, paper_class, *, max_chars=1200) -> str:
    # inject top-1 by problem-sig similarity, hard-capped ≤1-2. "" when disabled/empty.
```

**Red-line static-import guard (spec §11.1):** a test asserts `recipe_library` source contains NO read of grade fields (`overall_score`, `median_score`, `compute_adjusted_score`, `rubric_score`) except inside a `# copy-into-report` allowlisted line — i.e. admission reads only Tier-1 + validator inputs.

**What NOT to touch:** no `run.py`/`baseline_implementation.py` edits (W2-5).

- [ ] **Step 1 — failing tests:** a clean report (evidence_gate passed + success ledger row + validator clean + meets_target True) → `admit_recipe` writes a recipe; a high-GRADE but evidence_gate-FAILED report → NOT admitted (red line); a validator-`vetoed` report → NOT admitted; the static-import guard; `recipe_guidance_block` injects top-1 capped; `derive_paper_class` buckets SDAR via PAPER_HINTS; poison-proof (agent prose never enters the recipe body).
- [ ] **Step 2-5:** run-fail → implement → run-pass → ruff → commit.

**Acceptance:** spec §9.2/§11.1 `recipe_admission` row.

### Task P4.2 (W2-5): admission hooks (all 3 finalize paths) + injection

**Files:**
- Modify: `backend/agents/rlm/run.py` (`_finalize` ~3194; `_finalize_fatal_primitive_abort` ~1326; `_hard_stop_with_report` ~1451 — add `admit_recipe` to each, guarded on `ctx is not None`)
- Modify: `backend/agents/baseline_implementation.py` (append `recipe_guidance_block(...)` after the leaf-triage/lessons hooks ~2492-2550)
- Test: `tests/rlm/test_recipe_admission_hooks.py`

- [ ] **Step 1 — failing test:** a normal finalize on a clean run admits a recipe; the flag OFF admits nothing (byte-identical); a subsequent run of the same paper-class injects the recipe into the implementer guidance.
- [ ] **Step 2-5:** run-fail → wire → run-pass → ruff → commit. **Opus reviews the run.py diff.**

**Acceptance:** spec §9.2; admission survives a wall-clock kill (hook present in hard-stop path, unlike `mine_lessons`).

---

## P5 — recovery-aware detector polish + resume polish + docs

### Task P5.1: detector polish (code-side; the $2 experiment is operator-run)

**Files:** `backend/agents/rlm/root_progress.py`, `backend/agents/rlm/forced_iteration.py` (only if P0's operator experiment later indicates a tuning change — for now, code-side hardening: ensure the nudge cites the ledger across all stages; ensure the rope (generous-but-bounded) honors wall-clock).
- [ ] Harden + test the stage-directive coverage for all 5 stages; document the `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD=16` precondition experiment as an operator step in the runbook (do NOT run it).

### Task P5.2: docs + doc-update contract

**Files:** `CLAUDE.md` (feature-flag entries for every new flag), `docs/runbooks/2026-06-20-sdar-harness-refactor-and-external-validation-handoff.md` (mark superseded → point at this plan + the spec), a new `changelog.md`/`learn.md` entry, and the operator checklist for the deferred paid runs.
- [ ] Add a CLAUDE.md "Feature flags" block documenting: `OPENRESEARCH_ZERO_METRICS_GUARD`, `OPENRESEARCH_LIFECYCLE_LEDGER`, `OPENRESEARCH_EXTERNAL_VALIDATOR`, `OPENRESEARCH_VALIDATOR_BACKEND`/`_MODEL`/`_PANEL_N`/`_RERUN_ON_SUSPICION`, `OPENRESEARCH_REPAIR_MAX_ITERATIONS`, `OPENRESEARCH_POSITIVE_RECIPES` — each with default-OFF, the seam, and the {azure, oauth} canonical validator panel.
- [ ] Operator checklist: the precondition experiment + the ≥3 paired SDAR runs + the per-flag A/B before any default-flip.

---

## Wave 3 — verification + review

- [ ] Full hermetic suite: `.venv/bin/python -m pytest tests/ -n auto` green (socket-hermetic; `--reruns 2` for flakes).
- [ ] Lint: `uvx ruff@0.15.16 check .` clean on touched files.
- [ ] Type sanity on new modules (the repo is not strictly typed; at minimum no import/NameErrors).
- [ ] Integration dry-run: the P3.2 planted-fixture `--sandbox local` test (no GPU) demonstrates repair-then-honest-fail within budget.
- [ ] Default-OFF contract test (spec §11.2): with every new flag unset, the listed default-ON rails are the only active behavior; assert no new module changes a default path.
- [ ] **Codex adversarial review of the entire diff** (the user's explicit final gate).

## Self-review checklist (run before fan-out)

- Spec coverage: §4 ledger→P1; §5 reliability→P3; §6 floor→P0; §7 validator→P2; §8 repair loop→P3; §9 recipes→P4; §10 run-spec→P0; §11 invariants→every task's acceptance. ✓
- Red line enforced by the static-import guard (P4.1) + admission gate reading only Tier-1+validator. ✓
- {azure, oauth} cross-family panel is a first-class acceptance test (P2.1). ✓
- Every new flag default-OFF; byte-identical-to-baseline tests in P0.2, P1.2, P2.3, P3.1, P4.2. ✓
