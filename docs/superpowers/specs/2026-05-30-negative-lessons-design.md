# Per-paper Negative Lessons (MUSE-lite) — Design

**Date:** 2026-05-30
**Phase:** 9 of the RLM wedge-hardening-and-evolution plan
**Status:** Approved design, pre-implementation
**Flag:** `REPROLAB_NEGATIVE_LESSONS` (default **0** / off)
**Source paper (infra inspiration only):** MUSE (arXiv 2605.27366) — promotes/curates a skill bank. ReproLab ships only the cheap, high-value *negative* half: a per-paper file of failures-to-avoid, mined deterministically and injected into the next run's implementer prompt. No positive skill bank, no LLM curation.

---

## 1. Goal & non-goals

**Goal.** When an *agent-correctable* failure recurs across separate runs of the same paper, carry a structured, class-tagged lesson forward so the next run's `implement_baseline` agent avoids it. The same `failure_class` recurring across runs (BUG-NEW-042 prose-in-Dockerfile, a thin env spec, a missing per-model metric) is signal no intra-run mechanism can see — `repair_context` and `_prior_rubric_feedback_block` already cover *within*-run repair; this is the first **cross-run** learning surface (deliberately deferred out of Phase 8, which is intra-run only).

**Non-goals (v1, explicit):**
- No LLM distiller — lessons are mined deterministically from the *already-classified* `failure_class` records. MUSE's LLM curation half is rejected (costs an LLM call/run, produces free prose — the exact prompt-junk risk).
- No *positive* skill bank (MUSE's promote-good-skills half). Negative lessons only.
- No pytest-gating of skills before they ship (a later concern, not this phase).
- No cross-*paper* lessons — keyed strictly by `arxiv_id`. A run with no `arxiv_id` (PDF upload, unparsed id) does not participate (degrade silently).
- The lesson is **advisory guidance** to the implementer; it is never a correctness authority (the evidence gate + rubric remain the backstops — §7).

## 2. Architecture

One new module, two hooks, one new tiny classifier accessor, one flag. All reversible by unsetting the flag.

```
backend/agents/rlm/lesson_distiller.py     (NEW)   — owns runs/_lessons/<arxiv_id>.json
backend/agents/rlm/failure_classifier.py   (ADD)   — public suggested_fix_for_class(klass)
backend/agents/rlm/run.py                   (HOOK)  — mine_lessons() in the run finalize/finally path
backend/agents/baseline_implementation.py   (HOOK)  — _negative_lessons_block() in _compute_constraint_guidance
```

### 2.1 Module: `lesson_distiller.py`
Owns `runs/_lessons/<arxiv_id>.json` (gitignored — `/runs/` is in `.gitignore`, so this is runtime-only per the storage decision). Pure file I/O + a module-level `threading.Lock` + atomic `tmp`+`os.replace`; fail-soft on every path (mirrors `context_map.py` / `primitive_cache.py`). Public API:

- `is_enabled() -> bool` — `False` unless `REPROLAB_NEGATIVE_LESSONS` is truthy (`1`/`on`/`true`); default off.
- `mine_lessons(project_dir: Path, runs_root: Path, arxiv_id: str | None) -> None` — post-run: read this run's `experiment_runs.jsonl` + artifacts, upsert/retire lessons in the per-paper file. No-op when disabled or `arxiv_id` is None.
- `read_lessons(runs_root: Path, arxiv_id: str | None) -> dict` — return `{"version":"v1","arxiv_id":…,"lessons":[…]}`; empty shape when off / missing / corrupt / `arxiv_id` None.

### 2.2 Classifier accessor (`failure_classifier.py`)
Add a thin public wrapper so the distiller never reads an agent-authored fix string (§4):
```python
def suggested_fix_for_class(klass: str) -> str:
    """Public, deterministic class -> canonical suggested-fix string."""
    return _suggest(klass)
```

### 2.3 Mining hook (`run.py`)
Call `lesson_distiller.mine_lessons(project_dir, ctx.runs_root, ctx.arxiv_id)` in the run's **finalize/`finally` path** — the single chokepoint that runs exactly once at run end regardless of which `write_final_report_rlm` site fired (the normal path at run.py:1868 and the watchdog path at run.py:864 are mutually exclusive branches; the `finally` runs after either). Fail-soft; wrapped in try/except so it can never affect run teardown.

### 2.4 Injection hook (`baseline_implementation.py`)
A `_negative_lessons_block(arxiv_id, runs_root)` added to `_compute_constraint_guidance` beside the existing `_prior_rubric_feedback_block` (the near-exact intra-run precedent). Returns `""` when off / no `arxiv_id` / no active lessons. `runs_root` is derived from `project_dir.parent`.

### 2.5 Flag
`REPROLAB_NEGATIVE_LESSONS` — default `0`. Off ⇒ both hooks no-op (the miner writes nothing, the block returns `""`) — zero behavior change. Rollback = unset.

## 3. Eligibility allowlist (the noise filter — core)

Only **agent-correctable** classes become lessons. Environmental/transient classes won't recur deterministically and would be prompt junk.

**Eligible (`_LESSON_WORTHY_CLASSES`):**
`missing_module`, `torch_redundancy`, `requirements_not_found`, `missing_dataset`, `syntax_error`, `scope_shape_violation`, `contract_violation`, `dockerfile_invalid`.

**Excluded** (environmental/transient or already handled elsewhere):
`network_flake`, `runpod_capacity`, `runpod_transient_500`, `runpod_ssh_timeout`, `runpod_balance_too_low`, `cuda_oom` / `oom_killed` (handled by dynamic-GPU auto-escalation), `exec_timeout`, `watchdog_killed`, `preflight_blocked` (already a guard), `permission_denied` (FS/Docker-env), `unknown`.

The allowlist is an explicit, tunable constant.

## 4. Lesson schema + promotion + provenance

Per-paper file `runs/_lessons/<arxiv_id>.json`:
```json
{"version":"v1","arxiv_id":"2605.15155","lessons":[
  {"failure_class":"dockerfile_invalid",
   "suggested_fix":"<canonical classifier fix string>",
   "suggested_fix_source":"classifier",
   "occurrences":1,
   "status":"active",
   "first_seen_run":"prj_a","last_seen_run":"prj_a",
   "staleness":0}
]}
```

**`suggested_fix` provenance (no side-door).** `experiment_runs.jsonl::suggested_fix` is written with `setdefault`, so a pre-existing (possibly agent/LLM-authored) value can win over the classifier's. The distiller therefore **ignores the record's `suggested_fix` entirely** and re-derives it from `failure_classifier.suggested_fix_for_class(failure_class)`. The lesson keys on the `failure_class` enum (a controlled value) + a classifier-generated fix; `suggested_fix_source` is always `"classifier"`. No agent prose can reach the injected prompt.

**Promotion (candidate → active; recurrence-gated, NOT first-occurrence by default).** First occurrence stores the lesson as `status:"candidate"` — recorded but **not injected**. A lesson is `"active"` (injectable) only once `occurrences >= _promote_threshold(failure_class)`.
- Default threshold = **2** (a single allowlisted failure can still be a one-off from weird generated code — do not trust one).
- `_PROMOTE_THRESHOLD` carves out **structurally-deterministic** classes at **1** (promote on first): v1 = `{"dockerfile_invalid": 1}` (a malformed-Dockerfile failure is structural, not stochastic-code-dependent). Kept deliberately minimal/conservative; `torch_redundancy` is the obvious next candidate but stays at 2 in v1.

On each run, for every eligible class that **fired**: upsert the lesson → `occurrences += 1`, `last_seen_run = <run id>`, `staleness = 0`, recompute `status`.

## 5. Retirement — opportunity-aware staleness

A lesson should only age when its class **had a chance to fire**. Staleness is incremented only when the class's gating phase was reached this run and the class did not fire; the lesson is retired (dropped) at `staleness >= 3`.

`_phases_reached(project_dir) -> set[str]` — derived deterministically from this run's artifacts:
- `ENV_BUILT` — a Dockerfile was generated this run (`project_dir/Dockerfile` exists). *(Approximation: the run dir is reused across attempts of the same paper, so a stale Dockerfile can register the phase. Acceptable for a flag-gated prototype — worst case a lesson ages slightly faster than ideal.)*
- `EXPERIMENT_RAN` — `experiment_runs.jsonl` has ≥1 record (execution was attempted).
- `METRICS_PRODUCED` — ≥1 `experiment_runs.jsonl` record carries non-empty `metrics`/`per_model`, OR `final_report.json` has a `rubric` with non-empty `areas`.

`_OPPORTUNITY_PHASE: dict[str, str]` maps each eligible class to its gating phase:
| Class | Gating phase |
|---|---|
| `dockerfile_invalid`, `torch_redundancy`, `requirements_not_found` | `ENV_BUILT` |
| `missing_module`, `missing_dataset`, `syntax_error` | `EXPERIMENT_RAN` |
| `scope_shape_violation`, `contract_violation` | `METRICS_PRODUCED` |

Retirement pass (after the promotion pass): for each active/candidate lesson whose class did **not** fire this run, increment `staleness` **iff** `_OPPORTUNITY_PHASE[class] in phases_reached`; reset to 0 on a fire (already done in §4); drop any lesson with `staleness >= 3`.

## 6. Injection block (capped guardrail)

`_negative_lessons_block(arxiv_id, runs_root)` reads the per-paper file and renders only **`status:"active"`** lessons, top-**K=5** by (`occurrences` desc, then recency):
```
NEGATIVE LESSONS FROM PRIOR RUNS OF THIS PAPER — avoid repeating these failures:
- [dockerfile_invalid] <suggested_fix> (seen 3×)
- [scope_shape_violation] <suggested_fix> (seen 2×)
```
Length-bounded: each `suggested_fix` truncated to 200 chars; whole block ≤ ~1.5 KB. Class-tagged, never free prose. Returns `""` when off / no `arxiv_id` / no active lessons.

## 7. Safety / contamination

A bad lesson poisons *future runs of one paper* (cross-run blast radius — the reason it is flag-gated + runtime-only + per-paper). Bounded by:
1. **Recurrence-gated** (candidate until `occurrences >= 2`, except structurally-deterministic classes) — a one-off never injects.
2. **Structured + class-tagged + classifier-sourced fix** — no free agent prose can enter the prompt (§4).
3. **Allowlist** — no transient/environmental classes (§3).
4. **Capped** — top-5, length-bounded (§6).
5. **Opportunity-aware retire-on-stale** — self-cleaning; a fixed bug's lesson ages out (§5).
6. **Advisory only** — a lesson nudges the implementer; the Phase 3 evidence gate (`REPROLAB_EVIDENCE_GATE`) and the rubric remain the correctness backstops. A lesson cannot fabricate a passing verdict.

## 8. Testing (TDD)

- **Distiller unit:** an eligible class that fires → candidate lesson written (not injectable); same class fires again → `occurrences=2`, `status="active"`; `dockerfile_invalid` → `active` on first occurrence; an excluded class → ignored; `suggested_fix` always equals `suggested_fix_for_class(class)` regardless of the record's `suggested_fix` field; `suggested_fix_source=="classifier"`; non-firing on an opportunity run → `staleness++`; non-firing when the gating phase was NOT reached → `staleness` unchanged; `staleness>=3` → retired; `arxiv_id` None → no-op; flag off → no-op; corrupt file → fail-soft; concurrent-safe; atomic write.
- **Phase-detection unit:** `_phases_reached` returns `ENV_BUILT` when a Dockerfile exists, `EXPERIMENT_RAN` when experiment_runs has records, `METRICS_PRODUCED` when metrics/rubric present.
- **Classifier accessor unit:** `suggested_fix_for_class("dockerfile_invalid")` returns the canonical non-empty string; unknown class → `""`.
- **Injection unit:** empty/missing/flag-off → `""`; only `active` lessons render; capped at 5; class-tagged; length-bounded.
- **Mining-hook unit:** runs once post-finalize; fail-soft (raises nothing into teardown); no-op when off.

## 9. Validation (A/B)

Does a class-tagged lesson cut recurrence of its `failure_class` on the next SDAR run? `REPROLAB_NEGATIVE_LESSONS=1` vs `0` across ≥3 paired runs of the same paper — compare **recurrence rate of the targeted `failure_class`** and **iterations-to-first-rubric-pass**; the lesson earns its keep only if it cuts the failure **without lowering `final_report.json::rubric.overall_score`**. **Rollback:** `REPROLAB_NEGATIVE_LESSONS=0`.

## 10. Files

- **Create:** `backend/agents/rlm/lesson_distiller.py`
- **Create:** `tests/agents/rlm/test_lesson_distiller.py`
- **Create:** `tests/agents/rlm/test_negative_lessons_injection.py`
- **Modify:** `backend/agents/rlm/failure_classifier.py` (public `suggested_fix_for_class`)
- **Modify:** `backend/agents/rlm/run.py` (mining hook in the finalize/`finally` path)
- **Modify:** `backend/agents/baseline_implementation.py` (`_negative_lessons_block` in `_compute_constraint_guidance`)
- **Modify:** `CLAUDE.md` (document `REPROLAB_NEGATIVE_LESSONS`) + the master plan status table.
