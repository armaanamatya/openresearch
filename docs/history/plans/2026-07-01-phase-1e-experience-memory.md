# Phase 1e — Advisory ExperienceMemory (FailureAttribution + global-infra store + held-out gate) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Advisory-only cross-run self-improvement: (A) a `FailureAttribution{signature, root_cause, scope∈{infra,method}, confidence, evidence_refs}` schema (extends `failure_classifier`, which today returns only `(class, fix)`); (B) a global-infra memory store + an `ExperienceMemory` that WRAPS the existing per-paper `_lessons`/`recipes`/`capsules`; (C) an `EvidenceVector` held-out non-regression gate + `ReplayCase`/`CandidateLesson` contracts. **Never auto-mutates run mechanics** — memory is injected as agent/provisioning HINTS only. All default-OFF ⇒ byte-identical.

**Architecture:** The trust signal is the SAME multi-predicate deterministic evidence layer the harness already computes — **reuse `evidence_audit.EvidenceAudit`** (`backed_by_ledger`, `provenance_present`, `metrics_non_degenerate`, `metric_keys_real`, `rerun_agrees`, `run_level_clean`) as the `EvidenceVector`, never a scalar grade. Scope (`infra`|`method`) is the routing key: `infra` → global store (`runs/_memory/infra/`, keyed by `signature` not arxiv_id), `method` → the existing per-paper `_lessons`. `ExperienceMemory` orchestrates the existing memory modules + the one new global store; only that store is new.

**Tech Stack:** Python 3.12 / floor 3.11; `pytest` socket-hermetic; stdlib + existing rlm memory modules. No new deps.

## Global Constraints
- **Advisory only — never auto-mutates run mechanics.** Memory produces guidance strings injected into the implementer/provisioning prompt, exactly like `lesson_distiller.negative_lessons_block`/`recipe_library.recipe_guidance_block` do today.
- **Evidence, not grade (the red line):** promotion/admission reads the `EvidenceVector` (predicate-level, validator veto absolute), NEVER a scalar LLM grade. A brace/static check + a test enforce grade fields are never the admission signal.
- **Scope routing invariant:** a `method`-scoped lesson can NEVER enter global memory (a test enforces this).
- **Wrap, don't replace:** reuse `lesson_distiller`, `recipe_library`, `failure_capsule`, `evidence_audit` — only the global-infra store is a new surface. Do NOT rewrite them.
- All flag-gated default-OFF (`OPENRESEARCH_EXPERIENCE_MEMORY`); unset ⇒ byte-identical. Fail-soft: any memory error degrades to "no hint," never aborts a run. Env naming `OPENRESEARCH_*`.
- Commit at the milestone; no CC prefix; no `Co-Authored-By`; author `lolout1`; push to deepinvent after commit.

## Component → file map
| Component | New/extends | Where |
|---|---|---|
| `FailureAttribution` + `attribute_failure` + scope table | NEW (over `failure_classifier`) | `backend/agents/rlm/failure_attribution.py` |
| global-infra store + `ExperienceMemory` | NEW/orchestrates | `backend/agents/rlm/experience_memory.py` |
| `EvidenceVector` held-out gate + `ReplayCase`/`CandidateLesson` | NEW | `backend/agents/rlm/held_out_gate.py` |

---

## Unit A — FailureAttribution (root-cause + scope)

**Files:** Create `backend/agents/rlm/failure_attribution.py`; Test `tests/rlm/test_failure_attribution.py`.

**Consumes:** `failure_classifier.classify_failure(result) -> (failure_class, suggested_fix)` + `FAILURE_CLASSES` (read the list to build the scope table).

**Interfaces:**
```python
# Which failure classes are infra (paper-invariant, cross-paper) vs method (paper-specific).
_INFRA_CLASSES: frozenset[str] = frozenset({
    "missing_module", "requirements_not_found", "dockerfile_invalid", "cuda_shlib_load",
    "cuda_oom", "oom_killed", "network_flake", "runpod_capacity", "runpod_transient_500",
    "runpod_ssh_timeout", "disk_exhausted", "nccl_timeout", "cuda_device_assert",
})
_METHOD_CLASSES: frozenset[str] = frozenset({
    "scope_shape_violation", "contract_violation", "silent_oom", "insufficient_train_steps",
    "insufficient_training", "degenerate_training", "incomplete_metrics", "code_bug",
    "fabrication_suspected", "result_quality",
})
# unknown/ambiguous classes default to "method" (conservative: never pollute global infra memory).

@dataclass(frozen=True)
class FailureAttribution:
    signature: str            # stable hash of (failure_class + normalized error tail)
    root_cause: str           # the failure_class (first-decisive-error)
    scope: str                # "infra" | "method"
    confidence: float         # 0..1 (1.0 for a classifier hit, lower for the unknown default)
    evidence_refs: tuple[str, ...] = ()   # e.g. ("experiment_runs.jsonl#<n>",)

def attribute_failure(result: dict, *, arxiv_id: str | None = None,
                      evidence_refs: tuple[str, ...] = ()) -> FailureAttribution: ...
    # classify_failure(result) -> (klass, _fix); scope = infra/method/default-method;
    # signature = sha1(f"{klass}:{_normalize(error_tail)}")[:16]; confidence 1.0 hit / 0.5 unknown.
```
`_normalize` strips digits/paths/hex so the same root cause across runs yields ONE signature (AgentDebug: first-decisive-error, not a substring match).

- [ ] **Step 1: Write the failing tests**
```python
from backend.agents.rlm.failure_attribution import FailureAttribution, attribute_failure


def test_infra_class_routes_infra():
    att = attribute_failure({"success": False, "error": "ImportError: No module named 'flash_attn'",
                             "failure_class": "missing_module"})
    assert att.scope == "infra" and att.root_cause == "missing_module" and att.signature


def test_method_class_routes_method():
    att = attribute_failure({"success": False, "failure_class": "scope_shape_violation"})
    assert att.scope == "method"


def test_unknown_class_defaults_method_not_infra():
    # conservative: never let an unclassified failure pollute cross-paper infra memory.
    att = attribute_failure({"success": False, "failure_class": "totally_unknown_class"})
    assert att.scope == "method" and att.confidence < 1.0


def test_same_root_cause_same_signature_across_runs():
    a = attribute_failure({"failure_class": "cuda_shlib_load", "error": "libcupti.so.12: cannot open (pid 4821)"})
    b = attribute_failure({"failure_class": "cuda_shlib_load", "error": "libcupti.so.12: cannot open (pid 9930)"})
    assert a.signature == b.signature       # pid/path differences normalized away
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (read `failure_classifier.FAILURE_CLASSES` to keep the scope table exhaustive; `classify_failure` may return the class already in `result["failure_class"]` — prefer the explicit field, fall back to `classify_failure`). **Step 4:** Run → PASS. Lint clean.

---

## Unit C — EvidenceVector held-out gate + ReplayCase/CandidateLesson

**Files:** Create `backend/agents/rlm/held_out_gate.py`; Test `tests/rlm/test_held_out_gate.py`.

**Consumes:** `evidence_audit.EvidenceAudit` (the EvidenceVector: `backed_by_ledger`/`provenance_present`/`metrics_non_degenerate`/`metric_keys_real`/`rerun_agrees`/`run_level_clean`), `FailureAttribution` (Unit A).

**Interfaces:**
```python
_HELD_OUT_PREDICATES = ("backed_by_ledger", "provenance_present", "metrics_non_degenerate", "metric_keys_real")
# the VALIDATOR VETO predicate — an absolute gate, never outvoted:
_VETO_PREDICATE = "run_level_clean"

@dataclass(frozen=True)
class CandidateLesson:
    attribution: "FailureAttribution"
    patch: dict                 # the advisory hint payload (guidance text + provenance)
    admission_state: str = "candidate"   # "candidate" | "active" | "rejected"

@dataclass(frozen=True)
class ReplayCase:
    id: str
    expected_predicates: dict   # predicate_name -> bool (the baseline, held-out)
    # apply(lesson) -> EvidenceVector is injected as a callable in Phase 1e (CPU/cheap-tier only)

def evidence_vector(audit) -> dict:   # EvidenceAudit -> {predicate: bool}
def admit(candidate: CandidateLesson, replay_set: list[ReplayCase],
          apply_fn) -> CandidateLesson: ...
    # For each ReplayCase: v = evidence_vector(apply_fn(candidate, case)).
    # Promote to "active" IFF for EVERY held-out case: the VETO predicate is True,
    # AND no held-out predicate regressed (True->False) vs case.expected_predicates,
    # AND >=1 held-out predicate improved (False->True) in at least one case.
    # Otherwise "rejected" (logged, never applied). NEVER reads a scalar grade.
```

- [ ] **Step 1: Write the failing tests**
```python
from backend.agents.rlm.held_out_gate import CandidateLesson, ReplayCase, admit, evidence_vector
from backend.agents.rlm.failure_attribution import attribute_failure


def _cand():
    return CandidateLesson(attribution=attribute_failure({"failure_class": "missing_module"}), patch={"hint": "add flash_attn"})


def _case(preds):  # expected (baseline) predicates
    return ReplayCase(id="c1", expected_predicates=preds)


def test_promotes_on_improvement_no_regression():
    baseline = {"backed_by_ledger": True, "provenance_present": False, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    improved = dict(baseline, provenance_present=True)   # one predicate improved, veto True, none regressed
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: improved)
    assert out.admission_state == "active"


def test_rejects_on_any_regression():
    baseline = {"backed_by_ledger": True, "provenance_present": True, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    regressed = dict(baseline, metric_keys_real=False)   # a held-out predicate regressed
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: regressed)
    assert out.admission_state == "rejected"


def test_rejects_on_veto_false_even_if_others_improve():
    baseline = {"backed_by_ledger": False, "provenance_present": False, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    fabricated = dict(baseline, backed_by_ledger=True, provenance_present=True, run_level_clean=False)  # veto fails
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: fabricated)
    assert out.admission_state == "rejected"   # validator veto is absolute — never a scalar override


def test_no_improvement_stays_rejected():
    baseline = {"backed_by_ledger": True, "provenance_present": True, "metrics_non_degenerate": True,
                "metric_keys_real": True, "run_level_clean": True}
    out = admit(_cand(), [_case(baseline)], apply_fn=lambda c, case: dict(baseline))  # identical, nothing improved
    assert out.admission_state == "rejected"
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `held_out_gate.py` (`evidence_vector(audit)` reads the `EvidenceAudit` attributes into a `{predicate: bool}` dict — accept either a real `EvidenceAudit` or a plain dict from `apply_fn`; `admit` per the rules above). **Step 4:** Run → PASS. Lint clean.

---

## Unit B — global-infra store + ExperienceMemory (wraps existing)

**Files:** Create `backend/agents/rlm/experience_memory.py`; Test `tests/rlm/test_experience_memory.py`.

**Consumes:** `FailureAttribution` (A), `lesson_distiller` (per-paper `_lessons`), `recipe_library`, `held_out_gate` (C, for infra-lesson promotion), the new global-infra store.

**Interfaces:**
```python
def _infra_store_path(runs_root, signature) -> Path:   # runs/_memory/infra/<signature>.json (atomic write)

class ExperienceMemory:
    def __init__(self, runs_root, *, enabled: bool | None = None) -> None: ...   # enabled default = OPENRESEARCH_EXPERIENCE_MEMORY
    def record(self, attribution: "FailureAttribution", *, arxiv_id: str | None,
               hint: str) -> None: ...
        # scope=="infra" -> global-infra store (keyed by signature, recurrence-counted, caps<=5/<=200c);
        # scope=="method" -> DELEGATE to the existing per-paper lessons store (never global).
    def infra_hints(self, *, env_fingerprint: str = "") -> list[str]: ...    # top-k bounded infra hints
    def guidance_block(self, *, arxiv_id: str | None, env_fingerprint: str = "") -> str: ...
        # composes the existing recipe/lesson blocks + the new infra hints; advisory only.
```
Guardrails (mirror `lesson_distiller`): recurrence-gated promotion (`occurrences>=2`), dedup, caps (≤5 hints / ≤200 chars), staleness retirement. Fail-soft everywhere.

- [ ] **Step 1: Write the failing tests**
```python
from backend.agents.rlm.experience_memory import ExperienceMemory
from backend.agents.rlm.failure_attribution import attribute_failure


def test_infra_attribution_writes_global_store(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=True)
    att = attribute_failure({"failure_class": "cuda_shlib_load", "error": "libcupti.so.12"})
    mem.record(att, arxiv_id="2605.15155", hint="prepend venv CUDA lib dirs to LD_LIBRARY_PATH")
    mem.record(att, arxiv_id="2512.99999", hint="prepend venv CUDA lib dirs to LD_LIBRARY_PATH")  # recurrence>=2
    hints = mem.infra_hints()
    assert any("LD_LIBRARY_PATH" in h for h in hints)
    # cross-paper: the SAME signature from a DIFFERENT arxiv_id contributed.
    assert (tmp_path / "_memory" / "infra").exists()


def test_method_attribution_never_enters_global_store(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=True)
    att = attribute_failure({"failure_class": "scope_shape_violation"})
    mem.record(att, arxiv_id="2605.15155", hint="emit explicit model_key/env/baseline per cell")
    assert mem.infra_hints() == []                       # method scope NEVER global (the routing invariant)
    infra_dir = tmp_path / "_memory" / "infra"
    assert not infra_dir.exists() or not any(infra_dir.iterdir())


def test_disabled_is_noop(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=False)
    mem.record(attribute_failure({"failure_class": "cuda_shlib_load"}), arxiv_id="x", hint="h")
    assert mem.infra_hints() == [] and mem.guidance_block(arxiv_id="x") == ""


def test_infra_hints_bounded_and_deduped(tmp_path):
    mem = ExperienceMemory(tmp_path, enabled=True)
    for i in range(12):
        att = attribute_failure({"failure_class": "network_flake", "error": f"conn reset {i}"})
        mem.record(att, arxiv_id="a"); mem.record(att, arxiv_id="b")
    assert len(mem.infra_hints()) <= 5                   # cap enforced
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (global-infra store = atomic JSON per signature with `{occurrences, hint, scope, staleness}`; `record` routes by `attribution.scope` — infra→global, method→`lesson_distiller` per-paper path; `infra_hints` reads promoted (`occurrences>=2`) infra entries, dedup + cap; `guidance_block` composes recipe/lesson/infra; `enabled` gates everything, unset reads `OPENRESEARCH_EXPERIENCE_MEMORY`). **Step 4:** Run → PASS. Lint clean.

---

## Validation
- [ ] `.venv/bin/python -m pytest tests/rlm/test_failure_attribution.py tests/rlm/test_held_out_gate.py tests/rlm/test_experience_memory.py -q`
- [ ] Broad regression: `.venv/bin/python -m pytest tests/rlm/ -q` (catches any collateral).
- [ ] Ruff clean on all new files. Import smoke of the live path.
- [ ] Docs: CLAUDE.md note + memory update.

## Self-Review (against spec §7)
- §7 `FailureAttribution{signature,root_cause,scope,confidence,evidence_refs}` + method-never-global test → Unit A + B. ✓
- §7 global-infra store (keyed by signature) + `ExperienceMemory` wrapping existing memory → Unit B. ✓
- §7 held-out gate over an `EvidenceVector` (reuse `EvidenceAudit`), validator veto absolute, never a scalar → Unit C. ✓
- §7 `ReplayCase`/`CandidateLesson` contracts → Unit C. ✓
- §7 advisory-only (hints, never auto-mutate mechanics) → all units emit guidance strings only. ✓
- **Deferred (honest, per spec):** the live replay CORPUS (Phase 1e ships a minimal in-test set; a real cached-run corpus grows over runs — until then a lesson stays candidate/advisory-low-confidence); the staged harness self-edit tier (north-star, off).
