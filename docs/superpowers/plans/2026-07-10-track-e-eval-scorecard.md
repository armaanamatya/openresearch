# Track E — Eval Scorecard + Typed EvaluationReport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Subagent guardrails (2026-07-10 incident):** a delegated worker MUST NOT run any git state command (`commit`/`add`/`amend`/`checkout`/`stash`/`reset`/`rebase`), MUST restrict writes to the exact files its task names, and MUST NEVER edit or delete an existing test — if a test blocks you, STOP and report. If you think a production file outside your task's file list needs changing, STOP and report. The lead reviews every diff and owns all commits.

**Goal:** Emit a per-run diagnostic **scorecard** + typed **`EvaluationReport`** (json+md) that records all 11 proposed evaluator dimensions with artifact-anchored provenance — deterministic dimensions as downward-only verdict gates, LLM-judged dimensions as display-only — plus the new deterministic instrumentation (human-intervention, per-experiment efficiency) and an out-of-process re-grade ok-receipt, without any composite or new signal ever reaching `final_report.verdict`.

**Architecture:** Track E adds a thin typed **adapter/view** layer in `backend/evals/` that *composes* the existing `backend/agents/schemas.py` + `report.py` types (never forks them), consumes the deterministic evidence the VerdictAuthority already computes, and writes an `EvaluationReport` sidecar at finalize. The discrete evidence verdict (`verdict_authority.decide`) stays the sole gate; the composite is refactored to be deterministic-dominated and stays confined to `backend/evals/` (report/rank-only). Every new capability is `os.environ`-gated, default-OFF, byte-identical when off.

**Tech Stack:** Python ≥3.11, stdlib + pydantic v2 (existing), pytest (socket-hermetic, `-n auto` in CI). No new dependencies. No LLM in any gate; LLM-judged rows are advisory display-only.

## Global Constraints

- **North-star (structural):** `verdict_authority.decide()` stays the SOLE verdict writer. Every new signal runs *before* `decide()` as a **downward-only gate** (like `claim_gate_cap`) or is **display-only**. Nothing new may raise a verdict. **No new field may write the rubric surface** (`rubric`/`overall_score`/`target_score`/`meets_target`). Every new writer must pass the Track-0 tripwire `verdict_authority.assert_verdict_surface_unchanged` (governs `VERDICT_SURFACE_KEYS = ("verdict","implementation_verdict","replication_verdict")`, covers `final_report.json` + `demo_status.json`).
- **Flag idiom (default-OFF):** `backend/agents/rlm/feature_flags.py::env_truthy("FLAG")` — truthy tokens `1/true/yes/on`. Byte-identical when off or `=0`. Every flag ships a hermetic OFF+ON test pair (`tests/CLAUDE.md`).
- **Provenance is typed + three-way:** every scorecard value tagged `paper_reported | agent_measured | evaluator_computed`, anchored to an `evidence_bundle` sha256 receipt. Generated/inferred values are never treated as reported ground truth.
- **Skill = structure · paper text = values · deterministic evidence = pass/fail.** A skill supplies structure only; it can NEVER supply a pass. A dedicated test asserts a skill cannot flip a claim to pass.
- **Composite is report/rank-only.** It can never reach `final_report.verdict`/`demo_status.verdict`/`rubric`/`meets_target`. A guard test enforces this.
- **Determinism:** no `Date.now()`-style nondeterminism in scored output; timestamps come from the run's own clock at write time and are excluded from any equality/idempotency assertion.

**Real code map (verified 2026-07-10 via recon; match by symbol, lines drift ±10):**
- `backend/agents/rlm/primitives.py`: `_persist_experiment_result`:4880 (JSONL append :4954-4959, `entry={"timestamp",**result,"model_id","eval_env"}`); `_stamp_manifest_ids`:4867 (called :7471 — the run-scoped seam where `experiment_run_id`/`env_id`/`commands`/`gpu_plan`/`_retry_idx` are in scope); `_manifest_enrichment`:4834 (`metrics_sha256`:4862); `_classify_run_experiment_outcome`:589; gpu_plan load :7034-7040; final `return _persist_experiment_result(...)`:7949. **Absent today:** `start_ts`/`end_ts`/`gpu_hours`/`estimated_cost`/`retry_id`/`gpu_plan` on the row.
- `backend/agents/rlm/report.py`: `RLMFinalReport`:32 (fields 39-232); `_has_experiment_evidence`:1403; `_authority_evidence_gate`:1507; `_apply_evidence_gate`:1653 (forge test :1764-1767); **`run_experiment_success_count(ctx)`:1906-1926** (returns `None` when `ctx.cost_ledger is None` — the out-of-process seam); `write_final_report_rlm` finalize chokepoint; `_authority_active` block ~2400-2525 (authority decide :2416, demo_status mirror :2455-2468, tripwire :2519-2525).
- `backend/agents/schemas.py`: `MetricSpec`:60, `PaperClaimMap`:88, `ReproductionContract`:403, `ExperimentArtifacts`:475, `RubricVerification`+`from_areas`:651/668 (deterministic weight-normalized `overall_score`), `MetricDelta`:705 (`relative_error_vs_paper`:718, `effect_size` Cohen's d:721, `ci95_half_width`:722).
- `backend/evals/schemas.py`: `ReproductionScore`:26 (`composite_score`:42 = `0.1·build+0.2·run+0.4·metric_match+0.3·fidelity`), `HypothesisScore`:55, `IntegrityReport`:84 (`data_leakage`/`selective_reporting` via `IntegrityFlag`:76); `ResearchMapScore.composite_score`:112. **Callers:** only `backend/evals/store.py:115/124` (SQLite `composite` column) + `tests/test_eval_store.py` — NEVER the verdict surface.
- `backend/agents/rlm/reproduction_campaign.py`: `CampaignLedger.append_row`:205 (atomic write-ahead + torn-tail repair + fsync — the durable append to reuse); operator approval/resume seams :450-457 / :546-554 / :649-657.
- `backend/routes/messages.py`: `post_message`:64-99 (user_messages.jsonl append :88-97), `post_campaign_message`:102-150 (campaign steering :133-148).
- `backend/agents/rlm/result_fidelity.py`: `normalize_repro_spec_claims` (idempotent nested→flat lift, committed 15880345) + `evaluate`.
- `backend/agents/rlm/evidence_bundle.py`: `rlm_state/evidence_bundle.json` receipt (`metrics_sha256`/`code_tree_digest`/`coordinates`).
- Skills: `backend/agents/rlm/skill_selection.py` (`rlm_state/active_skills.json`), `consult_skill` primitive.

---

### Task 0: Prerequisite — normalize fix is landed; fix the stale Adam acceptance docstring

**Status:** `normalize_repro_spec_claims` (nested→flat idempotent claim lift) is already committed (`15880345`). It makes `result_fidelity.evaluate` read real extractor claims — without it the authority was inert (`inconclusive` on every real run). This is Track E's foundation.

**Files:**
- Modify: `tests/acceptance/test_adam_verdict_reground.py` (docstring only)

**Interfaces:** none (doc-only). Produces no code.

- [ ] **Step 1:** Read `tests/acceptance/test_adam_verdict_reground.py`'s module + `test_adam_headline_reground_is_inconclusive_not_reproduced` docstrings. They currently claim the Adam claim is "invisible to the reader" (`metric_name=None`, `is_primary=False`). With `normalize_repro_spec_claims` the claim is now VISIBLE (`is_primary=True`, `ambiguous=True`), so it lands `inconclusive` via **Rule 1 (ambiguous primary)** — the correct mechanism, not invisibility.
- [ ] **Step 2:** Update BOTH docstrings surgically: state that the nested→flat adapter now exists (`normalize_repro_spec_claims`, the "adapter inserted ahead of evaluate()" the old docstring predicted), so the claim is read as an ambiguous primary and Rule 1 fires for the RIGHT reason. Keep the assertion unchanged (`verdict == "inconclusive"`, `reason == "no_measurable_target"`). Do NOT touch any assertion or helper.
- [ ] **Step 3:** Run `.venv/bin/python -m pytest tests/acceptance/test_adam_verdict_reground.py -v` → all pass (assertions unchanged).
- [ ] **Step 4:** Commit — `git commit -m "Update Adam acceptance docstring: normalize adapter makes the claim a visible ambiguous primary (Rule 1), not invisible"`

---

### Task 1: Deterministic-dominated offline composite + verdict-surface guard (§6.2)

**Files:**
- Modify: `backend/evals/schemas.py` (`ReproductionScore.composite_score`:42, + module weight constants)
- Test: `tests/test_eval_composite.py` (new)

**Interfaces:**
- Produces: `ReproductionScore.composite_score(weights: CompositeWeights | None = None) -> float` — deterministic-dominated default; the LLM `fidelity_score` term dropped to `0.0` weight by default. Module constant `DEFAULT_COMPOSITE_WEIGHTS: CompositeWeights` (a frozen dataclass `{build, run, metric_match, fidelity}` summing to 1.0). All component fields (`build_success`/`run_success`/`metric_match`/`fidelity_score`) stay on the model, preserved separately.

**Design (locked):** re-weight onto the deterministic components. New default `{build:0.15, run:0.25, metric_match:0.60, fidelity:0.0}` (deterministic-dominated; fidelity_score, the only LLM term, drops out of the default blend). Weights are overridable per call for A/B tuning (spec §10 Q1). This is a deliberate change to the `backend/evals/` SQLite `composite` column (report/rank-only) — NOT byte-identical, but every component field is preserved separately and the composite still never reaches the verdict.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_composite.py
import ast
from pathlib import Path
from backend.evals.schemas import ReproductionScore, DEFAULT_COMPOSITE_WEIGHTS

def test_default_weights_sum_to_one():
    w = DEFAULT_COMPOSITE_WEIGHTS
    assert abs(w.build + w.run + w.metric_match + w.fidelity - 1.0) < 1e-9

def test_composite_is_deterministic_dominated_fidelity_zero_by_default():
    # A run with a HIGH LLM fidelity but ZERO deterministic evidence must score ~0.
    s = ReproductionScore(build_success=False, run_success=False,
                          metric_match=0.0, fidelity_score=1.0)
    assert s.composite_score() == 0.0  # the LLM term cannot lift the default composite

def test_composite_dominated_by_metric_match():
    s = ReproductionScore(build_success=True, run_success=True,
                          metric_match=1.0, fidelity_score=0.0)
    assert abs(s.composite_score() - 1.0) < 1e-9

def test_component_fields_preserved_separately():
    s = ReproductionScore(fidelity_score=0.77, metric_match=0.5)
    assert s.fidelity_score == 0.77 and s.metric_match == 0.5  # not collapsed into composite

def test_composite_never_reaches_verdict_surface():
    # Static guard: no backend/evals module writes a reproduction-verdict key,
    # and report.py never imports backend.evals.schemas.
    report_src = Path("backend/agents/rlm/report.py").read_text()
    assert "backend.evals.schemas" not in report_src and "from backend.evals" not in report_src
    store_src = Path("backend/evals/store.py").read_text()
    for banned in ('"verdict"', "'verdict'", "meets_target", "target_score"):
        assert banned not in store_src, f"{banned} must not be written by the evals store"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_eval_composite.py -v` → FAIL (`DEFAULT_COMPOSITE_WEIGHTS` missing; old composite folds fidelity 0.3).
- [ ] **Step 3: Implement** — add a frozen `@dataclass CompositeWeights(build, run, metric_match, fidelity)` + `DEFAULT_COMPOSITE_WEIGHTS = CompositeWeights(0.15, 0.25, 0.60, 0.0)` to `backend/evals/schemas.py`; rewrite `composite_score(self, weights=None)` to use `weights or DEFAULT_COMPOSITE_WEIGHTS`. Keep all fields. Do NOT touch `ResearchMapScore` or the verdict path.
- [ ] **Step 4: Run to verify pass** — `pytest tests/test_eval_composite.py tests/test_eval_store.py -v` → PASS (update `tests/test_eval_store.py`'s composite expectation to the new weights if it pins the old 0.3-fidelity value).
- [ ] **Step 5: Commit** — `git commit -m "Refactor ReproductionScore.composite_score deterministic-dominated (drop LLM fidelity term from default); guard it off the verdict surface"`

---

### Task 2: `HumanIntervention` deterministic capture (§6.5) — DELEGATABLE (guarded Sonnet)

**Files:**
- Create: `backend/agents/rlm/human_intervention.py`
- Modify: `backend/routes/messages.py` (`post_message`:88-97, `post_campaign_message`:141-148)
- Modify: `backend/agents/rlm/reproduction_campaign.py` (:453 resume, :553 awaiting-operator, :656 checkpoint-pause — add one call each)
- Test: `tests/agents/rlm/test_human_intervention.py`

**Interfaces:**
- Produces: `record_intervention(project_dir: Path, *, kind: str, what: str, why: str = "", artifact_diff: str = "", blocking: bool = False) -> bool` — atomically appends one JSON row to `runs/<id>/human_interventions.jsonl` (reuse the `CampaignLedger.append_row` durability recipe: temp/torn-tail/fsync via a small local helper; fail-soft returns False, never raises). Row: `{ts, kind, what, why, artifact_diff, blocking}`. `kind ∈ {credentials, clarification, dataset-approval, env-repair, code-fix, config-choice, scientific-judgment, compute-approval, manual-artifact, resume-approval}`. `human_intervention_enabled()` reads `OPENRESEARCH_HUMAN_INTERVENTION_LOG` (default-OFF). Also `autonomy_metric(project_dir) -> dict` = `{n_interventions, n_blocking, by_kind, autonomy_score}` (a DISPLAY stat; `autonomy_score` = `1.0 - weighted_intervention_density`, credentials/legally-required weighted lower than technical/scientific assistance). **Never gates.**

**Guardrails for the worker:** touch ONLY the 4 files above. Do NOT change `respond_to_user` (that is the assistant side, not operator). Do NOT alter `user_messages.jsonl` writes — ADD alongside. Off ⇒ every hook is a no-op (byte-identical).

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/rlm/test_human_intervention.py
import json
from pathlib import Path
import pytest
from backend.agents.rlm.human_intervention import (
    record_intervention, autonomy_metric, human_intervention_enabled,
)

def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", raising=False)
    assert human_intervention_enabled() is False

def test_record_appends_row(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", "1")
    assert record_intervention(tmp_path, kind="credentials", what="added HF token", blocking=True) is True
    rows = [json.loads(l) for l in (tmp_path / "human_interventions.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "credentials" and rows[0]["blocking"] is True and "ts" in rows[0]

def test_off_state_no_write(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", raising=False)
    assert record_intervention(tmp_path, kind="clarification", what="x") is False
    assert not (tmp_path / "human_interventions.jsonl").exists()

def test_autonomy_metric_weights_credentials_lower(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", "1")
    record_intervention(tmp_path, kind="credentials", what="token")       # legally-required
    record_intervention(tmp_path, kind="scientific-judgment", what="pick") # technical/scientific
    m = autonomy_metric(tmp_path)
    assert m["n_interventions"] == 2 and set(m["by_kind"]) == {"credentials", "scientific-judgment"}
    assert 0.0 <= m["autonomy_score"] <= 1.0

def test_record_never_raises_on_bad_dir(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_HUMAN_INTERVENTION_LOG", "1")
    assert record_intervention(Path("/nonexistent/xyz"), kind="clarification", what="x") is False
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/agents/rlm/test_human_intervention.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `human_intervention.py` (writer + `autonomy_metric` + flag). Then add the 5 hook calls (routes: after the 404/validation guard, alongside the existing appends; campaign: at the 3 approval/resume seams), each wrapped so an off-flag or failure is a silent no-op.
- [ ] **Step 4: Run to verify pass** — `pytest tests/agents/rlm/test_human_intervention.py tests/routes/ -v` → PASS; confirm routes tests unaffected off-flag.
- [ ] **Step 5: STOP and report the diff to the lead** (do not commit). Lead reviews + commits: `git commit -m "Add deterministic HumanIntervention capture (human_interventions.jsonl) at operator-ingress; autonomy display metric, never gates"`

---

### Task 3: Per-experiment efficiency instrumentation (§6.5) — DELEGATABLE (guarded Sonnet)

> **Status (2026-07-10):** split into **3a — DONE** (`gpu_ledger.py` writer +
> `aggregate_gpu_cost` + `tests/agents/rlm/test_gpu_ledger.py`, commit `4ff678b0`;
> flag-gated, byte-identical off since nothing calls it yet) and **3b — PENDING**
> (the `primitives.py` wiring: stamp `start_ts`/`end_ts`/`gpu_plan`/`retry_id` at
> the `_stamp_manifest_ids` call seam + `append_gpu_ledger` at the persist seam,
> flag-gated + off-flag byte-identical). Do 3b in-tree, not a stale worktree.

**Files (3b — remaining):**
- Create: `backend/agents/rlm/gpu_ledger.py`
- Modify: `backend/agents/rlm/primitives.py` (stamp row fields at `_stamp_manifest_ids` call site :7471; append GPU-ledger row near the persist :4943-4959)
- Test: `tests/agents/rlm/test_gpu_ledger.py`

**Interfaces:**
- Produces: on each `experiment_runs.jsonl` row (when `OPENRESEARCH_GPU_LEDGER` on), new keys `start_ts`, `end_ts`, `retry_id`, `gpu_plan` (a snapshot dict from `rlm_state/gpu_plan.json`), added via `setdefault` at the `_stamp_manifest_ids` seam so the base schema stays intact off-flag. PLUS a separate `gpu_ledger.py::append_gpu_ledger(project_dir, *, experiment_run_id, start_ts, end_ts, gpu_plan, provider, rate_usd_per_hr) -> bool` writing `runs/<id>/gpu_ledger.jsonl` rows `{experiment_run_id, start_ts, end_ts, gpu_hours, provider, rate_usd_per_hr, est_cost_usd}` (do NOT overload the token `cost_ledger`). `gpu_ledger_enabled()` reads `OPENRESEARCH_GPU_LEDGER`. `aggregate_gpu_cost(project_dir) -> dict` = `{total_gpu_hours, total_est_cost_usd, by_experiment}` (DISPLAY-only; never a fitness term, never gates).

**Guardrails:** touch ONLY the 3 files. `start_ts` is captured in `run_experiment` (near the gpu_plan load :7034-7040) and threaded to the stamp seam; `end_ts` ≈ the persist `timestamp`. Off ⇒ no new row keys, no gpu_ledger.jsonl (byte-identical).

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/rlm/test_gpu_ledger.py
import json
from pathlib import Path
import pytest
from backend.agents.rlm.gpu_ledger import (
    append_gpu_ledger, aggregate_gpu_cost, gpu_ledger_enabled,
)

def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GPU_LEDGER", raising=False)
    assert gpu_ledger_enabled() is False

def test_append_and_aggregate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    append_gpu_ledger(tmp_path, experiment_run_id="e1", start_ts="2026-07-10T00:00:00+00:00",
                      end_ts="2026-07-10T01:00:00+00:00", gpu_plan={"sku": "A100"},
                      provider="gcp", rate_usd_per_hr=3.0)
    agg = aggregate_gpu_cost(tmp_path)
    assert abs(agg["total_gpu_hours"] - 1.0) < 1e-6
    assert abs(agg["total_est_cost_usd"] - 3.0) < 1e-6
    assert agg["by_experiment"]["e1"]["gpu_hours"] == pytest.approx(1.0)

def test_off_state_no_write(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_GPU_LEDGER", raising=False)
    assert append_gpu_ledger(tmp_path, experiment_run_id="e1", start_ts="a", end_ts="b",
                             gpu_plan={}, provider="gcp", rate_usd_per_hr=3.0) is False
    assert not (tmp_path / "gpu_ledger.jsonl").exists()

def test_gpu_hours_zero_when_timestamps_unparseable(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_GPU_LEDGER", "1")
    append_gpu_ledger(tmp_path, experiment_run_id="e2", start_ts="bad", end_ts="also-bad",
                      gpu_plan={}, provider="local", rate_usd_per_hr=0.0)
    assert aggregate_gpu_cost(tmp_path)["by_experiment"]["e2"]["gpu_hours"] == 0.0
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).
- [ ] **Step 3: Implement** `gpu_ledger.py` (parse ISO timestamps → hours, fail-soft to 0.0; est_cost = hours×rate) + the primitives.py stamp/append hooks (flag-gated, `setdefault`, fail-soft).
- [ ] **Step 4: Run to verify pass** — `pytest tests/agents/rlm/test_gpu_ledger.py -v` + a targeted `tests/agents/rlm/test_*primitive*`/persist test off-flag to prove the row is byte-identical off → PASS.
- [ ] **Step 5: STOP and report the diff.** Lead commits: `git commit -m "Add per-experiment GPU ledger + start/end/gpu_plan/retry_id row fields (display-only efficiency, never a fitness term)"`

---

### Task 4: Out-of-process re-grade ok-receipt (§6.6) — lead-implemented (load-bearing)

**Files:**
- Create: `backend/agents/rlm/ok_receipt.py`
- Modify: `backend/agents/rlm/primitives.py` (persist a receipt on success at :4959)
- Modify: `backend/agents/rlm/report.py` (`run_experiment_success_count`:1906-1926 — fallback to receipts when `ctx.cost_ledger is None`)
- Test: `tests/agents/rlm/test_ok_receipt.py`

**Interfaces:**
- Produces: `write_ok_receipt(project_dir, *, experiment_run_id, ok, metrics_sha256, ts) -> bool` (atomic append to `rlm_state/experiment_ok_receipts.jsonl`, written ONLY on a genuine in-process success so forge-resistance holds). `count_ok_receipts(project_dir) -> int` (distinct `experiment_run_id` with `ok is True` AND a non-empty `metrics_sha256`). `ok_receipt_enabled()` reads `OPENRESEARCH_OK_RECEIPT`. In `report.py`, `run_experiment_success_count(ctx)` gains a fallback: when `ctx.cost_ledger is None` (out-of-process) AND the flag is on, return `count_ok_receipts(ctx.project_dir)` instead of `None` — lifting the `partial` ceiling for a genuine out-of-process re-grade.

**Design (locked):** a receipt is minted only inside the in-process success path (beside `_persist_experiment_result`), keyed to the same `metrics_sha256` the evidence bundle uses — so a replay cannot forge one (it never ran `run_experiment`). The `evidence_bundle` alone does NOT clear the ceiling (§9); the receipt is the new, forge-resistant proof. The `report.py` change MUST pass the Track-0 tripwire (it only affects the `evidence_gate` bool input to `decide()`, never writes a verdict key).

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/rlm/test_ok_receipt.py
import json
from pathlib import Path
import pytest
from backend.agents.rlm.ok_receipt import (
    write_ok_receipt, count_ok_receipts, ok_receipt_enabled,
)

def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_OK_RECEIPT", raising=False)
    assert ok_receipt_enabled() is False

def test_write_and_count(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    (tmp_path / "rlm_state").mkdir()
    assert write_ok_receipt(tmp_path, experiment_run_id="e1", ok=True,
                            metrics_sha256="abc", ts="t") is True
    assert count_ok_receipts(tmp_path) == 1

def test_failed_or_shaless_receipt_not_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    (tmp_path / "rlm_state").mkdir()
    write_ok_receipt(tmp_path, experiment_run_id="e1", ok=False, metrics_sha256="abc", ts="t")
    write_ok_receipt(tmp_path, experiment_run_id="e2", ok=True, metrics_sha256="", ts="t")
    assert count_ok_receipts(tmp_path) == 0  # neither is a forge-proof success

def test_distinct_run_ids_counted_once(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_OK_RECEIPT", "1")
    (tmp_path / "rlm_state").mkdir()
    write_ok_receipt(tmp_path, experiment_run_id="e1", ok=True, metrics_sha256="abc", ts="t1")
    write_ok_receipt(tmp_path, experiment_run_id="e1", ok=True, metrics_sha256="abc", ts="t2")
    assert count_ok_receipts(tmp_path) == 1
```

- [ ] **Step 2: Failing** → run, confirm module missing.
- [ ] **Step 3: Implement** `ok_receipt.py`; add the flag-gated success-path write in `primitives.py:4959` (`ok=result.get("success") is True`, `metrics_sha256`/`experiment_run_id` from the enriched `result`, `ts`=`entry["timestamp"]`); add the `report.py:1906` fallback (only when `ctx.cost_ledger is None` AND flag on).
- [ ] **Step 4: Pass** — new tests + `tests/agents/rlm/test_single_verdict_authority_guard.py` (the tripwire must still pass) + `tests/rlm/test_evidence_gate_forge.py` (forge-resistance intact) → all green.
- [ ] **Step 5: Commit** — `git commit -m "Add forge-resistant out-of-process ok-receipt; lift the partial re-grade ceiling when the in-memory ledger is absent (never writes a verdict key)"`

---

### Task 5: Typed `EvaluationReport` adapter + `ScorecardRow` model (§6.1, §6.3) — lead-implemented (core)

**Files:**
- Create: `backend/evals/evaluation_report.py`
- Test: `tests/test_evaluation_report.py`

**Interfaces:**
- Produces: `class ScorecardRow(BaseModel)` = `{dimension: str, status: Literal["pass","fail","unmeasured","excluded","display"], provenance: Literal["paper_reported","agent_measured","evaluator_computed"], gates: bool, evidence_refs: list[str], detail: str}`. `class EvaluationReport(BaseModel)` = a typed SUPERSET composing (not forking) `RLMFinalReport` — carries `verdict` (copied read-only from the authoritative report, NEVER recomputed), `scorecard: list[ScorecardRow]`, `composite: float | None` (display-only, from Task 1), `provenance_bundle_sha256: str | None`, `autonomy: dict | None`, `gpu_efficiency: dict | None`. Classmethod `EvaluationReport.from_run(project_dir: Path) -> EvaluationReport` composes it from on-disk artifacts (final_report.json + rubric + result_fidelity + evidence_bundle + Task 2/3 sidecars). `to_markdown() -> str`. **Invariant: `EvaluationReport` NEVER writes `final_report.verdict`; it reads it.** A `gate_caps() -> str | None` helper returns the min (most-severe, downward-only) verdict cap implied by any `gates=True` row that is `fail`/`unmeasured` — this is a pre-`decide()` INPUT (like `claim_gate_cap`), never a post-`decide()` write.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluation_report.py
import json
from pathlib import Path
from backend.evals.evaluation_report import EvaluationReport, ScorecardRow

def _run(tmp_path, verdict="partial"):
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": verdict, "baseline_metrics": {}}))
    return tmp_path

def test_evaluation_report_copies_verdict_read_only(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path, verdict="inconclusive"))
    assert er.verdict == "inconclusive"  # copied, not recomputed

def test_scorecard_row_shape():
    r = ScorecardRow(dimension="numerical_reproduction", status="pass",
                     provenance="agent_measured", gates=True, evidence_refs=["metrics.json#accuracy"], detail="")
    assert r.gates is True and r.status == "pass"

def test_gate_cap_is_downward_only_and_never_a_verdict_write(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path, verdict="reproduced"))
    er.scorecard = [ScorecardRow(dimension="execution_completeness", status="unmeasured",
                                 provenance="evaluator_computed", gates=True, evidence_refs=[], detail="")]
    cap = er.gate_caps()
    assert cap in ("partial", "inconclusive")  # a downward cap, computed BEFORE decide(), never written back
    # the report object's verdict field is untouched by gate_caps()
    assert er.verdict == "reproduced"

def test_display_rows_never_gate(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path))
    er.scorecard = [ScorecardRow(dimension="paper_understanding", status="display",
                                 provenance="evaluator_computed", gates=False, evidence_refs=[], detail="x")]
    assert er.gate_caps() is None  # display rows contribute no cap

def test_composite_is_display_only_optional(tmp_path):
    er = EvaluationReport.from_run(_run(tmp_path))
    assert er.composite is None or isinstance(er.composite, float)  # never a verdict driver
```

- [ ] **Step 2: Failing** → run.
- [ ] **Step 3: Implement** `evaluation_report.py` — pydantic models + `from_run` (read-only composition) + `gate_caps()` (returns the most-severe downward cap from `gates=True` non-pass rows via `reproducibility_verdict._ROLLUP_ORDER`; display rows ignored) + `to_markdown()`. NO import of anything that writes a verdict.
- [ ] **Step 4: Pass** — `pytest tests/test_evaluation_report.py -v`.
- [ ] **Step 5: Commit** — `git commit -m "Add typed EvaluationReport adapter + ScorecardRow (composes RLMFinalReport read-only; gate_caps is downward-only pre-decide input)"`

---

### Task 6: Populate the 11 dimensions + wire EvaluationReport at finalize (§6.1) — lead-implemented (core)

**Files:**
- Create: `backend/evals/scorecard.py`
- Modify: `backend/agents/rlm/report.py` (`write_final_report_rlm`: after the authority stamp, flag-gated, write the sidecar + pass any gate cap as a pre-`decide()` input — NOT a post-decide write)
- Test: `tests/test_scorecard_dimensions.py`, `tests/agents/rlm/test_scorecard_offstate.py`

**Interfaces:**
- Produces: `build_scorecard(project_dir: Path) -> list[ScorecardRow]` mapping each of the 11 dimensions to a row from EXISTING deterministic signals (per the spec §6.1 table): GATE rows — numerical (`result_fidelity`), execution (`_has_experiment_evidence` + ok-receipt), environment (`env_health.jsonl` exclusions), dataset (`_detect_data_unavailable_leaves`), tables/figures (`fig_*.json` sidecars, GATE-lite); DISPLAY rows — autonomy (Task 2), efficiency (Task 3), paper-understanding (`fidelity_score_from_rubric`), DAG-planning (post-hoc S0 graph from `experiment_runs.jsonl`; upgrades to Track G S1 later), debugging (`failure_capsules.jsonl` + `FailureAttribution`), scientific-analysis (`HypothesisScore`/`IntegrityReport`). `write_evaluation_report(project_dir) -> Path | None` (writes `evaluation_report.{json,md}`, flag-gated `OPENRESEARCH_EVAL_SCORECARD`).
- The finalize wiring computes any downward gate cap BEFORE `decide()` and folds it into the existing `claim_gate_cap` path — it does NOT add a post-`decide()` writer. Off ⇒ no sidecar, no gate-cap contribution (byte-identical, proven by the tripwire + offstate test).

- [ ] **Step 1: Write failing tests** — one row-mapping test per dimension family (GATE rows key on deterministic artifacts; DISPLAY rows always `status="display"`, `gates=False`); an offstate test (flag off ⇒ `write_evaluation_report` returns None, no file, `write_final_report_rlm` byte-identical + tripwire passes). Include the §8 edge cases: missing metrics → `unmeasured` (never auto-pass), fabricated/unverifiable artifact → severe (evidence gate), a DISPLAY row can never produce a gate cap.
- [ ] **Step 2: Failing** → run.
- [ ] **Step 3: Implement** `scorecard.py` + the flag-gated finalize hook (sidecar write + pre-decide gate-cap fold). Reuse existing detectors; add NO paper-specific logic.
- [ ] **Step 4: Pass** — new tests + `test_single_verdict_authority_guard.py` (tripwire) + `test_verdict_authority_offstate.py` → all green under `-n auto`.
- [ ] **Step 5: Commit** — `git commit -m "Emit EvaluationReport scorecard at finalize (11 dimensions, deterministic gates downward-only, LLM rows display-only); flag-gated OPENRESEARCH_EVAL_SCORECARD, byte-identical off"`

---

### Task 7: Skill-as-reference composition + leniency guard (§6.4) — lead-implemented

**Files:**
- Create: `backend/evals/reference_from_skills.py`
- Test: `tests/test_reference_from_skills.py`

**Interfaces:**
- Produces: `compose_reference(project_dir: Path) -> dict` reading `rlm_state/active_skills.json` (existing skill-select machinery) → a reference `{expected_metric_families, standard_baselines, eval_protocol, dataset_expectations}` STRUCTURE only, tagged provenance `evaluator_computed`, with paper-text-span offsets where values come from `parsed_full_text.txt`. **Leniency guard:** `reference_can_flip_pass()` is structurally impossible — the reference feeds only the scorecard's `detail`/structure, never the `status`. A dedicated test asserts that injecting a skill-supplied "expected pass" cannot change any `result_fidelity` per-claim `status` (pass/fail keys solely on measured artifacts).

- [ ] **Step 1: Write failing tests** — `compose_reference` returns structure from a fixture `active_skills.json`; the **load-bearing** test: build a `repro_spec` whose claim is `unmeasured`, supply a skill reference asserting the metric "should pass", run `result_fidelity.evaluate` → status stays `unmeasured` (skill cannot flip it).
- [ ] **Step 2: Failing** → run.
- [ ] **Step 3: Implement** `reference_from_skills.py` (structure-only composition; no path from reference → claim status).
- [ ] **Step 4: Pass** — `pytest tests/test_reference_from_skills.py -v`.
- [ ] **Step 5: Commit** — `git commit -m "Compose eval reference from selected skills (structure only); test-enforce a skill can never flip a claim to pass"`

---

### Task 8: Acceptance + §8 deterministic test battery — lead-implemented

**Files:**
- Test: `tests/acceptance/test_eval_scorecard_acceptance.py`

**Interfaces:** consumes Tasks 1-7; no new production code.

**Acceptance (spec §6.8 + §8):**
- Frozen **Adam** (`runs/prj_adam_local_1`) → `EvaluationReport` with a coherent scorecard whose headline verdict stays `inconclusive` (never lifted by any display row or composite).
- A **UCPO** artifact (`runs/prj_ucpo_optA_1`) re-grade → coherent scorecard + post-hoc observed-DAG rows.
- SDAR is a stress goal, never the correctness oracle.
- §8 battery (as deterministic units where not already covered): missing metrics → `unmeasured` (never auto-pass) · multi-seed aggregation · human-intervention weighting · GPU-ledger cost aggregation · **composite never reaches the verdict surface** · **no scorecard field alters `meets_target`/`AttemptAssessment`/`campaign_policy.decide()`** · **a skill cannot flip a claim to pass** · fabricated/unverifiable artifact → severe penalty · serialization round-trip.

- [ ] **Step 1: Write the acceptance tests** driving `EvaluationReport.from_run` + `write_evaluation_report` (flag ON) on copies of the Adam + UCPO run dirs (NEVER mutate `runs/` — copy to tmp_path); assert the coherent scorecard + unchanged verdict + the §8 invariants.
- [ ] **Step 2: Run** — `pytest tests/acceptance/test_eval_scorecard_acceptance.py -v`.
- [ ] **Step 3: Full suite + lint** — `.venv/bin/python -m pytest tests/ -n auto` and `uvx ruff@0.15.16 check .` green; confirm every off-state pair byte-identical.
- [ ] **Step 4: Commit** — `git commit -m "Acceptance: Adam/UCPO EvaluationReport scorecards coherent + verdict-preserving; §8 deterministic battery"`

---

## Self-review (spec coverage)

- §6.1 dimension contract → Tasks 5-6. §6.2 composite → Task 1. §6.3 data models → Tasks 5-6 (adapter/view, no fork). §6.4 skills reference → Task 7. §6.5 HumanIntervention + efficiency → Tasks 2-3. §6.6 ok-receipt → Task 4. §6.7 verdict levels → unchanged (authority owns them; EvaluationReport copies read-only). §6.8 acceptance → Task 8. §8 tests → Task 8. Non-goals honored: no composite-gates-verdict (Task 1 guard), no rubric-surface write (Global Constraints + tripwire), no new DECIDE signal (Task 8 regression).
- **Deferred to Track G:** the observed-DAG row starts S0-derived post-hoc (Task 6) and upgrades to S1-recorded when Track G lands — no throwaway coupling.
- **Delegation split (guarded hybrid):** lead implements Tasks 1, 4, 5, 6, 7, 8 (evaluator-core / verdict-adjacent / load-bearing); Tasks 2, 3 are delegatable to guarded Sonnet (new-file + fenced-hook, no verdict surface). Task 0 is doc-only.
