# Track A — Eval Integrity Implementation Plan (WS1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the reproduction verdict in exactly one grade-free place keyed on deterministic result-fidelity; demote the LLM grade to a diagnostic (`impl_fidelity`) that can never move the headline verdict.

**Architecture:** Three new pure stdlib modules (`metric_binding`, `result_fidelity`, `verdict_authority`) sit on top of the *already-built* claim extractor (`repro_spec_extractor`) and verdict engine (`reproducibility_verdict`/`two_axis_report`). `VerdictAuthority.decide` runs **once, last, unconditionally** at the finalize chokepoint (`report.write_final_report_rlm`) and stamps every verdict surface; every prior grade→verdict writer is either severed or converted to a pre-authority input. All behind `OPENRESEARCH_VERDICT_AUTHORITY` (new sub-flag) under the existing `OPENRESEARCH_TWO_AXIS_VERDICT` master; byte-identical when off.

**Tech Stack:** Python ≥3.11, stdlib-only for the three new modules (no LLM, no network), pytest (socket-hermetic), existing `grader_transport` for the grader-fallback task.

## Global Constraints

- **North-star invariant:** the verdict keys on the deterministic evidence layer, never the LLM grade. `VerdictAuthority.decide` takes **no** grade/`impl_fidelity` argument — the sever is structural, not a convention.
- **Flag idiom (default-OFF):** `os.environ.get("OPENRESEARCH_VERDICT_AUTHORITY","").strip().lower() in ("1","true","yes","on")`. Byte-identical when off or explicitly `=0`.
- **Master gate:** the authority only runs when BOTH `OPENRESEARCH_TWO_AXIS_VERDICT` (existing) and `OPENRESEARCH_VERDICT_AUTHORITY` (new) are truthy.
- **No false `contradicted`:** a `fail`/`contradicted` requires a scope-verified metric bind; any unverifiable/ambiguous bind stays `unmeasured` (asymmetry — a false contradiction is the worst error).
- **Diagnostic dual-write:** keep `rubric_score` as a back-compat alias of the new `impl_fidelity` for one release.
- **Reuse, don't rebuild:** the taxonomy severity order already exists as `reproducibility_verdict._ROLLUP_ORDER` (`contradicted` < `inconclusive`) — reuse it; do not reimplement a parallel ordering.
- **Corrected acceptance (see review):** on the **frozen** on-disk `runs/prj_adam_local_1` the ruler holds only one *ambiguous* primary claim, so the honest re-grade is **`inconclusive`**, not the `partial` the source spec §7 states. `partial` is exercised via a synthetic fixture where a *real* primary claim is unmeasured.

**Real code map (verified 2026-07-10; line numbers may drift ±10 — match by symbol):**
- `backend/agents/rlm/repro_spec_extractor.py`: `extract_and_write`:841, `seed_bundle_from_metrics`:502, `_extract_metric_value`:575, claim schema fields:428-443, auto-promote-primary:982.
- `backend/agents/rlm/two_axis_report.py`: `fidelity_score_from_rubric`:69, `load_claims`:161, `compute_and_attach`:271, **grade→headline overwrite** `report["verdict"]=verdict.legacy_verdict`:308-309.
- `backend/agents/rlm/reproducibility_verdict.py`: `compute_reproducibility_verdict`:437, `_legacy_verdict_from_fidelity`:429-434, `_ROLLUP_ORDER`:539-541, `_rollup_primaries`:544-546.
- `backend/agents/rlm/report.py`: `_VERDICT_REPRODUCED_MIN_SCORE=0.60`:264, `_reconcile_verdict`:989, evidence gate:1551, two-axis attach + upgrade-clamp:2160-2198, `report_claim_gate` call (post-attach mutation):2202-2231, `write_final_report_rlm` = finalize chokepoint.
- `backend/agents/rlm/run.py`: lifecycle verdict projector:1286-1300, `regrade_and_emit` call:4819, `write_final_report_rlm` call:4854, terminal `_write_demo_status` (no verdict passed):4920, demo_status default-derivation:1070.
- `backend/agents/rlm/finalize_regrade.py`: grade→verdict `report.verdict = reconcile_verdict_with_score(...)`:332.
- `backend/evals/paperbench/leaf_scorer.py`: `_grade_batch` silent all-zero on exception:1928-1952, `amend_final_report` `_attach_two_axis`:2357 + `report["verdict"]=reconcile_verdict_with_score(...)`:2369.
- `scripts/score_run.py`: `amend_final_report(run_dir, score)`:103.
- `backend/agents/rdr/controller.py`: grade→verdict:1417, shared `write_final_report_rlm`:1483.
- `backend/agents/rlm/campaign_policy.py`: REPRODUCED gate on `meets_target`:787, escalation gate:629. `backend/agents/rlm/attempt_assessment.py`: `meets_target`:332.
- `backend/agents/rlm/report_claim_gate.py`: caps to `partial`:100, "FINAL verdict mutation" docstring:9.
- `backend/agents/rlm/grader_transport.py`: single-swap builders only, **no fallback list today**.
- `backend/agents/rlm/leaderboard.py`: reads `reproduction_status` alias:2156.

---

### Task 1: `metric_binding` — bind prose `metric_name` → measurement path (closes D2)

**Files:**
- Create: `backend/agents/rlm/metric_binding.py`
- Test: `tests/agents/rlm/test_metric_binding.py`

**Interfaces:**
- Consumes: `repro_spec` dict (claims with `metric_name`, `scope{model,dataset,split}`, `direction`), `metrics` dict (the parsed `code/metrics.json`). Reuses the metric-key tokenizer already in `leaf_scorer` (import it; do not re-tokenize differently).
- Produces: `bind_claims(repro_spec: dict, metrics: dict) -> dict` — returns `repro_spec` with each claim gaining a `metric_binding: {metric_key, model_key?, env_key?, baseline_key?, agg?, bound: bool, reason}` object. Consumed by Task 2.

**Design (locked):** deterministic tokenized match first; a candidate path is **ACCEPTED only** when a pure check confirms the resolved scope keys (`model_key`/`env_key`/`baseline_key`/split) match the claim's declared `scope` AND the metric unit/`direction` are consistent. Ambiguous or scope-mismatched → `bound=False` (stays unmeasured). "Path exists in metrics.json" is necessary, not sufficient. (LLM-proposed fallback bind is **deferred** to a follow-up — the deterministic gate is the load-bearing piece; note it as a documented seam, do not build it here — YAGNI.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/rlm/test_metric_binding.py
from backend.agents.rlm.metric_binding import bind_claims

_METRICS = {"per_model": {"adam": {"mnist": {"test": {"accuracy": 0.991}}}},
            "accuracy": 0.991}

def test_unambiguous_scope_match_binds_path():
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "accuracy",
                        "direction": "higher_is_better",
                        "scope": {"model": "adam", "dataset": "mnist", "split": "test"}}]}
    out = bind_claims(spec, _METRICS)
    b = out["claims"][0]["metric_binding"]
    assert b["bound"] is True
    assert b["metric_key"] == "accuracy" and b["model_key"] == "adam" and b["env_key"] == "mnist"

def test_scope_mismatch_does_not_bind():
    # path 'accuracy' exists but the claim's split is 'train' — must NOT bind to the test-split number
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "accuracy",
                        "direction": "higher_is_better",
                        "scope": {"model": "adam", "dataset": "mnist", "split": "train"}}]}
    out = bind_claims(spec, _METRICS)
    assert out["claims"][0]["metric_binding"]["bound"] is False

def test_ambiguous_metric_name_does_not_bind():
    metrics = {"loss_a": 0.1, "loss_b": 0.2}
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "loss", "scope": {}}]}
    out = bind_claims(spec, metrics)
    assert out["claims"][0]["metric_binding"]["bound"] is False

def test_missing_metric_stays_unbound_never_raises():
    spec = {"claims": [{"claim_id": "primary_0", "metric_name": "perplexity", "scope": {}}]}
    out = bind_claims(spec, _METRICS)
    assert out["claims"][0]["metric_binding"]["bound"] is False
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/agents/rlm/test_metric_binding.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement `bind_claims`** — deterministic tokenized candidate search over the `metrics` key tree (reuse `leaf_scorer`'s tokenizer), then the scope+unit acceptance gate. On any ambiguity/mismatch/missing, set `{"bound": False, "reason": ...}`; never raise. Write the accepted bind back onto the claim.
- [ ] **Step 4: Run to verify pass** — `pytest tests/agents/rlm/test_metric_binding.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add backend/agents/rlm/metric_binding.py tests/agents/rlm/test_metric_binding.py && git commit -m "Add deterministic metric_binding (claim prose metric_name -> scope-verified metrics.json path)"`

---

### Task 2: `result_fidelity` — the deterministic per-claim checker (§4.2)

**Files:**
- Create: `backend/agents/rlm/result_fidelity.py`
- Test: `tests/agents/rlm/test_result_fidelity.py`

**Interfaces:**
- Consumes: `bind_claims` (Task 1); `repro_spec_extractor.seed_bundle_from_metrics` to read the measured value at the bound path; `code/metrics.json` under `run_dir`.
- Produces: `evaluate(repro_spec: dict, run_dir: Path) -> ResultFidelity` where `ResultFidelity` is a dict: `{"per_claim": [{"claim_id","kind","status": "pass"|"fail"|"unmeasured","measured","target","margin","reason"}], "result_fidelity_score": float, "primary_all_measured": bool, "any_contradicted": bool}`. Consumed by Task 3.

**Design (locked):** per-kind deterministic test — **numeric** `abs(measured-target) <= equivalence_margin`; **relative** ordering + ratio/effect within margin on the sign-folded `claimed_effect`; **trend** sign/slope of `history.*` series over the claimed window; **qualitative / ambiguous / unbound** → `unmeasured` (never auto-pass). `result_fidelity_score` = fraction of PRIMARY claims that `pass` (secondary weighted lower). Pure, no LLM.

- [ ] **Step 1: Write the failing tests** (one per kind + the asymmetry)

```python
# tests/agents/rlm/test_result_fidelity.py
from pathlib import Path
import json
from backend.agents.rlm.result_fidelity import evaluate

def _run(tmp_path, metrics):
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)
    (tmp_path / "code" / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path

def _claim(**kw):
    base = {"claim_id": "primary_0", "is_primary": True, "kind": "numeric",
            "metric_name": "accuracy", "estimate_kind": "point",
            "claimed_effect": 0.99, "equivalence_margin": 0.01,
            "direction": "higher_is_better", "scope": {}, "ambiguous": False}
    base.update(kw); return base

def test_numeric_within_margin_passes(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.991})
    rf = evaluate({"claims": [_claim()]}, run)
    assert rf["per_claim"][0]["status"] == "pass" and rf["any_contradicted"] is False

def test_numeric_outside_margin_fails_only_with_verified_bind(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.80})
    rf = evaluate({"claims": [_claim()]}, run)
    assert rf["per_claim"][0]["status"] == "fail" and rf["any_contradicted"] is True

def test_ambiguous_claim_is_unmeasured_never_fail(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.10})
    rf = evaluate({"claims": [_claim(ambiguous=True)]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured" and rf["any_contradicted"] is False

def test_unbound_metric_is_unmeasured(tmp_path):
    run = _run(tmp_path, {"other": 1.0})
    rf = evaluate({"claims": [_claim(metric_name="nope")]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/agents/rlm/test_result_fidelity.py -v` → FAIL.
- [ ] **Step 3: Implement `evaluate`** — call `bind_claims`, read via `seed_bundle_from_metrics` at bound paths, apply per-kind tests, aggregate. Reuse the extractor's sign-folding for `relative`.
- [ ] **Step 4: Run to verify pass** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add result_fidelity deterministic per-claim checker (numeric/relative/trend/unmeasured)"`

---

### Task 3: `verdict_authority` — the single grade-free reconciliation point (§4.3)

**Files:**
- Create: `backend/agents/rlm/verdict_authority.py`
- Test: `tests/agents/rlm/test_verdict_authority.py`

**Interfaces:**
- Consumes: `result_fidelity` (Task 2 output dict); `evidence_gate` (bool/struct: ≥1 success-compatible in-process `run_experiment` ledger row + real on-disk metrics); `fidelity_certificate`; optional `claim_gate_cap` (str verdict ceiling), `ruler_quality` (Spec-B seam, defaults `"trusted"`). Reuses `reproducibility_verdict._ROLLUP_ORDER` for severity.
- Produces: `decide(*, result_fidelity, evidence_gate, fidelity_certificate, claim_gate_cap=None, ruler_quality=None) -> Verdict` where `Verdict = {"verdict": "reproduced"|"contradicted"|"partial"|"inconclusive", "reason": str}`. **No grade/`impl_fidelity` parameter.** Plus `freeze_contract(run_dir: Path) -> Path` (Spec-C seam; writes/returns the frozen `repro_spec.json` path).

**Taxonomy precedence (locked):** `inconclusive(no_measurable_target) → contradicted → partial → reproduced`. Rule 1: no genuine `is_primary` claim, or only `ambiguous`/unbound primaries → `inconclusive`; the extractor's auto-promoted first claim (`repro_spec_extractor.py:982`) is **ignored for the verdict**. Rule 2: any primary `fail` → `contradicted`. Rule 3: any primary `unmeasured` (none fail) → `partial`. Rule 4: all primaries `pass` on `eval_split` AND `evidence_gate` satisfied → `reproduced`. `claim_gate_cap` caps the result downward only.

- [ ] **Step 1: Write the failing tests** (precedence matrix + the auto-promote-ignored rule)

```python
# tests/agents/rlm/test_verdict_authority.py
from backend.agents.rlm.verdict_authority import decide

def _rf(per_claim, **kw):
    return {"per_claim": per_claim, "result_fidelity_score": kw.get("score", 0.0),
            "primary_all_measured": kw.get("all_measured", False),
            "any_contradicted": any(c["status"] == "fail" for c in per_claim)}

def _c(status, primary=True, ambiguous=False):
    return {"claim_id": "x", "status": status, "is_primary": primary, "ambiguous": ambiguous}

def test_no_measurable_primary_is_inconclusive():
    v = decide(result_fidelity=_rf([_c("unmeasured", ambiguous=True)]),
               evidence_gate=True, fidelity_certificate=None)
    assert v["verdict"] == "inconclusive" and v["reason"] == "no_measurable_target"

def test_any_primary_fail_is_contradicted_even_with_unmeasured_sibling():
    v = decide(result_fidelity=_rf([_c("fail"), _c("unmeasured")]),
               evidence_gate=True, fidelity_certificate=None)
    assert v["verdict"] == "contradicted"

def test_mixed_pass_unmeasured_none_fail_is_partial():
    v = decide(result_fidelity=_rf([_c("pass"), _c("unmeasured")]),
               evidence_gate=True, fidelity_certificate=None)
    assert v["verdict"] == "partial"

def test_all_primary_pass_with_evidence_is_reproduced():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=True, fidelity_certificate=object())
    assert v["verdict"] == "reproduced"

def test_all_pass_but_no_evidence_is_not_reproduced():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=False, fidelity_certificate=None)
    assert v["verdict"] != "reproduced"

def test_claim_gate_cap_only_lowers():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=True, fidelity_certificate=object(), claim_gate_cap="partial")
    assert v["verdict"] == "partial"
```

- [ ] **Step 2: Run to verify failure** — → FAIL.
- [ ] **Step 3: Implement `decide` + `freeze_contract`** — pure precedence over primaries; import `_ROLLUP_ORDER` for severity; apply `claim_gate_cap` as a downward clamp; `reproduced` requires `evidence_gate` truthy. `decide` must accept a missing/empty `result_fidelity` (RDR/legacy) → return `inconclusive` (never pass through a grade-derived verdict).
- [ ] **Step 4: Run to verify pass** — → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add VerdictAuthority.decide: single grade-free reproduction verdict with explicit taxonomy precedence"`

---

### Task 4: D1 fix — `fidelity_score_from_rubric` flat-`sub_tasks` path (diagnostic only)

**Files:**
- Modify: `backend/agents/rlm/two_axis_report.py:69-96`
- Test: `tests/agents/rlm/test_two_axis_report.py` (add cases)

**Interfaces:**
- Produces: `fidelity_score_from_rubric` returns a real weighted score for a flat `sub_tasks` rubric (no `areas`). **This feeds only the `implementation_verdict`/`impl_fidelity` diagnostic — never the headline verdict (Task 6 severs that).**

- [ ] **Step 1: Write the failing test**

```python
def test_flat_sub_tasks_rubric_scores_by_leaf_weight():
    from backend.agents.rlm.two_axis_report import fidelity_score_from_rubric
    rubric = {"sub_tasks": [
        {"score": 1.0, "weight": 2.0, "sub_tasks": []},
        {"score": 0.0, "weight": 1.0, "sub_tasks": []}]}
    assert abs(fidelity_score_from_rubric(rubric) - (2.0 / 3.0)) < 1e-9
```

- [ ] **Step 2: Run to verify failure** — currently falls back to `overall_score`→0.0 → FAIL.
- [ ] **Step 3: Implement** — when `areas` is absent/empty, walk the `sub_tasks` leaf tree weighting by per-leaf `weight`; keep the `areas` path untouched.
- [ ] **Step 4: Run to verify pass** — → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Fix D1: fidelity_score_from_rubric scores flat sub_tasks rubrics (diagnostic axis only)"`

---

### Task 5: Grader hardening — cross-provider fallback, never silent-zero (§4.5)

**Files:**
- Modify: `backend/agents/rlm/grader_transport.py` (add ordered fallback list)
- Modify: `backend/evals/paperbench/leaf_scorer.py:1928-1952` (`_grade_batch`)
- Test: `tests/evals/paperbench/test_leaf_scorer_fallback.py`

**Interfaces:**
- Produces: `grader_transport.build_fallback_chain() -> list[TransportClient]` (ordered, e.g. Foundry-Sonnet → grok → oauth-Sonnet). `_grade_batch` on a parse/SDK failure retries down the chain before scoring; a leaf ungradable after the whole chain is marked `ungraded` (excluded from the diagnostic) with a loud `grader_unavailable` `run_warning` — **never silently `0.0`**.

- [ ] **Step 1: Write the failing test** — primary transport raises → chain falls back and grades; whole chain fails → leaves are `ungraded` (not `0.0`) and a warning is emitted. Use fakes for the transports (socket-hermetic).
- [ ] **Step 2: Run to verify failure** — → FAIL (today defaults to `0.0`).
- [ ] **Step 3: Implement** — add the fallback chain to `grader_transport`; wrap `_grade_batch`'s `except` to iterate the chain, then mark `ungraded`.
- [ ] **Step 4: Run to verify pass** — → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Harden grader: in-run cross-provider fallback; ungradable leaves marked ungraded, never silent 0.0"`

---

### Task 6: The sever — VerdictAuthority as the single, last, unconditional verdict writer (§4.3)

**Files:**
- Modify: `backend/agents/rlm/report.py` (`write_final_report_rlm`: insert authority as last writer; reorder `report_claim_gate`→pre-authority `claim_gate_cap`)
- Modify: `backend/agents/rlm/two_axis_report.py:308-309` (stop writing `report["verdict"]`)
- Modify: `backend/agents/rlm/reproducibility_verdict.py:429-434` (keep the two *diagnostic* axes; delete the headline projection)
- Modify: `backend/agents/rlm/finalize_regrade.py:332`, `backend/evals/paperbench/leaf_scorer.py:2357-2369` + `scripts/score_run.py:103`, `backend/agents/rdr/controller.py:1417` (refresh diagnostics only; never mint the reproduction verdict)
- Create: `tests/agents/rlm/test_single_verdict_authority_guard.py`

**Interfaces:**
- Consumes: Tasks 2–3. Produces: exactly one function stamps the reproduction verdict; a static + runtime guard asserts no module writes `report["verdict"]`/`demo_status["verdict"]`/`replication_verdict`/`implementation_verdict` **after** `decide()`.

**Design (locked — this is the crux; the review found the sever surface is wider than the source spec's two paths):**
1. In `write_final_report_rlm`, when the flag is on, call `VerdictAuthority.decide(...)` **unconditionally** (independent of two-axis-attach success or `repro_spec` presence) as the final step, and stamp all surfaces atomically (`report.verdict`, `implementation_verdict`, `replication_verdict`, and mirror into `demo_status.verdict` — which today is never stamped and defaults wrong).
2. Compute `report_claim_gate`'s cap **before** `decide` and pass it as `claim_gate_cap`; remove its post-attach mutation (`report.py:2202-2231`).
3. Sever the grade→headline writers: `two_axis_report:309` no longer assigns `report["verdict"]`; `reproducibility_verdict._legacy_verdict_from_fidelity` stops feeding the headline (keeps the two diagnostic axes); `finalize_regrade:332`, `leaf_scorer.amend_final_report:2369`, and `rdr/controller:1417` refresh only `impl_fidelity`/rubric diagnostics — the authority (or its RDR `inconclusive` default) owns the verdict.
4. The `0.60` `_reconcile_verdict` classifier feeds only the `impl_fidelity` diagnostic, not the headline.

- [ ] **Step 1: Write the failing guard + acceptance tests**

```python
# tests/agents/rlm/test_single_verdict_authority_guard.py
# (1) runtime guard: instrument write_final_report_rlm so any post-decide write to a
#     reproduction-verdict key raises; assert a synthetic post-authority mutation is caught.
# (2) static guard: grep the finalize call graph — no module in the post-decide set assigns
#     report["verdict"]. (3) grade-severance: with fixed deterministic artifacts, perturbing
#     the LLM grade / simulating a grader outage does NOT change report["verdict"].
```

- [ ] **Step 2: Run to verify failure** — → FAIL (multiple writers still mint the verdict).
- [ ] **Step 3: Implement the sever** per the locked design above; add the runtime assertion in `write_final_report_rlm`.
- [ ] **Step 4: Run to verify pass** — `pytest tests/agents/rlm/test_single_verdict_authority_guard.py -v` → PASS; full verdict suite green.
- [ ] **Step 5: Commit** — `git commit -m "Sever grade->verdict: VerdictAuthority is the single, last, unconditional reproduction-verdict writer (+ guard test)"`

---

### Task 7: Campaign consumers read the authority, not the grade (§4.3)

**Files:**
- Modify: `backend/agents/rlm/campaign_policy.py:787` (REPRODUCED gate), `:629` (escalation gate)
- Test: `tests/agents/rlm/test_campaign_policy.py` (add case)

**Interfaces:**
- Produces: the campaign `REPRODUCED`/escalation gates require the authoritative `verdict == "reproduced"` **AND** the existing `meets_target` (AND, not OR-replace) — a `meets_target=True` but `verdict=inconclusive` attempt cannot terminate the campaign as reproduced.

- [ ] **Step 1: Write the failing test** — an attempt with `meets_target=True, verdict="inconclusive"` does NOT yield `kind="REPRODUCED"`.
- [ ] **Step 2: Run to verify failure** — → FAIL (today keys on `meets_target` alone).
- [ ] **Step 3: Implement** — AND-condition `verdict=="reproduced"` onto both gates.
- [ ] **Step 4: Run to verify pass** — → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Campaign terminal/escalation gates require authoritative verdict==reproduced, not grade-derived meets_target"`

---

### Task 8: Flag plumbing + off-state byte-identical + diagnostic dual-write

**Files:**
- Modify: `backend/agents/rlm/report.py` (flag read helper), the finalize call site
- Modify wherever `rubric_score` is written to also write `impl_fidelity` (dual-write one release)
- Test: `tests/agents/rlm/test_verdict_authority_offstate.py`

**Interfaces:**
- Produces: `OPENRESEARCH_VERDICT_AUTHORITY` default-OFF; the authority path runs only when it AND `OPENRESEARCH_TWO_AXIS_VERDICT` are truthy. `final_report.json` gains `result_fidelity{per_claim,score}` and `impl_fidelity` (alias `rubric_score`).

- [ ] **Step 1: Write the failing test** — with `OPENRESEARCH_VERDICT_AUTHORITY=0`, `write_final_report_rlm` output is byte-identical to pre-change on a fixture run.
- [ ] **Step 2: Run to verify failure/pass** — establish the golden then assert equality.
- [ ] **Step 3: Implement** — gate every new code path on the flag; dual-write `impl_fidelity`/`rubric_score`.
- [ ] **Step 4: Run to verify pass** — → PASS; run `uvx ruff@0.15.16 check .`.
- [ ] **Step 5: Commit** — `git commit -m "Gate VerdictAuthority behind OPENRESEARCH_VERDICT_AUTHORITY (default-OFF, byte-identical); dual-write impl_fidelity"`

---

### Task 9: End-to-end acceptance on the Adam artifact + synthetic taxonomy fixtures

**Files:**
- Test: `tests/acceptance/test_adam_verdict_reground.py`

**Interfaces:** consumes all prior tasks; no new production code.

**Acceptance (corrected per review):**
- On the **frozen** `runs/prj_adam_local_1` (ruler = one *ambiguous* primary), re-running the verdict with the flag ON yields **`verdict == "inconclusive"`** (`reason="no_measurable_target"`) — NOT `reproduced`. If it still says `reproduced`, WS1 failed.
- Synthetic fixture with a *real* primary claim left unmeasured → `partial`.
- Synthetic fixture with a measured primary violating the claim → `contradicted`.
- Synthetic fixture with a measured primary passing + evidence gate → `reproduced`.

- [ ] **Step 1: Write the acceptance test** driving `write_final_report_rlm` (flag ON) on a copy of the Adam run dir + the three synthetic dirs; assert the four verdicts above.
- [ ] **Step 2: Run** — `pytest tests/acceptance/test_adam_verdict_reground.py -v`. Expected: the Adam case asserts `inconclusive`.
- [ ] **Step 3: Full suite + lint** — `.venv/bin/python -m pytest tests/ -n auto` and `uvx ruff@0.15.16 check .` green; confirm off-state suites unchanged.
- [ ] **Step 4: Commit** — `git commit -m "Acceptance: Adam re-grades reproduced->inconclusive under VerdictAuthority; taxonomy synthetics for partial/contradicted/reproduced"`

---

## Follow-on plans (sequenced; not in this plan)

- **WS2 — Track D flag promotion + default GCP path.** Re-tiered per review: `IMPL_ABANDON_GUARD`, `GKE_SYNTH_CELL`, `CELL_RESUME_AUTO` move to audited/A-B (not Tier-0); `PREFLIGHT_UNION_SCOPE` gets a population false-block audit; the Tier-1 vetoes flip only after WS1's authority lands + the σ-gate (make `REQUIRE_STAMPED_AB` CI-enforced). Guarded monolithic `k8s_job_backend.exec`; cell-synth default.
- **WS3 — Cloud-native durable driver (needs a DESIGN PASS first).** Close the 3 review blockers before bite-sizing: (a) add generation-aware writes to `gcs_blob` (`if_generation_match`) as the CAS primitive the lease needs; (b) controller self-heal (`restartPolicy=OnFailure`/supervisor + reaper of prior-generation Jobs) so controller death ≠ run death; (c) fencing — stamp the lease generation into Job names + blob paths + deterministic `(run_id,cell_id,attempt)` adopt-by-name — so a superseded owner can't race the evidence bus or leak A100s.
- **WS4 — CPU-class durable lane.** `gpu_count=0` cloud path under WS3 + routing so CPU-class papers run unattended on cloud, not the laptop.
- **WS5 — Repo-first grounding default-on.** Default the author-repo grounding path where a paper links code.
