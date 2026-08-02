# Tier-3 Sub-project B — Wire the scheduler to APPLY (local-first) Implementation Plan

> **STATUS 2026-08-01 — EXECUTED (2026-07-22/23).** The checkboxes below are the as-authored
> roadmap and were not ticked during execution. All 8 tasks (B1.1, B1.2, B2.1+B2.2, B2.3,
> B3.2, B3.3, and the Phase C checkpoint/resume substrate) landed; 506 tests green. Ground truth:
> `docs/progress/2026-07-22-tier3-adam-progress.md` and the spec's status banner. Phase C
> (billed ADAM A/B on real GPU) remains gated on operator GPU budget.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Every task here carries logic next to the evidence red line — run the FULL two-stage review (spec compliance, then code quality) on each task.**

**Goal:** Make a real `campaign --sandbox local` run with `OPENRESEARCH_SCHEDULER_TREE=1` + `OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1` drive a multi-branch cohort through the receipt-gated `SchedulerAuthorityController` — freezing an underperformer, promoting winners, and reviving a frozen branch — all from **verified receipts** built off the deterministic evidence layer, while the same invocation with the flags OFF stays **byte-identical** to today.

**Architecture:** The authority tree (`SchedulerAuthorityController` + `SchedulerTreeRuntime` + `scheduler_evidence`) is already built and hermetically tested; it has ZERO campaign call sites. Three layers wire it in: **B1** a harness-owned receipt producer that materializes a real 5-field checkpoint + evidence bundle from a completed local cell and assembles the exact `raw_receipt` dict; **B2** constructs the controller in `build_campaign` (flag+spec gated, byte-identical-OFF) and resolves the `branch-tree:<id>` double-writer; **B3** a flag-gated `_cohort_loop` the campaign dispatches to under authority (serial `_loop` untouched for OFF), reachable from a real `campaign` entrypoint, driving claim→run→receipt→decide→apply→revive.

**Tech Stack:** Python 3.11+, pytest (socket-hermetic — CPU-only stub cells, no GPU/network), the existing scheduler hermetic suite as the contract oracle.

---

## Scope / exit criterion — read first

**Exit for Phase B (the honest check):** a real `campaign --sandbox local` invocation over **≥2 tiny CPU-stub branches** with **both** `OPENRESEARCH_SCHEDULER_TREE=1` and `OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1` produces run artifacts showing **one branch frozen, one promoted, and one revived** from **verified receipts** — AND the *same* invocation with the flags unset is **byte-identical** to today's serial output. If B3 can't be reached from a `campaign` entrypoint, B is not done. (The controller's own hermetic tests remain the machinery oracle; B adds the campaign-reachable wiring + a real-cell receipt producer.)

**RED LINES (all four recon agents + advisor flagged these — they are REQUIRED tests, not notes):**
1. **Evidence-not-grade.** The receipt `metric_value` MUST be read from the cell's `metrics.json[ladder.metric_id]`, NEVER `final_report.score` or any LLM grade, and NEVER the shadow adapter's `observation_from_assessment` (which reads `final_report.score`). `write_verified_receipt` only checks hash *consistency* — it cannot tell a deterministic metric from a dumped grade, so the discipline lives entirely in the producer. **Required test:** plant `final_report.score: 0.01` in the cell metrics (mirror the controller test's `_raw_receipt`) and assert the receipt's `metric_value` comes from `metrics.json[metric_id]` and the grade is never consulted.
2. **Never bypass the deterministic terminal.** `_decide_impl` computes `campaign_decide(...)` FIRST; authority runs after and may only reorder the NEXT cohort's continue-space. It must preserve every base decision key and every terminal stop. A promote/freeze/kill can never flip or suppress a terminal verdict. (`test_authority_preserves_every_terminal_decision` already encodes this — keep it green.)
3. **Byte-identical-OFF.** Constructing `SchedulerAuthorityController` side-effects the run dir at `__init__` (writes `scheduler_ladder.json` + `scheduler_tree_state.json`, mkdirs `campaign/`). Gate construction behind BOTH flags AND a present `authority_spec_path`. OFF ⇒ no controller, no receipt files, no `attempts.jsonl` `scheduler_receipt` rows, no new SSE/decision keys. Proven by an OFF-pair test.
4. **`training_diverged`-only true-kill.** Do NOT extend `asha_campaign_adapter._BREAKAGE_CLASSES` (frozen `{"training_diverged"}`). Every other cause is a reversible FREEZE. The producer sets `termination_cause` only from the deterministic failure classifier.
5. **Fail-closed on absent/forged receipt.** No verified receipt for a rung ⇒ preserve the base campaign decision and keep `applied:false`. `load_verified_receipt` returns `None` on any mismatch; the wiring must honor that `None`, never default it. A CPU stub with NO real checkpoint file cannot mint a receipt (`write_verified_receipt` raises) — do NOT add a fabricated/zero checkpoint to make a demo pass.
6. **`gpu_cell_runner` stays import-clean.** It is copied verbatim into the agent sandbox (stdlib-only). The receipt emission lives at the CALLER (`primitives.py::_execute_cell_matrix`), never inside `gpu_cell_runner`. Any new import in `gpu_cell_runner` is a red line.
7. **Real provider GPU-$.** `decide_rung` requires `provider_gpu_usd_by_branch` and won't substitute wall-clock/grade. Feed it the deterministic per-branch `AttemptAssessment.cost.gpu_usd` (computed from `rlm_state/gpu_plan.json` rate × gpu_hours), never the Foundry-blind `cost_ledger.jsonl` GPU column. (For the CPU-stub demo, the spec supplies explicit small per-branch GPU-$ so width arithmetic is exercised deterministically.)

**The exact `raw_receipt` dict shape** (verbatim from `scheduler_evidence._receipt_from_mapping`; the producer MUST emit exactly this — a paraphrase yields a silent `None` receipt):
```python
{
  "schema_version": 1,
  "campaign_id": str, "branch_id": str, "parent_branch_id": str | None,
  "attempt_n": int, "cell_id": str, "paper_ref": str,
  "ladder_sha256": str, "from_step": int, "to_step": int, "seed": int,
  "termination_cause": str | None,
  "metric":  {"id": str, "direction": "maximize"|"minimize", "value": float,
              "artifact_path": "<rel-to-run_dir>", "sha256": str},          # value == json.load(artifact)[id]
  "checkpoint": {"path": "<rel-to-run_dir>", "sha256": str,                  # sha256 of the checkpoint file bytes
                 "state": {"model_sha256": str, "optimizer_sha256": str,      # EXACTLY these 5 keys, all sha256-hex
                           "lr_scheduler_sha256": str, "rng_sha256": str, "data_order_sha256": str},
                 "state_path": "<rel-to-run_dir>", "state_sha256": str},      # sha256 of the state manifest file
  "evidence_bundle": {"path": "<rel-to-run_dir>", "sha256": str},            # bundle JSON: schema:1, coherent:true,
                                                                             #   metrics_sha256==metric.sha256,
                                                                             #   code_tree_digest==fingerprints.code_sha256
  "fingerprints": {"code_sha256": str, "dataset_sha256": str, "dataset_manifest_path": "<rel>",
                   "run_spec_sha256": str, "run_spec_path": "<rel>"},
}
```
All `*_path` values are **relative to `run_dir`**, artifacts must be real files under `run_dir` (symlinks rejected), and `write_verified_receipt` re-hashes each. The `attest` callback MUST be the controller's ledger appender (`CampaignLedger.append_row`), never the worker's.

---

## File Structure

| File | Responsibility | Part |
|---|---|---|
| `backend/agents/rlm/scheduler_receipt_producer.py` (NEW) | Harness-owned: build the exact `raw_receipt` dict from a completed cell's `metrics.json` + a 5-field checkpoint dir + fingerprints. No campaign imports beyond `scheduler_evidence` types. | B1 |
| `tests/rlm/test_scheduler_receipt_producer.py` (NEW) | Round-trip through `write_verified_receipt`; evidence-not-grade; fail-closed on missing checkpoint | B1 |
| `backend/agents/rlm/campaign_composition.py` | `CampaignOptions.authority_spec_path`; `build_campaign` constructs the controller (gated); grow `_maybe_apply_asha_authority`; suppress the serial branch-spawn emit under authority | B2 |
| `backend/agents/rlm/reproduction_campaign.py` | Inject `scheduler_controller`; add the flag-gated `_cohort_loop` branch (serial `_loop` untouched) | B2, B3 |
| `backend/agents/rlm/primitives.py` | Flag-gated receipt emission at `_execute_cell_matrix` after `run_matrix` returns | B3 |
| `tests/rlm/test_campaign_composition.py`, `test_authority_controller_wiring.py` (NEW), `test_campaign_cohort_loop.py` (NEW) | OFF-byte-identical + construction + cohort drive | B2, B3 |
| `tests/rlm/test_local_freeze_promote_revive.py` (NEW) | **The Phase B exit test** — campaign-reachable, real CPU-stub cells, freeze/promote/revive from verified receipts | B3 |

**Test-support (CPU stub):** a tiny `train_cell.py`-shaped stub that writes `metrics.json` (with `metric_id` value + a planted `final_report.score`) and a 5-component checkpoint dir (`model/optimizer/lr_scheduler/rng/data_order` serialized bytes) — no GPU, no network, deterministic by seed. Lives under `tests/rlm/_stubs/` or inline in the test.

---

# PART B1 — Receipt producer + 5-field checkpoint substrate (the load-bearing foundation)

## Task B1.1: Checkpoint materializer — 5-field state manifest from a component dir

**Files:**
- Create: `backend/agents/rlm/scheduler_receipt_producer.py`
- Test: `tests/rlm/test_scheduler_receipt_producer.py`

**Contract:** the (stub or real) trainer writes a checkpoint directory containing 5 component files under a fixed contract — `model`, `optimizer`, `lr_scheduler`, `rng`, `data_order` (any serialization; bytes are what matter). The harness reads those bytes, computes `sha256` of each → the 5 `*_sha256` values, bundles the 5 components into ONE resumable checkpoint file (the `checkpoint.path` blob that `revive` restores), computes its sha, and writes the state manifest JSON (exactly the 5 keys). The harness OWNS the hashing + manifest — the trainer only produces raw component bytes.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_scheduler_receipt_producer.py
from __future__ import annotations
import hashlib, json, tarfile
from pathlib import Path
import pytest
from backend.agents.rlm.scheduler_receipt_producer import materialize_checkpoint

_COMPONENTS = ("model", "optimizer", "lr_scheduler", "rng", "data_order")

def _write_checkpoint_components(ckpt_dir: Path) -> dict[str, bytes]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payloads = {name: f"{name}-state-bytes".encode() for name in _COMPONENTS}
    for name, data in payloads.items():
        (ckpt_dir / name).write_bytes(data)
    return payloads

def test_materialize_checkpoint_builds_bundle_and_5field_manifest(tmp_path):
    run_dir = tmp_path
    ckpt_components = run_dir / "cellA" / "checkpoint_components"
    payloads = _write_checkpoint_components(ckpt_components)
    out = materialize_checkpoint(
        run_dir=run_dir, cell_output_dir=run_dir / "cellA",
        checkpoint_components_dir=ckpt_components,
    )
    # bundle file exists under run_dir and its sha matches
    bundle = run_dir / out["path"]
    assert bundle.is_file()
    assert out["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    # state manifest has EXACTLY the 5 keys, each the real sha of the component bytes
    state = out["state"]
    assert set(state) == {"model_sha256", "optimizer_sha256", "lr_scheduler_sha256", "rng_sha256", "data_order_sha256"}
    assert state["model_sha256"] == hashlib.sha256(payloads["model"]).hexdigest()
    # state manifest file exists, its sha matches, and its content == state dict
    state_file = run_dir / out["state_path"]
    assert json.loads(state_file.read_text()) == state
    assert out["state_sha256"] == hashlib.sha256(state_file.read_bytes()).hexdigest()
    # all paths are relative to run_dir
    assert not Path(out["path"]).is_absolute() and not Path(out["state_path"]).is_absolute()

def test_materialize_checkpoint_fails_closed_on_missing_component(tmp_path):
    ckpt = tmp_path / "cellB" / "checkpoint_components"
    ckpt.mkdir(parents=True)
    (ckpt / "model").write_bytes(b"only-model")   # missing the other 4
    with pytest.raises(ValueError, match="checkpoint component"):
        materialize_checkpoint(run_dir=tmp_path, cell_output_dir=tmp_path / "cellB",
                               checkpoint_components_dir=ckpt)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/rlm/test_scheduler_receipt_producer.py -v` → FAIL (module/function missing).

- [ ] **Step 3: Implement `materialize_checkpoint`**

```python
# backend/agents/rlm/scheduler_receipt_producer.py
"""Harness-owned producer of authority receipts from a completed LOCAL cell.

This module lives OUTSIDE gpu_cell_runner (which is stdlib-only, copied into the
agent sandbox). It reads the deterministic on-disk evidence a cell produced and
assembles the exact ``raw_receipt`` mapping ``scheduler_evidence.write_verified_receipt``
expects. It NEVER reads an LLM grade: the ladder metric comes from
``metrics.json[metric_id]`` only. The 5-field checkpoint is materialized here (the
harness owns the hashing + manifest); the trainer only produces raw component bytes.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Mapping

_CHECKPOINT_COMPONENTS = ("model", "optimizer", "lr_scheduler", "rng", "data_order")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path, run_dir: Path) -> str:
    return str(path.resolve().relative_to(run_dir.resolve()))


def materialize_checkpoint(
    *, run_dir: Path, cell_output_dir: Path, checkpoint_components_dir: Path,
) -> dict[str, Any]:
    """Bundle a 5-component checkpoint dir into one resumable blob + a state manifest.

    Returns the ``checkpoint`` sub-object for a raw_receipt: path/sha256/state/
    state_path/state_sha256. Fails closed (ValueError) if any of the five
    components is missing.
    """
    run_dir = Path(run_dir)
    components_dir = Path(checkpoint_components_dir)
    state: dict[str, str] = {}
    for name in _CHECKPOINT_COMPONENTS:
        comp = components_dir / name
        if not comp.is_file():
            raise ValueError(f"checkpoint component missing: {name}")
        state[f"{name}_sha256"] = _sha256_file(comp)

    # One resumable blob (what revive restores): a deterministic tar of the 5 files.
    bundle_path = Path(cell_output_dir) / "checkpoint.tar"
    with tarfile.open(bundle_path, "w") as tar:
        for name in _CHECKPOINT_COMPONENTS:            # sorted, fixed order → deterministic
            tar.add(components_dir / name, arcname=name)

    state_path = Path(cell_output_dir) / "checkpoint-state.json"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    return {
        "path": _rel(bundle_path, run_dir),
        "sha256": _sha256_file(bundle_path),
        "state": state,
        "state_path": _rel(state_path, run_dir),
        "state_sha256": _sha256_file(state_path),
    }
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/rlm/test_scheduler_receipt_producer.py -v` → PASS.
- [ ] **Step 5: Lint + commit** — `uvx ruff@0.15.16 check backend/agents/rlm/scheduler_receipt_producer.py tests/rlm/test_scheduler_receipt_producer.py`; then `git add` those two files + `git commit -m "Phase B1: harness-owned 5-field checkpoint materializer for authority receipts"`.

## Task B1.2: Full receipt producer — assemble the exact `raw_receipt` (metric from metrics.json, NOT the grade)

**Files:** extend `scheduler_receipt_producer.py` + its test.

**Before writing:** the implementer MUST read `backend/agents/rlm/scheduler_evidence.py` `_receipt_from_mapping` (~243-271), `_verify_metric`, `_verify_checkpoint`, `_verify_evidence_bundle`, `_verify_fingerprints` verbatim, and the controller test's `_raw_receipt` helper in `tests/rlm/test_scheduler_authority_controller.py`, to match the shape byte-for-byte. The plan's shape block above is the contract, but confirm against source.

- [ ] **Step 1: Write the failing test** (round-trip through `write_verified_receipt` + evidence-not-grade + fail-closed)

```python
def _ladder():
    from backend.agents.rlm.scheduler_evidence import PaperStepLadder
    return PaperStepLadder(paper_ref="1412.6980", metric_id="eval.accuracy", direction="maximize",
                           r_max_steps=50, rung_steps=(10, 50), schedule_source_sha256="a"*64)

def _write_cell_evidence(run_dir: Path, cell_dir: Path, *, metric_id: str, metric_value: float):
    cell_dir.mkdir(parents=True, exist_ok=True)
    # metrics.json carries the deterministic metric AND a planted grade that must be ignored
    (cell_dir / "metrics.json").write_text(json.dumps({metric_id: metric_value, "final_report": {"score": 0.01}}))
    comps = cell_dir / "checkpoint_components"; _write_checkpoint_components(comps)
    # dataset + run_spec fingerprint files under run_dir
    rlm = run_dir / "rlm_state"; rlm.mkdir(exist_ok=True)
    (rlm / "dataset-manifest.json").write_text('{"dataset":"mnist-pinned"}')
    (rlm / "run-spec.json").write_text('{"image":"pinned@sha256"}')
    return comps

def test_build_raw_receipt_round_trips_through_write_verified_receipt(tmp_path):
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
    from backend.agents.rlm.scheduler_evidence import write_verified_receipt, load_verified_receipt
    run_dir = tmp_path
    ladder = _ladder()
    comps = _write_cell_evidence(run_dir, run_dir / "cellA", metric_id=ladder.metric_id, metric_value=0.9)
    raw = build_raw_receipt(
        run_dir=run_dir, cell_output_dir=run_dir / "cellA", checkpoint_components_dir=comps,
        ladder=ladder, campaign_id="campaign-1", branch_id="faithful", parent_branch_id=None,
        attempt_n=1, cell_id="cell-faithful", from_step=0, to_step=10, seed=1, termination_cause=None,
        dataset_manifest_path=run_dir / "rlm_state" / "dataset-manifest.json",
        run_spec_path=run_dir / "rlm_state" / "run-spec.json",
    )
    # evidence-not-grade: metric_value is the metrics.json[metric_id], NOT final_report.score (0.01)
    assert raw["metric"]["value"] == 0.9 and raw["metric"]["id"] == ladder.metric_id
    # round-trips: a real fsync attest publishes it and load_verified_receipt returns it
    ledger = run_dir / "campaign" / "attempts.jsonl"
    def attest(row):
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.open("a").write(json.dumps(dict(row)) + "\n")
    path = write_verified_receipt(raw, ladder=ladder, run_dir=run_dir, campaign_id="campaign-1", attest=attest)
    assert load_verified_receipt(path, ladder=ladder, run_dir=run_dir, expected_campaign_id="campaign-1") is not None

def test_build_raw_receipt_ignores_grade_when_metric_present(tmp_path):
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
    run_dir = tmp_path; ladder = _ladder()
    comps = _write_cell_evidence(run_dir, run_dir / "cellB", metric_id=ladder.metric_id, metric_value=0.42)
    raw = build_raw_receipt(run_dir=run_dir, cell_output_dir=run_dir / "cellB", checkpoint_components_dir=comps,
        ladder=ladder, campaign_id="c", branch_id="b", parent_branch_id=None, attempt_n=1, cell_id="cell-b",
        from_step=0, to_step=10, seed=1, termination_cause=None,
        dataset_manifest_path=run_dir/"rlm_state"/"dataset-manifest.json",
        run_spec_path=run_dir/"rlm_state"/"run-spec.json")
    assert raw["metric"]["value"] == 0.42        # never 0.01 (the planted grade)

def test_build_raw_receipt_missing_metric_key_fails_closed(tmp_path):
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
    run_dir = tmp_path; ladder = _ladder()
    cell = run_dir / "cellC"; cell.mkdir()
    (cell / "metrics.json").write_text(json.dumps({"final_report": {"score": 0.99}}))  # NO metric_id key
    comps = cell / "checkpoint_components"; _write_checkpoint_components(comps)
    (run_dir/"rlm_state").mkdir(); (run_dir/"rlm_state"/"dataset-manifest.json").write_text("{}"); (run_dir/"rlm_state"/"run-spec.json").write_text("{}")
    with pytest.raises(ValueError, match="metric"):
        build_raw_receipt(run_dir=run_dir, cell_output_dir=cell, checkpoint_components_dir=comps, ladder=ladder,
            campaign_id="c", branch_id="b", parent_branch_id=None, attempt_n=1, cell_id="cc", from_step=0, to_step=10,
            seed=1, termination_cause=None, dataset_manifest_path=run_dir/"rlm_state"/"dataset-manifest.json",
            run_spec_path=run_dir/"rlm_state"/"run-spec.json")
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement `build_raw_receipt`** — reads `metrics.json`, extracts `metrics[ladder.metric_id]` (raise `ValueError("metric ...")` if absent/non-finite/bool — NEVER fall back to any grade key), writes the metric artifact (a JSON file whose `[metric_id]` == value; can reuse `metrics.json` itself as the artifact if its `[metric_id]` matches), calls `materialize_checkpoint`, builds the evidence bundle JSON (`schema:1, coherent:true, metrics_sha256=<sha of metric artifact>, code_tree_digest=<code_sha256>`), computes fingerprint SHAs, and assembles the full dict per the shape block. All `*_path` relative to `run_dir`. The `code_sha256`/`code_tree_digest` must be ONE consistent value (see red-line #note: reconcile with the canonical evidence bundle digest if `OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE` is on; for the stub/demo, compute a deterministic `sha256` over the cell's code tree or accept an injected `code_sha256`).

```python
def build_raw_receipt(
    *, run_dir: Path, cell_output_dir: Path, checkpoint_components_dir: Path,
    ladder, campaign_id: str, branch_id: str, parent_branch_id: str | None,
    attempt_n: int, cell_id: str, from_step: int, to_step: int, seed: int,
    termination_cause: str | None, dataset_manifest_path: Path, run_spec_path: Path,
    code_sha256: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir); cell_output_dir = Path(cell_output_dir)
    metrics_path = cell_output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping) or ladder.metric_id not in metrics:
        raise ValueError(f"metric {ladder.metric_id!r} absent from metrics.json — never fall back to a grade")
    value = metrics[ladder.metric_id]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {ladder.metric_id!r} is not a finite number")
    value = float(value)
    # metric artifact: metrics.json itself already satisfies payload[metric_id]==value
    metric_sha = _sha256_file(metrics_path)
    checkpoint = materialize_checkpoint(run_dir=run_dir, cell_output_dir=cell_output_dir,
                                        checkpoint_components_dir=checkpoint_components_dir)
    code_digest = code_sha256 or hashlib.sha256(_code_tree_bytes(cell_output_dir)).hexdigest()
    bundle_path = cell_output_dir / "evidence_bundle.json"
    bundle_path.write_text(json.dumps({"schema": 1, "coherent": True,
        "metrics_sha256": metric_sha, "code_tree_digest": code_digest}, sort_keys=True), encoding="utf-8")
    return {
        "schema_version": 1, "campaign_id": campaign_id, "branch_id": branch_id,
        "parent_branch_id": parent_branch_id, "attempt_n": attempt_n, "cell_id": cell_id,
        "paper_ref": ladder.paper_ref, "ladder_sha256": ladder.sha256,
        "from_step": from_step, "to_step": to_step, "seed": seed, "termination_cause": termination_cause,
        "metric": {"id": ladder.metric_id, "direction": ladder.direction, "value": value,
                   "artifact_path": _rel(metrics_path, run_dir), "sha256": metric_sha},
        "checkpoint": checkpoint,
        "evidence_bundle": {"path": _rel(bundle_path, run_dir), "sha256": _sha256_file(bundle_path)},
        "fingerprints": {"code_sha256": code_digest,
                         "dataset_sha256": _sha256_file(Path(dataset_manifest_path)),
                         "dataset_manifest_path": _rel(Path(dataset_manifest_path), run_dir),
                         "run_spec_sha256": _sha256_file(Path(run_spec_path)),
                         "run_spec_path": _rel(Path(run_spec_path), run_dir)},
    }
```
(Add a small `_code_tree_bytes(dir)` helper hashing the sorted cell source files, or require `code_sha256` injected. Keep it deterministic.)

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Lint + commit** — `git commit -m "Phase B1: receipt producer assembles verified raw_receipt from a cell's deterministic metric (never the grade)"`.

---

# PART B2 — Construct + inject the controller (flag+spec gated, byte-identical-OFF)

## Task B2.1: `CampaignOptions.authority_spec_path` + plumbing

**Before writing:** read `campaign_composition.py` `CampaignOptions` (frozen dataclass ~161) and `build_campaign` (~1272), and `cli.py`'s campaign flag block, verbatim.

- [ ] **Step 1: Write the failing test** — assert `CampaignOptions` accepts `authority_spec_path: str | None = None` defaulting to `None`, and that an existing construction with no such arg is unchanged (byte-identical default).
- [ ] **Step 2..4:** add the frozen-dataclass field (defaulted `None`) + a CLI flag `--authority-spec` / run-spec key wired into `CampaignOptions`; run tests.
- [ ] **Step 5: Commit** — `git commit -m "Phase B2: add CampaignOptions.authority_spec_path (default None, byte-identical off)"`.

## Task B2.2: Construct the controller in `build_campaign` (gated) + inject into `ReproductionCampaign`

**Before writing:** read `build_campaign` (~1272), `ReproductionCampaign.__init__` (~404-416, the `branch_tree_event_store=None` injection precedent), `_scheduler_tree_enabled()` (~105), and how `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` is checked (~1051-1053). Read `load_authority_spec` + `SchedulerAuthoritySpec` (in `scheduler_runtime.py`) for the spec-file format (see the plan's JSON shape block).

- [ ] **Step 1: Write the failing tests** — a NEW `tests/rlm/test_authority_controller_wiring.py`:
  - `test_off_constructs_no_controller`: with both flags unset (or spec path None), `build_campaign(...)` produces a campaign whose `scheduler_controller is None` AND the run dir has NO `campaign/scheduler_ladder.json` / `scheduler_tree_state.json` (construction never happened) — the byte-identical-OFF proof.
  - `test_on_with_spec_constructs_controller`: with both flags set AND a valid `authority_spec_path` (write a minimal spec JSON per the shape block, `paper_ref` matching the campaign), `build_campaign` constructs a `SchedulerAuthorityController` with `campaign_id == project_id`, and `bootstrap()` registers the spec branches.
  - `test_on_without_spec_still_no_apply`: both flags set but `authority_spec_path=None` ⇒ no controller (falls back to today's `applied:false` audit, no crash).

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — in `build_campaign`, gated behind `_scheduler_tree_enabled() and _scheduler_authoritative_enabled() and opts.authority_spec_path`: `spec = load_authority_spec(opts.authority_spec_path, paper_ref=opts.paper_ref)`; `controller = SchedulerAuthorityController(run_dir, campaign_id=project_id, spec=spec)`; pass `scheduler_controller=controller` into `ReproductionCampaign(...)`. Add `scheduler_controller: SchedulerAuthorityController | None = None` to `ReproductionCampaign.__init__` mirroring `branch_tree_event_store`; store on `self`. Lazy import inside the gate so OFF imports nothing. Add a `_scheduler_authoritative_enabled()` helper (the `.strip().lower() in ("1","true","yes")` contract on `OPENRESEARCH_SCHEDULER_AUTHORITATIVE`) if not already present.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "Phase B2: construct SchedulerAuthorityController in build_campaign (both-flags+spec gated), inject into ReproductionCampaign; OFF constructs nothing"`.

## Task B2.3: Resolve the `branch-tree:<id>` double-writer under authority

**Before writing:** read `reproduction_campaign.py::_maybe_emit_root_branch_spawned` (~463) and `branch_lineage.branch_tree_aggregate_id` (~29). Under authority `campaign_id == project_id`, so the controller runtime and the serial emit collide on one aggregate.

- [ ] **Step 1: Write the failing test** — when `self.scheduler_controller is not None` (authority live), `_maybe_emit_root_branch_spawned` is suppressed (the controller is the sole `branch-tree:<id>` writer); when it's `None` (today), the serial emit is unchanged. Assert no double `branch_spawned` on the aggregate under authority.
- [ ] **Step 2..4:** guard the serial emit with `if self.scheduler_controller is None:`; run tests.
- [ ] **Step 5: Commit** — `git commit -m "Phase B2: controller is the sole branch-tree lineage writer under authority (suppress serial branch-spawn emit)"`.

---

# PART B3 — Cohort driver + campaign-reachable local demo (= the Phase B exit)

## Task B3.1: Flag-gated receipt emission at `_execute_cell_matrix`

**Before writing:** read `primitives.py::_execute_cell_matrix` around the `_matrix_runner = gpu_cell_runner.run_matrix` seam (~6534) and the later `_matrix_runner(...)` call + result handling. Confirm what's in scope after `run_matrix` returns (`ctx`, `project_dir`/`code`, `campaign_id`, the result dict, `artifact_root`). `gpu_cell_runner` MUST stay import-clean — the emission is here.

- [ ] **Step 1: Write the failing test** — with both flags OFF, `_execute_cell_matrix` produces a byte-identical result dict + `cell_manifest.json` and writes NO receipt (`campaign/scheduler_receipts/` absent). With a controller present + flags ON, after `run_matrix` returns an `ok` cell, a receipt is materialized (via `build_raw_receipt`) and `controller.record_cell_receipt(raw, attest=CampaignLedger.append_row)` is called, transitioning the branch to `awaiting_receipt`. (Use a fake controller recording calls + a real `run_matrix` over a CPU stub, or a focused unit with a stub cell output dir.)
- [ ] **Step 2..4:** add a flag-gated block after `run_matrix` returns: only when the campaign passed a live controller (thread it through `ctx` or the cell-matrix call), for each `ok` cell build the raw_receipt from the cell's output dir (metrics.json + the checkpoint components the stub/trainer wrote per contract) and call `controller.record_cell_receipt`. OFF ⇒ the whole block is skipped, `run_matrix` untouched. Read the real provider GPU-$ per branch from the deterministic assessment (`AttemptAssessment.cost.gpu_usd`), not the ledger.
- [ ] **Step 5: Commit** — `git commit -m "Phase B3: flag-gated authority-receipt emission at _execute_cell_matrix (gpu_cell_runner untouched)"`.

## Task B3.2: The flag-gated `_cohort_loop` branch in the campaign

**Before writing:** read `reproduction_campaign.py::_loop` (~949) fully. Do NOT rewrite it — add a sibling `_cohort_loop` and dispatch to it only when `self.scheduler_controller is not None`.

- [ ] **Step 1: Write the failing test** (`tests/rlm/test_campaign_cohort_loop.py`) — with a live controller, the campaign runs `_cohort_loop`: `controller.bootstrap()` → `claim_launches(max_parallel=a100_cap)` → run each branch's cell (serially on one machine) → `record_cell_receipt` per branch → `decide_rung(rung, provider_gpu_usd_by_branch)` → apply promote (re-queue rung+1) / freeze (frozen pool) / kill → `claim_launches` again picks up promotions → `revive(branch_id)` returns a frozen branch → next claim re-enters it at its checkpoint. Assert the terminal deterministic campaign decision still wins (authority only reorders the CONTINUE cohort) and the fail-closed ledger write-ahead is intact per launch. Mirror the controller test's drive sequence.
- [ ] **Step 2..4:** implement `_cohort_loop` as that driver; the campaign's main entry dispatches `_cohort_loop` when `self.scheduler_controller is not None`, else the existing `_loop` (byte-identical OFF). Feed `provider_gpu_usd_by_branch` from deterministic per-branch assessments. On a launch/submit failure call `controller.release_launch(branch_id)`. Every non-terminal branch at a rung must have a recorded receipt before `decide_rung` (else it raises `incomplete` — fail-closed, correct).
- [ ] **Step 5: Commit** — `git commit -m "Phase B3: flag-gated _cohort_loop drives claim->receipt->decide->apply->revive (serial _loop untouched OFF)"`.

## Task B3.3: THE EXIT TEST — campaign-reachable local freeze/promote/revive from verified receipts

**Files:** `tests/rlm/test_local_freeze_promote_revive.py` (NEW) + a CPU-stub `train_cell.py`.

- [ ] **Step 1: Write the exit test** — drive a **real `campaign` entrypoint** (`--sandbox local`, both flags ON, `--authority-spec <tmp spec>`) over a 3-branch spec (mirror `_spec()`: faithful/ambiguity/discovery, `a100_cap=2`, `rung_steps=(10,50)`) whose cells run a **CPU-only stub** `train_cell.py` that writes `metrics.json` (with `metric_id` + a planted `final_report.score`) and the 5-component checkpoint dir — no GPU, no network (socket-hermetic per `tests/CLAUDE.md`). Assert from the run artifacts: at rung 0 the low-metric branch (`ambiguity`) is **frozen**, the others **promoted**; a subsequent `revive` of the frozen branch re-enters it at its checkpoint; and the `branch-tree` events include `branch_spawned`/`rung_climbed`/`branch_promoted`/`frozen_pool_eviction`/`branch_revived`. Assert `metric_value` in every receipt came from `metrics.json[metric_id]`, never the planted grade. **Also ship the OFF half:** the same campaign invocation with the flags unset produces byte-identical serial output (no `scheduler_receipts/`, no `scheduler_ladder.json`, no cohort behavior).
- [ ] **Step 2: Run — it exercises B1+B2+B3 end-to-end.** Iterate until green (this is where integration gaps surface). If `decide_rung` raises `incomplete`, the driver decided before all cohort receipts landed — fix the driver, not the guard. If a receipt fails to verify, the stub's checkpoint/metric shape drifted from `_receipt_from_mapping` — fix the producer/stub, never weaken the verifier.
- [ ] **Step 3: Full scheduler + campaign suite green** — `.venv/bin/python -m pytest tests/rlm/ -q -k "scheduler or asha or campaign or freeze_promote_revive"` → all pass; plus `tests/rlm/test_authority_preserves_every_terminal_decision` (terminal always wins) and the byte-identical-OFF pair green.
- [ ] **Step 4: Lint + commit** — `git commit -m "Phase B3: exit test — campaign-reachable local freeze/promote/revive from verified receipts; OFF byte-identical"`.

---

## Definition of done (Phase B)

- The exit test (B3.3) passes: a real `campaign --sandbox local` with both flags ON freezes one branch, promotes others, and revives a frozen one from **verified receipts**; flags-OFF is byte-identical.
- `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` remains **default-OFF**; a default-ON flip is explicitly OUT of scope (needs the ≥3 paired A/B + grader-σ + operator sign-off — that's Phase C's operator gate, never autonomous).
- All red-line tests green: evidence-not-grade (metric from `metrics.json[metric_id]`), terminal-always-wins, `training_diverged`-only kill, fail-closed on absent receipt, `gpu_cell_runner` import-clean, real provider GPU-$.
- The full scheduler + campaign hermetic suite is green.

## Self-review (inline)

- **Spec coverage:** B1 (producer + checkpoint) → B1.1/B1.2; B2 (construct+inject+double-writer) → B2.1/B2.2/B2.3; B3 (emission + cohort loop + campaign-reachable exit) → B3.1/B3.2/B3.3. The advisor's non-negotiable — the cohort loop must be reachable from a real `campaign` entrypoint, not a throwaway harness — is B3.2's dispatch + B3.3's exit test.
- **Red lines → required tests:** evidence-not-grade (B1.2 `test_build_raw_receipt_ignores_grade...`), byte-identical-OFF (B2.2 `test_off_constructs_no_controller` + B3.3 OFF half), fail-closed (B1.2 missing-metric + B1.1 missing-component), terminal-wins (B3.2 assertion), import-clean gpu_cell_runner (B3.1 places emission at the caller).
- **Type consistency:** the `raw_receipt` shape in B1.2 matches `_receipt_from_mapping` verbatim; `materialize_checkpoint` returns exactly the `checkpoint` sub-object B1.2 embeds; `build_raw_receipt` output feeds `write_verified_receipt` (B1.2 proves the round-trip) and `controller.record_cell_receipt` (B3.1).
