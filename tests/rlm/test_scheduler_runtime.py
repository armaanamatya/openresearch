"""Hermetic end-to-end tests for the receipt-gated authority tree runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.agents.rlm.scheduler_evidence import PaperStepLadder
from backend.agents.rlm.scheduler_runtime import (
    SchedulerRuntimeError,
    SchedulerTreeRuntime,
    load_authority_spec,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f10(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ladder() -> PaperStepLadder:
    return PaperStepLadder(
        paper_ref="2605.15155",
        metric_id="eval.accuracy",
        direction="maximize",
        r_max_steps=50,
        rung_steps=(10, 50),
        schedule_source_sha256="a" * 64,
    )


def _receipt(
    root: Path,
    ladder: PaperStepLadder,
    *,
    branch_id: str,
    attempt_n: int,
    value: float,
    termination_cause: str | None = None,
) -> Path:
    artifact = root / "artifact" / branch_id
    artifact.mkdir(parents=True, exist_ok=True)
    metrics = artifact / "metrics.json"
    metrics.write_text(json.dumps({ladder.metric_id: value, "final_report": {"score": 0.999}}), encoding="utf-8")
    checkpoint = artifact / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint:{branch_id}".encode())
    state = artifact / "checkpoint-state.json"
    state_payload = {
        "model_sha256": "b" * 64,
        "optimizer_sha256": "c" * 64,
        "lr_scheduler_sha256": "d" * 64,
        "rng_sha256": "e" * 64,
        "data_order_sha256": "f" * 64,
    }
    state.write_text(json.dumps(state_payload), encoding="utf-8")
    rlm_state = root / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    dataset = rlm_state / "dataset-manifest.json"
    run_spec = rlm_state / "run-spec.json"
    dataset.write_text('{"dataset":"pinned"}', encoding="utf-8")
    run_spec.write_text('{"image":"pinned@sha256"}', encoding="utf-8")
    bundle = rlm_state / f"evidence-{branch_id}.json"
    bundle.write_text(json.dumps({
        "schema": 1,
        "coherent": True,
        "metrics_sha256": _sha(metrics),
        "code_tree_digest": "1" * 64,
    }), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "campaign_id": "campaign-1",
        "branch_id": branch_id,
        "parent_branch_id": None,
        "attempt_n": attempt_n,
        "cell_id": f"cell-{branch_id}",
        "paper_ref": ladder.paper_ref,
        "ladder_sha256": ladder.sha256,
        "from_step": 0,
        "to_step": 10,
        "metric": {"id": ladder.metric_id, "direction": ladder.direction, "value": value,
                   "artifact_path": str(metrics.relative_to(root)), "sha256": _sha(metrics)},
        "checkpoint": {"path": str(checkpoint.relative_to(root)), "sha256": _sha(checkpoint),
                       "state": state_payload, "state_path": str(state.relative_to(root)),
                       "state_sha256": _sha(state)},
        "evidence_bundle": {"path": str(bundle.relative_to(root)), "sha256": _sha(bundle)},
        "fingerprints": {"code_sha256": "1" * 64, "dataset_sha256": _sha(dataset),
                         "dataset_manifest_path": str(dataset.relative_to(root)),
                         "run_spec_sha256": _sha(run_spec), "run_spec_path": str(run_spec.relative_to(root))},
        "seed": 7,
        "termination_cause": termination_cause,
    }
    receipt = root / "campaign" / "scheduler_receipts" / f"{branch_id}-{attempt_n}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    ledger = root / "campaign" / "attempts.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "status": "scheduler_receipt", "receipt_sha256": _sha(receipt),
            "campaign_id": "campaign-1", "branch_id": branch_id, "attempt_n": attempt_n,
            "paper_ref": ladder.paper_ref, "run_spec_sha256": _sha(run_spec),
        }) + "\n")
    return receipt


def _runtime(tmp_path: Path, events: list[tuple[str, dict]] | None = None) -> SchedulerTreeRuntime:
    sink = None if events is None else lambda kind, payload: events.append((kind, dict(payload)))
    return SchedulerTreeRuntime(tmp_path, campaign_id="campaign-1", ladder=_ladder(), event_sink=sink)


def test_authority_tree_promotes_freezes_and_emits_factual_events_after_wal(tmp_path):
    events: list[tuple[str, dict]] = []
    runtime = _runtime(tmp_path, events)
    ladder = _ladder()
    for attempt_n, (branch_id, score) in enumerate((("faithful", 0.9), ("ambiguity", 0.5), ("discovery", 0.1)), 1):
        assert runtime.register_branch(
            branch_id=branch_id,
            branch_type=branch_id if branch_id != "faithful" else "faithful",
            hypothesis_fingerprint=_f10(f"f10-{branch_id}"),
        )
        runtime.record_receipt(branch_id, _receipt(tmp_path, ladder, branch_id=branch_id, attempt_n=attempt_n, value=score))

    result = runtime.decide_ready_rung(
        rung=0,
        provider_gpu_usd_by_branch={"faithful": 1.0, "ambiguity": 1.0, "discovery": 1.0},
        gpu_usd_budget=1.0,
        a100_cap=2,
        discovery_gpu_usd_budget=1.0,
    )

    actions = {item.branch_id: item.action for item in result.actions}
    # Discovery is isolated by the core, and therefore does not compete with
    # the faithful/ambiguity budget cohort.
    assert actions == {"faithful": "promote", "ambiguity": "freeze", "discovery": "promote"}
    assert runtime.branches["faithful"].state == "queued"
    assert runtime.branches["faithful"].rung == 1
    assert runtime.branches["ambiguity"].state == "frozen"
    assert runtime.branches["discovery"].state == "queued"
    assert result.authority_audit()["applied"] is True
    assert {item.decision_evidence_sha256 for item in result.actions} == {result.decision_evidence_sha256}
    assert {item.authority_audit_sha256 for item in result.actions} == {result.authority_audit_sha256}
    assert (tmp_path / "campaign" / "scheduler_ladder.json").is_file()
    decision_evidence = tmp_path / result.decision_evidence_path
    assert decision_evidence.is_file()
    assert hashlib.sha256(decision_evidence.read_bytes()).hexdigest() == result.decision_evidence_sha256
    assert any(kind == "branch_promoted" for kind, _ in events)
    assert any(kind == "frozen_pool_eviction" for kind, _ in events)
    wal = (tmp_path / "campaign" / "scheduler_tree_actions.jsonl").read_text(encoding="utf-8")
    assert "authority_batch" in wal


def test_true_kill_requires_literal_training_diverged_and_is_never_grade_based(tmp_path):
    runtime = _runtime(tmp_path)
    ladder = _ladder()
    runtime.register_branch(branch_id="bad", branch_type="faithful", hypothesis_fingerprint=_f10("f10-bad"))
    runtime.register_branch(branch_id="good", branch_type="faithful", hypothesis_fingerprint=_f10("f10-good"))
    # A lower final_report score is embedded in every receipt fixture.  The
    # authority result is controlled only by eval.accuracy and the literal cause.
    runtime.record_receipt("bad", _receipt(tmp_path, ladder, branch_id="bad", attempt_n=1, value=0.99,
                                            termination_cause="training_diverged"))
    runtime.record_receipt("good", _receipt(tmp_path, ladder, branch_id="good", attempt_n=2, value=0.1))

    result = runtime.decide_ready_rung(
        rung=0, provider_gpu_usd_by_branch={"bad": 1.0, "good": 1.0}, gpu_usd_budget=1.0, a100_cap=1,
    )

    actions = {item.branch_id: item.action for item in result.actions}
    assert actions["bad"] == "kill"
    assert runtime.branches["bad"].state == "killed"
    assert runtime.branches["bad"].termination_cause == "training_diverged"
    assert actions["good"] == "promote"


def test_missing_provider_cost_or_receipt_fails_closed_without_authority_transition(tmp_path):
    runtime = _runtime(tmp_path)
    ladder = _ladder()
    runtime.register_branch(branch_id="b1", branch_type="faithful", hypothesis_fingerprint=_f10("f10-b1"))
    with pytest.raises(SchedulerRuntimeError, match="receipt"):
        runtime.record_receipt("b1", tmp_path / "campaign" / "scheduler_receipts" / "missing.json")
    assert runtime.branches["b1"].state == "queued"

    runtime.record_receipt("b1", _receipt(tmp_path, ladder, branch_id="b1", attempt_n=1, value=0.8))
    with pytest.raises(SchedulerRuntimeError, match="provider GPU cost"):
        runtime.decide_ready_rung(rung=0, provider_gpu_usd_by_branch={}, gpu_usd_budget=1.0, a100_cap=1)
    assert runtime.branches["b1"].state == "awaiting_receipt"


def test_write_ahead_wal_recovers_projection_and_f10_dedup_is_not_reinvented(tmp_path):
    events: list[tuple[str, dict]] = []
    runtime = _runtime(tmp_path, events)
    existing_f10 = _f10("existing-f10")
    assert runtime.register_branch(branch_id="first", branch_type="faithful", hypothesis_fingerprint=existing_f10)
    assert not runtime.register_branch(branch_id="duplicate", branch_type="ambiguity", hypothesis_fingerprint=existing_f10)
    assert set(runtime.branches) == {"first"}
    assert events[-1] == ("dedup_hit", {"hypothesis_fingerprint": existing_f10, "existing_branch_id": "first"})

    # Simulate a controller crash after fsync'ing the WAL but before a durable
    # snapshot reaches its next process. Re-opening reconstructs the same tree.
    (tmp_path / "campaign" / "scheduler_tree_state.json").unlink()
    recovered = _runtime(tmp_path)
    assert set(recovered.branches) == {"first"}
    assert recovered.branches["first"].hypothesis_fingerprint == existing_f10


def test_event_outbox_keeps_a_whole_cohort_durable_and_reconciles_after_sink_failure(tmp_path):
    runtime = _runtime(tmp_path)
    ladder = _ladder()
    for attempt_n, (branch_id, score) in enumerate((("top", 0.9), ("low", 0.1)), 1):
        runtime.register_branch(branch_id=branch_id, branch_type="faithful", hypothesis_fingerprint=_f10(branch_id))
        runtime.record_receipt(branch_id, _receipt(tmp_path, ladder, branch_id=branch_id, attempt_n=attempt_n, value=score))

    # Deliver setup facts first. The batch's first authority event then fails,
    # after its whole WAL decision was committed but before its EventStore fact.
    runtime.event_sink = lambda _kind, _payload: None
    assert runtime.reconcile_events() > 0
    runtime.event_sink = lambda kind, _payload: (_ for _ in ()).throw(RuntimeError("store down")) if kind == "branch_promoted" else None
    with pytest.raises(SchedulerRuntimeError, match="durable but event emission failed"):
        runtime.decide_ready_rung(
            rung=0, provider_gpu_usd_by_branch={"top": 1.0, "low": 1.0}, gpu_usd_budget=1.0, a100_cap=1,
        )
    # It is never a partly re-ranked cohort: both transitions were one batch.
    assert runtime.branches["top"].state == "queued"
    assert runtime.branches["low"].state == "frozen"

    delivered: list[str] = []
    runtime.event_sink = lambda kind, _payload: delivered.append(kind)
    assert runtime.reconcile_events() >= 2
    assert {"branch_promoted", "frozen_pool_eviction"}.issubset(delivered)
    assert runtime.reconcile_events() == 0


def test_frozen_branch_can_be_explicitly_revived_but_not_reuse_its_old_receipt(tmp_path):
    events: list[tuple[str, dict]] = []
    runtime = _runtime(tmp_path, events)
    ladder = _ladder()
    for attempt_n, (branch_id, score) in enumerate((("top", 0.9), ("low", 0.1)), 1):
        runtime.register_branch(branch_id=branch_id, branch_type="faithful", hypothesis_fingerprint=_f10(branch_id))
        runtime.record_receipt(branch_id, _receipt(tmp_path, ladder, branch_id=branch_id, attempt_n=attempt_n, value=score))
    runtime.decide_ready_rung(
        rung=0, provider_gpu_usd_by_branch={"top": 1.0, "low": 1.0}, gpu_usd_budget=1.0, a100_cap=1,
    )
    old_receipt = tmp_path / "campaign" / "scheduler_receipts" / "low-2.json"
    revived = runtime.revive_branch("low")
    assert revived.state == "queued"
    assert revived.checkpoint_path is not None
    assert revived.receipt_sha256 is None
    assert events[-1][0] == "branch_revived"
    with pytest.raises(SchedulerRuntimeError, match="already produced"):
        runtime.record_receipt("low", old_receipt)


def test_explicit_zero_gpu_width_freezes_a_single_ready_branch(tmp_path):
    runtime = _runtime(tmp_path)
    ladder = _ladder()
    runtime.register_branch(branch_id="only", branch_type="faithful", hypothesis_fingerprint=_f10("only"))
    runtime.record_receipt("only", _receipt(tmp_path, ladder, branch_id="only", attempt_n=1, value=0.9))
    result = runtime.decide_ready_rung(
        rung=0, provider_gpu_usd_by_branch={"only": 1.0}, gpu_usd_budget=0.0, a100_cap=1,
    )
    assert result.actions[0].action == "freeze"
    assert runtime.branches["only"].state == "frozen"


def test_adam_authority_spec_config_loads_with_four_branches_and_rungs():
    config = Path(__file__).resolve().parents[2] / "configs" / "adam_authority_spec.json"
    spec = load_authority_spec(config, paper_ref="1412.6980")
    assert len(spec.branches) == 4
    assert spec.ladder.paper_ref == "1412.6980"
    assert spec.ladder.rung_steps == (100, 300, 1000)
    assert spec.ladder.r_max_steps == 1000
    assert spec.ladder.direction == "minimize"
    # Exactly one faithful safety-bracket branch, two discovery, one ambiguity.
    types = sorted(branch.branch_type for branch in spec.branches)
    assert types == ["ambiguity", "discovery", "discovery", "faithful"]
    assert [b.branch_id for b in spec.branches if b.is_safety_bracket] == ["adam-faithful-lr1e-3"]
    # Distinct F10 fingerprints and distinct seeds.
    assert len({b.hypothesis_fingerprint for b in spec.branches}) == 4
    assert sorted(b.seed for b in spec.branches) == [1, 2, 3, 4]
    # The safety bracket forces safety_gpu_usd_budget; discovery forces discovery budget.
    assert spec.safety_gpu_usd_budget == 2.0
    assert spec.discovery_gpu_usd_budget == 2.0
    assert spec.a100_cap == 1


def test_authority_spec_requires_existing_f10s_and_separate_reservations(tmp_path):
    raw = {
        "schema_version": 1,
        "ladder": {
            "paper_ref": "2605.15155", "metric_id": "eval.accuracy", "direction": "maximize",
            "r_max_steps": 50, "rung_steps": [10, 50], "schedule_source_sha256": "a" * 64,
        },
        "width": {"gpu_usd_budget": 3.0, "a100_cap": 2, "discovery_gpu_usd_budget": 1.0},
        "branches": [
            {"branch_id": "faithful", "branch_type": "faithful", "hypothesis_fingerprint": _f10("faithful"), "seed": 1},
            {"branch_id": "discovery", "branch_type": "discovery", "hypothesis_fingerprint": _f10("discovery"), "seed": 2},
        ],
    }
    path = tmp_path / "authority-spec.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    spec = load_authority_spec(path, paper_ref="2605.15155")
    assert spec.discovery_gpu_usd_budget == 1.0
    raw["branches"][1]["hypothesis_fingerprint"] = "not-a-sha"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SchedulerRuntimeError, match="malformed"):
        load_authority_spec(path, paper_ref="2605.15155")
