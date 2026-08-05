"""Hermetic evidence tests for the offline ASHA authority A/B gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.agents.rlm.asha_authority_gate import evaluate_manifest, main
from backend.agents.rlm.branch_lineage import BranchPromoted, BranchTrueKilled, FrozenPoolEviction
from backend.agents.rlm.scheduler_evidence import PaperStepLadder
from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId, CorrelationId, EventEnvelope, new_event_id


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign_state(
    *, terminal_kind: str = "EXHAUSTED", paper: str = "2605.15155", project_id: str = "fixture"
) -> dict:
    return {
        "project_id": project_id,
        "paper_ref": paper,
        "state": "terminal",
        "rubric_sha256": "a" * 64,
        "terminal": {"kind": terminal_kind, "rule": "budget", "stop_reason": "cap"},
    }


def _advisory(*, action: str = "promote", branch_id: str = "1") -> dict:
    return {
        "rung": 0,
        "width_meter": {
            "gpu_usd_budget": 5.0,
            "a100_cap": 1,
            "eta": 3.0,
            "noise_floor": 0.0067,
        },
        "decisions": [
            {"branch_id": branch_id, "action": action, "reason": "deterministic_rank"}
        ],
    }


def _audit(
    *,
    applied: bool = False,
    action: str = "continue",
    basis: str = "no deterministic metric and optimizer-step lineage",
    failure_class: str | None = None,
) -> dict:
    out = {
        "enabled": True,
        "applied": applied,
        "action": action,
        "deterministic_evidence_basis": basis,
    }
    if failure_class is not None:
        out["failure_class"] = failure_class
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attach_verified_receipt(
    run: Path,
    *,
    action: str = "promote",
    termination_cause: str | None = None,
    controller_event_store: Path | None = None,
) -> None:
    """Add a controller-attested receipt to a fixture authority arm.

    This deliberately mirrors the independent receipt verifier rather than
    teaching the A/B gate a test-only shortcut.  It gives the positive fixture
    exactly the same paper-step, artifact, checkpoint, and ledger bindings a
    real authoritative run must carry.
    """
    ladder = PaperStepLadder(
        paper_ref="2605.15155",
        metric_id="eval.accuracy",
        direction="maximize",
        r_max_steps=100,
        rung_steps=(10, 50, 100),
        schedule_source_sha256="a" * 64,
    )
    campaign_id = json.loads((run / "campaign" / "campaign.json").read_text())["project_id"]
    ladder_path = run / "campaign" / "scheduler_step_ladder.json"
    _json(ladder_path, {
        "schema_version": 1,
        "paper_ref": ladder.paper_ref,
        "metric_id": ladder.metric_id,
        "direction": ladder.direction,
        "r_max_steps": ladder.r_max_steps,
        "rung_steps": list(ladder.rung_steps),
        "schedule_source_sha256": ladder.schedule_source_sha256,
    })
    metrics = run / "artifact" / "metrics.json"
    _json(metrics, {"eval.accuracy": 0.8, "final_report": {"score": 0.0}})
    checkpoint = run / "artifact" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_state = run / "artifact" / "checkpoint_state.json"
    checkpoint_state_data = {
        "model_sha256": "b" * 64,
        "optimizer_sha256": "c" * 64,
        "lr_scheduler_sha256": "d" * 64,
        "rng_sha256": "e" * 64,
        "data_order_sha256": "f" * 64,
    }
    _json(checkpoint_state, checkpoint_state_data)
    dataset = run / "rlm_state" / "dataset_manifest.json"
    run_spec = run / "rlm_state" / "run_spec.json"
    _json(dataset, {"dataset": "paper-pinned"})
    _json(run_spec, {"image": "pinned@sha256"})
    bundle = run / "rlm_state" / "evidence_bundle.json"
    _json(bundle, {
        "schema": 1,
        "coherent": True,
        "metrics_sha256": _sha(metrics),
        "code_tree_digest": "1" * 64,
    })
    receipt_path = run / "campaign" / "scheduler_receipts" / "1-1.json"
    _json(receipt_path, {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "branch_id": "1",
        "parent_branch_id": None,
        "attempt_n": 1,
        "cell_id": "cell-1",
        "paper_ref": ladder.paper_ref,
        "ladder_sha256": ladder.sha256,
        "from_step": 0,
        "to_step": 10,
        "metric": {
            "id": ladder.metric_id,
            "direction": ladder.direction,
            "value": 0.8,
            "artifact_path": "artifact/metrics.json",
            "sha256": _sha(metrics),
        },
        "checkpoint": {
            "path": "artifact/checkpoint.pt",
            "sha256": _sha(checkpoint),
            "state": checkpoint_state_data,
            "state_path": "artifact/checkpoint_state.json",
            "state_sha256": _sha(checkpoint_state),
        },
        "evidence_bundle": {"path": "rlm_state/evidence_bundle.json", "sha256": _sha(bundle)},
        "fingerprints": {
            "code_sha256": "1" * 64,
            "dataset_sha256": _sha(dataset),
            "dataset_manifest_path": "rlm_state/dataset_manifest.json",
            "run_spec_sha256": _sha(run_spec),
            "run_spec_path": "rlm_state/run_spec.json",
        },
        "seed": 7,
        "termination_cause": termination_cause,
    })
    ledger = run / "campaign" / "attempts.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows.insert(-1, {
        "status": "scheduler_receipt",
        "receipt_sha256": _sha(receipt_path),
        "branch_id": "1",
        "attempt_n": 1,
        "campaign_id": campaign_id,
        "paper_ref": ladder.paper_ref,
        "run_spec_sha256": _sha(run_spec),
    })
    audit = rows[-1]["decision"]["asha_authority_audit"]
    audit.update({
        "action": action,
        "applied": True,
        "source_branch_id": "1",
        "source_attempt_n": 1,
        "scheduler_receipt_path": "campaign/scheduler_receipts/1-1.json",
        "scheduler_ladder_path": "campaign/scheduler_step_ladder.json",
        "scheduler_receipt_sha256": _sha(receipt_path),
    })
    if action == "promote":
        rows[-1]["decision"]["next_plan"]["scope_rung"] = 1
    if action == "kill":
        audit["failure_class"] = "training_diverged"
    decision_evidence = run / "campaign" / "scheduler_decisions" / "1.json"
    _json(decision_evidence, {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "ladder_sha256": ladder.sha256,
        "metric_id": ladder.metric_id,
        "direction": ladder.direction,
        "rung": 0,
        "config": {
            "gpu_usd_budget": 5.0,
            "a100_cap": 1,
            "eta": 3.0,
            "noise_floor": 0.0067,
        },
        "cohort": [{
            "branch_id": "1",
            "receipt_path": "campaign/scheduler_receipts/1-1.json",
            "receipt_sha256": _sha(receipt_path),
            "branch_type": "faithful",
            "is_safety_bracket": False,
            "gpu_usd": 1.0,
        }],
        "selected_branch_id": "1",
        "action": action,
    })
    audit.update({
        "scheduler_decision_evidence_path": "campaign/scheduler_decisions/1.json",
        "scheduler_decision_evidence_sha256": _sha(decision_evidence),
    })
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if controller_event_store is not None:
        _append_controller_action_event(
            controller_event_store,
            campaign_id=campaign_id,
            receipt_sha256=_sha(receipt_path),
            audit=audit,
            action=action,
            checkpoint_path="artifact/checkpoint.pt",
            termination_cause=termination_cause,
        )


def _append_controller_action_event(
    path: Path,
    *,
    campaign_id: str,
    receipt_sha256: str,
    audit: dict,
    action: str,
    checkpoint_path: str,
    termination_cause: str | None,
) -> None:
    """Use the real controller EventStore, not a run-local JSON fixture."""
    store = SqliteEventStore(f"sqlite:///{path}")
    audit_sha256 = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if action == "promote":
        event = BranchPromoted(
            branch_id="1",
            from_rung=0,
            to_rung=1,
            receipt_sha256=receipt_sha256,
            authority_audit_sha256=audit_sha256,
            decision_evidence_sha256=audit["scheduler_decision_evidence_sha256"],
        )
    elif action == "freeze":
        event = FrozenPoolEviction(
            branch_id="1",
            rung=0,
            ckpt_uri=checkpoint_path,
            reason="deterministic_rank",
            receipt_sha256=receipt_sha256,
            authority_audit_sha256=audit_sha256,
            decision_evidence_sha256=audit["scheduler_decision_evidence_sha256"],
        )
    elif action == "kill":
        event = BranchTrueKilled(
            branch_id="1",
            termination_cause=str(termination_cause),
            rung=0,
            receipt_sha256=receipt_sha256,
            authority_audit_sha256=audit_sha256,
            decision_evidence_sha256=audit["scheduler_decision_evidence_sha256"],
        )
    else:
        raise AssertionError(f"unsupported controller action fixture: {action}")
    store.append(
        AggregateId(f"branch-tree:{campaign_id}"),
        "branch_tree",
        [event],
        expected_version=store.get_aggregate_version(AggregateId(f"branch-tree:{campaign_id}")),
        envelopes=[EventEnvelope(
            event_id=new_event_id(),
            correlation_id=CorrelationId(campaign_id),
            source="agents.rlm.reproduction_campaign",
        )],
    )


def _rewrite_audit_and_append_event(run: Path, controller_store: Path) -> None:
    """Refresh a fixture audit after changing its decision artifact/decision row."""
    ledger = run / "campaign" / "attempts.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    audit = rows[-1]["decision"]["asha_authority_audit"]
    evidence = run / audit["scheduler_decision_evidence_path"]
    audit["scheduler_decision_evidence_sha256"] = _sha(evidence)
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _append_controller_action_event(
        controller_store,
        campaign_id=run.name,
        receipt_sha256=audit["scheduler_receipt_sha256"],
        audit=audit,
        action="promote",
        checkpoint_path="artifact/checkpoint.pt",
        termination_cause=None,
    )


def _add_second_verified_same_rung_receipt(run: Path) -> None:
    """Add branch/attempt 2 with a valid rung-0 receipt, not in the decision cohort."""
    first_path = run / "campaign" / "scheduler_receipts" / "1-1.json"
    second_path = run / "campaign" / "scheduler_receipts" / "2-2.json"
    second = json.loads(first_path.read_text())
    metrics = run / "artifact" / "metrics-2.json"
    _json(metrics, {"eval.accuracy": 0.9})
    checkpoint = run / "artifact" / "checkpoint-2.pt"
    checkpoint.write_bytes(b"checkpoint-2")
    bundle = run / "rlm_state" / "evidence_bundle-2.json"
    _json(bundle, {
        "schema": 1,
        "coherent": True,
        "metrics_sha256": _sha(metrics),
        "code_tree_digest": "1" * 64,
    })
    second.update({
        "branch_id": "2",
        "attempt_n": 2,
        "cell_id": "cell-2",
        "metric": {
            "id": "eval.accuracy",
            "direction": "maximize",
            "value": 0.9,
            "artifact_path": "artifact/metrics-2.json",
            "sha256": _sha(metrics),
        },
        "checkpoint": {
            **second["checkpoint"],
            "path": "artifact/checkpoint-2.pt",
            "sha256": _sha(checkpoint),
        },
        "evidence_bundle": {"path": "rlm_state/evidence_bundle-2.json", "sha256": _sha(bundle)},
    })
    _json(second_path, second)
    ledger = run / "campaign" / "attempts.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    decision = rows[-1]
    receipt_row = {
        "status": "scheduler_receipt",
        "receipt_sha256": _sha(second_path),
        "branch_id": "2",
        "attempt_n": 2,
        "campaign_id": run.name,
        "paper_ref": "2605.15155",
        "run_spec_sha256": second["fingerprints"]["run_spec_sha256"],
    }
    rows[-1:-1] = [
        {"attempt_n": 2, "status": "launched"},
        {"attempt_n": 2, "status": "assessed", "assessment": {
            "failure_class": None,
            "cost": {"llm_usd": 0.0, "gpu_usd": 1.0, "gpu_hours": 0.0, "wall_s": 0.0},
        }},
        receipt_row,
    ]
    assert decision is rows[-1]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _calibration(scores: list[float], *, run_id: str) -> dict:
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return {
        "run_id": run_id,
        "k": len(scores),
        "overall": {
            "n": len(scores),
            "scores": scores,
            "mean": mean,
            "min": min(scores),
            "max": max(scores),
            "stdev": variance**0.5,
        },
    }


def _run(
    root: Path,
    name: str,
    *,
    authoritative: bool,
    score: float = 0.8,
    terminal_kind: str = "EXHAUSTED",
    advisory: dict | None = None,
    audit: dict | None = None,
    failure_class: str | None = None,
) -> Path:
    run = root / name
    _json(
        run / "campaign" / "campaign.json",
        _campaign_state(terminal_kind=terminal_kind, project_id=name),
    )
    decision = {
        "kind": "CONTINUE",
        "rule": "continue",
        "stop_reason": None,
        "next_plan": {"scope_rung": 0},
        "champion_attempt_n": None,
        "asha_advisory": advisory or _advisory(),
    }
    if authoritative:
        decision["asha_authority_audit"] = audit or _audit()
    rows = [
        {"attempt_n": 1, "status": "launched"},
        {
            "attempt_n": 1,
            "status": "assessed",
            "assessment": {
                "failure_class": failure_class,
                "cost": {"llm_usd": 0.0, "gpu_usd": 1.0, "gpu_hours": 0.0, "wall_s": 0.0},
            },
        },
        {"attempt_n": 1, "status": "decided", "decision": decision},
    ]
    (run / "campaign" / "attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _json(run / "final_report.json", {"rubric": {"overall_score": score}})
    _json(
        run / "tokens_total.json",
        {"schema_version": 1, "run_id": name, "grand_total": {"calls": 1}},
    )
    _json(run / "calibration.json", _calibration([0.79, 0.80, 0.81, 0.80, 0.80], run_id=name))
    return run


def _cost(run: Path, usd: float) -> dict:
    record_id = f"provider-{run.name}"
    _json(
        run / "gpu_observation.json",
        {
            "schema_version": 1,
            "source": "provider_bill_export",
            "records": [{
                "record_id": record_id,
                "run_id": run.name,
                "provider_record_id": f"bill-{run.name}",
                "gpu_sku": "NVIDIA-A100",
                "cost_usd": usd,
            }],
        },
    )
    return {
        "gpu_usd": usd,
        "run_id": run.name,
        "tokens_total_path": "tokens_total.json",
        "gpu_evidence_path": "gpu_observation.json",
        "source": "provider_bill_export",
        "evidence_record_id": record_id,
    }


def _pair(pair_id: str, shadow: Path, authority: Path, *, shadow_usd: float = 10.0, authority_usd: float = 8.0) -> dict:
    return {
        "pair_id": pair_id,
        "shadow": str(shadow),
        "authoritative": str(authority),
        "calibrations": {
            "shadow": "calibration.json",
            "authoritative": "calibration.json",
        },
        "verified_gpu_costs": {
            "shadow": _cost(shadow, shadow_usd),
            "authoritative": _cost(authority, authority_usd),
        },
    }


def _manifest(*pairs: dict) -> dict:
    return {"schema_version": 1, "pairs": list(pairs)}


def test_rejects_self_authored_applied_audit_even_with_bound_costs(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(
            tmp_path,
            f"authority-{n}",
            authoritative=True,
            audit=_audit(applied=n == 0, action="promote" if n == 0 else "continue"),
        )
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert report.applied_action_count == 0
    assert report.total_verified_gpu_savings_usd == 6.0
    assert "authoritative_evidence_receipt_missing" in report.failure_codes


def test_accepts_three_paired_runs_when_applied_action_has_verified_receipt(tmp_path):
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is True
    assert report.applied_action_count == 1
    assert report.total_verified_gpu_savings_usd == 6.0
    assert report.failure_codes == ()


def test_accepts_rung_bound_true_kill_only_for_provable_breakage(tmp_path):
    """A true-delete is valid only when its receipt and controller event agree.

    This positive case complements the malformed-kill rejection below: adding
    ``rung`` to ``BranchTrueKilled`` is not merely schema decoration; the
    adoption gate consumes it to bind the irreversible action to the measured
    fidelity checkpoint.
    """
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(
            tmp_path,
            f"authority-{n}",
            authoritative=True,
            advisory=_advisory(action="kill"),
            failure_class="training_diverged",
        )
        if n == 0:
            _attach_verified_receipt(
                authority,
                action="kill",
                termination_cause="training_diverged",
                controller_event_store=controller_store,
            )
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is True
    assert report.applied_action_count == 1
    assert report.failure_codes == ()


def test_rejects_grade_or_curve_fields_in_canonical_asha_decision_evidence(tmp_path):
    """Selection is recomputed from receipt metrics, never an LLM grade."""
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
            decision_path = authority / "campaign" / "scheduler_decisions" / "1.json"
            evidence = json.loads(decision_path.read_text())
            evidence["llm_grade"] = 0.999  # must not be accepted as a hidden tie-breaker
            _json(decision_path, evidence)
            ledger = authority / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            audit = rows[-1]["decision"]["asha_authority_audit"]
            audit["scheduler_decision_evidence_sha256"] = _sha(decision_path)
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            _append_controller_action_event(
                controller_store,
                campaign_id=authority.name,
                receipt_sha256=audit["scheduler_receipt_sha256"],
                audit=audit,
                action="promote",
                checkpoint_path="artifact/checkpoint.pt",
                termination_cause=None,
            )
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "authoritative_decision_evidence_invalid" in report.failure_codes


def test_rejects_reused_receipt_even_when_audit_hash_and_event_are_modified(tmp_path):
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
            ledger = authority / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            duplicate = json.loads(json.dumps(rows[-1]))
            duplicate_audit = duplicate["decision"]["asha_authority_audit"]
            duplicate_audit["deterministic_evidence_basis"] = "same receipt, altered audit prose"
            rows.append(duplicate)
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            _append_controller_action_event(
                controller_store,
                campaign_id=authority.name,
                receipt_sha256=duplicate_audit["scheduler_receipt_sha256"],
                audit=duplicate_audit,
                action="promote",
                checkpoint_path="artifact/checkpoint.pt",
                termination_cause=None,
            )
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "reused_authority_receipt" in report.failure_codes


def test_invalid_receipt_is_a_rejection_not_an_uncaught_attribute_error(tmp_path):
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
            # Break the receipt's measured-artifact hash after the controller
            # event exists. The gate must fail closed, never dereference None.
            _json(authority / "artifact" / "metrics.json", {"eval.accuracy": 0.7})
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "authoritative_evidence_receipt_invalid" in report.failure_codes


def test_rejects_cohort_that_omits_another_verified_same_rung_receipt_regardless_of_suffix(tmp_path):
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
            _add_second_verified_same_rung_receipt(authority)
            # Receipt verification accepts any regular file under the trusted
            # receipt root; inventory must not let a non-.json filename hide a
            # stronger competitor from the ASHA cohort.
            nested = authority / "campaign" / "scheduler_receipts" / "nested"
            nested.mkdir()
            (authority / "campaign" / "scheduler_receipts" / "2-2.json").rename(
                nested / "2-2.receipt"
            )
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "authoritative_decision_evidence_invalid" in report.failure_codes


def test_rejects_self_authored_width_metadata_that_differs_from_assessment(tmp_path):
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
            decision_path = authority / "campaign" / "scheduler_decisions" / "1.json"
            evidence = json.loads(decision_path.read_text())
            evidence["cohort"][0]["gpu_usd"] = 0.0
            _json(decision_path, evidence)
            _rewrite_audit_and_append_event(authority, controller_store)
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "authoritative_decision_evidence_invalid" in report.failure_codes


def test_rejects_promotion_event_or_live_scope_for_the_wrong_rung(tmp_path):
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
            ledger = authority / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["decision"]["next_plan"]["scope_rung"] = 0
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            _rewrite_audit_and_append_event(authority, controller_store)
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "controller_action_event_invalid" in report.failure_codes


def test_rejects_applied_receipt_without_external_controller_event(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority)
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert report.applied_action_count == 0
    assert "controller_action_event_missing" in report.failure_codes


def test_rejects_controller_store_inside_shadow_worker_arm(tmp_path):
    controller_store = tmp_path / "shadow-0" / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
        pairs.append(_pair(f"p{n}", shadow, authority))

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert report.applied_action_count == 0
    assert "controller_action_event_invalid" in report.failure_codes


def test_rejects_reused_campaign_identity_across_distinct_paths(tmp_path):
    controller_store = tmp_path / "controller-events.db"
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(tmp_path, f"authority-{n}", authoritative=True)
        if n == 0:
            _attach_verified_receipt(authority, controller_event_store=controller_store)
        pairs.append(_pair(f"p{n}", shadow, authority))
    duplicated = json.loads((tmp_path / "authority-1" / "campaign" / "campaign.json").read_text())
    duplicated["project_id"] = "authority-0"
    _json(tmp_path / "authority-1" / "campaign" / "campaign.json", duplicated)

    manifest = _manifest(*pairs)
    manifest["controller_event_store"] = str(controller_store)
    report = evaluate_manifest(manifest, manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "reused_campaign_identity" in report.failure_codes


def test_rejects_noop_audit_pasted_onto_an_incomplete_decision(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(
            tmp_path,
            f"authority-{n}",
            authoritative=True,
            audit=_audit(applied=n == 0, action="promote" if n == 0 else "continue"),
        )
        if n == 0:
            ledger = authority / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["decision"].pop("next_plan")
            rows[-1]["decision"].pop("champion_attempt_n")
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "decision_contract_missing" in report.failure_codes
    assert "unlinked_authority_audit" in report.failure_codes


def test_rejects_fewer_than_three_pairs(tmp_path):
    pairs = [
        _pair(f"p{n}", _run(tmp_path, f"s{n}", authoritative=False), _run(tmp_path, f"a{n}", authoritative=True))
        for n in range(2)
    ]
    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "minimum_pair_count" in report.failure_codes


def test_rejects_missing_shadow_advisory_coverage(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            ledger = shadow / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["decision"].pop("asha_advisory")
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "shadow_advisory_coverage" in report.failure_codes


def test_rejects_unpersisted_or_unapplied_authority_audit(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            ledger = authority / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["decision"].pop("asha_authority_audit")
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "authority_audit_missing" in report.failure_codes
    assert "meaningful_authority_action" in report.failure_codes


def test_rejects_any_nonliteral_training_diverged_kill(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            authority = _run(
                tmp_path,
                f"a{n}",
                authoritative=True,
                audit=_audit(applied=True, action="kill", failure_class="cell_execution_error"),
            )
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "invalid_true_kill" in report.failure_codes


def test_rejects_inadequate_or_noisy_calibration(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            _json(authority / "calibration.json", _calibration([0.5, 0.7, 0.5, 0.7, 0.5], run_id=authority.name))
        if n == 1:
            _json(shadow / "calibration.json", _calibration([0.8, 0.8, 0.8, 0.8], run_id=shadow.name))
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "grader_sigma" in report.failure_codes
    assert "grader_calibration_k" in report.failure_codes


def test_rejects_score_degradation_beyond_pair_sigma(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False, score=0.80)
        authority = _run(tmp_path, f"a{n}", authoritative=True, score=0.75 if n == 0 else 0.80)
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "score_degradation" in report.failure_codes


def test_rejects_ledger_only_or_nonpositive_cost_claims(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        pair = _pair(f"p{n}", shadow, authority, shadow_usd=8.0, authority_usd=8.0)
        if n == 0:
            pair["verified_gpu_costs"]["shadow"] = {
                "gpu_usd": 10.0,
                "run_id": shadow.name,
                "source": "cost_ledger",
                "tokens_total_path": "tokens_total.json",
                "gpu_evidence_path": "cost_ledger.jsonl",
            }
            (shadow / "cost_ledger.jsonl").write_text("{}\n", encoding="utf-8")
        pairs.append(pair)

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "verified_gpu_cost" in report.failure_codes
    assert "positive_verified_gpu_saving" in report.failure_codes


def test_rejects_empty_or_unbound_cost_evidence_even_when_manifest_claims_saving(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        pair = _pair(f"p{n}", shadow, authority)
        if n == 0:
            pair["verified_gpu_costs"]["shadow"]["evidence_record_id"] = "absent"
            _json(shadow / "gpu_observation.json", {"schema_version": 1, "source": "provider_bill_export", "records": []})
        pairs.append(pair)

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "verified_gpu_cost" in report.failure_codes


def test_rejects_unbound_direct_grader_calibration(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            calibration = _calibration([0.79, 0.80, 0.81, 0.80, 0.80], run_id="other-run")
            _json(authority / "calibration.json", calibration)
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "unbound_grader_calibration" in report.failure_codes


def test_rejects_mismatched_terminal_evidence_and_malformed_ledger(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(
            tmp_path,
            f"a{n}",
            authoritative=True,
            terminal_kind="CONTRADICTED" if n == 0 else "EXHAUSTED",
        )
        if n == 1:
            with (shadow / "campaign" / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{ broken json\n")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "terminal_evidence_mismatch" in report.failure_codes
    assert "malformed_campaign_ledger" in report.failure_codes


def test_cli_writes_machine_readable_rejection(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    _json(manifest_path, _manifest())
    assert main([str(manifest_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["eligible_for_operator_review"] is False
    assert "minimum_pair_count" in payload["failure_codes"]
