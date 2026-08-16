"""End-to-end authority-controller tests without a GPU, network, or grade."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.agents.rlm.scheduler_authority_controller import SchedulerAuthorityController
from backend.agents.rlm.scheduler_evidence import PaperStepLadder
from backend.agents.rlm.scheduler_runtime import (
    BranchTemplate,
    SchedulerAuthoritySpec,
    SchedulerRuntimeError,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f10(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _spec() -> SchedulerAuthoritySpec:
    ladder = PaperStepLadder(
        paper_ref="2605.15155", metric_id="eval.accuracy", direction="maximize",
        r_max_steps=50, rung_steps=(10, 50), schedule_source_sha256="a" * 64,
    )
    return SchedulerAuthoritySpec(
        ladder=ladder,
        branches=(
            BranchTemplate("faithful", "faithful", _f10("faithful"), seed=1),
            BranchTemplate("ambiguity", "ambiguity", _f10("ambiguity"), seed=2),
            BranchTemplate("discovery", "discovery", _f10("discovery"), seed=3),
        ),
        gpu_usd_budget=1.0,
        discovery_gpu_usd_budget=1.0,
        a100_cap=2,
    )


def _raw_receipt(root: Path, *, controller: SchedulerAuthorityController, branch_id: str, attempt_n: int, value: float) -> dict:
    branch = controller.branches[branch_id]
    ladder = controller.spec.ladder
    artifact = root / "artifact" / branch_id
    artifact.mkdir(parents=True, exist_ok=True)
    metrics = artifact / "metrics.json"
    # The tempting grade field is intentionally present but has no controller
    # reader. Only ladder.metric_id participates in authority.
    metrics.write_text(json.dumps({ladder.metric_id: value, "final_report": {"score": 0.01}}), encoding="utf-8")
    checkpoint = artifact / "checkpoint.pt"
    checkpoint.write_bytes(branch_id.encode("utf-8"))
    state = artifact / "checkpoint-state.json"
    state_payload = {
        "model_sha256": "b" * 64, "optimizer_sha256": "c" * 64,
        "lr_scheduler_sha256": "d" * 64, "rng_sha256": "e" * 64,
        "data_order_sha256": "f" * 64,
    }
    state.write_text(json.dumps(state_payload), encoding="utf-8")
    rlm_state = root / "rlm_state"
    rlm_state.mkdir(exist_ok=True)
    dataset = rlm_state / "dataset-manifest.json"
    run_spec = rlm_state / "run-spec.json"
    dataset.write_text('{"dataset":"pinned"}', encoding="utf-8")
    run_spec.write_text('{"image":"pinned@sha256"}', encoding="utf-8")
    bundle = rlm_state / f"evidence-{branch_id}.json"
    bundle.write_text(json.dumps({
        "schema": 1, "coherent": True, "metrics_sha256": _sha(metrics), "code_tree_digest": "1" * 64,
    }), encoding="utf-8")
    return {
        "schema_version": 1,
        "campaign_id": "campaign-1",
        "branch_id": branch_id,
        "parent_branch_id": branch.parent_branch_id,
        "attempt_n": attempt_n,
        "cell_id": f"cell-{branch_id}",
        "paper_ref": ladder.paper_ref,
        "ladder_sha256": ladder.sha256,
        "from_step": 0 if branch.rung == 0 else ladder.rung_steps[branch.rung - 1],
        "to_step": ladder.rung_steps[branch.rung],
        "metric": {"id": ladder.metric_id, "direction": ladder.direction, "value": value,
                   "artifact_path": str(metrics.relative_to(root)), "sha256": _sha(metrics)},
        "checkpoint": {"path": str(checkpoint.relative_to(root)), "sha256": _sha(checkpoint),
                       "state": state_payload, "state_path": str(state.relative_to(root)), "state_sha256": _sha(state)},
        "evidence_bundle": {"path": str(bundle.relative_to(root)), "sha256": _sha(bundle)},
        "fingerprints": {"code_sha256": "1" * 64, "dataset_sha256": _sha(dataset),
                         "dataset_manifest_path": str(dataset.relative_to(root)),
                         "run_spec_sha256": _sha(run_spec), "run_spec_path": str(run_spec.relative_to(root))},
        "seed": branch.seed,
        "termination_cause": None,
    }


def _attestor(root: Path):
    ledger = root / "campaign" / "attempts.jsonl"

    def attest(row):
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row)) + "\n")
            handle.flush()

    return attest


def test_controller_fans_out_under_cap_waits_for_full_cohort_and_requeues_promotions(tmp_path):
    events: list[str] = []
    controller = SchedulerAuthorityController(
        tmp_path, campaign_id="campaign-1", spec=_spec(), event_sink=lambda kind, _payload: events.append(kind),
    )
    assert controller.bootstrap() == 3
    assert controller.bootstrap() == 0

    first_wave = controller.claim_launches()
    assert {item.branch_id for item in first_wave} == {"ambiguity", "discovery"}
    assert all(item.to_step == 10 for item in first_wave)
    for attempt_n, launch in enumerate(first_wave, 1):
        score = 0.5 if launch.branch_id == "ambiguity" else 0.1
        controller.record_cell_receipt(_raw_receipt(tmp_path, controller=controller, branch_id=launch.branch_id,
                                                     attempt_n=attempt_n, value=score), attest=_attestor(tmp_path))
    with pytest.raises(SchedulerRuntimeError, match="incomplete"):
        controller.decide_rung(rung=0, provider_gpu_usd_by_branch={"ambiguity": 1.0, "discovery": 1.0})

    second_wave = controller.claim_launches()
    assert [item.branch_id for item in second_wave] == ["faithful"]
    controller.record_cell_receipt(_raw_receipt(tmp_path, controller=controller, branch_id="faithful",
                                                 attempt_n=3, value=0.9), attest=_attestor(tmp_path))
    result = controller.decide_rung(
        rung=0,
        provider_gpu_usd_by_branch={"faithful": 1.0, "ambiguity": 1.0, "discovery": 1.0},
    )
    assert {item.branch_id: item.action for item in result.actions} == {
        "faithful": "promote", "ambiguity": "freeze", "discovery": "promote",
    }
    next_wave = controller.claim_launches()
    assert {item.branch_id for item in next_wave} == {"faithful", "discovery"}
    assert all(item.from_step == 10 and item.to_step == 50 for item in next_wave)
    assert {"branch_spawned", "rung_climbed", "branch_promoted", "frozen_pool_eviction"}.issubset(events)
