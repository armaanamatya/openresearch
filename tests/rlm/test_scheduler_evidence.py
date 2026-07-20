"""Hermetic authority-receipt verification tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.agents.rlm.scheduler_evidence import PaperStepLadder, load_verified_receipt


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ladder() -> PaperStepLadder:
    return PaperStepLadder(
        paper_ref="2605.15155",
        metric_id="eval.accuracy",
        direction="maximize",
        r_max_steps=100,
        rung_steps=(10, 50, 100),
        schedule_source_sha256="a" * 64,
    )


def _receipt(root: Path, ladder: PaperStepLadder) -> Path:
    metrics = root / "artifact" / "metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps({"eval.accuracy": 0.8, "final_report": {"score": 0.01}}), encoding="utf-8")
    checkpoint = root / "artifact" / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    checkpoint_state = root / "artifact" / "checkpoint-state.json"
    checkpoint_state.write_text(
        json.dumps(
            {
                "model_sha256": "b" * 64,
                "optimizer_sha256": "c" * 64,
                "lr_scheduler_sha256": "d" * 64,
                "rng_sha256": "e" * 64,
                "data_order_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    dataset_manifest = root / "rlm_state" / "dataset-manifest.json"
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset_manifest.write_text('{"dataset":"pinned"}', encoding="utf-8")
    run_spec = root / "rlm_state" / "run-spec.json"
    run_spec.write_text('{"image":"pinned@sha256"}', encoding="utf-8")
    bundle = root / "rlm_state" / "evidence_bundle.json"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(
        json.dumps(
            {
                "schema": 1,
                "coherent": True,
                "metrics_sha256": _sha(metrics),
                "code_tree_digest": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "campaign_id": "campaign-1",
        "branch_id": "branch-1",
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
            "state": {
                "model_sha256": "b" * 64,
                "optimizer_sha256": "c" * 64,
                "lr_scheduler_sha256": "d" * 64,
                "rng_sha256": "e" * 64,
                "data_order_sha256": "f" * 64,
            },
            "state_path": "artifact/checkpoint-state.json",
            "state_sha256": _sha(checkpoint_state),
        },
        "evidence_bundle": {"path": "rlm_state/evidence_bundle.json", "sha256": _sha(bundle)},
        "fingerprints": {
            "code_sha256": "1" * 64,
            "dataset_sha256": _sha(dataset_manifest),
            "dataset_manifest_path": "rlm_state/dataset-manifest.json",
            "run_spec_sha256": _sha(run_spec),
            "run_spec_path": "rlm_state/run-spec.json",
        },
        "seed": 7,
        "termination_cause": None,
    }
    receipt = root / "campaign" / "scheduler_receipts" / "branch-1-1.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    ledger = root / "campaign" / "attempts.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "status": "scheduler_receipt",
                "receipt_sha256": _sha(receipt),
                "branch_id": "branch-1",
                "attempt_n": 1,
                "paper_ref": ladder.paper_ref,
                "run_spec_sha256": _sha(run_spec),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def test_accepts_complete_harness_bound_receipt_and_never_uses_grade(tmp_path):
    ladder = _ladder()
    receipt_path = _receipt(tmp_path, ladder)

    receipt = load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path)

    assert receipt is not None
    assert receipt.metric_value == 0.8
    assert receipt.to_step == 10


def test_rejects_tampered_metric_checkpoint_bundle_and_ladder(tmp_path):
    ladder = _ladder()
    receipt_path = _receipt(tmp_path, ladder)
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is not None

    (tmp_path / "artifact" / "metrics.json").write_text('{"eval.accuracy": 0.9}', encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    payload = json.loads(receipt_path.read_text())
    payload["checkpoint"]["state"].pop("rng_sha256")
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    payload = json.loads(receipt_path.read_text())
    payload["to_step"] = 11
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    payload = json.loads(receipt_path.read_text())
    payload["from_step"] = 9
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None


def test_rejects_uncoherent_bundle_and_outside_artifact_path(tmp_path):
    ladder = _ladder()
    receipt_path = _receipt(tmp_path, ladder)
    bundle = tmp_path / "rlm_state" / "evidence_bundle.json"
    bundle.write_text(json.dumps({"schema": 1, "coherent": False}), encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    payload = json.loads(receipt_path.read_text())
    payload["metric"]["artifact_path"] = "../../outside.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    bundle = tmp_path / "rlm_state" / "evidence_bundle.json"
    payload = json.loads(bundle.read_text())
    payload["metrics_sha256"] = "0" * 64
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None


def test_rejects_unattested_absolute_symlink_and_boolean_metric_receipts(tmp_path):
    ladder = _ladder()
    receipt_path = _receipt(tmp_path, ladder)
    (tmp_path / "campaign" / "attempts.jsonl").unlink()
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    payload = json.loads(receipt_path.read_text())
    payload["metric"]["artifact_path"] = str(tmp_path / "artifact" / "metrics.json")
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    metrics = tmp_path / "artifact" / "metrics.json"
    metrics.write_text('{"eval.accuracy": true}', encoding="utf-8")
    payload = json.loads(receipt_path.read_text())
    payload["metric"]["sha256"] = _sha(metrics)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    ledger = tmp_path / "campaign" / "attempts.jsonl"
    row = json.loads(ledger.read_text())
    row["receipt_sha256"] = _sha(receipt_path)
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    real_metrics = tmp_path / "artifact" / "real-metrics.json"
    real_metrics.write_text('{"eval.accuracy": 0.8}', encoding="utf-8")
    metrics_link = tmp_path / "artifact" / "metrics-link.json"
    metrics_link.symlink_to(real_metrics)
    payload = json.loads(receipt_path.read_text())
    payload["metric"]["artifact_path"] = "artifact/metrics-link.json"
    payload["metric"]["sha256"] = _sha(real_metrics)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    row = json.loads((tmp_path / "campaign" / "attempts.jsonl").read_text())
    row["receipt_sha256"] = _sha(receipt_path)
    (tmp_path / "campaign" / "attempts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_verified_receipt(receipt_path, ladder=ladder, run_dir=tmp_path) is None

    receipt_path = _receipt(tmp_path, ladder)
    receipt_link = tmp_path / "campaign" / "scheduler_receipts" / "receipt-link.json"
    receipt_link.symlink_to(receipt_path)
    assert load_verified_receipt(receipt_link, ladder=ladder, run_dir=tmp_path) is None
