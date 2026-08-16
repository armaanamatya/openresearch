"""The trainer-facing 5-field checkpoint contract (Phase C prerequisite).

Proves the harness-owned WRITER (``cell_checkpoint.write_checkpoint``) emits a
layout the receipt producer + verifier accept — i.e. a trainer using the
contract produces a VERIFIABLE scheduler receipt end-to-end.
"""
from __future__ import annotations

import json

import pytest

from backend.agents.rlm import cell_checkpoint

_COMPONENTS = ("model", "optimizer", "lr_scheduler", "rng", "data_order")


def test_write_then_latest_then_read_round_trips(tmp_path):
    ckpt = tmp_path / "checkpoints"
    cell_checkpoint.write_checkpoint(
        ckpt, 0, model=b"m0", optimizer=b"o0", lr_scheduler=b"l0", rng=b"r0", data_order=b"d0"
    )
    dir10 = cell_checkpoint.write_checkpoint(
        ckpt, 10, model=b"m10", optimizer=b"o10", lr_scheduler=b"l10", rng=b"r10", data_order=b"d10"
    )
    latest = cell_checkpoint.latest_checkpoint_dir(ckpt)
    assert latest == dir10                       # numeric max, not lexical (step_10 > step_2/step_0)
    assert latest.name == "step_10"
    blobs = cell_checkpoint.read_checkpoint(latest)
    assert blobs == {
        "model": b"m10", "optimizer": b"o10", "lr_scheduler": b"l10",
        "rng": b"r10", "data_order": b"d10",
    }
    # The dir directly contains the 5 exact component filenames the producer reads.
    assert {p.name for p in latest.iterdir()} == set(_COMPONENTS)


def test_latest_checkpoint_dir_none_when_empty(tmp_path):
    assert cell_checkpoint.latest_checkpoint_dir(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert cell_checkpoint.latest_checkpoint_dir(tmp_path / "empty") is None


def test_write_checkpoint_fails_closed_on_non_bytes(tmp_path):
    ckpt = tmp_path / "checkpoints"
    with pytest.raises(TypeError):
        cell_checkpoint.write_checkpoint(
            ckpt, 5, model=b"m", optimizer="not-bytes", lr_scheduler=b"l", rng=b"r", data_order=b"d"
        )
    # No partial step_<n> dir left behind — validation runs before any mkdir.
    assert not (ckpt / "step_5").exists()
    assert cell_checkpoint.latest_checkpoint_dir(ckpt) is None


def test_read_checkpoint_fails_closed_on_missing_component(tmp_path):
    comps = tmp_path / "step_1"
    comps.mkdir()
    (comps / "model").write_bytes(b"only-model")
    with pytest.raises(ValueError, match="checkpoint component missing"):
        cell_checkpoint.read_checkpoint(comps)


def _ladder():
    from backend.agents.rlm.scheduler_evidence import PaperStepLadder

    return PaperStepLadder(
        paper_ref="1412.6980", metric_id="eval.accuracy", direction="maximize",
        r_max_steps=50, rung_steps=(10, 50), schedule_source_sha256="a" * 64,
    )


def test_written_checkpoint_is_receipt_ready(tmp_path):
    """A stub trainer's checkpoint round-trips into a VERIFIED scheduler receipt.

    This is the load-bearing test: it proves the writer's on-disk layout is
    exactly what ``materialize_checkpoint`` + ``_verify_checkpoint`` expect.
    """
    from backend.agents.rlm.scheduler_evidence import (
        load_verified_receipt,
        write_verified_receipt,
    )
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt

    run_dir = tmp_path
    ladder = _ladder()

    cell_dir = run_dir / "cellA"
    cell_dir.mkdir()
    (cell_dir / "metrics.json").write_text(
        json.dumps({ladder.metric_id: 0.9, "final_report": {"score": 0.01}})
    )
    rlm = run_dir / "rlm_state"
    rlm.mkdir()
    (rlm / "dataset-manifest.json").write_text('{"dataset":"mnist-pinned"}')
    (rlm / "run-spec.json").write_text('{"image":"pinned@sha256"}')

    # A stub "trainer" uses the contract to emit a receipt-ready checkpoint. The
    # checkpoint dir lives OUTSIDE cell_output_dir so the raw component bytes never
    # enter the code_tree digest.
    ckpt_dir = run_dir / "cell_ckpt"
    cell_checkpoint.write_checkpoint(
        ckpt_dir, 10, model=b"m", optimizer=b"o", lr_scheduler=b"l", rng=b"r", data_order=b"d"
    )
    components_dir = cell_checkpoint.latest_checkpoint_dir(ckpt_dir)
    assert components_dir is not None

    raw = build_raw_receipt(
        run_dir=run_dir, cell_output_dir=cell_dir, checkpoint_components_dir=components_dir,
        ladder=ladder, campaign_id="campaign-1", branch_id="faithful", parent_branch_id=None,
        attempt_n=1, cell_id="cell-faithful", from_step=0, to_step=10, seed=1, termination_cause=None,
        dataset_manifest_path=rlm / "dataset-manifest.json", run_spec_path=rlm / "run-spec.json",
    )
    assert raw["metric"]["value"] == 0.9 and raw["metric"]["id"] == ladder.metric_id
    assert set(raw["checkpoint"]["state"]) == {
        "model_sha256", "optimizer_sha256", "lr_scheduler_sha256", "rng_sha256", "data_order_sha256",
    }

    ledger = run_dir / "campaign" / "attempts.jsonl"

    def attest(row):
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as f:
            f.write(json.dumps(dict(row)) + "\n")

    path = write_verified_receipt(
        raw, ladder=ladder, run_dir=run_dir, campaign_id="campaign-1", attest=attest
    )
    assert load_verified_receipt(
        path, ladder=ladder, run_dir=run_dir, expected_campaign_id="campaign-1"
    ) is not None
