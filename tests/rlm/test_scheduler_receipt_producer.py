from __future__ import annotations
import hashlib
import json
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
    (run_dir / "cellA").mkdir()
    ckpt_components = run_dir / "cellA" / "checkpoint_components"
    payloads = _write_checkpoint_components(ckpt_components)
    out = materialize_checkpoint(
        run_dir=run_dir, cell_output_dir=run_dir / "cellA",
        checkpoint_components_dir=ckpt_components,
    )
    bundle = run_dir / out["path"]
    assert bundle.is_file()
    assert out["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    state = out["state"]
    assert set(state) == {"model_sha256", "optimizer_sha256", "lr_scheduler_sha256", "rng_sha256", "data_order_sha256"}
    assert state["model_sha256"] == hashlib.sha256(payloads["model"]).hexdigest()
    state_file = run_dir / out["state_path"]
    assert json.loads(state_file.read_text()) == state
    assert out["state_sha256"] == hashlib.sha256(state_file.read_bytes()).hexdigest()
    assert not Path(out["path"]).is_absolute() and not Path(out["state_path"]).is_absolute()

def test_materialize_checkpoint_fails_closed_on_missing_component(tmp_path):
    (tmp_path / "cellB").mkdir()
    ckpt = tmp_path / "cellB" / "checkpoint_components"
    ckpt.mkdir(parents=True)
    (ckpt / "model").write_bytes(b"only-model")   # missing the other 4
    with pytest.raises(ValueError, match="checkpoint component"):
        materialize_checkpoint(run_dir=tmp_path, cell_output_dir=tmp_path / "cellB",
                               checkpoint_components_dir=ckpt)

def _ladder():
    from backend.agents.rlm.scheduler_evidence import PaperStepLadder
    return PaperStepLadder(paper_ref="1412.6980", metric_id="eval.accuracy", direction="maximize",
                           r_max_steps=50, rung_steps=(10, 50), schedule_source_sha256="a"*64)

def _write_cell_evidence(run_dir, cell_dir, *, metric_id, metric_value):
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "metrics.json").write_text(json.dumps({metric_id: metric_value, "final_report": {"score": 0.01}}))
    comps = cell_dir / "checkpoint_components"; _write_checkpoint_components(comps)
    rlm = run_dir / "rlm_state"; rlm.mkdir(exist_ok=True)
    (rlm / "dataset-manifest.json").write_text('{"dataset":"mnist-pinned"}')
    (rlm / "run-spec.json").write_text('{"image":"pinned@sha256"}')
    return comps

def test_build_raw_receipt_round_trips_through_write_verified_receipt(tmp_path):
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
    from backend.agents.rlm.scheduler_evidence import write_verified_receipt, load_verified_receipt
    run_dir = tmp_path; ladder = _ladder()
    comps = _write_cell_evidence(run_dir, run_dir / "cellA", metric_id=ladder.metric_id, metric_value=0.9)
    raw = build_raw_receipt(run_dir=run_dir, cell_output_dir=run_dir / "cellA", checkpoint_components_dir=comps,
        ladder=ladder, campaign_id="campaign-1", branch_id="faithful", parent_branch_id=None, attempt_n=1,
        cell_id="cell-faithful", from_step=0, to_step=10, seed=1, termination_cause=None,
        dataset_manifest_path=run_dir/"rlm_state"/"dataset-manifest.json", run_spec_path=run_dir/"rlm_state"/"run-spec.json")
    assert raw["metric"]["value"] == 0.9 and raw["metric"]["id"] == ladder.metric_id
    ledger = run_dir / "campaign" / "attempts.jsonl"
    def attest(row):
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as f:
            f.write(json.dumps(dict(row)) + "\n")
    path = write_verified_receipt(raw, ladder=ladder, run_dir=run_dir, campaign_id="campaign-1", attest=attest)
    assert load_verified_receipt(path, ladder=ladder, run_dir=run_dir, expected_campaign_id="campaign-1") is not None

def test_build_raw_receipt_ignores_grade_when_metric_present(tmp_path):
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
    run_dir = tmp_path; ladder = _ladder()
    comps = _write_cell_evidence(run_dir, run_dir / "cellB", metric_id=ladder.metric_id, metric_value=0.42)
    raw = build_raw_receipt(run_dir=run_dir, cell_output_dir=run_dir / "cellB", checkpoint_components_dir=comps,
        ladder=ladder, campaign_id="c", branch_id="b", parent_branch_id=None, attempt_n=1, cell_id="cell-b",
        from_step=0, to_step=10, seed=1, termination_cause=None,
        dataset_manifest_path=run_dir/"rlm_state"/"dataset-manifest.json", run_spec_path=run_dir/"rlm_state"/"run-spec.json")
    assert raw["metric"]["value"] == 0.42          # never 0.01 (the planted grade)

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

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_build_raw_receipt_non_finite_metric_fails_closed(tmp_path, bad):
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
    run_dir = tmp_path; ladder = _ladder()
    cell = run_dir / "cellN"; cell.mkdir()
    # json.dump can't emit NaN/Inf by default; write it so json.loads(...) parses it back
    (cell / "metrics.json").write_text(json.dumps({ladder.metric_id: bad}))  # allow_nan=True default emits NaN/Infinity
    comps = cell / "checkpoint_components"; _write_checkpoint_components(comps)
    (run_dir/"rlm_state").mkdir(); (run_dir/"rlm_state"/"dataset-manifest.json").write_text("{}"); (run_dir/"rlm_state"/"run-spec.json").write_text("{}")
    with pytest.raises(ValueError, match="finite"):
        build_raw_receipt(run_dir=run_dir, cell_output_dir=cell, checkpoint_components_dir=comps, ladder=ladder,
            campaign_id="c", branch_id="b", parent_branch_id=None, attempt_n=1, cell_id="cn", from_step=0, to_step=10,
            seed=1, termination_cause=None, dataset_manifest_path=run_dir/"rlm_state"/"dataset-manifest.json",
            run_spec_path=run_dir/"rlm_state"/"run-spec.json")

def test_code_sha256_is_code_only_stable_across_checkpoint_contents(tmp_path):
    from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
    ladder = _ladder()
    def _one(run_dir, cell_name, ckpt_seed):
        cell = run_dir / cell_name; cell.mkdir(parents=True)
        (cell / "metrics.json").write_text(json.dumps({ladder.metric_id: 0.5}))
        (cell / "train_cell.py").write_text("print('identical code')\n")   # identical CODE
        comps = cell / "checkpoint_components"; comps.mkdir()
        for name in _COMPONENTS:
            (comps / name).write_bytes(f"{name}-{ckpt_seed}".encode())      # DIFFERENT weights
        (run_dir/"rlm_state").mkdir(exist_ok=True)
        (run_dir/"rlm_state"/"dataset-manifest.json").write_text("{}"); (run_dir/"rlm_state"/"run-spec.json").write_text("{}")
        return build_raw_receipt(run_dir=run_dir, cell_output_dir=cell, checkpoint_components_dir=comps, ladder=ladder,
            campaign_id="c", branch_id=cell_name, parent_branch_id=None, attempt_n=1, cell_id=cell_name, from_step=0,
            to_step=10, seed=1, termination_cause=None, dataset_manifest_path=run_dir/"rlm_state"/"dataset-manifest.json",
            run_spec_path=run_dir/"rlm_state"/"run-spec.json")
    r1 = _one(tmp_path / "r1", "cellX", "aaa")
    r2 = _one(tmp_path / "r2", "cellX", "bbb")
    assert r1["fingerprints"]["code_sha256"] == r2["fingerprints"]["code_sha256"]
