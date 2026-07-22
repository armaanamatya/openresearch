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
