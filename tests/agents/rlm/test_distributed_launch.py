"""_resolve_distributed_launch: dynamically re-launch FSDP/accelerate train
scripts under ``accelerate launch`` + a harness FSDP2 config on multi-GPU.

Replaces the torchrun-era wrap (2026-05-30). Bare ``torchrun`` only *replicates*
(DDP) unless the agent hand-wrote FSDP, so big models still OOM; the harness now
owns a correct FSDP2 launch and the agent only has to call
``accelerator.prepare(...)``. Dynamic by design: <=1 GPU runs verbatim (FSDP on a
single card is pure all-gather overhead), >=2 GPUs shards. Deterministic — no
sandbox, no LLM.
"""
from __future__ import annotations

import re
from pathlib import Path

from backend.agents.rlm.primitives import (
    _free_tcp_port,
    _resolve_distributed_launch,
    _write_fsdp2_accelerate_config,
)

_CFG = "_reprolab_fsdp2.yaml"


def _train(tmp_path: Path, body: str) -> Path:
    (tmp_path / "train.py").write_text(body, encoding="utf-8")
    return tmp_path


def test_wraps_fsdp_script_on_multi_gpu(tmp_path):
    code = _train(tmp_path, "from torch.distributed.fsdp import fully_shard\n")
    out = _resolve_distributed_launch(["python train.py --foo 1"], code, 4)
    assert len(out) == 1
    cmd = out[0]
    assert cmd.startswith(f"accelerate launch --config_file {_CFG} ")
    assert "--num_processes 4" in cmd
    assert "--num_machines 1" in cmd
    assert re.search(r"--main_process_port \d+", cmd)
    assert cmd.endswith("train.py --foo 1")
    # harness FSDP2 config materialized
    assert (code / _CFG).exists()
    assert "fsdp_version: 2" in (code / _CFG).read_text(encoding="utf-8")


def test_wraps_accelerate_api_script(tmp_path):
    code = _train(tmp_path, "from accelerate import Accelerator\nacc = Accelerator()\n")
    out = _resolve_distributed_launch(["python train.py"], code, 2)
    assert out[0].startswith(f"accelerate launch --config_file {_CFG} ")
    assert "--num_processes 2" in out[0]


def test_noop_single_gpu(tmp_path):
    code = _train(tmp_path, "fully_shard\n")
    assert _resolve_distributed_launch(["python train.py"], code, 1) == ["python train.py"]
    assert not (code / _CFG).exists()  # no launch rewrite → no config written


def test_noop_zero_gpu(tmp_path):
    code = _train(tmp_path, "from accelerate import Accelerator\n")
    assert _resolve_distributed_launch(["python train.py"], code, 0) == ["python train.py"]


def test_noop_non_distributed(tmp_path):
    code = _train(tmp_path, "import torch\nmodel.train()\n")
    assert _resolve_distributed_launch(["python train.py"], code, 4) == ["python train.py"]
    assert not (code / _CFG).exists()


def test_noop_already_accelerate_launch(tmp_path):
    code = _train(tmp_path, "from accelerate import Accelerator\n")
    cmds = ["accelerate launch --num_processes 4 train.py"]
    assert _resolve_distributed_launch(cmds, code, 4) == cmds


def test_noop_already_torchrun(tmp_path):
    code = _train(tmp_path, "DistributedDataParallel\n")
    cmds = ["torchrun --nproc_per_node=4 train.py"]
    assert _resolve_distributed_launch(cmds, code, 4) == cmds


def test_other_commands_untouched(tmp_path):
    code = _train(tmp_path, "init_process_group('nccl')\n")
    out = _resolve_distributed_launch(
        ["pip install -r requirements.txt", "python train.py"], code, 2
    )
    assert out[0] == "pip install -r requirements.txt"
    assert out[1].startswith(f"accelerate launch --config_file {_CFG} ")
    assert "--num_processes 2" in out[1]


def test_missing_script_noop(tmp_path):
    # referenced script absent → no markers → no wrap (never raises)
    assert _resolve_distributed_launch(["python ghost.py"], tmp_path, 4) == ["python ghost.py"]


def test_disable_toggle(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_DISABLE_TORCHRUN_WRAP", "1")
    code = _train(tmp_path, "fully_shard\n")
    assert _resolve_distributed_launch(["python train.py"], code, 4) == ["python train.py"]


def test_free_tcp_port_is_distinct_and_valid():
    p1, p2 = _free_tcp_port(), _free_tcp_port()
    assert isinstance(p1, int) and 1024 <= p1 <= 65535
    assert isinstance(p2, int) and 1024 <= p2 <= 65535


def test_fsdp2_config_contents(tmp_path):
    path = _write_fsdp2_accelerate_config(tmp_path, 4)
    txt = path.read_text(encoding="utf-8")
    assert "distributed_type: FSDP" in txt
    assert "fsdp_version: 2" in txt
    assert "num_processes: 4" in txt
    assert "mixed_precision: bf16" in txt
    assert "TRANSFORMER_BASED_WRAP" in txt
    assert "fsdp_offload_params: false" in txt
