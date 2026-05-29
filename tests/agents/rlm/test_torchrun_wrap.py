"""_maybe_torchrun_wrap: re-launch FSDP/DDP train scripts under torchrun on multi-GPU.

The 2026-05-29 SDAR bug: the agent wrote FSDP code but commands.json was
`python train.py`, so it ran single-process (WORLD_SIZE=1) → only GPU 0 used →
large models OOM. The safety-net rewrites the launch to torchrun.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm.primitives import _maybe_torchrun_wrap


def _train(tmp_path: Path, body: str) -> Path:
    (tmp_path / "train.py").write_text(body, encoding="utf-8")
    return tmp_path


def test_wraps_fsdp_script_on_multi_gpu(tmp_path):
    code = _train(tmp_path, "from torch.distributed.fsdp import FullyShardedDataParallel\n")
    out = _maybe_torchrun_wrap(["python train.py --foo 1"], code, 4)
    assert out == ["torchrun --standalone --nproc_per_node=4 train.py --foo 1"]


def test_noop_single_gpu(tmp_path):
    code = _train(tmp_path, "FullyShardedDataParallel\n")
    assert _maybe_torchrun_wrap(["python train.py"], code, 1) == ["python train.py"]


def test_noop_non_distributed(tmp_path):
    code = _train(tmp_path, "import torch\nmodel.train()\n")
    assert _maybe_torchrun_wrap(["python train.py"], code, 4) == ["python train.py"]


def test_noop_already_torchrun(tmp_path):
    code = _train(tmp_path, "DistributedDataParallel\n")
    cmds = ["torchrun --nproc_per_node=4 train.py"]
    assert _maybe_torchrun_wrap(cmds, code, 4) == cmds


def test_other_commands_untouched(tmp_path):
    code = _train(tmp_path, "init_process_group('nccl')\n")
    out = _maybe_torchrun_wrap(["pip install -r requirements.txt", "python train.py"], code, 2)
    assert out == [
        "pip install -r requirements.txt",
        "torchrun --standalone --nproc_per_node=2 train.py",
    ]


def test_missing_script_noop(tmp_path):
    # referenced script absent → no markers → no wrap (never raises)
    assert _maybe_torchrun_wrap(["python ghost.py"], tmp_path, 4) == ["python ghost.py"]
