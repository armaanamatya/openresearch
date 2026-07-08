"""framework_detector — pure, no-I/O-beyond-reading-code_path fingerprinting.

Covers: a verl fixture (bundled verl/ dir + hydra-signature scripts) detects
with high confidence; a plain non-verl fixture detects nothing; a missing
directory fails soft instead of raising.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm import framework_detector


def _write_verl_fixture(code_dir: Path) -> None:
    verl_dir = code_dir / "verl"
    verl_dir.mkdir(parents=True)
    (verl_dir / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='verl')\n", encoding="utf-8"
    )

    scripts_dir = code_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run.sh").write_text(
        "#!/bin/bash\n"
        "python3 -m demo.main_run \\\n"
        "    algorithm.adv_estimator=grpo \\\n"
        "    actor_rollout_ref.rollout.n=8 \\\n"
        "    data.max_response_length=3072 \\\n"
        "    data.train_files=dataset/train.parquet \\\n"
        "    trainer.n_gpus_per_node=4 \\\n"
        "    trainer.logger=['console','tensorboard']\n",
        encoding="utf-8",
    )

    demo_dir = code_dir / "demo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "main_run.py").write_text("# stub entrypoint\n", encoding="utf-8")

    dataset_dir = code_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.parquet").write_bytes(b"")


def _write_non_verl_fixture(code_dir: Path) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "train.py").write_text(
        "import torch\n\n\ndef main():\n    print('plain training loop')\n",
        encoding="utf-8",
    )
    scripts_dir = code_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run.sh").write_text(
        "#!/bin/bash\npython3 train.py --epochs 1\n", encoding="utf-8"
    )


def test_detect_framework_verl_fixture(tmp_path):
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)

    framework, confidence, evidence = framework_detector.detect_framework(code_dir)

    assert framework == "verl"
    assert confidence >= 0.7
    assert evidence["bundled_verl_dir"] is True
    assert evidence["signature_files"]


def test_detect_framework_verl_signature_only_no_bundled_dir(tmp_path):
    code_dir = tmp_path / "code"
    scripts_dir = code_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run.sh").write_text(
        "#!/bin/bash\npython3 -m some.main_run algorithm.adv_estimator=grpo\n",
        encoding="utf-8",
    )

    framework, confidence, evidence = framework_detector.detect_framework(code_dir)

    assert framework == "verl"
    assert confidence == 0.7
    assert evidence["bundled_verl_dir"] is False


def test_detect_framework_non_verl_fixture(tmp_path):
    code_dir = tmp_path / "code"
    _write_non_verl_fixture(code_dir)

    framework, confidence, evidence = framework_detector.detect_framework(code_dir)

    assert framework == "unknown"
    assert confidence == 0.0
    assert evidence["signature_files"] == []


def test_detect_framework_missing_dir_fails_soft(tmp_path):
    missing = tmp_path / "does_not_exist"

    framework, confidence, evidence = framework_detector.detect_framework(missing)

    assert framework == "unknown"
    assert confidence == 0.0
