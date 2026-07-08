"""execute_planner.plan — pure/deterministic launch + downscale extraction.

Covers: a verl fixture yields an ExecuteSpec whose launch.command carries the
fixture's module verbatim, whose overrides include the 1xA100 downscale knobs,
and which NEVER downscales data.max_response_length (the authors' value must
survive verbatim — truncating it is a known false-degenerate-reward trap);
backslash-continuation joining (the fixture script's real shape); a non-verl
fixture falls through to None.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm import execute_planner


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


def test_plan_returns_execute_spec_for_verl_fixture(tmp_path):
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)

    spec = execute_planner.plan(code_dir)

    assert spec is not None
    assert spec.framework == "verl"
    assert spec.image_key == "verl"
    assert spec.source == "deterministic"
    assert spec.confidence >= 0.7

    # the authors' module + args survive verbatim
    assert "demo.main_run" in spec.launch.command
    assert "algorithm.adv_estimator=grpo" in spec.launch.command
    # backslash-continuations across 7 physical lines joined into one command
    assert "trainer.logger=['console','tensorboard']" in spec.launch.command

    # CRITICAL: max_response_length is never downscaled — authors' value kept
    assert "data.max_response_length=3072" in spec.launch.command
    assert "data.max_response_length" not in spec.launch.overrides

    # the 1xA100 downscale knobs are present, applied AFTER the authors' args
    assert spec.launch.overrides["trainer.n_gpus_per_node"] == "1"
    assert (
        spec.launch.overrides["actor_rollout_ref.actor.fsdp_config.param_offload"]
        == "True"
    )
    assert (
        spec.launch.overrides["actor_rollout_ref.actor.fsdp_config.optimizer_offload"]
        == "True"
    )
    assert (
        spec.launch.overrides["actor_rollout_ref.ref.fsdp_config.param_offload"]
        == "True"
    )

    assert spec.data_slice == {"train_file": "dataset/train.parquet", "slice_rows": 32}
    assert spec.reward.kind == "verl"
    assert spec.reward.keys[0] == "critic/rewards/mean"
    assert spec.setup == ("pip install -e verl --no-deps --no-build-isolation",)


def test_plan_data_slice_none_when_train_file_missing(tmp_path):
    code_dir = tmp_path / "code"
    _write_verl_fixture(code_dir)
    (code_dir / "dataset" / "train.parquet").unlink()

    spec = execute_planner.plan(code_dir)

    assert spec is not None
    assert spec.data_slice is None


def test_plan_returns_none_for_non_verl_fixture(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("print('hello')\n", encoding="utf-8")

    assert execute_planner.plan(code_dir) is None


def test_plan_returns_none_when_no_launch_script_found(tmp_path):
    code_dir = tmp_path / "code"
    verl_dir = code_dir / "verl"
    verl_dir.mkdir(parents=True)
    (verl_dir / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    # a signature token exists in a scanned .py file (so detect_framework says
    # verl, confidently) but no scripts*/run*/train*/examples/** script
    # actually invokes a `python -m <module>` entrypoint — plan() must fall
    # through to None rather than raise or guess.
    config_dir = code_dir / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "defaults.py").write_text(
        "DEFAULTS = {'actor_rollout_ref': {}}\n", encoding="utf-8"
    )

    framework, confidence, _ = execute_planner.detect_framework(code_dir)
    assert framework == "verl" and confidence >= 0.7  # sanity: detection fired
    assert execute_planner.plan(code_dir) is None


# --- shell-variable resolution (the REAL run_ucpo_1.5b.sh shape) ---------------------

def _write_shell_var_verl_fixture(code_dir: Path) -> None:
    """A fixture in the real launch-script shape: VAR=literal assignments
    (incl. a nested X=${Y}/z) above a backslash-continued python3 -m invocation
    whose hydra args reference those vars, with a trailing $@."""
    verl_dir = code_dir / "verl"
    verl_dir.mkdir(parents=True)
    (verl_dir / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='verl')\n", encoding="utf-8"
    )

    scripts_dir = code_dir / "scripts_c"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run_ucpo_1.5b.sh").write_text(
        "#!/bin/bash\n"
        "set -x\n"
        "NUM_EPISODES=3\n"
        "EXP_NAME=run_ucpo_0.2\n"
        "TRAIN_DATADIR=./dataset/train_data_10k.parquet\n"
        "VAL_DATADIR=./dataset/valid_data.parquet\n"
        "MODELDIR=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\n"
        "SAVE_DIR=../checkpoint_ds/ds_1.5b_${EXP_NAME}/\n"
        "export HYDRA_FULL_ERROR=1\n"
        "python3 -m demo.main_run \\\n"
        "    algorithm.adv_estimator=iq \\\n"
        "    data.train_files=$TRAIN_DATADIR \\\n"
        "    data.val_files=$VAL_DATADIR \\\n"
        "    data.max_response_length=3072 \\\n"
        "    actor_rollout_ref.model.path=$MODELDIR \\\n"
        "    trainer.default_local_dir=$SAVE_DIR \\\n"
        "    trainer.total_epochs=$NUM_EPISODES $@\n",
        encoding="utf-8",
    )

    demo_dir = code_dir / "demo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "main_run.py").write_text("# stub entrypoint\n", encoding="utf-8")

    dataset_dir = code_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train_data_10k.parquet").write_bytes(b"")


def test_plan_resolves_shell_vars_in_launch_command(tmp_path):
    code_dir = tmp_path / "code"
    _write_shell_var_verl_fixture(code_dir)

    spec = execute_planner.plan(code_dir)

    assert spec is not None
    cmd = spec.launch.command

    # resolved literals substituted for the shell vars
    assert "data.train_files=./dataset/train_data_10k.parquet" in cmd
    assert (
        "actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" in cmd
    )
    assert "trainer.total_epochs=3" in cmd
    # nested VAR (${EXP_NAME}) resolved inside SAVE_DIR
    assert "trainer.default_local_dir=../checkpoint_ds/ds_1.5b_run_ucpo_0.2/" in cmd

    # no unresolved shell vars and no shell arg-passthrough survive
    assert "$" not in cmd
    assert "$@" not in cmd

    # the authors' response budget is still preserved verbatim (never downscaled)
    assert "data.max_response_length=3072" in cmd

    # slicing now sees the resolved .parquet
    assert spec.data_slice == {
        "train_file": "./dataset/train_data_10k.parquet",
        "slice_rows": 32,
    }


def test_plan_leaves_command_substitution_unresolved(tmp_path):
    code_dir = tmp_path / "code"
    verl_dir = code_dir / "verl"
    verl_dir.mkdir(parents=True)
    (verl_dir / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='verl')\n", encoding="utf-8"
    )
    scripts_dir = code_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    # STAMP is command-substitution -> must NOT be resolved, and must not crash.
    (scripts_dir / "run.sh").write_text(
        "#!/bin/bash\n"
        "STAMP=$(date)\n"
        "python3 -m demo.main_run \\\n"
        "    algorithm.adv_estimator=iq \\\n"
        "    trainer.experiment_name=$STAMP\n",
        encoding="utf-8",
    )
    demo_dir = code_dir / "demo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "main_run.py").write_text("# stub\n", encoding="utf-8")

    spec = execute_planner.plan(code_dir)

    assert spec is not None  # does not crash
    # unresolved command-substitution var stays literal (never guessed)
    assert "trainer.experiment_name=$STAMP" in spec.launch.command


def test_collect_shell_vars_rejects_metachars_and_resolves_nested():
    text = (
        "A=hello\n"
        "export B='quoted value'\n"
        "C=${A}_world\n"
        "D=$(uname)\n"
        "E=a|b\n"
        "F=$C/leaf\n"
    )
    resolved = execute_planner._collect_shell_vars(text)

    assert resolved["A"] == "hello"
    assert resolved["B"] == "quoted value"      # one quote layer stripped
    assert resolved["C"] == "hello_world"        # nested ${A} resolved
    assert resolved["F"] == "hello_world/leaf"   # multi-pass nested resolution
    assert "D" not in resolved                   # command substitution rejected
    assert "E" not in resolved                   # pipe metachar rejected
