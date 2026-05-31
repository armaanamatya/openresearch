from argparse import Namespace

from backend.cli import (
    _blacklist_entries_from_arg,
    _with_reproduce_defaults,
)


def test_reproduce_defaults_accept_generated_namespace_without_cli_fields() -> None:
    args = _with_reproduce_defaults(
        Namespace(source="paper.pdf"),
    )

    assert args.source_kind == "auto"
    assert args.agent == "default"
    assert args.mode == "rlm"
    assert args.model is None
    assert args.provider is None
    assert args.verification_provider is None
    assert args.hints is None
    assert args.n_paths == 3
    assert args.execution_mode == "max"  # default flipped efficient→max in 738478a
    assert args.sandbox == "runpod"
    assert args.gpu_mode == "auto"
    assert args.command_timeout is None
    assert args.allow_sandbox_network is False
    assert args.sandbox_platform is None
    assert args.sandbox_memory is None
    assert args.sandbox_cpus is None
    assert args.seed is None
    assert args.attempt_id is None
    assert args.run_group_id is None
    assert args.blacklist is None


def test_blacklist_entries_from_inline_arg_and_file(tmp_path) -> None:
    blacklist = tmp_path / "blacklist.txt"
    blacklist.write_text(
        "# comment\nhttps://github.com/BartekCupial/finetuning-RL-as-CL\n",
        encoding="utf-8",
    )

    assert _blacklist_entries_from_arg(str(blacklist)) == (
        "https://github.com/BartekCupial/finetuning-RL-as-CL",
    )
    assert _blacklist_entries_from_arg("one, two") == ("one", "two")
