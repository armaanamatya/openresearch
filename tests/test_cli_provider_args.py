from argparse import Namespace

from backend.cli import _resolve_sdk_providers, _with_reproduce_defaults


def test_reproduce_defaults_accept_generated_namespace_without_cli_fields() -> None:
    args = _with_reproduce_defaults(
        Namespace(source="paper.pdf"),
    )

    assert args.source_kind == "auto"
    assert args.agent == "default"
    assert args.mode == "sdk"
    assert args.model is None
    assert args.provider is None
    assert args.verification_provider is None
    assert args.hints is None
    assert args.n_paths == 3
    assert args.execution_mode == "efficient"
    assert args.sandbox == "auto"
    assert args.gpu_mode == "auto"
    assert args.command_timeout is None
    assert args.allow_sandbox_network is False
    assert args.sandbox_platform is None
    assert args.sandbox_memory is None
    assert args.sandbox_cpus is None


def test_resolve_sdk_providers_accepts_generated_namespace_without_provider(
    monkeypatch,
) -> None:
    monkeypatch.setenv("REPROLAB_LLM_PROVIDER", "anthropic")

    provider, verification_provider = _resolve_sdk_providers(
        Namespace(mode="sdk"),
    )

    assert provider == "anthropic"
    assert verification_provider is None
