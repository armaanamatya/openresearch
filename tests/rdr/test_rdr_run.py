"""Tests for the rdr run entry (``backend/agents/rdr/run.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.rdr.run import _resolve_bundle_path, _effective_provider, _build_llm_client


def test_resolve_bundle_path_finds_vendored_bundle() -> None:
    """A bare paper_id resolves to the vendored third_party/paperbench bundle."""
    path = _resolve_bundle_path("sequential-neural-score-estimation")
    assert path.is_dir(), f"bundle dir not found: {path}"
    assert (path / "rubric.json").is_file(), "resolved bundle has no rubric.json"


def test_resolve_bundle_path_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_bundle_path("no-such-paper-bundle-xyz")


def test_resolve_bundle_path_custom_root(tmp_path: Path) -> None:
    """A custom bundles_root is used instead of the default vendored root."""
    # Create a fake bundle directory under the custom root
    custom_root = tmp_path / "my_bundles"
    fake_bundle = custom_root / "my-paper"
    fake_bundle.mkdir(parents=True)
    (fake_bundle / "rubric.json").write_text("{}", encoding="utf-8")

    resolved = _resolve_bundle_path("my-paper", bundles_root=custom_root)
    assert resolved == fake_bundle.resolve()


def test_resolve_bundle_path_custom_root_missing_raises(tmp_path: Path) -> None:
    """FileNotFoundError when paper_id does not exist under custom bundles_root."""
    custom_root = tmp_path / "my_bundles"
    custom_root.mkdir()

    with pytest.raises(FileNotFoundError, match="my-bundles-root"):
        _resolve_bundle_path("my-bundles-root", bundles_root=custom_root)


def test_effective_provider_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit provider=openai wins over presence of ANTHROPIC_API_KEY."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _effective_provider("openai") == "openai"


def test_effective_provider_explicit_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert _effective_provider("anthropic") == "anthropic"


def test_effective_provider_auto_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detect: OPENAI_API_KEY set → openai."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_PATH", raising=False)
    assert _effective_provider(None) == "openai"


def test_effective_provider_auto_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detect: ANTHROPIC_API_KEY set, no OPENAI_API_KEY → anthropic."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY_PATH", raising=False)
    assert _effective_provider(None) == "anthropic"


def test_effective_provider_none_when_no_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detect: no env vars → None."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_PATH", raising=False)
    assert _effective_provider(None) is None


def test_build_llm_client_openai_uses_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_llm_client with openai provider uses the supplied model at construction."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    client, model, label = _build_llm_client("openai", "gpt-5")
    assert model == "gpt-5"
    assert label == "openai"
    # The client's internal _model must be set to what we passed.
    assert client._model == "gpt-5"


def test_build_llm_client_openai_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_llm_client with openai provider defaults to gpt-4o-mini when model=None."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    client, model, label = _build_llm_client("openai", None)
    assert model == "gpt-4o-mini"
    assert client._model == "gpt-4o-mini"


def test_build_llm_client_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_llm_client with anthropic/None provider returns ClaudeLlmClient."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _client, model, label = _build_llm_client("anthropic", "claude-sonnet-4-6")
    assert label == "anthropic"
    assert model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Feature A: resume flag threading through run_pipeline_rdr
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_rdr_resume_flag_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_pipeline_rdr accepts resume=True without raising.

    Monkeypatches run_rdr and load_paperbench_bundle so no LLM or bundle I/O
    occurs. Verifies the resume flag is forwarded to run_rdr.
    """
    from backend.agents.rdr.models import RdrResult

    captured: dict = {}

    async def _fake_run_rdr(bundle, *, ctx, **kwargs):  # type: ignore[override]
        captured.update(kwargs)
        return RdrResult(
            project_id=ctx.project_id,
            status="partial",
            rubric_score=0.0,
            clusters_total=0,
            clusters_failed=0,
            repair_iterations=0,
            final_report_path="",
            cost_usd=None,
        )

    class _FakeBundle:
        paper_id = "test-paper"

        def rubric(self):
            return {"id": "root", "requirements": "r", "weight": 1.0, "sub_tasks": []}

        def read_paper_markdown(self):
            return "# Test"

        def metadata(self):
            return {"id": "test-paper", "title": "Test Paper"}

    monkeypatch.setattr("backend.agents.rdr.run.run_rdr", _fake_run_rdr)
    # load_paperbench_bundle is a local (lazy) import inside run_pipeline_rdr;
    # patch it on the bundle module where it lives.
    monkeypatch.setattr(
        "backend.evals.paperbench.bundle.load_paperbench_bundle",
        lambda path: _FakeBundle(),
    )
    # Also patch _resolve_bundle_path to avoid FileNotFoundError
    monkeypatch.setattr(
        "backend.agents.rdr.run._resolve_bundle_path",
        lambda paper_id, bundles_root=None: tmp_path,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from backend.agents.rdr.run import run_pipeline_rdr

    result = await run_pipeline_rdr(
        "test_project",
        tmp_path,
        paper_id="test-paper",
        resume=True,
    )

    assert result.status == "partial"
    assert captured.get("resume") is True, (
        f"resume=True was not forwarded to run_rdr; captured kwargs: {captured}"
    )


@pytest.mark.asyncio
async def test_run_pipeline_rdr_resume_defaults_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_pipeline_rdr defaults resume=False (backward compatibility)."""
    from backend.agents.rdr.models import RdrResult

    captured: dict = {}

    async def _fake_run_rdr(bundle, *, ctx, **kwargs):  # type: ignore[override]
        captured.update(kwargs)
        return RdrResult(
            project_id=ctx.project_id,
            status="partial",
            rubric_score=0.0,
            clusters_total=0,
            clusters_failed=0,
            repair_iterations=0,
            final_report_path="",
            cost_usd=None,
        )

    class _FakeBundle:
        paper_id = "test-paper"

        def rubric(self):
            return {"id": "root", "requirements": "r", "weight": 1.0, "sub_tasks": []}

        def read_paper_markdown(self):
            return "# Test"

        def metadata(self):
            return {"id": "test-paper", "title": "Test Paper"}

    monkeypatch.setattr("backend.agents.rdr.run.run_rdr", _fake_run_rdr)
    monkeypatch.setattr(
        "backend.evals.paperbench.bundle.load_paperbench_bundle",
        lambda path: _FakeBundle(),
    )
    monkeypatch.setattr(
        "backend.agents.rdr.run._resolve_bundle_path",
        lambda paper_id, bundles_root=None: tmp_path,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from backend.agents.rdr.run import run_pipeline_rdr

    result = await run_pipeline_rdr(
        "test_project_default",
        tmp_path,
        paper_id="test-paper",
        # resume not specified — should default to False
    )

    assert result.status == "partial"
    assert captured.get("resume") is False, (
        f"resume should default to False; captured kwargs: {captured}"
    )
