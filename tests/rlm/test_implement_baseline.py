import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import implement_baseline


class _FakeBaselineResult:
    commands_to_run = ["python train.py", "python eval.py"]


def test_implement_baseline_writes_commands_manifest(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    ctx.runtime = object()

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    code_path = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )
    manifest = json.loads((tmp_path / "test_proj" / "code" / "commands.json").read_text())
    assert manifest == ["python train.py", "python eval.py"]
    assert code_path.endswith("code")


def test_implement_baseline_passes_agent_model_as_override(make_context, tmp_path, monkeypatch):
    # ctx.agent_model must reach run_with_sdk as `model` — it becomes the
    # model_override that beats the agent registry's Opus default for the
    # baseline-implementation agent. RLM run 3 burned the OAuth quota for
    # lack of this override.
    ctx = make_context(tmp_path)
    ctx.runtime = object()
    ctx.agent_model = "claude-sonnet-4-6"
    captured: dict = {}

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        captured.update(kw)
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )
    assert captured.get("model") == "claude-sonnet-4-6"


def test_implement_baseline_writes_manifest_beside_the_code(make_context, tmp_path, monkeypatch):
    # run_with_sdk writes the code to runs_root/project_id/code. commands.json
    # must land THERE — not at ctx.project_dir/code — even when project_dir
    # diverges from runs_root/project_id (RunContext does not enforce that
    # invariant). This pins the fix: reverting it to ctx.project_dir/code
    # makes this test fail.
    ctx = make_context(tmp_path)
    ctx.runtime = object()
    ctx.project_dir = tmp_path / "elsewhere"  # diverge from runs_root/project_id

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    code_path = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )
    # ctx.project_id is "test_proj"; runs_root is tmp_path.
    assert (tmp_path / "test_proj" / "code" / "commands.json").exists()
    assert not (tmp_path / "elsewhere" / "code" / "commands.json").exists()
    assert code_path == str(tmp_path / "test_proj" / "code")
