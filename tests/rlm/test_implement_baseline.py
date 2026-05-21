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
