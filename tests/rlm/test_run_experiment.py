import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import run_experiment


def test_run_experiment_reads_commands_and_returns_metrics(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id):
        assert env_id == "reprolab/test:env-check"
        assert commands == ["python train.py"]
        assert project_id  # run_experiment threads ctx.project_id through
        return {"metrics": {"mean_reward": 200.0}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    result = run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)
    assert result["success"] is True
    assert result["metrics"]["mean_reward"] == 200.0


def test_run_experiment_missing_commands_json(make_context, tmp_path):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "nocode"
    code_dir.mkdir()
    result = run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)
    assert result["success"] is False
    assert "error" in result


def test_run_experiment_empty_commands_json(make_context, tmp_path):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "emptycode"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text("[]")
    result = run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)
    assert result["success"] is False
    assert "error" in result
