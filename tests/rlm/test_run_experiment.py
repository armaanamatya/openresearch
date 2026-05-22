import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import _MAX_LOG_CHARS, _cap_logs, run_experiment


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


def test_run_experiment_persists_result_to_disk(make_context, tmp_path, monkeypatch):
    # Every run_experiment call must leave an on-disk trace — its result
    # otherwise lives only in the root's REPL, so a failed experiment cannot
    # be diagnosed post-run. One JSONL line per call (repair retries included).
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id):
        return {"metrics": {}, "success": False, "logs": "boom: traceback here"}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)

    log = ctx.project_dir / "experiment_runs.jsonl"
    assert log.exists()
    entry = json.loads(log.read_text().strip())
    assert entry["success"] is False
    assert "boom" in entry["logs"]
    assert "timestamp" in entry


def test_cap_logs_bounds_unbounded_experiment_output():
    # run_experiment's container stdout is unbounded; verify_against_rubric and
    # propose_improvements feed the result into an LLM prompt, so the logs must
    # be capped before they leave run_experiment.
    small = "ok\n" * 10
    assert _cap_logs(small) == small  # under the cap: untouched
    huge = "x" * (_MAX_LOG_CHARS * 4)
    capped = _cap_logs(huge)
    assert len(capped) < _MAX_LOG_CHARS + 100  # head+tail window + marker
    assert "truncated" in capped
    assert capped.startswith("x") and capped.endswith("x")
