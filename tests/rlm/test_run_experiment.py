import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import _MAX_LOG_CHARS, _cap_logs, run_experiment


def test_run_experiment_reads_commands_and_returns_metrics(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id, sandbox_mode=None):
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

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id, sandbox_mode=None):
        return {"metrics": {}, "success": False, "logs": "boom: traceback here"}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)

    log = ctx.project_dir / "experiment_runs.jsonl"
    assert log.exists()
    entry = json.loads(log.read_text().strip())
    assert entry["success"] is False
    assert "boom" in entry["logs"]
    assert "timestamp" in entry


def test_run_experiment_returns_real_metrics_from_artifact_dir(
        make_context, tmp_path, monkeypatch):
    """Symptom: every score caps at 0.35 because run_experiment hard-returns metrics={}.

    Without this fix _execute_in_sandbox returns {} regardless of what the
    paper's code wrote to $OUTPUT_DIR/metrics.json (handoff P0-I1 / review C1).
    Verify: when the sandbox writes metrics.json into artifact_root, run_experiment
    returns those metrics (not {}), and the degraded backstop would not fire.
    """
    import json
    from pathlib import Path

    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    # Monkeypatch the runtime service so we don't need Docker.  Instead, we
    # intercept create_sandbox / execute / destroy and have execute() write
    # metrics.json to the artifact_root the production code computed.
    captured = {}

    from datetime import datetime, timezone

    from backend.services.runtime.interface import ExecResult, Sandbox

    def _make_result(cmd_str: str) -> ExecResult:
        now = datetime.now(timezone.utc)
        return ExecResult(
            command=cmd_str,
            exit_code=0,
            stdout="trained successfully\n",
            stderr="",
            started_at=now,
            finished_at=now,
            duration_seconds=0.1,
        )

    class _FakeService:
        def __init__(self, backend):  # signature matches RuntimeAppService(backend)
            self._backend = backend
        async def create_sandbox(self, cmd):
            captured["artifact_root"] = cmd.config.artifact_root
            captured["env"] = cmd.config.environment
            return Sandbox(
                sandbox_id="fake",
                name="fake",
                image="fake",
                config=cmd.config,
            )
        async def execute(self, cmd):
            # Simulate the paper's code writing metrics to the contract path.
            (Path(captured["artifact_root"]) / "metrics.json").write_text(
                json.dumps({"mean_reward": 487.3}), encoding="utf-8")
            return _make_result(cmd.command)
        async def destroy(self, cmd):
            return None

    import backend.agents.rlm.primitives as primitives_mod
    monkeypatch.setattr(primitives_mod, "RuntimeAppService", _FakeService)

    result = primitives_mod.run_experiment(
        str(code_dir), "reprolab/test:env-check", ctx=ctx)

    # The metrics the paper's code wrote must reach the primitive's return.
    assert result["success"] is True
    assert result["metrics"] == {"mean_reward": 487.3}
    # The container OUTPUT_DIR contract was honored.
    assert captured["env"]["OUTPUT_DIR"] == "/artifacts"
    # And the degraded condition would NOT fire:
    assert not ((not result["success"]) or (not result["metrics"]))


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
