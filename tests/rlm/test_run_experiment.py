import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import _MAX_LOG_CHARS, _cap_logs, run_experiment


def test_run_experiment_reads_commands_and_returns_metrics(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id,
                        sandbox_mode=None, run_budget=None, gpu_plan=None, gpu_mode=None):
        assert env_id == "reprolab/test:env-check"
        assert commands == ["python train.py"]
        assert project_id  # run_experiment threads ctx.project_id through
        return {"metrics": {"mean_reward": 200.0}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    result = run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)
    assert result["success"] is True
    assert result["metrics"]["mean_reward"] == 200.0


def test_run_experiment_accepts_ok_code_envelope(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id,
                        sandbox_mode=None, run_budget=None, gpu_plan=None, gpu_mode=None):
        assert code_path == str(code_dir)
        return {"metrics": {"mean_reward": 10.0}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    result = run_experiment(
        {"ok": True, "code_path": str(code_dir), "files": ["commands.json"]},
        "reprolab/test:env-check",
        ctx=ctx,
    )
    assert result["success"] is True
    assert result["metrics"]["mean_reward"] == 10.0


def test_run_experiment_invalid_code_path_does_not_emit_experiment_completed_or_spawn(
    make_context, tmp_path, monkeypatch
):
    ctx = make_context(tmp_path)
    called = {"execute": False}

    async def fake_exec(*args, **kwargs):
        called["execute"] = True
        return {"metrics": {}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    result = run_experiment(
        {"ok": False, "error": "sdk_pre_emit_stall"},
        "reprolab/test:env-check",
        ctx=ctx,
    )

    assert result["success"] is False
    assert result["failure_class"] == "contract_guard"
    assert called["execute"] is False
    events_path = tmp_path / "test_proj" / "dashboard_events.jsonl"
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        assert not any(e.get("event") == "experiment_completed" for e in events)
    assert not (tmp_path / "test_proj" / "experiment_runs.jsonl").exists()


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

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id,
                        sandbox_mode=None, run_budget=None, gpu_plan=None, gpu_mode=None):
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


def test_run_experiment_returns_empty_metrics_when_file_missing(
        make_context, tmp_path, monkeypatch):
    """Symptom: a paper's code that fails to write metrics.json should not crash run_experiment.

    The post-loop reader must return metrics={} when $OUTPUT_DIR/metrics.json
    does not exist on the host — fail-soft, not exception (handoff P0-I1 / review C1).
    """
    import json
    from pathlib import Path

    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    from datetime import datetime, timezone

    from backend.services.runtime.interface import ExecResult, Sandbox

    def _make_result(cmd_str: str) -> ExecResult:
        now = datetime.now(timezone.utc)
        return ExecResult(
            command=cmd_str,
            exit_code=0,
            stdout="",
            stderr="",
            started_at=now,
            finished_at=now,
            duration_seconds=0.1,
        )

    class _FakeService:
        def __init__(self, backend):
            self._backend = backend
        async def create_sandbox(self, cmd):
            return Sandbox(sandbox_id="fake", name="fake", image="fake", config=cmd.config)
        async def execute(self, cmd):
            # Deliberately do NOT write metrics.json.
            return _make_result(cmd.command)
        async def destroy(self, cmd):
            return None

    import backend.agents.rlm.primitives as primitives_mod
    monkeypatch.setattr(primitives_mod, "RuntimeAppService", _FakeService)

    result = primitives_mod.run_experiment(
        str(code_dir), "reprolab/test:env-check", ctx=ctx)

    assert result["success"] is True
    assert result["metrics"] == {}  # fail-soft, not a crash


def test_run_experiment_returns_empty_metrics_when_file_malformed(
        make_context, tmp_path, monkeypatch):
    """Symptom: a corrupted metrics.json should degrade fail-soft, not crash.

    The post-loop reader must return metrics={} when $OUTPUT_DIR/metrics.json
    contains invalid JSON on the host (handoff P0-I1 / review C1).
    """
    import json
    from pathlib import Path

    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    captured = {}

    from datetime import datetime, timezone

    from backend.services.runtime.interface import ExecResult, Sandbox

    def _make_result(cmd_str: str) -> ExecResult:
        now = datetime.now(timezone.utc)
        return ExecResult(
            command=cmd_str,
            exit_code=0,
            stdout="",
            stderr="",
            started_at=now,
            finished_at=now,
            duration_seconds=0.1,
        )

    class _FakeService:
        def __init__(self, backend):
            self._backend = backend
        async def create_sandbox(self, cmd):
            captured["artifact_root"] = cmd.config.artifact_root
            return Sandbox(sandbox_id="fake", name="fake", image="fake", config=cmd.config)
        async def execute(self, cmd):
            # Write invalid JSON to the contract path.
            (Path(captured["artifact_root"]) / "metrics.json").write_text(
                "not valid json {", encoding="utf-8")
            return _make_result(cmd.command)
        async def destroy(self, cmd):
            return None

    import backend.agents.rlm.primitives as primitives_mod
    monkeypatch.setattr(primitives_mod, "RuntimeAppService", _FakeService)

    result = primitives_mod.run_experiment(
        str(code_dir), "reprolab/test:env-check", ctx=ctx)

    assert result["success"] is True
    assert result["metrics"] == {}  # fail-soft on JSONDecodeError


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


def test_sandbox_mode_is_threaded_from_ctx_to_backend(make_context, tmp_path, monkeypatch):
    """Symptom: --sandbox runpod silently uses LocalDockerBackend in RLM mode.

    _execute_in_sandbox hardcoded LocalDockerBackend; ctx.sandbox_mode was
    accepted but never consulted (handoff P1-I7 / T12). Verify: the sandbox_mode
    stored in ctx is the value forwarded to _backend_for_sandbox_mode.
    """
    import backend.agents.rlm.primitives as primitives_mod

    captured: dict = {}

    real_backend_for = primitives_mod._backend_for_sandbox_mode

    def spy_backend_for(mode, *, run_budget=None, gpu_plan=None):
        captured["mode"] = mode
        # Return a minimal fake backend so we don't need Docker.
        class _FakeBackend:
            pass
        return _FakeBackend()

    monkeypatch.setattr(primitives_mod, "_backend_for_sandbox_mode", spy_backend_for)

    # Also stub RuntimeAppService so no real container work happens.
    from datetime import datetime, timezone
    from backend.services.runtime.interface import ExecResult, Sandbox

    def _make_result(cmd_str: str) -> ExecResult:
        now = datetime.now(timezone.utc)
        return ExecResult(
            command=cmd_str, exit_code=0, stdout="", stderr="",
            started_at=now, finished_at=now, duration_seconds=0.01,
        )

    class _FakeService:
        def __init__(self, backend):
            pass
        async def create_sandbox(self, cmd):
            return Sandbox(sandbox_id="fake", name="fake", image="fake", config=cmd.config)
        async def execute(self, cmd):
            return _make_result(cmd.command)
        async def destroy(self, cmd):
            return None

    monkeypatch.setattr(primitives_mod, "RuntimeAppService", _FakeService)

    ctx = make_context(tmp_path)
    ctx.sandbox_mode = "runpod"  # non-default value — must survive the thread
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["echo hi"]))

    primitives_mod.run_experiment(str(code_dir), "img:tag", ctx=ctx)

    assert captured.get("mode") == "runpod", (
        f"ctx.sandbox_mode was not forwarded to _backend_for_sandbox_mode; "
        f"got mode={captured.get('mode')!r}"
    )


def test_run_budget_is_threaded_from_ctx_to_backend(make_context, tmp_path, monkeypatch):
    """Regression guard: ctx.run_budget must reach _backend_for_sandbox_mode.

    Mirrors test_sandbox_mode_is_threaded_from_ctx_to_backend but captures the
    run_budget kwarg. Without this, accidentally dropping run_budget=ctx.run_budget
    from the _execute_in_sandbox(...) call inside run_experiment would silently
    disarm the max_pod_seconds protection in production with no test failure.
    """
    import backend.agents.rlm.primitives as primitives_mod
    from backend.agents.resilience.budget import RunBudget

    captured: dict = {}

    def spy_backend_for(mode, *, run_budget=None, gpu_plan=None):
        captured["run_budget"] = run_budget

        class _FakeBackend:
            pass

        return _FakeBackend()

    monkeypatch.setattr(primitives_mod, "_backend_for_sandbox_mode", spy_backend_for)

    from datetime import datetime, timezone
    from backend.services.runtime.interface import ExecResult, Sandbox

    def _make_result(cmd_str: str) -> ExecResult:
        now = datetime.now(timezone.utc)
        return ExecResult(
            command=cmd_str, exit_code=0, stdout="", stderr="",
            started_at=now, finished_at=now, duration_seconds=0.01,
        )

    class _FakeService:
        def __init__(self, backend):
            pass
        async def create_sandbox(self, cmd):
            return Sandbox(sandbox_id="fake", name="fake", image="fake", config=cmd.config)
        async def execute(self, cmd):
            return _make_result(cmd.command)
        async def destroy(self, cmd):
            return None

    monkeypatch.setattr(primitives_mod, "RuntimeAppService", _FakeService)

    ctx = make_context(tmp_path)
    budget = RunBudget(max_pod_seconds=600.0)
    ctx.run_budget = budget
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["echo hi"]))

    primitives_mod.run_experiment(str(code_dir), "img:tag", ctx=ctx)

    assert captured.get("run_budget") is budget, (
        f"ctx.run_budget did not reach _backend_for_sandbox_mode; "
        f"got run_budget={captured.get('run_budget')!r}"
    )
