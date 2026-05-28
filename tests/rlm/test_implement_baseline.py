import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import implement_baseline


class _FakeBaselineResult:
    commands_to_run = ["python train.py", "python eval.py"]


def _write_minimal_code(project_id, runs_root):
    code_dir = runs_root / project_id / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")


def _assert_ok_envelope(result, tmp_path):
    assert result["ok"] is True
    assert result["code_path"] == str(tmp_path / "test_proj" / "code")
    assert "commands.json" in result["files"]
    assert "train.py" in result["files"]


def test_implement_baseline_writes_commands_manifest(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    ctx.runtime = object()

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        _write_minimal_code(project_id, runs_root)
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    result = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )
    manifest = json.loads((tmp_path / "test_proj" / "code" / "commands.json").read_text())
    assert manifest == ["python train.py", "python eval.py"]
    _assert_ok_envelope(result, tmp_path)


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
        _write_minimal_code(project_id, runs_root)
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )
    assert captured.get("model") == "claude-sonnet-4-6"


def test_implement_baseline_threads_repair_context(make_context, tmp_path, monkeypatch):
    # plan["repair_context"] must reach run_with_sdk as `repair_context` — it is
    # the signal that switches the code-writing agent into fix-existing-code mode
    # rather than a fresh implementation.
    ctx = make_context(tmp_path)
    ctx.runtime = object()
    captured: dict = {}

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        captured.update(kw)
        _write_minimal_code(project_id, runs_root)
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    implement_baseline(
        {
            "paper_claim_map": {},
            "environment_spec": {},
            "reproduction_contract": None,
            "repair_context": {"success": False, "logs": "boom"},
        },
        ctx=ctx,
    )
    assert captured.get("repair_context") == {"success": False, "logs": "boom"}


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
        _write_minimal_code(project_id, runs_root)
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)
    result = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )
    # ctx.project_id is "test_proj"; runs_root is tmp_path.
    assert (tmp_path / "test_proj" / "code" / "commands.json").exists()
    assert not (tmp_path / "elsewhere" / "code" / "commands.json").exists()
    _assert_ok_envelope(result, tmp_path)


def test_implement_baseline_pre_emit_stall_returns_error_envelope(make_context, tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    ctx.runtime = object()

    class _Future:
        def result(self, timeout=None):
            raise primitives.concurrent.futures.TimeoutError()

    class _Pool:
        def __init__(self, max_workers=1):
            pass
        def submit(self, *args, **kwargs):
            if len(args) > 1 and hasattr(args[1], "close"):
                args[1].close()
            return _Future()
        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(primitives.concurrent.futures, "ThreadPoolExecutor", _Pool)
    monkeypatch.setattr(primitives, "_pre_emit_stall_s", lambda: -1.0)

    result = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )

    assert result["ok"] is False
    assert result["error_code"] == "sdk_pre_emit_stall"
    assert result["repairable"] is True
    assert "commands.json" in result["missing_files"]


def test_implement_baseline_harvests_usable_artifacts_after_sdk_failure(
    make_context, tmp_path, monkeypatch
):
    ctx = make_context(tmp_path)
    ctx.runtime = object()

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        code_dir = runs_root / project_id / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        (code_dir / "commands.json").write_text(json.dumps(["python train.py"]), encoding="utf-8")
        raise RuntimeError("aclose(): asynchronous generator is already running")

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

    result = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )

    _assert_ok_envelope(result, tmp_path)


def test_implement_baseline_reports_missing_artifacts_after_sdk_failure(
    make_context, tmp_path, monkeypatch
):
    ctx = make_context(tmp_path)
    ctx.runtime = object()

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                artifact_index, **kw):
        code_dir = runs_root / project_id / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        raise RuntimeError("aclose(): asynchronous generator is already running")

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

    result = implement_baseline(
        {"paper_claim_map": {}, "environment_spec": {}, "reproduction_contract": None},
        ctx=ctx,
    )

    assert result["ok"] is False
    assert result["error_code"] == "sdk_failure_incomplete_artifacts"
    assert "commands.json" in result["missing_files"]
    assert "RuntimeError" in result["sdk_error"]


# ---------------------------------------------------------------------------
# Lane A — warm-retry cache (2026-05-24)
# ---------------------------------------------------------------------------


class TestImplementBaselineCache:
    """The cache short-circuits the ~5 min Sonnet sub-agent when the same plan
    + repair_context + arxiv_id + sandbox_mode + gpu_mode have already produced
    a result.  Hit-then-verify: if code/commands.json was archived between
    cache write and re-read, recompute from scratch."""

    def _plan(self):
        return {
            "paper_claim_map": {"core_contribution": "x"},
            "environment_spec": {},
            "reproduction_contract": None,
        }

    def test_cache_hit_skips_sub_agent(self, make_context, tmp_path, monkeypatch):
        """Second call with identical inputs must NOT invoke run_with_sdk."""
        ctx = make_context(tmp_path)
        ctx.runtime = object()

        call_count = {"n": 0}

        async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                    artifact_index, **kw):
            call_count["n"] += 1
            _write_minimal_code(project_id, runs_root)
            return _FakeBaselineResult()

        monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

        plan = self._plan()
        # First call — should invoke run_with_sdk and write commands.json.
        path1 = implement_baseline(plan, ctx=ctx)
        assert call_count["n"] == 1
        assert (tmp_path / "test_proj" / "code" / "commands.json").exists()

        # Second call — same inputs → cache hit → no new sub-agent call.
        path2 = implement_baseline(plan, ctx=ctx)
        assert call_count["n"] == 1, (
            "Second identical call must hit cache, not invoke run_with_sdk"
        )
        assert path1["code_path"] == path2["code_path"]
        assert path2["ok"] is True

    def test_cache_miss_when_code_dir_archived(self, make_context, tmp_path, monkeypatch):
        """Cache hit + missing code/commands.json → fall back to recompute.

        Simulates the race where attempt_isolation moved code/ AFTER the
        cache wrote.  The cache must NOT return a stale path.
        """
        ctx = make_context(tmp_path)
        ctx.runtime = object()

        call_count = {"n": 0}

        async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                    artifact_index, **kw):
            call_count["n"] += 1
            _write_minimal_code(project_id, runs_root)
            return _FakeBaselineResult()

        monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

        plan = self._plan()
        implement_baseline(plan, ctx=ctx)  # writes cache entry
        assert call_count["n"] == 1

        # Now simulate attempt_isolation archiving code/ after the cache wrote.
        import shutil
        code_dir = tmp_path / "test_proj" / "code"
        archived = tmp_path / "test_proj" / "attempts" / "20260524T000000"
        archived.mkdir(parents=True)
        shutil.move(str(code_dir), str(archived / "code"))
        assert not (tmp_path / "test_proj" / "code" / "commands.json").exists()

        # Second call — same inputs — cache hit BUT code/ is gone, so the
        # primitive must recompute and bump the call count.
        implement_baseline(plan, ctx=ctx)
        assert call_count["n"] == 2, (
            "Cache miss on commands.json verification must trigger recompute"
        )
        # And the recompute restored commands.json on disk.
        assert (tmp_path / "test_proj" / "code" / "commands.json").exists()

    def test_cache_key_excludes_remaining_s(self, make_context, tmp_path, monkeypatch):
        """Two calls with the same plan but different ctx.remaining_s() must
        still hit the same cache entry — remaining_s changes every call and
        would otherwise defeat the cache."""
        ctx = make_context(tmp_path)
        ctx.runtime = object()

        call_count = {"n": 0}

        async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                    artifact_index, **kw):
            call_count["n"] += 1
            _write_minimal_code(project_id, runs_root)
            return _FakeBaselineResult()

        monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

        plan = self._plan()

        # Patch ctx.remaining_s to return different values across the two calls.
        from datetime import datetime, timedelta, timezone

        ctx.deadline_utc = datetime.now(tz=timezone.utc) + timedelta(seconds=3600)
        implement_baseline(plan, ctx=ctx)
        ctx.deadline_utc = datetime.now(tz=timezone.utc) + timedelta(seconds=120)
        implement_baseline(plan, ctx=ctx)

        assert call_count["n"] == 1, (
            "remaining_s difference must NOT defeat cache — second call must hit"
        )

    def test_cache_key_includes_repair_context(self, make_context, tmp_path, monkeypatch):
        """Different repair_context → different cache key → cache miss."""
        ctx = make_context(tmp_path)
        ctx.runtime = object()

        call_count = {"n": 0}

        async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                    artifact_index, **kw):
            call_count["n"] += 1
            _write_minimal_code(project_id, runs_root)
            return _FakeBaselineResult()

        monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

        plan_fresh = self._plan()
        plan_repair = dict(plan_fresh)
        plan_repair["repair_context"] = {"success": False, "logs": "boom"}

        implement_baseline(plan_fresh, ctx=ctx)
        implement_baseline(plan_repair, ctx=ctx)

        assert call_count["n"] == 2, (
            "Different repair_context must be a cache MISS (different key)"
        )
