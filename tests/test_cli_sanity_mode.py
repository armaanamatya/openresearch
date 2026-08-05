from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace


def test_cmd_reproduce_sanity_dispatches_before_ingest(tmp_path, monkeypatch):
    import backend.cli as cli

    called = {}

    def fake_sanity(args, runs_root):
        called["source"] = args.source
        called["runs_root"] = runs_root
        return 0

    def fail_ingest(*args, **kwargs):
        raise AssertionError("sanity mode must not invoke ingestion")

    monkeypatch.setattr(cli, "_cmd_reproduce_sanity", fake_sanity)
    monkeypatch.setattr(cli, "build_intake_service", fail_ingest, raising=False)

    result = cli.cmd_reproduce(Namespace(
        source="2512.24601",
        sanity=True,
        runs_root=str(tmp_path),
    ))

    assert result == 0
    assert called["source"] == "2512.24601"
    assert called["runs_root"] == tmp_path


def test_cmd_reproduce_sanity_writes_stable_artifacts(tmp_path, monkeypatch):
    import backend.cli as cli
    import backend.agents.execution as execution
    import backend.agents.rlm.primitives as primitives

    monkeypatch.setattr(execution, "resolve_sandbox_mode", lambda sandbox, pipeline_mode: SimpleNamespace(value="docker"))
    monkeypatch.setattr(execution, "ensure_sandbox_mode_available", lambda mode: None)

    def fake_run_experiment(code_path, env_id, *, model_id, eval_env, ctx):
        return {"success": True, "metrics": {"sanity_ok": 1.0}, "logs": "", "outcome": "ok"}

    monkeypatch.setattr(primitives, "run_experiment", fake_run_experiment)

    result = cli._cmd_reproduce_sanity(Namespace(
        source="2512.24601",
        project_id="prj_sanity",
        sandbox="docker",
        max_usd=None,
        max_wall_clock=1800,
        max_pod_seconds=900,
        max_run_gpu_usd=1.0,
    ), tmp_path)

    run_dir = tmp_path / "prj_sanity"
    assert result == 0
    assert (run_dir / "code" / "sanity.py").exists()
    assert (run_dir / "code" / "commands.json").exists()
    assert (run_dir / "cost_ledger.jsonl").exists()
    assert (run_dir / "demo_status.json").exists()
    assert "REQUIRE_GPU = False" in (run_dir / "code" / "sanity.py").read_text()


def test_cmd_reproduce_sanity_requires_and_threads_cloud_gpu(tmp_path, monkeypatch):
    import backend.cli as cli
    import backend.agents.execution as execution
    import backend.agents.rlm.primitives as primitives

    monkeypatch.setattr(
        execution,
        "resolve_sandbox_mode",
        lambda sandbox, pipeline_mode: SimpleNamespace(value="gcp"),
    )
    monkeypatch.setattr(execution, "ensure_sandbox_mode_available", lambda mode: None)
    observed = {}

    def fake_run_experiment(code_path, env_id, *, model_id, eval_env, ctx):
        observed["script"] = (tmp_path / "prj_cloud_sanity" / "code" / "sanity.py").read_text()
        observed["gpu_mode"] = ctx.gpu_mode
        return {"success": True, "metrics": {"sanity_ok": 1.0, "gpu_visible": 1.0}}

    monkeypatch.setattr(primitives, "run_experiment", fake_run_experiment)

    result = cli._cmd_reproduce_sanity(Namespace(
        source="2512.24601",
        project_id="prj_cloud_sanity",
        sandbox="gcp",
        gpu_mode="prefer",
        max_usd=None,
        max_wall_clock=1800,
        max_pod_seconds=900,
        max_run_gpu_usd=1.0,
    ), tmp_path)

    assert result == 0
    assert "REQUIRE_GPU = True" in observed["script"]
    assert observed["gpu_mode"] == "prefer"
