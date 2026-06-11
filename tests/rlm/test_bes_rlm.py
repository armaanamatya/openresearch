"""BES competing candidates on the RLM path (bes_rlm.py).

Covers the full contract: gating (should_compete), the compete loop
(snapshot/clear/grade/select/restore), idempotency via the winner marker,
fail-soft behaviours (all-fail, grade failure), guidance env restoration,
pool truncation on low wall-clock, and the experiment-arm stamp.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm import bes_rlm


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _settings(enabled: bool = True, n: int = 2, metric: str = "cluster_score"):
    return SimpleNamespace(
        bes_enabled=enabled,
        bes_candidates_per_cluster=n,
        bes_select_metric=metric,
    )


def _ctx(tmp_path: Path, *, remaining: float | None = 99999.0):
    project_dir = tmp_path / "prj_test"
    (project_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_id="prj_test",
        project_dir=project_dir,
        runs_root=tmp_path,
        llm_client=SimpleNamespace(complete=lambda **kw: "{}"),
        remaining_s=lambda: remaining,
    )


def _write_rubric(ctx) -> None:
    (Path(ctx.project_dir) / "generated_rubric.json").write_text(
        json.dumps({"source": "generated", "leaves": []}), encoding="utf-8"
    )


def _fake_implement(code_dir: Path, *, fail_on: set[int] = frozenset()):
    """An implement_fn that writes a valid candidate tree per call."""
    calls: list[dict] = []

    def fn(plan, *, ctx, _bes_inner=False):
        idx = plan.get("_bes_candidate_idx", -1)
        calls.append({"idx": idx, "guidance": os.environ.get("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE", "")})
        if idx in fail_on:
            return {"ok": False, "error_code": "boom", "error": f"candidate {idx} failed", "repairable": True}
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))
        (code_dir / "train.py").write_text(f"# candidate {idx}\n")
        return {"ok": True, "code_path": str(code_dir), "files": ["train.py", "commands.json"]}

    fn.calls = calls
    return fn


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    for var in ("OPENRESEARCH_AB_ARM", "OPENRESEARCH_AB_PAIR_ID", "OPENRESEARCH_BES_MIN_REMAINING_S",
                "OPENRESEARCH_BES_CONTINUE_MIN_S", "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# should_compete gating
# ---------------------------------------------------------------------------


def test_should_compete_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(enabled=False))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    assert bes_rlm.should_compete(ctx, {}) is False


def test_should_compete_requires_n_above_one(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(enabled=True, n=1))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    assert bes_rlm.should_compete(ctx, {}) is False


def test_should_compete_skips_repairs(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings())
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    assert bes_rlm.should_compete(ctx, {"repair_context": {"error": "x"}}) is False


def test_should_compete_requires_rubric_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings())
    ctx = _ctx(tmp_path)
    assert bes_rlm.should_compete(ctx, {}) is False


def test_should_compete_requires_wall_clock_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings())
    ctx = _ctx(tmp_path, remaining=100.0)
    _write_rubric(ctx)
    assert bes_rlm.should_compete(ctx, {}) is False


def test_should_compete_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings())
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    assert bes_rlm.should_compete(ctx, {}) is True


# ---------------------------------------------------------------------------
# compete — selection, restore, persistence
# ---------------------------------------------------------------------------


def test_compete_selects_higher_scored_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir)
    scores = {0: (0.3, ["leaf_a", "leaf_b"]), 1: (0.7, ["leaf_a"])}
    seen = []

    def fake_grade(rubric, cand_dir, _ctx):
        idx = int(str(cand_dir).rsplit("_", 1)[-1])
        seen.append(idx)
        return scores[idx]

    monkeypatch.setattr(bes_rlm, "_static_grade", fake_grade)

    result = bes_rlm.compete({"paper_claim_map": {}}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert result["bes"]["selected"] == "rlm_impl#1"
    assert result["bes"]["n_candidates"] == 2
    assert seen == [0, 1]
    # Winner's code restored into the real code/ tree.
    assert (code_dir / "train.py").read_text() == "# candidate 1\n"
    assert (code_dir / "commands.json").exists()
    # Pool + marker persisted for the report stamp and idempotency.
    state = json.loads((Path(ctx.project_dir) / "rlm_state" / "bes_candidates.json").read_text())
    assert state["winner"] == "rlm_impl#1"
    assert len(state["candidates"]) == 2
    assert (Path(ctx.project_dir) / "rlm_state" / "bes_winner.json").exists()
    # Snapshots kept for post-hoc inspection.
    assert (Path(ctx.project_dir) / "candidates" / "rlm_impl_0" / "code" / "train.py").exists()


def test_compete_is_idempotent_after_winner(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir)
    monkeypatch.setattr(bes_rlm, "_static_grade", lambda r, d, c: (0.5, []))

    first = bes_rlm.compete({}, ctx=ctx, implement_fn=impl)
    n_calls = len(impl.calls)
    second = bes_rlm.compete({}, ctx=ctx, implement_fn=impl)

    assert first["ok"] is True and second["ok"] is True
    assert len(impl.calls) == n_calls  # no re-implementation
    assert second["bes"]["selected"] == first["bes"]["selected"]


def test_compete_all_candidates_failed_returns_failure_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir, fail_on={0, 1})

    result = bes_rlm.compete({}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is False
    assert result.get("repairable") is True
    assert not (Path(ctx.project_dir) / "rlm_state" / "bes_winner.json").exists()


def test_compete_single_survivor_wins(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir, fail_on={0})
    monkeypatch.setattr(bes_rlm, "_static_grade", lambda r, d, c: (0.4, []))

    result = bes_rlm.compete({}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert result["bes"]["selected"] == "rlm_impl#1"
    assert (code_dir / "train.py").read_text() == "# candidate 1\n"


def test_compete_grade_failure_is_unscored_not_fatal(tmp_path, monkeypatch):
    """A grading crash leaves the candidate unscored; a scored peer wins."""
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir)

    def flaky_grade(rubric, cand_dir, _ctx):
        idx = int(str(cand_dir).rsplit("_", 1)[-1])
        if idx == 0:
            raise RuntimeError("grader down")
        return 0.6, []

    monkeypatch.setattr(bes_rlm, "_static_grade", flaky_grade)
    result = bes_rlm.compete({}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert result["bes"]["selected"] == "rlm_impl#1"


def test_compete_truncates_pool_when_wall_clock_runs_low(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=3))
    # First consultation happens before candidate #1 (i==0 never checks).
    ctx = _ctx(tmp_path)
    ctx.remaining_s = lambda: 100.0
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir)
    monkeypatch.setattr(bes_rlm, "_static_grade", lambda r, d, c: (0.5, []))

    result = bes_rlm.compete({}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert result["bes"]["n_candidates"] == 1  # truncated before candidate #1
    assert result["bes"]["selected"] == "rlm_impl#0"


def test_compete_falls_back_single_shot_on_internal_error(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    ctx = _ctx(tmp_path)
    # No rubric on disk → _compete_inner raises at rubric load; compete's
    # fallback must run ONE normal implementation.
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir)

    result = bes_rlm.compete({}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert "bes" not in result
    assert len(impl.calls) == 1


def test_angle_guidance_appends_and_restores(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE", "operator text")
    with bes_rlm._angle_guidance("ANGLE X"):
        merged = os.environ["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"]
        assert "operator text" in merged and "ANGLE X" in merged
    assert os.environ["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"] == "operator text"


def test_angle_guidance_unset_restores_to_absent(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE", raising=False)
    with bes_rlm._angle_guidance("ANGLE Y"):
        assert "ANGLE Y" in os.environ["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"]
    assert "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE" not in os.environ


def test_candidates_see_distinct_guidance(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir)
    monkeypatch.setattr(bes_rlm, "_static_grade", lambda r, d, c: (0.5, []))

    bes_rlm.compete({}, ctx=ctx, implement_fn=impl)

    assert impl.calls[0]["guidance"] == ""  # parity candidate
    assert "CANDIDATE ANGLE" in impl.calls[1]["guidance"]


# ---------------------------------------------------------------------------
# implement_baseline hook routing
# ---------------------------------------------------------------------------


def test_implement_baseline_routes_to_compete_when_enabled(tmp_path, monkeypatch):
    from backend.agents.rlm import primitives

    sentinel = {"ok": True, "code_path": str(tmp_path / "code"), "files": [], "bes": {"selected": "rlm_impl#0"}}
    monkeypatch.setattr(bes_rlm, "should_compete", lambda ctx, plan: True)
    monkeypatch.setattr(bes_rlm, "compete", lambda plan, *, ctx, implement_fn: sentinel)
    ctx = _ctx(tmp_path)

    result = primitives.implement_baseline({"paper_claim_map": {}}, ctx=ctx)

    assert result is sentinel


def test_implement_baseline_inner_calls_bypass_compete(tmp_path, monkeypatch):
    """_bes_inner re-entry must not consult the gate (no recursion)."""
    from backend.agents.rlm import primitives

    consulted = []
    monkeypatch.setattr(bes_rlm, "should_compete", lambda ctx, plan: consulted.append(1) or True)
    sentinel = {"ok": True, "code_path": "x", "files": []}
    monkeypatch.setattr(bes_rlm, "compete", lambda plan, *, ctx, implement_fn: sentinel)
    ctx = _ctx(tmp_path)

    # Outer call consults the gate and routes to compete.
    assert primitives.implement_baseline({"paper_claim_map": {}}, ctx=ctx) is sentinel
    assert consulted == [1]


# ---------------------------------------------------------------------------
# experiment_arm_stamp
# ---------------------------------------------------------------------------


def test_stamp_control_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(enabled=False, n=1))
    stamp = bes_rlm.experiment_arm_stamp(tmp_path)
    assert stamp["arm"] == "control"
    assert stamp["bes"]["enabled"] is False
    assert stamp["ab_pair_id"] is None


def test_stamp_bes_when_flags_on(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(enabled=True, n=2))
    stamp = bes_rlm.experiment_arm_stamp(tmp_path)
    assert stamp["arm"] == "bes"
    assert stamp["bes"]["candidates_per_cluster"] == 2


def test_stamp_env_overrides_and_pool_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(enabled=True, n=2))
    monkeypatch.setenv("OPENRESEARCH_AB_ARM", "bes")
    monkeypatch.setenv("OPENRESEARCH_AB_PAIR_ID", "allcnn-ab-1")
    state_dir = tmp_path / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "bes_candidates.json").write_text(json.dumps({
        "winner": "rlm_impl#1",
        "candidates": [
            {"candidate_id": "rlm_impl#0", "ok": True, "score": 0.3},
            {"candidate_id": "rlm_impl#1", "ok": True, "score": 0.7},
        ],
    }))
    stamp = bes_rlm.experiment_arm_stamp(tmp_path)
    assert stamp["ab_pair_id"] == "allcnn-ab-1"
    assert stamp["bes"]["winner"] == "rlm_impl#1"
    assert [c["score"] for c in stamp["bes"]["pool"]] == [0.3, 0.7]


# ---------------------------------------------------------------------------
# Rubric reuse (OPENRESEARCH_REUSE_RUBRIC — A/B arms grade against ONE rubric)
# ---------------------------------------------------------------------------


def test_reuse_rubric_off_by_default(tmp_path, monkeypatch):
    from backend.agents.rlm.run import _load_reusable_rubric
    monkeypatch.delenv("OPENRESEARCH_REUSE_RUBRIC", raising=False)
    (tmp_path / "generated_rubric.json").write_text('{"source": "generated"}')
    assert _load_reusable_rubric(tmp_path) is None


def test_reuse_rubric_returns_seeded_rubric(tmp_path, monkeypatch):
    from backend.agents.rlm.run import _load_reusable_rubric
    monkeypatch.setenv("OPENRESEARCH_REUSE_RUBRIC", "1")
    (tmp_path / "generated_rubric.json").write_text(
        json.dumps({"source": "generated", "leaves": [{"id": "x"}]})
    )
    rubric = _load_reusable_rubric(tmp_path)
    assert rubric is not None and rubric["leaves"][0]["id"] == "x"


def test_reuse_rubric_corrupt_file_falls_through(tmp_path, monkeypatch):
    from backend.agents.rlm.run import _load_reusable_rubric
    monkeypatch.setenv("OPENRESEARCH_REUSE_RUBRIC", "1")
    (tmp_path / "generated_rubric.json").write_text("{not json")
    assert _load_reusable_rubric(tmp_path) is None


def test_reuse_rubric_missing_file_falls_through(tmp_path, monkeypatch):
    from backend.agents.rlm.run import _load_reusable_rubric
    monkeypatch.setenv("OPENRESEARCH_REUSE_RUBRIC", "1")
    assert _load_reusable_rubric(tmp_path) is None


# ---------------------------------------------------------------------------
# Adaptive gating (OPENRESEARCH_BES_ADAPTIVE — pool only where selection pays)
# ---------------------------------------------------------------------------


def _adaptive_settings(skip_score: float = 0.5, adaptive: bool = True, n: int = 2):
    return SimpleNamespace(
        bes_enabled=True,
        bes_candidates_per_cluster=n,
        bes_select_metric="cluster_score",
        bes_adaptive=adaptive,
        bes_adaptive_skip_score=skip_score,
    )


def _write_attempt_report(project_dir: Path, score: float) -> None:
    att = project_dir / "attempts" / "20260610T000000-000000-aaaaaa"
    att.mkdir(parents=True, exist_ok=True)
    (att / "final_report.json").write_text(json.dumps({
        "paper": {"id": "x"},
        "verdict": "reproduced",
        "rubric": {"overall_score": score, "meets_target": True, "areas": []},
    }))


def test_adaptive_off_competes_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _adaptive_settings(adaptive=False))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    _write_attempt_report(Path(ctx.project_dir), 0.9)  # strong history, but adaptive off
    assert bes_rlm.should_compete(ctx, {}) is True


def test_adaptive_engages_on_first_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _adaptive_settings())
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    assert bes_rlm.should_compete(ctx, {}) is True
    decision = json.loads((Path(ctx.project_dir) / "rlm_state" / "bes_adaptive.json").read_text())
    assert decision["engage"] is True and decision["reason"] == "no_prior_history"


def test_adaptive_skips_on_strong_history(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _adaptive_settings(skip_score=0.5))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    _write_attempt_report(Path(ctx.project_dir), 0.74)
    assert bes_rlm.should_compete(ctx, {}) is False
    decision = json.loads((Path(ctx.project_dir) / "rlm_state" / "bes_adaptive.json").read_text())
    assert decision["engage"] is False
    assert decision["reason"].startswith("strong_history")
    assert decision["best_score"] == pytest.approx(0.74)


def test_adaptive_engages_on_weak_history(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _adaptive_settings(skip_score=0.5))
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    _write_attempt_report(Path(ctx.project_dir), 0.3)
    assert bes_rlm.should_compete(ctx, {}) is True
    decision = json.loads((Path(ctx.project_dir) / "rlm_state" / "bes_adaptive.json").read_text())
    assert decision["engage"] is True and decision["reason"].startswith("weak_history")


def test_adaptive_decision_lands_in_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _adaptive_settings())
    state = tmp_path / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "bes_adaptive.json").write_text(json.dumps(
        {"engage": False, "reason": "strong_history(0.740>=0.5)", "best_score": 0.74, "threshold": 0.5}
    ))
    stamp = bes_rlm.experiment_arm_stamp(tmp_path)
    assert stamp["bes"]["adaptive"]["engage"] is False
    assert stamp["bes"]["adaptive"]["best_score"] == pytest.approx(0.74)
