"""Task 10: lifecycle driver edge-case hardening.

Covers three additions to backend/agents/rlm/lifecycle_driver.py:
  (a) a bounded re-drive when implement_baseline itself returns a repairable
      result (distinct from the existing run_experiment repair loop);
  (b) run_lifecycle_primary persists plan_reproduction's ordered steps to
      rlm_state/reproduction_plan.json (fail-soft) and reads them back;
  (c) an honest stopped_reason="repair_exhausted" when the evidence
      fingerprint stops changing across repairs.

Default-safe: existing behaviour is unchanged when implement_baseline never
returns a repairable result and/or ctx has no project_dir.

NOTE on the `tools` dict shape: `_get_tool` (and every other test in
tests/rlm/test_lifecycle_driver.py, and the real `binding.build_custom_tools`)
expects `{name: {"tool": callable, ...}}` — a bare callable is NOT
supported. Tools below are wrapped accordingly.
"""

from __future__ import annotations

import json

from backend.agents.rlm import lifecycle_driver as ld


def _tool(seq):
    calls = {"n": 0}

    def fn(*a, **k):
        r = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return r

    return fn, calls


def _wrap(fn):
    return {"tool": fn}


# ---------------------------------------------------------------------------
# (a) repairable implement_baseline triggers a bounded re-drive
# ---------------------------------------------------------------------------


def test_repairable_implement_baseline_triggers_redrive(tmp_path):
    # implement_baseline first returns repairable, then ok; driver should retry it.
    impl, impl_calls = _tool([{"ok": False, "outcome": "repairable"},
                              {"ok": True, "code_path": str(tmp_path)}])
    tools = {
        "understand_section": _wrap(_tool([{"ok": True}])[0]),
        "detect_environment": _wrap(_tool([{"ok": True}])[0]),
        "plan_reproduction": _wrap(_tool([{"ok": True, "steps": ["s1"]}])[0]),
        "implement_baseline": _wrap(impl),
        "run_experiment": _wrap(_tool([{"ok": True, "metrics": {"val/success_rate": 0.46}}])[0]),
        "verify_against_rubric": _wrap(_tool([{"overall_score": 0.46}])[0]),
    }
    summary = ld.drive_lifecycle_chain(
        tools=tools, ctx=None, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=2)
    assert impl_calls["n"] >= 2  # implement was re-driven on the repairable result
    assert summary["rubric_score"] == 0.46


def test_repairable_implement_baseline_repair_context_matches_failure(tmp_path):
    """The re-drive's repair_context must equal the failed implement result."""
    repairable = {"ok": False, "outcome": "repairable", "failure_class": "preflight_blocked"}
    ok_result = {"ok": True, "code_path": str(tmp_path)}
    impl_plans: list = []

    def _impl(plan):
        impl_plans.append(dict(plan))
        return repairable if len(impl_plans) == 1 else ok_result

    tools = {
        "understand_section": _wrap(_tool([{"ok": True}])[0]),
        "detect_environment": _wrap(_tool([{"ok": True}])[0]),
        "plan_reproduction": _wrap(_tool([{"ok": True, "steps": ["s1"]}])[0]),
        "implement_baseline": _wrap(_impl),
        "run_experiment": _wrap(_tool([{"ok": True, "metrics": {"acc": 0.5}}])[0]),
        "verify_against_rubric": _wrap(_tool([{"overall_score": 0.5}])[0]),
    }

    ld.drive_lifecycle_chain(
        tools=tools, ctx=None, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=2)

    assert len(impl_plans) == 2
    assert impl_plans[1].get("repair_context") == repairable


def test_repairable_implement_baseline_exhausts_budget_stops_honestly(tmp_path):
    """implement_baseline stays repairable forever -> stops with repair_exhausted,
    run_experiment is never reached."""
    always_repairable = {"ok": False, "outcome": "repairable", "failure_class": "x"}
    run_fn, run_calls = _tool([{"ok": True, "metrics": {}}])
    tools = {
        "understand_section": _wrap(_tool([{"ok": True}])[0]),
        "detect_environment": _wrap(_tool([{"ok": True}])[0]),
        "plan_reproduction": _wrap(_tool([{"ok": True, "steps": ["s1"]}])[0]),
        "implement_baseline": _wrap(_tool([always_repairable])[0]),
        "run_experiment": _wrap(run_fn),
        "verify_against_rubric": _wrap(_tool([{"overall_score": 0.0}])[0]),
    }

    summary = ld.drive_lifecycle_chain(
        tools=tools, ctx=None, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=2)

    assert summary["stopped_at"] == "implement_baseline"
    assert summary["stopped_reason"] == "repair_exhausted"
    assert run_calls["n"] == 0  # run_experiment must never be reached


def test_non_repairable_implement_failure_unaffected(tmp_path):
    """A plain ok=False (no outcome=='repairable') behaves exactly as before:
    the chain stops immediately at implement_baseline, no re-drive attempted."""
    impl, impl_calls = _tool([{"ok": False, "error": "boom"}])
    run_fn, run_calls = _tool([{"ok": True, "metrics": {}}])
    tools = {
        "understand_section": _wrap(_tool([{"ok": True}])[0]),
        "detect_environment": _wrap(_tool([{"ok": True}])[0]),
        "plan_reproduction": _wrap(_tool([{"ok": True, "steps": ["s1"]}])[0]),
        "implement_baseline": _wrap(impl),
        "run_experiment": _wrap(run_fn),
        "verify_against_rubric": _wrap(_tool([{"overall_score": 0.0}])[0]),
    }

    summary = ld.drive_lifecycle_chain(
        tools=tools, ctx=None, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=2)

    assert impl_calls["n"] == 1  # no re-drive for a non-repairable failure
    assert summary["stopped_at"] == "implement_baseline"
    assert "boom" in summary["stopped_reason"]
    assert run_calls["n"] == 0


# ---------------------------------------------------------------------------
# (b) run_lifecycle_primary persists the ordered plan_reproduction steps
# ---------------------------------------------------------------------------


def _make_full_tools(*, plan_steps=("understand", "implement", "run"),
                      verify_ret=None):
    verify_ret = verify_ret or {"overall_score": 0.9, "target_score": 0.7, "meets_target": True}
    return {
        "understand_section": _wrap(_tool([{"sections": ["intro"]}])[0]),
        "detect_environment": _wrap(_tool([{"environment": "conda"}])[0]),
        "plan_reproduction": _wrap(_tool([{"steps": list(plan_steps), "method_spec": "X"}])[0]),
        "implement_baseline": _wrap(_tool([{"ok": True, "code_path": "/fake/code"}])[0]),
        "run_experiment": _wrap(_tool([{"success": True, "metrics": {"acc": 0.9}}])[0]),
        "verify_against_rubric": _wrap(_tool([verify_ret])[0]),
        "propose_improvements": _wrap(_tool([[]])[0]),
    }


def _make_ctx(tmp_path):
    from types import SimpleNamespace
    project_dir = tmp_path / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(project_dir=project_dir, remaining_s=lambda: None)


def test_run_lifecycle_primary_persists_reproduction_plan(tmp_path):
    tools = _make_full_tools(plan_steps=("understand", "implement", "run"))
    ctx = _make_ctx(tmp_path)

    summary = ld.run_lifecycle_primary(
        tools=tools, ctx=ctx, paper_text="p", rubric_spec={"leaves": []},
        emit=lambda e: None, max_improve_iterations=0,
    )

    plan_path = ctx.project_dir / "rlm_state" / "reproduction_plan.json"
    assert plan_path.is_file()
    persisted = json.loads(plan_path.read_text(encoding="utf-8"))
    assert persisted["steps"] == ["understand", "implement", "run"]
    # Read back into the summary for dispatch.
    assert summary["reproduction_plan"] == ["understand", "implement", "run"]


def test_run_lifecycle_primary_no_project_dir_is_fail_soft(tmp_path):
    """ctx without a usable project_dir must not raise; reproduction_plan stays None."""
    from types import SimpleNamespace

    tools = _make_full_tools()
    ctx = SimpleNamespace(project_dir=None, remaining_s=lambda: None)

    summary = ld.run_lifecycle_primary(
        tools=tools, ctx=ctx, paper_text="p", rubric_spec={"leaves": []},
        emit=lambda e: None, max_improve_iterations=0,
    )

    assert summary["reproduction_plan"] is None
    assert summary["rubric_score"] == 0.9


def test_run_lifecycle_primary_no_plan_steps_is_noop(tmp_path):
    """plan_reproduction result with no 'steps' key -> no file written, key stays None."""
    tools = _make_full_tools()
    tools["plan_reproduction"] = _wrap(_tool([{"method_spec": "X"}])[0])  # no "steps"
    ctx = _make_ctx(tmp_path)

    summary = ld.run_lifecycle_primary(
        tools=tools, ctx=ctx, paper_text="p", rubric_spec={"leaves": []},
        emit=lambda e: None, max_improve_iterations=0,
    )

    plan_path = ctx.project_dir / "rlm_state" / "reproduction_plan.json"
    assert not plan_path.exists()
    assert summary["reproduction_plan"] is None


# ---------------------------------------------------------------------------
# (c) stopped_reason == "repair_exhausted" on evidence-fingerprint stagnation
# ---------------------------------------------------------------------------


def test_repair_exhausted_when_run_experiment_evidence_stagnates(tmp_path):
    """run_experiment stays repairable with IDENTICAL metrics across repeated
    repairs -> stopped_reason == repair_exhausted, an honest early stop rather
    than silently grinding out the full iteration cap."""
    stuck = {"outcome": "repairable", "failure_class": "preflight_blocked",
              "metrics": {"acc": 0.1}}
    tools = {
        "understand_section": _wrap(_tool([{"ok": True}])[0]),
        "detect_environment": _wrap(_tool([{"ok": True}])[0]),
        "plan_reproduction": _wrap(_tool([{"ok": True, "steps": ["s1"]}])[0]),
        "implement_baseline": _wrap(_tool([{"ok": True, "code_path": str(tmp_path)}])[0]),
        "run_experiment": _wrap(_tool([stuck])[0]),  # always returns the identical dict
        "verify_against_rubric": _wrap(_tool([{"overall_score": 0.1}])[0]),
    }

    summary = ld.drive_lifecycle_chain(
        tools=tools, ctx=None, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=5)

    assert summary["stopped_reason"] == "repair_exhausted"
    # Stopped BEFORE burning the full 5-iteration budget (stagnation detected at repair 2).
    assert summary["repaired"] < 5
    # Verify still ran against whatever partial evidence exists.
    assert summary["rubric_score"] == 0.1


def test_repair_continues_when_evidence_changes_each_time(tmp_path):
    """Evidence genuinely changes each repair -> no premature repair_exhausted stop;
    the loop runs to the natural cap (existing behaviour, unaffected)."""
    results = [
        {"outcome": "repairable", "metrics": {"acc": 0.1}},
        {"outcome": "repairable", "metrics": {"acc": 0.2}},
        {"outcome": "repairable", "metrics": {"acc": 0.3}},
    ]
    tools = {
        "understand_section": _wrap(_tool([{"ok": True}])[0]),
        "detect_environment": _wrap(_tool([{"ok": True}])[0]),
        "plan_reproduction": _wrap(_tool([{"ok": True, "steps": ["s1"]}])[0]),
        "implement_baseline": _wrap(_tool([{"ok": True, "code_path": str(tmp_path)}])[0]),
        "run_experiment": _wrap(_tool(results)[0]),
        "verify_against_rubric": _wrap(_tool([{"overall_score": 0.3}])[0]),
    }

    summary = ld.drive_lifecycle_chain(
        tools=tools, ctx=None, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=2)

    # Cap reached normally (2 repairs), not the fingerprint-stagnation path.
    assert summary["repaired"] == 2
    assert summary["stopped_reason"] is None


# ---------------------------------------------------------------------------
# (d) reviewer fix I3: _evidence_fingerprint must track the CURRENT result,
#     never a persisted (and possibly stale) evidence_bundle.json.
# ---------------------------------------------------------------------------


def test_evidence_fingerprint_reflects_current_metrics_change():
    """Two different results with different metrics must fingerprint
    differently, and identical metrics must fingerprint identically -- this
    is the core contract the repair-stagnation check relies on."""
    ctx = None
    fp_low = ld._evidence_fingerprint({"metrics": {"acc": 0.10}}, ctx)
    fp_high = ld._evidence_fingerprint({"metrics": {"acc": 0.90}}, ctx)
    assert fp_low != fp_high

    fp_low_again = ld._evidence_fingerprint({"metrics": {"acc": 0.10}}, ctx)
    assert fp_low == fp_low_again


def test_evidence_fingerprint_ignores_stale_persisted_bundle(tmp_path):
    """Reviewer regression (I3): a COHERENT (i.e. genuinely resolvable) but
    STALE evidence_bundle.json left on disk by an earlier finalize in a
    resumed project_dir must not mask a real metrics change between two
    results. Before the fix, _evidence_fingerprint preferred
    evidence_bundle.resolve_bundle(ctx.project_dir) whenever it resolved,
    so both results below fingerprinted IDENTICALLY (the stagnation detector
    then falsely fired after 2 repairs even with genuine improvement)."""
    from backend.agents.rlm import evidence_bundle

    project_dir = tmp_path / "proj"
    artifact_dir = project_dir / "outputs" / "attempt_1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"acc": 0.5}), encoding="utf-8")

    import hashlib

    on_disk_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    stale_bundle = {
        "schema": 1,
        "attempt_id": "stale-attempt",
        "ledger_sequence": 0,
        "metrics_sha256": on_disk_sha,
        "code_tree_digest": None,
        "artifact_dir": str(artifact_dir),
        "coordinates": {},
        "coherent": True,
    }
    assert evidence_bundle.persist_bundle(project_dir, stale_bundle)
    # Sanity check: the bundle really is resolvable, so this test exercises
    # the actual bug rather than a bundle that silently no-ops.
    assert evidence_bundle.resolve_bundle(project_dir) is not None

    from types import SimpleNamespace

    ctx = SimpleNamespace(project_dir=project_dir, remaining_s=lambda: None)

    fp_low = ld._evidence_fingerprint({"metrics": {"acc": 0.10}}, ctx)
    fp_high = ld._evidence_fingerprint({"metrics": {"acc": 0.90}}, ctx)
    assert fp_low != fp_high

    fp_low_again = ld._evidence_fingerprint({"metrics": {"acc": 0.10}}, ctx)
    assert fp_low == fp_low_again


def test_repair_exhausted_stagnation_not_masked_by_stale_bundle(tmp_path):
    """End-to-end regression: even with a coherent-but-stale bundle on disk,
    repairs that genuinely improve metrics must NOT be stopped early as
    'repair_exhausted' -- the driver should run to its natural cap."""
    from backend.agents.rlm import evidence_bundle

    project_dir = tmp_path / "proj"
    artifact_dir = project_dir / "outputs" / "attempt_1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"acc": 0.5}), encoding="utf-8")

    import hashlib

    on_disk_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    stale_bundle = {
        "schema": 1,
        "attempt_id": "stale-attempt",
        "ledger_sequence": 0,
        "metrics_sha256": on_disk_sha,
        "code_tree_digest": None,
        "artifact_dir": str(artifact_dir),
        "coordinates": {},
        "coherent": True,
    }
    evidence_bundle.persist_bundle(project_dir, stale_bundle)

    from types import SimpleNamespace

    ctx = SimpleNamespace(project_dir=project_dir, remaining_s=lambda: None)

    results = [
        {"outcome": "repairable", "metrics": {"acc": 0.1}},
        {"outcome": "repairable", "metrics": {"acc": 0.2}},
        {"outcome": "repairable", "metrics": {"acc": 0.3}},
    ]
    tools = {
        "understand_section": _wrap(_tool([{"ok": True}])[0]),
        "detect_environment": _wrap(_tool([{"ok": True}])[0]),
        "plan_reproduction": _wrap(_tool([{"ok": True, "steps": ["s1"]}])[0]),
        "implement_baseline": _wrap(_tool([{"ok": True, "code_path": str(tmp_path)}])[0]),
        "run_experiment": _wrap(_tool(results)[0]),
        "verify_against_rubric": _wrap(_tool([{"overall_score": 0.3}])[0]),
    }

    summary = ld.drive_lifecycle_chain(
        tools=tools, ctx=ctx, paper_text="p", rubric_spec={"leaves": []},
        start_stage="need_baseline", emit=lambda e: None, min_remaining_s=0.0,
        max_repair_iterations=2)

    # Genuine improvement across repairs -> cap reached normally, no
    # premature repair_exhausted stop despite the stale bundle on disk.
    assert summary["repaired"] == 2
    assert summary["stopped_reason"] is None
