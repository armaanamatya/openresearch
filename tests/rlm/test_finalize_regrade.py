"""Finalize-time freshness re-grade (finalize_regrade.py).

2026-06-13 All-CNN v5: a complete 13/14-converged grid shipped at 0.558
because it was graded ONCE on a partial grid and never re-graded. This rail
re-grades grown evidence at finalize and adopts a strictly-higher score.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm import finalize_regrade as fr


def _grid_metrics(n_converged: int = 13, n_dead: int = 1) -> dict:
    pm: dict = {}
    for i in range(n_converged):
        pm[f"m{i}"] = {"cifar10": {"base": {"test_error_pct": 12.0 + i, "test_accuracy": 0.88}}}
    for j in range(n_dead):
        pm[f"d{j}"] = {"cifar10": {"base": {"test_error_pct": 90.0, "test_accuracy": 0.1}}}
    return {"status": "completed", "per_model": pm}


def _project(tmp_path: Path, *, graded_at: float | None, metrics_at: float | None,
             recorded: float = 0.5413, target: float = 0.7437) -> Path:
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (tmp_path / "generated_rubric.json").write_text(json.dumps(
        {"source": "generated", "id": "r", "sub_tasks": []}))
    mp = code / "metrics.json"
    mp.write_text(json.dumps(_grid_metrics()))
    if metrics_at is not None:
        os.utime(mp, (metrics_at, metrics_at))
    if graded_at is not None:
        ev = tmp_path / "rubric_evaluation.json"
        ev.write_text(json.dumps({"overall_score": recorded, "target_score": target,
                                  "graded": 22, "leaf_count": 22}))
        os.utime(ev, (graded_at, graded_at))
    return tmp_path


# ---------------------------------------------------------------------------
# should_regrade gate
# ---------------------------------------------------------------------------


def test_fires_when_evidence_grew_after_grade(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    fire, reason = fr.should_regrade(p, recorded_score=0.5413, target=0.7437)
    assert fire is True
    assert "evidence_grew" in reason


def test_skips_when_grade_is_fresh(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 10, metrics_at=now)  # within margin
    fire, reason = fr.should_regrade(p, recorded_score=0.5413, target=0.7437)
    assert fire is False
    assert reason == "grade_is_fresh"


def test_skips_when_already_meets_target_and_fresh(tmp_path, monkeypatch):
    # At/above target with NO material new evidence since the grade → skip (no-op).
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 10, metrics_at=now)  # within margin
    fire, reason = fr.should_regrade(p, recorded_score=0.78, target=0.7437)
    assert fire is False
    assert reason == "already_meets_target"


def test_regrades_past_target_when_evidence_grew(tmp_path, monkeypatch):
    # Maximization (2026-06-14 Codex review): a grid that GREW after the grade is
    # re-graded EVEN at/above the floored target — best-of-run MAX adopts only if
    # the fresh grade is strictly higher, so the floor can never be lost.
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)  # grew
    fire, reason = fr.should_regrade(p, recorded_score=0.78, target=0.7437)
    assert fire is True
    assert "evidence_grew" in reason


def test_fires_when_no_recorded_grade(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=None, metrics_at=now)
    (p / "rubric_evaluation.json").unlink(missing_ok=True)
    fire, reason = fr.should_regrade(p, recorded_score=None, target=0.7437)
    assert fire is True


def test_skips_without_metrics(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    fire, reason = fr.should_regrade(tmp_path, recorded_score=0.5, target=0.74)
    assert fire is False
    assert reason == "no_metrics_on_disk"


def test_flag_disables(tmp_path, monkeypatch):
    monkeypatch.setenv(fr.ENV_FLAG, "0")
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    assert fr.should_regrade(p, recorded_score=0.54, target=0.74)[0] is False
    assert fr.is_enabled() is False


# ---------------------------------------------------------------------------
# converged-cell proxy
# ---------------------------------------------------------------------------


def test_converged_cell_count_allcnn_shape(tmp_path):
    # All-CNN per_model[model][env][baseline] = {test_error_pct, ...}; dead
    # cells still count (the grader judges quality, the gate counts evidence).
    assert fr._converged_cell_count(_grid_metrics(13, 1)) == 14
    assert fr._converged_cell_count({"per_model": {}}) == 0
    assert fr._converged_cell_count({}) == 0


def test_converged_cell_count_adam_flat_map_shape():
    # Adam per_model[family] = {optimizer: bare_scalar} — no metric key (rule b).
    m = {"per_model": {
        "mnist_logreg": {"adam": 0.33, "sgd_nesterov": 0.31, "adagrad": 0.41},
        "cifar10_cnn": {"adam": 0.53, "sgd_nesterov": 0.12},
        "vae": {"adam_bc": 104.1, "rmsprop": 126.9},
    }}
    assert fr._converged_cell_count(m) == 3  # one measured leaf per family


def test_converged_cell_count_ignores_empty_and_nonnumeric():
    assert fr._converged_cell_count({"per_model": {"x": {"note": "pending"}}}) == 0
    assert fr._converged_cell_count({"per_model": {"x": {}}}) == 0


# ---------------------------------------------------------------------------
# maybe_regrade — adopt / keep semantics
# ---------------------------------------------------------------------------


def _report(score=0.5413, target=0.7437, verdict="partial"):
    return SimpleNamespace(
        rubric={"overall_score": score, "target_score": target, "meets_target": False},
        verdict=verdict,
    )


def _ctx(project_dir, fresh_score):
    # llm_client is opaque to maybe_regrade; score_reproduction is monkeypatched.
    return SimpleNamespace(project_dir=project_dir, llm_client=object(),
                           paper_hint_invariants=[])


def test_adopts_strictly_higher_regrade(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.731, "target_score": 0.7437,
                      "graded": 22, "leaf_count": 22, "leaf_scores": [], "areas": []},
    )
    report = _report()
    fresh = fr.maybe_regrade(_ctx(p, 0.731), report)
    assert fresh is not None
    assert report.rubric["overall_score"] == pytest.approx(0.731)
    # Persisted for the report merge.
    saved = json.loads((p / "rubric_evaluation.json").read_text())
    assert saved["overall_score"] == pytest.approx(0.731)


def test_regrade_passes_degraded_false(tmp_path, monkeypatch):
    """The _converged_cell_count gate proves real converged cells, so the regrade
    MUST pass degraded=False explicitly. Otherwise score_reproduction's degraded=None
    auto-detect reads a stale failed/empty-baseline final_report.json and caps every
    leaf at 0.35 — nuking the very complete-grid grade the regrade exists to recover."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    captured: dict = {}

    def _spy(**kw):
        captured.update(kw)
        return {"overall_score": 0.80, "target_score": 0.7437,
                "graded": 22, "leaf_count": 22, "leaf_scores": [], "areas": []}

    monkeypatch.setattr("backend.evals.paperbench.leaf_scorer.score_reproduction", _spy)
    fr.maybe_regrade(_ctx(p, 0.80), _report())
    assert captured.get("degraded") is False


def test_keeps_recorded_when_regrade_not_higher(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.52, "target_score": 0.7437},
    )
    report = _report()
    assert fr.maybe_regrade(_ctx(p, 0.52), report) is None
    assert report.rubric["overall_score"] == pytest.approx(0.5413)  # untouched


def test_adopted_regrade_meeting_target_flips_meets(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now, target=0.60)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.731, "target_score": 0.60,
                      "leaf_scores": [], "areas": []},
    )
    report = _report(target=0.60)
    fresh = fr.maybe_regrade(_ctx(p, 0.731), report)
    assert fresh["meets_target"] is True
    assert report.rubric["meets_target"] is True


def test_skips_regrade_when_no_converged_cells(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    (p / "code" / "metrics.json").write_text(json.dumps({"per_model": {}}))
    os.utime(p / "code" / "metrics.json", (now, now))
    called = []
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: called.append(1) or {"overall_score": 0.9},
    )
    assert fr.maybe_regrade(_ctx(p, 0.9), _report()) is None
    assert called == []  # no LLM call spent on empty evidence


def test_never_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    bad_ctx = SimpleNamespace(project_dir="/nonexistent/xyz", llm_client=None)
    assert fr.maybe_regrade(bad_ctx, _report()) is None


# ---------------------------------------------------------------------------
# regrade_and_emit — always emits a reason (observability)
# ---------------------------------------------------------------------------


def test_regrade_and_emit_emits_skip_reason(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 10, metrics_at=now)  # fresh → skip
    events = []
    fr.regrade_and_emit(_ctx(p, 0.5), _report(), lambda e, pl: events.append((e, pl["code"])))
    assert ("run_warning", "finalize_regrade_skipped") in events


def test_regrade_and_emit_emits_adopt(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.73, "target_score": 0.7437, "leaf_scores": [], "areas": []},
    )
    events = []
    fr.regrade_and_emit(_ctx(p, 0.73), _report(), lambda e, pl: events.append(pl["code"]))
    assert "finalize_regrade_adopted" in events


def test_regrade_and_emit_disabled_is_silent(tmp_path, monkeypatch):
    monkeypatch.setenv(fr.ENV_FLAG, "0")
    events = []
    assert fr.regrade_and_emit(_ctx(tmp_path, 0.5), _report(), lambda e, pl: events.append(e)) is None
    assert events == []


# ---------------------------------------------------------------------------
# regrade_for_hard_stop — re-grade a completed grid with no ctx
# ---------------------------------------------------------------------------


def test_hard_stop_regrade_grades_completed_grid(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    p = _project(tmp_path, graded_at=None, metrics_at=time.time())
    (p / "rubric_evaluation.json").unlink(missing_ok=True)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.71, "target_score": 0.7437},
    )
    fresh = fr.regrade_for_hard_stop(p, llm_client=object())
    assert fresh["overall_score"] == pytest.approx(0.71)
    # Persisted so the salvage floor + report merge read it.
    assert json.loads((p / "rubric_evaluation.json").read_text())["overall_score"] == pytest.approx(0.71)


def test_hard_stop_regrade_skips_without_client(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    p = _project(tmp_path, graded_at=None, metrics_at=time.time())
    assert fr.regrade_for_hard_stop(p, llm_client=None) is None


def test_hard_stop_regrade_skips_empty_grid(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    p = _project(tmp_path, graded_at=None, metrics_at=time.time())
    (p / "code" / "metrics.json").write_text(json.dumps({"per_model": {}}))
    called = []
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: called.append(1) or {"overall_score": 0.9},
    )
    assert fr.regrade_for_hard_stop(p, llm_client=object()) is None
    assert called == []


# ---------------------------------------------------------------------------
# P0 (2026-07-13): the failed→reproduced UPGRADE CLAMP
#
# code/metrics.json is ROOT-WRITABLE and _converged_cell_count only proves NUMBERS
# ARE IN THE FILE — not that a container ever ran. So the regrade used to hand a
# run that never produced one clean success row a full-credit (degraded=False)
# re-grade AND flip its verdict failed→reproduced. Both now require the same
# unforgeable signal report.py's two-axis clamp uses: >=1 success-compatible
# in-process run_experiment cost-ledger call. Downgrades stay free.
# ---------------------------------------------------------------------------


class _Entry:
    def __init__(self, agent_id: str, outcome: str) -> None:
        self.agent_id = agent_id
        self.outcome = outcome


class _Ledger:
    """The in-process cost ledger shape report.py's canonical counters read.

    Deliberately does NOT implement session_*_count, so the counters fall through to
    their entry-scanning path — proving the clamp works against the real ledger
    contract (agent_id + per-row `outcome` provenance stamp), not a stub of it.
    """

    def __init__(self, entries: list[_Entry]) -> None:
        self.entries = entries


def _ctx_with_ledger(project_dir, outcomes: list[str]):
    """A ctx whose run_experiment ledger rows carry these outcome stamps."""
    return SimpleNamespace(
        project_dir=project_dir,
        llm_client=object(),
        paper_hint_invariants=[],
        cost_ledger=_Ledger([_Entry("run_experiment", o) for o in outcomes]),
    )


def _fresh_grade(score: float = 0.82):
    return lambda **kw: {"overall_score": score, "target_score": 0.60,
                         "leaf_scores": [], "areas": []}


def test_no_ledger_keeps_content_only_trust(tmp_path, monkeypatch):
    """None (replay/postmortem — no ledger) → today's behaviour, exactly like
    report._apply_evidence_gate's posture. This is what every pre-existing test hits."""
    from backend.agents.rlm import finalize_regrade as _fr
    assert _fr._experiment_backed(SimpleNamespace(project_dir=tmp_path)) is None
    assert _fr._experiment_backed(None) is None


def test_ledger_counts_only_success_compatible_rows(tmp_path):
    from backend.agents.rlm import finalize_regrade as _fr
    assert _fr._experiment_backed(_ctx_with_ledger(tmp_path, ["ok"])) is True
    assert _fr._experiment_backed(_ctx_with_ledger(tmp_path, [""])) is True  # legacy/unknown
    assert _fr._experiment_backed(_ctx_with_ledger(tmp_path, ["failed"])) is False
    assert _fr._experiment_backed(_ctx_with_ledger(tmp_path, ["raised", "failed"])) is False
    assert _fr._experiment_backed(_ctx_with_ledger(tmp_path, [])) is False
    # A harness-finalized timeout partial IS real work (the harness loaded the metrics
    # off disk itself) — the salvage tier this module exists for.
    assert _fr._experiment_backed(_ctx_with_ledger(tmp_path, ["partial_timeout"])) is True


def test_regrade_refused_when_no_backing_experiment_call(tmp_path, monkeypatch):
    """THE P0: a run whose every run_experiment call FAILED, but which wrote a plausible
    metrics.json, gets NO full-credit re-grade — and no LLM call is spent on it."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    called = []
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: called.append(1) or _fresh_grade()(**kw),
    )
    report = _report(verdict="failed")
    assert fr.maybe_regrade(_ctx_with_ledger(p, ["failed"]), report) is None
    assert called == [], "no LLM call may be spent grading unbacked numbers"
    assert report.verdict == "failed"
    assert report.rubric["overall_score"] == pytest.approx(0.5413)  # untouched


def test_cannot_flip_failed_to_reproduced_without_a_ledger_row(tmp_path, monkeypatch):
    """The flip site: even if a grade IS adopted, the verdict upgrade is clamped when no
    success-compatible run_experiment call backs it — and the refusal is stamped."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction", _fresh_grade(0.95)
    )
    ctx = _ctx_with_ledger(p, ["failed"])
    report = _report(verdict="failed")
    events: list[dict] = []
    fr.regrade_and_emit(ctx, report, lambda e, pl: events.append(pl))

    assert report.verdict == "failed", "a root-writable metrics.json cannot buy 'reproduced'"
    assert report.rubric["overall_score"] == pytest.approx(0.5413)  # score not lifted either
    # The refusal is STAMPED and EMITTED, never silent (report.py's verdict_clamped key).
    assert "zero success-compatible run_experiment calls" in report.rubric["verdict_clamped"]
    assert any(e.get("code") == "finalize_regrade_verdict_clamped" for e in events)
    assert not any(e.get("code") == "finalize_regrade_adopted" for e in events)


def test_clamp_stamps_verdict_clamped_directly(tmp_path):
    """Unit-level: the clamp itself refuses the upgrade and stamps, mirroring
    report.py::write_final_report_rlm's two-axis clamp convention."""
    report = _report(verdict="failed")
    fr._apply_regrade_verdict(_ctx_with_ledger(tmp_path, ["raised"]), report, 0.99, None)
    assert report.verdict == "failed"
    assert report.rubric["verdict_clamped"].startswith(
        "upgrade from 'failed' to 'reproduced' refused"
    )


def test_genuine_success_row_still_regrades_and_upgrades(tmp_path, monkeypatch):
    """NO REGRESSION: a real grid backed by a successful in-process run_experiment call
    still gets its full-credit re-grade AND its earned verdict — the All-CNN v5 / Adam v10
    recovery this module exists for."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    captured: dict = {}

    def _spy(**kw):
        captured.update(kw)
        return {"overall_score": 0.82, "target_score": 0.60, "leaf_scores": [], "areas": []}

    monkeypatch.setattr("backend.evals.paperbench.leaf_scorer.score_reproduction", _spy)
    ctx = _ctx_with_ledger(p, ["ok", "failed"])  # one real success among the calls
    report = _report(target=0.60, verdict="failed")
    events: list[dict] = []
    fresh = fr.regrade_and_emit(ctx, report, lambda e, pl: events.append(pl))

    assert fresh is not None
    assert captured.get("degraded") is False       # the ceiling bypass is EARNED here
    assert report.verdict == "reproduced"          # 0.82 >= the reproduced band
    assert "verdict_clamped" not in report.rubric
    assert any(e.get("code") == "finalize_regrade_adopted" for e in events)


def test_partial_timeout_only_run_is_capped_at_partial(tmp_path, monkeypatch):
    """The harness-finalized timeout tier: a container DID run and the HARNESS loaded its
    metrics, so the salvage still lands — but seeded at 'partial', never 'reproduced'
    (mirrors _apply_evidence_gate's partial-timeout cap)."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction", _fresh_grade(0.95)
    )
    ctx = _ctx_with_ledger(p, ["partial_timeout"])
    report = _report(verdict="failed")
    fr.regrade_and_emit(ctx, report, lambda e, pl: None)
    assert report.verdict == "partial"
    assert "verdict_clamped" not in report.rubric


def test_downgrade_stays_free(tmp_path, monkeypatch):
    """Fail-closed: the clamp only blocks UPGRADES. A low adopted score still drags a
    'reproduced' verdict down — a guard must never be able to raise a verdict."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    report = _report(score=None, target=0.60, verdict="reproduced")
    # A real backed call — so no clamp — but the fresh score only supports 'partial'.
    fr._apply_regrade_verdict(_ctx_with_ledger(p, ["ok"]), report, 0.20, None)
    assert report.verdict == "partial"

    # And with NO backing call at all, the downgrade must STILL land — the clamp refuses
    # to LIFT a verdict, it never freezes one in place.
    report2 = _report(score=None, target=0.60, verdict="reproduced")
    fr._apply_regrade_verdict(_ctx_with_ledger(p, ["failed"]), report2, 0.20, None)
    assert report2.verdict == "partial"
    assert "verdict_clamped" not in report2.rubric  # nothing was refused — it went down


def test_hard_stop_regrade_refused_without_backing_call(tmp_path, monkeypatch):
    """The no-ctx salvage path: when the ledger IS threaded through and shows no real
    call, the degraded=False full-credit bypass is refused there too."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    p = _project(tmp_path, graded_at=None, metrics_at=time.time())
    called = []
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: called.append(1) or {"overall_score": 0.9},
    )
    assert fr.regrade_for_hard_stop(
        p, llm_client=object(), ctx=_ctx_with_ledger(p, ["failed"])
    ) is None
    assert called == []
    # ...and byte-identical (today's caller passes no ctx → content-only trust).
    assert fr.regrade_for_hard_stop(p, llm_client=object())["overall_score"] == pytest.approx(0.9)
