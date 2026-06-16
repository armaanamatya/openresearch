"""Smoke-gated BES SELECT on the RLM path (bes_rlm.py §D2, 2026-06-16).

End-to-end coverage of the three §D2 behaviours through the real
``bes_rlm.compete`` loop (the candidates.py unit layer is covered by
``tests/rdr/test_candidates_smoke_select.py``):

  * smoke-gated select — a statically-faithful but non-runnable candidate loses
    to a runnable one even at a higher static grade;
  * degenerate_pool fall-through — every candidate non-runnable ⇒ a coded
    ``degenerate_pool`` run_warning + ONE single-shot repair, no doomed winner;
  * off-path parity — with the flag unset, no smoke is computed and the legacy
    SELECT decision is recorded (no regression to ``test_bes_rlm.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm import bes_rlm


# ---------------------------------------------------------------------------
# Harness (mirrors tests/rlm/test_bes_rlm.py)
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
    events: list[dict] = []
    ctx = SimpleNamespace(
        project_id="prj_test",
        project_dir=project_dir,
        runs_root=tmp_path,
        llm_client=SimpleNamespace(complete=lambda **kw: "{}"),
        remaining_s=lambda: remaining,
        events=events,
    )
    return ctx


def _write_rubric(ctx) -> None:
    (Path(ctx.project_dir) / "generated_rubric.json").write_text(
        json.dumps({"source": "generated", "leaves": []}), encoding="utf-8"
    )


# Per-candidate code bodies keyed by candidate index. A "broken" body carries a
# guaranteed-AttributeError call (a HARD preflight_ast violation → not runnable).
_RUNNABLE_BODY = "import os\n\n\ndef main():\n    return os.getcwd()\n"
_BROKEN_BODY = (
    "class Model:\n"
    "    def forward(self, x):\n"
    "        return x\n\n"
    "m = Model()\n"
    "m.reparameterize(1, 2, 3)\n"  # no such method → hard violation
)


def _fake_implement(code_dir: Path, *, bodies: dict[int, str], fail_on: set[int] = frozenset()):
    """An implement_fn that writes a per-index code body into code/train.py."""
    calls: list[dict] = []

    def fn(plan, *, ctx, _bes_inner=False):
        idx = plan.get("_bes_candidate_idx", -1)
        calls.append({"idx": idx})
        if idx in fail_on:
            return {"ok": False, "error_code": "boom", "error": f"candidate {idx} failed", "repairable": True}
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))
        (code_dir / "train.py").write_text(bodies.get(idx, _RUNNABLE_BODY))
        return {"ok": True, "code_path": str(code_dir), "files": ["train.py", "commands.json"]}

    fn.calls = calls
    return fn


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    for var in ("REPROLAB_AB_ARM", "REPROLAB_AB_PAIR_ID", "REPROLAB_BES_MIN_REMAINING_S",
                "REPROLAB_BES_CONTINUE_MIN_S", "REPROLAB_BASELINE_EXTRA_GUIDANCE",
                "REPROLAB_BES_SMOKE_SELECT", "REPROLAB_BES_SELECT_MIN_SPREAD"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Smoke-gated select — a non-runnable candidate cannot win
# ---------------------------------------------------------------------------


def test_smoke_select_non_runnable_high_score_loses(tmp_path, monkeypatch):
    """Candidate #0 grades HIGHER but is non-runnable (hard AST violation);
    the runnable #1 must win once REPROLAB_BES_SMOKE_SELECT is on."""
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir, bodies={0: _BROKEN_BODY, 1: _RUNNABLE_BODY})
    # #0 (broken) scores higher than #1 (runnable).
    scores = {0: (0.9, []), 1: (0.4, [])}
    monkeypatch.setattr(
        bes_rlm, "_static_grade",
        lambda r, d, c: scores[int(str(d).rsplit("_", 1)[-1])],
    )

    result = bes_rlm.compete({"paper_claim_map": {}}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert result["bes"]["selected"] == "rlm_impl#1"  # runnable wins
    decision = result["bes"]["select_decision"]
    assert decision["path"] == "smoke_gated"
    assert "rlm_impl#0" in decision["smoke_dropped"]
    # Winner's runnable code restored into the real code/.
    assert "reparameterize" not in (code_dir / "train.py").read_text()


def test_smoke_select_off_keeps_legacy_winner(tmp_path, monkeypatch):
    """Flag OFF: the higher static grade wins even if non-runnable (byte-for-byte
    today). The decision is recorded as 'legacy'; no smoke is consulted."""
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    # REPROLAB_BES_SMOKE_SELECT unset by the autouse fixture.
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir, bodies={0: _BROKEN_BODY, 1: _RUNNABLE_BODY})
    scores = {0: (0.9, []), 1: (0.4, [])}
    monkeypatch.setattr(
        bes_rlm, "_static_grade",
        lambda r, d, c: scores[int(str(d).rsplit("_", 1)[-1])],
    )

    result = bes_rlm.compete({"paper_claim_map": {}}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert result["bes"]["selected"] == "rlm_impl#0"  # legacy: highest score wins
    assert result["bes"]["select_decision"] == {"path": "legacy"}


# ---------------------------------------------------------------------------
# Degenerate pool fall-through
# ---------------------------------------------------------------------------


def test_degenerate_pool_falls_through_to_single_shot_repair(tmp_path, monkeypatch):
    """Every candidate non-runnable → a degenerate_pool run_warning + ONE extra
    single-shot repair implementation, instead of selecting a doomed winner."""
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    # Both competing candidates are broken; the repair shot writes runnable code.
    impl = _fake_implement(code_dir, bodies={0: _BROKEN_BODY, 1: _BROKEN_BODY, -1: _RUNNABLE_BODY})
    monkeypatch.setattr(bes_rlm, "_static_grade", lambda r, d, c: (0.7, []))

    warnings: list[dict] = []
    monkeypatch.setattr(
        bes_rlm, "_emit",
        lambda ctx, ev, payload: warnings.append(payload) if ev == "run_warning" else None,
    )

    result = bes_rlm.compete({"paper_claim_map": {}}, ctx=ctx, implement_fn=impl)

    # The single-shot repair ran (a 3rd implement call, idx -1) and succeeded.
    assert result["ok"] is True
    assert result["bes"]["selected"] is None
    assert result["bes"]["degenerate_pool"] is True
    assert [c["idx"] for c in impl.calls] == [0, 1, -1]
    # A degenerate_pool warning was emitted naming the smoke-dropped candidates.
    codes = [w.get("code") for w in warnings]
    assert "degenerate_pool" in codes
    dp = next(w for w in warnings if w.get("code") == "degenerate_pool")
    assert set(dp["smoke_dropped"]) == {"rlm_impl#0", "rlm_impl#1"}
    # No winner marker persisted (the pool produced no winner to be idempotent on).
    assert not (Path(ctx.project_dir) / "rlm_state" / "bes_winner.json").exists()


def test_all_runnable_records_smoke_decision(tmp_path, monkeypatch):
    """Sanity: a healthy pool still selects and stamps a smoke_gated decision
    (not degenerate) so the A/B stamp carries the SELECT provenance."""
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(n=2))
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    ctx = _ctx(tmp_path)
    _write_rubric(ctx)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    impl = _fake_implement(code_dir, bodies={0: _RUNNABLE_BODY, 1: _RUNNABLE_BODY})
    scores = {0: (0.3, []), 1: (0.7, [])}  # wide spread → score wins
    monkeypatch.setattr(
        bes_rlm, "_static_grade",
        lambda r, d, c: scores[int(str(d).rsplit("_", 1)[-1])],
    )

    result = bes_rlm.compete({"paper_claim_map": {}}, ctx=ctx, implement_fn=impl)

    assert result["ok"] is True
    assert result["bes"]["selected"] == "rlm_impl#1"
    decision = result["bes"]["select_decision"]
    assert decision["path"] == "smoke_gated"
    assert decision["tie_break"] == "score"
    assert decision["smoke_dropped"] == []
    # Persisted pool state carries the decision for the report stamp.
    state = json.loads((Path(ctx.project_dir) / "rlm_state" / "bes_candidates.json").read_text())
    assert state["select_decision"]["path"] == "smoke_gated"
