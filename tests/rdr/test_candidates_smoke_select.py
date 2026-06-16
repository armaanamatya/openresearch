"""Smoke-gated BES SELECT (candidates.py §D2, 2026-06-16).

Pure unit coverage for the three §D2 levers, all gated on
``REPROLAB_BES_SMOKE_SELECT`` (default OFF):

  1. ``smoke_check_candidate`` — the read-only AST construct/import smoke;
  2. ``select_best_gated`` smoke gate — a non-runnable candidate (hard AST
     violation) cannot outrank a runnable one even at a higher static score;
  3. ``select_best_gated`` sub-σ tie-break — a top-2 spread below the σ_grader
     proxy is broken DETERMINISTICALLY (AST-completeness → fewest violations →
     lowest index), never on the noisy score;
  4. degenerate-pool verdict — no runnable candidate ⇒ ``(None, {degenerate})``.

Off-path: with the flag unset, ``select_best_gated`` is a pass-through to
``select_best`` (asserted explicitly).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.rdr.candidates import (
    Candidate,
    SmokeResult,
    select_best,
    select_best_gated,
    smoke_check_candidate,
)
from backend.agents.rdr.models import Artifacts


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _cand(cid: str, *, score: float | None, failed: bool = False, leaves=()) -> Candidate:
    return Candidate(
        candidate_id=cid,
        cluster_id="c",
        scratch_dir=Path("/tmp") / cid,
        artifacts=Artifacts(cluster_id="c", failed=failed),
        score=score,
        failed_leaves=list(leaves),
    )


def _smoke(cid: str, *, runnable: bool, complete: bool = True, hard: int = 0, soft: int = 0) -> SmokeResult:
    return SmokeResult(
        candidate_id=cid,
        checked=True,
        runnable=runnable,
        ast_complete=complete,
        hard_violations=hard,
        soft_violations=soft,
    )


@pytest.fixture(autouse=True)
def _no_flag_leak(monkeypatch):
    monkeypatch.delenv("REPROLAB_BES_SMOKE_SELECT", raising=False)
    monkeypatch.delenv("REPROLAB_BES_SELECT_MIN_SPREAD", raising=False)


# ---------------------------------------------------------------------------
# smoke_check_candidate — the construct/import smoke over a snapshot
# ---------------------------------------------------------------------------


def test_smoke_clean_code_is_runnable_and_complete(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "train.py").write_text("import os\n\n\ndef main():\n    return os.getcwd()\n")
    res = smoke_check_candidate("c#0", code)
    assert res.checked is True
    assert res.runnable is True
    assert res.ast_complete is True
    assert res.hard_violations == 0


def test_smoke_missing_method_is_not_runnable(tmp_path):
    """A guaranteed AttributeError (call to an undefined method on a same-file
    class) is a HARD violation → runnable=False — the VAE-crash-class bug."""
    code = tmp_path / "code"
    code.mkdir()
    (code / "train.py").write_text(
        "class Model:\n"
        "    def forward(self, x):\n"
        "        return x\n\n"
        "m = Model()\n"
        "m.reparameterize(1, 2, 3)\n"
    )
    res = smoke_check_candidate("c#0", code)
    assert res.checked is True
    assert res.runnable is False
    assert res.ast_complete is False
    assert res.hard_violations >= 1
    assert "reparameterize" in res.detail


def test_smoke_missing_code_dir_is_fail_soft(tmp_path):
    res = smoke_check_candidate("c#0", tmp_path / "does_not_exist")
    assert res.checked is False
    assert res.runnable is True  # not disqualifying


# ---------------------------------------------------------------------------
# select_best_gated — OFF path (byte-for-byte select_best)
# ---------------------------------------------------------------------------


def test_gated_off_is_passthrough_to_select_best(monkeypatch):
    # Flag unset by the autouse fixture.
    pool = [_cand("c#0", score=0.3), _cand("c#1", score=0.7)]
    winner, decision = select_best_gated(pool)
    assert decision == {"path": "legacy"}
    assert winner is select_best(pool)
    assert winner.candidate_id == "c#1"


def test_gated_off_ignores_smokes(monkeypatch):
    """Off path must not consult smokes at all — even a non-runnable top still
    wins, exactly as today (the gate is the opt-in)."""
    pool = [_cand("c#0", score=0.9), _cand("c#1", score=0.2)]
    smokes = {"c#0": _smoke("c#0", runnable=False, complete=False, hard=2)}
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert winner.candidate_id == "c#0"
    assert decision["path"] == "legacy"


# ---------------------------------------------------------------------------
# select_best_gated — smoke gate (non-runnable loses)
# ---------------------------------------------------------------------------


def test_gated_non_runnable_top_loses_to_runnable(monkeypatch):
    """The headline §D2 case: a statically-faithful but non-runnable candidate
    (higher static grade) cannot win over a runnable one."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    pool = [_cand("c#0", score=0.9), _cand("c#1", score=0.2)]
    smokes = {
        "c#0": _smoke("c#0", runnable=False, complete=False, hard=2),
        "c#1": _smoke("c#1", runnable=True),
    }
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert winner.candidate_id == "c#1"  # runnable wins despite lower score
    assert decision["path"] == "smoke_gated"
    assert "c#0" in decision["smoke_dropped"]


def test_gated_unsmoked_candidate_is_not_disqualified(monkeypatch):
    """A candidate with no smoke entry is fail-soft 'runnable' and competes on
    its score exactly as today (so a partial smoke map never drops a peer)."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    pool = [_cand("c#0", score=0.8), _cand("c#1", score=0.2)]
    winner, decision = select_best_gated(pool, smokes={})  # no smokes computed
    assert winner.candidate_id == "c#0"
    assert decision["tie_break"] == "score"


# ---------------------------------------------------------------------------
# select_best_gated — sub-σ tie-break determinism
# ---------------------------------------------------------------------------


def test_gated_sub_sigma_tie_breaks_on_completeness_not_score(monkeypatch):
    """Top-2 within the σ proxy → the COMPLETE candidate wins even though it has
    the (marginally) lower score — the grader noise is not banked."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    # spread 0.557 - 0.549 = 0.008 < 0.05 default → tie.
    pool = [_cand("c#0", score=0.549), _cand("c#1", score=0.557)]
    smokes = {
        "c#0": _smoke("c#0", runnable=True, complete=True),       # complete
        "c#1": _smoke("c#1", runnable=True, complete=False, soft=1),  # incomplete
    }
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert winner.candidate_id == "c#0"  # complete beats the higher noisy score
    assert decision["tie_break"] == "sub_sigma"
    assert set(decision["tie_set"]) == {"c#0", "c#1"}


def test_gated_sub_sigma_falls_to_lowest_index_when_all_equal(monkeypatch):
    """Equal completeness within the tie band → deterministic lowest index."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    pool = [_cand("c#0", score=0.560), _cand("c#1", score=0.561), _cand("c#2", score=0.559)]
    smokes = {c.candidate_id: _smoke(c.candidate_id, runnable=True, complete=True) for c in pool}
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert winner.candidate_id == "c#0"  # lowest index, NOT the 0.561 top
    assert decision["tie_break"] == "sub_sigma"


def test_gated_sub_sigma_is_deterministic_across_input_orderings(monkeypatch):
    """The whole point: same pool, any presentation order → same winner."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    a = _cand("c#0", score=0.560)
    b = _cand("c#1", score=0.561)
    smokes = {"c#0": _smoke("c#0", runnable=True, complete=True),
              "c#1": _smoke("c#1", runnable=True, complete=True)}
    w1, _ = select_best_gated([a, b], smokes=smokes)
    w2, _ = select_best_gated([b, a], smokes=smokes)  # reversed input
    # c#0 is the lowest-index member of the tie set in BOTH orderings.
    assert w1.candidate_id == "c#0"
    assert w2.candidate_id == "c#0"


def test_gated_wide_spread_keeps_the_higher_score(monkeypatch):
    """Above the σ proxy the real score wins — the tie-break must NOT fire."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    pool = [_cand("c#0", score=0.3), _cand("c#1", score=0.7)]  # spread 0.4
    smokes = {c.candidate_id: _smoke(c.candidate_id, runnable=True, complete=True) for c in pool}
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert winner.candidate_id == "c#1"
    assert decision["tie_break"] == "score"


def test_gated_min_spread_flag_widens_tie_band(monkeypatch):
    """A larger REPROLAB_BES_SELECT_MIN_SPREAD pulls a wider gap into the tie."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    monkeypatch.setenv("REPROLAB_BES_SELECT_MIN_SPREAD", "0.30")
    pool = [_cand("c#0", score=0.5), _cand("c#1", score=0.7)]  # spread 0.2 < 0.30
    smokes = {
        "c#0": _smoke("c#0", runnable=True, complete=True),
        "c#1": _smoke("c#1", runnable=True, complete=False, soft=1),
    }
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert decision["tie_break"] == "sub_sigma"
    assert winner.candidate_id == "c#0"  # complete wins the now-tied band


# ---------------------------------------------------------------------------
# select_best_gated — degenerate pool
# ---------------------------------------------------------------------------


def test_gated_all_non_runnable_is_degenerate(monkeypatch):
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    pool = [_cand("c#0", score=0.8), _cand("c#1", score=0.6)]
    smokes = {
        "c#0": _smoke("c#0", runnable=False, hard=1),
        "c#1": _smoke("c#1", runnable=False, hard=3),
    }
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert winner is None
    assert decision["degenerate"] is True
    assert decision["n_runnable"] == 0
    assert set(decision["smoke_dropped"]) == {"c#0", "c#1"}


def test_gated_all_failed_implementation_is_degenerate(monkeypatch):
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    pool = [_cand("c#0", score=None, failed=True), _cand("c#1", score=None, failed=True)]
    winner, decision = select_best_gated(pool, smokes={})
    assert winner is None
    assert decision["degenerate"] is True


def test_gated_one_runnable_survivor_is_not_degenerate(monkeypatch):
    """A single runnable survivor among non-runnable peers still wins (not
    degenerate) — the pool produced something it can ship."""
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    pool = [_cand("c#0", score=0.9), _cand("c#1", score=0.4)]
    smokes = {
        "c#0": _smoke("c#0", runnable=False, hard=2),
        "c#1": _smoke("c#1", runnable=True),
    }
    winner, decision = select_best_gated(pool, smokes=smokes)
    assert winner is not None and winner.candidate_id == "c#1"
    assert not decision.get("degenerate")


def test_gated_empty_pool_is_degenerate(monkeypatch):
    monkeypatch.setenv("REPROLAB_BES_SMOKE_SELECT", "1")
    winner, decision = select_best_gated([], smokes={})
    assert winner is None
    assert decision["degenerate"] is True
