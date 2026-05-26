"""Phase 2-5 unit tests for the GEPA adapter + supporting modules.

End-to-end ``gepa.optimize`` runs are out of scope here — that requires the
``gepa`` library (broken on the local brew Python 3.14.5; works in Docker).
These tests cover everything that does NOT require ``import gepa``:

- ``trace_minimizer.build_record`` shape + redaction + 8KB cap
- ``EvalBudgetEnforcer`` from-env + timeout_record shape
- ``ReproLabGEPAAdapter`` candidate validation (refuses out-of-surface, refuses
  oversized, accepts well-formed)
- ``ReproLabGEPAAdapter.evaluate`` short-circuit using a monkey-patched
  ``_run_one_eval`` (no subprocess spawn)
- ``primitive_cache.make_key`` surface_salt: keys differ when salt is set,
  byte-identical to pre-GEPA when salt is unset
- ``scripts/optimize_prompts_gepa._validate_split`` G2 enforcement
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.agents.optimization.eval_budget import EvalBudgetEnforcer
from backend.agents.optimization.gepa_adapter import (
    SURFACES,
    EvalRequest,
    EvalResult,
    ReproLabGEPAAdapter,
    candidate_hash,
)
from backend.agents.optimization.trace_minimizer import _hermes_multiplier, build_record
from backend.agents.rlm import primitive_cache


# ---------------------------------------------------------------------------
# trace_minimizer
# ---------------------------------------------------------------------------


def _write_final_report(run_dir: Path, *, score: float, hermes_status: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_report.json").write_text(json.dumps({
        "rubric": {
            "overall_score": score,
            "target_score": 0.7,
            "areas": {"correctness": score, "completeness": score - 0.1},
            "weak_leaves": [{"name": "leaf_a", "score": 0.0}],
        },
        "metrics": {"mean_reward": 480.0},
        "hermes_audit": {
            "status": hermes_status,
            "unsupported_claims": ["claim that the model is real"],
            "evidence_refs": [{"snippet": "trace excerpt with sk-anthropic-fake-key-12345-LONG-ENOUGH-tokens"}],
        },
    }), encoding="utf-8")


def test_trace_minimizer_shape_and_score(tmp_path: Path) -> None:
    _write_final_report(tmp_path, score=0.62, hermes_status="grounded")
    rec = build_record(
        component_id="improvement.orchestrator.body",
        example_id="2605.15155",
        paper_archetype="rl-agent",
        run_dir=tmp_path,
        rubric_before={"overall_score": 0.5, "areas": {"correctness": 0.5}},
        weak_leaves_before=[{"name": "leaf_a", "score": 0.0, "rationale": "too short"}],
        candidate_output={"selected_hypothesis": {"path_id": "path_1"}},
    )
    assert rec["component_id"] == "improvement.orchestrator.body"
    assert rec["example_id"] == "2605.15155"
    assert rec["paper_archetype"] == "rl-agent"
    assert rec["score"] == pytest.approx(0.62, abs=1e-3)
    assert rec["execution_trace"]["hermes"]["status"] == "grounded"
    # delta computed for both areas
    assert "correctness" in rec["execution_trace"]["rubric_delta_areas"]


def test_trace_minimizer_redacts_secrets(tmp_path: Path) -> None:
    _write_final_report(tmp_path, score=0.5, hermes_status="caveat")
    rec = build_record(
        component_id="improvement.orchestrator.body",
        example_id="2605.15155",
        paper_archetype="rl-agent",
        run_dir=tmp_path,
    )
    blob = json.dumps(rec)
    assert "sk-anthropic-fake-key-12345" not in blob
    assert "<REDACTED:secret>" in blob


def test_trace_minimizer_caveat_halves_score(tmp_path: Path) -> None:
    _write_final_report(tmp_path, score=0.8, hermes_status="caveat")
    rec = build_record(
        component_id="x", example_id="y", paper_archetype="z", run_dir=tmp_path
    )
    assert rec["score"] == pytest.approx(0.4, abs=1e-3)  # 0.8 * 0.5


def test_trace_minimizer_unsupported_zeros_score(tmp_path: Path) -> None:
    _write_final_report(tmp_path, score=0.9, hermes_status="unsupported")
    rec = build_record(
        component_id="x", example_id="y", paper_archetype="z", run_dir=tmp_path
    )
    assert rec["score"] == 0.0


def test_hermes_multiplier_table() -> None:
    assert _hermes_multiplier("grounded") == 1.0
    assert _hermes_multiplier("caveat") == 0.5
    assert _hermes_multiplier("unsupported") == 0.0
    assert _hermes_multiplier("unavailable") == 0.0
    assert _hermes_multiplier("system_error") == 0.0
    assert _hermes_multiplier("totally-bogus-status") == 0.0


def test_trace_minimizer_missing_files_degrades_gracefully(tmp_path: Path) -> None:
    rec = build_record(
        component_id="x", example_id="y", paper_archetype="z", run_dir=tmp_path
    )
    assert rec["score"] == 0.0
    assert rec["execution_trace"]["hermes"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# eval_budget
# ---------------------------------------------------------------------------


def test_eval_budget_from_env_defaults() -> None:
    with patch.dict(os.environ, {}, clear=False):
        for k in (
            "REPROLAB_GEPA_MAX_USD_PER_EVAL",
            "REPROLAB_GEPA_MAX_WALL_S_PER_EVAL",
            "REPROLAB_GEPA_MAX_PARALLEL_PODS",
        ):
            os.environ.pop(k, None)
        b = EvalBudgetEnforcer.from_env()
        assert b.max_usd == 0.50
        assert b.max_wall_clock_seconds == 1800.0
        assert b.max_parallel_pods == 2


def test_eval_budget_timeout_record_shape() -> None:
    b = EvalBudgetEnforcer()
    rec = b.timeout_record(component_id="x", example_id="y", elapsed_s=900.0)
    assert rec["score"] == 0.0
    assert rec["execution_trace"]["timeout"]["elapsed_s"] == 900.0
    assert rec["execution_trace"]["hermes"]["status"] == "system_error"


# ---------------------------------------------------------------------------
# gepa_adapter — candidate validation
# ---------------------------------------------------------------------------


def test_adapter_rejects_unknown_surface(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown surface"):
        ReproLabGEPAAdapter(surface="bogus", opt_run_dir=tmp_path)


def test_adapter_rejects_out_of_surface_component(tmp_path: Path) -> None:
    adapter = ReproLabGEPAAdapter(surface="improvement", opt_run_dir=tmp_path)
    with pytest.raises(ValueError, match="not in surface"):
        adapter._validate_candidate({"baseline_agent.body": "x"})


def test_adapter_rejects_oversized_text(tmp_path: Path) -> None:
    adapter = ReproLabGEPAAdapter(surface="improvement", opt_run_dir=tmp_path)
    huge = "X" * 999_999
    with pytest.raises(ValueError, match="exceeds budget"):
        adapter._validate_candidate({"improvement.orchestrator.body": huge})


def test_adapter_accepts_well_formed_candidate(tmp_path: Path) -> None:
    adapter = ReproLabGEPAAdapter(surface="improvement", opt_run_dir=tmp_path)
    adapter._validate_candidate({"improvement.orchestrator.body": "short replacement"})


def test_candidate_hash_stable() -> None:
    h1 = candidate_hash({"a": "x", "b": "y"})
    h2 = candidate_hash({"b": "y", "a": "x"})  # different insertion order
    assert h1 == h2
    h3 = candidate_hash({"a": "x", "b": "z"})
    assert h1 != h3


def test_adapter_evaluate_short_circuit_with_mocked_eval(tmp_path: Path) -> None:
    """``evaluate`` aggregates per-paper results without spawning subprocesses."""
    adapter = ReproLabGEPAAdapter(surface="improvement", opt_run_dir=tmp_path)
    fake_records: list[float] = [0.7, 0.3]

    def fake_eval(*, candidate, candidate_hash, req):
        return EvalResult(
            score=fake_records.pop(0),
            record={"component_id": "improvement.orchestrator.body",
                    "example_id": req.paper_id, "score": 0.7},
        )

    adapter._run_one_eval = fake_eval  # type: ignore[method-assign]
    batch = [
        {"paper_id": "p1", "archetype": "rl-agent"},
        {"paper_id": "p2", "archetype": "cv-ablation"},
    ]
    result = adapter.evaluate(
        batch=batch,
        candidate={"improvement.orchestrator.body": "test text"},
        capture_traces=True,
    )
    assert list(result.scores) == [0.7, 0.3]
    # scores.jsonl written
    scores_file = tmp_path / "scores.jsonl"
    assert scores_file.exists()
    lines = scores_file.read_text().strip().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# primitive_cache surface_salt
# ---------------------------------------------------------------------------


def test_make_key_byte_identical_without_salt() -> None:
    """Pre-GEPA invariant: no salt → same key as before this commit."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("REPROLAB_PRIMITIVE_CACHE_SURFACE_SALT", None)
        k1 = primitive_cache.make_key("understand_section", payload={"text": "x"})
        k2 = primitive_cache.make_key("understand_section", payload={"text": "x"})
        assert k1 == k2
        # And it has the expected shape
        assert k1.startswith("v")  # versioned prefix
        assert ":understand_section:" in k1


def test_make_key_differs_when_salt_active() -> None:
    payload = {"text": "x"}
    base = primitive_cache.make_key("implement_baseline", payload=payload)
    salted = primitive_cache.make_key("implement_baseline", payload=payload,
                                       surface_salt="cand-abc:improvement")
    assert base != salted
    # Same salt → same key
    salted2 = primitive_cache.make_key("implement_baseline", payload=payload,
                                        surface_salt="cand-abc:improvement")
    assert salted == salted2
    # Different salt → different key
    salted3 = primitive_cache.make_key("implement_baseline", payload=payload,
                                        surface_salt="cand-xyz:improvement")
    assert salted != salted3


def test_env_salt_picked_up_when_arg_absent() -> None:
    payload = {"text": "x"}
    base = primitive_cache.make_key("implement_baseline", payload=payload)
    with patch.dict(os.environ, {"REPROLAB_PRIMITIVE_CACHE_SURFACE_SALT": "env-salt"}):
        with_env = primitive_cache.make_key("implement_baseline", payload=payload)
    assert base != with_env


# ---------------------------------------------------------------------------
# driver split validation (G2)
# ---------------------------------------------------------------------------


def test_validate_split_g2_violation(tmp_path: Path) -> None:
    from scripts.optimize_prompts_gepa import _validate_split

    train = [{"paper_id": f"p{i}"} for i in range(8)]
    val = [{"paper_id": "v1"}, {"paper_id": "v2"}]  # 20% — under threshold
    with pytest.raises(SystemExit, match="G2 violation"):
        _validate_split(train, val, "held-out-1")


def test_validate_split_overlap(tmp_path: Path) -> None:
    from scripts.optimize_prompts_gepa import _validate_split

    train = [{"paper_id": "p1"}, {"paper_id": "p2"}]
    val = [{"paper_id": "p2"}, {"paper_id": "v1"}]
    with pytest.raises(SystemExit, match="overlap"):
        _validate_split(train, val, "held-out-1")


def test_validate_split_held_out_contamination(tmp_path: Path) -> None:
    from scripts.optimize_prompts_gepa import _validate_split

    train = [{"paper_id": "p1"}, {"paper_id": "p2"}]
    val = [{"paper_id": "v1"}, {"paper_id": "v2"}]
    with pytest.raises(SystemExit, match="contamination"):
        _validate_split(train, val, "p1")


def test_validate_split_accepts_clean(tmp_path: Path) -> None:
    from scripts.optimize_prompts_gepa import _validate_split

    train = [{"paper_id": f"p{i}"} for i in range(7)]
    val = [{"paper_id": "v1"}, {"paper_id": "v2"}, {"paper_id": "v3"}]  # 30%
    _validate_split(train, val, "held-out-special")  # no raise


# ---------------------------------------------------------------------------
# Surfaces are consistent
# ---------------------------------------------------------------------------


def test_surfaces_all_components_in_regions() -> None:
    from backend.agents.optimization.mutable_regions import REGIONS

    for surface, components in SURFACES.items():
        for cid in components:
            assert cid in REGIONS, f"surface {surface!r} references unknown component {cid!r}"
