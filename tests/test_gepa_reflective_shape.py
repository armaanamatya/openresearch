"""Regression tests for the gepa-canonical reflective dataset shape.

Gepa's reflection LM expects each example as exactly three keys:
``{"Inputs", "Generated Outputs", "Feedback"}``. Returning our raw §4.4
record (with keys like ``component_id``, ``execution_trace`` etc.) would
feed the LM unstructured trace dumps and produce nonsense mutations.

These tests pin the shape so a future refactor cannot silently break the
contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.optimization.gepa_adapter import (
    EvalResult,
    ReproLabGEPAAdapter,
    _to_gepa_reflective_record,
)


_RAW_RECORD = {
    "component_id": "improvement.orchestrator.body",
    "example_id": "2605.15155",
    "paper_archetype": "rl-agent",
    "score": 0.62,
    "hermes_clamped_score": 0.62,
    "input": {
        "rubric_overall_before": 0.50,
        "rubric_areas_before": {"correctness": 0.5, "completeness": 0.4},
        "weak_leaves_before": [
            {"name": "leaf_a", "score": 0.0, "rationale": "missing artifact"},
        ],
        "current_results_digest": "mean_reward=475",
    },
    "candidate_output": {"selected_hypothesis": {"path_id": "path_1"}},
    "execution_trace": {
        "rubric_overall_after": 0.62,
        "rubric_delta_areas": {"correctness": 0.12, "completeness": 0.05},
        "weak_leaves_after": [],
        "run_experiment_success": True,
        "repair_summaries": ["fixed missing artifact"],
        "forced_iteration_warnings": 0,
        "blanket_decline_count": 1,
        "hermes": {
            "status": "grounded",
            "unsupported_claims": [],
            "evidence_refs_snippets": [],
        },
    },
}


def test_reflective_record_has_exactly_three_canonical_keys() -> None:
    rec = _to_gepa_reflective_record(_RAW_RECORD, candidate_text="some prompt text")
    assert set(rec.keys()) == {"Inputs", "Generated Outputs", "Feedback"}


def test_reflective_record_inputs_carry_situation() -> None:
    rec = _to_gepa_reflective_record(_RAW_RECORD, candidate_text="prompt")
    inp = rec["Inputs"]
    # Inputs is a dict — structured fields the reflection LM can parse cleanly.
    assert isinstance(inp, dict)
    assert inp["paper_archetype"] == "rl-agent"
    assert inp["rubric_overall_before"] == 0.5
    assert any(leaf["name"] == "leaf_a" for leaf in inp["weak_leaves_before"])
    assert "mean_reward=475" in inp["prior_results_digest"]


def test_reflective_record_outputs_carry_candidate_and_metrics() -> None:
    rec = _to_gepa_reflective_record(_RAW_RECORD, candidate_text="CANDIDATE_PROMPT_X")
    out = rec["Generated Outputs"]
    # Generated Outputs is a dict — gepa canonical adapter pattern.
    assert isinstance(out, dict)
    assert out["candidate_prompt_text"] == "CANDIDATE_PROMPT_X"
    assert out["candidate_truncated"] is False
    assert out["candidate_metrics"]["selected_hypothesis"]["path_id"] == "path_1"
    assert out["rubric_overall_after"] == 0.62
    assert out["run_experiment_success"] is True


def test_reflective_record_feedback_carries_outcome_signal() -> None:
    rec = _to_gepa_reflective_record(_RAW_RECORD, candidate_text="prompt")
    fb = rec["Feedback"]
    # Feedback stays as a natural-language instruction (per canonical adapters).
    assert isinstance(fb, str)
    assert "0.620" in fb                    # Hermes-clamped score
    assert "grounded" in fb                 # Hermes status
    assert "fixed missing artifact" in fb   # repair summary
    assert "Improve the prompt" in fb       # actionable instruction


def test_reflective_record_truncates_overlong_candidate() -> None:
    long_text = "X" * 5000
    rec = _to_gepa_reflective_record(_RAW_RECORD, candidate_text=long_text)
    out = rec["Generated Outputs"]
    assert len(out["candidate_prompt_text"]) == 1500
    assert out["candidate_truncated"] is True


def test_make_reflective_dataset_returns_gepa_shape(tmp_path: Path) -> None:
    """End-to-end: adapter.evaluate then make_reflective_dataset → 3-key records."""
    adapter = ReproLabGEPAAdapter(surface="improvement", opt_run_dir=tmp_path)

    def fake_eval(*, candidate, candidate_hash, req):
        return EvalResult(score=0.62, record=dict(_RAW_RECORD, example_id=req.paper_id))

    adapter._run_one_eval = fake_eval  # type: ignore[method-assign]
    eb = adapter.evaluate(
        batch=[{"paper_id": "p1", "archetype": "rl-agent"}],
        candidate={"improvement.orchestrator.body": "prompt"},
        capture_traces=True,
    )
    rd = adapter.make_reflective_dataset(
        candidate={"improvement.orchestrator.body": "prompt"},
        eval_batch=eb,
        components_to_update=["improvement.orchestrator.body"],
    )
    assert "improvement.orchestrator.body" in rd
    examples = rd["improvement.orchestrator.body"]
    assert len(examples) == 1
    assert set(examples[0].keys()) == {"Inputs", "Generated Outputs", "Feedback"}
    # Phase-2 dict-shape contract — Inputs + Generated Outputs are dicts.
    assert isinstance(examples[0]["Inputs"], dict)
    assert isinstance(examples[0]["Generated Outputs"], dict)
    assert isinstance(examples[0]["Feedback"], str)


def test_make_reflective_dataset_persists_raw_records_for_audit(tmp_path: Path) -> None:
    """G6 audit trail: raw §4.4 records must land in reflective_dataset.jsonl."""
    adapter = ReproLabGEPAAdapter(surface="improvement", opt_run_dir=tmp_path)

    def fake_eval(*, candidate, candidate_hash, req):
        return EvalResult(score=0.62, record=dict(_RAW_RECORD, example_id=req.paper_id))

    adapter._run_one_eval = fake_eval  # type: ignore[method-assign]
    eb = adapter.evaluate(
        batch=[{"paper_id": "p1", "archetype": "rl-agent"}],
        candidate={"improvement.orchestrator.body": "prompt"},
        capture_traces=True,
    )
    adapter.make_reflective_dataset(
        candidate={"improvement.orchestrator.body": "prompt"},
        eval_batch=eb,
        components_to_update=["improvement.orchestrator.body"],
    )
    audit = tmp_path / "reflective_dataset.jsonl"
    assert audit.exists()
    lines = audit.read_text().strip().splitlines()
    assert len(lines) == 1
    raw = json.loads(lines[0])
    # Raw record preserves all §4.4 keys for human inspection (NOT the 3-key shape).
    assert "execution_trace" in raw
    assert "input" in raw
    assert raw["component_id"] == "improvement.orchestrator.body"
