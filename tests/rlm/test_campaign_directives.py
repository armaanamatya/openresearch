"""Tests for backend.agents.rlm.campaign_directives (Unit 6).

Covers spec §9 (PLAN_ATTEMPT directive synthesis, the clean-context
contract) + §8.4 (typed novelty-fingerprint inputs). Hermetic: every test
writes only under ``tmp_path``; no network; no imports from sibling
campaign modules under active parallel edit -- only the frozen
``campaign_policy`` surface (``AttemptEnvelope``, ``NextAttemptPlan``,
``directives_fingerprint``) this unit composes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.agents.rlm.campaign_directives import (
    AttemptDirectives,
    DirectiveContractError,
    synthesize_directives,
)
from backend.agents.rlm.campaign_policy import AttemptEnvelope, NextAttemptPlan, directives_fingerprint

# The exact clean-context forbidden-marker vocabulary from the brief (spec
# §9): any path containing one of these as a substring is transcript/REPL/
# prompt-shaped and must fail the build. ".log" and "attempt_" are BOTH
# listed independently (deliberately redundant) so the campaign's own
# per-attempt driver log (``attempt_<n>.log``, see the campaign file
# layout) is rejected from either direction.
_FORBIDDEN_MARKERS: tuple[str, ...] = (
    "dashboard_events.jsonl",
    "user_messages.jsonl",
    "iterations.jsonl",
    "repl_state",
    ".log",
    "prompt",
    "transcript",
    "attempt_",
)

# Path-carrying inputs the clean-context contract checks, each mapped to a
# basename that keeps the check isolated to the marker (not an incidental
# basename mismatch). Covers all five prior_run_artifacts keys from
# _ALLOWED_ARTIFACT_BASENAMES (leaf_triage, failure_capsules, metrics,
# prior_report, understanding) plus the two other path-carrying top-level
# fields (understanding_ref, run_spec_path).
_KIND_BASENAME: dict[str, str] = {
    "prior_run_artifacts_leaf_triage": "leaf_triage.json",
    "prior_run_artifacts_failure_capsules": "failure_capsules.jsonl",
    "prior_run_artifacts_metrics": "metrics.json",
    "prior_run_artifacts_prior_report": "final_report.json",
    "prior_run_artifacts_understanding": "understanding.json",
    "understanding_ref": "understanding.json",
    "run_spec_path": "run_spec.json",
}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _plan(**overrides: Any) -> NextAttemptPlan:
    defaults: dict[str, Any] = dict(
        lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1
    )
    defaults.update(overrides)
    return NextAttemptPlan(**defaults)


def _envelope(**overrides: Any) -> AttemptEnvelope:
    defaults: dict[str, Any] = dict(
        llm_usd=5.0, gpu_usd=2.0, gpu_hours=1.0, wall_s=3600.0, vm_ceiling_s=5400.0
    )
    defaults.update(overrides)
    return AttemptEnvelope(**defaults)


def _base_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        attempt_n=1,
        project_id="proj_1",
        paper_ref="2605.15155",
        plan=_plan(),
        envelope=_envelope(),
        enforcement={"cli_args": [["--max-usd", "5"]], "env": {}},
        run_spec_path=None,
        understanding_ref=None,
        unresolved_warnings=(),
        prior_run_artifacts={},
        improvement_notes=(),
        memory_hints=(),
        injected_lesson_signatures=(),
        failure_classes=(),
        scope_spec=None,
        target_floor=None,
        out_dir=tmp_path / "campaign",
    )
    defaults.update(overrides)
    return defaults


def _write_leaf_triage(path: Path, plan_entries: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"plan": plan_entries, "facts": {}, "summary": "n weak leaves"}),
        encoding="utf-8",
    )
    return path


def _apply_kind(kwargs: dict[str, Any], kind: str, tainted_path: str) -> None:
    if kind == "prior_run_artifacts_leaf_triage":
        kwargs["prior_run_artifacts"] = {"leaf_triage": tainted_path}
    elif kind == "prior_run_artifacts_failure_capsules":
        kwargs["prior_run_artifacts"] = {"failure_capsules": tainted_path}
    elif kind == "prior_run_artifacts_metrics":
        kwargs["prior_run_artifacts"] = {"metrics": tainted_path}
    elif kind == "prior_run_artifacts_prior_report":
        kwargs["prior_run_artifacts"] = {"prior_report": tainted_path}
    elif kind == "prior_run_artifacts_understanding":
        kwargs["prior_run_artifacts"] = {"understanding": tainted_path}
    elif kind == "understanding_ref":
        kwargs["understanding_ref"] = tainted_path
    elif kind == "run_spec_path":
        kwargs["run_spec_path"] = tainted_path
    else:
        raise AssertionError(f"unhandled kind {kind!r}")


# ---------------------------------------------------------------------------
# Clean-context contract: forbidden transcript-shaped markers (§14 test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", _FORBIDDEN_MARKERS)
@pytest.mark.parametrize("kind", sorted(_KIND_BASENAME))
def test_transcript_path_fails_build(tmp_path: Path, marker: str, kind: str) -> None:
    basename = _KIND_BASENAME[kind]
    tainted_path = f"/runs/proj/{marker}/{basename}"
    kwargs = _base_kwargs(tmp_path)
    _apply_kind(kwargs, kind, tainted_path)

    with pytest.raises(DirectiveContractError):
        synthesize_directives(**kwargs)


def test_real_attempt_log_path_rejected(tmp_path: Path) -> None:
    """The motivating real-world case: the campaign's own per-attempt
    driver log (``attempt_<n>.log``, campaign file layout) must never be
    accepted as a directive input."""
    kwargs = _base_kwargs(
        tmp_path, understanding_ref=str(tmp_path / "campaign" / "attempt_3.log")
    )
    with pytest.raises(DirectiveContractError):
        synthesize_directives(**kwargs)


# ---------------------------------------------------------------------------
# Clean-context contract: unknown key / wrong basename
# ---------------------------------------------------------------------------


def test_unknown_artifact_key_fails(tmp_path: Path) -> None:
    kwargs = _base_kwargs(
        tmp_path, prior_run_artifacts={"weird_key": "/runs/proj/weird_key.json"}
    )
    with pytest.raises(DirectiveContractError, match="weird_key"):
        synthesize_directives(**kwargs)


def test_wrong_basename_for_key_fails(tmp_path: Path) -> None:
    kwargs = _base_kwargs(
        tmp_path,
        prior_run_artifacts={"leaf_triage": "/runs/proj/rlm_state/wrong_name.json"},
    )
    with pytest.raises(DirectiveContractError, match="basename"):
        synthesize_directives(**kwargs)


# ---------------------------------------------------------------------------
# Leaf repair plan reduction
# ---------------------------------------------------------------------------


def test_leaf_plan_reduced_drops_justification(tmp_path: Path) -> None:
    leaf_path = _write_leaf_triage(
        tmp_path / "rlm_state" / "leaf_triage.json",
        [
            {
                "leaf_id": "leaf_1",
                "score": 0.2,
                "repair_class": "render_artifact",
                "cost": "none",
                "directive": "render the figure from disk data",
                "justification": "the grader said the figure was missing entirely",
            }
        ],
    )

    directives = synthesize_directives(
        **_base_kwargs(tmp_path, prior_run_artifacts={"leaf_triage": str(leaf_path)})
    )

    assert directives.leaf_repair_plan == {
        "plan": [
            {
                "leaf_id": "leaf_1",
                "repair_class": "render_artifact",
                "cost": "none",
                "directive": "render the figure from disk data",
            }
        ]
    }
    persisted_text = (tmp_path / "campaign" / "directives" / "1.json").read_text(encoding="utf-8")
    assert "justification" not in persisted_text
    assert "grader said" not in persisted_text


def test_corrupt_leaf_triage_fails_build(tmp_path: Path) -> None:
    leaf_path = tmp_path / "rlm_state" / "leaf_triage.json"
    leaf_path.parent.mkdir(parents=True)
    leaf_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(DirectiveContractError):
        synthesize_directives(
            **_base_kwargs(tmp_path, prior_run_artifacts={"leaf_triage": str(leaf_path)})
        )


@pytest.mark.parametrize(
    "content",
    [
        "{not valid json",
        json.dumps({"facts": {}, "summary": "no plan key"}),
        json.dumps({"plan": "not-a-list"}),
        json.dumps({"plan": [{"leaf_id": "x"}]}),
        json.dumps({"plan": [["not", "an", "object"]]}),
        json.dumps(
            {
                "plan": [
                    {
                        "leaf_id": "leaf_1",
                        "repair_class": ["render_artifact", "protocol_gap"],
                        "cost": "none",
                        "directive": "d",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "plan": [
                    {"leaf_id": 1, "repair_class": "render_artifact", "cost": "none", "directive": "d"}
                ]
            }
        ),
    ],
    ids=[
        "bad-json",
        "missing-plan-key",
        "plan-not-list",
        "entry-missing-fields",
        "entry-not-object",
        "repair-class-not-str",
        "leaf-id-not-str",
    ],
)
def test_corrupt_leaf_triage_shapes_fail_build(tmp_path: Path, content: str) -> None:
    leaf_path = tmp_path / "rlm_state" / "leaf_triage.json"
    leaf_path.parent.mkdir(parents=True)
    leaf_path.write_text(content, encoding="utf-8")

    with pytest.raises(DirectiveContractError):
        synthesize_directives(
            **_base_kwargs(tmp_path, prior_run_artifacts={"leaf_triage": str(leaf_path)})
        )


def test_missing_leaf_triage_file_fails_build(tmp_path: Path) -> None:
    missing_path = tmp_path / "rlm_state" / "leaf_triage.json"  # never written
    with pytest.raises(DirectiveContractError):
        synthesize_directives(
            **_base_kwargs(tmp_path, prior_run_artifacts={"leaf_triage": str(missing_path)})
        )


# ---------------------------------------------------------------------------
# Fingerprint (F10): typed kinds only, never prose
# ---------------------------------------------------------------------------


def test_fingerprint_uses_typed_kinds_not_prose(tmp_path: Path) -> None:
    path_a = _write_leaf_triage(
        tmp_path / "a" / "leaf_triage.json",
        [
            {
                "leaf_id": "leaf_1",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "render the missing whitening step",
                "justification": "the grader flagged whitening as absent",
            }
        ],
    )
    path_b = _write_leaf_triage(
        tmp_path / "b" / "leaf_triage.json",
        [
            {
                "leaf_id": "leaf_1",
                "score": 0.9,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "a completely different directive string entirely",
                "justification": "a totally unrelated justification blob",
            }
        ],
    )

    directives_a = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            out_dir=tmp_path / "campaign_a",
            prior_run_artifacts={"leaf_triage": str(path_a)},
        )
    )
    directives_b = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            out_dir=tmp_path / "campaign_b",
            prior_run_artifacts={"leaf_triage": str(path_b)},
        )
    )

    assert directives_a.fingerprint == directives_b.fingerprint
    assert directives_a.leaf_repair_plan != directives_b.leaf_repair_plan


def test_fingerprint_changes_with_lineage_rung_kinds_classes(tmp_path: Path) -> None:
    leaf_path_protocol = _write_leaf_triage(
        tmp_path / "lt1" / "leaf_triage.json",
        [
            {
                "leaf_id": "l1",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "d",
                "justification": "j",
            }
        ],
    )
    leaf_path_render = _write_leaf_triage(
        tmp_path / "lt2" / "leaf_triage.json",
        [
            {
                "leaf_id": "l1",
                "score": 0.1,
                "repair_class": "render_artifact",
                "cost": "none",
                "directive": "d",
                "justification": "j",
            }
        ],
    )

    base_plan = _plan(lineage="champion", seed_attempt_n=2, scope_rung=1)
    base_kwargs = _base_kwargs(
        tmp_path,
        out_dir=tmp_path / "c_base",
        plan=base_plan,
        prior_run_artifacts={"leaf_triage": str(leaf_path_protocol)},
        failure_classes=["cuda_oom"],
    )
    base_fp = synthesize_directives(**base_kwargs).fingerprint

    lineage_kwargs = dict(base_kwargs)
    lineage_kwargs["out_dir"] = tmp_path / "c_lineage"
    lineage_kwargs["plan"] = _plan(lineage="runner_up", seed_attempt_n=2, scope_rung=1)
    lineage_fp = synthesize_directives(**lineage_kwargs).fingerprint
    assert lineage_fp != base_fp

    rung_kwargs = dict(base_kwargs)
    rung_kwargs["out_dir"] = tmp_path / "c_rung"
    rung_kwargs["plan"] = _plan(lineage="champion", seed_attempt_n=2, scope_rung=2)
    rung_fp = synthesize_directives(**rung_kwargs).fingerprint
    assert rung_fp != base_fp

    classes_kwargs = dict(base_kwargs)
    classes_kwargs["out_dir"] = tmp_path / "c_classes"
    classes_kwargs["failure_classes"] = ["cuda_oom", "dockerfile_invalid"]
    classes_fp = synthesize_directives(**classes_kwargs).fingerprint
    assert classes_fp != base_fp

    kinds_kwargs = dict(base_kwargs)
    kinds_kwargs["out_dir"] = tmp_path / "c_kinds"
    kinds_kwargs["prior_run_artifacts"] = {"leaf_triage": str(leaf_path_render)}
    kinds_fp = synthesize_directives(**kinds_kwargs).fingerprint
    assert kinds_fp != base_fp

    seed_kwargs = dict(base_kwargs)
    seed_kwargs["out_dir"] = tmp_path / "c_seed"
    seed_kwargs["plan"] = _plan(lineage="champion", seed_attempt_n=5, scope_rung=1)
    seed_fp = synthesize_directives(**seed_kwargs).fingerprint
    assert seed_fp != base_fp


# ---------------------------------------------------------------------------
# Content caps
# ---------------------------------------------------------------------------


def test_memory_hint_caps_enforced(tmp_path: Path) -> None:
    hints = [f"hint-{i}" for i in range(6)]
    hints[0] = "x" * 250

    directives = synthesize_directives(**_base_kwargs(tmp_path, memory_hints=hints))

    assert len(directives.memory_hints) == 5
    assert directives.memory_hints[0] == "x" * 200
    assert list(directives.memory_hints[1:]) == ["hint-1", "hint-2", "hint-3", "hint-4"]


def test_improvement_note_caps_enforced(tmp_path: Path) -> None:
    notes = [f"note-{i}" for i in range(10)]
    notes[0] = "y" * 500

    directives = synthesize_directives(**_base_kwargs(tmp_path, improvement_notes=notes))

    assert len(directives.improvement_notes) == 8
    assert directives.improvement_notes[0] == "y" * 400
    assert list(directives.improvement_notes[1:]) == [f"note-{i}" for i in range(1, 8)]


# ---------------------------------------------------------------------------
# extra_guidance assembly + cap
# ---------------------------------------------------------------------------


def test_extra_guidance_sections_and_cap(tmp_path: Path) -> None:
    minimal = synthesize_directives(**_base_kwargs(tmp_path, out_dir=tmp_path / "c_min"))
    assert minimal.extra_guidance == "[campaign] Attempt 1 (fresh, scope rung 0)."
    for header in (
        "[unresolved-understanding]",
        "[leaf-repairs]",
        "[prior-improvements]",
        "[memory]",
        "[scope]",
        "[target-floor]",
    ):
        assert header not in minimal.extra_guidance

    leaf_path = _write_leaf_triage(
        tmp_path / "rlm_state" / "leaf_triage.json",
        [
            {
                "leaf_id": "l1",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "fix the dropout rate",
                "justification": "j",
            }
        ],
    )
    full = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            out_dir=tmp_path / "c_full",
            unresolved_warnings=["field X could not be resolved"],
            prior_run_artifacts={"leaf_triage": str(leaf_path)},
            improvement_notes=["try a lower lr"],
            memory_hints=["past infra failure: dataset mirror flaky"],
            scope_spec="models=[a,b]",
            target_floor=0.62,
        )
    )
    headers_in_order = [
        "[campaign]",
        "[unresolved-understanding]",
        "[leaf-repairs]",
        "[prior-improvements]",
        "[memory]",
        "[scope]",
        "[target-floor]",
    ]
    positions = [full.extra_guidance.index(h) for h in headers_in_order]
    assert positions == sorted(positions)
    assert "l1 [protocol_gap] fix the dropout rate" in full.extra_guidance
    assert "field X could not be resolved" in full.extra_guidance
    assert "try a lower lr" in full.extra_guidance
    assert "past infra failure: dataset mirror flaky" in full.extra_guidance
    assert "models=[a,b]" in full.extra_guidance
    assert "0.62" in full.extra_guidance

    big_leaf_path = _write_leaf_triage(
        tmp_path / "rlm_state_big" / "leaf_triage.json",
        [
            {
                "leaf_id": f"leaf_{i}",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "d" * 300,
                "justification": "j",
            }
            for i in range(8)
        ],
    )
    huge = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            out_dir=tmp_path / "c_huge",
            unresolved_warnings=["w" * 300 for _ in range(5)],
            prior_run_artifacts={"leaf_triage": str(big_leaf_path)},
            improvement_notes=["n" * 400 for _ in range(8)],
            memory_hints=["m" * 200 for _ in range(5)],
            scope_spec="s" * 300,
            target_floor=0.5,
        )
    )
    assert len(huge.extra_guidance) <= 4000
    assert huge.extra_guidance.endswith("[truncated]")


# ---------------------------------------------------------------------------
# Persist round-trip + atomicity
# ---------------------------------------------------------------------------


def test_persist_roundtrip_and_atomicity(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path, out_dir=tmp_path / "campaign", target_floor=0.5, scope_spec="models=[a]"
        )
    )

    path = tmp_path / "campaign" / "directives" / "1.json"
    assert path.is_file()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == directives.to_dict()
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    alt_path = tmp_path / "alt" / "directives.json"
    directives.persist(alt_path)
    assert json.loads(alt_path.read_text(encoding="utf-8")) == directives.to_dict()
    assert not alt_path.with_suffix(alt_path.suffix + ".tmp").exists()


def test_scheduler_defaults_omit_durable_keys_but_explicit_metadata_roundtrips(tmp_path: Path) -> None:
    defaults = synthesize_directives(**_base_kwargs(tmp_path))
    default_payload = defaults.to_dict()
    assert "branch_type" not in default_payload
    assert "is_safety_bracket" not in default_payload

    safety = synthesize_directives(
        **_base_kwargs(tmp_path, attempt_n=2, plan=_plan(is_safety_bracket=True))
    )
    assert safety.to_dict()["is_safety_bracket"] is True
    assert "branch_type" not in safety.to_dict()  # faithful remains legacy-default implicit

    typed = synthesize_directives(
        **_base_kwargs(tmp_path, attempt_n=3, plan=_plan(branch_type="ambiguity"))
    )
    assert typed.to_dict()["branch_type"] == "ambiguity"
    assert typed.fingerprint != defaults.fingerprint  # F10 includes branch type

    with pytest.raises(DirectiveContractError):
        AttemptDirectives(
            **{
                **defaults.__dict__,
                "branch_type": "free-text",  # type: ignore[arg-type]
            }
        )
    with pytest.raises(DirectiveContractError):
        AttemptDirectives(
            **{
                **defaults.__dict__,
                "branch_type": "discovery",
                "is_safety_bracket": True,
            }
        )


# ---------------------------------------------------------------------------
# Fresh attempt 1, no prior artifacts
# ---------------------------------------------------------------------------


def test_empty_prior_artifacts_fresh_attempt_ok(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            attempt_n=1,
            plan=_plan(lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0),
        )
    )

    assert directives.leaf_repair_plan is None
    assert directives.failure_capsules_ref is None
    assert directives.prior_evidence_ref is None
    assert directives.understanding_ref is None
    assert directives.improvement_notes == ()
    assert directives.memory_hints == ()
    assert directives.seed_lineage == "fresh"
    assert directives.seed_pointer is None
    assert directives.extra_guidance == "[campaign] Attempt 1 (fresh, scope rung 0)."
    assert isinstance(directives.fingerprint, str) and len(directives.fingerprint) == 64


# ---------------------------------------------------------------------------
# Field-mapping edge cases (not in the §14 catalog by name, but load-bearing
# design decisions this unit had to make against the frozen AttemptDirectives
# shape -- locked in explicitly so a regression is caught)
# ---------------------------------------------------------------------------


def test_prior_evidence_ref_prefers_metrics_over_prior_report(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            prior_run_artifacts={
                "metrics": "/runs/proj/code/metrics.json",
                "prior_report": "/runs/proj/final_report.json",
            },
        )
    )
    assert directives.prior_evidence_ref == "/runs/proj/code/metrics.json"


def test_prior_evidence_ref_falls_back_to_prior_report(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path, prior_run_artifacts={"prior_report": "/runs/proj/final_report.json"}
        )
    )
    assert directives.prior_evidence_ref == "/runs/proj/final_report.json"


def test_failure_capsules_ref_passthrough(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            prior_run_artifacts={
                "failure_capsules": "/runs/proj/rlm_state/failure_capsules.jsonl"
            },
        )
    )
    assert directives.failure_capsules_ref == "/runs/proj/rlm_state/failure_capsules.jsonl"


def test_understanding_key_in_prior_artifacts_validated_not_overriding(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            understanding_ref="/runs/proj/campaign/understanding.json",
            prior_run_artifacts={"understanding": "/runs/proj/campaign/understanding.json"},
        )
    )
    assert directives.understanding_ref == "/runs/proj/campaign/understanding.json"


def test_injected_lesson_signatures_passthrough(tmp_path: Path) -> None:
    sigs = ("sig-a", "sig-b", "sig-c")
    directives = synthesize_directives(**_base_kwargs(tmp_path, injected_lesson_signatures=sigs))
    assert directives.injected_lesson_signatures == sigs


def test_enforcement_and_envelope_stored_verbatim(tmp_path: Path) -> None:
    enforcement = {"cli_args": [["--max-usd", "5"], ["--max-wall-clock", "3600"]], "env": {"X": "1"}}
    envelope = _envelope(llm_usd=3.0)
    directives = synthesize_directives(
        **_base_kwargs(tmp_path, enforcement=enforcement, envelope=envelope)
    )
    assert dict(directives.enforcement) == enforcement
    assert directives.envelope == envelope


def test_returns_frozen_attempt_directives_instance(tmp_path: Path) -> None:
    directives = synthesize_directives(**_base_kwargs(tmp_path))
    assert isinstance(directives, AttemptDirectives)
    with pytest.raises(Exception):  # noqa: B017, PT011 -- frozen dataclass: any mutation attempt raises
        directives.attempt_n = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# repair-mode directive (MLE-STAR-style localized refinement)
# ---------------------------------------------------------------------------


def test_repair_mode_section_for_seeded_lineage_with_leaf_repairs(tmp_path: Path) -> None:
    leaf_path = _write_leaf_triage(
        tmp_path / "rlm_state" / "leaf_triage.json",
        [
            {
                "leaf_id": "l1",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "fix the dropout rate",
                "justification": "j",
            }
        ],
    )
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            plan=_plan(lineage="champion", seed_attempt_n=2, scope_rung=1),
            prior_run_artifacts={"leaf_triage": str(leaf_path)},
        )
    )
    assert "[repair-mode]" in directives.extra_guidance
    assert "code/_best_attempt/" in directives.extra_guidance
    assert (
        directives.extra_guidance.index("[campaign]")
        < directives.extra_guidance.index("[repair-mode]")
        < directives.extra_guidance.index("[leaf-repairs]")
    )


def test_repair_mode_section_for_seeded_lineage_with_memory_hints_only(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            plan=_plan(lineage="runner_up", seed_attempt_n=1, scope_rung=0),
            memory_hints=["past infra failure: dataset mirror flaky"],
        )
    )
    assert "[repair-mode]" in directives.extra_guidance
    assert (
        directives.extra_guidance.index("[repair-mode]")
        < directives.extra_guidance.index("[memory]")
    )


def test_repair_mode_section_absent_for_fresh_lineage(tmp_path: Path) -> None:
    leaf_path = _write_leaf_triage(
        tmp_path / "rlm_state" / "leaf_triage.json",
        [
            {
                "leaf_id": "l1",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "fix the dropout rate",
                "justification": "j",
            }
        ],
    )
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            plan=_plan(lineage="fresh", seed_attempt_n=None, scope_rung=0),
            prior_run_artifacts={"leaf_triage": str(leaf_path)},
            memory_hints=["hint"],
        )
    )
    assert "[repair-mode]" not in directives.extra_guidance


def test_repair_mode_section_absent_when_seeded_but_no_leaf_or_memory(tmp_path: Path) -> None:
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            plan=_plan(lineage="champion", seed_attempt_n=2, scope_rung=1),
        )
    )
    assert "[repair-mode]" not in directives.extra_guidance


def test_repair_mode_section_counts_toward_cap(tmp_path: Path) -> None:
    big_leaf_path = _write_leaf_triage(
        tmp_path / "rlm_state_big2" / "leaf_triage.json",
        [
            {
                "leaf_id": f"leaf_{i}",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "d" * 300,
                "justification": "j",
            }
            for i in range(8)
        ],
    )
    huge = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            out_dir=tmp_path / "c_huge_repair",
            plan=_plan(lineage="champion", seed_attempt_n=2, scope_rung=1),
            unresolved_warnings=["w" * 300 for _ in range(5)],
            prior_run_artifacts={"leaf_triage": str(big_leaf_path)},
            improvement_notes=["n" * 400 for _ in range(8)],
            memory_hints=["m" * 200 for _ in range(5)],
            scope_spec="s" * 300,
            target_floor=0.5,
        )
    )
    assert len(huge.extra_guidance) <= 4000
    assert huge.extra_guidance.endswith("[truncated]")
    assert "[repair-mode]" in huge.extra_guidance


def test_fingerprint_unaffected_by_repair_mode_section(tmp_path: Path) -> None:
    """The repair-mode prose lives only in extra_guidance; the fingerprint is
    computed from typed inputs only (spec F10) via
    ``campaign_policy.directives_fingerprint`` -- it must match a direct call
    with the same typed inputs regardless of whether the repair-mode section
    fires in extra_guidance."""
    leaf_path = _write_leaf_triage(
        tmp_path / "rlm_state" / "leaf_triage.json",
        [
            {
                "leaf_id": "l1",
                "score": 0.1,
                "repair_class": "protocol_gap",
                "cost": "targeted_rerun",
                "directive": "d",
                "justification": "j",
            }
        ],
    )
    plan = _plan(lineage="champion", seed_attempt_n=2, scope_rung=1)
    envelope = _envelope()
    directives = synthesize_directives(
        **_base_kwargs(
            tmp_path,
            plan=plan,
            envelope=envelope,
            prior_run_artifacts={"leaf_triage": str(leaf_path)},
            failure_classes=["cuda_oom"],
        )
    )
    assert "[repair-mode]" in directives.extra_guidance

    expected_fp = directives_fingerprint(
        seed_lineage=f"{plan.lineage}:{plan.seed_attempt_n or 0}",
        scope_rung=plan.scope_rung,
        repair_action_kinds=["protocol_gap"],
        failure_classes=["cuda_oom"],
        envelope=envelope.to_dict(),
    )
    assert directives.fingerprint == expected_fp
