"""Tests for backend.agents.rlm.campaign_report (Unit 8).

Hermetic: every test writes only under ``tmp_path``. No network, no imports
from sibling campaign modules (those are implemented in parallel units) —
this module consumes plain dicts shaped like the ledger row vocabulary
documented in the shared design context.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backend.agents.rlm import campaign_report

_FIXED_NOW = 1_750_000_000.0


def _now() -> float:
    return _FIXED_NOW


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base = {
        "paper_ref": "2601.00000",
        "project_id": "proj_default",
        "mode": "unattended",
        "driver": "live",
        "terminal": None,
        "budget": {"llm_usd": 10.0, "gpu_usd": 10.0, "gpu_hours": 2.0, "wall_s": 7200.0},
        "spent": {"llm_usd": 1.0, "gpu_usd": 1.0, "gpu_hours": 0.5, "wall_s": 1800.0},
    }
    base.update(overrides)
    return base


def _final_report(**overrides) -> dict:
    base = {
        "score": 0.5,
        "target": 0.6,
        "meets_target": False,
        "implementation_verdict": "partial",
        "replication_verdict": "inconclusive",
        "verdict": "partial",
        "stop_reason": None,
        "exclusions": [],
        "path": "final_report.json",
    }
    base.update(overrides)
    return base


def _assessment(attempt_n: int = 1, **overrides) -> dict:
    base = {
        "attempt_n": attempt_n,
        "driver": "live",
        "project_id": "proj_default",
        "directives_sha256": "sha_abc",
        "final_report": _final_report(),
        "evidence_predicates": {"backed_by_ledger": True, "provenance_present": False},
        "guard_flags": {},
        "validator": {"status": "clean", "fingerprint": "fp1", "fresh": True},
        "leaf_pass_count": 3,
        "leaf_vector_ref": None,
        "failure_class": None,
        "failure_signature": None,
        "failure_scope": None,
        "cost": {"llm_usd": 1.0, "gpu_usd": 2.0, "gpu_hours": 0.5, "wall_s": 1800.0},
        "rubric_sha256_ok": True,
        "hard_quarantined": False,
        "soft_quarantined": False,
        "quarantine_reasons": [],
    }
    base.update(overrides)
    return base


def _launched_row(attempt_n: int, driver: str = "live", **extra) -> dict:
    row = {
        "attempt_n": attempt_n,
        "status": "launched",
        "directives_sha256": "sha_abc",
        "envelope": {},
        "driver": driver,
        "project_id": "proj_default",
        "launched_at": 1_749_000_000.0,
    }
    row.update(extra)
    return row


def _assessed_row(attempt_n: int, assessment: dict, **extra) -> dict:
    row = {
        "attempt_n": attempt_n,
        "status": "assessed",
        "assessment": assessment,
        "assessed_at": 1_749_100_000.0,
    }
    row.update(extra)
    return row


def _decided_row(attempt_n: int, decision: dict, **extra) -> dict:
    row = {"attempt_n": attempt_n, "status": "decided", "decision": decision}
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# write_plan_only_report
# ---------------------------------------------------------------------------


def test_plan_only_writer_writes_verdict_plan_only(tmp_path):
    state = _state(paper_ref="2601.00001", project_id="proj_abc", terminal=None, spent={"llm_usd": 0.0})

    json_path, md_path = campaign_report.write_plan_only_report(
        tmp_path,
        stop_reason="infeasible:asset_gap",
        what_would_unblock=["dataset X access", "GPU quota increase"],
        state=state,
        now=_now,
    )

    assert json_path == tmp_path / "final_report.json"
    assert md_path == tmp_path / "final_report.md"
    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["verdict"] == "plan_only"
    assert data["stop_reason"] == "infeasible:asset_gap"
    assert data["what_would_unblock"] == ["dataset X access", "GPU quota increase"]
    assert data["paper"]["ref"] == "2601.00001"
    assert data["campaign"]["project_id"] == "proj_abc"
    assert data["campaign"]["terminal"] is None
    assert data["campaign"]["spent"] == {"llm_usd": 0.0}
    # generated_at must round-trip as a real ISO-8601 timestamp.
    datetime.fromisoformat(data["generated_at"])

    md_text = md_path.read_text(encoding="utf-8")
    assert "plan_only" in md_text
    assert "dataset X access" in md_text
    assert "infeasible:asset_gap" in md_text


def test_plan_only_never_clobbers_existing_report(tmp_path):
    sentinel = '{"sentinel": true, "verdict": "reproduced"}'
    json_path = tmp_path / "final_report.json"
    json_path.write_text(sentinel, encoding="utf-8")
    md_path = tmp_path / "final_report.md"
    assert not md_path.exists()

    result_json, result_md = campaign_report.write_plan_only_report(
        tmp_path,
        stop_reason="infeasible:should_not_apply",
        what_would_unblock=[],
        state=_state(),
        now=_now,
    )

    # Paths are always returned, even when nothing was written.
    assert result_json == json_path
    assert result_md == md_path
    # The real attempt's report survives byte-identically.
    assert json_path.read_text(encoding="utf-8") == sentinel
    # Neither file is touched -- the run's own report+md stand.
    assert not md_path.exists()


def test_plan_only_report_survives_leaderboard_extractor(tmp_path):
    from backend.services.runs.report_resolution import extract_scores

    json_path, _ = campaign_report.write_plan_only_report(
        tmp_path,
        stop_reason="infeasible:no_repo",
        what_would_unblock=["operator review"],
        state=_state(),
        now=_now,
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    overall, adjusted = extract_scores(data)
    assert overall is None
    assert adjusted is None


# ---------------------------------------------------------------------------
# write_campaign_report
# ---------------------------------------------------------------------------


def test_report_renders_zero_attempts_infeasible(tmp_path):
    state = _state(
        terminal={"kind": "INFEASIBLE", "rule": "understand_gate_block", "stop_reason": "asset_gap"},
    )

    path = campaign_report.write_campaign_report(tmp_path, state=state, rows=[], now=_now)

    assert path == tmp_path / "campaign_report.md"
    text = path.read_text(encoding="utf-8")
    assert "INFEASIBLE" in text
    assert "understand_gate_block" in text
    assert "asset_gap" in text
    assert "no attempts recorded" in text
    assert "none (no guard-clean attempt)" in text


def test_report_renders_attempt_table_with_superseding_rows(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(1, _assessment(1, final_report=_final_report(score=0.1, target=0.6))),
        _assessed_row(1, _assessment(1, final_report=_final_report(score=0.55, target=0.6))),
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    text = path.read_text(encoding="utf-8")

    assert "0.550" in text
    assert "0.100" not in text


def test_report_marks_quarantined_attempts(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(
            1,
            _assessment(1, hard_quarantined=True, quarantine_reasons=["rubric_hash_mismatch"]),
        ),
        _launched_row(2),
        _assessed_row(
            2,
            _assessment(2, soft_quarantined=True, quarantine_reasons=["validator_stale"]),
        ),
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    text = path.read_text(encoding="utf-8")

    assert "hard-quarantined: rubric_hash_mismatch" in text
    assert "soft-quarantined: validator_stale" in text


def test_report_evidence_trajectory_improvement_flags(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(
            1,
            _assessment(1, evidence_predicates={"a": True, "b": False}, leaf_pass_count=2),
        ),
        _launched_row(2),
        _assessed_row(
            2,
            _assessment(2, evidence_predicates={"a": True, "b": True}, leaf_pass_count=5),
        ),
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    text = path.read_text(encoding="utf-8")

    trajectory = text.split("## Evidence Trajectory", 1)[1].split("##", 1)[0]
    line1 = next(line for line in trajectory.splitlines() if "attempt 1:" in line)
    line2 = next(line for line in trajectory.splitlines() if "attempt 2:" in line)

    assert not line1.rstrip().endswith("+")
    assert line2.rstrip().endswith("+")


def test_report_renders_launched_but_unassessed_attempt(tmp_path):
    rows = [_launched_row(1, driver="live")]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    text = path.read_text(encoding="utf-8")

    assert "in-flight/unassessed" in text


def _attempts_section(text: str) -> list[str]:
    """Return the ``## Attempts`` section's non-blank lines (isolated from
    the ``## Budget`` table, which also contains pipe-delimited rows).
    """
    section = text.split("## Attempts", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if line.strip()]


def test_report_attempt_table_rows_match_header_column_count(tmp_path):
    """Every rendered attempt row (assessed, unassessed, and the
    zero-attempts placeholder) must have the same pipe-delimited column
    count as the header -- a malformed row would silently break the table.
    """
    rows = [
        _launched_row(1),  # unassessed row
        _launched_row(2),
        _assessed_row(2, _assessment(2)),  # assessed row
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    section_lines = _attempts_section(path.read_text(encoding="utf-8"))

    header_line = next(line for line in section_lines if line.startswith("| n |"))
    expected_cols = header_line.count("|")
    data_rows = [
        line
        for line in section_lines
        if line.startswith("|") and line != header_line and not set(line) <= {"|", "-"}
    ]
    assert len(data_rows) == 2, data_rows
    for row in data_rows:
        assert row.count("|") == expected_cols, row

    # Zero-attempts placeholder row also matches the header width.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_path = campaign_report.write_campaign_report(empty_dir, state=_state(), rows=[], now=_now)
    empty_section = _attempts_section(empty_path.read_text(encoding="utf-8"))
    empty_header = next(line for line in empty_section if line.startswith("| n |"))
    empty_data = next(line for line in empty_section if "no attempts recorded" in line)
    assert empty_data.count("|") == empty_header.count("|")


def test_report_deterministic_bytes(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(1, _assessment(1)),
        _decided_row(1, {"kind": "CONTINUE", "rule": "r1", "stop_reason": None, "next_plan": None}),
    ]
    state = _state()

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    path_a = campaign_report.write_campaign_report(dir_a, state=state, rows=rows, now=_now)
    path_b = campaign_report.write_campaign_report(dir_b, state=state, rows=rows, now=_now)

    assert path_a.read_bytes() == path_b.read_bytes()


def test_report_budget_table_math(tmp_path):
    state = _state(
        budget={"llm_usd": 10.0, "gpu_usd": 20.0, "gpu_hours": 5.0, "wall_s": 36000.0},
        spent={"llm_usd": 1.234, "gpu_usd": 2.5, "gpu_hours": 1.1, "wall_s": 3600.0},
    )

    path = campaign_report.write_campaign_report(tmp_path, state=state, rows=[], now=_now)
    text = path.read_text(encoding="utf-8")

    assert "10.00" in text
    assert "20.00" in text
    assert "5.00" in text
    assert "1.23" in text
    assert "2.50" in text
    assert "1.10" in text
    assert "10.0" in text  # budget wall-clock, seconds -> hours
    assert "1.0" in text  # spent wall-clock, seconds -> hours


# ---------------------------------------------------------------------------
# Misc: return type + path sanity for write_campaign_report
# ---------------------------------------------------------------------------


def test_report_path_is_under_run_dir(tmp_path):
    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=[], now=_now)
    assert isinstance(path, Path)
    assert path.parent == tmp_path


def _row_cells(row: str) -> list[str]:
    """Split one ``| a | b | c |`` markdown row into clean cell strings, in
    the same left-to-right order ``_render_attempt_row`` builds them."""
    inner = row.strip()
    assert inner.startswith("|") and inner.endswith("|"), row
    return [c.strip() for c in inner[1:-1].split("|")]


# ---------------------------------------------------------------------------
# U8 review: content assertions (not just "some text appears somewhere")
# ---------------------------------------------------------------------------


def test_report_attempt_row_renders_literal_decision_rule(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(1, _assessment(1)),
        _decided_row(
            1, {"kind": "CONTINUE", "rule": "budget_not_exhausted", "stop_reason": None, "next_plan": None}
        ),
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    row = next(line for line in _attempts_section(path.read_text(encoding="utf-8")) if line.startswith("| 1 |"))

    assert _row_cells(row)[-1] == "budget_not_exhausted"


def test_report_champion_present_renders_score_and_evidence_line(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(
            1,
            _assessment(
                1,
                final_report=_final_report(score=0.71, target=0.6),
                evidence_predicates={"a": True, "b": True, "c": False},
                leaf_pass_count=5,
            ),
        ),
    ]
    state = _state(
        terminal={"kind": "CONVERGED", "rule": "target_met", "stop_reason": None, "champion_attempt_n": 1}
    )

    path = campaign_report.write_campaign_report(tmp_path, state=state, rows=rows, now=_now)
    champion_section = path.read_text(encoding="utf-8").split("## Champion", 1)[1].split("\n## ", 1)[0]

    assert "Attempt 1" in champion_section
    assert "0.710/0.600" in champion_section
    assert "2/3 predicates" in champion_section
    assert "5 leaves" in champion_section


def test_report_header_in_progress_when_terminal_is_none(tmp_path):
    state = _state(terminal=None)

    path = campaign_report.write_campaign_report(tmp_path, state=state, rows=[], now=_now)
    text = path.read_text(encoding="utf-8")

    assert "- Terminal: in progress" in text


def test_report_assessed_row_with_missing_report_renders_em_dash_scores(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(1, _assessment(1, final_report=None)),
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    row = next(line for line in _attempts_section(path.read_text(encoding="utf-8")) if line.startswith("| 1 |"))
    cells = _row_cells(row)

    assert cells[2] == "—/—"  # score/target
    assert cells[3] == "—"  # meets
    assert cells[4] == "—/—"  # impl/repl verdicts


# ---------------------------------------------------------------------------
# FIX 1: exclusions with verification status + claims-vs-measured (§12
# locked decision 9)
# ---------------------------------------------------------------------------


def test_report_exclusions_render_both_ways(tmp_path):
    # (a) exclusions_detail present -> structured, verification-status bullets
    # take priority over the plain fallback strings.
    detail_dir = tmp_path / "detail"
    detail_dir.mkdir()
    rows_detail = [
        _launched_row(1),
        _assessed_row(
            1,
            _assessment(
                1,
                final_report=_final_report(
                    exclusions=["legacy string exclusion"],
                    exclusions_detail=[
                        {
                            "item": "ImageNet", "axis": "dataset", "kind": "capacity_vram",
                            "verified": True, "reason": "24GB budget exceeded",
                        },
                        {
                            "item": "COCO", "axis": "dataset", "kind": "dataset_dead",
                            "verified": False, "reason": "agent-declared, uncorroborated",
                        },
                    ],
                ),
            ),
        ),
    ]

    path_detail = campaign_report.write_campaign_report(detail_dir, state=_state(), rows=rows_detail, now=_now)
    exclusions_detail_section = (
        path_detail.read_text(encoding="utf-8").split("## Exclusions", 1)[1].split("\n## ", 1)[0]
    )

    assert "- ImageNet [dataset/capacity_vram] — verified: 24GB budget exceeded" in exclusions_detail_section
    assert (
        "- COCO [dataset/dataset_dead] — UNVERIFIED: agent-declared, uncorroborated"
        in exclusions_detail_section
    )
    assert "legacy string exclusion" not in exclusions_detail_section

    # (b) exclusions_detail absent -> falls back to the plain exclusion strings.
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    rows_fallback = [
        _launched_row(1),
        _assessed_row(1, _assessment(1, final_report=_final_report(exclusions=["ImageNet: model unavailable"]))),
    ]

    path_fallback = campaign_report.write_campaign_report(
        fallback_dir, state=_state(), rows=rows_fallback, now=_now
    )
    exclusions_fallback_section = (
        path_fallback.read_text(encoding="utf-8").split("## Exclusions", 1)[1].split("\n## ", 1)[0]
    )

    assert "- ImageNet: model unavailable" in exclusions_fallback_section


def test_report_claims_vs_measured_table(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(
            1,
            _assessment(
                1,
                final_report=_final_report(
                    per_claim=[
                        {
                            "claim_id": "table1_row2_alfworld_acc", "status": "reproduced",
                            "credit": 1.0, "eligible": True, "measured_mean": 0.71,
                            "ci_low": 0.65, "ci_high": 0.77,
                        },
                        {
                            "claim_id": "table1_row3_webshop_acc", "status": "contradicted",
                            "credit": 0.0, "eligible": True, "measured_mean": 0.12,
                            "ci_low": None, "ci_high": None,
                        },
                    ],
                ),
            ),
        ),
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    section = path.read_text(encoding="utf-8").split("## Claims vs measured", 1)[1].split("\n## ", 1)[0]

    assert "| claim | status | credit | measured | CI |" in section
    assert "| table1_row2_alfworld_acc | reproduced | 1.000 | 0.710 | [0.650, 0.770] |" in section
    assert "| table1_row3_webshop_acc | contradicted | 0.000 | 0.120 | — |" in section


def test_report_claims_vs_measured_empty_is_explicit(tmp_path):
    rows = [
        _launched_row(1),
        _assessed_row(1, _assessment(1)),  # default _final_report() carries no per_claim key
    ]

    path = campaign_report.write_campaign_report(tmp_path, state=_state(), rows=rows, now=_now)
    section = path.read_text(encoding="utf-8").split("## Claims vs measured", 1)[1].split("\n## ", 1)[0]

    assert "- no per-claim data recorded" in section

    # Zero-attempts campaign also renders the explicit empty state (no header row).
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_path = campaign_report.write_campaign_report(empty_dir, state=_state(), rows=[], now=_now)
    empty_section = empty_path.read_text(encoding="utf-8").split("## Claims vs measured", 1)[1].split("\n## ", 1)[0]
    assert "- no per-claim data recorded" in empty_section


# ---------------------------------------------------------------------------
# FIX 2/3: plan-only writer atomic claim + crash-stranded md recovery
# ---------------------------------------------------------------------------


def test_plan_only_writer_atomic_claim_second_call_does_not_overwrite(tmp_path):
    """A second (racing / retried) call on the same run_dir must never
    clobber the first call's already-claimed json+md -- the exists()-check-
    then-write race is replaced by an O_CREAT|O_EXCL atomic claim."""
    state1 = _state(paper_ref="2601.00002", project_id="proj_first")
    json_path, md_path = campaign_report.write_plan_only_report(
        tmp_path, stop_reason="infeasible:first", what_would_unblock=["a"], state=state1, now=_now,
    )
    original_json = json_path.read_text(encoding="utf-8")
    original_md = md_path.read_text(encoding="utf-8")

    state2 = _state(paper_ref="2601.99999", project_id="proj_second")
    json_path2, md_path2 = campaign_report.write_plan_only_report(
        tmp_path, stop_reason="infeasible:second", what_would_unblock=["b"], state=state2, now=_now,
    )

    assert json_path2 == json_path
    assert md_path2 == md_path
    assert json_path.read_text(encoding="utf-8") == original_json
    assert md_path.read_text(encoding="utf-8") == original_md


def test_plan_only_recovers_stranded_md_from_plan_only_json(tmp_path):
    """F14 crash recovery: a prior claim wrote final_report.json (verdict
    plan_only) but the process died before writing final_report.md. The next
    call must regenerate ONLY the md, from the STRANDED JSON'S OWN content --
    never from this call's (possibly different) arguments."""
    json_path = tmp_path / "final_report.json"
    stranded_payload = {
        "verdict": "plan_only",
        "stop_reason": "infeasible:asset_gap",
        "what_would_unblock": ["dataset access"],
        "paper": {"ref": "2601.00007"},
        "campaign": {"project_id": "proj_stranded", "terminal": None, "spent": {}},
        "generated_at": "2026-07-01T00:00:00+00:00",
    }
    json_path.write_text(json.dumps(stranded_payload), encoding="utf-8")
    md_path = tmp_path / "final_report.md"
    assert not md_path.exists()

    result_json, result_md = campaign_report.write_plan_only_report(
        tmp_path,
        stop_reason="infeasible:should_not_apply",
        what_would_unblock=["should not appear"],
        state=_state(paper_ref="2601.99999", project_id="proj_new"),
        now=_now,
    )

    assert result_json == json_path
    assert result_md == md_path
    # The stranded json is untouched byte-for-byte (still just the claim).
    assert json.loads(json_path.read_text(encoding="utf-8")) == stranded_payload
    # The md is regenerated from the STRANDED json's own content...
    md_text = md_path.read_text(encoding="utf-8")
    assert "2601.00007" in md_text
    assert "proj_stranded" in md_text
    assert "infeasible:asset_gap" in md_text
    assert "dataset access" in md_text
    # ...never from this call's arguments.
    assert "2601.99999" not in md_text
    assert "should not appear" not in md_text
    assert "infeasible:should_not_apply" not in md_text


def test_plan_only_does_not_write_md_for_a_real_stranded_report(tmp_path):
    """The crash-recovery path is scoped ONLY to a stranded plan_only json --
    a real attempt report (any other verdict) with a missing md sibling is
    left strictly alone; writing its md is not this function's job."""
    json_path = tmp_path / "final_report.json"
    real_payload = {"verdict": "reproduced", "rubric": {"overall_score": 0.8}}
    json_path.write_text(json.dumps(real_payload), encoding="utf-8")
    md_path = tmp_path / "final_report.md"
    assert not md_path.exists()

    result_json, result_md = campaign_report.write_plan_only_report(
        tmp_path, stop_reason="infeasible:should_not_apply", what_would_unblock=[], state=_state(), now=_now,
    )

    assert result_json == json_path
    assert result_md == md_path
    assert json.loads(json_path.read_text(encoding="utf-8")) == real_payload
    assert not md_path.exists()


def test_atomic_write_text_tmp_filename_is_pid_namespaced(tmp_path, monkeypatch):
    """The shared _atomic_write_text tmp file must be namespaced with the
    writer's pid so two processes racing on the same target path never share
    (and clobber) one another's in-flight tmp file."""
    monkeypatch.setattr(campaign_report.os, "getpid", lambda: 987654)
    seen_tmp_names = []
    real_replace = campaign_report.os.replace

    def _spy_replace(src, dst):
        seen_tmp_names.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(campaign_report.os, "replace", _spy_replace)

    path = tmp_path / "out.txt"
    campaign_report._atomic_write_text(path, "hello")

    assert seen_tmp_names == ["out.txt.tmp.987654"]
    assert path.read_text(encoding="utf-8") == "hello"
    assert not (tmp_path / "out.txt.tmp.987654").exists()  # renamed away, nothing left behind
