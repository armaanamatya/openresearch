"""OFF-state half of the VerdictAuthority sever's hermetic flag-pair test
(Track A §4.3, ``backend/agents/rlm/verdict_authority.py``). Every flag in
this repo ships a hermetic OFF+ON pair (``tests/CLAUDE.md``); the ON-state
depth already lives in
``tests/agents/rlm/test_single_verdict_authority_guard.py`` (grade-
severance, runtime/static guards, fail-closed behaviour, per-severed-writer
coverage). This file proves the complementary claim: with either flag off,
the sever is completely inert.

Covers:
  (a) ``is_enabled()``'s gate matrix -- BOTH flags must be truthy, driven
      through every unset/single-flag/explicit-zero combination.
  (b) OFF-state inertness through ``write_final_report_rlm`` (the same
      harness/fixture shape as the guard test): no ``verdict_authority`` key
      ships in ``final_report.json``, and ``demo_status.json`` is never
      touched by the authority mirror -- whether or not one already exists
      on disk.
  (c) a minimal ON-state sanity check (the flag pair really does flip the
      behaviour the rest of this file holds inert); full ON-state depth
      stays owned by the guard test file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
from backend.agents.rlm.verdict_authority import is_enabled

BOTH_FLAGS = ("OPENRESEARCH_TWO_AXIS_VERDICT", "OPENRESEARCH_VERDICT_AUTHORITY")


def _disable_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in BOTH_FLAGS:
        monkeypatch.delenv(flag, raising=False)


def _enable_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", "1")
    monkeypatch.setenv("OPENRESEARCH_VERDICT_AUTHORITY", "1")


def _write_claim_fixture(project_dir: Path, *, accuracy: float = 0.991) -> None:
    """A single genuinely-primary numeric claim (claimed 0.99 +/- 0.01) plus a
    real success+metrics experiment_runs.jsonl row (the evidence-gate signal).

    Copied verbatim from ``test_single_verdict_authority_guard.py`` -- the
    same minimal fixture shape ``write_final_report_rlm`` needs regardless of
    the flag state, so the OFF-state harness here matches the ON-state guard
    test exactly.
    """
    (project_dir / "code").mkdir(parents=True, exist_ok=True)
    (project_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
    (project_dir / "code" / "metrics.json").write_text(
        json.dumps({"accuracy": accuracy}), encoding="utf-8"
    )
    (project_dir / "rlm_state" / "repro_spec.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "primary_0",
                        "is_primary": True,
                        "kind": "numeric",
                        "metric_name": "accuracy",
                        "claimed_effect": 0.99,
                        "equivalence_margin": 0.01,
                        "direction": "higher_is_better",
                        "scope": {},
                        "ambiguous": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": accuracy}}) + "\n",
        encoding="utf-8",
    )


def _default_rubric(overall_score: float | None, *, meets_target: bool | None = None) -> dict:
    return {
        "overall_score": overall_score,
        "target_score": 0.6,
        "meets_target": meets_target,
        "degraded": None,
        "areas": [],
    }


def _shipped(project_dir: Path) -> dict:
    return json.loads((project_dir / "final_report.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# (a) is_enabled(): the full gate matrix -- BOTH flags must be truthy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "two_axis, authority, expected",
    [
        (None, None, False),
        ("1", None, False),
        (None, "1", False),
        ("1", "1", True),
        ("0", "0", False),
    ],
    ids=[
        "both-unset",
        "two-axis-only",
        "authority-only",
        "both-truthy",
        "both-explicit-zero",
    ],
)
def test_is_enabled_gate_matrix(
    monkeypatch: pytest.MonkeyPatch,
    two_axis: str | None,
    authority: str | None,
    expected: bool,
) -> None:
    if two_axis is None:
        monkeypatch.delenv("OPENRESEARCH_TWO_AXIS_VERDICT", raising=False)
    else:
        monkeypatch.setenv("OPENRESEARCH_TWO_AXIS_VERDICT", two_axis)
    if authority is None:
        monkeypatch.delenv("OPENRESEARCH_VERDICT_AUTHORITY", raising=False)
    else:
        monkeypatch.setenv("OPENRESEARCH_VERDICT_AUTHORITY", authority)

    assert is_enabled() is expected


# --------------------------------------------------------------------------- #
# (b) OFF-state inertness through write_final_report_rlm
# --------------------------------------------------------------------------- #


def test_off_state_final_report_has_no_verdict_authority_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)

    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )

    shipped = _shipped(project_dir)
    assert "verdict_authority" not in shipped
    assert shipped["verdict"] == "reproduced"  # the pre-authority verdict stands, untouched
    assert not (project_dir / "demo_status.json").exists()  # mirror never ran, nothing to create


def test_off_state_preexisting_demo_status_untouched_by_authority_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stronger than plain absence: even when demo_status.json ALREADY exists
    on disk (as it would mid-run, from the live pipeline's own status
    snapshots), the OFF-state path must never touch it -- proving the mirror
    is genuinely skipped, not merely that it happened to have nothing to
    create.
    """
    _disable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)
    seed_status = {"status": "running", "verdict": "unknown"}
    (project_dir / "demo_status.json").write_text(json.dumps(seed_status), encoding="utf-8")

    write_final_report_rlm(
        RLMFinalReport(verdict="reproduced", rubric=_default_rubric(0.95, meets_target=True)),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )

    shipped = _shipped(project_dir)
    assert "verdict_authority" not in shipped
    demo_status = json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert demo_status == seed_status  # byte-identical -- authority mirror never ran


# --------------------------------------------------------------------------- #
# (c) ON-state sanity: proves the flag pair actually gates the behaviour above
# --------------------------------------------------------------------------- #


def test_on_state_sanity_final_report_has_verdict_authority_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minimal ON-state check -- proves the flag pair this file's OFF-state
    tests hold inert genuinely gates the behaviour. Full ON-state depth
    (grade-severance, taxonomy, fail-closed, per-writer coverage) stays owned
    by test_single_verdict_authority_guard.py.
    """
    _enable_authority(monkeypatch)
    project_dir = tmp_path / "run"
    _write_claim_fixture(project_dir)

    write_final_report_rlm(
        RLMFinalReport(verdict="partial"),
        project_dir,
        run_experiment_ok_calls=1,
        run_experiment_calls=1,
    )

    assert "verdict_authority" in _shipped(project_dir)
