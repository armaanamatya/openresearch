"""Unit tests for scripts/regrade_backfill.py (Q6 regrade-backfill lane).

Fully self-contained: no real LLM, no network. The grader is injected two ways:
  * a stub ``llm_client`` (only needs a ``.complete(system=, user=)`` method), and
  * a fake ``score_fn`` standing in for ``leaf_scorer.score_reproduction`` — so the
    test never depends on the scorer's evidence-gathering or its on-disk contract.
    (The real scorer is exercised by the live CLI; here we verify the backfill
    *plumbing*: stamping, v0 preservation, dry-run safety, skip-on-no-rubric.)

Run: /home/sww35/openresearch/.venv/bin/python -m pytest tests/rlm/test_regrade_backfill.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import regrade_backfill as rb


# ---------------------------------------------------------------------------
# Stubs / fixtures
# ---------------------------------------------------------------------------


class _StubLlm:
    """Satisfies the grader transport Protocol; records nothing, returns nothing useful.

    The fake ``score_fn`` ignores it entirely — it exists only to prove the seam
    (the backfill threads a client through to the scorer) without any network.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:  # pragma: no cover - never parsed
        self.calls += 1
        return "{}"


def _make_score_fn(new_overall: float = 0.85, *, target: float | None = 0.7):
    """Return a deterministic fake ``score_reproduction`` producing a known result.

    Records the kwargs it was called with so a test can assert ``degraded=False`` is
    passed explicitly and ``rubric_source`` is inferred correctly.
    """
    calls: list[dict] = []

    def _score(rubric_tree, run_dir, llm_client, **kwargs):
        calls.append(
            {
                "run_dir": Path(run_dir),
                "rubric_source": kwargs.get("rubric_source"),
                "degraded": kwargs.get("degraded"),
            }
        )
        return {
            "overall_score": new_overall,
            "target_score": target,
            "leaf_count": 3,
            "graded": 3,
            "coverage_pct": 1.0,
            "degraded": False,
            "rubric_source": kwargs.get("rubric_source"),
            "leaf_scores": [
                {"id": "leaf-a", "score": new_overall, "justification": "regraded"},
            ],
            "eligible_count": 3,
            "unavailable_count": 0,
            "invariant_results": [],
            "invariant_gate_applied": False,
        }

    _score.calls = calls  # type: ignore[attr-defined]
    return _score


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _make_scored_run(
    run_dir: Path,
    *,
    final_overall: float = 0.42,
    eval_overall: float = 0.42,
    rubric_filename: str = "rubric_tree.json",
    with_final: bool = True,
    with_eval: bool = True,
) -> None:
    """Create a run dir with a rubric tree + scored final_report/rubric_evaluation."""
    _write(run_dir / rubric_filename, {"target_score": 0.7, "criteria": []})
    if with_final:
        _write(
            run_dir / "final_report.json",
            {
                "mode": "rlm",
                "rubric": {
                    "overall_score": final_overall,
                    "target_score": 0.7,
                    "meets_target": final_overall >= 0.7,
                    "areas": [{"name": "fidelity", "score": 1.0, "notes": ""}],
                    "leaf_scores": [{"id": "leaf-a", "score": final_overall}],
                    "rubric_source": "paperbench_bundle",
                },
            },
        )
    if with_eval:
        _write(
            run_dir / "rubric_evaluation.json",
            {
                "iteration": 7,
                "overall_score": eval_overall,
                "target_score": 0.7,
                "meets_target": eval_overall >= 0.7,
                "leaf_count": 1,
                "graded": 1,
                "rubric_source": "paperbench_bundle",
                "degraded": False,
                "compute_scope": None,
                "leaf_scores": [{"id": "leaf-a", "score": eval_overall}],
            },
        )


# ---------------------------------------------------------------------------
# 1. apply: stamp written + fresh score adopted on BOTH artifact shapes
# ---------------------------------------------------------------------------


def test_apply_stamps_both_artifacts_and_adopts_new_score(tmp_path: Path) -> None:
    run = tmp_path / "prj_a"
    _make_scored_run(run, final_overall=0.42, eval_overall=0.42)
    score_fn = _make_score_fn(new_overall=0.85, target=0.7)

    outcome = rb.backfill_run(
        run, _StubLlm(), grader_version="v1", grader_samples=1, apply=True, score_fn=score_fn
    )

    assert outcome.status == "regraded"
    # score_reproduction must be called exactly once, with degraded=False explicit.
    assert len(score_fn.calls) == 1  # type: ignore[attr-defined]
    assert score_fn.calls[0]["degraded"] is False  # type: ignore[attr-defined]
    assert score_fn.calls[0]["rubric_source"] == "paperbench_bundle"  # type: ignore[attr-defined]

    # final_report.json: stamp lands on the nested "rubric" block; score adopted.
    fr = json.loads((run / "final_report.json").read_text())
    assert fr["rubric"]["grader_version"] == "v1"
    assert fr["rubric"]["grader_samples"] == 1
    assert fr["rubric"]["grader_temperature"] == 0
    assert fr["rubric"]["overall_score"] == 0.85
    assert fr["rubric"]["meets_target"] is True  # 0.85 >= 0.7

    # rubric_evaluation.json: stamp lands at the TOP level (flat dict); score adopted.
    re = json.loads((run / "rubric_evaluation.json").read_text())
    assert re["grader_version"] == "v1"
    assert re["grader_samples"] == 1
    assert re["grader_temperature"] == 0
    assert re["overall_score"] == 0.85
    assert re["meets_target"] is True
    # Non-scoring keys preserved through the merge.
    assert re["iteration"] == 7
    assert re["compute_scope"] is None


# ---------------------------------------------------------------------------
# 2. apply: v0 sidecar created and the ORIGINAL value preserved verbatim
# ---------------------------------------------------------------------------


def test_apply_preserves_v0_sidecar_with_original_values(tmp_path: Path) -> None:
    run = tmp_path / "prj_b"
    _make_scored_run(run, final_overall=0.42, eval_overall=0.42)
    score_fn = _make_score_fn(new_overall=0.91)

    rb.backfill_run(run, _StubLlm(), apply=True, score_fn=score_fn)

    fr_v0 = run / "final_report.v0.json"
    re_v0 = run / "rubric_evaluation.v0.json"
    assert fr_v0.exists() and re_v0.exists()

    # The v0 sidecars hold the ORIGINAL (pre-backfill) numbers, untouched.
    assert json.loads(fr_v0.read_text())["rubric"]["overall_score"] == 0.42
    assert json.loads(re_v0.read_text())["overall_score"] == 0.42
    # And the v0 must NOT carry the new stamp (it's a pristine snapshot).
    assert "grader_version" not in json.loads(re_v0.read_text())

    # The live files DID change to the new score.
    assert json.loads((run / "final_report.json").read_text())["rubric"]["overall_score"] == 0.91


def test_v0_archive_is_idempotent(tmp_path: Path) -> None:
    """A second backfill must not overwrite the v0 snapshot with a v1 value."""
    run = tmp_path / "prj_c"
    _make_scored_run(run, final_overall=0.30, eval_overall=0.30)

    rb.backfill_run(run, _StubLlm(), apply=True, score_fn=_make_score_fn(new_overall=0.60))
    # Second pass grades higher; v0 must still show the FIRST original (0.30).
    second = rb.backfill_run(
        run, _StubLlm(), apply=True, score_fn=_make_score_fn(new_overall=0.99)
    )

    re_v0 = json.loads((run / "rubric_evaluation.v0.json").read_text())
    assert re_v0["overall_score"] == 0.30  # original preserved, not 0.60
    # Outcomes on the 2nd pass report v0_archived False (already archived).
    assert all(not a.v0_archived for a in second.artifacts)
    # Live file advanced to the newest grade.
    assert json.loads((run / "rubric_evaluation.json").read_text())["overall_score"] == 0.99


# ---------------------------------------------------------------------------
# 3. dry-run: writes NOTHING (no stamp, no v0 sidecar, score untouched)
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    run = tmp_path / "prj_d"
    _make_scored_run(run, final_overall=0.42, eval_overall=0.42)
    before_fr = (run / "final_report.json").read_text()
    before_re = (run / "rubric_evaluation.json").read_text()

    outcome = rb.backfill_run(
        run, _StubLlm(), apply=False, score_fn=_make_score_fn(new_overall=0.88)
    )

    # Status is still "regraded" (we computed the new score) but nothing was written.
    assert outcome.status == "regraded"
    assert all(a.written is False for a in outcome.artifacts)
    # The would-be new score is reported for the diff...
    assert {a.new_score for a in outcome.artifacts} == {0.88}
    # ...and the old score is reported too.
    assert {a.old_score for a in outcome.artifacts} == {0.42}
    # No v0 sidecars created.
    assert not (run / "final_report.v0.json").exists()
    assert not (run / "rubric_evaluation.v0.json").exists()
    # Original files byte-for-byte unchanged.
    assert (run / "final_report.json").read_text() == before_fr
    assert (run / "rubric_evaluation.json").read_text() == before_re


# ---------------------------------------------------------------------------
# 4. skip: a dir with no rubric (and a dir with rubric but no scored artifact)
# ---------------------------------------------------------------------------


def test_skip_run_with_no_rubric(tmp_path: Path) -> None:
    run = tmp_path / "prj_no_rubric"
    # A scored report but NO rubric tree → nothing to re-grade against.
    _write(run / "final_report.json", {"rubric": {"overall_score": 0.5, "target_score": 0.7}})
    score_fn = _make_score_fn()

    outcome = rb.backfill_run(run, _StubLlm(), apply=True, score_fn=score_fn)

    assert outcome.status == "skipped"
    assert outcome.reason == "no rubric tree"
    assert outcome.artifacts == []
    # Grader never invoked; original file untouched.
    assert len(score_fn.calls) == 0  # type: ignore[attr-defined]
    assert json.loads((run / "final_report.json").read_text())["rubric"]["overall_score"] == 0.5


def test_skip_run_with_rubric_but_no_scored_artifact(tmp_path: Path) -> None:
    run = tmp_path / "prj_unscored"
    # Rubric present but no final_report/rubric_evaluation carrying a score.
    _write(run / "rubric_tree.json", {"target_score": 0.7, "criteria": []})
    _write(run / "demo_status.json", {"status": "interrupted"})
    score_fn = _make_score_fn()

    outcome = rb.backfill_run(run, _StubLlm(), apply=True, score_fn=score_fn)

    assert outcome.status == "skipped"
    assert outcome.reason == "no scored artifact"
    assert len(score_fn.calls) == 0  # type: ignore[attr-defined]


def test_generated_rubric_infers_generated_source(tmp_path: Path) -> None:
    """A run with generated_rubric.json (no rubric_tree.json) grades as source=generated."""
    run = tmp_path / "prj_generated"
    _make_scored_run(run, rubric_filename="generated_rubric.json")
    score_fn = _make_score_fn()

    rb.backfill_run(run, _StubLlm(), apply=True, score_fn=score_fn)

    assert score_fn.calls[0]["rubric_source"] == "generated"  # type: ignore[attr-defined]


def test_eval_only_run_is_regraded(tmp_path: Path) -> None:
    """A run with rubric_evaluation.json but no final_report.json still backfills."""
    run = tmp_path / "prj_eval_only"
    _make_scored_run(run, with_final=False, with_eval=True, eval_overall=0.5)

    outcome = rb.backfill_run(
        run, _StubLlm(), apply=True, score_fn=_make_score_fn(new_overall=0.77)
    )

    assert outcome.status == "regraded"
    names = {a.name for a in outcome.artifacts}
    assert names == {"rubric_evaluation.json"}
    assert json.loads((run / "rubric_evaluation.json").read_text())["overall_score"] == 0.77


# ---------------------------------------------------------------------------
# 5. discovery / sweep: top-level runs + attempts/*; --only filter
# ---------------------------------------------------------------------------


def test_discover_includes_attempts_and_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_scored_run(runs_root / "prj_x")
    _make_scored_run(runs_root / "prj_x" / "attempts" / "20260611T000001-aa")
    _make_scored_run(runs_root / "prj_y")
    (runs_root / "not_a_run.txt").parent.mkdir(parents=True, exist_ok=True)
    (runs_root / "not_a_run.txt").write_text("x")

    found = rb.discover_run_dirs(runs_root)
    names = {p.name for p in found}
    assert "prj_x" in names
    assert "prj_y" in names
    assert "20260611T000001-aa" in names  # the attempt sub-dir is included
    # The stray file is not a dir → excluded.
    assert all(p.is_dir() for p in found)


def test_only_filter_scopes_to_one_run_and_its_attempts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_scored_run(runs_root / "prj_x")
    _make_scored_run(runs_root / "prj_x" / "attempts" / "att1")
    _make_scored_run(runs_root / "prj_y")

    found = rb.discover_run_dirs(runs_root, only="prj_x")
    names = {p.name for p in found}
    assert names == {"prj_x", "att1"}  # prj_y excluded; prj_x's attempt rides along


def test_backfill_root_regrades_all_and_skips_gracefully(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_scored_run(runs_root / "prj_scored")
    # An un-gradeable dir mixed in.
    (runs_root / "prj_empty").mkdir(parents=True, exist_ok=True)
    _write(runs_root / "prj_empty" / "demo_status.json", {"status": "queued"})

    outcomes = rb.backfill_root(
        runs_root, _StubLlm(), apply=True, score_fn=_make_score_fn(new_overall=0.8)
    )

    by_name = {o.run_dir.name: o for o in outcomes}
    assert by_name["prj_scored"].status == "regraded"
    assert by_name["prj_empty"].status == "skipped"


# ---------------------------------------------------------------------------
# 6. stamp value helpers
# ---------------------------------------------------------------------------


def test_grader_samples_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "3")
    assert rb._grader_samples_from_env() == 3
    stamp = rb.build_stamp("v1")
    assert stamp == {"grader_version": "v1", "grader_samples": 3, "grader_temperature": 0}


def test_grader_samples_defaults_to_one_when_unset_or_bad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENRESEARCH_GRADER_SAMPLES", raising=False)
    assert rb._grader_samples_from_env() == 1
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "not-a-number")
    assert rb._grader_samples_from_env() == 1
    monkeypatch.setenv("OPENRESEARCH_GRADER_SAMPLES", "0")
    assert rb._grader_samples_from_env() == 1  # non-positive clamps to 1


def test_meets_target_derivation() -> None:
    assert rb._derive_meets_target(0.8, 0.7) is True
    assert rb._derive_meets_target(0.6, 0.7) is False
    assert rb._derive_meets_target(0.8, None) is None  # unknown target
    assert rb._derive_meets_target(None, 0.7) is None  # non-numeric overall


# ---------------------------------------------------------------------------
# 7. CLI: dry-run is the default; --help touches no network
# ---------------------------------------------------------------------------


def test_cli_help_is_offline() -> None:
    parser = rb._build_arg_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_main_defaults_to_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --apply the CLI writes nothing, and never builds a real client."""
    runs_root = tmp_path / "runs"
    _make_scored_run(runs_root / "prj_z", final_overall=0.4, eval_overall=0.4)
    before = (runs_root / "prj_z" / "final_report.json").read_text()

    # Inject the stub client + fake scorer so main() touches no network even when
    # it (incorrectly) tries to build a real client. We patch the builder + scorer.
    monkeypatch.setattr(rb, "_build_real_llm_client", lambda: _StubLlm())
    fake = _make_score_fn(new_overall=0.95)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction", fake, raising=False
    )

    rc = rb.main(["--runs-root", str(runs_root)])
    assert rc == 0
    # Dry-run default → file unchanged, no v0 sidecar.
    assert (runs_root / "prj_z" / "final_report.json").read_text() == before
    assert not (runs_root / "prj_z" / "final_report.v0.json").exists()


def test_cli_dry_run_wins_over_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If both --apply and --dry-run are passed, the safe (no-write) path wins."""
    runs_root = tmp_path / "runs"
    _make_scored_run(runs_root / "prj_q", final_overall=0.4, eval_overall=0.4)
    before = (runs_root / "prj_q" / "rubric_evaluation.json").read_text()

    monkeypatch.setattr(rb, "_build_real_llm_client", lambda: _StubLlm())
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        _make_score_fn(new_overall=0.9),
        raising=False,
    )

    rc = rb.main(["--runs-root", str(runs_root), "--apply", "--dry-run"])
    assert rc == 0
    assert (runs_root / "prj_q" / "rubric_evaluation.json").read_text() == before


def test_cli_apply_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    _make_scored_run(runs_root / "prj_w", final_overall=0.4, eval_overall=0.4)

    monkeypatch.setattr(rb, "_build_real_llm_client", lambda: _StubLlm())
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        _make_score_fn(new_overall=0.66),
        raising=False,
    )

    rc = rb.main(["--runs-root", str(runs_root), "--apply", "--grader-version", "v1"])
    assert rc == 0
    re = json.loads((runs_root / "prj_w" / "rubric_evaluation.json").read_text())
    assert re["grader_version"] == "v1"
    assert re["overall_score"] == 0.66
    assert (runs_root / "prj_w" / "rubric_evaluation.v0.json").exists()
