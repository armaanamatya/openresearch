"""Track E Task 6: ``OPENRESEARCH_EVAL_SCORECARD`` hermetic OFF+ON pair.

Off (unset, or any falsy token) -> ``write_evaluation_report`` is a pure
no-op: returns ``None`` and writes NEITHER ``evaluation_report.json`` NOR
``evaluation_report.md`` -- byte-identical to a build with no scorecard
module at all. On -> it composes the ``EvaluationReport`` (``verdict``
copied read-only from ``final_report.json``), attaches the 11-dimension
scorecard, and writes both sidecars, returning the json ``Path``.
``final_report.json`` itself is never mutated (read-only contract).
"""

import json

from backend.evals.scorecard import scorecard_enabled, write_evaluation_report


def test_scorecard_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EVAL_SCORECARD", raising=False)
    assert scorecard_enabled() is False


def test_off_state_returns_none_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EVAL_SCORECARD", raising=False)
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": "partial"}))
    assert write_evaluation_report(tmp_path) is None
    assert not (tmp_path / "evaluation_report.json").exists()
    assert not (tmp_path / "evaluation_report.md").exists()


def test_explicit_zero_is_also_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EVAL_SCORECARD", "0")
    assert write_evaluation_report(tmp_path) is None
    assert not (tmp_path / "evaluation_report.json").exists()


def test_off_state_on_a_populated_run_dir_still_writes_nothing(tmp_path, monkeypatch):
    # Byte-identical claim must hold even when every other sidecar this module
    # can read (experiment_runs.jsonl, gpu_ledger.jsonl, ...) already exists.
    monkeypatch.delenv("OPENRESEARCH_EVAL_SCORECARD", raising=False)
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": "reproduced"}))
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"a": 1}}) + "\n"
    )
    before = sorted(p.name for p in tmp_path.iterdir())
    assert write_evaluation_report(tmp_path) is None
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after  # no new file appeared


def test_on_state_writes_json_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EVAL_SCORECARD", "1")
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": "inconclusive"}))
    out = write_evaluation_report(tmp_path)
    assert out == tmp_path / "evaluation_report.json"
    assert out.exists()
    md_path = tmp_path / "evaluation_report.md"
    assert md_path.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["verdict"] == "inconclusive"
    assert len(data["scorecard"]) == 11
    assert "Verdict:** inconclusive" in md_path.read_text(encoding="utf-8")


def test_on_state_never_mutates_final_report_json(tmp_path, monkeypatch):
    # The sidecar's "verdict" is a verbatim copy -- writing it must never
    # touch final_report.json itself (read-only contract: rubric/verdict are
    # read, never recomputed or written back).
    monkeypatch.setenv("OPENRESEARCH_EVAL_SCORECARD", "1")
    original = json.dumps({"verdict": "reproduced", "rubric": {"areas": []}})
    (tmp_path / "final_report.json").write_text(original)
    write_evaluation_report(tmp_path)
    assert (tmp_path / "final_report.json").read_text(encoding="utf-8") == original
