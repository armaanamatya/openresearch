"""Tests for backend.agents.rlm.repl_snapshot.

Issue #62 DC#4 — runs produced no repl_state.pickle / iterations/ directory.

Critical tests:
- iterations/iteration_NNNN.json is written with safe variables.
- context and _private keys are excluded from both artifacts.
- Corpus sentinel NEVER appears in the JSON file text (redact_corpus guard).
- repl_state.pickle is written and loadable; un-picklable values are tombstoned.
- OpenResearchRLMLogger with snapshot_writer=None still works (back-compat).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rlm.core.types import CodeBlock, REPLResult, RLMIteration

from backend.agents.rlm.repl_snapshot import ReplSnapshotWriter
from backend.agents.rlm.sse_bridge import OpenResearchRLMLogger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_SENTINEL = "SECRETCORPUSTOKEN abcdefghij"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_iteration(locals_: dict | None = None) -> RLMIteration:
    """Build an RLMIteration whose last code block has the given locals."""
    result = REPLResult(
        stdout="ok\n",
        stderr="",
        locals=locals_ or {},
        execution_time=0.5,
        rlm_calls=[],
    )
    block = CodeBlock(code="baseline_result = run_baseline()", result=result)
    return RLMIteration(
        prompt={"role": "user", "content": "reproduce"},
        response="running baseline",
        code_blocks=[block],
        final_answer=None,
        iteration_time=1.23,
    )


# ---------------------------------------------------------------------------
# Test: per-iteration JSON snapshot (issue #62 DC#4)
# ---------------------------------------------------------------------------

class TestIterationJsonSnapshot:
    """Issue #62 DC#4 — iterations/iteration_NNNN.json not written."""

    def _make_writer(self, tmp_path: Path) -> ReplSnapshotWriter:
        return ReplSnapshotWriter(
            project_dir=tmp_path,
            sentinels=[CORPUS_SENTINEL],
        )

    def test_iterations_dir_created_on_init(self, tmp_path: Path):
        _make_writer = lambda: ReplSnapshotWriter(project_dir=tmp_path)  # noqa: E731
        assert not (tmp_path / "iterations").exists()
        _make_writer()
        assert (tmp_path / "iterations").is_dir()

    def test_json_file_created_at_correct_path(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"baseline_result": {"success": True}})
        writer.write(iteration, 1)
        assert (tmp_path / "iterations" / "iteration_0001.json").exists()

    def test_json_file_zero_padded_index(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 1})
        writer.write(iteration, 42)
        assert (tmp_path / "iterations" / "iteration_0042.json").exists()

    def test_json_contains_safe_variable(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
        })
        writer.write(iteration, 1)

        path = tmp_path / "iterations" / "iteration_0001.json"
        data = json.loads(path.read_text())
        assert "baseline_result" in data["variables"]

    def test_json_excludes_context_key(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
            "bad": (lambda: 1),
        })
        writer.write(iteration, 1)

        path = tmp_path / "iterations" / "iteration_0001.json"
        data = json.loads(path.read_text())
        assert "context" not in data["variables"]

    def test_json_excludes_underscore_key(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
        })
        writer.write(iteration, 1)

        path = tmp_path / "iterations" / "iteration_0001.json"
        data = json.loads(path.read_text())
        assert "_hidden" not in data["variables"]

    def test_corpus_sentinel_never_in_json_text(self, tmp_path: Path):
        """Corpus-safety guard: the sentinel string must NOT appear anywhere in the file."""
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
            "bad": (lambda: 1),
        })
        writer.write(iteration, 1)

        path = tmp_path / "iterations" / "iteration_0001.json"
        raw_text = path.read_text()
        assert CORPUS_SENTINEL not in raw_text, (
            f"Corpus sentinel leaked into iteration JSON: {raw_text[:500]!r}"
        )

    def test_json_has_corpus_ref_field(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 1})
        writer.write(iteration, 1)

        data = json.loads((tmp_path / "iterations" / "iteration_0001.json").read_text())
        assert data["corpus_ref"] == "parsed_full_text.txt"

    def test_json_has_timing_field(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 1})
        writer.write(iteration, 1)

        data = json.loads((tmp_path / "iterations" / "iteration_0001.json").read_text())
        assert data["timing"] == pytest.approx(1.23)

    def test_json_code_blocks_have_char_counts(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 1})
        writer.write(iteration, 1)

        data = json.loads((tmp_path / "iterations" / "iteration_0001.json").read_text())
        assert len(data["code_blocks"]) == 1
        cb = data["code_blocks"][0]
        assert "stdout_chars" in cb
        assert "stderr_chars" in cb
        assert "code" in cb


# ---------------------------------------------------------------------------
# Test: repl_state.pickle (issue #62 DC#4)
# ---------------------------------------------------------------------------

class TestReplStatePickle:
    """Issue #62 DC#4 — repl_state.pickle not written."""

    def _make_writer(self, tmp_path: Path) -> ReplSnapshotWriter:
        return ReplSnapshotWriter(
            project_dir=tmp_path,
            sentinels=[CORPUS_SENTINEL],
        )

    def test_pickle_file_created(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 1})
        writer.write(iteration, 1)
        assert (tmp_path / "repl_state.pickle").exists()

    def test_pickle_is_loadable(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 42})
        writer.write(iteration, 1)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert isinstance(payload, dict)

    def test_pickle_schema_version(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 1})
        writer.write(iteration, 1)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert payload["schema_version"] == 1

    def test_pickle_contains_safe_variable(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
        })
        writer.write(iteration, 1)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert payload["variables"]["baseline_result"] == {"success": True, "metrics": {"acc": 0.9}}

    def test_pickle_excludes_context(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
        })
        writer.write(iteration, 1)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert "context" not in payload["variables"]

    def test_pickle_excludes_underscore_keys(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
        })
        writer.write(iteration, 1)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert "_hidden" not in payload["variables"]

    def test_pickle_tombstones_unpicklable_value(self, tmp_path: Path):
        """Un-picklable values (e.g. lambdas) are replaced with a tombstone dict."""
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({
            "baseline_result": {"success": True, "metrics": {"acc": 0.9}},
            "context": {"paper_text": CORPUS_SENTINEL},
            "_hidden": 123,
            "bad": (lambda: 1),
        })
        writer.write(iteration, 1)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        bad_entry = payload["variables"]["bad"]
        assert isinstance(bad_entry, dict)
        assert "__unpicklable__" in bad_entry

    def test_pickle_corpus_ref_field(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        iteration = _make_iteration({"x": 1})
        writer.write(iteration, 1)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert payload["corpus_ref"] == "parsed_full_text.txt"

    def test_pickle_overwritten_on_each_call(self, tmp_path: Path):
        """repl_state.pickle always reflects the LATEST iteration."""
        writer = self._make_writer(tmp_path)
        writer.write(_make_iteration({"x": 1}), 1)
        writer.write(_make_iteration({"x": 999}), 2)

        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert payload["iteration"] == 2
        assert payload["variables"]["x"] == 999


# ---------------------------------------------------------------------------
# Test: exception safety — write() must never crash a run
# ---------------------------------------------------------------------------

class TestWriteExceptionSafe:
    """Issue #62 DC#4 — snapshotting must never crash a run."""

    def test_write_swallows_exceptions(self, tmp_path: Path, monkeypatch):
        """If _write raises, write() must catch and log — never re-raise."""
        writer = ReplSnapshotWriter(project_dir=tmp_path)

        def _explode(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(writer, "_write", _explode)
        iteration = _make_iteration({"x": 1})
        # Must not raise
        writer.write(iteration, 1)

    def test_empty_code_blocks_writes_json_not_pickle(self, tmp_path: Path):
        """A pure-reasoning iteration (no code blocks) records the iteration JSON
        but does not create repl_state.pickle — there is no REPL state to snap."""
        writer = ReplSnapshotWriter(project_dir=tmp_path)
        iteration = RLMIteration(
            prompt="p",
            response="thinking",
            code_blocks=[],
            final_answer=None,
            iteration_time=0.5,
        )
        writer.write(iteration, 1)
        assert (tmp_path / "iterations" / "iteration_0001.json").exists()
        # No prior state and a no-code iteration → no pickle written.
        assert not (tmp_path / "repl_state.pickle").exists()

    def test_no_code_iteration_preserves_prior_repl_state(self, tmp_path: Path):
        """A no-code iteration must NOT clobber a prior repl_state.pickle.

        Symptom: the RLM root interleaves pure-reasoning iterations (no code
        blocks) with code iterations; a no-code iteration was overwriting
        repl_state.pickle with an empty {} variables dict, destroying the
        accumulated REPL state (run 1: iter 9 had 52 vars, iter 10 wiped it).
        """
        writer = ReplSnapshotWriter(project_dir=tmp_path)
        # Iteration 1 — real code, real vars.
        writer.write(_make_iteration({"baseline_result": {"ok": True}}), 1)
        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            assert pickle.load(fh)["variables"]["baseline_result"] == {"ok": True}
        # Iteration 2 — pure reasoning, no code blocks.
        no_code = RLMIteration(
            prompt="p",
            response="thinking",
            code_blocks=[],
            final_answer=None,
            iteration_time=0.5,
        )
        writer.write(no_code, 2)
        # repl_state.pickle must STILL hold iteration 1's state.
        with open(tmp_path / "repl_state.pickle", "rb") as fh:
            payload = pickle.load(fh)
        assert payload["iteration"] == 1
        assert payload["variables"]["baseline_result"] == {"ok": True}


# ---------------------------------------------------------------------------
# Test: OpenResearchRLMLogger back-compat — snapshot_writer=None
# ---------------------------------------------------------------------------

class TestOpenResearchRLMLoggerBackCompat:
    """Issue #62 DC#4 — OpenResearchRLMLogger with snapshot_writer=None must still work."""

    def test_logger_without_snapshot_writer_logs_normally(self):
        """snapshot_writer=None (default) must not break emit or checkpoint."""
        emitted = []
        mock_checkpointer = MagicMock()

        logger = OpenResearchRLMLogger(
            emit=lambda e: emitted.append(e),
            checkpointer=mock_checkpointer,
        )
        iteration = _make_iteration({"score": 0.95})
        logger.log(iteration)

        assert len(emitted) == 1
        assert emitted[0]["event"] == "repl_iteration"
        mock_checkpointer.record.assert_called_once()

    def test_logger_with_snapshot_writer_calls_write(self, tmp_path: Path):
        """When a snapshot_writer is provided, write() is called after record()."""
        emitted = []
        mock_checkpointer = MagicMock()
        writer = ReplSnapshotWriter(project_dir=tmp_path)

        logger = OpenResearchRLMLogger(
            emit=lambda e: emitted.append(e),
            checkpointer=mock_checkpointer,
            snapshot_writer=writer,
        )
        iteration = _make_iteration({"score": 0.95})
        logger.log(iteration)

        assert (tmp_path / "repl_state.pickle").exists()
        assert (tmp_path / "iterations" / "iteration_0001.json").exists()
