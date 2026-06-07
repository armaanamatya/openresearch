"""Tests for backend.agents.rlm.sse_bridge.

Critical tests:
- C2 regression: the paper corpus sentinel NEVER appears in sanitize_iteration output.
- stdout/stderr reduced to metadata.
- response bounding to ≤4 000 chars.
- OpenResearchRLMLogger.log() emits and checkpoints, does NOT call super().log().
- make_emit serializes via a threading.Lock.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rlm.core.types import CodeBlock, REPLResult, RLMIteration

from backend.agents.rlm.sse_bridge import (
    OpenResearchRLMLogger,
    build_candidate_outcome_event,
    build_candidate_proposed_event,
    build_rubric_score_event,
    build_run_complete_event,
    build_sub_rlm_complete_event,
    build_sub_rlm_spawned_event,
    make_emit,
    make_on_subcall_complete,
    make_on_subcall_start,
    sanitize_iteration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORPUS_SENTINEL = "PAPER_CORPUS_SENTINEL_xyzzy_DO_NOT_LEAK_abcdefg"


def _make_repl_result(
    *,
    stdout: str = "ok\n",
    stderr: str = "",
    locals_: dict | None = None,
    rlm_calls: list | None = None,
) -> REPLResult:
    """Build a REPLResult with the given locals."""
    return REPLResult(
        stdout=stdout,
        stderr=stderr,
        locals=locals_ or {},
        execution_time=0.1,
        rlm_calls=rlm_calls or [],
    )


def _make_iteration(
    *,
    response: str = "The root model reasoning text.",
    code_blocks: list[CodeBlock] | None = None,
    iteration_time: float = 1.0,
) -> RLMIteration:
    return RLMIteration(
        prompt={"role": "user", "content": "reproduce the paper"},
        response=response,
        code_blocks=code_blocks or [],
        final_answer=None,
        iteration_time=iteration_time,
    )


# ---------------------------------------------------------------------------
# C2 regression test — the single most important test in this file
# ---------------------------------------------------------------------------

class TestC2CorpusLeak:
    """The corpus sentinel must NEVER appear anywhere in sanitize_iteration output."""

    def _assert_no_sentinel(self, obj: Any, path: str = "") -> None:
        """Recursively assert the sentinel string is not present in obj."""
        if isinstance(obj, str):
            assert CORPUS_SENTINEL not in obj, (
                f"Corpus sentinel leaked at path {path!r}: {obj[:200]!r}"
            )
        elif isinstance(obj, dict):
            for k, v in obj.items():
                # Key names themselves must not contain the sentinel
                assert CORPUS_SENTINEL not in str(k), (
                    f"Corpus sentinel leaked in key at path {path!r}: {k!r}"
                )
                self._assert_no_sentinel(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self._assert_no_sentinel(item, f"{path}[{i}]")

    def test_context_locals_value_never_leaks(self):
        """An RLMIteration with `context` in locals must produce no sentinel in output."""
        large_corpus = CORPUS_SENTINEL + ("X" * 10_000)
        result = _make_repl_result(
            locals_={
                "context": large_corpus,            # the paper corpus — must be excluded
                "claim_map": {"title": "BERT"},     # a primitive output — values dropped
                "x": 42,                            # a plain variable
            }
        )
        block = CodeBlock(code="claim_map = understand_section(context['abstract'])", result=result)
        iteration = _make_iteration(code_blocks=[block])

        clean = sanitize_iteration(iteration, 1)

        # Recursively assert no sentinel anywhere in the output
        self._assert_no_sentinel(clean)

    def test_context_key_itself_excluded(self):
        """The key 'context' must not appear as a variable name in vars."""
        result = _make_repl_result(
            locals_={
                "context": CORPUS_SENTINEL,
                "context_window": CORPUS_SENTINEL,
                "contextual_data": CORPUS_SENTINEL,
            }
        )
        block = CodeBlock(code="pass", result=result)
        iteration = _make_iteration(code_blocks=[block])

        clean = sanitize_iteration(iteration, 1)

        # No block's 'vars' should have a key starting with 'context'
        for cb in clean["code_blocks"]:
            for var_name in cb["vars"]:
                assert not var_name.startswith("context"), (
                    f"Variable name starting with 'context' leaked: {var_name!r}"
                )

    def test_underscore_keys_excluded(self):
        """Keys starting with _ are excluded from vars."""
        result = _make_repl_result(
            locals_={
                "_builtins_": "...",
                "__doc__": None,
                "visible": "hello",
            }
        )
        block = CodeBlock(code="pass", result=result)
        iteration = _make_iteration(code_blocks=[block])

        clean = sanitize_iteration(iteration, 1)
        assert len(clean["code_blocks"]) == 1
        vars_ = clean["code_blocks"][0]["vars"]
        assert "visible" in vars_
        assert "_builtins_" not in vars_
        assert "__doc__" not in vars_

    def test_prompt_dropped(self):
        """iteration.prompt (full message history) is never in the output."""
        iteration = _make_iteration(
            response="reasoning here",
            code_blocks=[],
        )
        clean = sanitize_iteration(iteration, 1)
        # 'prompt' key must not exist in the output
        assert "prompt" not in clean

    def test_final_answer_dropped(self):
        """iteration.final_answer is never in the output."""
        it = RLMIteration(
            prompt="some prompt",
            response="reasoning",
            code_blocks=[],
            final_answer="this should not appear",
            iteration_time=0.5,
        )
        clean = sanitize_iteration(it, 1)
        assert "final_answer" not in clean


# ---------------------------------------------------------------------------
# stdout/stderr → metadata
# ---------------------------------------------------------------------------

class TestStdoutStderrMetadata:

    def test_stdout_reduced_to_metadata(self):
        long_stdout = "line output\n" * 500
        result = _make_repl_result(stdout=long_stdout)
        block = CodeBlock(code="print('hi')", result=result)
        iteration = _make_iteration(code_blocks=[block])
        clean = sanitize_iteration(iteration, 1)

        meta = clean["code_blocks"][0]["stdout_meta"]
        assert meta["length"] == len(long_stdout)
        assert len(meta["prefix"]) <= 200
        assert meta["prefix"] == long_stdout[:200]
        assert isinstance(meta["has_traceback"], bool)

    def test_traceback_detected(self):
        stderr = "Traceback (most recent call last):\n  File ...\nValueError: bad\n"
        result = _make_repl_result(stderr=stderr)
        block = CodeBlock(code="raise ValueError('bad')", result=result)
        iteration = _make_iteration(code_blocks=[block])
        clean = sanitize_iteration(iteration, 1)
        assert clean["code_blocks"][0]["stderr_meta"]["has_traceback"] is True

    def test_no_traceback(self):
        result = _make_repl_result(stdout="all good")
        block = CodeBlock(code="x = 1", result=result)
        iteration = _make_iteration(code_blocks=[block])
        clean = sanitize_iteration(iteration, 1)
        assert clean["code_blocks"][0]["stdout_meta"]["has_traceback"] is False

    def test_empty_stdout_stderr(self):
        result = _make_repl_result(stdout="", stderr="")
        block = CodeBlock(code="pass", result=result)
        iteration = _make_iteration(code_blocks=[block])
        clean = sanitize_iteration(iteration, 1)
        cb = clean["code_blocks"][0]
        assert cb["stdout_meta"]["length"] == 0
        assert cb["stderr_meta"]["length"] == 0


# ---------------------------------------------------------------------------
# Response bounding
# ---------------------------------------------------------------------------

class TestResponseBounding:

    def test_response_bounded_to_4000(self):
        long_response = "A" * 5000
        iteration = _make_iteration(response=long_response)
        clean = sanitize_iteration(iteration, 1)
        assert len(clean["response"]) == 4000

    def test_response_short_unchanged(self):
        short = "short reasoning"
        iteration = _make_iteration(response=short)
        clean = sanitize_iteration(iteration, 1)
        assert clean["response"] == short

    def test_response_exactly_4000_unchanged(self):
        resp = "B" * 4000
        iteration = _make_iteration(response=resp)
        clean = sanitize_iteration(iteration, 1)
        assert len(clean["response"]) == 4000


# ---------------------------------------------------------------------------
# sanitize_iteration output shape
# ---------------------------------------------------------------------------

class TestSanitizeIterationShape:

    def test_output_keys(self):
        iteration = _make_iteration()
        clean = sanitize_iteration(iteration, 3)
        assert set(clean.keys()) == {"iteration", "response", "code_blocks", "sub_calls", "timing"}

    def test_iteration_index(self):
        iteration = _make_iteration()
        clean = sanitize_iteration(iteration, 7)
        assert clean["iteration"] == 7

    def test_timing_preserved(self):
        iteration = _make_iteration(iteration_time=2.5)
        clean = sanitize_iteration(iteration, 1)
        assert clean["timing"] == 2.5

    def test_vars_shape(self):
        result = _make_repl_result(locals_={"score": 0.95, "model": "bert"})
        block = CodeBlock(code="score = verify()", result=result)
        iteration = _make_iteration(code_blocks=[block])
        clean = sanitize_iteration(iteration, 1)
        vars_ = clean["code_blocks"][0]["vars"]
        assert "score" in vars_
        assert "model" in vars_
        assert vars_["score"]["type"] == "float"
        assert vars_["model"]["type"] == "str"
        # values are never present — only type + size
        assert "value" not in vars_["score"]

    def test_sub_calls_counted(self):
        # Build a mock rlm_call object
        mock_call = MagicMock()
        result = _make_repl_result(rlm_calls=[mock_call, mock_call])
        block = CodeBlock(code="rlm_query('something')", result=result)
        iteration = _make_iteration(code_blocks=[block])
        clean = sanitize_iteration(iteration, 1)
        assert clean["sub_calls"] == 2
        assert clean["code_blocks"][0]["sub_calls"] == 2

    def test_multiple_blocks(self):
        b1 = CodeBlock(code="a = 1", result=_make_repl_result())
        b2 = CodeBlock(code="b = 2", result=_make_repl_result())
        iteration = _make_iteration(code_blocks=[b1, b2])
        clean = sanitize_iteration(iteration, 1)
        assert len(clean["code_blocks"]) == 2
        assert clean["code_blocks"][0]["code"] == "a = 1"
        assert clean["code_blocks"][1]["code"] == "b = 2"


# ---------------------------------------------------------------------------
# OpenResearchRLMLogger
# ---------------------------------------------------------------------------

class TestOpenResearchRLMLogger:

    def _make_logger(self):
        emitted = []
        checkpointed = []

        emit = lambda event: emitted.append(event)  # noqa: E731

        mock_checkpointer = MagicMock()
        mock_checkpointer.record.side_effect = lambda clean: checkpointed.append(clean)

        logger = OpenResearchRLMLogger(emit=emit, checkpointer=mock_checkpointer)
        return logger, emitted, checkpointed, mock_checkpointer

    def test_log_emits_repl_iteration_event(self):
        logger, emitted, _, _ = self._make_logger()
        iteration = _make_iteration(
            response="model thinking",
            code_blocks=[CodeBlock(code="x=1", result=_make_repl_result())],
        )
        logger.log(iteration)
        assert len(emitted) == 1
        assert emitted[0]["event"] == "repl_iteration"
        assert emitted[0]["iteration"] == 1

    def test_log_calls_checkpointer_record(self):
        logger, _, checkpointed, mock_cp = self._make_logger()
        iteration = _make_iteration()
        logger.log(iteration)
        mock_cp.record.assert_called_once()
        clean = checkpointed[0]
        assert clean["iteration"] == 1

    def test_log_does_not_call_super_log(self):
        """OpenResearchRLMLogger.log() must NEVER call super().log() — that would
        capture the raw RLMIteration (corpus) in the base class's _iterations list."""
        logger, _, _, _ = self._make_logger()

        with patch.object(type(logger).__bases__[0], "log") as mock_super_log:
            iteration = _make_iteration()
            logger.log(iteration)
            mock_super_log.assert_not_called()

    def test_log_does_not_populate_base_iterations(self):
        """The base _iterations list must remain empty — no raw data stored."""
        logger, _, _, _ = self._make_logger()
        iteration = _make_iteration()
        logger.log(iteration)
        # Base class _iterations should stay empty
        assert logger._iterations == []

    def test_index_increments_per_call(self):
        logger, emitted, _, _ = self._make_logger()
        for _ in range(3):
            logger.log(_make_iteration())
        assert [e["iteration"] for e in emitted] == [1, 2, 3]

    def test_emitted_event_is_corpus_free(self):
        """Even if locals contains the corpus, the emitted event is clean."""
        large_corpus = CORPUS_SENTINEL + "X" * 5000
        result = _make_repl_result(locals_={"context": large_corpus})
        block = CodeBlock(code="pass", result=result)
        iteration = _make_iteration(code_blocks=[block])

        logger, emitted, _, _ = self._make_logger()
        logger.log(iteration)

        # Convert the emitted event to a JSON string and check
        event_str = json.dumps(emitted[0])
        assert CORPUS_SENTINEL not in event_str


# ---------------------------------------------------------------------------
# make_emit — lock serialization
# ---------------------------------------------------------------------------

class TestMakeEmit:

    def test_make_emit_calls_dashboard_emit(self, tmp_path):
        """make_emit forwards the event to DashboardEmitter._emit."""
        from backend.agents.dashboard_emitter import DashboardEmitter
        dashboard = DashboardEmitter("proj1", tmp_path)
        emit = make_emit(dashboard)
        event = {"event": "test", "value": 42}
        emit(event)

        # Check the JSONL file was written
        events_file = tmp_path / "proj1" / "dashboard_events.jsonl"
        assert events_file.exists()
        line = events_file.read_text().strip()
        loaded = json.loads(line)
        assert loaded["event"] == "test"
        assert loaded["value"] == 42

    def test_make_emit_serializes_via_lock(self, tmp_path):
        """Two threads calling emit simultaneously must not interleave writes."""
        from backend.agents.dashboard_emitter import DashboardEmitter
        dashboard = DashboardEmitter("proj2", tmp_path)
        emit = make_emit(dashboard)

        errors: list[Exception] = []
        call_order: list[int] = []

        def writer(n: int):
            try:
                for i in range(20):
                    emit({"event": "test", "thread": n, "seq": i})
                    call_order.append(n)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"

        # All 40 events should be in the file
        events_file = tmp_path / "proj2" / "dashboard_events.jsonl"
        lines = events_file.read_text().strip().splitlines()
        assert len(lines) == 40

    def test_make_emit_each_call_independent(self, tmp_path):
        """Each make_emit() call produces an independent closure with its own lock."""
        from backend.agents.dashboard_emitter import DashboardEmitter
        d1 = DashboardEmitter("p1", tmp_path)
        d2 = DashboardEmitter("p2", tmp_path)
        emit1 = make_emit(d1)
        emit2 = make_emit(d2)
        emit1({"event": "e1"})
        emit2({"event": "e2"})

        f1 = json.loads((tmp_path / "p1" / "dashboard_events.jsonl").read_text())
        f2 = json.loads((tmp_path / "p2" / "dashboard_events.jsonl").read_text())
        assert f1["event"] == "e1"
        assert f2["event"] == "e2"


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------

class TestEventBuilders:

    def test_sub_rlm_spawned_event(self):
        event = build_sub_rlm_spawned_event(1, "gpt-5", "A" * 300)
        assert event["event"] == "sub_rlm_spawned"
        assert event["depth"] == 1
        assert event["model"] == "gpt-5"
        assert len(event["prompt_preview"]) == 200

    def test_sub_rlm_complete_event(self):
        event = build_sub_rlm_complete_event(2, "gpt-5-mini", 1.5, None)
        assert event["event"] == "sub_rlm_complete"
        assert event["depth"] == 2
        assert event["duration_ms"] == 1500
        assert event["error"] is None

    def test_sub_rlm_complete_with_error(self):
        event = build_sub_rlm_complete_event(1, "gpt-5", 0.0, "timeout")
        assert event["error"] == "timeout"

    def test_run_complete_event(self):
        event = build_run_complete_event(
            status="completed",
            iterations=5,
            rubric_score=0.85,
            cost_usd=0.12,
            final_report_path="/tmp/report.json",
        )
        assert event["event"] == "run_complete"
        assert event["status"] == "completed"
        assert event["iterations"] == 5
        assert event["rubric_score"] == 0.85


# ---------------------------------------------------------------------------
# on_subcall_* callback builders
# ---------------------------------------------------------------------------

class TestSubcallCallbacks:

    def test_on_subcall_start_emits_event(self):
        emitted = []
        emit = lambda e: emitted.append(e)  # noqa: E731
        cb = make_on_subcall_start(emit)
        cb(1, "gpt-5", "hello world")
        assert len(emitted) == 1
        assert emitted[0]["event"] == "sub_rlm_spawned"

    def test_on_subcall_complete_emits_event(self):
        emitted = []
        emit = lambda e: emitted.append(e)  # noqa: E731
        cb = make_on_subcall_complete(emit)
        cb(1, "gpt-5", 2.0, None)
        assert len(emitted) == 1
        assert emitted[0]["event"] == "sub_rlm_complete"
        assert emitted[0]["duration_ms"] == 2000


# ---------------------------------------------------------------------------
# T10: M-REDACT egress — response must be redacted
# ---------------------------------------------------------------------------


def test_sanitize_iteration_redacts_corpus_in_response():
    """Symptom: up to 4000 chars of paper corpus per iteration leak via response.

    sanitize_iteration redacted stdout/stderr prefixes via _stream_metadata
    but NOT the response — the root's natural-language response can quote
    paper slices it read via REPL code (e.g. print(context['paper_text'][:N])),
    and that response goes verbatim into every repl_iteration event
    (review I2 / T10). Verify: a response containing a corpus sentinel
    is redacted to [REDACTED] before egress.
    """
    sentinel = "x" * 200
    result = _make_repl_result()
    block = CodeBlock(code="", result=result)
    iteration = _make_iteration(
        response=f"I read this in the paper: {sentinel} and then ...",
        code_blocks=[block],
    )

    clean = sanitize_iteration(iteration, index=1, sentinels=[sentinel])
    assert sentinel not in clean["response"]
    assert "[REDACTED]" in clean["response"]


# ---------------------------------------------------------------------------
# New event builders — candidate_proposed, candidate_outcome, rubric_score
# ---------------------------------------------------------------------------

class TestNewEventBuilders:

    def test_build_candidate_proposed_event_shape(self):
        ev = build_candidate_proposed_event(
            iteration=3, round=1, parent_id="baseline",
            candidate={"id": "c1", "title": "tune lr", "category": "optimizer",
                       "description": "raise the learning rate", "reasoning": "loss plateaued"},
        )
        assert ev["event"] == "candidate_proposed"
        assert ev["iteration"] == 3 and ev["round"] == 1
        assert ev["parent_id"] == "baseline"
        assert set(ev["candidate"]) == {"id", "title", "category", "description", "reasoning"}
        assert "timestamp" in ev

    def test_build_candidate_proposed_event_no_parent_id(self):
        ev = build_candidate_proposed_event(
            iteration=1, round=1,
            candidate={"id": "c1", "title": "tune lr", "category": "optimizer",
                       "description": "raise the learning rate", "reasoning": "loss plateaued"},
        )
        assert ev["event"] == "candidate_proposed"
        # parent_id omitted when None — the key must not appear in the dict
        assert "parent_id" not in ev

    def test_build_candidate_outcome_event_shape(self):
        ev = build_candidate_outcome_event(
            iteration=5, candidate_id="c1", outcome="promoted", rubric_delta=0.08,
        )
        assert ev["event"] == "candidate_outcome"
        assert ev["candidate_id"] == "c1" and ev["outcome"] == "promoted"
        assert ev["rubric_delta"] == 0.08
        assert "timestamp" in ev

    def test_build_candidate_outcome_event_null_rubric_delta(self):
        ev = build_candidate_outcome_event(
            iteration=2, candidate_id="c2", outcome="failed", rubric_delta=None,
        )
        assert ev["rubric_delta"] is None

    def test_build_rubric_score_event_derives_area_status(self):
        ev = build_rubric_score_event(
            iteration=4, score=0.55, target=0.7,
            areas=[{"area": "method", "score": 0.8, "weight": 0.5},
                   {"area": "results", "score": 0.3, "weight": 0.5}],
        )
        assert ev["event"] == "rubric_score"
        assert ev["score"] == 0.55 and ev["target"] == 0.7
        statuses = {a["area"]: a["status"] for a in ev["areas"]}
        assert statuses == {"method": "pass", "results": "fail"}
        assert "timestamp" in ev

    def test_build_rubric_score_event_partial_status(self):
        ev = build_rubric_score_event(
            iteration=2, score=0.5, target=0.7,
            areas=[{"area": "setup", "score": 0.5, "weight": 1.0}],
        )
        statuses = {a["area"]: a["status"] for a in ev["areas"]}
        assert statuses == {"setup": "partial"}
