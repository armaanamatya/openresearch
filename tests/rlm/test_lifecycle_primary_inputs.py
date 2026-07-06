"""T9: gate the OPENRESEARCH_LIFECYCLE_PRIMARY branch on ready inputs.

_primary_inputs_ready(tools, paper_text, rubric_spec) validates the inputs
run_lifecycle_primary needs; a missing input must fall through to the normal
loop (loud run_warning), never silently no-op.
"""
import backend.agents.rlm.run as run_mod


def test_primary_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_PRIMARY", "1")
    assert run_mod._lifecycle_primary_enabled() is True
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_PRIMARY", "0")
    assert run_mod._lifecycle_primary_enabled() is False
    monkeypatch.delenv("OPENRESEARCH_LIFECYCLE_PRIMARY", raising=False)
    assert run_mod._lifecycle_primary_enabled() is False  # default OFF (byte-identical)


def test_primary_requires_inputs(monkeypatch):
    # A helper that validates inputs and returns a reason on missing ones.
    ok, reason = run_mod._primary_inputs_ready(tools={"implement_baseline": lambda: None},
                                               paper_text="abc", rubric_spec={"leaves": []})
    assert ok and reason is None
    ok, reason = run_mod._primary_inputs_ready(tools={}, paper_text="", rubric_spec=None)
    assert not ok and reason


def test_primary_requires_inputs_partial_missing():
    """Only tools missing -> still not ready, reason names it."""
    ok, reason = run_mod._primary_inputs_ready(
        tools={}, paper_text="abc", rubric_spec={"leaves": []}
    )
    assert not ok
    assert "tools" in reason


def test_primary_requires_inputs_paper_text_missing():
    ok, reason = run_mod._primary_inputs_ready(
        tools={"x": 1}, paper_text="", rubric_spec={"leaves": []}
    )
    assert not ok
    assert "paper_text" in reason


def test_primary_requires_inputs_rubric_spec_missing():
    ok, reason = run_mod._primary_inputs_ready(
        tools={"x": 1}, paper_text="abc", rubric_spec={}
    )
    assert not ok
    assert "rubric_spec" in reason
