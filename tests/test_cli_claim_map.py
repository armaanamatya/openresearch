"""Tests for backend.cli._build_workspace_claim_map.

Locks in: paper_text variable → one entry, full untruncated text; no
paper_text variable → fallback per-variable entries, un-truncated.
The RLM orchestrator is the only pipeline; SDK/offline truncating behavior
has been removed.
"""

from __future__ import annotations

from backend.cli import _build_workspace_claim_map


# ---------------------------------------------------------------------------
# Minimal Cited stub — mirrors the real Cited interface (.value attribute).
# ---------------------------------------------------------------------------

class _Cited:
    def __init__(self, value):
        self.value = value


# ---------------------------------------------------------------------------
# Test 1: paper_text dict → one entry, full untruncated text
# ---------------------------------------------------------------------------


def test_paper_text_dict_full_text():
    """paper_text dict value returns one entry with the full text."""
    long_text = "X" * 5000
    variables = {
        "paper_text": _Cited({"text": long_text, "project_id": "prj_abc"}),
        "other_var": _Cited("other value"),
    }
    result = _build_workspace_claim_map(variables, "prj_abc")

    assert result["project_id"] == "prj_abc"
    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert entry["title"] == "paper_text"
    assert entry["excerpt"] == long_text
    assert "..." not in entry["excerpt"], "text must not be truncated"


# ---------------------------------------------------------------------------
# Test 2: paper_text string value → one entry, full text
# ---------------------------------------------------------------------------


def test_paper_text_string_value():
    """A bare string paper_text value also returns full text."""
    long_text = "Y" * 5000
    variables = {"paper_text": _Cited(long_text)}
    result = _build_workspace_claim_map(variables, "prj_xyz")

    assert len(result["entries"]) == 1
    assert result["entries"][0]["excerpt"] == long_text


# ---------------------------------------------------------------------------
# Test 3: no paper_text variable → fallback, un-truncated
# ---------------------------------------------------------------------------


def test_no_paper_text_fallback_untruncated():
    """Without paper_text falls back to one entry per variable, not truncated."""
    long_text = "B" * 5000
    variables = {
        "claim_map": _Cited(long_text),
        "metadata": _Cited("meta"),
    }
    result = _build_workspace_claim_map(variables, "prj_fb")

    assert result["project_id"] == "prj_fb"
    # Fallback: one entry per variable
    assert len(result["entries"]) == 2

    excerpts = {e["title"]: e["excerpt"] for e in result["entries"]}
    # Long text NOT truncated in fallback
    assert excerpts["claim_map"] == long_text
    assert "..." not in excerpts["claim_map"]


def test_prefers_parsed_full_text_blob(tmp_path):
    """Reads parsed_full_text.txt (the parser's clean output) in preference
    to the workspace paper_text variable.

    Regression: the workspace paper_text variable is reassembled from indexed
    chunks and can lose content; the parser blob is the clean source of truth.
    """
    pid = "prj_blobtest"
    blob_dir = tmp_path / pid
    blob_dir.mkdir()
    (blob_dir / "parsed_full_text.txt").write_text(
        "CLEAN parser full text body.", encoding="utf-8"
    )
    # The workspace variable holds DIFFERENT (degraded) text — must be ignored.
    variables = {"paper_text": _Cited({"text": "degraded variable text"})}

    result = _build_workspace_claim_map(variables, pid, runs_root=tmp_path)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["excerpt"] == "CLEAN parser full text body."


def test_falls_back_to_variable_when_no_blob(tmp_path):
    """With no parsed_full_text.txt falls back to the paper_text variable."""
    pid = "prj_noblob"
    (tmp_path / pid).mkdir()
    variables = {"paper_text": _Cited({"text": "variable fallback text"})}
    result = _build_workspace_claim_map(variables, pid, runs_root=tmp_path)
    assert result["entries"][0]["excerpt"] == "variable fallback text"
