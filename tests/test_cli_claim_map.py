"""Tests for backend.cli._build_workspace_claim_map.

Locks in: RLM mode with paper_text variable → one entry, full untruncated text;
SDK mode → per-variable entries truncated to 600 chars; RLM mode with no
paper_text variable → fallback per-variable entries, un-truncated.
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
# Test 1: RLM mode + paper_text dict → one entry, full untruncated text
# ---------------------------------------------------------------------------


def test_rlm_mode_paper_text_dict_full_text():
    """RLM mode with paper_text dict value returns one entry with the full text."""
    long_text = "X" * 5000
    variables = {
        "paper_text": _Cited({"text": long_text, "project_id": "prj_abc"}),
        "other_var": _Cited("other value"),
    }
    result = _build_workspace_claim_map(variables, "prj_abc", "rlm")

    assert result["project_id"] == "prj_abc"
    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert entry["title"] == "paper_text"
    assert entry["excerpt"] == long_text
    assert "..." not in entry["excerpt"], "text must not be truncated in RLM mode"


# ---------------------------------------------------------------------------
# Test 2: RLM mode + paper_text string value → one entry, full text
# ---------------------------------------------------------------------------


def test_rlm_mode_paper_text_string_value():
    """RLM mode with a bare string paper_text value also returns full text."""
    long_text = "Y" * 5000
    variables = {"paper_text": _Cited(long_text)}
    result = _build_workspace_claim_map(variables, "prj_xyz", "rlm")

    assert len(result["entries"]) == 1
    assert result["entries"][0]["excerpt"] == long_text


# ---------------------------------------------------------------------------
# Test 3: SDK mode → per-variable entries, each truncated to 600 chars
# ---------------------------------------------------------------------------


def test_sdk_mode_truncates_long_values():
    """SDK mode truncates each excerpt to 600 chars with trailing '...'."""
    long_text = "Z" * 5000
    variables = {
        "paper_text": _Cited(long_text),
        "short_var": _Cited("short"),
    }
    result = _build_workspace_claim_map(variables, "prj_sdk", "sdk")

    assert result["project_id"] == "prj_sdk"
    # One entry per variable
    assert len(result["entries"]) == 2

    excerpts = {e["title"]: e["excerpt"] for e in result["entries"]}
    # Long text truncated
    assert len(excerpts["paper_text"]) == 603  # 600 chars + "..."
    assert excerpts["paper_text"].endswith("...")
    # Short text unchanged
    assert excerpts["short_var"] == "short"


# ---------------------------------------------------------------------------
# Test 4: Offline mode → same truncation behavior as SDK
# ---------------------------------------------------------------------------


def test_offline_mode_truncates_like_sdk():
    """Offline mode is byte-identical to SDK: truncated at 600 chars."""
    long_text = "A" * 700
    variables = {"var1": _Cited(long_text)}
    result = _build_workspace_claim_map(variables, "prj_off", "offline")

    assert len(result["entries"]) == 1
    assert result["entries"][0]["excerpt"].endswith("...")
    assert len(result["entries"][0]["excerpt"]) == 603


# ---------------------------------------------------------------------------
# Test 5: RLM mode + no paper_text variable → fallback, un-truncated
# ---------------------------------------------------------------------------


def test_rlm_mode_no_paper_text_fallback_untruncated():
    """RLM mode without paper_text falls back to one entry per variable, not truncated."""
    long_text = "B" * 5000
    variables = {
        "claim_map": _Cited(long_text),
        "metadata": _Cited("meta"),
    }
    result = _build_workspace_claim_map(variables, "prj_fb", "rlm")

    assert result["project_id"] == "prj_fb"
    # Fallback: one entry per variable
    assert len(result["entries"]) == 2

    excerpts = {e["title"]: e["excerpt"] for e in result["entries"]}
    # Long text NOT truncated in RLM fallback
    assert excerpts["claim_map"] == long_text
    assert "..." not in excerpts["claim_map"]


# ---------------------------------------------------------------------------
# Test 6: SDK mode with dict value → json.dumps, truncated
# ---------------------------------------------------------------------------


def test_sdk_mode_dict_value_json_dumped_and_truncated():
    """SDK mode json.dumps dict values and then truncates to 600 chars."""
    big_dict = {"key": "V" * 5000}
    variables = {"data": _Cited(big_dict)}
    result = _build_workspace_claim_map(variables, "prj_dict", "sdk")

    entry = result["entries"][0]
    assert entry["excerpt"].endswith("...")
    assert len(entry["excerpt"]) == 603
