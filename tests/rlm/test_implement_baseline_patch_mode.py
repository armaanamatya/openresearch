"""Tests for PR-ι.2 — patch-mode implement_baseline.

Covers:
1. _extract_violations_from_repair_context extracts contract_violations list.
2. _extract_violations_from_repair_context extracts preflight_violations list.
3. _extract_violations_from_repair_context falls back to error string.
4. _extract_violations_from_repair_context handles dicts with message/hint keys.
5. _apply_unified_diff applies a simple single-hunk diff correctly.
6. _apply_unified_diff raises ValueError when no hunks found.
7. _apply_unified_diff raises ValueError when hunk range out of range.
8. patch_mode_run_with_sdk returns (False, ...) when response has no diff fence.
9. patch_mode_run_with_sdk returns (True, patched) when response has a valid diff.
10. Untouched lines preserved after diff apply.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Violation extraction
# ---------------------------------------------------------------------------

from backend.agents.baseline_implementation import (
    _extract_violations_from_repair_context,
    _apply_unified_diff,
)


def test_extract_contract_violations_list() -> None:
    rc = {"contract_violations": ["missing key 'adam_nll'", "wrong shape"]}
    result = _extract_violations_from_repair_context(rc)
    assert result == ["missing key 'adam_nll'", "wrong shape"]


def test_extract_preflight_violations_list() -> None:
    rc = {"preflight_violations": ["load_dataset('imdb') → use 'stanfordnlp/imdb'"]}
    result = _extract_violations_from_repair_context(rc)
    assert len(result) == 1
    assert "stanfordnlp" in result[0]


def test_extract_preflight_violation_dicts() -> None:
    rc = {
        "preflight_violations": [
            {"message": "wrong dataset", "location": "train.py:486"},
            {"hint": "use torch.device('cuda')", "file": "train.py:12"},
        ]
    }
    result = _extract_violations_from_repair_context(rc)
    assert len(result) == 2
    assert "train.py:486" in result[0]
    assert "train.py:12" in result[1]


def test_extract_falls_back_to_error_string() -> None:
    rc = {"error": "AttributeError: model has no reparameterize"}
    result = _extract_violations_from_repair_context(rc)
    assert len(result) == 1
    assert "reparameterize" in result[0]


def test_extract_empty_repair_context() -> None:
    result = _extract_violations_from_repair_context({})
    assert result == []


def test_extract_both_keys_combined() -> None:
    rc = {
        "contract_violations": ["missing key"],
        "preflight_violations": ["wrong dataset"],
    }
    result = _extract_violations_from_repair_context(rc)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _apply_unified_diff
# ---------------------------------------------------------------------------


def _make_file(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_apply_unified_diff_simple_replacement() -> None:
    original = _make_file(
        "line1",
        "BAD_LINE",
        "line3",
    )
    diff = (
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-BAD_LINE\n"
        "+GOOD_LINE\n"
    )
    result = _apply_unified_diff(original, diff)
    assert "GOOD_LINE" in result
    assert "BAD_LINE" not in result
    assert "line1" in result
    assert "line3" in result


def test_apply_unified_diff_preserves_untouched_lines() -> None:
    original = _make_file(
        "import torch",
        "import os",
        "dataset = load_dataset('imdb')",
        "model = MyModel()",
    )
    diff = (
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -3,1 +3,1 @@\n"
        "-dataset = load_dataset('imdb')\n"
        "+dataset = load_dataset('stanfordnlp/imdb')\n"
    )
    result = _apply_unified_diff(original, diff)
    # Fixed line
    assert "stanfordnlp/imdb" in result
    assert "load_dataset('imdb')" not in result
    # Untouched lines preserved
    assert "import torch" in result
    assert "import os" in result
    assert "model = MyModel()" in result


def test_apply_unified_diff_no_hunks_raises() -> None:
    original = "some code\n"
    diff = "--- a/train.py\n+++ b/train.py\n"  # no @@ hunk
    with pytest.raises(ValueError, match="no hunks found"):
        _apply_unified_diff(original, diff)


def test_apply_unified_diff_out_of_range_raises() -> None:
    original = "line1\nline2\n"
    diff = (
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -50,1 +50,1 @@\n"
        "-bad\n"
        "+good\n"
    )
    with pytest.raises(ValueError):
        _apply_unified_diff(original, diff)


def test_apply_unified_diff_addition_only() -> None:
    """Adding lines without removing any."""
    original = "line1\nline3\n"
    diff = (
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -1,1 +1,2 @@\n"
        " line1\n"
        "+line2\n"
    )
    result = _apply_unified_diff(original, diff)
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result


# ---------------------------------------------------------------------------
# patch_mode_run_with_sdk — mock the LLM call
# ---------------------------------------------------------------------------


import asyncio
from unittest.mock import patch as mock_patch
from pathlib import Path
import tempfile


async def _run_patch_mode(
    project_id: str,
    runs_root: Path,
    prior_train_py: str,
    violations: list[str],
    repair_context: dict,
    response_text: str,
) -> tuple[bool, str]:
    """Helper that mocks collect_agent_text and runs patch_mode_run_with_sdk."""
    from backend.agents.baseline_implementation import patch_mode_run_with_sdk

    # create required directory structure
    (runs_root / project_id / "code").mkdir(parents=True, exist_ok=True)

    collected: list[str] = []

    async def _fake_collect(agent_name, prompt, **kwargs):
        collected.append(response_text)
        return response_text

    with mock_patch(
        "backend.agents.runtime.invoke.collect_agent_text",
        side_effect=_fake_collect,
    ):
        return await patch_mode_run_with_sdk(
            project_id,
            runs_root,
            prior_train_py,
            violations,
            repair_context,
        )


def test_patch_mode_no_diff_in_response() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_root = Path(tmpdir)
        success, result = asyncio.run(_run_patch_mode(
            "prj_test",
            runs_root,
            prior_train_py="import os\n",
            violations=["wrong dataset"],
            repair_context={"error": "bad"},
            response_text="I suggest changing the dataset name. Here is the fix: just update line 5.",
        ))
    assert success is False
    assert "no unified diff" in result


def test_patch_mode_valid_diff_applied() -> None:
    prior = "import os\ndataset = load_dataset('imdb')\nmodel = MyModel()\n"
    diff_block = (
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-dataset = load_dataset('imdb')\n"
        "+dataset = load_dataset('stanfordnlp/imdb')\n"
    )
    response = f"Here is the minimal fix:\n\n```diff\n{diff_block}\n```\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        runs_root = Path(tmpdir)
        success, result = asyncio.run(_run_patch_mode(
            "prj_test",
            runs_root,
            prior_train_py=prior,
            violations=["load_dataset('imdb') → use 'stanfordnlp/imdb'"],
            repair_context={"preflight_violations": ["wrong dataset"]},
            response_text=response,
        ))
    assert success is True
    assert "stanfordnlp/imdb" in result
    assert "load_dataset('imdb')" not in result
    assert "import os" in result
    assert "model = MyModel()" in result


def test_patch_mode_untouched_lines_preserved() -> None:
    """Lines not mentioned in the diff survive unchanged."""
    prior = "\n".join([f"line{i}" for i in range(1, 21)]) + "\n"
    diff_block = (
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -10,1 +10,1 @@\n"
        "-line10\n"
        "+FIXED_LINE10\n"
    )
    response = f"Fix:\n```diff\n{diff_block}\n```"

    with tempfile.TemporaryDirectory() as tmpdir:
        runs_root = Path(tmpdir)
        success, result = asyncio.run(_run_patch_mode(
            "prj_test",
            runs_root,
            prior_train_py=prior,
            violations=["line10 is wrong"],
            repair_context={"contract_violations": ["line10 is wrong"]},
            response_text=response,
        ))
    assert success is True
    assert "FIXED_LINE10" in result
    assert "line10" not in result
    # Untouched lines
    for i in [1, 5, 15, 20]:
        assert f"line{i}" in result
