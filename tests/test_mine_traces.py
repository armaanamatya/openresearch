"""Tests for scripts/mine_traces.py — focused on the idempotency contract
that re-running the miner MUST NOT clobber human-authored narrative findings.

The aggregation logic itself is exercised by running the miner against the
real ``runs/`` corpus; this test suite pins the narrative-preservation
behavior specifically because losing the narrative on a re-run was a real
bug shipped briefly on 2026-05-27.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_miner():
    """Load scripts/mine_traces.py as a module — it's a script, not in a package."""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "mine_traces", repo_root / "scripts" / "mine_traces.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mine_traces"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_merge_writes_full_report_when_file_missing(tmp_path: Path) -> None:
    mt = _load_miner()
    out = tmp_path / "findings.md"
    new_report = "stats\n\n" + mt._HUMAN_SENTINEL + "\n\n## 8. Narrative findings\n\n_To be filled in after reviewing the stats above._\n"
    merged = mt._merge_with_existing(new_report, out)
    assert merged == new_report


def test_merge_preserves_narrative_when_sentinel_present(tmp_path: Path) -> None:
    mt = _load_miner()
    out = tmp_path / "findings.md"
    existing = (
        "OLD stats\n\n"
        + mt._HUMAN_SENTINEL
        + "\n\n## 8. Narrative findings\n\n### Finding 1 — my analysis\n\nThis is precious.\n"
    )
    out.write_text(existing)
    new_report = (
        "NEW stats\n\n"
        + mt._HUMAN_SENTINEL
        + "\n\n## 8. Narrative findings\n\n_To be filled in after reviewing the stats above._\n"
    )
    merged = mt._merge_with_existing(new_report, out)
    assert "NEW stats" in merged
    assert "OLD stats" not in merged
    assert "### Finding 1 — my analysis" in merged
    assert "This is precious." in merged
    assert "_To be filled in" not in merged


def test_merge_preserves_narrative_via_header_fallback(tmp_path: Path) -> None:
    """For files written by the OLD miner (no sentinel) — the header fallback path."""
    mt = _load_miner()
    out = tmp_path / "findings.md"
    existing = (
        "OLD stats no sentinel\n\n"
        "## 8. Narrative findings\n\n"
        "### Finding 1 — handcrafted\n\nMy careful analysis.\n"
    )
    out.write_text(existing)
    new_report = (
        "NEW stats\n\n"
        + mt._HUMAN_SENTINEL
        + "\n\n## 8. Narrative findings\n\n_To be filled in after reviewing the stats above._\n"
    )
    merged = mt._merge_with_existing(new_report, out)
    assert "NEW stats" in merged
    assert "### Finding 1 — handcrafted" in merged
    assert "My careful analysis." in merged


def test_merge_does_not_preserve_empty_narrative(tmp_path: Path) -> None:
    """If the existing §8 only contains the placeholder, the new template wins."""
    mt = _load_miner()
    out = tmp_path / "findings.md"
    existing = (
        "OLD stats\n\n"
        + mt._HUMAN_SENTINEL
        + "\n\n## 8. Narrative findings\n\n_To be filled in after reviewing the stats above._\n"
    )
    out.write_text(existing)
    new_report = (
        "NEW stats\n\n"
        + mt._HUMAN_SENTINEL
        + "\n\n## 8. Narrative findings\n\n_To be filled in after reviewing the stats above._\n"
    )
    merged = mt._merge_with_existing(new_report, out)
    # Both have the same placeholder; the splice path runs but result is the
    # new report (with the existing-identical placeholder body).
    assert "_To be filled in after reviewing the stats above._" in merged
    assert "NEW stats" in merged
