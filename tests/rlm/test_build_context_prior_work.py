"""OFF+ON pair for the prior_work_refs context mount + sentinel recursion.

Convention (tests/CLAUDE.md): every new flag ships a hermetic OFF+ON test
pair — OFF must be byte-identical to the prior baseline, ON must actually
change behavior. Flag: OPENRESEARCH_LITERATURE_CORPUS.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.run import _build_context, _corpus_sentinels

_CLAIM_MAP = {
    "project_id": "prj_x",
    "entries": [{"source_id": "s1", "title": "Abstract", "excerpt": "We study things."}],
}


def _write_spec(project_dir: Path) -> None:
    state = project_dir / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "literature_spec.json").write_text(
        json.dumps(
            {
                "target": {"id": "arxiv:2605.15155"},
                "papers": [
                    {
                        "id": "arxiv:2101.00001",
                        "title": "A Sufficiently Long Related Paper Title For Sentinels",
                        "relation": "direct_ref",
                        "abstract": (
                            "This abstract is corpus text that must never reach the "
                            "SSE stream verbatim under any circumstances at all."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_off_state_prior_work_refs_stays_reserved_empty(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_CORPUS", raising=False)
    project = tmp_path / "prj_x"
    _write_spec(project)  # spec present but flag off → must be ignored
    ctx = _build_context(_CLAIM_MAP, project_dir=project)
    assert ctx["prior_work_refs"] == []


def test_on_state_mounts_bounded_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    project = tmp_path / "prj_x"
    _write_spec(project)
    ctx = _build_context(_CLAIM_MAP, project_dir=project)
    assert len(ctx["prior_work_refs"]) == 1
    assert ctx["prior_work_refs"][0]["id"] == "arxiv:2101.00001"


def test_on_state_without_spec_degrades_to_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    ctx = _build_context(_CLAIM_MAP, project_dir=tmp_path / "prj_nospec")
    assert ctx["prior_work_refs"] == []


def test_corpus_sentinels_cover_nested_prior_work_strings(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    project = tmp_path / "prj_x"
    _write_spec(project)
    ctx = _build_context(_CLAIM_MAP, project_dir=project)

    sentinels = _corpus_sentinels(ctx)
    # Long nested strings (title, abstract) are sentinel-registered...
    assert any(s.startswith("A Sufficiently Long Related Paper Title") for s in sentinels)
    assert any(s.startswith("This abstract is corpus text") for s in sentinels)
    # ...but short nested strings are not (a year or "direct_ref" must never
    # cause blanket redaction of legitimate output).
    assert "direct_ref" not in sentinels
    assert "arxiv:2101.00001" not in sentinels


def test_corpus_sentinels_top_level_behavior_unchanged():
    ctx = {"paper_text": "x" * 500, "prior_work_refs": [], "repo_files": None}
    sentinels = _corpus_sentinels(ctx)
    assert sentinels == ["x" * 200]
