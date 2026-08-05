"""Tests for backend.services.knowledge.corpus.manifest — the ≤8KB mount."""

from __future__ import annotations

import json
from pathlib import Path

from backend.services.knowledge.corpus.manifest import (
    MAX_CONTEXT_BYTES,
    MAX_ENTRIES,
    build_prior_work_refs,
)


def _write_spec(project_dir: Path, papers: list[dict]) -> None:
    state = project_dir / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "literature_spec.json").write_text(
        json.dumps({"target": {"id": "arxiv:x"}, "papers": papers}), encoding="utf-8"
    )


def test_missing_spec_returns_empty(tmp_path: Path):
    assert build_prior_work_refs(tmp_path / "prj_none") == []


def test_corrupt_spec_returns_empty(tmp_path: Path):
    project = tmp_path / "prj_bad"
    (project / "rlm_state").mkdir(parents=True)
    (project / "rlm_state" / "literature_spec.json").write_text("{not json", encoding="utf-8")
    assert build_prior_work_refs(project) == []


def test_entries_carry_bounded_fields(tmp_path: Path):
    project = tmp_path / "prj_ok"
    _write_spec(
        project,
        [
            {
                "id": "arxiv:2101.00001",
                "title": "T" * 500,
                "year": 2021,
                "relation": "direct_ref",
                "abstract": "A" * 1000,
                "score": 3.2,
            }
        ],
    )
    entries = build_prior_work_refs(project)
    assert len(entries) == 1
    e = entries[0]
    assert e["id"] == "arxiv:2101.00001"
    assert len(e["title"]) <= 200
    assert len(e["abstract"]) <= 300
    assert e["relation"] == "direct_ref"
    assert e["year"] == 2021
    assert "score" not in e  # ranking internals stay out of the root's context


def test_ceiling_drops_abstracts_then_tail_entries(tmp_path: Path):
    project = tmp_path / "prj_big"
    papers = [
        {
            "id": f"arxiv:21{i:02d}.0000{i % 10}",
            "title": f"Long Paper Title Number {i} " + "x" * 150,
            "relation": "direct_ref",
            "abstract": "B" * 300,
        }
        for i in range(MAX_ENTRIES)
    ]
    _write_spec(project, papers)
    entries = build_prior_work_refs(project)
    assert entries, "trimming must never empty a non-empty spec entirely"
    size = len(json.dumps(entries, ensure_ascii=False).encode("utf-8"))
    assert size <= MAX_CONTEXT_BYTES
    # Rank order preserved: the first (highest-ranked) entry survives trimming.
    assert entries[0]["id"] == papers[0]["id"]


def test_more_than_max_entries_is_capped(tmp_path: Path):
    project = tmp_path / "prj_many"
    papers = [
        {"id": f"arxiv:2200.{i:05d}", "title": f"P{i}", "relation": "search"}
        for i in range(MAX_ENTRIES + 30)
    ]
    _write_spec(project, papers)
    assert len(build_prior_work_refs(project)) <= MAX_ENTRIES
