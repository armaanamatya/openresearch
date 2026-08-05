"""Tests for the cross-machine preserved-artifact discovery helper."""

from __future__ import annotations

from pathlib import Path

from tests.agents.preserved_artifacts import candidate_runs_roots, find_preserved_file


def test_override_root_wins_and_glob_matches_preserved_rename(tmp_path: Path, monkeypatch):
    preserved = tmp_path / "_preserved_vae_score_0.6457_prj_03271ba130d423fe"
    preserved.mkdir(parents=True)
    (preserved / "parsed_full_text.txt").write_text("real vae text", encoding="utf-8")
    monkeypatch.setenv("OPENRESEARCH_PRESERVED_RUNS_ROOT", str(tmp_path))

    found = find_preserved_file("*prj_03271ba130d423fe/parsed_full_text.txt")
    assert found is not None
    assert found.read_text(encoding="utf-8") == "real vae text"
    assert candidate_runs_roots()[0] == tmp_path


def test_plain_project_dir_also_matches(tmp_path: Path, monkeypatch):
    plain = tmp_path / "prj_03271ba130d423fe" / "code"
    plain.mkdir(parents=True)
    (plain / "train.py").write_text("print('observed')", encoding="utf-8")
    monkeypatch.setenv("OPENRESEARCH_PRESERVED_RUNS_ROOT", str(tmp_path))

    assert find_preserved_file("*prj_03271ba130d423fe/code/train.py") is not None


def test_no_match_anywhere_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_PRESERVED_RUNS_ROOT", str(tmp_path))
    assert find_preserved_file("*prj_ffffffffffffffff/parsed_full_text.txt") is None


def test_directories_never_returned(tmp_path: Path, monkeypatch):
    (tmp_path / "prj_03271ba130d423fe" / "parsed_full_text.txt").mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_PRESERVED_RUNS_ROOT", str(tmp_path))
    assert find_preserved_file("*prj_03271ba130d423fe/parsed_full_text.txt") is None
