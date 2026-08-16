"""Phase 4 — advisory field-plausibility band tests.

Hermetic: a real ``CorpusStore`` over a tmp ``runs/_corpus/corpus.db`` (no
network, no LLM), a fabricated ``experiment_runs.jsonl``, and the flag set
per-test via monkeypatch (the conftest scrubs ``OPENRESEARCH_*``).
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.field_plausibility import (
    gather_plausibility_findings,
    run_field_plausibility,
)
from backend.services.knowledge.corpus.store import CorpusStore, corpus_root

FLAG = "OPENRESEARCH_FIELD_PLAUSIBILITY"


def _make_project(runs_root: Path, metrics: dict) -> Path:
    project_dir = runs_root / "prj_fieldband"
    project_dir.mkdir(parents=True)
    rows = [
        {"success": False, "metrics": {"per_model": {"m": {"alfworld": {"success_rate": 0.99}}}}},
        {"success": True, "metrics": metrics},
    ]
    (project_dir / "experiment_runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return project_dir


def _seed_corpus(runs_root: Path, values: list[float], *, dataset: str = "alfworld",
                 metric: str = "success_rate") -> None:
    store = CorpusStore(corpus_root(runs_root))
    store.initialize()
    for i, v in enumerate(values):
        store.put_result(
            f"arxiv:250{i}.0000{i}",
            method=f"method-{i}",
            dataset=dataset,
            metric=metric,
            value=v,
            span_quote=f"achieves {v} on ALFWorld",
        )
    store.close()


NESTED_OUTLIER = {"per_model": {"qwen-1.7b": {"alfworld": {"grpo": {"success_rate": 0.94}}}}}


# ---------------------------------------------------------------------------
# Off-state
# ---------------------------------------------------------------------------


def test_disabled_returns_empty_and_touches_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    project_dir = _make_project(tmp_path, NESTED_OUTLIER)
    _seed_corpus(tmp_path, [0.31, 0.35, 0.42])
    emitted: list[tuple[str, str]] = []
    findings = run_field_plausibility(
        project_dir, emit_warning=lambda c, m: emitted.append((c, m))
    )
    assert findings == []
    assert emitted == []


def test_disabled_never_creates_a_corpus(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    project_dir = _make_project(tmp_path, NESTED_OUTLIER)
    run_field_plausibility(project_dir)
    assert not corpus_root(tmp_path).exists()


def test_enabled_without_corpus_never_creates_one(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    project_dir = _make_project(tmp_path, NESTED_OUTLIER)
    assert run_field_plausibility(project_dir) == []
    assert not corpus_root(tmp_path).exists()


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def test_outlier_flagged_with_warning(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    project_dir = _make_project(tmp_path, NESTED_OUTLIER)
    _seed_corpus(tmp_path, [0.31, 0.35, 0.42])
    emitted: list[tuple[str, str]] = []
    findings = run_field_plausibility(
        project_dir, emit_warning=lambda c, m: emitted.append((c, m))
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.dataset == "alfworld" and f.metric == "success_rate"
    assert f.reproduced_value == 0.94
    assert emitted and emitted[0][0] == "metric_outside_field_band"
    assert "advisory" in emitted[0][1]


def test_in_band_value_not_flagged(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    metrics = {"per_model": {"qwen-1.7b": {"alfworld": {"grpo": {"success_rate": 0.40}}}}}
    project_dir = _make_project(tmp_path, metrics)
    _seed_corpus(tmp_path, [0.31, 0.35, 0.42])
    assert run_field_plausibility(project_dir) == []


def test_fewer_than_three_field_values_never_flagged(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    project_dir = _make_project(tmp_path, NESTED_OUTLIER)
    _seed_corpus(tmp_path, [0.31, 0.35])
    assert run_field_plausibility(project_dir) == []


def test_scale_ambiguity_resolved_in_runs_favor(tmp_path: Path, monkeypatch):
    """Field values in percent, reproduced value as a fraction: not flagged."""
    monkeypatch.setenv(FLAG, "1")
    metrics = {"per_model": {"m": {"alfworld": {"grpo": {"success_rate": 0.72}}}}}
    project_dir = _make_project(tmp_path, metrics)
    _seed_corpus(tmp_path, [70.0, 72.0, 74.0])
    assert run_field_plausibility(project_dir) == []


def test_tight_cluster_plausible_value_not_flagged(tmp_path: Path, monkeypatch):
    """|z|>3 alone (tight cluster) must NOT flag a value inside the ±20% band —
    the documented tightening of the plan's `or` to `and`."""
    monkeypatch.setenv(FLAG, "1")
    metrics = {"per_model": {"m": {"alfworld": {"grpo": {"success_rate": 72.0}}}}}
    project_dir = _make_project(tmp_path, metrics)
    _seed_corpus(tmp_path, [71.0, 71.1, 71.2])
    assert run_field_plausibility(project_dir) == []


def test_flat_dataset_metric_key_matches(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    project_dir = _make_project(tmp_path, {"alfworld_success_rate": 0.94})
    _seed_corpus(tmp_path, [0.31, 0.35, 0.42])
    findings = run_field_plausibility(project_dir)
    assert len(findings) == 1
    assert findings[0].metric_path == "alfworld_success_rate"


def test_unrelated_dataset_never_matches(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    metrics = {"per_model": {"m": {"webshop": {"grpo": {"success_rate": 0.94}}}}}
    project_dir = _make_project(tmp_path, metrics)
    _seed_corpus(tmp_path, [0.31, 0.35, 0.42])  # alfworld band only
    assert run_field_plausibility(project_dir) == []


def test_dataset_alias_matches(tmp_path: Path, monkeypatch):
    """Corpus says 'cifar-10', the run's path says 'cifar10' — curated aliases bridge."""
    monkeypatch.setenv(FLAG, "1")
    metrics = {"per_model": {"resnet": {"cifar10": {"base": {"accuracy": 94.0}}}}}
    project_dir = _make_project(tmp_path, metrics)
    _seed_corpus(tmp_path, [71.0, 73.5, 76.0], dataset="cifar-10", metric="accuracy")
    findings = run_field_plausibility(project_dir)
    assert len(findings) == 1
    assert findings[0].reproduced_value == 94.0


# ---------------------------------------------------------------------------
# Fail-soft + purity
# ---------------------------------------------------------------------------


def test_failsoft_on_corrupt_corpus_db(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    project_dir = _make_project(tmp_path, NESTED_OUTLIER)
    root = corpus_root(tmp_path)
    root.mkdir(parents=True)
    (root / "corpus.db").write_bytes(b"this is not a sqlite database")
    assert run_field_plausibility(project_dir) == []


def test_gather_is_pure_and_bounded():
    field_rows = {("alfworld", "success_rate"): [0.31, 0.35, 0.42]}
    findings = gather_plausibility_findings(NESTED_OUTLIER, field_rows)
    assert len(findings) == 1
    assert findings[0].band[0] < 0.31 and findings[0].band[1] > 0.42
    # Non-dict / junk inputs degrade to [] without raising.
    assert gather_plausibility_findings(None, field_rows) == []
    assert gather_plausibility_findings("nope", field_rows) == []
    assert gather_plausibility_findings(NESTED_OUTLIER, {}) == []
