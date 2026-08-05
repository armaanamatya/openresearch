"""Phase 3 — runtime web expansion tests (hermetic: injected fakes, no network)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from backend.services.knowledge.corpus.store import CorpusStore, corpus_root
from backend.services.knowledge.corpus.web_expand import (
    MAX_RUNTIME_FETCHES,
    discover_papers,
    fetch_paper,
    fetches_remaining,
)

WEB_FLAG = "OPENRESEARCH_LITERATURE_WEB"
NET_FLAG = "OPENRESEARCH_LITERATURE_GROUNDING"


def _store(tmp_path: Path) -> CorpusStore:
    store = CorpusStore(corpus_root(tmp_path))
    store.initialize()
    return store


def _project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "prj_webx"
    (project_dir / "rlm_state").mkdir(parents=True)
    return project_dir


class _RaisingConnector:
    """A connector double that fails the test if any network path is taken."""

    def fetch(self, record_id):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("network path taken")

    def search(self, query, *, limit=10):  # pragma: no cover
        raise AssertionError("network path taken")


def _fake_download(url, *, runs_root, project_id, fetched_via):
    (Path(runs_root) / project_id / "raw_paper.pdf").write_bytes(b"%PDF-1.4 fake")


class _FakeParser:
    def parse(self, *, project_id, paper_path):
        return SimpleNamespace(full_text="AEVB on the Frey Face dataset and MNIST. " * 30)


# ---------------------------------------------------------------------------
# Flag gates
# ---------------------------------------------------------------------------


def test_web_flag_off_disables_both_modes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(WEB_FLAG, raising=False)
    monkeypatch.setenv(NET_FLAG, "1")
    store = _store(tmp_path)
    assert discover_papers("vae") == {"status": "disabled"}
    out = fetch_paper("2101.00001", store=store, project_dir=_project(tmp_path))
    assert out == {"status": "disabled"}
    store.close()


def test_grounding_off_never_dials_out(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(WEB_FLAG, "1")
    monkeypatch.delenv(NET_FLAG, raising=False)
    store = _store(tmp_path)
    assert discover_papers("vae", search_connector=_RaisingConnector())["status"] == "network_disabled"
    out = fetch_paper(
        "2101.00001",
        store=store,
        project_dir=_project(tmp_path),
        s2_connector=_RaisingConnector(),
    )
    assert out["status"] == "network_disabled"
    store.close()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_returns_bounded_fetchable_hits(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(WEB_FLAG, "1")
    monkeypatch.setenv(NET_FLAG, "1")

    class _Search:
        def search(self, query, *, limit=10):
            # Field name matches LiteratureRecord (abstract_snippet, not
            # abstract) — a mock with the wrong attribute masked a real
            # AttributeError in discover_papers.
            return [
                SimpleNamespace(
                    id="2101.00001", source="arxiv", title="T" * 500,
                    year=2021, abstract_snippet="A" * 900,
                )
            ]

    out = discover_papers("variational autoencoder", search_connector=_Search())
    assert out["status"] == "ok"
    (hit,) = out["hits"]
    assert hit["fetch_id"] == "arxiv:2101.00001"
    assert len(hit["title"]) <= 200 and len(hit["snippet"]) <= 300
    assert hit["snippet"] == "A" * 300


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def test_fetch_happy_path_lands_like_a_build_fetch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(WEB_FLAG, "1")
    monkeypatch.setenv(NET_FLAG, "1")
    store = _store(tmp_path)
    project_dir = _project(tmp_path)

    class _S2:
        def fetch(self, record_id):
            assert record_id == "ARXIV:2101.00001"
            return SimpleNamespace(
                title="Auto-Encoding Variational Bayes", year=2014, abstract_snippet="abs"
            )

    out = fetch_paper(
        "arxiv:2101.00001v3",
        store=store,
        project_dir=project_dir,
        s2_connector=_S2(),
        download_pdf_fn=_fake_download,
        parser=_FakeParser(),
    )
    assert out["status"] == "ok" and out["already_present"] is False
    assert out["fetches_remaining"] == MAX_RUNTIME_FETCHES - 1
    row = store.get_paper("arxiv:2101.00001")
    assert row is not None and row["fetched_level"] == "fulltext"
    assert row["abstract"] == "abs"   # S2 metadata (abstract_snippet) persisted
    assert (store.paper_dir("arxiv:2101.00001") / "parsed_full_text.txt").exists()
    # deterministic dictionary extraction ran over the fetched text
    ents = store.connection.execute(
        "SELECT name FROM lit_entities WHERE paper_id = ? AND source = 'deterministic'",
        ("arxiv:2101.00001",),
    ).fetchall()
    assert {e["name"] for e in ents} >= {"mnist"}
    # chunks were indexed
    n_chunks = store.connection.execute(
        "SELECT COUNT(*) FROM lit_chunks WHERE paper_id = ?", ("arxiv:2101.00001",)
    ).fetchone()[0]
    assert n_chunks >= 1
    store.close()


def test_already_present_fulltext_is_free(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(WEB_FLAG, "1")
    monkeypatch.delenv(NET_FLAG, raising=False)  # even with network off
    store = _store(tmp_path)
    store.upsert_paper("arxiv:2101.00001", title="Known", fetched_level="fulltext")
    project_dir = _project(tmp_path)
    out = fetch_paper(
        "2101.00001", store=store, project_dir=project_dir,
        s2_connector=_RaisingConnector(),
    )
    assert out["status"] == "ok" and out["already_present"] is True
    assert fetches_remaining(project_dir) == MAX_RUNTIME_FETCHES
    store.close()


def test_budget_exhausted_blocks_before_network(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(WEB_FLAG, "1")
    monkeypatch.setenv(NET_FLAG, "1")
    store = _store(tmp_path)
    project_dir = _project(tmp_path)
    ledger = project_dir / "rlm_state" / "literature_web_fetches.json"
    ledger.write_text(
        json.dumps({"fetched": [f"arxiv:2000.0000{i}" for i in range(MAX_RUNTIME_FETCHES)]}),
        encoding="utf-8",
    )
    out = fetch_paper(
        "2101.00001", store=store, project_dir=project_dir,
        s2_connector=_RaisingConnector(),
    )
    assert out["status"] == "budget_exhausted"
    store.close()


def test_failed_fetch_still_charges_the_budget(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(WEB_FLAG, "1")
    monkeypatch.setenv(NET_FLAG, "1")
    store = _store(tmp_path)
    project_dir = _project(tmp_path)

    def _broken_download(url, *, runs_root, project_id, fetched_via):
        raise OSError("download failed")

    class _S2None:
        def fetch(self, record_id):
            return None

    out = fetch_paper(
        "2101.00001", store=store, project_dir=project_dir,
        s2_connector=_S2None(), download_pdf_fn=_broken_download, parser=_FakeParser(),
    )
    assert out["status"] == "error"
    assert fetches_remaining(project_dir) == MAX_RUNTIME_FETCHES - 1
    store.close()


def test_unusable_fetch_id_rejected_without_charge(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(WEB_FLAG, "1")
    monkeypatch.setenv(NET_FLAG, "1")
    store = _store(tmp_path)
    project_dir = _project(tmp_path)
    out = fetch_paper(
        "https://evil.example/paper.pdf", store=store, project_dir=project_dir,
        s2_connector=_RaisingConnector(),
    )
    assert out["status"] == "error"
    assert fetches_remaining(project_dir) == MAX_RUNTIME_FETCHES
    store.close()


# ---------------------------------------------------------------------------
# Primitive dispatch (SimpleNamespace ctx — only runs_root/project_dir used)
# ---------------------------------------------------------------------------


def test_primitive_dispatch_web_modes_gated(tmp_path: Path, monkeypatch):
    from backend.agents.rlm.primitives import search_literature

    monkeypatch.setenv("OPENRESEARCH_LITERATURE_SURVEY", "1")
    monkeypatch.delenv(WEB_FLAG, raising=False)
    ctx = SimpleNamespace(runs_root=tmp_path, project_dir=_project(tmp_path))
    assert search_literature(fetch_id="2101.00001", ctx=ctx) == {"status": "disabled"}
    assert search_literature(web_query="vae", ctx=ctx) == {"status": "disabled"}
    # survey off wins over everything, web on or not
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_SURVEY", raising=False)
    monkeypatch.setenv(WEB_FLAG, "1")
    assert search_literature(web_query="vae", ctx=ctx) == {"status": "disabled"}
