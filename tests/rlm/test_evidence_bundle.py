"""Canonical evidence-bundle receipt (OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE).

Hermetic, on-disk fixtures only — no network/GPU/LLM. Builds small real run
directories (metrics.json + code/ files + experiment_runs.jsonl) and asserts:

1. OFF (default) → no bundle file written, scorer selection byte-for-byte legacy.
2. ON + single coherent attempt → bundle minted with correct sha/digest/
   coordinates, AND the scorer's metrics selection is identical to the legacy
   pick (single attempt ⇒ the bundle coordinate == today's newest-by-mtime pick).
3. ON + cross-attempt incoherence (metrics from B, code/ledger from A) →
   bundle is stamped bundle_unverified + resolution returns None (legacy
   fallback), never crashes.
4. mint error path → fail-soft (returns None / does not raise).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from backend.agents.rlm import evidence_bundle as eb
from backend.evals.paperbench.leaf_scorer import _latest_metrics_path

_FLAG = "OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE"


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    """Default OFF for every test; cases that need it ON set it explicitly."""
    monkeypatch.delenv(_FLAG, raising=False)
    yield


def _write_metrics(run_dir: Path, run_id: str, payload: dict, mtime: float) -> Path:
    d = run_dir / "code" / "outputs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "metrics.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def _write_code_file(run_dir: Path, rel: str, content: str) -> Path:
    p = run_dir / "code" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _append_experiment_row(run_dir: Path, row: dict) -> None:
    log = run_dir / "experiment_runs.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coherent_single_attempt(run_dir: Path) -> Path:
    """One successful attempt: metrics.json + a ledger row whose sha matches."""
    (run_dir / "code").mkdir(parents=True, exist_ok=True)
    _write_code_file(run_dir, "train.py", "print('hi')\n")
    mp = _write_metrics(
        run_dir,
        "exp_A",
        {"per_model": {"m": {"acc": 0.9}}, "comparison": {"m": {"delta": 0.05}}},
        mtime=2000,
    )
    _append_experiment_row(
        run_dir,
        {
            "success": True,
            "experiment_run_id": "exp_A",
            "artifact_dir": str(mp.parent),
            "metrics_sha256": _sha(mp),
            "model_id": "m",
            "env_id": "e",
        },
    )
    return mp


# ---------------------------------------------------------------------------
# 1. OFF (default) → no bundle file, selection unchanged
# ---------------------------------------------------------------------------


def test_flag_off_no_bundle_and_legacy_selection(tmp_path, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    mp = _coherent_single_attempt(tmp_path)

    # is_enabled is False; mint_and_persist is a no-op; no file written.
    assert eb.is_enabled() is False
    assert eb.mint_and_persist(tmp_path) is None
    assert not eb.bundle_path(tmp_path).exists()

    # Scorer selection is the legacy newest-by-mtime pick (byte-for-byte).
    legacy = _latest_metrics_path(tmp_path)
    assert legacy == mp


def test_flag_off_helper_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    _coherent_single_attempt(tmp_path)
    # resolve_metrics_path short-circuits to None when the flag is off even if a
    # stale bundle file happens to exist.
    eb.persist_bundle(tmp_path, eb.mint_bundle(tmp_path))
    assert eb.resolve_metrics_path(tmp_path) is None


# ---------------------------------------------------------------------------
# 2. ON + single coherent attempt → bundle minted; score selection identical
# ---------------------------------------------------------------------------


def test_on_single_coherent_mint_fields_and_same_selection(tmp_path, monkeypatch):
    mp = _coherent_single_attempt(tmp_path)

    # Capture the legacy selection BEFORE enabling the flag.
    legacy = _latest_metrics_path(tmp_path)
    assert legacy == mp

    monkeypatch.setenv(_FLAG, "1")
    bundle = eb.mint_and_persist(tmp_path)
    assert bundle is not None
    assert bundle["coherent"] is True
    assert "status" not in bundle  # not bundle_unverified

    # Correct sha / digest / coordinates.
    assert bundle["attempt_id"] == "exp_A"
    assert bundle["ledger_sequence"] == 0
    assert bundle["metrics_sha256"] == _sha(mp)
    assert bundle["artifact_dir"] == str(mp.parent)
    assert bundle["coordinates"]["model_id"] == "m"
    assert bundle["coordinates"]["env_id"] == "e"
    # code_tree_digest is the deterministic recompute of code/.
    assert bundle["code_tree_digest"] == eb.compute_code_tree_digest(tmp_path / "code")

    # Persisted file exists.
    assert eb.bundle_path(tmp_path).exists()

    # The scorer now resolves THROUGH the bundle — and for a single attempt that
    # coordinate is the SAME metrics file the legacy pick chose (so the score is
    # identical, only a bundle artifact is written).
    via_bundle = _latest_metrics_path(tmp_path)
    assert via_bundle == legacy == mp


def test_resolve_bundle_returns_canonical_set(tmp_path, monkeypatch):
    mp = _coherent_single_attempt(tmp_path)
    monkeypatch.setenv(_FLAG, "1")
    eb.mint_and_persist(tmp_path)

    resolved = eb.resolve_bundle(tmp_path)
    assert resolved is not None
    assert resolved["attempt_id"] == "exp_A"
    assert resolved["metrics_path"] == mp
    assert resolved["metrics_sha256"] == _sha(mp)
    assert resolved["ledger_sequence"] == 0


def test_code_tree_digest_changes_with_code(tmp_path):
    (tmp_path / "code").mkdir(parents=True)
    _write_code_file(tmp_path, "train.py", "a = 1\n")
    d1 = eb.compute_code_tree_digest(tmp_path / "code")
    _write_code_file(tmp_path, "train.py", "a = 2\n")  # content changed
    d2 = eb.compute_code_tree_digest(tmp_path / "code")
    assert d1 and d2 and d1 != d2


def test_code_tree_digest_excludes_caches(tmp_path):
    (tmp_path / "code").mkdir(parents=True)
    _write_code_file(tmp_path, "train.py", "a = 1\n")
    base = eb.compute_code_tree_digest(tmp_path / "code")
    # Adding cache/dataset/weight junk must NOT change the digest.
    _write_code_file(tmp_path, "__pycache__/x.pyc", "junk")
    _write_code_file(tmp_path, "datasets/big.bin", "junk")
    _write_code_file(tmp_path, ".venv/lib/foo.py", "junk")
    assert eb.compute_code_tree_digest(tmp_path / "code") == base


# ---------------------------------------------------------------------------
# 3. ON + cross-attempt incoherence → bundle_unverified + legacy fallback
# ---------------------------------------------------------------------------


def test_on_cross_attempt_incoherence_stamps_unverified(tmp_path, monkeypatch):
    """Ledger row (attempt A) records sha for A's metrics, but the on-disk
    metrics.json at that artifact_dir has been overwritten by attempt B's bytes —
    so the recorded sha no longer matches. Bundle must be stamped
    bundle_unverified and resolution must fall back (None), never crash."""
    (tmp_path / "code").mkdir(parents=True)
    _write_code_file(tmp_path, "train.py", "print('A')\n")

    # Attempt A metrics + its true sha recorded in the ledger.
    mp = _write_metrics(tmp_path, "exp_A", {"per_model": {"m": {"acc": 0.1}}}, mtime=1000)
    sha_a = _sha(mp)
    _append_experiment_row(
        tmp_path,
        {
            "success": True,
            "experiment_run_id": "exp_A",
            "artifact_dir": str(mp.parent),
            "metrics_sha256": sha_a,
            "model_id": "m",
            "env_id": "e",
        },
    )
    # Attempt B overwrites the SAME artifact_dir's metrics.json with different
    # bytes after the row was written — cross-attempt incoherence.
    mp.write_text(json.dumps({"per_model": {"m": {"acc": 0.99}}}), encoding="utf-8")
    os.utime(mp, (2000, 2000))
    assert _sha(mp) != sha_a

    monkeypatch.setenv(_FLAG, "1")
    bundle = eb.mint_bundle(tmp_path)
    assert bundle is not None
    assert bundle["coherent"] is False
    assert bundle["status"] == eb.BUNDLE_UNVERIFIED
    assert "mismatch" in bundle["incoherence_reason"]

    # Persist the unverified bundle; resolution refuses it → legacy fallback.
    eb.persist_bundle(tmp_path, bundle)
    assert eb.resolve_bundle(tmp_path) is None
    assert eb.resolve_metrics_path(tmp_path) is None  # no crash, legacy wins

    # The scorer therefore still returns its legacy newest-by-mtime selection.
    assert _latest_metrics_path(tmp_path) == mp


def test_on_missing_artifact_dir_is_unverified(tmp_path, monkeypatch):
    """A ledger row pointing at an artifact_dir whose metrics.json is gone is
    incoherent, not a crash."""
    (tmp_path / "code").mkdir(parents=True)
    _append_experiment_row(
        tmp_path,
        {
            "success": True,
            "experiment_run_id": "exp_gone",
            "artifact_dir": str(tmp_path / "code" / "outputs" / "exp_gone"),
            "metrics_sha256": "deadbeef",
        },
    )
    monkeypatch.setenv(_FLAG, "1")
    bundle = eb.mint_bundle(tmp_path)
    assert bundle is not None
    assert bundle["coherent"] is False
    assert bundle["status"] == eb.BUNDLE_UNVERIFIED


def test_no_successful_row_mints_none(tmp_path, monkeypatch):
    (tmp_path / "code").mkdir(parents=True)
    _append_experiment_row(tmp_path, {"success": False, "experiment_run_id": "exp_fail"})
    monkeypatch.setenv(_FLAG, "1")
    assert eb.mint_bundle(tmp_path) is None


# ---------------------------------------------------------------------------
# 4. mint error → fail-soft
# ---------------------------------------------------------------------------


def test_mint_error_is_fail_soft(tmp_path, monkeypatch):
    """Any internal error in mint must be swallowed → returns None, no raise."""
    _coherent_single_attempt(tmp_path)
    monkeypatch.setenv(_FLAG, "1")

    def _boom(_rows):
        raise RuntimeError("synthetic mint failure")

    monkeypatch.setattr(eb, "_select_canonical_row", _boom)
    # Must not raise.
    assert eb.mint_bundle(tmp_path) is None
    assert eb.mint_and_persist(tmp_path) is None


def test_resolve_error_is_fail_soft(tmp_path, monkeypatch):
    """A corrupt persisted bundle file must resolve to None, not raise."""
    monkeypatch.setenv(_FLAG, "1")
    bp = eb.bundle_path(tmp_path)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text("{not valid json", encoding="utf-8")
    assert eb.resolve_bundle(tmp_path) is None
    assert eb.resolve_metrics_path(tmp_path) is None


def test_persist_bundle_atomic_roundtrip(tmp_path):
    bundle = {"schema": 1, "attempt_id": "x", "coherent": True}
    assert eb.persist_bundle(tmp_path, bundle) is True
    loaded = json.loads(eb.bundle_path(tmp_path).read_text())
    assert loaded["attempt_id"] == "x"
