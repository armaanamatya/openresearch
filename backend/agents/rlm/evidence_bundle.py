"""Canonical evidence-bundle receipt (feature flag-gated, default OFF).

WHY this exists
---------------
Scoring and reporting today each *independently* resolve "the evidence" the run
produced:

  * the leaf scorer (``leaf_scorer._latest_metrics_path``) picks the
    NEWEST-by-mtime ``metrics.json`` it can find on disk;
  * the final report (``report._canonical_experiment_provenance``) separately
    picks the LATEST SUCCESSFUL ``experiment_runs.jsonl`` row and trusts its
    ``metrics_sha256`` + ``artifact_dir``;
  * the code tree (``code/``) is read as whatever happens to be on disk now.

Each of those is a reasonable heuristic in isolation, but they are *separate*
selections over the SAME run. On a multi-attempt run they can disagree — e.g.
the newest metrics.json belongs to attempt B (an exploratory re-run) while the
latest *successful* ledger row + the restored ``code/`` belong to attempt A.
The grader then scores attempt-B numbers against attempt-A code/provenance:
mixed evidence, an incoherent grade.

The fix is a single immutable RECEIPT minted ONCE on a successful experiment
that binds the evidence together::

    {attempt_id, ledger_sequence, metrics_sha256, code_tree_digest,
     artifact_dir, coordinates}

Score and report can then resolve evidence THROUGH the receipt instead of each
re-deriving "newest". Resolution re-verifies coherence (the recorded
``metrics_sha256`` must still match the bytes on disk at ``artifact_dir``); on
ANY mismatch or error we stamp ``bundle_unverified`` and fall back to today's
independent legacy selection — so a bundle can only ever make evidence MORE
coherent, never assert something the disk doesn't support.

Design constraints
------------------
* **Gated** behind ``OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE`` (truthy
  ``1``/``true``/``yes``/``on``). Default OFF ⇒ scorer + report behave
  byte-for-byte as today (callers must check :func:`is_enabled` first).
* **Fail-soft**: nothing here may raise into scoring/reporting. Every public
  function swallows its own exceptions and returns ``None`` (or stamps
  ``bundle_unverified``) on any error.
* **stdlib-only** (``hashlib``/``json``/``os``) — deterministic digests, no new
  deps. The bundle is the cross-attempt SUPERSET of the existing
  ``_canonical_experiment_provenance`` flow; it is built ALONGSIDE it, not in
  place of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BUNDLE_REL_PATH = ("rlm_state", "evidence_bundle.json")
_EXPERIMENT_LOG = "experiment_runs.jsonl"

# Directory names excluded from the code-tree digest — caches/venvs/datasets/
# weights are non-deterministic, huge, and not part of "the implementation".
_CODE_TREE_EXCLUDE_DIRS = frozenset(
    {
        "__pycache__",
        ".venv",
        "venv",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "datasets",
        "weights",
        "node_modules",
        "outputs",  # per-attempt experiment outputs, not source
        ".ipynb_checkpoints",
    }
)

# Sentinel value callers may stamp on a report/result when a bundle was expected
# but could not be coherently resolved.
BUNDLE_UNVERIFIED = "bundle_unverified"


def is_enabled() -> bool:
    """WHY: a single truthy gate so OFF (default) is byte-for-byte unchanged.

    Reads ``OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE`` each call (no import-time
    capture) so tests can monkeypatch the env per-case.
    """
    return os.environ.get("OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def bundle_path(project_dir: Path | str) -> Path:
    """Path to the persisted receipt: ``<project_dir>/rlm_state/evidence_bundle.json``."""
    return Path(project_dir).joinpath(*_BUNDLE_REL_PATH)


def _sha256_file(path: Path) -> str | None:
    """sha256 of a file's raw bytes (the same approach ``primitives._manifest_enrichment``
    uses for ``metrics_sha256``). ``None`` on any read error."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def compute_code_tree_digest(code_dir: Path | str) -> str | None:
    """Deterministic digest of ``code/``: sorted hash of (relpath, sha256) pairs.

    WHY sorted (path, content-hash) pairs: the digest must be stable across
    filesystem walk order and must change iff a tracked source file's PATH or
    CONTENT changes — so two attempts with different code produce different
    digests (the coherence signal), while re-reading the same tree is identical.
    Caches/venvs/datasets/weights are excluded (see ``_CODE_TREE_EXCLUDE_DIRS``):
    they are non-deterministic and not "the implementation".

    Returns ``None`` (fail-soft) when ``code_dir`` is missing or unreadable.
    """
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return None
    try:
        pairs: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(code_dir):
            # Prune excluded dirs in place so os.walk does not descend into them.
            dirs[:] = [d for d in dirs if d not in _CODE_TREE_EXCLUDE_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                rel = fpath.relative_to(code_dir).as_posix()
                fsha = _sha256_file(fpath)
                if fsha is None:
                    continue
                pairs.append((rel, fsha))
        pairs.sort()
        canonical = json.dumps(pairs, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001 — digest must never raise into scoring
        logger.debug("evidence_bundle: code-tree digest failed", exc_info=True)
        return None


def _read_experiment_rows(project_dir: Path) -> list[tuple[int, dict[str, Any]]]:
    """Return ``(line_index, row)`` for each JSONL row in ``experiment_runs.jsonl``.

    The ``line_index`` (0-based, counting only non-blank lines) IS the
    ``ledger_sequence`` — a per-run monotonic ordering of run_experiment calls.
    Torn/partial lines are skipped. ``[]`` on any error.
    """
    log = project_dir / _EXPERIMENT_LOG
    if not log.exists():
        return []
    out: list[tuple[int, dict[str, Any]]] = []
    try:
        idx = 0
        for raw in log.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:  # noqa: BLE001 — tolerate a torn/partial line
                idx += 1
                continue
            if isinstance(rec, dict):
                out.append((idx, rec))
            idx += 1
    except OSError:
        return []
    return out


def _select_canonical_row(
    rows: list[tuple[int, dict[str, Any]]],
) -> tuple[int, dict[str, Any]] | None:
    """The canonical experiment row: the LATEST SUCCESSFUL row carrying an
    ``experiment_run_id`` — mirrors ``report._latest_successful_experiment_record``
    so the bundle names the SAME run the legacy provenance flow does (the bundle
    is its coherent superset, not a competing pick)."""
    chosen: tuple[int, dict[str, Any]] | None = None
    for seq, rec in rows:
        if rec.get("success") and rec.get("experiment_run_id"):
            chosen = (seq, rec)  # keep the latest successful
    return chosen


def _coordinates(row: dict[str, Any]) -> dict[str, Any]:
    """The model/env coordinates that locate this evidence in the run matrix.

    Best-effort: ``model_id``/``env_id`` are stamped by
    ``primitives._stamp_manifest_ids`` / ``_persist_experiment_result``.
    """
    coords: dict[str, Any] = {}
    for key in ("model_id", "env_id", "experiment_run_id"):
        val = row.get(key)
        if val is not None:
            coords[key] = val
    return coords


def mint_bundle(project_dir: Path | str) -> dict[str, Any] | None:
    """Mint ONE receipt binding the canonical evidence for this run, or ``None``.

    WHY this shape: the receipt names exactly the artifact set scoring + report
    must agree on — ``attempt_id`` (the experiment_run_id of the canonical
    successful run), ``ledger_sequence`` (its row index in
    ``experiment_runs.jsonl``), ``metrics_sha256`` (sha of the metrics.json bytes
    at its artifact_dir), ``code_tree_digest`` (the implementation that produced
    it), ``artifact_dir``, and ``coordinates`` (model/env).

    Coherence is enforced at mint time: the canonical row must point at an
    on-disk ``artifact_dir`` whose ``metrics.json`` exists; the recorded
    ``metrics_sha256`` (if present) must match the actual bytes. On ANY
    mismatch/missing artifact the bundle is marked ``coherent: False`` (callers
    treat that exactly as ``bundle_unverified`` and fall back to legacy). Returns
    ``None`` only when there is no canonical successful row to mint from.

    Fail-soft: never raises.
    """
    try:
        project_dir = Path(project_dir)
        rows = _read_experiment_rows(project_dir)
        selected = _select_canonical_row(rows)
        if selected is None:
            return None
        ledger_sequence, row = selected

        attempt_id = str(row.get("experiment_run_id"))
        artifact_dir = row.get("artifact_dir")
        recorded_sha = row.get("metrics_sha256")

        coherent = True
        reason = ""

        # Resolve the metrics file the receipt binds to. The artifact_dir is the
        # authoritative location; fall back to nothing (incoherent) if absent.
        metrics_file: Path | None = None
        if artifact_dir:
            candidate = Path(str(artifact_dir)) / "metrics.json"
            if candidate.is_file():
                metrics_file = candidate
            else:
                coherent = False
                reason = "artifact_dir metrics.json missing"
        else:
            coherent = False
            reason = "no artifact_dir on canonical row"

        on_disk_sha: str | None = None
        if metrics_file is not None:
            on_disk_sha = _sha256_file(metrics_file)
            if on_disk_sha is None:
                coherent = False
                reason = "metrics.json unreadable"
            elif recorded_sha and recorded_sha != on_disk_sha:
                # Cross-attempt incoherence: the ledger row's recorded hash does
                # not match the bytes currently on disk — the evidence has been
                # mixed across attempts. Stamp incoherent; legacy fallback wins.
                coherent = False
                reason = "metrics_sha256 mismatch (cross-attempt incoherence)"

        metrics_sha256 = on_disk_sha or (recorded_sha if isinstance(recorded_sha, str) else None)

        code_tree_digest = compute_code_tree_digest(project_dir / "code")

        bundle: dict[str, Any] = {
            "schema": 1,
            "attempt_id": attempt_id,
            "ledger_sequence": int(ledger_sequence),
            "metrics_sha256": metrics_sha256,
            "code_tree_digest": code_tree_digest,
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
            "coordinates": _coordinates(row),
            "coherent": bool(coherent),
        }
        if not coherent:
            bundle["status"] = BUNDLE_UNVERIFIED
            bundle["incoherence_reason"] = reason
        return bundle
    except Exception:  # noqa: BLE001 — minting must never break a run
        logger.debug("evidence_bundle: mint failed", exc_info=True)
        return None


def persist_bundle(project_dir: Path | str, bundle: dict[str, Any]) -> bool:
    """Atomically write ``bundle`` to ``rlm_state/evidence_bundle.json``.

    Atomic (tmp + ``os.replace``) so a concurrent reader never sees a torn file.
    Returns ``True`` on success, ``False`` (fail-soft) on any error.
    """
    try:
        target = bundle_path(project_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
        return True
    except Exception:  # noqa: BLE001 — persistence must never break a run
        logger.debug("evidence_bundle: persist failed", exc_info=True)
        return False


def mint_and_persist(project_dir: Path | str) -> dict[str, Any] | None:
    """Convenience: mint then persist (only when enabled). Returns the bundle or None."""
    if not is_enabled():
        return None
    bundle = mint_bundle(project_dir)
    if bundle is None:
        return None
    persist_bundle(project_dir, bundle)
    return bundle


def resolve_bundle(project_dir: Path | str) -> dict[str, Any] | None:
    """Return the canonical artifact set from the persisted receipt, or ``None``.

    Resolution RE-VERIFIES coherence at read time: the persisted
    ``metrics_sha256`` must still match the bytes at ``artifact_dir/metrics.json``
    on disk. This is what protects against the evidence drifting AFTER the bundle
    was minted (a later attempt overwriting metrics.json). On any mismatch,
    missing artifact, an already-``bundle_unverified`` receipt, or any read
    error, returns ``None`` so the caller falls back to legacy selection.

    The returned dict carries the resolved coordinates the caller should use:
    ``{attempt_id, ledger_sequence, metrics_path, metrics_sha256,
       code_tree_digest, artifact_dir, coordinates}``.
    """
    try:
        path = bundle_path(project_dir)
        if not path.is_file():
            return None
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(bundle, dict):
            return None
        if bundle.get("status") == BUNDLE_UNVERIFIED or not bundle.get("coherent", False):
            return None

        artifact_dir = bundle.get("artifact_dir")
        if not artifact_dir:
            return None
        metrics_file = Path(str(artifact_dir)) / "metrics.json"
        if not metrics_file.is_file():
            return None

        recorded_sha = bundle.get("metrics_sha256")
        on_disk_sha = _sha256_file(metrics_file)
        if on_disk_sha is None:
            return None
        if recorded_sha and recorded_sha != on_disk_sha:
            # The evidence drifted after minting — refuse to resolve; legacy wins.
            logger.debug("evidence_bundle: resolve sha drift — falling back to legacy")
            return None

        return {
            "attempt_id": bundle.get("attempt_id"),
            "ledger_sequence": bundle.get("ledger_sequence"),
            "metrics_path": metrics_file,
            "metrics_sha256": on_disk_sha,
            "code_tree_digest": bundle.get("code_tree_digest"),
            "artifact_dir": str(artifact_dir),
            "coordinates": bundle.get("coordinates") or {},
        }
    except Exception:  # noqa: BLE001 — resolution must never break scoring
        logger.debug("evidence_bundle: resolve failed", exc_info=True)
        return None


def resolve_metrics_path(project_dir: Path | str) -> Path | None:
    """Thin helper: the bundle's canonical ``metrics.json`` path, or ``None``.

    The single integration point the leaf scorer uses — it replaces the
    independent newest-by-mtime pick with the bundle's coordinate WHEN the bundle
    is enabled AND coherently resolvable, else returns ``None`` (legacy wins).
    """
    if not is_enabled():
        return None
    resolved = resolve_bundle(project_dir)
    if resolved is None:
        return None
    mp = resolved.get("metrics_path")
    return mp if isinstance(mp, Path) else None
