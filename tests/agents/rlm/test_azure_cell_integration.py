"""Integration test: k8s_job_cell_runner.run_matrix → cell_matrix.aggregate_cell_metrics.

Proves the FULL seam end-to-end:
  k8s_job_cell_runner.run_matrix (FakeK8s + monkeypatched azure_blob)
  → per-cell metrics.json written to output_root/<cell_id>/metrics.json
  → cell_matrix.aggregate_cell_metrics folds into canonical per_model[...][env][baseline] shape

Three scenarios required by the integration brief:
  1. Happy-path: 2 cells both Succeed (exit 0) → status "ok", metrics land,
     aggregate produces the canonical leaf shape with correct SDAR values.
  2. Mixed outcome: cell-A ok, cell-B wrapper-exit-42 (oom_failed) → aggregate
     folds ok cell; gap entry for the oom cell.
  3. Capacity exhaustion: all cells stuck-Pending past timeout → every cell
     error with "capacity_exhausted:" prefix (the stop_reason the wiring layer
     promotes to capacity_exhausted).

All K8s calls go through the FakeK8s objects copied from test_k8s_job_cell_runner.py.
azure_blob helpers are monkeypatched on the runner module's internal wrappers.
kubernetes and azure SDKs need NOT be installed; the _k8s_clients_override seam
plus _blob_* monkeypatching ensure no real imports are attempted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import backend.agents.rlm.k8s_job_cell_runner as kjcr
from backend.agents.rlm.k8s_job_cell_runner import _K8sClients, run_matrix
from backend.agents.rlm.cell_matrix import aggregate_cell_metrics

# ---------------------------------------------------------------------------
# Fixture: real SDAR-style per-cell metrics.json content (flat leaf shape).
# This is what the in-Job wrapper writes and the runner pulls from Blob.
# Shape mirrors what gpu_cell_runner cells write: flat dict with "status",
# "metric", and SDAR-specific keys (reward_mean, steps_run, etc.).
# ---------------------------------------------------------------------------
_FIXTURE_PATH = Path(__file__).resolve().parent / "__fixtures__" / "azure_cell_metrics.json"

# Load once at import time so tests share the same dict.
_CELL_METRICS_FIXTURE: dict[str, Any] = json.loads(
    _FIXTURE_PATH.read_text(encoding="utf-8")
)


# ---------------------------------------------------------------------------
# FakeK8s helpers (copied from test_k8s_job_cell_runner.py pattern)
# ---------------------------------------------------------------------------

class _FakePodStatus:
    def __init__(self, phase: str = "Running", exit_code: int | None = 0) -> None:
        self.phase = phase
        term = MagicMock()
        term.exit_code = exit_code
        cs = MagicMock()
        cs.state.terminated = term if exit_code is not None else None
        self.container_statuses = [cs]


class _FakePodSpec:
    def __init__(self, node_name: str = "aks-gpunode-0") -> None:
        self.node_name = node_name


class _FakePod:
    def __init__(
        self,
        name: str = "pod-0",
        phase: str = "Running",
        exit_code: int | None = 0,
        node_name: str = "aks-gpunode-0",
    ) -> None:
        self.metadata = MagicMock()
        self.metadata.name = name
        self.status = _FakePodStatus(phase=phase, exit_code=exit_code)
        self.spec = _FakePodSpec(node_name=node_name)


class _FakePodList:
    def __init__(self, pods: list[_FakePod]) -> None:
        self.items = pods


class _FakeJobCondition:
    def __init__(self, type_: str, status: str = "True") -> None:
        self.type = type_
        self.status = status


class _FakeJobStatus:
    def __init__(
        self,
        conditions: list[_FakeJobCondition] | None = None,
        succeeded: int = 0,
        failed: int = 0,
    ) -> None:
        self.conditions = conditions or []
        self.succeeded = succeeded
        self.failed = failed


class _FakeJob:
    def __init__(self, status: _FakeJobStatus) -> None:
        self.status = status


class _FakeK8sBatch:
    """Minimal BatchV1Api stub.  Polls job_sequence in order on each status read."""

    def __init__(self, job_sequence: list[_FakeJob]) -> None:
        self._jobs = job_sequence
        self._call_count = 0
        self.created_jobs: list[dict] = []

    def create_namespaced_job(self, namespace: str, body: dict) -> None:
        self.created_jobs.append(body)

    def read_namespaced_job_status(self, name: str, namespace: str) -> _FakeJob:
        if self._call_count < len(self._jobs):
            job = self._jobs[self._call_count]
        else:
            job = self._jobs[-1]
        self._call_count += 1
        return job

    def delete_namespaced_job(self, name: str, namespace: str, **kwargs: Any) -> None:
        pass


class _FakeK8sCore:
    """Minimal CoreV1Api stub."""

    def __init__(
        self,
        pods: list[_FakePod] | None = None,
        log_text: str = "training ok\n",
    ) -> None:
        self._pods = pods or []
        self._log_text = log_text

    def list_namespaced_pod(
        self, namespace: str, label_selector: str = ""
    ) -> _FakePodList:
        return _FakePodList(self._pods)

    def read_namespaced_pod_log(
        self, name: str, namespace: str, **kwargs: Any
    ) -> str:
        return self._log_text


def _make_k8s(
    *,
    job_sequence: list[_FakeJob],
    pods: list[_FakePod] | None = None,
    log_text: str = "training ok\n",
) -> _K8sClients:
    batch = _FakeK8sBatch(job_sequence)
    core = _FakeK8sCore(pods=pods, log_text=log_text)
    return _K8sClients(batch=batch, core=core, watch_cls=None)


# ---------------------------------------------------------------------------
# FakeBlob per-cell metrics registry
# ---------------------------------------------------------------------------

def _make_fake_blob(
    *,
    per_cell_metrics: dict[str, dict[str, Any]] | None = None,
    default_metrics: dict[str, Any] | None = None,
    raise_on_upload: bool = False,
) -> dict[str, Any]:
    """Return patched azure_blob callables.

    per_cell_metrics: {cell_id: metrics_dict} — served by cell_id extracted
    from the blob_name path ``runs/<run>/cells/<cell_id>/metrics.json``.
    default_metrics: served for any cell_id not found in per_cell_metrics.
    If a cell_id has no entry and default_metrics is None, raises FileNotFoundError.
    """

    def fake_upload_prefix(
        local_root: Any,
        *,
        blob_prefix: str,
        account_name: str,
        container_name: str,
        client: Any = None,
    ) -> list[str]:
        if raise_on_upload:
            raise RuntimeError("upload failed (test)")
        return ["train_cell.py"]

    def fake_download_bytes(
        blob_name: str,
        *,
        account_name: str,
        container_name: str,
        client: Any = None,
    ) -> bytes:
        if "metrics.json" in blob_name:
            # Extract cell_id from path: runs/<run>/cells/<cell_id>/metrics.json
            parts = blob_name.split("/")
            # Find the element after "cells" in the path.
            try:
                cells_idx = parts.index("cells")
                cell_id = parts[cells_idx + 1]
            except (ValueError, IndexError):
                cell_id = ""
            if per_cell_metrics and cell_id in per_cell_metrics:
                return json.dumps(per_cell_metrics[cell_id]).encode()
            if default_metrics is not None:
                return json.dumps(default_metrics).encode()
            raise FileNotFoundError(f"no metrics for blob_name={blob_name!r}")
        # status.json, logs, anything else → not found (fail-soft).
        raise FileNotFoundError(blob_name)

    def fake_download_artifact(
        blob_name: str,
        destination: Any,
        *,
        account_name: str,
        container_name: str,
        client: Any = None,
    ) -> Path:
        return Path(destination)

    return {
        "upload_prefix": fake_upload_prefix,
        "download_bytes": fake_download_bytes,
        "download_artifact": fake_download_artifact,
    }


# ---------------------------------------------------------------------------
# Convenience: patch blob helpers directly on the runner module
# ---------------------------------------------------------------------------

def _patch_blob(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    fake = _make_fake_blob(**kwargs)
    monkeypatch.setattr(kjcr, "_blob_upload_prefix", fake["upload_prefix"])
    monkeypatch.setattr(kjcr, "_blob_download_bytes", fake["download_bytes"])
    monkeypatch.setattr(kjcr, "_blob_download_artifact", fake["download_artifact"])


# ---------------------------------------------------------------------------
# Convenience: build job sequences
# ---------------------------------------------------------------------------

def _succeeded_job() -> list[_FakeJob]:
    """Single poll → Succeeded."""
    return [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))]


def _failed_job_exit(exit_code: int) -> tuple[list[_FakeJob], list[_FakePod]]:
    jobs = [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))]
    pods = [_FakePod(exit_code=exit_code)]
    return jobs, pods


def _pending_jobs(n: int = 60) -> tuple[list[_FakeJob], list[_FakePod]]:
    jobs = [_FakeJob(_FakeJobStatus()) for _ in range(n)]
    pods = [_FakePod(phase="Pending", exit_code=None)]
    return jobs, pods


# ---------------------------------------------------------------------------
# Cell helpers — mirror the cell dict shape from cells.json / test_cell_matrix.py
# ---------------------------------------------------------------------------

def _cell(
    model_key: str,
    env: str,
    baseline: str,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "id": f"{model_key}__{baseline}__{env}__s{seed}",
        "model_key": model_key,
        "baseline": baseline,
        "env": env,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Autouse fixture: reset the global K8s override after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_k8s_override():
    original = kjcr._k8s_clients_override
    yield
    kjcr._k8s_clients_override = original


# ===========================================================================
# 1. Happy path end-to-end — 2 cells Succeed, metrics land, aggregate correct
# ===========================================================================

class TestHappyPathEndToEnd:
    """Two SDAR-style cells (qwen3_1_7b × search_qa × sdar and × alfworld × sdar).

    Both K8s Jobs Succeed (exit 0).  FakeBlob serves the real fixture for each.
    Then aggregate_cell_metrics folds into the canonical leaf shape.
    """

    def _build_cells(self) -> list[dict[str, Any]]:
        return [
            _cell("qwen3_1_7b", "search_qa", "sdar"),
            _cell("qwen3_1_7b", "alfworld", "sdar"),
        ]

    def test_run_matrix_returns_ok_status_for_both_cells(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells = self._build_cells()
        pods = [_FakePod(exit_code=0, node_name="aks-gpunode-0")]
        # Each cell gets one Succeeded poll.
        job_seq = _succeeded_job() + _succeeded_job()
        k8s = _make_k8s(job_sequence=job_seq, pods=pods)
        kjcr._k8s_clients_override = k8s

        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        result = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "outputs",
        )

        for cell in cells:
            cid = cell["id"]
            assert cid in result, f"cell {cid!r} missing from result"
            r = result[cid]
            assert r["status"] == "ok", f"cell {cid!r} status={r['status']!r}"
            assert r["error"] is None
            assert isinstance(r["metrics"], dict)

    def test_metrics_json_written_at_canonical_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-cell metrics.json must land at output_root/<cell_id>/metrics.json."""
        cells = self._build_cells()
        output_root = tmp_path / "outputs"
        pods = [_FakePod(exit_code=0)]
        k8s = _make_k8s(job_sequence=_succeeded_job() + _succeeded_job(), pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        run_matrix(cells, tmp_path / "train_cell.py", output_root=output_root)

        for cell in cells:
            cid = cell["id"]
            metrics_path = output_root / cid / "metrics.json"
            assert metrics_path.is_file(), (
                f"metrics.json not found at {metrics_path} for cell {cid!r}"
            )
            on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert on_disk == _CELL_METRICS_FIXTURE, (
                f"on-disk metrics for {cid!r} do not match fixture"
            )

    def test_run_matrix_result_dict_has_required_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells = self._build_cells()
        k8s = _make_k8s(
            job_sequence=_succeeded_job() + _succeeded_job(),
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        result = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        for cell in cells:
            cid = cell["id"]
            assert {"status", "metrics", "gpu", "retries", "error"} <= set(result[cid])

    def test_aggregate_produces_canonical_leaf_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The main integration assertion: run_matrix result → aggregate → correct leaf."""
        cells = self._build_cells()
        k8s = _make_k8s(
            job_sequence=_succeeded_job() + _succeeded_job(),
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        matrix_result = run_matrix(
            cells, tmp_path / "train_cell.py", output_root=tmp_path / "out"
        )

        # ---- AGGREGATE ----
        agg = aggregate_cell_metrics(matrix_result, cells)

        # Top-level shape
        assert "status" in agg
        assert "per_model" in agg
        assert "scope" in agg

        # Both cells ok → status "complete"
        assert agg["status"] == "complete", (
            f"expected complete, got {agg['status']!r}"
        )

        # The canonical path: per_model[model_key][env][baseline]
        # No per_dataset wrapper (matches SDAR real sample and postflight).
        per_qwen = agg["per_model"]["qwen3_1_7b"]
        assert "per_dataset" not in per_qwen, "unexpected per_dataset wrapper"

        # search_qa leaf
        leaf_sq = per_qwen["search_qa"]["sdar"]
        assert leaf_sq["status"] == "ok"
        # Metric coerced to float from fixture value 0.37
        assert leaf_sq["metric"] == pytest.approx(_CELL_METRICS_FIXTURE["metric"])
        # Fixture passthrough keys preserved in the ok leaf
        assert leaf_sq["reward_mean"] == pytest.approx(_CELL_METRICS_FIXTURE["reward_mean"])
        assert leaf_sq["steps_run"] == _CELL_METRICS_FIXTURE["steps_run"]

        # alfworld leaf
        leaf_aw = per_qwen["alfworld"]["sdar"]
        assert leaf_aw["status"] == "ok"
        assert leaf_aw["metric"] == pytest.approx(_CELL_METRICS_FIXTURE["metric"])

        # scope
        assert agg["scope"]["models_run"] == ["qwen3_1_7b"]
        assert agg["scope"]["models_skipped"] == []
        assert agg["scope"]["environments_skipped"] == []
        assert agg["scope"]["gaps"] == []

    def test_aggregate_output_is_json_serialisable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells = self._build_cells()
        k8s = _make_k8s(
            job_sequence=_succeeded_job() + _succeeded_job(),
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        matrix_result = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        agg = aggregate_cell_metrics(matrix_result, cells)
        # Must round-trip through JSON without error.
        assert json.loads(json.dumps(agg)) == agg

    def test_gpu_label_present_in_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells = self._build_cells()
        k8s = _make_k8s(
            job_sequence=_succeeded_job() + _succeeded_job(),
            pods=[_FakePod(exit_code=0, node_name="aks-gpunode-7")],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        result = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        for cell in cells:
            assert result[cell["id"]]["gpu"].startswith("aks:"), (
                "gpu label should start with 'aks:' for K8s-dispatched cells"
            )


# ===========================================================================
# 2. Mixed outcome — cell-A ok, cell-B oom_failed (exit 42)
# ===========================================================================

class TestMixedOutcome:
    """cell-A Succeeds with real metrics; cell-B wrapper-exit-42 → oom_failed.

    aggregate_cell_metrics must:
    - fold cell-A into an ok leaf with correct metric value
    - fold cell-B into a failed leaf
    - emit top-level status "partial"
    """

    def _cells(self) -> list[dict[str, Any]]:
        return [
            _cell("qwen3_1_7b", "search_qa", "sdar"),   # will succeed
            _cell("qwen2_5_3b", "alfworld", "sdar"),     # will oom_fail
        ]

    def _fake_k8s_for_mixed(self) -> _K8sClients:
        # Two separate batch objects won't work with one global override.
        # We need a single batch that serves Succeeded for cell-A and Failed for cell-B.
        # Because FakeK8sBatch is FIFO and both cells share one batch, we give it
        # 2 reads: first Succeeded (for whichever cell polls first), then Failed.
        # Actual ordering may vary due to threading; we use per_cell_metrics in blob
        # to distinguish, and check the aggregate result rather than individual ordering.
        # For the K8s side: provide one Succeeded + one Failed sequence (6 total to be safe).
        succeed = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))
        fail = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))
        # Pods list: first pod exit 0 (ok cell), last pod exit 42 (oom cell).
        # The core fake always returns the last pod in the list for exit code extraction.
        # Since cells run in parallel threads with a single core, we need to be careful.
        # Simplest: use max_parallel=1 so they run sequentially.
        return _make_k8s(
            job_sequence=[succeed, fail],
            pods=[_FakePod(exit_code=0), _FakePod(exit_code=42)],
        )

    def test_run_matrix_mixed_statuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells = self._cells()
        cell_a_id = cells[0]["id"]
        cell_b_id = cells[1]["id"]

        # Separate metrics: cell-A has real metrics, cell-B will OOM (no metrics)
        per_cell: dict[str, dict[str, Any]] = {
            cell_a_id: _CELL_METRICS_FIXTURE,
        }
        # cell_b_id absent → FakeBlob raises FileNotFoundError → _try_download_metrics → None

        # Build a deterministic two-cell fake with max_parallel=1.
        # Cell-A gets the first job (Succeeded), Cell-B gets the second (Failed+exit42).
        succeed = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))
        fail = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))

        # For max_parallel=1 the cells are processed sequentially.  The pod list
        # is shared; the last pod in the list is always returned.  We use two
        # distinct cores with separate pod lists routed by job name — but FakeK8sCore
        # is stateless.  Workaround: the _FakeK8sBatch is call-count indexed so
        # job 1 → succeed, job 2 → fail; the _FakeK8sCore always returns its pod list.
        # We want pod exit=0 for read-1 and pod exit=42 for read-2 but the core
        # is shared.  Accept: use a stateful core that tracks call count.

        class _SequentialCore(_FakeK8sCore):
            def __init__(self) -> None:
                super().__init__()
                self._call = 0

            def list_namespaced_pod(
                self, namespace: str, label_selector: str = ""
            ) -> _FakePodList:
                self._call += 1
                if self._call <= 1:
                    return _FakePodList([_FakePod(exit_code=0, name="pod-ok")])
                return _FakePodList([_FakePod(exit_code=42, name="pod-oom")])

        batch = _FakeK8sBatch([succeed, fail])
        core = _SequentialCore()
        k8s = _K8sClients(batch=batch, core=core, watch_cls=None)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, per_cell_metrics=per_cell)

        result = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            max_parallel=1,
        )

        assert cell_a_id in result
        assert cell_b_id in result

        r_a = result[cell_a_id]
        r_b = result[cell_b_id]

        assert r_a["status"] == "ok", f"cell-A status={r_a['status']!r}"
        assert r_b["status"] == "oom_failed", f"cell-B status={r_b['status']!r}"

    def test_aggregate_partial_status_with_mixed_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Aggregate must emit status='partial' and the ok leaf must carry metrics."""
        cells = self._cells()
        cell_a_id = cells[0]["id"]
        cell_b_id = cells[1]["id"]

        # Build matrix_result directly (independent of run_matrix threading concerns).
        # This is the pure-aggregation integration assertion.
        matrix_result: dict[str, dict[str, Any]] = {
            cell_a_id: {
                "status": "ok",
                "metrics": _CELL_METRICS_FIXTURE,
                "gpu": "aks:aks-gpunode-0",
                "retries": 0,
                "error": None,
            },
            cell_b_id: {
                "status": "oom_failed",
                "metrics": None,
                "gpu": "aks:aks-gpunode-1",
                "retries": 2,
                "error": "CUDA out of memory. Tried to allocate 1.11 GiB.",
            },
        }

        agg = aggregate_cell_metrics(matrix_result, cells)

        # Top-level: one ok, one failed → partial
        assert agg["status"] == "partial", f"expected partial, got {agg['status']!r}"

        # cell-A ok leaf
        leaf_a = agg["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]
        assert leaf_a["status"] == "ok"
        assert leaf_a["metric"] == pytest.approx(_CELL_METRICS_FIXTURE["metric"])
        assert leaf_a["reward_mean"] == pytest.approx(_CELL_METRICS_FIXTURE["reward_mean"])

        # cell-B failed leaf
        leaf_b = agg["per_model"]["qwen2_5_3b"]["alfworld"]["sdar"]
        assert leaf_b["status"] == "failed"
        assert leaf_b["metric"] is None
        assert "CUDA out of memory" in leaf_b["error"]

        # scope: cell-A's model is in models_run; cell-B's model is NOT (it had no ok cell)
        assert "qwen3_1_7b" in agg["scope"]["models_run"]
        assert "qwen2_5_3b" not in agg["scope"]["models_run"]

    def test_ok_cell_metrics_json_on_disk_when_mixed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even in a mixed run, the ok cell's metrics.json lands at the canonical path."""
        cells = self._cells()
        cell_a_id = cells[0]["id"]
        cell_b_id = cells[1]["id"]
        output_root = tmp_path / "out"

        # Run with real run_matrix using max_parallel=1.
        succeed = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))
        fail = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))

        class _SequentialCore2(_FakeK8sCore):
            def __init__(self) -> None:
                super().__init__()
                self._call = 0

            def list_namespaced_pod(
                self, namespace: str, label_selector: str = ""
            ) -> _FakePodList:
                self._call += 1
                if self._call <= 1:
                    return _FakePodList([_FakePod(exit_code=0)])
                return _FakePodList([_FakePod(exit_code=42)])

        batch = _FakeK8sBatch([succeed, fail])
        core = _SequentialCore2()
        k8s = _K8sClients(batch=batch, core=core, watch_cls=None)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, per_cell_metrics={cell_a_id: _CELL_METRICS_FIXTURE})

        run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=output_root,
            max_parallel=1,
        )

        # cell-A must have metrics.json
        metrics_path_a = output_root / cell_a_id / "metrics.json"
        assert metrics_path_a.is_file(), f"ok cell missing metrics.json at {metrics_path_a}"
        assert json.loads(metrics_path_a.read_text()) == _CELL_METRICS_FIXTURE


# ===========================================================================
# 3. Capacity exhaustion — all cells stuck-Pending past timeout
# ===========================================================================

class TestCapacityExhaustion:
    """All cells get stuck in Pending state past the pending_timeout.

    Expected:
    - Every cell result has status "error"
    - Every cell error string starts with "capacity_exhausted:"
    - aggregate_cell_metrics sees all cells as failed → top-level "failed"
    """

    def _cells(self) -> list[dict[str, Any]]:
        return [
            _cell("qwen3_1_7b", "search_qa", "sdar"),
            _cell("qwen3_1_7b", "alfworld", "sdar"),
        ]

    def _patch_fast_pending_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            kjcr,
            "_setting",
            lambda name, default=None: {
                "azure_pending_timeout_seconds": 0.05,  # 50ms → triggers fast
                "azure_max_nodes": 4,
                "azure_gpu_usd_per_hour": 3.67,
                "azure_namespace": "reprolab",
                "azure_service_account": "reprolab-sa",
                "azure_node_pool_name": "gpunodes",
                "azure_base_image": "reprolab.azurecr.io/reprolab-aks-cell:latest",
                "azure_storage_account": "",
                "azure_blob_container": "reprolab-artifacts",
                "azure_files_share": "reprolab-cache",
                "azure_boot_timeout_seconds": 900,
            }.get(name, default),
        )

    def test_all_cells_error_with_capacity_exhausted_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells = self._cells()
        jobs, pods = _pending_jobs(n=100)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=None)
        self._patch_fast_pending_timeout(monkeypatch)

        result = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            per_cell_timeout_s=60.0,
        )

        for cell in cells:
            cid = cell["id"]
            assert cid in result, f"cell {cid!r} missing from result"
            r = result[cid]
            assert r["status"] == "error", (
                f"cell {cid!r}: expected error, got {r['status']!r}"
            )
            assert r["error"] is not None, f"cell {cid!r}: error field is None"
            assert r["error"].startswith("capacity_exhausted:"), (
                f"cell {cid!r}: error={r['error']!r} does not start with 'capacity_exhausted:'"
            )

    def test_aggregate_status_failed_when_all_capacity_exhausted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """aggregate_cell_metrics must emit top-level 'failed' when all cells are error."""
        cells = self._cells()
        jobs, pods = _pending_jobs(n=100)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=None)
        self._patch_fast_pending_timeout(monkeypatch)

        matrix_result = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            per_cell_timeout_s=60.0,
        )

        agg = aggregate_cell_metrics(matrix_result, cells)

        assert agg["status"] == "failed", (
            f"expected 'failed' when all cells are capacity_exhausted, got {agg['status']!r}"
        )
        # No model made it to an ok result.
        assert agg["scope"]["models_run"] == []

        # Each cell has a failed leaf in per_model.
        for cell in cells:
            model_key = cell["model_key"]
            env = cell["env"]
            baseline = cell["baseline"]
            leaf = agg["per_model"][model_key][env][baseline]
            assert leaf["status"] == "failed"
            assert leaf["metric"] is None
            # The error string from the run_matrix result (capacity_exhausted: ...) is
            # truncated into the leaf.
            assert leaf["error"].startswith("capacity_exhausted:")

    def test_error_field_starts_with_capacity_exhausted_in_leaf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 'capacity_exhausted:' prefix must survive intact into the aggregate leaf.

        This is the key wiring assertion: the stop_reason promotion logic in run.py
        detects this prefix in the aggregate leaf to set the terminal stop_reason.
        """
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar")]
        jobs, pods = _pending_jobs(n=100)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=None)
        self._patch_fast_pending_timeout(monkeypatch)

        matrix_result = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            per_cell_timeout_s=60.0,
        )

        agg = aggregate_cell_metrics(matrix_result, cells)

        # The leaf error must carry the capacity_exhausted prefix so the wiring layer
        # can detect it.
        leaf = agg["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]
        assert leaf["status"] == "failed"
        assert leaf.get("error", "").startswith("capacity_exhausted:"), (
            f"leaf error must start with 'capacity_exhausted:', got: {leaf.get('error')!r}"
        )

    def test_both_cells_present_in_result_after_capacity_exhaustion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_matrix must return ALL cells in the result dict even on capacity exhaustion."""
        cells = self._cells()
        jobs, pods = _pending_jobs(n=100)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=None)
        self._patch_fast_pending_timeout(monkeypatch)

        result = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            per_cell_timeout_s=60.0,
        )

        assert set(result.keys()) == {cell["id"] for cell in cells}


# ===========================================================================
# 4. Seam verification — run_matrix output shape exactly matches what
#    aggregate_cell_metrics expects.  This is the integration-risk assertion.
# ===========================================================================

class TestSeamCompatibility:
    """Verify the CellResult.to_dict() schema from run_matrix is consumed
    correctly by aggregate_cell_metrics without any adapter layer.

    These tests build the matrix_result directly from run_matrix (not manually)
    and pass it unchanged to aggregate_cell_metrics to prove no intermediate
    translation is needed.
    """

    def test_result_keys_are_what_aggregate_expects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """aggregate_cell_metrics reads: status, metrics, error (from the record).

        The to_dict() schema must include all three without renaming.
        """
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar")]
        k8s = _make_k8s(
            job_sequence=_succeeded_job(),
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        result = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        r = result[cells[0]["id"]]
        # aggregate_cell_metrics reads these three keys from the record dict:
        assert "status" in r
        assert "metrics" in r
        assert "error" in r
        # gpu and retries are passed through but not consumed by aggregate:
        assert "gpu" in r
        assert "retries" in r

    def test_no_adapter_needed_between_run_matrix_and_aggregate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass run_matrix output DIRECTLY into aggregate_cell_metrics with no mapping."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar"),
            _cell("qwen2_5_3b", "alfworld", "sdar"),
        ]
        k8s = _make_k8s(
            job_sequence=_succeeded_job() + _succeeded_job(),
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, default_metrics=_CELL_METRICS_FIXTURE)

        matrix_result = run_matrix(
            cells, tmp_path / "train_cell.py", output_root=tmp_path / "out"
        )

        # Pass DIRECTLY — no transformation.
        agg = aggregate_cell_metrics(matrix_result, cells)

        # Both cells ok → complete.
        assert agg["status"] == "complete"
        # Both leaves present and ok.
        assert agg["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]["status"] == "ok"
        assert agg["per_model"]["qwen2_5_3b"]["alfworld"]["sdar"]["status"] == "ok"

    def test_metrics_field_is_none_for_pending_timeout_cells(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FakeBlob raises FileNotFoundError (no metrics blob), result.metrics=None.

        aggregate_cell_metrics must tolerate None metrics and produce a failed leaf.
        """
        cells = [_cell("qwen3_1_7b", "search_qa", "sdar")]

        # Patch blob to raise on metrics download.
        monkeypatch.setattr(kjcr, "_blob_upload_prefix", lambda *a, **kw: ["f.py"])
        monkeypatch.setattr(
            kjcr, "_blob_download_bytes",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no blob")),
        )
        monkeypatch.setattr(
            kjcr, "_blob_download_artifact",
            lambda *a, **kw: Path(kw.get("destination", "/tmp")),
        )

        jobs, pods = _failed_job_exit(1)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s

        result = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        r = result[cells[0]["id"]]
        assert r["status"] == "error"
        assert r["metrics"] is None

        agg = aggregate_cell_metrics(result, cells)
        leaf = agg["per_model"]["qwen3_1_7b"]["search_qa"]["sdar"]
        assert leaf["status"] == "failed"
        assert leaf["metric"] is None  # None metrics → null metric in leaf
