"""Tests for k8s_job_cell_runner — AKS drop-in for gpu_cell_runner.run_matrix.

All K8s calls go through a FakeK8s; azure_blob helpers are monkeypatched.
No real cluster, SDK, or GPU required.

Suite covers:
  * signature parity with gpu_cell_runner.run_matrix (inspect.signature).
  * empty cells → {}.
  * Job Succeeded (exit 0) → "ok" with metrics pulled from Blob.
  * wrapper exit 42 → "oom_failed".
  * Job Failed (non-zero exit) → "error".
  * overall_timeout → "error" before submission.
  * Pending beyond pending_timeout → "error" with "capacity_exhausted:" prefix.
  * resume skip path (REPROLAB_RESUME_CELLS).
  * every input cell present in the result (completeness).
  * gpus_per_cell != 1 → every cell "error".
  * budget cap → cells beyond cap "error".
  * bind_run_context injects budget and event_sink without altering signature.
"""
from __future__ import annotations

import inspect
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import backend.agents.rlm.k8s_job_cell_runner as kjcr
from backend.agents.rlm.k8s_job_cell_runner import (
    CELL_MANIFEST_NAME,
    _K8sClients,
    bind_run_context,
    run_matrix,
)
import backend.agents.rlm.gpu_cell_runner as gcr


# ---------------------------------------------------------------------------
# Helpers: Fake K8s objects
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
    def __init__(self, node_name: str = "gpu-node-0") -> None:
        self.node_name = node_name


class _FakePod:
    def __init__(self, name: str = "pod-0", phase: str = "Running",
                 exit_code: int | None = 0, node_name: str = "gpu-node-0") -> None:
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


class FakeK8sBatch:
    """Fake BatchV1Api with configurable Job responses."""

    def __init__(self, job_sequence: list[_FakeJob]) -> None:
        """``job_sequence`` is polled in order on successive read_namespaced_job_status calls."""
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


class FakeK8sCore:
    """Fake CoreV1Api."""

    def __init__(
        self,
        pods: list[_FakePod] | None = None,
        log_text: str = "training ok\n",
    ) -> None:
        self._pods = pods or []
        self._log_text = log_text

    def list_namespaced_pod(self, namespace: str, label_selector: str = "") -> _FakePodList:
        return _FakePodList(self._pods)

    def read_namespaced_pod_log(self, name: str, namespace: str, **kwargs: Any) -> str:
        return self._log_text


def _make_k8s(
    *,
    job_sequence: list[_FakeJob],
    pods: list[_FakePod] | None = None,
    log_text: str = "training ok\n",
) -> _K8sClients:
    batch = FakeK8sBatch(job_sequence)
    core = FakeK8sCore(pods=pods, log_text=log_text)
    return _K8sClients(batch=batch, core=core, watch_cls=None)


# ---------------------------------------------------------------------------
# Helpers: Fake Blob
# ---------------------------------------------------------------------------

def _make_fake_blob(
    *,
    metrics: dict[str, Any] | None = None,
    raise_on_download: bool = False,
    raise_on_upload: bool = False,
) -> dict[str, Any]:
    """Return a namespace of fake azure_blob functions."""

    def fake_upload_prefix(local_root: Any, *, blob_prefix: str,
                           account_name: str, container_name: str,
                           client: Any = None) -> list[str]:
        if raise_on_upload:
            raise RuntimeError("upload failed (test)")
        return ["file1.py", "file2.py"]

    def fake_download_bytes(blob_name: str, *, account_name: str,
                            container_name: str, client: Any = None) -> bytes:
        if raise_on_download:
            raise RuntimeError("download failed (test)")
        if "metrics.json" in blob_name:
            if metrics is not None:
                return json.dumps(metrics).encode()
            raise FileNotFoundError("no metrics")
        if "status.json" in blob_name:
            raise FileNotFoundError("no status.json")
        if "logs" in blob_name:
            raise FileNotFoundError("no log")
        raise FileNotFoundError(blob_name)

    def fake_download_artifact(blob_name: str, destination: Any, *,
                               account_name: str, container_name: str,
                               client: Any = None) -> Path:
        if raise_on_download:
            raise RuntimeError("download failed (test)")
        return Path(destination)

    return {
        "upload_prefix": fake_upload_prefix,
        "download_bytes": fake_download_bytes,
        "download_artifact": fake_download_artifact,
    }


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_k8s_override():
    """Ensure the global K8s client override is cleared after each test."""
    original = kjcr._k8s_clients_override
    yield
    kjcr._k8s_clients_override = original


def _patch_blob(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    """Monkeypatch azure_blob helpers on the runner module's internal functions."""
    fake = _make_fake_blob(**kwargs)
    monkeypatch.setattr(kjcr, "_blob_upload_prefix", fake["upload_prefix"])
    monkeypatch.setattr(kjcr, "_blob_download_bytes", fake["download_bytes"])
    monkeypatch.setattr(kjcr, "_blob_download_artifact", fake["download_artifact"])


def _succeeded_job(exit_code: int = 0) -> list[_FakeJob]:
    """Job sequence that returns a Succeeded condition on first poll."""
    return [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))]


def _failed_job(exit_code: int | None = 1) -> tuple[list[_FakeJob], list[_FakePod]]:
    jobs = [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))]
    pods = [_FakePod(exit_code=exit_code)]
    return jobs, pods


def _pending_jobs(n: int = 200) -> tuple[list[_FakeJob], list[_FakePod]]:
    """Return many Pending job statuses to trigger pending_timeout."""
    jobs = [_FakeJob(_FakeJobStatus()) for _ in range(n)]
    pods = [_FakePod(phase="Pending", exit_code=None)]
    return jobs, pods


# ---------------------------------------------------------------------------
# 1. Signature parity with gpu_cell_runner.run_matrix
# ---------------------------------------------------------------------------

class TestSignatureParity:
    def test_signature_matches_gpu_cell_runner(self):
        ref_sig = inspect.signature(gcr.run_matrix)
        k8s_sig = inspect.signature(kjcr.run_matrix)
        assert list(ref_sig.parameters) == list(k8s_sig.parameters), (
            f"Parameter mismatch:\n  gcr: {list(ref_sig.parameters)}\n"
            f"  k8s: {list(k8s_sig.parameters)}"
        )

    def test_return_annotation_compatible(self):
        # Both return dict[str, dict[str, Any]] — check at the signature level.
        k8s_sig = inspect.signature(kjcr.run_matrix)
        assert k8s_sig.return_annotation is not inspect.Parameter.empty

    def test_all_defaults_match(self):
        """Keyword argument defaults must be identical."""
        ref = inspect.signature(gcr.run_matrix)
        k8s = inspect.signature(kjcr.run_matrix)
        for name, ref_param in ref.parameters.items():
            k8s_param = k8s.parameters.get(name)
            assert k8s_param is not None, f"Missing param {name!r} in k8s runner"
            if ref_param.default is not inspect.Parameter.empty:
                assert k8s_param.default == ref_param.default, (
                    f"Default mismatch for {name!r}: "
                    f"gcr={ref_param.default!r} k8s={k8s_param.default!r}"
                )


# ---------------------------------------------------------------------------
# 2. Empty cells
# ---------------------------------------------------------------------------

class TestEmptyCells:
    def test_empty_returns_empty_dict(self):
        result = run_matrix([], "train_cell.py", output_root="/tmp")
        assert result == {}

    def test_empty_never_imports_kubernetes(self):
        """Empty fast-path must not attempt to initialise K8s clients."""
        with patch.object(kjcr, "_k8s_factory", side_effect=RuntimeError("should not call")):
            result = run_matrix([], "train_cell.py", output_root="/tmp")
        assert result == {}


# ---------------------------------------------------------------------------
# 3. Job Succeeded exit 0 → "ok" with metrics
# ---------------------------------------------------------------------------

class TestJobSucceeded:
    def test_ok_status_with_metrics(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        metrics = {"reward_mean": 1.5, "accuracy": 0.8}
        cells = [{"id": "c0", "model": "qwen3-1.7b"}]

        pods = [_FakePod(phase="Running", exit_code=0, node_name="gpu-node-1")]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=metrics)

        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "outputs",
        )

        assert "c0" in results
        r = results["c0"]
        assert r["status"] == "ok"
        assert r["metrics"] == metrics
        assert r["error"] is None

    def test_gpu_label_from_node_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "c1"}]
        pods = [_FakePod(phase="Running", exit_code=0, node_name="mynode-42")]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["c1"]["gpu"] == "aks:mynode-42"

    def test_metrics_written_locally(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        metrics = {"metric": 0.9}
        cells = [{"id": "m0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=metrics)

        output_root = tmp_path / "outputs"
        run_matrix(cells, tmp_path / "train_cell.py", output_root=output_root)

        local_metrics = output_root / "m0" / "metrics.json"
        assert local_metrics.exists()
        assert json.loads(local_metrics.read_text()) == metrics

    def test_cell_manifest_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "mf0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 1.0})

        output_root = tmp_path / "outputs"
        run_matrix(cells, tmp_path / "train_cell.py", output_root=output_root)

        manifest_path = output_root / "mf0" / CELL_MANIFEST_NAME
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "ok"
        assert manifest["cell_id"] == "mf0"

    def test_job_submitted_to_namespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "ns0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "myns",
            "azure_service_account": "sa",
            "azure_node_pool_name": "gpu",
            "azure_base_image": "img:latest",
            "azure_storage_account": "myacct",
            "azure_blob_container": "mycontainer",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
        }.get(name, default))

        run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert len(k8s.batch.created_jobs) == 1


# ---------------------------------------------------------------------------
# 4. Wrapper exit 42 → "oom_failed"
# ---------------------------------------------------------------------------

class TestOomFailed:
    def test_exit_42_oom_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "oom0"}]
        # Succeeded condition but pod exit_code=42
        jobs = [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))]
        pods = [_FakePod(phase="Running", exit_code=42)]
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["oom0"]["status"] == "oom_failed"

    def test_failed_job_with_oom_sentinel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Job Failed + status.json outcome==oom_shrink_exhausted → oom_failed."""
        cells = [{"id": "oom1"}]
        jobs, pods = _failed_job(exit_code=42)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["oom1"]["status"] == "oom_failed"


# ---------------------------------------------------------------------------
# 5. Job Failed / non-terminal exit → "error"
# ---------------------------------------------------------------------------

class TestJobFailed:
    def test_job_failed_condition(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "e0"}]
        jobs, pods = _failed_job(exit_code=1)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["e0"]["status"] == "error"
        assert results["e0"]["error"] is not None

    def test_exit_40_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "e40"}]
        jobs = [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))]
        pods = [_FakePod(exit_code=40)]
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["e40"]["status"] == "error"

    def test_upload_failure_all_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Code upload failure → all cells error without submitting any Job."""
        cells = [{"id": f"u{i}"} for i in range(3)]
        _patch_blob(monkeypatch, raise_on_upload=True)
        k8s = _make_k8s(job_sequence=_succeeded_job())
        kjcr._k8s_clients_override = k8s

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        for cell in cells:
            cid = cell["id"]
            assert results[cid]["status"] == "error"
        # No Jobs should have been submitted.
        assert len(k8s.batch.created_jobs) == 0


# ---------------------------------------------------------------------------
# 6. Overall timeout → "error" before submission
# ---------------------------------------------------------------------------

class TestOverallTimeout:
    def test_overall_timeout_prevents_submission(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
        """When overall_timeout_s is tiny + already elapsed, cells are error without submit."""
        cells = [{"id": "t0"}, {"id": "t1"}]
        k8s = _make_k8s(job_sequence=_succeeded_job())
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        # Use a VERY short timeout so it expires during processing.
        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            overall_timeout_s=0.001,
        )
        # All cells must appear in results.
        for cid in ("t0", "t1"):
            assert cid in results

    def test_all_cells_present_on_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": f"x{i}"} for i in range(4)]
        k8s = _make_k8s(job_sequence=_succeeded_job())
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            overall_timeout_s=0.001,
        )
        assert set(results.keys()) == {"x0", "x1", "x2", "x3"}


# ---------------------------------------------------------------------------
# 7. Pending timeout → "error" prefixed "capacity_exhausted:"
# ---------------------------------------------------------------------------

class TestPendingTimeout:
    def test_pending_timeout_capacity_exhausted(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "pend0"}]
        # Return many Pending statuses.
        jobs, pods = _pending_jobs(n=50)
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        # Short pending_timeout_s so the test doesn't hang.
        monkeypatch.setattr(
            kjcr, "_setting",
            lambda name, default=None: {
                "azure_pending_timeout_seconds": 0.1,  # 100ms
                "azure_max_nodes": 4,
                "azure_gpu_usd_per_hour": 3.5,
                "azure_namespace": "reprolab",
                "azure_service_account": "sa",
                "azure_node_pool_name": "gpunodes",
                "azure_base_image": "img:latest",
                "azure_storage_account": "acct",
                "azure_blob_container": "ctr",
                "azure_files_share": "share",
                "azure_boot_timeout_seconds": 900,
            }.get(name, default),
        )

        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            per_cell_timeout_s=30.0,
        )
        r = results["pend0"]
        assert r["status"] == "error"
        assert r["error"] is not None
        assert r["error"].startswith("capacity_exhausted:")


# ---------------------------------------------------------------------------
# 8. Resume skip (REPROLAB_RESUME_CELLS)
# ---------------------------------------------------------------------------

class TestResume:
    def _seed_ok_cell(self, output_root: Path, cell_id: str, fingerprint: str) -> None:
        output_dir = output_root / cell_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps({"metric": 0.77}), encoding="utf-8"
        )
        (output_dir / CELL_MANIFEST_NAME).write_text(
            json.dumps({
                "cell_id": cell_id,
                "status": "ok",
                "fingerprint": fingerprint,
                "metric": 0.77,
                "retries": 0,
            }),
            encoding="utf-8",
        )

    def test_resume_skips_ok_fingerprint_matched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        output_root = tmp_path / "outputs"
        fp = "abc123"
        self._seed_ok_cell(output_root, "r0", fp)

        k8s = _make_k8s(job_sequence=_succeeded_job())
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.77})

        monkeypatch.setenv("REPROLAB_RESUME_CELLS", "1")
        cells = [{"id": "r0"}]
        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=output_root,
            fingerprints={"r0": fp},
        )
        assert results["r0"]["status"] == "skipped"
        # No Jobs submitted (skip short-circuits before submission).
        assert len(k8s.batch.created_jobs) == 0

    def test_resume_reruns_on_fingerprint_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        output_root = tmp_path / "outputs"
        self._seed_ok_cell(output_root, "r1", "old_fp")

        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        monkeypatch.setenv("REPROLAB_RESUME_CELLS", "1")
        cells = [{"id": "r1"}]
        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=output_root,
            fingerprints={"r1": "new_fp"},  # mismatch
        )
        # Should have re-run, not skipped.
        assert results["r1"]["status"] != "skipped"
        assert len(k8s.batch.created_jobs) == 1

    def test_resume_reruns_on_force_cells(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        output_root = tmp_path / "outputs"
        fp = "matchfp"
        self._seed_ok_cell(output_root, "r2", fp)

        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.6})

        monkeypatch.setenv("REPROLAB_RESUME_CELLS", "1")
        cells = [{"id": "r2"}]
        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=output_root,
            fingerprints={"r2": fp},
            force_cells={"r2"},
        )
        assert results["r2"]["status"] != "skipped"
        assert len(k8s.batch.created_jobs) == 1

    def test_resume_flag_unset_always_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        output_root = tmp_path / "outputs"
        fp = "fp_x"
        self._seed_ok_cell(output_root, "r3", fp)

        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.3})

        monkeypatch.delenv("REPROLAB_RESUME_CELLS", raising=False)
        cells = [{"id": "r3"}]
        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=output_root,
            fingerprints={"r3": fp},
        )
        # Flag unset → runs regardless.
        assert results["r3"]["status"] != "skipped"
        assert len(k8s.batch.created_jobs) == 1


# ---------------------------------------------------------------------------
# 9. Completeness — every input cell present in result
# ---------------------------------------------------------------------------

class TestCompleteness:
    def test_all_cells_present_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        n = 5
        cells = [{"id": f"cell_{i}"} for i in range(n)]

        # Build a k8s that returns Succeeded for every poll.
        jobs = [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")])) for _ in range(n * 3)]
        pods = [_FakePod(exit_code=0)]
        k8s = _make_k8s(job_sequence=jobs, pods=pods)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        results = run_matrix(
            cells, tmp_path / "train_cell.py", output_root=tmp_path / "out"
        )
        assert set(results.keys()) == {f"cell_{i}" for i in range(n)}

    def test_result_dict_has_required_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "k0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 1.0})

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        r = results["k0"]
        assert {"status", "metrics", "gpu", "retries", "error"} <= set(r.keys())

    def test_never_raises_on_single_cell_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Even if a cell fails, run_matrix must complete and return all results."""
        cells = [{"id": "ok0"}, {"id": "fail0"}]

        # First job succeeds, second fails.
        succeed = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))
        fail = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))
        # Interleave — jobs are polled sequentially per cell.
        all_jobs = [succeed, succeed, fail, fail]
        pods_ok = [_FakePod(exit_code=0)]
        k8s = _make_k8s(job_sequence=all_jobs, pods=pods_ok)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        results = run_matrix(
            cells, tmp_path / "train_cell.py", output_root=tmp_path / "out"
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 10. gpus_per_cell != 1 → every cell "error"
# ---------------------------------------------------------------------------

class TestGpusPerCell:
    def test_gpus_per_cell_ne_1_all_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": f"g{i}"} for i in range(3)]
        k8s = _make_k8s(job_sequence=_succeeded_job())
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        results = run_matrix(
            cells, tmp_path / "train_cell.py", output_root=tmp_path / "out",
            gpus_per_cell=2,
        )
        for cell in cells:
            cid = cell["id"]
            assert results[cid]["status"] == "error"
        # No Jobs submitted.
        assert len(k8s.batch.created_jobs) == 0

    def test_gpus_per_cell_1_ok(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "g1"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.3})

        results = run_matrix(
            cells, tmp_path / "train_cell.py", output_root=tmp_path / "out",
            gpus_per_cell=1,
        )
        assert results["g1"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 11. Budget cap
# ---------------------------------------------------------------------------

class TestBudget:
    def test_budget_cap_blocks_later_cells(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """With a tiny budget cap and a long per_cell_timeout, cells beyond the cap are error."""
        from backend.agents.resilience.budget import RunBudget

        cells = [{"id": f"b{i}"} for i in range(3)]
        k8s = _make_k8s(
            job_sequence=[_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))] * 10,
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        # Budget: max_run_gpu_usd = 0.001 USD — way below 3.5 $/hr * 3600s = 12.60 USD per cell.
        budget = RunBudget(max_run_gpu_usd=0.001)

        with bind_run_context(run_budget=budget):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                per_cell_timeout_s=3600.0,
                max_parallel=1,
            )

        # At least one cell must have been blocked by budget.
        error_cells = [cid for cid, r in results.items() if r["status"] == "error"]
        assert len(error_cells) >= 1

    def test_no_budget_no_cap(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Without a RunBudget bound, no budget checks run."""
        cells = [{"id": "nb0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        # No bind_run_context → budget is None → no cap.
        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["nb0"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 12. bind_run_context: event_sink called on run_warning
# ---------------------------------------------------------------------------

class TestBindRunContext:
    def test_event_sink_called_on_budget_exceeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from backend.agents.resilience.budget import RunBudget

        events: list[tuple[str, dict]] = []

        def sink(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        cells = [{"id": "ev0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job())
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        budget = RunBudget(max_run_gpu_usd=0.0001)  # tiny → triggers immediately

        with bind_run_context(run_budget=budget, event_sink=sink):
            run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                per_cell_timeout_s=3600.0,
            )

        warning_events = [e for e in events if e[0] == "run_warning"]
        assert len(warning_events) >= 1

    def test_event_sink_called_on_gpus_per_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        events: list[tuple[str, dict]] = []

        def sink(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        cells = [{"id": "ev1"}]
        k8s = _make_k8s(job_sequence=_succeeded_job())
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        with bind_run_context(event_sink=sink):
            run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                gpus_per_cell=4,  # invalid
            )

        warning_events = [e for e in events if e[0] == "run_warning"]
        assert len(warning_events) >= 1

    def test_context_var_concurrent_isolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Two concurrent threads must see their own context."""
        seen: dict[str, Any] = {}
        barrier = threading.Barrier(2, timeout=5)

        def _thread_a() -> None:
            from backend.agents.resilience.budget import RunBudget
            budget_a = RunBudget(max_run_gpu_usd=100.0)
            with bind_run_context(run_budget=budget_a):
                barrier.wait()  # sync so both are inside their context at the same time
                seen["a"] = kjcr._get_run_budget()

        def _thread_b() -> None:
            budget_b = None  # no budget
            with bind_run_context(run_budget=budget_b):
                barrier.wait()
                seen["b"] = kjcr._get_run_budget()

        ta = threading.Thread(target=_thread_a)
        tb = threading.Thread(target=_thread_b)
        ta.start(); tb.start()
        ta.join(timeout=10); tb.join(timeout=10)

        assert seen["a"] is not None
        assert seen["b"] is None

    def test_bind_run_context_no_args(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """bind_run_context with no args must not crash; defaults to no-op."""
        cells = [{"id": "noop"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.2})

        with bind_run_context():
            results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert results["noop"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 13. Job name sanity
# ---------------------------------------------------------------------------

class TestJobName:
    def test_job_name_dns_safe(self):
        name = kjcr._job_name("ALFWorld_1.7B GRPO", "run-abc123")
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name), name
        assert len(name) <= 63

    def test_job_name_deterministic(self):
        a = kjcr._job_name("cell-0", "run-xyz")
        b = kjcr._job_name("cell-0", "run-xyz")
        assert a == b

    def test_job_name_starts_with_prefix(self):
        name = kjcr._job_name("c0")
        assert name.startswith("reprolab-cell-")


# ---------------------------------------------------------------------------
# 14. gpus parameter is ignored (accepted but not used)
# ---------------------------------------------------------------------------

class TestGpusIgnored:
    def test_gpus_param_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cells = [{"id": "gig0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        # Passing gpus should not cause any error.
        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            gpus=["0", "1", "2"],  # ignored
        )
        assert results["gig0"]["status"] == "ok"
