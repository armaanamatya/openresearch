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
  * resume skip path (OPENRESEARCH_RESUME_CELLS).
  * every input cell present in the result (completeness).
  * gpus_per_cell != 1 → every cell "error".
  * budget cap → cells beyond cap "error".
  * bind_run_context injects budget and event_sink without altering signature.
  * cell_scheduler symbols adopted (CellResult, headline_metric, etc.).
  * gpu_plan bound → manifest nodeSelector reprolab/sku + correct gpu_count.
  * no gpu_plan → default manifest (back-compat, agentpool nodeSelector).
  * SKU escalation: oom_failed + ladder → resubmit on bigger pool, gpu_escalated event.
  * escalation graceful-degrade: empty ladder / not-provisioned → stays oom_failed.
  * escalation cap: bounded by dynamic_gpu_max_escalations.
  * bind_run_context(gpu_plan=...) round-trips via _get_gpu_plan().
"""
from __future__ import annotations

import inspect
import json
import threading
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
from backend.agents.rlm import cell_scheduler


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
# 8. Resume skip (OPENRESEARCH_RESUME_CELLS)
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

        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
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

        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
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

        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
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

        monkeypatch.delenv("OPENRESEARCH_RESUME_CELLS", raising=False)
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


# ---------------------------------------------------------------------------
# 15. cell_scheduler symbols adoption
# ---------------------------------------------------------------------------

class TestCellSchedulerAdoption:
    """Verify that the runner uses the shared cell_scheduler symbols, not local copies."""

    def test_cell_manifest_name_is_scheduler_constant(self):
        assert kjcr.CELL_MANIFEST_NAME is cell_scheduler.CELL_MANIFEST_NAME

    def test_cell_result_is_scheduler_class(self):
        # CellResult imported by kjcr must be the same class as in cell_scheduler.
        assert kjcr.CellResult is cell_scheduler.CellResult

    def test_headline_metric_is_scheduler_function(self):
        assert kjcr.headline_metric is cell_scheduler.headline_metric

    def test_load_cell_manifest_is_scheduler_function(self):
        assert kjcr.load_cell_manifest is cell_scheduler.load_cell_manifest

    def test_should_skip_cell_is_scheduler_function(self):
        assert kjcr.should_skip_cell is cell_scheduler.should_skip_cell

    def test_write_cell_manifest_is_scheduler_function(self):
        assert kjcr.write_cell_manifest is cell_scheduler.write_cell_manifest

    def test_is_resume_armed_is_scheduler_function(self):
        assert kjcr.is_resume_armed is cell_scheduler.is_resume_armed

    def test_deadline_from_timeout_is_scheduler_function(self):
        assert kjcr.deadline_from_timeout is cell_scheduler.deadline_from_timeout

    def test_clamp_cell_timeout_is_scheduler_function(self):
        assert kjcr.clamp_cell_timeout is cell_scheduler.clamp_cell_timeout

    def test_status_constants_match_scheduler(self):
        assert kjcr.STATUS_OK is cell_scheduler.STATUS_OK
        assert kjcr.STATUS_OOM_FAILED is cell_scheduler.STATUS_OOM_FAILED
        assert kjcr.STATUS_SKIPPED is cell_scheduler.STATUS_SKIPPED
        assert kjcr.STATUS_ERROR is cell_scheduler.STATUS_ERROR
        assert kjcr.STATUS_TIMEOUT is cell_scheduler.STATUS_TIMEOUT

    def test_write_cell_manifest_caller_k8s(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            caplog: pytest.LogCaptureFixture):
        """write_cell_manifest called by the runner must use caller='k8s_job_cell_runner'."""
        cells = [{"id": "wm0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        output_root = tmp_path / "out"
        run_matrix(cells, tmp_path / "train_cell.py", output_root=output_root)

        # The manifest must have been written (not raise).
        manifest_path = output_root / "wm0" / CELL_MANIFEST_NAME
        assert manifest_path.exists()

    def test_existing_run_matrix_tests_still_pass_with_scheduler_symbols(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Smoke: a basic happy-path still works after adopting cell_scheduler."""
        cells = [{"id": "cs0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.9})

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["cs0"]["status"] == "ok"
        assert results["cs0"]["metrics"] == {"metric": 0.9}


# ---------------------------------------------------------------------------
# 16. gpu_plan → Job manifest nodeSelector + gpu_count
# ---------------------------------------------------------------------------

class _FakeGpuPlan:
    """Minimal GpuPlan stub for testing — only the fields _build_job_manifest reads."""

    def __init__(
        self,
        short_name: str = "azure_a100_80",
        gpu_count: int = 1,
        ladder_remaining: tuple[str, ...] = (),
    ) -> None:
        self.short_name = short_name
        self.gpu_count = gpu_count
        self.ladder_remaining = ladder_remaining


class TestGpuPlanManifest:
    """Job manifest must honour gpu_plan when one is bound."""

    def test_gpu_plan_sets_node_selector_sku(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """nodeSelector must include reprolab/sku=<plan.short_name> when gpu_plan is bound."""
        cells = [{"id": "gpm0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        plan = _FakeGpuPlan(short_name="azure_a100_80", gpu_count=1)

        with bind_run_context(gpu_plan=plan):
            run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert len(k8s.batch.created_jobs) == 1
        job_body = k8s.batch.created_jobs[0]
        node_selector = (
            job_body["spec"]["template"]["spec"]["nodeSelector"]
        )
        assert node_selector == {"reprolab/sku": "azure_a100_80"}, (
            f"expected reprolab/sku nodeSelector, got {node_selector!r}"
        )

    def test_gpu_plan_sets_gpu_count_in_resources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """GPU resource request/limit must equal plan.gpu_count when gpu_plan is bound."""
        cells = [{"id": "gpm1"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.2})

        plan = _FakeGpuPlan(short_name="azure_a100_80x2", gpu_count=2)

        with bind_run_context(gpu_plan=plan):
            run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert len(k8s.batch.created_jobs) == 1
        job_body = k8s.batch.created_jobs[0]
        container = job_body["spec"]["template"]["spec"]["containers"][0]
        resources = container["resources"]
        assert resources["requests"]["nvidia.com/gpu"] == "2", (
            f"expected gpu_count=2 in requests, got {resources['requests']!r}"
        )
        assert resources["limits"]["nvidia.com/gpu"] == "2", (
            f"expected gpu_count=2 in limits, got {resources['limits']!r}"
        )

    def test_gpu_plan_taint_toleration_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Manifest must always tolerate nvidia.com/gpu taint (Exists, NoSchedule)."""
        cells = [{"id": "gpm2"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        plan = _FakeGpuPlan(short_name="azure_a100_80", gpu_count=1)

        with bind_run_context(gpu_plan=plan):
            run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        tolerations = k8s.batch.created_jobs[0]["spec"]["template"]["spec"]["tolerations"]
        nvidia_tols = [
            t for t in tolerations
            if t.get("key") == "nvidia.com/gpu" and t.get("operator") == "Exists"
        ]
        assert nvidia_tols, f"no nvidia.com/gpu taint toleration found in {tolerations!r}"

    def test_no_gpu_plan_uses_reprolab_sku_default_selector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Without a gpu_plan the manifest must fall back to reprolab/sku=<default_sku>
        (P0-fix-3: ``agentpool`` label does NOT match the infra pool label contract;
        ``reprolab/sku`` ensures the Pod lands on a real GPU node even without a plan).
        """
        cells = [{"id": "gpm3"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        # Provide a settings override so we get a predictable default_sku.
        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            # One provisioned SKU — this becomes the default.
            "azure_gpu_skus": ["azure_a100_80"],
            "dynamic_gpu_max_escalations": 2,
            "azure_ttl_seconds_after_finished": 3600,
            "azure_job_backoff_limit": 0,
            "azure_cache_mount_path": "/mnt/reprolab-cache",
            "azure_watch_poll_interval_s": 5.0,
        }.get(name, default))

        # No bind_run_context at all → gpu_plan is None.
        run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        node_selector = k8s.batch.created_jobs[0]["spec"]["template"]["spec"]["nodeSelector"]
        # P0-fix-3: must use reprolab/sku, NOT agentpool.
        assert "reprolab/sku" in node_selector, (
            f"expected reprolab/sku nodeSelector without gpu_plan, got {node_selector!r}"
        )
        assert "agentpool" not in node_selector, (
            f"agentpool must not be present; reprolab/sku is the infra label, got {node_selector!r}"
        )

    def test_no_gpu_plan_uses_single_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Without a gpu_plan the Job must request exactly 1 GPU."""
        cells = [{"id": "gpm4"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        container = k8s.batch.created_jobs[0]["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"


# ---------------------------------------------------------------------------
# 17. bind_run_context gpu_plan round-trip
# ---------------------------------------------------------------------------

class TestBindRunContextGpuPlan:
    def test_gpu_plan_round_trips_via_get_gpu_plan(self):
        """bind_run_context(gpu_plan=...) must be readable via _get_gpu_plan()."""
        plan = _FakeGpuPlan(short_name="azure_a10_24", gpu_count=1)
        with bind_run_context(gpu_plan=plan):
            retrieved = kjcr._get_gpu_plan()
        assert retrieved is plan

    def test_no_gpu_plan_returns_none(self):
        with bind_run_context():
            assert kjcr._get_gpu_plan() is None

    def test_gpu_plan_none_explicit_returns_none(self):
        with bind_run_context(gpu_plan=None):
            assert kjcr._get_gpu_plan() is None

    def test_gpu_plan_concurrent_isolation(self, tmp_path: Path):
        """Two concurrent threads must see their own gpu_plan."""
        seen: dict[str, Any] = {}
        barrier = threading.Barrier(2, timeout=5)

        def _thread_a() -> None:
            plan_a = _FakeGpuPlan(short_name="azure_a10_24")
            with bind_run_context(gpu_plan=plan_a):
                barrier.wait()
                seen["a"] = kjcr._get_gpu_plan()

        def _thread_b() -> None:
            with bind_run_context(gpu_plan=None):
                barrier.wait()
                seen["b"] = kjcr._get_gpu_plan()

        ta = threading.Thread(target=_thread_a)
        tb = threading.Thread(target=_thread_b)
        ta.start(); tb.start()
        ta.join(timeout=10); tb.join(timeout=10)

        assert seen["a"] is not None
        assert seen["a"].short_name == "azure_a10_24"
        assert seen["b"] is None

    def test_all_three_context_vars_coexist(self):
        """run_budget, event_sink, and gpu_plan can all be bound simultaneously."""
        from backend.agents.resilience.budget import RunBudget
        events: list = []
        plan = _FakeGpuPlan(short_name="azure_a100_80")
        budget = RunBudget(max_run_gpu_usd=100.0)

        with bind_run_context(
            run_budget=budget,
            event_sink=lambda t, p: events.append((t, p)),
            gpu_plan=plan,
        ):
            assert kjcr._get_run_budget() is budget
            assert kjcr._get_gpu_plan() is plan
            # event_sink: call it directly to verify it's wired.
            kjcr._get_event_sink()("run_warning", {"code": "test"})

        assert any(e[0] == "run_warning" for e in events)


# ---------------------------------------------------------------------------
# 18. SKU escalation on oom_failed
# ---------------------------------------------------------------------------

class _OomThenSucceedBatch:
    """FakeK8sBatch that returns Failed(exit 42) for the first cell and Succeeded for resubmit."""

    def __init__(self) -> None:
        self.created_jobs: list[dict] = []
        self._call_count = 0

    def create_namespaced_job(self, namespace: str, body: dict) -> None:
        self.created_jobs.append(body)

    def read_namespaced_job_status(self, name: str, namespace: str) -> Any:
        # First submit → oom (Failed + exit 42)
        # Escalated submit → Succeeded (exit 0)
        # Distinguish by number of created jobs so far.
        # call_count tracks how many status reads have happened.
        self._call_count += 1
        if len(self.created_jobs) <= 1:
            # First job: return Failed condition
            return _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))
        else:
            # Escalated job: return Succeeded
            return _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))

    def delete_namespaced_job(self, name: str, namespace: str, **kwargs: Any) -> None:
        pass


class _StatefulCore:
    """Fake CoreV1Api that returns exit-42 pod for job 1, exit-0 pod for job 2+."""

    def __init__(self) -> None:
        self._batch_ref: _OomThenSucceedBatch | None = None

    def list_namespaced_pod(self, namespace: str, label_selector: str = "") -> _FakePodList:
        # Determine which job number we're on based on batch.created_jobs.
        n_jobs = len(self._batch_ref.created_jobs) if self._batch_ref else 1
        if n_jobs <= 1:
            return _FakePodList([_FakePod(exit_code=42, phase="Running")])
        return _FakePodList([_FakePod(exit_code=0, phase="Running")])

    def read_namespaced_pod_log(self, name: str, namespace: str, **kwargs: Any) -> str:
        return "log text\n"


class TestSkuEscalation:
    """OOM escalation: cell oom_failed → escalate to bigger pool, emit gpu_escalated."""

    def _make_oom_then_ok_k8s(self) -> tuple[_OomThenSucceedBatch, _K8sClients]:
        batch = _OomThenSucceedBatch()
        core = _StatefulCore()
        core._batch_ref = batch
        return batch, _K8sClients(batch=batch, core=core, watch_cls=None)

    def test_escalation_resubmits_on_oom_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When a cell oom_fails and ladder has a provisioned SKU, a second Job is submitted."""
        cells = [{"id": "esc0"}]
        batch, k8s = self._make_oom_then_ok_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.4})

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2",),
        )

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": ["azure_a100_80x2"],  # provisioned
            "dynamic_gpu_max_escalations": 2,
        }.get(name, default))

        with bind_run_context(gpu_plan=plan):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        # Two Jobs must have been created: original + escalated.
        assert len(batch.created_jobs) == 2, (
            f"expected 2 Jobs (original + escalated), got {len(batch.created_jobs)}"
        )

    def test_escalation_targets_bigger_sku_node_selector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The escalated Job must have nodeSelector reprolab/sku=<next_sku>."""
        cells = [{"id": "esc1"}]
        batch, k8s = self._make_oom_then_ok_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.3})

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2",),
        )

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": ["azure_a100_80x2"],
            "dynamic_gpu_max_escalations": 2,
        }.get(name, default))

        with bind_run_context(gpu_plan=plan):
            run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        assert len(batch.created_jobs) == 2
        escalated_job = batch.created_jobs[1]
        node_selector = escalated_job["spec"]["template"]["spec"]["nodeSelector"]
        assert node_selector.get("reprolab/sku") == "azure_a100_80x2", (
            f"escalated job must target azure_a100_80x2, got {node_selector!r}"
        )

    def test_escalation_emits_gpu_escalated_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Each escalation must emit a gpu_escalated event via the event_sink."""
        cells = [{"id": "esc2"}]
        batch, k8s = self._make_oom_then_ok_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.3})

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2",),
        )

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": ["azure_a100_80x2"],
            "dynamic_gpu_max_escalations": 2,
        }.get(name, default))

        events: list[tuple[str, dict]] = []

        with bind_run_context(gpu_plan=plan, event_sink=lambda t, p: events.append((t, p))):
            run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        escalated_events = [e for e in events if e[0] == "gpu_escalated"]
        assert len(escalated_events) == 1, (
            f"expected 1 gpu_escalated event, got {escalated_events!r}"
        )
        payload = escalated_events[0][1]
        assert payload["cell_id"] == "esc2"
        assert payload["from_sku"] == "azure_a100_80"
        assert payload["to_sku"] == "azure_a100_80x2"

    def test_escalation_final_result_ok_after_escalate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """After successful escalation the cell result must be 'ok'."""
        cells = [{"id": "esc3"}]
        batch, k8s = self._make_oom_then_ok_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.8})

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2",),
        )

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": ["azure_a100_80x2"],
            "dynamic_gpu_max_escalations": 2,
        }.get(name, default))

        with bind_run_context(gpu_plan=plan):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        assert results["esc3"]["status"] == "ok", (
            f"expected ok after escalation, got {results['esc3']['status']!r}"
        )


# ---------------------------------------------------------------------------
# 19. Escalation graceful degrade
# ---------------------------------------------------------------------------

class TestEscalationGraceDegrade:
    """When escalation cannot proceed, cell stays oom_failed — never crashes/loops."""

    def _make_always_oom_k8s(self) -> _K8sClients:
        """K8s that always returns Failed + exit 42."""
        jobs = [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Failed")]))] * 20
        pods = [_FakePod(exit_code=42, phase="Running")]
        return _make_k8s(job_sequence=jobs, pods=pods)

    def _common_settings(self, monkeypatch: pytest.MonkeyPatch, provisioned: list[str]) -> None:
        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": provisioned,
            "dynamic_gpu_max_escalations": 2,
        }.get(name, default))

    def test_empty_ladder_stays_oom_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Empty ladder_remaining → single Job, cell stays oom_failed."""
        cells = [{"id": "dg0"}]
        k8s = self._make_always_oom_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)
        self._common_settings(monkeypatch, provisioned=["azure_a100_80x2"])

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=(),  # empty
        )

        with bind_run_context(gpu_plan=plan):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        assert results["dg0"]["status"] == "oom_failed", (
            f"expected oom_failed with empty ladder, got {results['dg0']['status']!r}"
        )
        # Only ONE Job submitted (no escalation).
        assert len(k8s.batch.created_jobs) == 1, (
            f"expected 1 Job with empty ladder, got {len(k8s.batch.created_jobs)}"
        )

    def test_not_provisioned_stays_oom_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Ladder has next SKU but it's not in azure_gpu_skus → stays oom_failed."""
        cells = [{"id": "dg1"}]
        k8s = self._make_always_oom_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)
        # azure_gpu_skus does NOT include azure_a100_80x2.
        self._common_settings(monkeypatch, provisioned=["azure_a10_24"])

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2",),  # not provisioned
        )

        with bind_run_context(gpu_plan=plan):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        assert results["dg1"]["status"] == "oom_failed"
        # Only ONE Job submitted.
        assert len(k8s.batch.created_jobs) == 1

    def test_escalation_cap_respected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Escalations are bounded by dynamic_gpu_max_escalations (default 2)."""
        cells = [{"id": "dg2"}]
        k8s = self._make_always_oom_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)
        # All SKUs provisioned.
        self._common_settings(
            monkeypatch,
            provisioned=["azure_a100_80x2", "azure_a100_80x4"],
        )
        # Override max_escalations to 1.
        _orig_setting = kjcr._setting

        def _patched_setting(name: str, default: Any = None) -> Any:
            if name == "dynamic_gpu_max_escalations":
                return 1
            return {
                "azure_namespace": "reprolab",
                "azure_service_account": "reprolab-sa",
                "azure_node_pool_name": "gpunodes",
                "azure_base_image": "img:latest",
                "azure_storage_account": "acct",
                "azure_blob_container": "ctr",
                "azure_files_share": "share",
                "azure_max_nodes": 4,
                "azure_gpu_usd_per_hour": 3.5,
                "azure_pending_timeout_seconds": 900,
                "azure_gpu_skus": ["azure_a100_80x2", "azure_a100_80x4"],
            }.get(name, default)

        monkeypatch.setattr(kjcr, "_setting", _patched_setting)

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2", "azure_a100_80x4"),
        )

        with bind_run_context(gpu_plan=plan):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        # With max_escalations=1: original + 1 escalation = 2 Jobs max.
        assert len(k8s.batch.created_jobs) <= 2, (
            f"expected ≤2 Jobs with max_escalations=1, got {len(k8s.batch.created_jobs)}"
        )
        # Cell stays oom_failed (all attempts OOM'd).
        assert results["dg2"]["status"] == "oom_failed"

    def test_no_gpu_plan_no_escalation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Without a gpu_plan, oom_failed cells are not escalated."""
        cells = [{"id": "dg3"}]
        k8s = self._make_always_oom_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)
        self._common_settings(monkeypatch, provisioned=["azure_a100_80x2"])

        # No gpu_plan bound.
        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            max_parallel=1,
        )

        assert results["dg3"]["status"] == "oom_failed"
        # Only ONE Job (no escalation without a plan).
        assert len(k8s.batch.created_jobs) == 1

    def test_escalation_no_crash_no_loop_when_every_attempt_ooms(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Even if every escalated attempt also OOMs, run_matrix returns cleanly."""
        cells = [{"id": "dg4"}]
        k8s = self._make_always_oom_k8s()
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)
        self._common_settings(
            monkeypatch,
            provisioned=["azure_a100_80x2", "azure_a100_80x4"],
        )

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2", "azure_a100_80x4"),
        )

        # Must complete without hanging or raising.
        with bind_run_context(gpu_plan=plan):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        assert "dg4" in results
        assert results["dg4"]["status"] == "oom_failed"
        # Jobs ≤ 1 (original) + max_escalations (2 from default) = 3 max.
        assert len(k8s.batch.created_jobs) <= 3


# ---------------------------------------------------------------------------
# 20. _trim_ladder helper
# ---------------------------------------------------------------------------

class TestTrimLadder:
    def test_trim_removes_used_and_prior(self):
        ladder = ("a", "b", "c", "d")
        result = kjcr._trim_ladder(ladder, "b")
        assert result == ("c", "d")

    def test_trim_last_element_returns_empty(self):
        ladder = ("a", "b")
        result = kjcr._trim_ladder(ladder, "b")
        assert result == ()

    def test_trim_not_found_returns_full_ladder(self):
        ladder = ("a", "b", "c")
        result = kjcr._trim_ladder(ladder, "x")
        assert result == ("a", "b", "c")

    def test_trim_first_element(self):
        ladder = ("a", "b", "c")
        result = kjcr._trim_ladder(ladder, "a")
        assert result == ("b", "c")


# ---------------------------------------------------------------------------
# 21. _resolve_escalation_sku helper
# ---------------------------------------------------------------------------

class TestResolveEscalationSku:
    def test_returns_first_provisioned(self):
        ladder = ("azure_a100_80x2", "azure_a100_80x4")
        provisioned = ["azure_a100_80x2", "azure_a100_80x4"]
        assert kjcr._resolve_escalation_sku(ladder, provisioned) == "azure_a100_80x2"

    def test_skips_unprovisioned(self):
        ladder = ("azure_a100_80x2", "azure_a100_80x4")
        provisioned = ["azure_a100_80x4"]  # only the bigger one
        assert kjcr._resolve_escalation_sku(ladder, provisioned) == "azure_a100_80x4"

    def test_empty_ladder_returns_none(self):
        assert kjcr._resolve_escalation_sku((), ["azure_a100_80x2"]) is None

    def test_none_provisioned_returns_none(self):
        ladder = ("azure_a100_80x2",)
        assert kjcr._resolve_escalation_sku(ladder, []) is None

    def test_no_overlap_returns_none(self):
        ladder = ("azure_a100_80x2",)
        provisioned = ["azure_a10_24"]
        assert kjcr._resolve_escalation_sku(ladder, provisioned) is None


# ---------------------------------------------------------------------------
# 22. P0-fix-2 — escalation Job-name uniqueness
# ---------------------------------------------------------------------------

class TestEscalationJobNameUniqueness:
    """P0-fix-2: escalated Jobs must have a different K8s name to avoid 409."""

    def test_escalated_job_has_unique_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The original and escalated Jobs must have distinct names."""
        cells = [{"id": "uniq0"}]
        batch = _OomThenSucceedBatch()
        core = _StatefulCore()
        core._batch_ref = batch
        k8s = _K8sClients(batch=batch, core=core, watch_cls=None)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2",),
        )

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": ["azure_a100_80x2"],
            "dynamic_gpu_max_escalations": 2,
            "azure_ttl_seconds_after_finished": 3600,
            "azure_job_backoff_limit": 0,
            "azure_cache_mount_path": "/mnt/reprolab-cache",
            "azure_watch_poll_interval_s": 0.001,
        }.get(name, default))

        with bind_run_context(gpu_plan=plan):
            run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                max_parallel=1,
            )

        assert len(batch.created_jobs) == 2, "expected original + 1 escalated Job"
        name0 = batch.created_jobs[0]["metadata"]["name"]
        name1 = batch.created_jobs[1]["metadata"]["name"]
        assert name0 != name1, (
            f"P0-fix-2: escalated Job must have a different name; got both={name0!r}"
        )
        # Escalated name must carry the '-e1' suffix to identify it.
        assert "e1" in name1, (
            f"P0-fix-2: escalated name should contain suffix 'e1', got {name1!r}"
        )


# ---------------------------------------------------------------------------
# 23. P0-fix-3 — no-gpu_plan nodeSelector fallback uses reprolab/sku
# ---------------------------------------------------------------------------

class TestNodeSelectorFallback:
    """P0-fix-3: without gpu_plan the nodeSelector must be reprolab/sku=<default_sku>."""

    def test_fallback_uses_reprolab_sku_not_agentpool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cells = [{"id": "ns_fallback0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.3})

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": ["azure_a100_80"],
            "dynamic_gpu_max_escalations": 2,
            "azure_ttl_seconds_after_finished": 3600,
            "azure_job_backoff_limit": 0,
            "azure_cache_mount_path": "/mnt/reprolab-cache",
            "azure_watch_poll_interval_s": 0.001,
        }.get(name, default))

        # No gpu_plan bound.
        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert len(k8s.batch.created_jobs) == 1
        node_selector = k8s.batch.created_jobs[0]["spec"]["template"]["spec"]["nodeSelector"]
        assert "reprolab/sku" in node_selector, (
            f"P0-fix-3: fallback must use reprolab/sku, got {node_selector!r}"
        )
        assert node_selector["reprolab/sku"] == "azure_a100_80", (
            f"P0-fix-3: fallback sku must be first in azure_gpu_skus, "
            f"got {node_selector['reprolab/sku']!r}"
        )
        assert "agentpool" not in node_selector, (
            f"P0-fix-3: agentpool must not appear, got {node_selector!r}"
        )


# ---------------------------------------------------------------------------
# 24. P0-fix-4 — escalated SKU billed at correct rate + budget recheck
# ---------------------------------------------------------------------------

class TestEscalationBudgetRecheck:
    """P0-fix-4: escalated rate is read from catalog; budget rechecked before resubmit."""

    def test_escalation_blocked_when_escalated_rate_exceeds_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If the escalated SKU's rate would push over budget, no second Job is submitted."""
        from backend.agents.resilience.budget import RunBudget

        cells = [{"id": "budget_esc0"}]
        # Always OOM.
        k8s = _make_k8s(
            job_sequence=[_FakeJob(_FakeJobStatus(
                conditions=[_FakeJobCondition("Failed")]
            ))] * 10,
            pods=[_FakePod(exit_code=42)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            # Budget: tiny — base rate ok at small timeout, escalated rate over.
            "azure_gpu_usd_per_hour": 0.001,
            "azure_pending_timeout_seconds": 900,
            "azure_gpu_skus": ["azure_a100_80x2"],
            "dynamic_gpu_max_escalations": 2,
            "azure_ttl_seconds_after_finished": 3600,
            "azure_job_backoff_limit": 0,
            "azure_cache_mount_path": "/mnt/reprolab-cache",
            "azure_watch_poll_interval_s": 0.001,
        }.get(name, default))

        # Patch catalog lookup so the escalated SKU returns a very high rate.
        def _fake_lookup(short_name: str) -> Any:
            class _FakeSku:
                approx_usd_per_hr = 1000.0
                gpu_count = 2
            return _FakeSku() if short_name == "azure_a100_80x2" else None

        monkeypatch.setattr(kjcr, "_lookup_sku_by_short_name", _fake_lookup)

        plan = _FakeGpuPlan(
            short_name="azure_a100_80",
            gpu_count=1,
            ladder_remaining=("azure_a100_80x2",),
        )
        # Budget that is already nearly consumed.
        budget = RunBudget(max_run_gpu_usd=0.0001)

        events: list[tuple[str, dict]] = []

        with bind_run_context(
            gpu_plan=plan,
            run_budget=budget,
            event_sink=lambda t, p: events.append((t, p)),
        ):
            results = run_matrix(
                cells,
                tmp_path / "train_cell.py",
                output_root=tmp_path / "out",
                per_cell_timeout_s=1.0,
                max_parallel=1,
            )

        # Either budget stopped the cell from being submitted at all (budget_exceeded),
        # or the escalation was blocked (oom_failed with only 1 Job).
        # Either way, at most 1 Job was submitted (no escalation over budget).
        assert len(k8s.batch.created_jobs) <= 1, (
            f"P0-fix-4: escalation over budget must not submit a second Job; "
            f"got {len(k8s.batch.created_jobs)} Jobs"
        )


# ---------------------------------------------------------------------------
# 25. P1-fix-7 — job.status None guard
# ---------------------------------------------------------------------------

class TestJobStatusNoneGuard:
    """P1-fix-7: job.status=None in transitional states must keep polling, not crash."""

    def test_status_none_then_succeeded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Polling should continue past a None-status response and eventually succeed."""
        cells = [{"id": "null_status0"}]

        # First read returns a job with status=None; second read returns Complete.
        null_job = _FakeJob(None)  # type: ignore[arg-type]
        succeed_job = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))
        k8s = _make_k8s(
            job_sequence=[null_job, succeed_job],
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
        )
        # Must not crash; must eventually resolve.
        assert "null_status0" in results
        assert results["null_status0"]["status"] == "ok"

    def test_status_none_does_not_raise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """job.status=None must not raise AttributeError."""
        cells = [{"id": "null_status1"}]
        null_job = _FakeJob(None)  # type: ignore[arg-type]
        # Always None then finally succeed.
        succeed = _FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")]))
        k8s = _make_k8s(
            job_sequence=[null_job, null_job, succeed],
            pods=[_FakePod(exit_code=0)],
        )
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        # Should complete without AttributeError.
        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert "null_status1" in results


# ---------------------------------------------------------------------------
# 26. P0-fix-1 — env-var names in Job manifest (runner injects OPENRESEARCH_AZURE_*)
# ---------------------------------------------------------------------------

class TestEnvVarNamesInManifest:
    """P0-fix-1: verify the canonical env-var names appear in the Job manifest."""

    # Canonical contract: runner injects these names; entrypoint reads the same names.
    _REQUIRED_ENV_NAMES = {
        "OPENRESEARCH_CELL_ID",
        "OPENRESEARCH_CELL_PARAMS",
        "OPENRESEARCH_CELL_OUTPUT_DIR",
        "OPENRESEARCH_CELL_MAX_OOM_RETRIES",
        "OPENRESEARCH_AZURE_STORAGE_ACCOUNT",   # P0-fix-1: was OPENRESEARCH_BLOB_ACCOUNT
        "OPENRESEARCH_AZURE_BLOB_CONTAINER",     # P0-fix-1: was OPENRESEARCH_BLOB_CONTAINER
        "OPENRESEARCH_BLOB_CODE_PREFIX",
        "OPENRESEARCH_BLOB_OUTPUT_PREFIX",
        "OPENRESEARCH_CACHE_MOUNT",
    }

    def test_manifest_contains_all_required_env_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cells = [{"id": "env0"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert len(k8s.batch.created_jobs) == 1
        env_list: list[dict] = (
            k8s.batch.created_jobs[0]["spec"]["template"]["spec"]["containers"][0]["env"]
        )
        env_names = {e["name"] for e in env_list}

        missing = self._REQUIRED_ENV_NAMES - env_names
        assert not missing, (
            f"P0-fix-1: manifest is missing these required env var names: {missing!r}\n"
            f"All injected names: {sorted(env_names)}"
        )

    def test_old_blob_account_name_not_injected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The old OPENRESEARCH_BLOB_ACCOUNT name (pre-fix) must NOT appear."""
        cells = [{"id": "env1"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        env_list: list[dict] = (
            k8s.batch.created_jobs[0]["spec"]["template"]["spec"]["containers"][0]["env"]
        )
        env_names = {e["name"] for e in env_list}
        assert "OPENRESEARCH_BLOB_ACCOUNT" not in env_names, (
            "OPENRESEARCH_BLOB_ACCOUNT is the old name (pre-fix); entrypoint reads "
            "OPENRESEARCH_AZURE_STORAGE_ACCOUNT.  Remove the old name."
        )
        assert "OPENRESEARCH_BLOB_CONTAINER" not in env_names, (
            "OPENRESEARCH_BLOB_CONTAINER is the old name (pre-fix); entrypoint reads "
            "OPENRESEARCH_AZURE_BLOB_CONTAINER.  Remove the old name."
        )


# ---------------------------------------------------------------------------
# 27. P1-fix-5 — empty base_image raises at submit, not silently uses :latest
# ---------------------------------------------------------------------------

class TestEmptyBaseImageError:
    def test_empty_base_image_raises_value_error(self):
        """_build_job_manifest must raise ValueError when base_image is empty."""
        with pytest.raises(ValueError, match="azure_base_image is empty"):
            kjcr._build_job_manifest(
                job_name="test-job",
                namespace="reprolab",
                service_account="reprolab-sa",
                node_pool_name="gpunodes",
                base_image="",  # empty → should raise
                storage_account="myacct",
                blob_container="myctr",
                files_share="share",
                cell_id="c0",
                cell_params_json="{}",
                output_blob_prefix="runs/r1/cells",
                code_blob_prefix="runs/r1/code",
                active_deadline_seconds=3600,
                max_oom_retries=2,
                fingerprint=None,
                now_iso=None,
            )

    def test_non_empty_base_image_does_not_raise(self):
        """_build_job_manifest must succeed when base_image is non-empty."""
        manifest = kjcr._build_job_manifest(
            job_name="test-job",
            namespace="reprolab",
            service_account="reprolab-sa",
            node_pool_name="gpunodes",
            base_image="myregistry.io/image:v1",
            storage_account="myacct",
            blob_container="myctr",
            files_share="share",
            cell_id="c0",
            cell_params_json="{}",
            output_blob_prefix="runs/r1/cells",
            code_blob_prefix="runs/r1/code",
            active_deadline_seconds=3600,
            max_oom_retries=2,
            fingerprint=None,
            now_iso=None,
        )
        assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == \
            "myregistry.io/image:v1"


# ---------------------------------------------------------------------------
# 28. P0-scale-1 — watch loop pod-list call count is bounded (not per-poll)
# ---------------------------------------------------------------------------

class _CountingCore:
    """FakeK8sCore that counts list_namespaced_pod calls."""

    def __init__(
        self,
        pods: list[_FakePod] | None = None,
        log_text: str = "ok\n",
    ) -> None:
        self._pods = pods or []
        self._log_text = log_text
        self.list_pod_call_count = 0

    def list_namespaced_pod(self, namespace: str, label_selector: str = "") -> _FakePodList:
        self.list_pod_call_count += 1
        return _FakePodList(self._pods)

    def read_namespaced_pod_log(self, name: str, namespace: str, **kwargs: Any) -> str:
        return self._log_text


class TestWatchLoopPodListCount:
    """P0-scale-1: list_namespaced_pod must NOT be called on every poll of a running job."""

    def test_healthy_job_pod_list_bounded_not_per_poll(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A job that runs for N polls then completes must call list_namespaced_pod
        a SMALL number of times (for terminal info collection only), not once per poll.

        We use 5 'running' polls then 1 terminal poll.  With the old code that would
        be 5*2 + 1*1 = 11+ list calls.  With the fix it should be ≤ 2 (one for
        _collect_pod_info at terminal time, possibly one for Pending check on poll 1
        before active>0 is seen — but once active>0 no more pod-list calls).
        """
        cells = [{"id": "sc1"}]

        # Build a job sequence: 5 polls with active=1 (running), then terminal.
        running_status = _FakeJobStatus(succeeded=0, failed=0)
        # Patch active counter — the fake class doesn't have it by default so we add it.
        running_status.active = 1  # type: ignore[attr-defined]

        terminal_status = _FakeJobStatus(conditions=[_FakeJobCondition("Complete")])
        terminal_status.active = 0  # type: ignore[attr-defined]

        running_job = _FakeJob(running_status)
        terminal_job = _FakeJob(terminal_status)

        n_running_polls = 5
        job_sequence = [running_job] * n_running_polls + [terminal_job]

        batch = FakeK8sBatch(job_sequence)
        counting_core = _CountingCore(pods=[_FakePod(exit_code=0, phase="Running")])
        k8s = _K8sClients(batch=batch, core=counting_core, watch_cls=None)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.5})

        # Fast poll interval so the test doesn't take too long.
        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_watch_poll_interval_s": 0.001,
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
        }.get(name, default))

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert results["sc1"]["status"] == "ok", (
            f"expected ok, got {results['sc1']['status']!r}"
        )
        # The key assertion: pod-list call count must be <= 2 (terminal collection
        # + at most one Pending check on the very first poll before active>0 is seen).
        # It must NOT be >= n_running_polls (which the old code would produce).
        assert counting_core.list_pod_call_count <= 2, (
            f"P0-scale-1: list_namespaced_pod called {counting_core.list_pod_call_count} times "
            f"for {n_running_polls} running polls + 1 terminal poll; "
            f"expected ≤ 2 (terminal collection only, no per-poll pod listing)"
        )

    def test_pending_timeout_still_uses_pod_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When the job is truly stuck (active=0, succeeded=0, failed=0), the watch loop
        MUST call list_namespaced_pod to check the pod phase for Pending-timeout.
        This verifies the stuck-Pending path is not accidentally skipped.
        """
        cells = [{"id": "sc_pend"}]

        # All polls: no active/succeeded/failed — pod is stuck Pending.
        empty_status = _FakeJobStatus(succeeded=0, failed=0)
        empty_status.active = 0  # type: ignore[attr-defined]
        pending_job = _FakeJob(empty_status)

        job_sequence = [pending_job] * 50
        batch = FakeK8sBatch(job_sequence)
        # Core returns a Pending pod.
        counting_core = _CountingCore(pods=[_FakePod(phase="Pending", exit_code=None)])
        k8s = _K8sClients(batch=batch, core=counting_core, watch_cls=None)
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics=None)

        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_watch_poll_interval_s": 0.001,
            "azure_pending_timeout_seconds": 0.05,  # 50ms → fires quickly
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "acct",
            "azure_blob_container": "ctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
        }.get(name, default))

        results = run_matrix(
            cells,
            tmp_path / "train_cell.py",
            output_root=tmp_path / "out",
            per_cell_timeout_s=30.0,
        )

        r = results["sc_pend"]
        assert r["status"] == "error", f"expected error, got {r['status']!r}"
        assert r["error"] is not None and "capacity_exhausted" in r["error"], (
            f"expected capacity_exhausted prefix, got {r['error']!r}"
        )
        # Pending detection MUST have called list_namespaced_pod at least once.
        assert counting_core.list_pod_call_count >= 1, (
            "P0-scale-1: Pending-timeout detection must use list_namespaced_pod"
        )


# ---------------------------------------------------------------------------
# 29. P0-scale-2 — ContainerClient built at most once per run_matrix
# ---------------------------------------------------------------------------

class TestContainerClientReuse:
    """P0-scale-2: _make_blob_client must be called at most once per run_matrix invocation,
    even when multiple cells share the same matrix call.
    """

    def test_blob_client_constructed_once_for_multi_cell_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A 3-cell run must call _make_blob_client exactly once."""
        n_cells = 3
        cells = [{"id": f"cl{i}"} for i in range(n_cells)]

        jobs = [_FakeJob(_FakeJobStatus(conditions=[_FakeJobCondition("Complete")])) for _ in range(n_cells * 2)]
        k8s = _make_k8s(job_sequence=jobs, pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.3})

        # Replace _make_blob_client with a counter; return a fake client object.
        construction_count = [0]
        fake_client = object()  # sentinel — any object will do

        def _fake_make_blob_client(account_name: str, container_name: str) -> Any:
            construction_count[0] += 1
            return fake_client

        monkeypatch.setattr(kjcr, "_make_blob_client", _fake_make_blob_client)
        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_watch_poll_interval_s": 0.001,
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "myacct",
            "azure_blob_container": "myctr",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
        }.get(name, default))

        results = run_matrix(
            cells, tmp_path / "train_cell.py", output_root=tmp_path / "out",
            max_parallel=n_cells,
        )

        # All cells should succeed.
        for i in range(n_cells):
            assert results[f"cl{i}"]["status"] == "ok", (
                f"cl{i}: expected ok, got {results[f'cl{i}']['status']!r}"
            )

        # The factory must have been called exactly once, regardless of cell count.
        assert construction_count[0] == 1, (
            f"P0-scale-2: _make_blob_client called {construction_count[0]} times "
            f"for {n_cells} cells; expected exactly 1 (shared client)"
        )

    def test_blob_client_none_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When _make_blob_client returns None (storage unconfigured), run_matrix
        must still succeed — the helpers fall back to constructing their own client
        or handling the error gracefully.
        """
        cells = [{"id": "cl_none"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.7})

        # Force client to be None (storage not configured).
        monkeypatch.setattr(kjcr, "_make_blob_client", lambda a, c: None)

        results = run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")
        assert results["cl_none"]["status"] == "ok", (
            f"expected ok with None blob client, got {results['cl_none']['status']!r}"
        )

    def test_blob_client_factory_called_with_correct_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """_make_blob_client must receive the configured account_name and container_name."""
        cells = [{"id": "cl_args"}]
        k8s = _make_k8s(job_sequence=_succeeded_job(), pods=[_FakePod(exit_code=0)])
        kjcr._k8s_clients_override = k8s
        _patch_blob(monkeypatch, metrics={"metric": 0.1})

        received_args: list[tuple[str, str]] = []

        def _capturing_factory(account_name: str, container_name: str) -> None:
            received_args.append((account_name, container_name))
            return None

        monkeypatch.setattr(kjcr, "_make_blob_client", _capturing_factory)
        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {
            "azure_watch_poll_interval_s": 0.001,
            "azure_namespace": "reprolab",
            "azure_service_account": "reprolab-sa",
            "azure_node_pool_name": "gpunodes",
            "azure_base_image": "img:latest",
            "azure_storage_account": "correct-account",
            "azure_blob_container": "correct-container",
            "azure_files_share": "share",
            "azure_max_nodes": 4,
            "azure_gpu_usd_per_hour": 3.5,
            "azure_pending_timeout_seconds": 900,
        }.get(name, default))

        run_matrix(cells, tmp_path / "train_cell.py", output_root=tmp_path / "out")

        assert len(received_args) == 1, f"expected 1 factory call, got {len(received_args)}"
        assert received_args[0] == ("correct-account", "correct-container"), (
            f"P0-scale-2: factory called with wrong args: {received_args[0]!r}"
        )
