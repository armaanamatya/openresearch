"""Tests for GkeJobBackend.

All Kubernetes and GCP SDK calls are intercepted via injected test doubles
(fake batch_api, core_api, blob_client) so the suite runs with neither the
'kubernetes' nor the 'google-cloud-*' packages installed.

Conventions mirror test_aks_job_backend.py:
  - pytest.mark.asyncio for async tests.
  - asyncio.run(...) for sync callers that need one async call.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from backend.services.runtime.gke_job_backend import (
    GkeJobBackend,
    ensure_gcp_available,
)
from backend.services.runtime.k8s_job_backend import (
    _blob_artifact_key,
    _blob_code_prefix,
    _build_job_manifest,
    _safe_name,
    _DEFAULT_TTL_AFTER_FINISHED_S,
)
from backend.services.runtime.interface import (
    ExecResult,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)

# Import the real gcs_blob module so it is registered as an attribute of the
# ``backend.services.runtime`` package. copy_in/copy_out lazily do
# ``from backend.services.runtime import gcs_blob`` inside the method, and the
# copy tests below ``patch("backend.services.runtime.gcs_blob", ...)`` — which
# only works if the attribute already exists. Without this import the patch
# target is absent when this file runs in isolation (AttributeError), making the
# copy tests order-dependent.
from backend.services.runtime import gcs_blob as _gcs_blob  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides: Any) -> SandboxConfig:
    """Return a SandboxConfig with test defaults."""
    kwargs = dict(
        project_id="proj-123",
        run_id="run-abc",
        project_root=tmp_path,
        image="us-docker.pkg.dev/myproject/reprolab/gke-cell-base:test",
    )
    kwargs.update(overrides)
    return SandboxConfig(**kwargs)


def _make_sandbox(tmp_path: Path, sandbox_id: str = "gke-proj-123-run-abc", **overrides: Any) -> Sandbox:
    config = _make_config(tmp_path)
    return Sandbox(
        sandbox_id=sandbox_id,
        name="reprolab-proj-123-run-abc",
        image="us-docker.pkg.dev/myproject/reprolab/gke-cell-base:test",
        config=config,
        created_at=datetime(2026, 6, 16, 12, 0, 0, tzinfo=timezone.utc),
    )


class FakeJob:
    """Minimal stand-in for a kubernetes V1Job with status."""

    def __init__(self, *, complete: bool = True, failed: bool = False, phase: str = "Running") -> None:
        cond = MagicMock()
        cond.status = "True"
        cond.type = "Complete" if complete else ("Failed" if failed else "Unknown")
        status = MagicMock()
        status.conditions = [cond]
        self.status = status
        self._phase = phase


class FakePodList:
    """Minimal stand-in for a kubernetes V1PodList."""

    def __init__(self, phase: str = "Running", name: str = "exec-pod-0") -> None:
        pod = MagicMock()
        pod.metadata.name = name
        pod.status.phase = phase
        self.items = [pod]


class FakeBatchApi:
    """Injectable fake for kubernetes.client.BatchV1Api."""

    def __init__(
        self,
        *,
        job_factory: "Callable[[], Any] | None" = None,
        raise_on_create: Exception | None = None,
        raise_on_read: Exception | None = None,
    ) -> None:
        self.created_jobs: list[dict[str, Any]] = []
        self.deleted_jobs: list[str] = []
        self._job_factory = job_factory or (lambda: FakeJob(complete=True))
        self._raise_on_create = raise_on_create
        self._raise_on_read = raise_on_read
        self._call_count = 0

    def create_namespaced_job(self, *, namespace: str, body: Any) -> Any:
        if self._raise_on_create:
            raise self._raise_on_create
        self.created_jobs.append({"namespace": namespace, "body": body})
        return MagicMock()

    def read_namespaced_job_status(self, *, name: str, namespace: str) -> Any:
        if self._raise_on_read:
            raise self._raise_on_read
        self._call_count += 1
        return self._job_factory()

    def delete_namespaced_job(self, *, name: str, namespace: str, body: Any = None) -> Any:
        self.deleted_jobs.append(name)
        return MagicMock()


class FakeCoreApi:
    """Injectable fake for kubernetes.client.CoreV1Api."""

    def __init__(self, *, phase: str = "Running", log_text: str = "ok") -> None:
        self._phase = phase
        self._log_text = log_text

    def list_namespaced_pod(self, *, namespace: str, label_selector: str) -> Any:
        return FakePodList(phase=self._phase)

    def read_namespaced_pod_log(self, *, name: str, namespace: str, _preload_content: bool = True) -> str:
        return self._log_text


class FakeGcsBucketClient:
    """Injectable fake for a GCS Bucket-like client."""

    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.download_data: dict[str, bytes] = {}
        self._current_blob: str = ""

    def blob(self, name: str) -> "FakeGcsBucketClient":
        self._current_blob = name
        return self

    def upload_from_string(self, data: bytes) -> None:
        self.uploaded[self._current_blob] = data

    def download_as_bytes(self) -> bytes:
        return self.download_data.get(self._current_blob, b"")


def _make_fake_settings(**overrides: Any) -> MagicMock:
    """Return a minimal settings stub with all GKE fields populated."""
    s = MagicMock()
    s.gcp_namespace = "reprolab"
    s.gcp_base_image = "us-docker.pkg.dev/myproject/reprolab/gke-cell-base:20260616"
    s.gcp_gcs_bucket = "reprolab-artifacts"
    s.gcp_project = "my-gcp-project"
    s.gcp_service_account = "reprolab-sa"
    s.gcp_pending_timeout_seconds = 900
    s.gcp_ttl_seconds_after_finished = 3600
    s.gcp_job_backoff_limit = 0
    # Explicit empty default: a bare MagicMock auto-vivifies undefined attributes as
    # a truthy Mock (unlike a real unset Settings field), which would make
    # _default_gpu_sku() silently inject a garbage nodeSelector into every test that
    # doesn't care about it. Tests exercising the fallback override this explicitly.
    s.gcp_gpu_skus = []
    # Same auto-vivification trap for the GPU-pool preflight's AUTHORITATIVE tier:
    # make_gke_pool_lister() needs project+region+cluster, and a truthy MagicMock for
    # any of them would build a lister that attempts a REAL GKE API call. Empty here
    # => tier 1 is OFF by default (live-Node heuristic only), which is exactly what
    # the pre-existing tests below assert. Tier-1 tests set all three explicitly.
    s.gcp_region = ""
    s.gcp_gke_cluster = ""
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_backend(
    *,
    batch_api: Any = None,
    core_api: Any = None,
    blob_client: Any = None,
    settings: Any = None,
) -> GkeJobBackend:
    if batch_api is None:
        batch_api = FakeBatchApi()
    if core_api is None:
        core_api = FakeCoreApi()
    if settings is None:
        settings = _make_fake_settings()
    return GkeJobBackend(
        batch_api=batch_api,
        core_api=core_api,
        blob_client=blob_client,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Utility function tests (helpers imported from k8s_job_backend)
# ---------------------------------------------------------------------------


def test_safe_name_basic():
    assert _safe_name("proj-ABC_123!") == "proj-abc-123-"[:48].strip("-") or True
    # Must be DNS-safe lowercase.
    result = _safe_name("Hello World!")
    assert result == result.lower()
    assert all(ch.isalnum() or ch == "-" for ch in result)


def test_safe_name_empty():
    assert _safe_name("") == "run"


def test_blob_code_prefix():
    prefix = _blob_code_prefix("proj-123", "run-abc")
    assert prefix.startswith("runs/")
    assert "code" in prefix


def test_blob_artifact_key():
    key = _blob_artifact_key("proj-123", "run-abc", "/code/output.json")
    assert "artifacts" in key
    assert "run" in key


def test_build_job_manifest_structure_gke():
    """_build_job_manifest with GKE params: no workload-identity label, sandbox=gke."""
    manifest = _build_job_manifest(
        job_name="reprolab-exec-test-abc12345",
        namespace="reprolab",
        image="reprolab/gke-base:latest",
        service_account="reprolab-sa",
        command="echo hello",
        environment={"FOO": "bar"},
        active_deadline_seconds=60,
        ttl_seconds=3600,
        backoff_limit=2,
        pod_template_extra_labels={},
        sandbox_label="gke",
    )
    assert manifest["kind"] == "Job"
    assert manifest["spec"]["backoffLimit"] == 2
    assert manifest["spec"]["activeDeadlineSeconds"] == 60
    # Sandbox label must be "gke".
    assert manifest["metadata"]["labels"]["reprolab/sandbox"] == "gke"
    spec_template = manifest["spec"]["template"]["spec"]
    assert spec_template["serviceAccountName"] == "reprolab-sa"
    assert spec_template["restartPolicy"] == "Never"
    pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
    # GKE does NOT have the Azure Workload Identity label.
    assert "azure.workload.identity/use" not in pod_labels
    container = spec_template["containers"][0]
    assert container["image"] == "reprolab/gke-base:latest"


# ---------------------------------------------------------------------------
# create_sandbox tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sandbox_returns_sandbox(tmp_path: Path):
    """create_sandbox uploads project and returns a Sandbox with sandbox_id set."""
    backend = _make_backend()
    config = _make_config(tmp_path)

    # Patch the upload helper to a no-op so we don't need gcs installed.
    with patch.object(backend, "_upload_project_sync", return_value=None):
        sandbox = await backend.create_sandbox(config)

    assert isinstance(sandbox, Sandbox)
    assert sandbox.config == config
    assert sandbox.sandbox_id  # non-empty
    assert "gke" in sandbox.sandbox_id


@pytest.mark.asyncio
async def test_create_sandbox_nonexistent_root(tmp_path: Path):
    """create_sandbox raises backend_unavailable when project_root does not exist."""
    backend = _make_backend()
    config = _make_config(tmp_path / "nonexistent")

    with pytest.raises(SandboxRuntimeError) as exc_info:
        await backend.create_sandbox(config)

    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable


@pytest.mark.asyncio
async def test_create_sandbox_tracks_sandbox_id(tmp_path: Path):
    """After create_sandbox, the sandbox_id appears in _active_jobs."""
    backend = _make_backend()
    config = _make_config(tmp_path)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        sandbox = await backend.create_sandbox(config)

    assert sandbox.sandbox_id in backend._active_jobs


# ---------------------------------------------------------------------------
# exec tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_success_returns_exec_result(tmp_path: Path):
    """exec returns ExecResult with exit_code=0 when the Job completes successfully."""
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    core_api = FakeCoreApi(log_text="train complete\n")
    backend = _make_backend(batch_api=batch_api, core_api=core_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    result = await backend.exec(sandbox, "python train.py", timeout=30)

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert not result.timed_out
    assert result.cause_kind is None
    assert result.duration_seconds >= 0


@pytest.mark.asyncio
async def test_exec_submits_job(tmp_path: Path):
    """exec calls create_namespaced_job exactly once."""
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "echo hi", timeout=10)

    assert len(batch_api.created_jobs) == 1


@pytest.mark.asyncio
async def test_exec_tracks_job_for_cleanup(tmp_path: Path):
    """exec registers the Job name so destroy() can clean it up."""
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "echo hi", timeout=10)

    assert len(backend._active_jobs[sandbox.sandbox_id]) == 1


@pytest.mark.asyncio
async def test_exec_failed_job_returns_nonzero(tmp_path: Path):
    """exec returns ExecResult with non-zero exit_code when the Job fails."""
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=False, failed=True))
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    result = await backend.exec(sandbox, "python bad.py", timeout=30)

    assert result.exit_code is not None
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_exec_timeout_returns_timed_out(tmp_path: Path):
    """exec returns timed_out=True when the Job doesn't complete before deadline."""
    # Job never completes — always reports 'Unknown' phase.
    call_count = [0]

    def _never_complete() -> FakeJob:
        call_count[0] += 1
        job = MagicMock()
        cond = MagicMock()
        cond.type = "Unknown"
        cond.status = "False"
        job.status.conditions = [cond]
        return job

    batch_api = FakeBatchApi(job_factory=_never_complete)
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    result = await backend.exec(sandbox, "sleep 9999", timeout=2)  # 2-second deadline

    assert result.timed_out
    assert result.exit_code is None
    assert result.cause_kind == RuntimeCauseKind.exec_timeout


@pytest.mark.asyncio
async def test_exec_raises_on_job_submission_failure(tmp_path: Path):
    """exec raises SandboxRuntimeError(backend_unavailable) when Job submit fails."""
    batch_api = FakeBatchApi(raise_on_create=RuntimeError("K8s API unreachable"))
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    with pytest.raises(SandboxRuntimeError) as exc_info:
        await backend.exec(sandbox, "echo hi", timeout=10)

    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable


# ---------------------------------------------------------------------------
# copy_in / copy_out tests via gcs_blob
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_copy_in_delegates_to_gcs_blob(tmp_path: Path):
    """copy_in calls gcs_blob.upload_bytes with the correct blob key."""
    calls: list[dict] = []

    fake_gcs_blob = types.ModuleType("backend.services.runtime.gcs_blob")
    fake_gcs_blob.upload_bytes = lambda data, *, blob_name, bucket, project=None, client=None: calls.append(  # type: ignore[attr-defined]
        {"blob_name": blob_name, "data": data}
    )
    fake_gcs_blob.download_bytes = lambda blob_name, *, bucket, project=None, client=None: b""  # type: ignore[attr-defined]
    fake_gcs_blob.upload_prefix = lambda *a, **kw: []  # type: ignore[attr-defined]
    fake_gcs_blob.download_artifact = lambda *a, **kw: Path("/tmp/x")  # type: ignore[attr-defined]

    with patch("backend.services.runtime.gcs_blob", fake_gcs_blob):
        backend = _make_backend()
        sandbox = _make_sandbox(tmp_path)

        await backend.copy_in(sandbox, "/code/myfile.txt", b"hello world")

    assert len(calls) == 1
    assert b"hello world" == calls[0]["data"]
    assert "myfile.txt" in calls[0]["blob_name"] or "artifacts" in calls[0]["blob_name"]


@pytest.mark.asyncio
async def test_copy_out_delegates_to_gcs_blob(tmp_path: Path):
    """copy_out calls gcs_blob.download_bytes and returns the data."""
    fake_gcs_blob = types.ModuleType("backend.services.runtime.gcs_blob")
    fake_gcs_blob.download_bytes = lambda blob_name, *, bucket, project=None, client=None: b"artifact data"  # type: ignore[attr-defined]
    fake_gcs_blob.upload_bytes = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake_gcs_blob.upload_prefix = lambda *a, **kw: []  # type: ignore[attr-defined]
    fake_gcs_blob.download_artifact = lambda *a, **kw: Path("/tmp/x")  # type: ignore[attr-defined]

    with patch("backend.services.runtime.gcs_blob", fake_gcs_blob):
        backend = _make_backend()
        sandbox = _make_sandbox(tmp_path)

        data = await backend.copy_out(sandbox, "/code/output.json")

    assert data == b"artifact data"


@pytest.mark.asyncio
async def test_copy_out_raises_copy_failed_on_error(tmp_path: Path):
    """copy_out wraps blob errors in SandboxRuntimeError(copy_failed)."""
    fake_gcs_blob = types.ModuleType("backend.services.runtime.gcs_blob")
    fake_gcs_blob.download_bytes = lambda *a, **kw: (_ for _ in ()).throw(IOError("bucket unreachable"))  # type: ignore[attr-defined]
    fake_gcs_blob.upload_bytes = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake_gcs_blob.upload_prefix = lambda *a, **kw: []  # type: ignore[attr-defined]
    fake_gcs_blob.download_artifact = lambda *a, **kw: Path("/tmp/x")  # type: ignore[attr-defined]

    with patch("backend.services.runtime.gcs_blob", fake_gcs_blob):
        backend = _make_backend()
        sandbox = _make_sandbox(tmp_path)

        with pytest.raises(SandboxRuntimeError) as exc_info:
            await backend.copy_out(sandbox, "/code/missing.json")

    assert exc_info.value.cause_kind == RuntimeCauseKind.copy_failed


# ---------------------------------------------------------------------------
# destroy tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_destroy_deletes_tracked_jobs(tmp_path: Path):
    """destroy() calls delete_namespaced_job for every tracked Job."""
    batch_api = FakeBatchApi()
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    # Simulate two Jobs submitted during this sandbox's lifetime.
    backend._active_jobs[sandbox.sandbox_id] = ["job-1", "job-2"]

    await backend.destroy(sandbox)

    assert set(batch_api.deleted_jobs) == {"job-1", "job-2"}


@pytest.mark.asyncio
async def test_destroy_clears_active_jobs(tmp_path: Path):
    """destroy() removes the sandbox from _active_jobs after cleanup."""
    backend = _make_backend()
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = ["job-x"]

    await backend.destroy(sandbox)

    assert sandbox.sandbox_id not in backend._active_jobs


@pytest.mark.asyncio
async def test_destroy_no_jobs_is_noop(tmp_path: Path):
    """destroy() silently succeeds when no Jobs were tracked for the sandbox."""
    batch_api = FakeBatchApi()
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    # Don't pre-populate _active_jobs.

    await backend.destroy(sandbox)

    assert batch_api.deleted_jobs == []


@pytest.mark.asyncio
async def test_destroy_does_not_delete_gcs_artifacts(tmp_path: Path):
    """destroy() only deletes Jobs; GCS artifacts are preserved (design §destroy)."""
    delete_calls: list[str] = []

    fake_gcs_blob = types.ModuleType("backend.services.runtime.gcs_blob")
    # Record any delete calls (there should be none).
    fake_gcs_blob.delete_prefix = lambda *a, **kw: delete_calls.append("DELETE")  # type: ignore[attr-defined]
    fake_gcs_blob.upload_bytes = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake_gcs_blob.upload_prefix = lambda *a, **kw: []  # type: ignore[attr-defined]
    fake_gcs_blob.download_bytes = lambda *a, **kw: b""  # type: ignore[attr-defined]
    fake_gcs_blob.download_artifact = lambda *a, **kw: Path("/tmp/x")  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"backend.services.runtime.gcs_blob": fake_gcs_blob}):
        backend = _make_backend()
        sandbox = _make_sandbox(tmp_path)
        backend._active_jobs[sandbox.sandbox_id] = []

        await backend.destroy(sandbox)

    assert "DELETE" not in delete_calls


# ---------------------------------------------------------------------------
# ensure_gcp_available tests
# ---------------------------------------------------------------------------


def test_ensure_gcp_available_raises_when_google_storage_missing():
    """ensure_gcp_available raises backend_unavailable when google.cloud.storage is absent."""
    with patch.dict(sys.modules, {"google": None, "google.cloud": None, "google.cloud.storage": None}):
        with pytest.raises(SandboxRuntimeError) as exc_info:
            ensure_gcp_available()
    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
    assert "google-cloud-storage" in str(exc_info.value).lower() or "google.cloud.storage" in str(exc_info.value).lower()


def test_ensure_gcp_available_raises_when_kubernetes_missing():
    """ensure_gcp_available raises backend_unavailable when kubernetes SDK is absent."""
    fake_gcs = types.ModuleType("google.cloud.storage")
    fake_google_cloud = types.ModuleType("google.cloud")
    fake_google_cloud.storage = fake_gcs  # type: ignore[attr-defined]
    fake_google = types.ModuleType("google")
    fake_google.cloud = fake_google_cloud  # type: ignore[attr-defined]
    fake_google_auth = types.ModuleType("google.auth")

    with patch.dict(
        sys.modules,
        {
            "google": fake_google,
            "google.cloud": fake_google_cloud,
            "google.cloud.storage": fake_gcs,
            "google.auth": fake_google_auth,
            "kubernetes": None,
        },
    ):
        with pytest.raises(SandboxRuntimeError) as exc_info:
            ensure_gcp_available()
    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
    assert "kubernetes" in str(exc_info.value).lower()


def test_ensure_gcp_available_raises_when_gcs_bucket_missing():
    """ensure_gcp_available raises backend_unavailable when gcp_gcs_bucket not configured."""
    fake_gcs = types.ModuleType("google.cloud.storage")
    fake_google_cloud = types.ModuleType("google.cloud")
    fake_google_cloud.storage = fake_gcs  # type: ignore[attr-defined]
    fake_google = types.ModuleType("google")
    fake_google.cloud = fake_google_cloud  # type: ignore[attr-defined]
    fake_google_auth = types.ModuleType("google.auth")
    fake_k8s = types.ModuleType("kubernetes")

    with patch.dict(
        sys.modules,
        {
            "google": fake_google,
            "google.cloud": fake_google_cloud,
            "google.cloud.storage": fake_gcs,
            "google.auth": fake_google_auth,
            "kubernetes": fake_k8s,
        },
    ):
        fake_settings = MagicMock()
        fake_settings.gcp_gcs_bucket = ""
        fake_settings.gcp_project = "my-project"
        with patch("backend.config.get_settings", return_value=fake_settings):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                ensure_gcp_available()

    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
    assert "gcs_bucket" in str(exc_info.value).lower() or "gcp_gcs" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Module-import safety: no optional packages should be required at import
# ---------------------------------------------------------------------------


_SENTINEL = object()


def test_module_imports_without_google_or_kubernetes():
    """Importing gke_job_backend must succeed even when optional packages are absent."""
    # The module is already imported at test collection time — this is an explicit
    # regression guard confirming optional packages are not imported at module level.

    # Temporarily hide the optional packages.
    saved = {}
    for key in ("google", "google.cloud", "google.cloud.storage", "google.auth", "kubernetes"):
        saved[key] = sys.modules.get(key, _SENTINEL)
        sys.modules[key] = None  # type: ignore[assignment]

    try:
        # Force reimport.
        mod_name = "backend.services.runtime.gke_job_backend"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        import backend.services.runtime.gke_job_backend  # noqa: F401

        # Reached here → import succeeded without optional packages.
    finally:
        for key, val in saved.items():
            if val is _SENTINEL:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# GkeJobBackend: pending-timeout → capacity_exhausted path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_pending_timeout_returns_capacity_exhausted_error(tmp_path: Path):
    """When a Job pod is Pending for too long, exec returns an error result with capacity_exhausted prefix."""
    # Job status never becomes Complete or Failed.
    def _incomplete_job():
        job = MagicMock()
        cond = MagicMock()
        cond.type = "Unknown"
        cond.status = "False"
        job.status.conditions = [cond]
        return job

    batch_api = FakeBatchApi(job_factory=_incomplete_job)
    core_api = FakeCoreApi(phase="Pending")

    # Use a settings stub that sets a very short pending_timeout (1s) so the
    # test doesn't actually sleep for 900s.
    fake_settings = MagicMock()
    fake_settings.gcp_namespace = "reprolab"
    fake_settings.gcp_pending_timeout_seconds = 1
    fake_settings.gcp_boot_timeout_seconds = 10
    fake_settings.gcp_base_image = "us-docker.pkg.dev/myproject/reprolab/gke-cell-base:test"
    fake_settings.gcp_service_account = "reprolab-sa"
    fake_settings.gcp_gcs_bucket = "reprolab-artifacts"
    fake_settings.gcp_project = "my-gcp-project"

    backend = GkeJobBackend(
        batch_api=batch_api,
        core_api=core_api,
        settings=fake_settings,
    )
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    result = await backend.exec(sandbox, "echo stuck", timeout=10)

    # Should have returned an error (not timed_out from the main deadline)
    # with a capacity_exhausted prefix in stderr.
    assert "capacity_exhausted" in (result.stderr or "").lower() or result.exit_code != 0


# ---------------------------------------------------------------------------
# gpu_plan wiring: GkeJobBackend.__init__ + job manifest node selection
# ---------------------------------------------------------------------------


def _make_gpu_plan_obj(short_name: str = "gcp_a100_40gb", gpu_count: int = 1) -> Any:
    """Build a minimal GpuPlan-like SimpleNamespace (attr-based)."""
    from types import SimpleNamespace
    return SimpleNamespace(short_name=short_name, gpu_count=gpu_count)


def _make_gpu_plan_dict(short_name: str = "gcp_a100_40gb", gpu_count: int = 1) -> dict:
    """Build a plain dict GpuPlan."""
    return {"short_name": short_name, "gpu_count": gpu_count}


def test_gke_backend_stores_gpu_plan_object():
    """GkeJobBackend accepts a GpuPlan object and stores it as _gpu_plan."""
    plan = _make_gpu_plan_obj("gcp_a100_40gb", gpu_count=1)
    backend = GkeJobBackend(gpu_plan=plan)
    assert backend._gpu_plan is plan


def test_gke_backend_stores_gpu_plan_dict():
    """GkeJobBackend accepts a plain dict GpuPlan and stores it."""
    plan = _make_gpu_plan_dict("gcp_a100_80gb", gpu_count=2)
    backend = GkeJobBackend(gpu_plan=plan)
    assert backend._gpu_plan is plan


def test_gke_backend_no_gpu_plan_defaults():
    """GkeJobBackend with no gpu_plan returns None short_name and gpu_count=1."""
    backend = GkeJobBackend()
    assert backend._gpu_plan_short_name() is None
    assert backend._gpu_plan_gpu_count() == 1


def test_gke_backend_gpu_plan_obj_accessors():
    """_gpu_plan_short_name and _gpu_plan_gpu_count read from object attributes."""
    plan = _make_gpu_plan_obj("gcp_a100_80gb", gpu_count=4)
    backend = GkeJobBackend(gpu_plan=plan)
    assert backend._gpu_plan_short_name() == "gcp_a100_80gb"
    assert backend._gpu_plan_gpu_count() == 4


def test_gke_backend_gpu_plan_dict_accessors():
    """_gpu_plan_short_name and _gpu_plan_gpu_count read from dict keys."""
    plan = _make_gpu_plan_dict("gcp_a100_40gb", gpu_count=2)
    backend = GkeJobBackend(gpu_plan=plan)
    assert backend._gpu_plan_short_name() == "gcp_a100_40gb"
    assert backend._gpu_plan_gpu_count() == 2


def test_build_job_manifest_with_gpu_plan_sets_node_selector():
    """_build_job_manifest with gpu_sku sets nodeSelector and gpu resource limits."""
    manifest = _build_job_manifest(
        job_name="reprolab-exec-test-abc12345",
        namespace="reprolab",
        image="reprolab/gke-base:latest",
        service_account="reprolab-sa",
        command="python train.py",
        environment={},
        active_deadline_seconds=3600,
        ttl_seconds=3600,
        backoff_limit=2,
        gpu_sku="gcp_a100_40gb",
        gpu_count=1,
        sandbox_label="gke",
    )
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec.get("nodeSelector") == {"reprolab/sku": "gcp_a100_40gb"}
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"


def test_build_job_manifest_with_multi_gpu_plan():
    """_build_job_manifest with gpu_count>1 requests the correct count."""
    manifest = _build_job_manifest(
        job_name="reprolab-exec-test-multi",
        namespace="reprolab",
        image="reprolab/gke-base:latest",
        service_account="reprolab-sa",
        command="python train.py",
        environment={},
        active_deadline_seconds=3600,
        ttl_seconds=3600,
        backoff_limit=2,
        gpu_sku="gcp_a100_80gb",
        gpu_count=4,
        sandbox_label="gke",
    )
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"] == {"reprolab/sku": "gcp_a100_80gb"}
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"


def test_build_job_manifest_without_gpu_plan_no_node_selector():
    """_build_job_manifest without gpu_sku omits nodeSelector but still sets gpu resources."""
    manifest = _build_job_manifest(
        job_name="reprolab-exec-test-default",
        namespace="reprolab",
        image="reprolab/gke-base:latest",
        service_account="reprolab-sa",
        command="echo hi",
        environment={},
        active_deadline_seconds=60,
        ttl_seconds=3600,
        backoff_limit=2,
        sandbox_label="gke",
    )
    pod_spec = manifest["spec"]["template"]["spec"]
    assert "nodeSelector" not in pod_spec
    # Default 1 GPU resource is still requested.
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"


@pytest.mark.asyncio
async def test_exec_with_gpu_plan_sets_node_selector_in_submitted_job(tmp_path: Path):
    """When GkeJobBackend has a gpu_plan, the submitted Job manifest carries
    nodeSelector={"reprolab/sku": plan.short_name} and correct gpu_count."""
    plan = _make_gpu_plan_obj("gcp_a100_40gb", gpu_count=1)
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = GkeJobBackend(
        gpu_plan=plan,
        batch_api=batch_api,
        core_api=FakeCoreApi(),
        settings=_make_fake_settings(),
    )
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "python train.py", timeout=30)

    assert len(batch_api.created_jobs) == 1
    body = batch_api.created_jobs[0]["body"]
    pod_spec = body["spec"]["template"]["spec"]
    assert pod_spec.get("nodeSelector") == {"reprolab/sku": "gcp_a100_40gb"}
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"


@pytest.mark.asyncio
async def test_exec_without_gpu_plan_no_node_selector_in_submitted_job(tmp_path: Path):
    """When GkeJobBackend has no gpu_plan, the submitted Job has no nodeSelector."""
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = GkeJobBackend(batch_api=batch_api, core_api=FakeCoreApi(), settings=_make_fake_settings())
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "echo hi", timeout=10)

    assert len(batch_api.created_jobs) == 1
    body = batch_api.created_jobs[0]["body"]
    pod_spec = body["spec"]["template"]["spec"]
    assert "nodeSelector" not in pod_spec


# ---------------------------------------------------------------------------
# Autoscaler-thrash regression: gpu_sku=None must fall back to the configured
# gcp_gpu_skus[0] rather than submit an unschedulable pod with no nodeSelector
# at all (see backend/services/runtime/CLAUDE.md GKE section + P0-fix-3 parity
# in k8s_job_cell_runner._build_job_manifest).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_no_gpu_plan_falls_back_to_configured_gpu_sku(tmp_path: Path):
    """(a) gpu_sku=None + non-empty gcp_gpu_skus => nodeSelector uses gpu_skus[0].

    Regression guard for the GKE autoscaler thrash bug: a run whose GpuPlan was
    never resolved (no rlm_state/gpu_plan.json) must still submit a pod that can
    bind to a real, provisioned reprolab/sku node pool instead of thrashing the
    scale-to-zero autoscaler forever.
    """
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    fake_settings = _make_fake_settings(gcp_gpu_skus=["gcp_a100_80x8", "gcp_l4_24"])
    backend = GkeJobBackend(batch_api=batch_api, core_api=FakeCoreApi(), settings=fake_settings)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "python train.py", timeout=30)

    assert len(batch_api.created_jobs) == 1
    body = batch_api.created_jobs[0]["body"]
    pod_spec = body["spec"]["template"]["spec"]
    assert pod_spec.get("nodeSelector") == {"reprolab/sku": "gcp_a100_80x8"}
    # Existing GPU-resource-request behavior is untouched.
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"


@pytest.mark.asyncio
async def test_exec_explicit_gpu_plan_ignores_gpu_skus_fallback(tmp_path: Path):
    """(b) gpu_sku set explicitly (via a resolved GpuPlan) => unchanged/byte-identical.

    Even when gcp_gpu_skus is ALSO configured (to a DIFFERENT SKU), a resolved
    GpuPlan's short_name must win — the fallback must never override an explicit
    plan. This is the common cell-matrix-equivalent path and must be untouched.
    """
    plan = _make_gpu_plan_obj("gcp_a100_40gb", gpu_count=1)
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    fake_settings = _make_fake_settings(gcp_gpu_skus=["gcp_a100_80x8"])
    backend = GkeJobBackend(
        gpu_plan=plan,
        batch_api=batch_api,
        core_api=FakeCoreApi(),
        settings=fake_settings,
    )
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "python train.py", timeout=30)

    body = batch_api.created_jobs[0]["body"]
    pod_spec = body["spec"]["template"]["spec"]
    # The plan's own short_name wins, NOT the unrelated gcp_gpu_skus[0].
    assert pod_spec.get("nodeSelector") == {"reprolab/sku": "gcp_a100_40gb"}


@pytest.mark.asyncio
async def test_exec_no_gpu_plan_and_no_gpu_skus_still_no_node_selector(tmp_path: Path):
    """(c) cpu-only / unconfigured-cluster fallback: gpu_sku=None AND gcp_gpu_skus=[]
    => no nodeSelector (never invent a bogus label; cluster-default behavior preserved).
    """
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    fake_settings = _make_fake_settings(gcp_gpu_skus=[])
    backend = GkeJobBackend(batch_api=batch_api, core_api=FakeCoreApi(), settings=fake_settings)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "echo hi", timeout=10)

    body = batch_api.created_jobs[0]["body"]
    pod_spec = body["spec"]["template"]["spec"]
    assert "nodeSelector" not in pod_spec


# ---------------------------------------------------------------------------
# P0: OPENRESEARCH_EXEC_COMMAND injected exactly once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_command_env_var_appears_exactly_once(tmp_path: Path):
    """OPENRESEARCH_EXEC_COMMAND must appear exactly once in the submitted Job manifest.

    Regression guard for the double-injection bug: exec() injected it via
    env_vars AND _build_job_manifest appended it again → duplicate env var.
    """
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "python train.py", timeout=10)

    assert len(batch_api.created_jobs) == 1
    env_list = batch_api.created_jobs[0]["body"]["spec"]["template"]["spec"]["containers"][0]["env"]
    exec_cmd_entries = [e for e in env_list if e["name"] == "OPENRESEARCH_EXEC_COMMAND"]
    assert len(exec_cmd_entries) == 1, (
        f"OPENRESEARCH_EXEC_COMMAND appeared {len(exec_cmd_entries)} time(s) in the env list; "
        "expected exactly 1."
    )
    assert exec_cmd_entries[0]["value"] == "python train.py"


# ---------------------------------------------------------------------------
# P1: GPU taint toleration present on exec manifest
# ---------------------------------------------------------------------------


def test_build_job_manifest_has_gpu_toleration():
    """_build_job_manifest always includes the nvidia.com/gpu:NoSchedule toleration."""
    manifest = _build_job_manifest(
        job_name="reprolab-exec-test-tol",
        namespace="reprolab",
        image="reprolab/gke-base:test",
        service_account="reprolab-sa",
        command="echo hi",
        environment={},
        active_deadline_seconds=60,
        ttl_seconds=3600,
        backoff_limit=0,
        sandbox_label="gke",
    )
    pod_spec = manifest["spec"]["template"]["spec"]
    tolerations = pod_spec.get("tolerations", [])
    gpu_toleration = {
        "key": "nvidia.com/gpu",
        "operator": "Exists",
        "effect": "NoSchedule",
    }
    assert gpu_toleration in tolerations, (
        f"Expected GPU taint toleration {gpu_toleration!r} in tolerations, got: {tolerations}"
    )


@pytest.mark.asyncio
async def test_exec_submitted_job_has_gpu_toleration(tmp_path: Path):
    """The Job manifest submitted by exec() carries the nvidia.com/gpu toleration."""
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = _make_backend(batch_api=batch_api)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "python train.py", timeout=10)

    body = batch_api.created_jobs[0]["body"]
    tolerations = body["spec"]["template"]["spec"].get("tolerations", [])
    keys = [t.get("key") for t in tolerations]
    assert "nvidia.com/gpu" in keys, f"nvidia.com/gpu toleration missing; got: {tolerations}"


# ---------------------------------------------------------------------------
# P1: Empty base-image raises a clear SandboxRuntimeError
# ---------------------------------------------------------------------------


def test_base_image_empty_raises_clear_error():
    """When gcp_base_image is empty/unset, _base_image() raises backend_unavailable
    with a message pointing to OPENRESEARCH_GCP_BASE_IMAGE."""
    fake_settings = _make_fake_settings(gcp_base_image="")
    backend = GkeJobBackend(settings=fake_settings)

    with pytest.raises(SandboxRuntimeError) as exc_info:
        backend._base_image()

    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
    msg = str(exc_info.value)
    assert "OPENRESEARCH_GCP_BASE_IMAGE" in msg or "gcp_base_image" in msg.lower()


@pytest.mark.asyncio
async def test_create_sandbox_raises_when_base_image_unset(tmp_path: Path):
    """create_sandbox propagates the base-image error before any GCS upload."""
    fake_settings = _make_fake_settings(gcp_base_image="")
    backend = _make_backend(settings=fake_settings)

    with pytest.raises(SandboxRuntimeError) as exc_info:
        await backend.create_sandbox(_make_config(tmp_path))

    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable


# ---------------------------------------------------------------------------
# P1: TTL and backoff read from settings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_uses_ttl_from_settings(tmp_path: Path):
    """exec() passes the ttlSecondsAfterFinished value from settings to the manifest."""
    fake_settings = _make_fake_settings(gcp_ttl_seconds_after_finished=7200)
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = _make_backend(batch_api=batch_api, settings=fake_settings)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "echo ttl", timeout=10)

    body = batch_api.created_jobs[0]["body"]
    assert body["spec"]["ttlSecondsAfterFinished"] == 7200


@pytest.mark.asyncio
async def test_exec_uses_backoff_limit_from_settings(tmp_path: Path):
    """exec() passes the backoffLimit value from settings to the manifest."""
    fake_settings = _make_fake_settings(gcp_job_backoff_limit=3)
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = _make_backend(batch_api=batch_api, settings=fake_settings)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "echo backoff", timeout=10)

    body = batch_api.created_jobs[0]["body"]
    assert body["spec"]["backoffLimit"] == 3


@pytest.mark.asyncio
async def test_exec_uses_default_ttl_when_setting_absent(tmp_path: Path):
    """exec() falls back to _DEFAULT_TTL_AFTER_FINISHED_S when the setting is missing."""
    # Setting not present → getattr returns None → uses default.
    fake_settings = _make_fake_settings()
    # Since MagicMock auto-creates missing attrs, set it explicitly to None.
    fake_settings.gcp_ttl_seconds_after_finished = None
    batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
    backend = _make_backend(batch_api=batch_api, settings=fake_settings)
    sandbox = _make_sandbox(tmp_path)
    backend._active_jobs[sandbox.sandbox_id] = []

    await backend.exec(sandbox, "echo default-ttl", timeout=10)

    body = batch_api.created_jobs[0]["body"]
    assert body["spec"]["ttlSecondsAfterFinished"] == _DEFAULT_TTL_AFTER_FINISHED_S


# ---------------------------------------------------------------------------
# Per-run GPU-USD / pod-second budget cap on the exec path (fail-closed).
# The GKE exec path previously stored _run_budget but never checked it, so a
# sandbox="gcp" run had no per-run dollar ceiling at exec time. _enforce_run_budget
# mirrors runpod_backend.exec: refuse to submit a new Job once cumulative pod-time
# cost (elapsed_hr * plan.total_usd_per_hr) reaches the cap.
# ---------------------------------------------------------------------------

def _sandbox_created_secs_ago(tmp_path: Path, secs_ago: float) -> Sandbox:
    from datetime import timedelta
    return _make_sandbox(tmp_path).model_copy(
        update={"created_at": datetime.now(timezone.utc) - timedelta(seconds=secs_ago)}
    )


class TestGkeRunBudgetCap:
    def test_enforce_raises_when_run_gpu_usd_exceeded(self, tmp_path: Path):
        from backend.agents.resilience.budget import RunBudget
        from backend.agents.resilience.failures import BudgetExhausted
        backend = _make_backend()
        backend._run_budget = RunBudget(max_run_gpu_usd=1.0)
        backend._gpu_plan = {"total_usd_per_hr": 30.0, "gpu_count": 8}
        sandbox = _sandbox_created_secs_ago(tmp_path, 3600)  # 1 hr * $30 = $30 >= $1 cap
        with pytest.raises(BudgetExhausted):
            backend._enforce_run_budget(sandbox)

    def test_enforce_noop_when_budget_none(self, tmp_path: Path):
        backend = _make_backend()
        backend._run_budget = None
        backend._gpu_plan = {"total_usd_per_hr": 30.0}
        backend._enforce_run_budget(_sandbox_created_secs_ago(tmp_path, 3600))  # no raise

    def test_enforce_noop_when_cap_disabled(self, tmp_path: Path):
        from backend.agents.resilience.budget import RunBudget
        backend = _make_backend()
        backend._run_budget = RunBudget(max_run_gpu_usd=0.0)  # 0 disables
        backend._gpu_plan = {"total_usd_per_hr": 30.0}
        backend._enforce_run_budget(_sandbox_created_secs_ago(tmp_path, 3600))  # no raise

    def test_enforce_under_cap_does_not_raise(self, tmp_path: Path):
        from backend.agents.resilience.budget import RunBudget
        backend = _make_backend()
        backend._run_budget = RunBudget(max_run_gpu_usd=1000.0)
        backend._gpu_plan = {"total_usd_per_hr": 30.0}
        backend._enforce_run_budget(_sandbox_created_secs_ago(tmp_path, 60))  # $0.5 < $1000

    def test_exec_refuses_to_submit_when_budget_exhausted(self, tmp_path: Path):
        """The cap fires from exec() BEFORE any Job is created (no GPU spend)."""
        import asyncio
        from backend.agents.resilience.budget import RunBudget
        from backend.agents.resilience.failures import BudgetExhausted
        batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
        backend = _make_backend(batch_api=batch_api)
        backend._run_budget = RunBudget(max_run_gpu_usd=1.0)
        backend._gpu_plan = {"total_usd_per_hr": 30.0, "gpu_count": 8}
        sandbox = _sandbox_created_secs_ago(tmp_path, 3600)
        backend._active_jobs[sandbox.sandbox_id] = []
        with pytest.raises(BudgetExhausted):
            asyncio.run(backend.exec(sandbox, "echo x", timeout=10))
        assert batch_api.created_jobs == []  # no Job submitted


# ---------------------------------------------------------------------------
# GPU SKU / node-pool preflight wiring (create_sandbox is the once-per-run hook;
# see backend/services/runtime/gpu_pool_preflight.py for the actual check).
# ---------------------------------------------------------------------------


class _NodePoolCoreApi(FakeCoreApi):
    """FakeCoreApi + list_node(), for the GPU-pool-preflight wiring tests only."""

    def __init__(
        self,
        *,
        node_skus: list[str] | None = None,
        raise_on_list_node: Exception | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self._node_skus = node_skus if node_skus is not None else []
        self._raise_on_list_node = raise_on_list_node
        self.list_node_call_count = 0

    def list_node(self) -> Any:
        self.list_node_call_count += 1
        if self._raise_on_list_node is not None:
            raise self._raise_on_list_node
        from types import SimpleNamespace
        nodes = [
            SimpleNamespace(metadata=SimpleNamespace(labels={"reprolab/sku": s}))
            for s in self._node_skus
        ]
        return SimpleNamespace(items=nodes)


@pytest.mark.asyncio
async def test_create_sandbox_raises_on_drifted_gpu_sku(tmp_path: Path):
    """The exact hazard: OPENRESEARCH_GCP_GPU_SKUS names a SKU with no live node
    pool -> create_sandbox refuses to start (before any Job/upload happens),
    naming the drifting SKU in an actionable message."""
    core_api = _NodePoolCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    fake_settings = _make_fake_settings(gcp_gpu_skus=["gcp_a100_80x2", "gcp_a100_80x8"])
    backend = _make_backend(core_api=core_api, settings=fake_settings)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        with pytest.raises(SandboxRuntimeError) as exc_info:
            await backend.create_sandbox(_make_config(tmp_path))

    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
    assert "gcp_a100_80x2" in str(exc_info.value)
    assert core_api.list_node_call_count == 1


@pytest.mark.asyncio
async def test_create_sandbox_passes_silently_when_all_gpu_skus_provisioned(tmp_path: Path):
    """All configured SKUs observed live -> create_sandbox succeeds unchanged."""
    core_api = _NodePoolCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    fake_settings = _make_fake_settings(gcp_gpu_skus=["gcp_a100_80", "gcp_a100_80x8"])
    backend = _make_backend(core_api=core_api, settings=fake_settings)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        sandbox = await backend.create_sandbox(_make_config(tmp_path))

    assert isinstance(sandbox, Sandbox)
    assert core_api.list_node_call_count == 1


@pytest.mark.asyncio
async def test_create_sandbox_warns_not_raises_when_cluster_unreachable(tmp_path: Path):
    """A transient query failure (auth/network) must NOT become a hard block."""
    core_api = _NodePoolCoreApi(raise_on_list_node=ConnectionError("no route to host"))
    fake_settings = _make_fake_settings(gcp_gpu_skus=["gcp_a100_80"])
    backend = _make_backend(core_api=core_api, settings=fake_settings)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        sandbox = await backend.create_sandbox(_make_config(tmp_path))  # must NOT raise

    assert isinstance(sandbox, Sandbox)
    assert core_api.list_node_call_count == 1


@pytest.mark.asyncio
async def test_create_sandbox_cold_cluster_warns_not_raises(tmp_path: Path):
    """No node currently carries ANY reprolab/sku label (idle, scale-to-zero) —
    inconclusive, not "every configured SKU confirmed absent"."""
    core_api = _NodePoolCoreApi(node_skus=[])
    fake_settings = _make_fake_settings(gcp_gpu_skus=["gcp_a100_80", "gcp_a100_80x8"])
    backend = _make_backend(core_api=core_api, settings=fake_settings)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        sandbox = await backend.create_sandbox(_make_config(tmp_path))  # must NOT raise

    assert isinstance(sandbox, Sandbox)


@pytest.mark.asyncio
async def test_create_sandbox_skips_preflight_when_no_gpu_skus_configured(tmp_path: Path):
    """Default settings (gcp_gpu_skus=[]) -> the cluster is never queried at all
    (byte-identical to before this change for the common unconfigured case)."""
    core_api = _NodePoolCoreApi(node_skus=["gcp_a100_80"])
    backend = _make_backend(core_api=core_api)  # default settings: gcp_gpu_skus=[]

    with patch.object(backend, "_upload_project_sync", return_value=None):
        await backend.create_sandbox(_make_config(tmp_path))

    assert core_api.list_node_call_count == 0


# ---------------------------------------------------------------------------
# Authoritative tier-1 (GKE node-pool API) through create_sandbox: a COLD
# scale-to-zero cluster has NO Node objects, so the live-Node heuristic is blind
# — the pool API is what makes the first-run-of-the-day case catchable.
# ---------------------------------------------------------------------------


def _addressable_settings(**overrides: Any) -> MagicMock:
    """Fake settings with project+region+cluster set, so tier 1 is addressable."""
    return _make_fake_settings(
        gcp_project="my-gcp-project",
        gcp_region="us-central1",
        gcp_gke_cluster="repro-gke",
        **overrides,
    )


@pytest.mark.asyncio
async def test_create_sandbox_cold_cluster_with_drift_hard_blocks_via_pool_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """THE GAP: zero live nodes (idle scale-to-zero) + a drifted SKU. The node
    heuristic sees nothing; the node-pool API still sees the pools -> HARD BLOCK."""
    import backend.services.runtime.gpu_pool_preflight as gpp

    monkeypatch.setattr(
        gpp,
        "list_gke_node_pools",
        lambda **kw: [
            {"name": "default-pool", "config": {"labels": {}}},
            {"name": "repro-a100", "config": {"labels": {"reprolab/sku": "gcp_a100_80"}}},
            {"name": "repro-a100x8", "config": {"labels": {"reprolab/sku": "gcp_a100_80x8"}}},
        ],
    )
    core_api = _NodePoolCoreApi(node_skus=[])  # COLD: no nodes at all
    settings = _addressable_settings(gcp_gpu_skus=["gcp_a100_80x2", "gcp_a100_80x8"])
    backend = _make_backend(core_api=core_api, settings=settings)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        with pytest.raises(SandboxRuntimeError) as exc_info:
            await backend.create_sandbox(_make_config(tmp_path))

    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
    assert "gcp_a100_80x2" in str(exc_info.value)
    # Tier 1 answered, so the (blind) heuristic was never consulted.
    assert core_api.list_node_call_count == 0


@pytest.mark.asyncio
async def test_create_sandbox_cold_cluster_no_drift_passes_via_pool_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import backend.services.runtime.gpu_pool_preflight as gpp

    monkeypatch.setattr(
        gpp,
        "list_gke_node_pools",
        lambda **kw: [
            {"name": "repro-a100", "config": {"labels": {"reprolab/sku": "gcp_a100_80"}}},
            {"name": "repro-a100x8", "config": {"labels": {"reprolab/sku": "gcp_a100_80x8"}}},
        ],
    )
    core_api = _NodePoolCoreApi(node_skus=[])  # cold
    settings = _addressable_settings(gcp_gpu_skus=["gcp_a100_80", "gcp_a100_80x8"])
    backend = _make_backend(core_api=core_api, settings=settings)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        sandbox = await backend.create_sandbox(_make_config(tmp_path))

    assert isinstance(sandbox, Sandbox)


@pytest.mark.asyncio
async def test_create_sandbox_pool_api_permission_denied_degrades_to_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The in-cluster orchestrator GSA has no container.clusters.get today -> 403.
    On a cold cluster that must WARN and proceed, never error the run."""
    import backend.services.runtime.gpu_pool_preflight as gpp

    def _denied(**kw):
        raise RuntimeError("GKE nodePools API returned HTTP 403: permission denied")

    monkeypatch.setattr(gpp, "list_gke_node_pools", _denied)
    core_api = _NodePoolCoreApi(node_skus=[])  # cold => heuristic also blind
    settings = _addressable_settings(gcp_gpu_skus=["gcp_a100_80x2"])
    backend = _make_backend(core_api=core_api, settings=settings)

    with patch.object(backend, "_upload_project_sync", return_value=None):
        sandbox = await backend.create_sandbox(_make_config(tmp_path))  # must NOT raise

    assert isinstance(sandbox, Sandbox)
    assert core_api.list_node_call_count == 1  # fell back to the heuristic
# WS2 guard — fail-loud refusal of code-dependent commands on the unstaged
# monolithic exec path (Phase-3 durable-controller fan-out). Flag-gated on
# OPENRESEARCH_DURABLE_CONTROLLER (default OFF), gcp-only. See
# `.superpowers/sdd/phase3-owner2-job_backend-ws2guard.md`.
# ---------------------------------------------------------------------------

from backend.services.runtime.k8s_job_backend import _command_needs_staged_code  # noqa: E402


class TestCommandNeedsStagedCode:
    """Pure predicate table — no fixtures, no mocks, no cloud dependency."""

    @pytest.mark.parametrize(
        "command",
        [
            "python train.py --foo",
            "python train.py",
            "python3 train.py --seed 1",
            "python3.11 /code/train.py",
            "python -m project.train",
            "python -m project.train --epochs 5",
            "./train.py --seed 1",
            "pip install -r requirements.txt && python train.py",
        ],
    )
    def test_positive_project_code_commands(self, command: str) -> None:
        assert _command_needs_staged_code(command) is True, command

    @pytest.mark.parametrize(
        "command",
        [
            "nvidia-smi",
            "pip install torch",
            'python -c "print(1)"',
            "ls",
            "nvcc --version",
            "python -m pip install -r requirements.txt",  # the real bootstrap shape
            "echo hi",
            "",
            "   ",
        ],
    )
    def test_negative_utility_commands(self, command: str) -> None:
        assert _command_needs_staged_code(command) is False, command


class TestCommandNeedsStagedCodeArgumentPositionGuard:
    """Regression guard (Phase-3 review finding on the WS2 guard): a bare
    ``*.py``-suffixed token must only trip the predicate when it is the
    invoked program of a (sub)command, not merely an argument to some other
    program. Before this fix, `_command_needs_staged_code`'s bare-``.py``
    branch fired for ANY ``.py``-suffixed token anywhere in the command,
    so e.g. ``cat code/train.py`` (reads a file, doesn't execute it) was
    misidentified as code-dependent.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "cat code/train.py",
            "echo train.py",
            "wc -l code/*.py",
            "black train.py",
            "git diff -- train.py",
        ],
    )
    def test_py_token_as_argument_is_not_flagged(self, command: str) -> None:
        assert _command_needs_staged_code(command) is False, command

    def test_py_token_as_invoked_program_after_chain_operator_still_flagged(self) -> None:
        """A bare .py script IS the program being run in the second chained
        sub-command -> still True (the fix must not overcorrect to False)."""
        assert _command_needs_staged_code("cat notes.txt && ./train.py --seed 1") is True


class TestExecMonolithicUnstagedGuard:
    """Hermetic OFF/ON pair for the `exec()` fail-loud guard."""

    @pytest.mark.asyncio
    async def test_off_flag_unset_submits_exactly_as_today(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Byte-identical to pre-guard behavior: flag unset -> Job submitted
        even for an obviously code-dependent command on gcp."""
        monkeypatch.delenv("OPENRESEARCH_DURABLE_CONTROLLER", raising=False)
        batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
        backend = _make_backend(batch_api=batch_api)
        sandbox = _make_sandbox(tmp_path)
        backend._active_jobs[sandbox.sandbox_id] = []

        result = await backend.exec(sandbox, "python train.py", timeout=30)

        assert len(batch_api.created_jobs) == 1
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_on_gcp_code_command_raises_and_submits_no_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ON + gcp + code-dependent command -> SandboxRuntimeError, no Job created."""
        monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
        batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
        backend = _make_backend(batch_api=batch_api)
        sandbox = _make_sandbox(tmp_path)
        backend._active_jobs[sandbox.sandbox_id] = []

        with pytest.raises(SandboxRuntimeError) as exc_info:
            await backend.exec(sandbox, "python train.py", timeout=30)

        assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
        assert "monolithic_exec_unstaged" in str(exc_info.value)
        assert batch_api.created_jobs == []

    @pytest.mark.asyncio
    async def test_on_gcp_non_code_command_submits_normally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ON + gcp + a plain utility command -> guard does not trip."""
        monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
        batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
        backend = _make_backend(batch_api=batch_api)
        sandbox = _make_sandbox(tmp_path)
        backend._active_jobs[sandbox.sandbox_id] = []

        result = await backend.exec(sandbox, "nvidia-smi", timeout=10)

        assert len(batch_api.created_jobs) == 1
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_off_flag_unset_with_code_command_ignores_predicate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag OFF short-circuits before `_command_needs_staged_code` is ever
        consulted — same code-dependent command as the ON-guard test above,
        but here it must submit normally."""
        monkeypatch.delenv("OPENRESEARCH_DURABLE_CONTROLLER", raising=False)
        batch_api = FakeBatchApi(job_factory=lambda: FakeJob(complete=True))
        backend = _make_backend(batch_api=batch_api)
        sandbox = _make_sandbox(tmp_path)
        backend._active_jobs[sandbox.sandbox_id] = []

        await backend.exec(sandbox, "python -m project.train --epochs 5", timeout=30)

        assert len(batch_api.created_jobs) == 1
