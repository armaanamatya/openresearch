import pytest
from pydantic import ValidationError

from backend.services.events.live_runs import StartRunRequest, FileLiveRunService


def _svc(tmp_path):
    # Build a *real* instance (not __new__). The brief's helper used
    # FileLiveRunService.__new__(FileLiveRunService) to skip __init__, but
    # _subprocess_env reads self.repo_root (live_runs.py:507, the .env-precedence
    # block) before it reaches the budget-field logic — a __new__ instance has no
    # repo_root and raised AttributeError. __init__ is cheap + hermetic (only
    # Path.resolve() on repo_root/runs_root); tmp_path has no .env → loads nothing.
    return FileLiveRunService(repo_root=tmp_path)


def test_subprocess_env_threads_budget_fields(tmp_path):
    req = StartRunRequest(dynamic_gpu=True, force_single_gpu=True,
                          max_gpu_usd_per_hour=7.5, vram_gb=24)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), req)
    assert env["OPENRESEARCH_DYNAMIC_GPU"] == "true"
    assert env["OPENRESEARCH_FORCE_SINGLE_GPU"] == "true"
    assert env["OPENRESEARCH_MAX_GPU_USD_PER_HOUR"] == "7.5"
    assert env["OPENRESEARCH_VRAM_OVERRIDE_GB"] == "24"


def test_subprocess_env_omits_unset_budget_fields(tmp_path, monkeypatch):
    # _subprocess_env seeds env from os.environ (:498), so a shell export of any
    # budget key would leak in and cause a false failure — clear them first.
    for k in ("OPENRESEARCH_DYNAMIC_GPU", "OPENRESEARCH_FORCE_SINGLE_GPU",
              "OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "OPENRESEARCH_VRAM_OVERRIDE_GB"):
        monkeypatch.delenv(k, raising=False)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), StartRunRequest())
    for k in ("OPENRESEARCH_DYNAMIC_GPU", "OPENRESEARCH_FORCE_SINGLE_GPU",
              "OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "OPENRESEARCH_VRAM_OVERRIDE_GB"):
        assert k not in env


def test_arxiv_request_declares_advanced_fields():
    from backend.app import StartArxivRunRequest
    r = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155",
                             dynamic_gpu=True, vram_gb=48)
    assert r.dynamic_gpu is True and r.vram_gb == 48


# --------------------------------------------------------------------------- #
# gpu_count — user-selectable GPU count (1-8, optional). None => existing
# auto-resolution (gpu_resolver), byte-identical to today.
# --------------------------------------------------------------------------- #


def test_subprocess_env_threads_gpu_count(tmp_path):
    req = StartRunRequest(gpu_count=4)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), req)
    assert env["OPENRESEARCH_GPU_COUNT"] == "4"


def test_subprocess_env_omits_unset_gpu_count(tmp_path, monkeypatch):
    # Same rationale as test_subprocess_env_omits_unset_budget_fields: clear
    # any shell export first so it can't leak into the seeded env and produce
    # a false failure.
    monkeypatch.delenv("OPENRESEARCH_GPU_COUNT", raising=False)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), StartRunRequest())
    assert "OPENRESEARCH_GPU_COUNT" not in env


@pytest.mark.parametrize("bad_value", [0, 9])
def test_start_run_request_rejects_out_of_range_gpu_count(bad_value):
    with pytest.raises(ValidationError):
        StartRunRequest(gpu_count=bad_value)


@pytest.mark.parametrize("bad_value", [0, 9])
def test_start_arxiv_request_rejects_out_of_range_gpu_count(bad_value):
    from backend.app import StartArxivRunRequest
    with pytest.raises(ValidationError):
        StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155", gpu_count=bad_value)


def test_arxiv_request_declares_gpu_count():
    from backend.app import StartArxivRunRequest
    r = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155", gpu_count=2)
    assert r.gpu_count == 2


# --------------------------------------------------------------------------- #
# HTTP-level regression guard for /runs/upload's gpuCount form field. The
# multipart path has no pydantic coercion at the FastAPI-parameter layer (the
# form is parsed manually), so an out-of-range value must be *rejected to
# None* by `_optional_form_gpu_count` before it reaches StartRunRequest's
# `ge=1, le=8` field — otherwise it would 500 via the generic exception
# handler instead of falling back to auto-resolution. Mirrors the
# `_FakeUploadService` + TestClient pattern in
# tests/routes/test_autonomous_request_threading.py.
# --------------------------------------------------------------------------- #


def _upload_client():
    from starlette.testclient import TestClient
    from backend.app import create_app
    from backend.services.events.live_runs import LiveRunState

    class _FakeUploadService:
        def __init__(self) -> None:
            self.started: StartRunRequest | None = None
            self.state = LiveRunState(
                projectId="prj_gpu",
                outputDir="runs/prj_gpu",
                runMode="rlm",
                llmProvider="anthropic",
                status="queued",
                payload=None,
                log="",
            )

        async def start_uploaded_run(self, request: StartRunRequest, *, file_name: str, content: bytes):
            self.started = request
            self.state.sourceKind = "uploaded_pdf"
            self.state.sourceLabel = file_name
            return self.state

    service = _FakeUploadService()
    return TestClient(create_app(run_service=service)), service


def test_upload_route_forwards_valid_gpu_count():
    client, service = _upload_client()
    response = client.post(
        "/runs/upload",
        data={"mode": "rlm", "provider": "anthropic", "gpuCount": "4"},
        files={"paper": ("paper.pdf", b"%PDF-demo", "application/pdf")},
    )
    assert response.status_code == 202, response.text
    assert service.started is not None
    assert service.started.gpu_count == 4


def test_upload_route_omitting_gpu_count_defaults_none_not_422():
    client, service = _upload_client()
    response = client.post(
        "/runs/upload",
        data={"mode": "rlm", "provider": "anthropic"},
        files={"paper": ("paper.pdf", b"%PDF-demo", "application/pdf")},
    )
    assert response.status_code == 202, response.text
    assert service.started is not None
    assert service.started.gpu_count is None


@pytest.mark.parametrize("bad_value", ["0", "9", "not-a-number"])
def test_upload_route_rejects_out_of_range_gpu_count_to_none(bad_value):
    client, service = _upload_client()
    response = client.post(
        "/runs/upload",
        data={"mode": "rlm", "provider": "anthropic", "gpuCount": bad_value},
        files={"paper": ("paper.pdf", b"%PDF-demo", "application/pdf")},
    )
    assert response.status_code == 202, response.text
    assert service.started is not None
    assert service.started.gpu_count is None
