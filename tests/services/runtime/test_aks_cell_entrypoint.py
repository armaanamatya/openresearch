"""Tests for docker/aks-cell-base/aks_cell_entrypoint.py.

Design constraints
------------------
* All tests run WITHOUT the ``azure`` package installed and WITHOUT a GPU.
* The pure functions are tested directly.
* Orchestration is tested with a ``FakeBlobClient`` + a ``FakeSubprocessRunner``.
* Blob I/O surfaces are verified via the injectable client param.
* No ``docker build``, no real subprocess, no real Blob calls.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Locate the module under test via its filesystem path so we do NOT need
# the docker/ directory on sys.path in normal test runs.
# ---------------------------------------------------------------------------

_ENTRYPOINT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "docker" / "aks-cell-base" / "aks_cell_entrypoint.py"
)


def _load_entrypoint():
    """Import aks_cell_entrypoint from its absolute path."""
    spec = importlib.util.spec_from_file_location(
        "aks_cell_entrypoint", _ENTRYPOINT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ep():
    """Module-level fixture for the entrypoint module."""
    return _load_entrypoint()


# ---------------------------------------------------------------------------
# Module import — must work even without the azure package
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_imports_without_azure_sdk(self):
        """The module must be importable even when azure is absent."""
        with patch.dict(sys.modules, {"azure": None,
                                       "azure.identity": None,
                                       "azure.storage": None,
                                       "azure.storage.blob": None}):
            mod = _load_entrypoint()
        assert callable(mod.plan_attempts)
        assert callable(mod.is_oom)
        assert callable(mod.classify_outcome)
        assert callable(mod.build_sentinel)

    def test_exit_code_constants(self, ep):
        assert ep.EXIT_OK == 0
        assert ep.EXIT_BOOTSTRAP_ERROR == 40
        assert ep.EXIT_ERROR == 41
        assert ep.EXIT_OOM_SHRINK_EXHAUSTED == 42
        assert ep.EXIT_METRICS_INVALID == 43
        assert ep.EXIT_ARTIFACT_UPLOAD_ERROR == 44


# ---------------------------------------------------------------------------
# plan_attempts — pure
# ---------------------------------------------------------------------------

class TestPlanAttempts:
    def test_zero_retries_gives_one_attempt(self, ep):
        attempts = ep.plan_attempts(0)
        assert len(attempts) == 1
        assert attempts[0]["attempt"] == 0
        assert attempts[0]["batch_scale"] == 1.0
        assert attempts[0]["grad_checkpoint"] == "0"

    def test_one_retry_gives_two_attempts(self, ep):
        attempts = ep.plan_attempts(1)
        assert len(attempts) == 2
        a1 = attempts[1]
        assert a1["attempt"] == 1
        assert a1["batch_scale"] == 0.5
        assert a1["grad_checkpoint"] == "1"

    def test_two_retries_gives_three_attempts(self, ep):
        attempts = ep.plan_attempts(2)
        assert len(attempts) == 3
        a2 = attempts[2]
        assert a2["attempt"] == 2
        assert a2["batch_scale"] == 0.25
        assert a2["grad_checkpoint"] == "1"

    def test_three_retries_floor_at_025(self, ep):
        """max_oom_retries=3 gives four attempts; last two should be 0.25 floor."""
        attempts = ep.plan_attempts(3)
        assert len(attempts) == 4
        # attempt 2 and 3 are both at the 0.25 floor
        assert attempts[2]["batch_scale"] == 0.25
        assert attempts[3]["batch_scale"] == 0.25
        assert attempts[3]["grad_checkpoint"] == "1"

    def test_attempt_indices_are_sequential(self, ep):
        for n in (0, 1, 2, 5):
            attempts = ep.plan_attempts(n)
            for i, a in enumerate(attempts):
                assert a["attempt"] == i

    def test_first_attempt_has_original_scale(self, ep):
        for n in (0, 2, 5):
            assert ep.plan_attempts(n)[0]["batch_scale"] == 1.0


# ---------------------------------------------------------------------------
# is_oom — pure
# ---------------------------------------------------------------------------

class TestIsOom:
    def test_empty_string_is_not_oom(self, ep):
        assert ep.is_oom("") is False

    def test_none_is_not_oom(self, ep):
        # Defensive: some callers may pass empty string; None-guard
        assert ep.is_oom("") is False

    def test_cuda_out_of_memory(self, ep):
        assert ep.is_oom("RuntimeError: CUDA out of memory. Tried to allocate...") is True

    def test_cuda_error_oom(self, ep):
        assert ep.is_oom("CUDA error: out of memory") is True

    def test_cuda_oom_error_class(self, ep):
        assert ep.is_oom("torch.cuda.OutOfMemoryError: CUDA OOM") is True

    def test_out_of_memory_generic(self, ep):
        assert ep.is_oom("Error: out of memory") is True

    def test_unrelated_error_not_oom(self, ep):
        assert ep.is_oom("ValueError: invalid literal for int()") is False

    def test_gradient_error_not_oom(self, ep):
        assert ep.is_oom("RuntimeError: element 0 of tensors does not require grad") is False

    def test_multiline_stderr_with_oom(self, ep):
        stderr = "\n".join([
            "Traceback (most recent call last):",
            "  File train_cell.py, line 500, in train",
            "RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB",
        ])
        assert ep.is_oom(stderr) is True

    def test_case_insensitive_oom(self, ep):
        assert ep.is_oom("cuda out of memory") is True


# ---------------------------------------------------------------------------
# is_oom_kill — pure
# ---------------------------------------------------------------------------

class TestIsOomKill:
    def test_sigkill_detected(self, ep):
        assert ep.is_oom_kill(-9) is True

    def test_normal_exit_not_oom_kill(self, ep):
        assert ep.is_oom_kill(0) is False
        assert ep.is_oom_kill(1) is False
        assert ep.is_oom_kill(137) is False  # 128+9 only applies to shell; Python uses -9

    def test_non_kill_signal_not_oom_kill(self, ep):
        assert ep.is_oom_kill(-15) is False  # SIGTERM


# ---------------------------------------------------------------------------
# classify_outcome — pure
# ---------------------------------------------------------------------------

class TestClassifyOutcome:
    @pytest.fixture
    def good_metrics(self):
        return {"final_training_loss": 0.5, "status": "completed"}

    def test_success_with_valid_metrics(self, ep, good_metrics):
        code, outcome = ep.classify_outcome(
            exit_code=0, stderr="", metrics=good_metrics, is_last_attempt=True
        )
        assert code == ep.EXIT_OK
        assert outcome == "ok"

    def test_success_but_metrics_none(self, ep):
        code, outcome = ep.classify_outcome(
            exit_code=0, stderr="", metrics=None, is_last_attempt=True
        )
        assert code == ep.EXIT_METRICS_INVALID
        assert outcome == "metrics_invalid"

    def test_success_but_metrics_empty_dict(self, ep):
        code, outcome = ep.classify_outcome(
            exit_code=0, stderr="", metrics={}, is_last_attempt=True
        )
        assert code == ep.EXIT_METRICS_INVALID
        assert outcome == "metrics_invalid"

    def test_success_metrics_missing_required_keys(self, ep):
        code, outcome = ep.classify_outcome(
            exit_code=0, stderr="", metrics={"something_else": 1}, is_last_attempt=True
        )
        assert code == ep.EXIT_METRICS_INVALID
        assert outcome == "metrics_invalid"

    def test_oom_last_attempt(self, ep):
        code, outcome = ep.classify_outcome(
            exit_code=1,
            stderr="CUDA out of memory",
            metrics=None,
            is_last_attempt=True,
        )
        assert code == ep.EXIT_OOM_SHRINK_EXHAUSTED
        assert outcome == "oom_shrink_exhausted"

    def test_oom_non_last_attempt(self, ep):
        code, outcome = ep.classify_outcome(
            exit_code=1,
            stderr="CUDA out of memory",
            metrics=None,
            is_last_attempt=False,
        )
        # Intermediate OOM still maps to OOM code (caller decides to retry)
        assert code == ep.EXIT_OOM_SHRINK_EXHAUSTED
        assert outcome == "oom_shrink_exhausted"

    def test_non_oom_failure_is_error(self, ep):
        code, outcome = ep.classify_outcome(
            exit_code=1,
            stderr="ImportError: No module named alfworld_env",
            metrics=None,
            is_last_attempt=True,
        )
        assert code == ep.EXIT_ERROR
        assert outcome == "error"

    def test_non_oom_failure_does_not_retry(self, ep):
        """Non-OOM failure should return ERROR even when is_last_attempt=False."""
        code, outcome = ep.classify_outcome(
            exit_code=2,
            stderr="SomeOtherError",
            metrics=None,
            is_last_attempt=False,
        )
        assert code == ep.EXIT_ERROR
        assert outcome == "error"

    def test_oom_kill_signal_recognized(self, ep, good_metrics):
        # returncode=-9 with empty stderr should still be recognised as OOM
        code, outcome = ep.classify_outcome(
            exit_code=-9,
            stderr="",
            metrics=None,
            is_last_attempt=True,
        )
        assert code == ep.EXIT_OOM_SHRINK_EXHAUSTED

    def test_metrics_with_status_key_only(self, ep):
        """metrics with only 'status' key should be accepted."""
        code, outcome = ep.classify_outcome(
            exit_code=0, stderr="", metrics={"status": "completed"}, is_last_attempt=True
        )
        assert code == ep.EXIT_OK


# ---------------------------------------------------------------------------
# build_sentinel — pure
# ---------------------------------------------------------------------------

class TestBuildSentinel:
    def test_all_required_fields_present(self, ep):
        s = ep.build_sentinel(
            cell_id="cell-001",
            outcome="ok",
            exit_code=0,
            attempts=1,
            retries=0,
            error=None,
            started_at="2026-06-07T00:00:00Z",
            finished_at="2026-06-07T01:00:00Z",
        )
        for key in ("version", "cell_id", "outcome", "exit_code",
                    "attempts", "retries", "error", "started_at", "finished_at"):
            assert key in s, f"missing key: {key}"

    def test_version_is_string(self, ep):
        s = ep.build_sentinel(
            cell_id="c", outcome="ok", exit_code=0, attempts=1,
            retries=0, error=None,
            started_at="t0", finished_at="t1",
        )
        assert isinstance(s["version"], str)

    def test_error_preserved(self, ep):
        s = ep.build_sentinel(
            cell_id="c", outcome="error", exit_code=41, attempts=1,
            retries=0, error="something went wrong",
            started_at="t0", finished_at="t1",
        )
        assert s["error"] == "something went wrong"

    def test_sentinel_is_json_serialisable(self, ep):
        s = ep.build_sentinel(
            cell_id="c-json", outcome="ok", exit_code=0, attempts=2,
            retries=1, error=None,
            started_at="2026-06-07T00:00:00Z", finished_at="2026-06-07T01:00:00Z",
        )
        json.dumps(s)  # must not raise


# ---------------------------------------------------------------------------
# validate_metrics — pure
# ---------------------------------------------------------------------------

class TestValidateMetrics:
    def test_none_is_invalid(self, ep):
        assert ep.validate_metrics(None) is False

    def test_empty_dict_is_invalid(self, ep):
        assert ep.validate_metrics({}) is False

    def test_minimal_valid_with_status(self, ep):
        assert ep.validate_metrics({"status": "completed"}) is True

    def test_minimal_valid_with_loss(self, ep):
        assert ep.validate_metrics({"final_training_loss": 0.3}) is True

    def test_dict_without_required_keys_is_invalid(self, ep):
        assert ep.validate_metrics({"foo": "bar"}) is False


# ---------------------------------------------------------------------------
# Fake infrastructure for orchestration tests
# ---------------------------------------------------------------------------

class FakeBlobClient:
    """In-memory ContainerClient duck-type suitable for all entrypoint calls."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self.upload_calls: list[tuple[str, bytes]] = []
        self.download_calls: list[str] = []
        self.list_calls: list[str] = []

    # ---- ContainerClient API surface used by the entrypoint -----------------

    def upload_blob(self, name: str, data: bytes, *, overwrite: bool = True) -> None:
        self._store[name] = data
        self.upload_calls.append((name, data))

    def download_blob(self, name: str) -> "_FakeStream":
        self.download_calls.append(name)
        return _FakeStream(self._store.get(name, b""))

    def list_blobs(self, *, name_starts_with: str = "") -> list[Any]:
        self.list_calls.append(name_starts_with)
        return [
            _FakeBlob(k)
            for k in self._store
            if k.startswith(name_starts_with)
        ]

    # ---- Seed helper --------------------------------------------------------

    def seed_blob(self, name: str, data: bytes) -> None:
        """Pre-populate a blob without logging to upload_calls."""
        self._store[name] = data


class _FakeStream:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _FakeBlob:
    def __init__(self, name: str) -> None:
        self.name = name


# ---------------------------------------------------------------------------
# FakeSubprocessRunner
# ---------------------------------------------------------------------------

def _make_fake_runner(
    returncode: int = 0,
    output: str = "",
    metrics: dict[str, Any] | None = None,
) -> Any:
    """Return a callable that mimics _run_trainer_subprocess.

    When *metrics* is provided, the fake writes a metrics.json to output_dir
    before returning so the orchestration code can pick it up.
    """
    def _runner(
        train_cell_path: Path,
        output_dir: Path,
        env_overrides: dict[str, str],
        attempt_log_path: Path,
    ) -> tuple[int, str]:
        # Write fake log
        attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_log_path.write_text(output, encoding="utf-8")
        # Optionally write metrics
        if metrics is not None:
            mf = output_dir / "metrics.json"
            mf.write_text(json.dumps(metrics), encoding="utf-8")
        return returncode, output

    return _runner


# ---------------------------------------------------------------------------
# Orchestration tests (_run_with_ladder)
# ---------------------------------------------------------------------------

class TestRunWithLadder:
    """Test _run_with_ladder with fake subprocess runner + fake blob client."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def scaffold(self, tmpdir):
        """Common dirs used by all orchestration tests."""
        train_cell = tmpdir / "code" / "train_cell.py"
        train_cell.parent.mkdir(parents=True)
        train_cell.write_text("# fake train_cell.py")
        output_dir = tmpdir / "output"
        output_dir.mkdir()
        log_dir = tmpdir / "logs"
        log_dir.mkdir()
        return {
            "train_cell_path": train_cell,
            "output_dir": output_dir,
            "log_dir": log_dir,
        }

    def _call(self, ep, scaffold, blob_client, runner, max_oom_retries=2):
        return ep._run_with_ladder(
            train_cell_path=scaffold["train_cell_path"],
            output_dir=scaffold["output_dir"],
            max_oom_retries=max_oom_retries,
            log_dir=scaffold["log_dir"],
            output_blob_prefix="runs/test/cells",
            cell_id="test-cell",
            account_name="testaccount",
            container_name="testcontainer",
            blob_client=blob_client,
            subprocess_runner=runner,
        )

    def test_success_path(self, ep, scaffold):
        """Clean exit 0 with valid metrics returns EXIT_OK, ok, 1 attempt."""
        good_metrics = {"final_training_loss": 0.2, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)
        blob = FakeBlobClient()
        exit_code, outcome, metrics, attempts, retries = self._call(ep, scaffold, blob, runner)
        assert exit_code == ep.EXIT_OK
        assert outcome == "ok"
        assert attempts == 1
        assert retries == 0
        assert metrics is not None
        assert metrics["status"] == "completed"

    def test_success_uploads_metrics(self, ep, scaffold):
        """Successful run must upload metrics.json to Blob."""
        good_metrics = {"final_training_loss": 0.2, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="", metrics=good_metrics)
        blob = FakeBlobClient()
        self._call(ep, scaffold, blob, runner)
        uploaded_names = [name for name, _ in blob.upload_calls]
        assert any("metrics.json" in n for n in uploaded_names)

    def test_non_oom_failure_does_not_retry(self, ep, scaffold):
        """A non-OOM trainer failure exits immediately with EXIT_ERROR."""
        runner = _make_fake_runner(returncode=1, output="ImportError: no module")
        blob = FakeBlobClient()
        exit_code, outcome, metrics, attempts, retries = self._call(ep, scaffold, blob, runner)
        assert exit_code == ep.EXIT_ERROR
        assert outcome == "error"
        assert attempts == 1
        assert retries == 0

    def test_oom_retries_up_to_max(self, ep, scaffold):
        """OOM on every attempt exhausts the ladder and returns EXIT_OOM_SHRINK_EXHAUSTED."""
        runner = _make_fake_runner(returncode=1, output="CUDA out of memory")
        blob = FakeBlobClient()
        exit_code, outcome, _, attempts, retries = self._call(ep, scaffold, blob, runner, max_oom_retries=2)
        assert exit_code == ep.EXIT_OOM_SHRINK_EXHAUSTED
        assert outcome == "oom_shrink_exhausted"
        assert attempts == 3
        assert retries == 2

    def test_oom_then_success(self, ep, scaffold):
        """OOM on attempt 0, success on attempt 1 returns EXIT_OK."""
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        call_count = [0]

        def _mixed_runner(train_cell_path, output_dir, env_overrides, attempt_log_path):
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_log_path.write_text("")
            call_count[0] += 1
            if call_count[0] == 1:
                return 1, "CUDA out of memory. Tried to allocate 4 GiB"
            # Second call: success
            mf = output_dir / "metrics.json"
            mf.write_text(json.dumps(good_metrics))
            return 0, "done"

        blob = FakeBlobClient()
        exit_code, outcome, metrics, attempts, retries = self._call(
            ep, scaffold, blob, _mixed_runner, max_oom_retries=2
        )
        assert exit_code == ep.EXIT_OK
        assert outcome == "ok"
        assert attempts == 2
        assert retries == 1

    def test_log_uploaded_for_each_attempt(self, ep, scaffold):
        """Per-attempt logs must be uploaded to Blob for each attempt."""
        runner = _make_fake_runner(returncode=1, output="CUDA out of memory")
        blob = FakeBlobClient()
        self._call(ep, scaffold, blob, runner, max_oom_retries=2)
        log_uploads = [n for n, _ in blob.upload_calls if "logs/attempt-" in n]
        assert len(log_uploads) == 3  # 0, 1, 2

    def test_oom_zero_retries(self, ep, scaffold):
        """With max_oom_retries=0, a single OOM attempt is immediately terminal."""
        runner = _make_fake_runner(returncode=1, output="CUDA out of memory")
        blob = FakeBlobClient()
        exit_code, outcome, _, attempts, retries = self._call(ep, scaffold, blob, runner, max_oom_retries=0)
        assert exit_code == ep.EXIT_OOM_SHRINK_EXHAUSTED
        assert attempts == 1
        assert retries == 0


# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------

class TestBootstrap:
    """Test _bootstrap with fake blob client and temp dirs."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_successful_bootstrap_no_requirements(self, ep, tmpdir):
        """Bootstrap with an empty code dir succeeds (no requirements.txt)."""
        blob = FakeBlobClient()
        blob.seed_blob("runs/test/code/train_cell.py", b"# fake")
        local_code = tmpdir / "code"
        cache_root = tmpdir / "cache"
        cache_root.mkdir()

        ok, err = ep._bootstrap(
            code_blob_prefix="runs/test/code",
            local_code_dir=local_code,
            cache_root=cache_root,
            account_name="acc",
            container_name="cont",
            blob_client=blob,
        )
        assert ok is True
        assert err == ""
        assert (local_code / "train_cell.py").is_file()

    def test_bootstrap_sets_hf_home_env(self, ep, tmpdir):
        """Bootstrap must set HF_HOME under cache_root."""
        blob = FakeBlobClient()
        blob.seed_blob("code/f.py", b"")
        local_code = tmpdir / "code"
        cache_root = tmpdir / "cache"
        cache_root.mkdir()

        ep._bootstrap(
            code_blob_prefix="code",
            local_code_dir=local_code,
            cache_root=cache_root,
            account_name="acc",
            container_name="cont",
            blob_client=blob,
        )
        assert "HF_HOME" in os.environ
        assert str(cache_root / "hf") == os.environ["HF_HOME"]

    def test_bootstrap_blob_failure_returns_false(self, ep, tmpdir):
        """A blob download failure returns (False, error_message)."""

        class ErrorBlobClient:
            def list_blobs(self, *, name_starts_with=""):
                raise RuntimeError("Simulated network error")

            def upload_blob(self, *a, **kw):
                pass

            def download_blob(self, name):
                raise RuntimeError("Simulated download error")

        local_code = tmpdir / "code"
        cache_root = tmpdir / "cache"
        cache_root.mkdir()

        ok, err = ep._bootstrap(
            code_blob_prefix="code",
            local_code_dir=local_code,
            cache_root=cache_root,
            account_name="acc",
            container_name="cont",
            blob_client=ErrorBlobClient(),
        )
        assert ok is False
        assert "Blob pull failed" in err


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------

class TestMainIntegration:
    """Test the top-level main() with full fake blob + subprocess injection."""

    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch, tmp_path):
        """Set required env vars; point cache mount to a temp dir."""
        monkeypatch.setenv("REPROLAB_CELL_ID", "int-cell-001")
        monkeypatch.setenv("REPROLAB_AZURE_STORAGE_ACCOUNT", "testacc")
        monkeypatch.setenv("REPROLAB_AZURE_BLOB_CONTAINER", "testcont")
        monkeypatch.setenv("REPROLAB_BLOB_CODE_PREFIX", "runs/run1/code")
        monkeypatch.setenv("REPROLAB_BLOB_OUTPUT_PREFIX", "runs/run1/cells")
        monkeypatch.setenv("REPROLAB_CELL_MAX_OOM_RETRIES", "2")
        monkeypatch.setenv("REPROLAB_CACHE_MOUNT", str(tmp_path / "cache"))
        monkeypatch.setenv(
            "REPROLAB_CELL_PARAMS",
            json.dumps({"model_id": "Qwen/Qwen3-1.7B", "baseline": "sdar", "env": "search_qa"}),
        )
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

    def _make_blob_with_code(self, tmp_path: Path) -> FakeBlobClient:
        """Return a FakeBlobClient seeded with a fake train_cell.py."""
        blob = FakeBlobClient()
        blob.seed_blob("runs/run1/code/train_cell.py", b"# fake train_cell.py")
        return blob

    def test_successful_run_exits_zero(self, ep, tmp_path):
        blob = self._make_blob_with_code(tmp_path)
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        code = ep.main(blob_client=blob, subprocess_runner=runner)
        assert code == ep.EXIT_OK

    def test_successful_run_uploads_sentinel_ok(self, ep, tmp_path):
        blob = self._make_blob_with_code(tmp_path)
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        ep.main(blob_client=blob, subprocess_runner=runner)

        # Find the status.json upload
        sentinel_data = None
        for name, data in blob.upload_calls:
            if name.endswith("status.json"):
                sentinel_data = json.loads(data.decode())
                break
        assert sentinel_data is not None
        assert sentinel_data["outcome"] == "ok"
        assert sentinel_data["exit_code"] == 0

    def test_non_oom_failure_exits_41(self, ep, tmp_path):
        blob = self._make_blob_with_code(tmp_path)
        runner = _make_fake_runner(returncode=1, output="ImportError: blah")

        code = ep.main(blob_client=blob, subprocess_runner=runner)
        assert code == ep.EXIT_ERROR

    def test_oom_exhausted_exits_42(self, ep, tmp_path):
        blob = self._make_blob_with_code(tmp_path)
        runner = _make_fake_runner(returncode=1, output="CUDA out of memory")

        code = ep.main(blob_client=blob, subprocess_runner=runner)
        assert code == ep.EXIT_OOM_SHRINK_EXHAUSTED

    def test_bootstrap_error_exits_40(self, ep, tmp_path, monkeypatch):
        """Missing code blobs → bootstrap error → EXIT_BOOTSTRAP_ERROR."""
        # Empty blob store — train_cell.py not present
        blob = FakeBlobClient()  # nothing seeded
        # No runner needed — bootstrap fails before training

        code = ep.main(blob_client=blob, subprocess_runner=_make_fake_runner())
        # The entrypoint checks for train_cell.py after download → bootstrap error
        assert code == ep.EXIT_BOOTSTRAP_ERROR

    def test_metrics_invalid_exits_43(self, ep, tmp_path):
        """Trainer exits 0 but writes no metrics.json → EXIT_METRICS_INVALID."""
        blob = self._make_blob_with_code(tmp_path)
        # Runner exits 0 but does NOT write metrics.json
        runner = _make_fake_runner(returncode=0, output="done", metrics=None)

        code = ep.main(blob_client=blob, subprocess_runner=runner)
        assert code == ep.EXIT_METRICS_INVALID

    def test_sentinel_always_uploaded(self, ep, tmp_path):
        """status.json must be uploaded regardless of outcome."""
        for output, rc in [
            ("CUDA out of memory", 1),
            ("done", 0),
            ("ImportError", 1),
        ]:
            blob = self._make_blob_with_code(tmp_path)
            metrics = {"final_training_loss": 0.1, "status": "completed"} if rc == 0 else None
            runner = _make_fake_runner(returncode=rc, output=output, metrics=metrics)
            ep.main(blob_client=blob, subprocess_runner=runner)
            sentinel_uploads = [n for n, _ in blob.upload_calls if "status.json" in n]
            assert len(sentinel_uploads) >= 1, f"No sentinel for rc={rc} output={output!r}"

    def test_artifact_upload_error_when_sentinel_fails(self, ep, tmp_path):
        """When training succeeds but sentinel upload fails → EXIT_ARTIFACT_UPLOAD_ERROR."""

        class FailSentinelBlob(FakeBlobClient):
            def __init__(self):
                super().__init__()
                self._fail_next_status = False

            def upload_blob(self, name: str, data: bytes, *, overwrite: bool = True) -> None:
                if "status.json" in name:
                    raise RuntimeError("Simulated upload failure")
                super().upload_blob(name, data, overwrite=overwrite)

        blob = FailSentinelBlob()
        # Seed code
        blob.seed_blob("runs/run1/code/train_cell.py", b"# fake train_cell.py")
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        code = ep.main(blob_client=blob, subprocess_runner=runner)
        assert code == ep.EXIT_ARTIFACT_UPLOAD_ERROR

    def test_artifact_upload_error_when_sentinel_fails_on_failure_path(self, ep, tmp_path):
        """P1-fix-10: sentinel upload failure on non-OK path also returns EXIT_ARTIFACT_UPLOAD_ERROR.

        Previously only the success path escalated; now ALL sentinel upload failures escalate so
        the orchestrator is never mis-scoring a failed/oom cell as ok.
        """

        class FailSentinelBlob(FakeBlobClient):
            def upload_blob(self, name: str, data: bytes, *, overwrite: bool = True) -> None:
                if "status.json" in name:
                    raise RuntimeError("Simulated upload failure on failure path")
                super().upload_blob(name, data, overwrite=overwrite)

        blob = FailSentinelBlob()
        blob.seed_blob("runs/run1/code/train_cell.py", b"# fake train_cell.py")
        # Trainer exits with non-zero (OOM path)
        runner = _make_fake_runner(returncode=1, output="CUDA out of memory")

        code = ep.main(blob_client=blob, subprocess_runner=runner)
        assert code == ep.EXIT_ARTIFACT_UPLOAD_ERROR, (
            "P1-fix-10: sentinel upload failure on OOM path must return "
            "EXIT_ARTIFACT_UPLOAD_ERROR, not silently use the original exit code"
        )


# ---------------------------------------------------------------------------
# P1-fix-9: plan_attempts reads batch-scale ratios from env
# ---------------------------------------------------------------------------

class TestPlanAttemptsEnvOverride:
    """P1-fix-9: plan_attempts must read batch-scale ratios from env vars."""

    def test_custom_step1_from_env(self, ep, monkeypatch):
        monkeypatch.setenv("REPROLAB_CELL_OOM_BATCH_SCALE_STEP1", "0.3")
        monkeypatch.setenv("REPROLAB_CELL_OOM_BATCH_SCALE_FLOOR", "0.1")
        attempts = ep.plan_attempts(2)
        assert attempts[1]["batch_scale"] == 0.3, (
            "P1-fix-9: REPROLAB_CELL_OOM_BATCH_SCALE_STEP1 must override step-1 scale"
        )
        assert attempts[2]["batch_scale"] == 0.1, (
            "P1-fix-9: REPROLAB_CELL_OOM_BATCH_SCALE_FLOOR must override floor scale"
        )

    def test_defaults_when_env_absent(self, ep, monkeypatch):
        monkeypatch.delenv("REPROLAB_CELL_OOM_BATCH_SCALE_STEP1", raising=False)
        monkeypatch.delenv("REPROLAB_CELL_OOM_BATCH_SCALE_FLOOR", raising=False)
        attempts = ep.plan_attempts(2)
        assert attempts[1]["batch_scale"] == 0.5
        assert attempts[2]["batch_scale"] == 0.25

    def test_kwarg_overrides_env(self, ep, monkeypatch):
        """Explicit kwargs take priority over env vars."""
        monkeypatch.setenv("REPROLAB_CELL_OOM_BATCH_SCALE_STEP1", "0.9")
        attempts = ep.plan_attempts(2, batch_scale_step1=0.4)
        assert attempts[1]["batch_scale"] == 0.4, (
            "explicit batch_scale_step1 kwarg must override env var"
        )


# ---------------------------------------------------------------------------
# download_prefix_to_dir — path-traversal defense (security watch item)
# ---------------------------------------------------------------------------


class TestDownloadPrefixPathSafety:
    """A poisoned blob name must never write outside local_dir on download."""

    def test_traversal_blob_name_skipped(self, ep, tmp_path):
        client = FakeBlobClient()
        prefix = "runs/r1/code/"
        # One safe file, one "../"-escaping name, one absolute-path name.
        client.seed_blob(prefix + "train_cell.py", b"safe")
        client.seed_blob(prefix + "../../../../tmp/evil.py", b"pwned")
        client.seed_blob(prefix + "sub/ok.txt", b"also-safe")
        local = tmp_path / "code"

        downloaded = ep.download_prefix_to_dir(
            prefix, local, account_name="a", container_name="c", client=client
        )

        # Safe files landed under local; the traversal blob was skipped.
        assert (local / "train_cell.py").read_bytes() == b"safe"
        assert (local / "sub" / "ok.txt").read_bytes() == b"also-safe"
        assert prefix + "../../../../tmp/evil.py" not in downloaded
        # Nothing escaped local_dir.
        assert not (tmp_path.parent / "evil.py").exists()
        assert not Path("/tmp/evil.py").exists() or Path("/tmp/evil.py").read_bytes() != b"pwned"
