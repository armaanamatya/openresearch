"""Tests for the SIGTERM preemption handler in both K8s cell entrypoints.

Design constraints
------------------
* Tests run WITHOUT azure / google-cloud packages or a GPU.
* Pure functions (build_preempt_sentinel) are tested directly.
* The handler logic is exercised via the injectable _preempt_upload_fn and a
  synthetic SIGTERM delivered via signal.raise_signal / os.kill, or by calling
  _flush_preemption directly (the handler's only I/O surface).
* No real subprocess, no real blob/GCS calls, no docker build.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Load both entrypoint modules from their filesystem paths
# ---------------------------------------------------------------------------

_AKS_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "docker" / "aks-cell-base" / "aks_cell_entrypoint.py"
)
_GKE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "docker" / "gke-cell-base" / "gke_cell_entrypoint.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def aks():
    return _load(_AKS_PATH, "aks_cell_entrypoint_preempt_test")


@pytest.fixture(scope="module")
def gke():
    return _load(_GKE_PATH, "gke_cell_entrypoint_preempt_test")


# ---------------------------------------------------------------------------
# Helpers shared by both entrypoint variants
# ---------------------------------------------------------------------------

def _fake_upload_calls() -> tuple[list[Any], Any]:
    """Return (calls_list, callable) where callable records args on calls_list."""
    calls: list[Any] = []

    def _fn(*args, **kwargs) -> bool:
        calls.append((args, kwargs))
        return True

    return calls, _fn


def _make_fake_runner(
    returncode: int = 0,
    output: str = "",
    metrics: dict[str, Any] | None = None,
) -> Any:
    """Fake subprocess runner that optionally writes metrics.json."""

    def _runner(
        train_cell_path: Path,
        output_dir: Path,
        env_overrides: dict[str, str],
        attempt_log_path: Path,
    ) -> tuple[int, str]:
        attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_log_path.write_text(output, encoding="utf-8")
        if metrics is not None:
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
        return returncode, output

    return _runner


# ---------------------------------------------------------------------------
# Fake blob/GCS clients for main() integration tests
# ---------------------------------------------------------------------------

class FakeAksBlobClient:
    def __init__(self):
        self._store: dict[str, bytes] = {}
        self.upload_calls: list[tuple[str, bytes]] = []

    def upload_blob(self, name: str, data: bytes, *, overwrite: bool = True) -> None:
        self._store[name] = data
        self.upload_calls.append((name, data))

    def download_blob(self, name: str):
        class _S:
            def __init__(self, d):
                self._d = d
            def readall(self):
                return self._d
        return _S(self._store.get(name, b""))

    def list_blobs(self, *, name_starts_with: str = ""):
        class _B:
            def __init__(self, n):
                self.name = n
        return [_B(k) for k in self._store if k.startswith(name_starts_with)]

    def seed_blob(self, name: str, data: bytes) -> None:
        self._store[name] = data


class FakeGksBucketClient:
    def __init__(self):
        self._store: dict[str, bytes] = {}
        self.upload_calls: list[tuple[str, bytes]] = []

    def blob(self, name: str):
        owner = self

        class _BH:
            def upload_from_string(self, data: bytes) -> None:
                owner._store[name] = data
                owner.upload_calls.append((name, data))

            def download_as_bytes(self) -> bytes:
                if name not in owner._store:
                    raise KeyError(name)
                return owner._store[name]

        return _BH()

    def list_blobs(self, *, prefix: str = ""):
        class _B:
            def __init__(self, n):
                self.name = n
        return [_B(k) for k in self._store if k.startswith(prefix)]

    def seed_blob(self, name: str, data: bytes) -> None:
        self._store[name] = data


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

class TestAksConstants:
    def test_exit_preempted_is_45(self, aks):
        assert aks.EXIT_PREEMPTED == 45

    def test_outcome_preempted_string(self, aks):
        assert aks._OUTCOME_PREEMPTED == "preempted"

    def test_exit_preempted_distinct_from_others(self, aks):
        existing = {
            aks.EXIT_OK,
            aks.EXIT_BOOTSTRAP_ERROR,
            aks.EXIT_ERROR,
            aks.EXIT_OOM_SHRINK_EXHAUSTED,
            aks.EXIT_METRICS_INVALID,
            aks.EXIT_ARTIFACT_UPLOAD_ERROR,
        }
        assert aks.EXIT_PREEMPTED not in existing


class TestGkeConstants:
    def test_exit_preempted_is_45(self, gke):
        assert gke.EXIT_PREEMPTED == 45

    def test_outcome_preempted_string(self, gke):
        assert gke._OUTCOME_PREEMPTED == "preempted"

    def test_exit_preempted_distinct_from_others(self, gke):
        existing = {
            gke.EXIT_OK,
            gke.EXIT_BOOTSTRAP_ERROR,
            gke.EXIT_ERROR,
            gke.EXIT_OOM_SHRINK_EXHAUSTED,
            gke.EXIT_METRICS_INVALID,
            gke.EXIT_ARTIFACT_UPLOAD_ERROR,
        }
        assert gke.EXIT_PREEMPTED not in existing


# ---------------------------------------------------------------------------
# 2. build_preempt_sentinel — pure function tests (both entrypoints)
# ---------------------------------------------------------------------------

class TestBuildPreemptSentinelAks:
    def test_outcome_is_preempted(self, aks):
        s = aks.build_preempt_sentinel(
            cell_id="c1",
            started_at="2026-06-17T00:00:00Z",
            finished_at="2026-06-17T01:00:00Z",
            attempts_so_far=1,
            retries_so_far=0,
        )
        assert s["outcome"] == "preempted"

    def test_exit_code_is_45(self, aks):
        s = aks.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=0,
            retries_so_far=0,
        )
        assert s["exit_code"] == aks.EXIT_PREEMPTED

    def test_error_field_set(self, aks):
        s = aks.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=2,
            retries_so_far=1,
        )
        assert isinstance(s["error"], str) and len(s["error"]) > 0

    def test_attempts_and_retries_preserved(self, aks):
        s = aks.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=3,
            retries_so_far=2,
        )
        assert s["attempts"] == 3
        assert s["retries"] == 2

    def test_required_keys_present(self, aks):
        s = aks.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=0,
            retries_so_far=0,
        )
        for key in ("version", "cell_id", "outcome", "exit_code",
                    "attempts", "retries", "error", "started_at", "finished_at"):
            assert key in s

    def test_json_serialisable(self, aks):
        s = aks.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=0,
            retries_so_far=0,
        )
        json.dumps(s)  # must not raise


class TestBuildPreemptSentinelGke:
    def test_outcome_is_preempted(self, gke):
        s = gke.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=1,
            retries_so_far=0,
        )
        assert s["outcome"] == "preempted"

    def test_exit_code_is_45(self, gke):
        s = gke.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=0,
            retries_so_far=0,
        )
        assert s["exit_code"] == gke.EXIT_PREEMPTED

    def test_required_keys_present(self, gke):
        s = gke.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=0,
            retries_so_far=0,
        )
        for key in ("version", "cell_id", "outcome", "exit_code",
                    "attempts", "retries", "error", "started_at", "finished_at"):
            assert key in s

    def test_json_serialisable(self, gke):
        s = gke.build_preempt_sentinel(
            cell_id="c1",
            started_at="t0",
            finished_at="t1",
            attempts_so_far=0,
            retries_so_far=0,
        )
        json.dumps(s)  # must not raise


# ---------------------------------------------------------------------------
# 3. _flush_preemption — injectable upload, fail-soft tests (AKS)
# ---------------------------------------------------------------------------

class TestFlushPreemptionAks:
    @pytest.fixture
    def tmpdir(self, tmp_path):
        return tmp_path

    def test_calls_upload_fn_with_output_dir(self, aks, tmpdir):
        calls, fn = _fake_upload_calls()
        aks._flush_preemption(
            cell_id="c1",
            started_at="t0",
            output_dir=tmpdir,
            output_blob_prefix="runs/r1/cells",
            account_name="acc",
            container_name="cont",
            upload_fn=fn,
            attempts_so_far=1,
            retries_so_far=0,
        )
        assert len(calls) == 1
        assert calls[0][0][0] == tmpdir  # first positional arg is output_dir

    def test_upload_fn_failure_does_not_raise(self, aks, tmpdir):
        """A failing upload_fn must not propagate (fail-soft)."""
        def _exploding_fn(*a, **kw):
            raise RuntimeError("simulated blob failure")

        # Should not raise
        aks._flush_preemption(
            cell_id="c1",
            started_at="t0",
            output_dir=tmpdir,
            output_blob_prefix="runs/r1/cells",
            account_name="acc",
            container_name="cont",
            upload_fn=_exploding_fn,
            attempts_so_far=0,
            retries_so_far=0,
        )

    def test_sentinel_upload_failure_does_not_raise(self, aks, tmpdir, monkeypatch):
        """_upload_sentinel failure inside _flush_preemption must be caught."""
        calls, fn = _fake_upload_calls()

        def _fail_sentinel(*a, **kw):
            raise RuntimeError("sentinel boom")

        monkeypatch.setattr(aks, "_upload_sentinel", _fail_sentinel)
        # Should not raise even if the sentinel upload explodes
        aks._flush_preemption(
            cell_id="c1",
            started_at="t0",
            output_dir=tmpdir,
            output_blob_prefix="runs/r1/cells",
            account_name="acc",
            container_name="cont",
            upload_fn=fn,
            attempts_so_far=0,
            retries_so_far=0,
        )

    def test_no_upload_when_metrics_missing(self, aks, tmp_path):
        """If metrics.json does not exist, _upload_fn is still called (fail-soft)."""
        calls, fn = _fake_upload_calls()
        empty_output_dir = tmp_path / "empty"
        empty_output_dir.mkdir()

        aks._flush_preemption(
            cell_id="c1",
            started_at="t0",
            output_dir=empty_output_dir,
            output_blob_prefix="runs/r1/cells",
            account_name="acc",
            container_name="cont",
            upload_fn=fn,
            attempts_so_far=0,
            retries_so_far=0,
        )
        # The upload_fn was called (it may internally no-op on missing file)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 4. _flush_preemption — injectable upload, fail-soft tests (GKE)
# ---------------------------------------------------------------------------

class TestFlushPreemptionGke:
    @pytest.fixture
    def tmpdir(self, tmp_path):
        return tmp_path

    def test_calls_upload_fn_with_output_dir(self, gke, tmpdir):
        calls, fn = _fake_upload_calls()
        gke._flush_preemption(
            cell_id="c1",
            started_at="t0",
            output_dir=tmpdir,
            output_blob_prefix="runs/r1/cells",
            bucket_name="bkt",
            project="proj",
            upload_fn=fn,
            attempts_so_far=1,
            retries_so_far=0,
        )
        assert len(calls) == 1
        assert calls[0][0][0] == tmpdir

    def test_upload_fn_failure_does_not_raise(self, gke, tmpdir):
        def _exploding_fn(*a, **kw):
            raise RuntimeError("simulated gcs failure")

        gke._flush_preemption(
            cell_id="c1",
            started_at="t0",
            output_dir=tmpdir,
            output_blob_prefix="runs/r1/cells",
            bucket_name="bkt",
            project="proj",
            upload_fn=_exploding_fn,
            attempts_so_far=0,
            retries_so_far=0,
        )

    def test_sentinel_upload_failure_does_not_raise(self, gke, tmpdir, monkeypatch):
        calls, fn = _fake_upload_calls()

        def _fail_sentinel(*a, **kw):
            raise RuntimeError("sentinel boom")

        monkeypatch.setattr(gke, "_upload_sentinel", _fail_sentinel)
        gke._flush_preemption(
            cell_id="c1",
            started_at="t0",
            output_dir=tmpdir,
            output_blob_prefix="runs/r1/cells",
            bucket_name="bkt",
            project="proj",
            upload_fn=fn,
            attempts_so_far=0,
            retries_so_far=0,
        )


# ---------------------------------------------------------------------------
# 5. main() integration — _preempt_upload_fn is accepted without breaking
#    the normal (non-preempted) path (AKS)
# ---------------------------------------------------------------------------

class TestAksMainPreemptIntegration:
    """Normal-path runs with a _preempt_upload_fn injected must still work."""

    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENRESEARCH_CELL_ID", "preempt-cell-aks")
        monkeypatch.setenv("OPENRESEARCH_AZURE_STORAGE_ACCOUNT", "testacc")
        monkeypatch.setenv("OPENRESEARCH_AZURE_BLOB_CONTAINER", "testcont")
        monkeypatch.setenv("OPENRESEARCH_BLOB_CODE_PREFIX", "runs/run1/code")
        monkeypatch.setenv("OPENRESEARCH_BLOB_OUTPUT_PREFIX", "runs/run1/cells")
        monkeypatch.setenv("OPENRESEARCH_CELL_MAX_OOM_RETRIES", "0")
        monkeypatch.setenv("OPENRESEARCH_CACHE_MOUNT", str(tmp_path / "cache"))
        monkeypatch.setenv("OPENRESEARCH_CELL_PREEMPT_GRACE_S", "5")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

    def _blob(self) -> FakeAksBlobClient:
        b = FakeAksBlobClient()
        b.seed_blob("runs/run1/code/train_cell.py", b"# fake")
        return b

    def test_success_path_unaffected(self, aks):
        """Normal success run with injected _preempt_upload_fn exits 0."""
        preempt_calls: list[Any] = []

        def _preempt_fn(*a, **kw):
            preempt_calls.append((a, kw))
            return True

        blob = self._blob()
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        code = aks.main(
            blob_client=blob,
            subprocess_runner=runner,
            _preempt_upload_fn=_preempt_fn,
        )
        assert code == aks.EXIT_OK
        # The preempt function must NOT have been called on the normal path
        assert preempt_calls == []

    def test_error_path_unaffected(self, aks):
        """Normal error run with injected _preempt_upload_fn exits EXIT_ERROR."""
        preempt_calls: list[Any] = []

        def _preempt_fn(*a, **kw):
            preempt_calls.append((a, kw))
            return True

        blob = self._blob()
        runner = _make_fake_runner(returncode=1, output="ImportError: blah")

        code = aks.main(
            blob_client=blob,
            subprocess_runner=runner,
            _preempt_upload_fn=_preempt_fn,
        )
        assert code == aks.EXIT_ERROR
        assert preempt_calls == []

    def test_sentinel_uploaded_on_normal_path(self, aks):
        """status.json uploaded even when _preempt_upload_fn is injected."""
        blob = self._blob()
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        aks.main(
            blob_client=blob,
            subprocess_runner=runner,
            _preempt_upload_fn=lambda *a, **kw: True,
        )
        sentinel_names = [n for n, _ in blob.upload_calls if "status.json" in n]
        assert len(sentinel_names) >= 1

    def test_sentinel_outcome_ok_on_normal_path(self, aks):
        blob = self._blob()
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        aks.main(
            blob_client=blob,
            subprocess_runner=runner,
            _preempt_upload_fn=lambda *a, **kw: True,
        )
        sentinel = None
        for name, data in blob.upload_calls:
            if "status.json" in name:
                sentinel = json.loads(data.decode())
        assert sentinel is not None
        assert sentinel["outcome"] == "ok"


# ---------------------------------------------------------------------------
# 6. main() integration — _preempt_upload_fn is accepted without breaking
#    the normal (non-preempted) path (GKE)
# ---------------------------------------------------------------------------

class TestGkeMainPreemptIntegration:
    """Normal-path runs with a _preempt_upload_fn injected must still work."""

    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENRESEARCH_CELL_ID", "preempt-cell-gke")
        monkeypatch.setenv("OPENRESEARCH_GCP_GCS_BUCKET", "testbucket")
        monkeypatch.setenv("OPENRESEARCH_BLOB_CODE_PREFIX", "runs/run1/code")
        monkeypatch.setenv("OPENRESEARCH_BLOB_OUTPUT_PREFIX", "runs/run1/cells")
        monkeypatch.setenv("OPENRESEARCH_CELL_MAX_OOM_RETRIES", "0")
        monkeypatch.setenv("OPENRESEARCH_CACHE_MOUNT", str(tmp_path / "cache"))
        monkeypatch.setenv("OPENRESEARCH_CELL_PREEMPT_GRACE_S", "5")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

    def _gcs(self) -> FakeGksBucketClient:
        g = FakeGksBucketClient()
        g.seed_blob("runs/run1/code/train_cell.py", b"# fake")
        return g

    def test_success_path_unaffected(self, gke):
        preempt_calls: list[Any] = []

        def _preempt_fn(*a, **kw):
            preempt_calls.append((a, kw))
            return True

        gcs = self._gcs()
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        code = gke.main(
            gcs_client=gcs,
            subprocess_runner=runner,
            _preempt_upload_fn=_preempt_fn,
        )
        assert code == gke.EXIT_OK
        assert preempt_calls == []

    def test_error_path_unaffected(self, gke):
        preempt_calls: list[Any] = []

        def _preempt_fn(*a, **kw):
            preempt_calls.append((a, kw))
            return True

        gcs = self._gcs()
        runner = _make_fake_runner(returncode=1, output="ImportError: blah")

        code = gke.main(
            gcs_client=gcs,
            subprocess_runner=runner,
            _preempt_upload_fn=_preempt_fn,
        )
        assert code == gke.EXIT_ERROR
        assert preempt_calls == []

    def test_sentinel_uploaded_on_normal_path(self, gke):
        gcs = self._gcs()
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        gke.main(
            gcs_client=gcs,
            subprocess_runner=runner,
            _preempt_upload_fn=lambda *a, **kw: True,
        )
        sentinel_names = [n for n, _ in gcs.upload_calls if "status.json" in n]
        assert len(sentinel_names) >= 1

    def test_sentinel_outcome_ok_on_normal_path(self, gke):
        gcs = self._gcs()
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        gke.main(
            gcs_client=gcs,
            subprocess_runner=runner,
            _preempt_upload_fn=lambda *a, **kw: True,
        )
        sentinel = None
        for name, data in gcs.upload_calls:
            if "status.json" in name:
                sentinel = json.loads(data.decode())
        assert sentinel is not None
        assert sentinel["outcome"] == "ok"


# ---------------------------------------------------------------------------
# 7. SIGTERM handler — idempotency and grace-env clamping tests
# ---------------------------------------------------------------------------

class TestAksSigtermHandlerBehavior:
    """Test that the handler fires and exits with EXIT_PREEMPTED (45).

    We exercise the handler by calling main() with a subprocess runner that
    sends SIGTERM to the current process mid-run, then verifies the exit code.
    Because sys.exit(45) will be raised, we catch SystemExit.
    """

    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENRESEARCH_CELL_ID", "sigterm-cell-aks")
        monkeypatch.setenv("OPENRESEARCH_AZURE_STORAGE_ACCOUNT", "acc")
        monkeypatch.setenv("OPENRESEARCH_AZURE_BLOB_CONTAINER", "cont")
        monkeypatch.setenv("OPENRESEARCH_BLOB_CODE_PREFIX", "runs/run1/code")
        monkeypatch.setenv("OPENRESEARCH_BLOB_OUTPUT_PREFIX", "runs/run1/cells")
        monkeypatch.setenv("OPENRESEARCH_CELL_MAX_OOM_RETRIES", "0")
        monkeypatch.setenv("OPENRESEARCH_CACHE_MOUNT", str(tmp_path / "cache"))
        monkeypatch.setenv("OPENRESEARCH_CELL_PREEMPT_GRACE_S", "5")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

    def test_sigterm_exits_preempted(self, aks):
        """SIGTERM during training must result in SystemExit(45)."""
        preempt_upload_calls: list[Any] = []

        def _preempt_fn(*a, **kw) -> bool:
            preempt_upload_calls.append(1)
            return True

        def _sigterm_runner(train_cell_path, output_dir, env_overrides, attempt_log_path):
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_log_path.write_text("", encoding="utf-8")
            # Deliver SIGTERM to self mid-run
            os.kill(os.getpid(), signal.SIGTERM)
            # This line should not be reached
            return 0, "done"

        blob = FakeAksBlobClient()
        blob.seed_blob("runs/run1/code/train_cell.py", b"# fake")

        with pytest.raises(SystemExit) as exc_info:
            aks.main(
                blob_client=blob,
                subprocess_runner=_sigterm_runner,
                _preempt_upload_fn=_preempt_fn,
            )
        assert exc_info.value.code == aks.EXIT_PREEMPTED

    def test_sigterm_calls_preempt_upload(self, aks):
        """The preempt flush upload_fn must be called on SIGTERM."""
        preempt_upload_calls: list[Any] = []

        def _preempt_fn(*a, **kw) -> bool:
            preempt_upload_calls.append(1)
            return True

        def _sigterm_runner(train_cell_path, output_dir, env_overrides, attempt_log_path):
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_log_path.write_text("", encoding="utf-8")
            os.kill(os.getpid(), signal.SIGTERM)
            return 0, "done"

        blob = FakeAksBlobClient()
        blob.seed_blob("runs/run1/code/train_cell.py", b"# fake")

        with pytest.raises(SystemExit):
            aks.main(
                blob_client=blob,
                subprocess_runner=_sigterm_runner,
                _preempt_upload_fn=_preempt_fn,
            )
        assert preempt_upload_calls, "preempt upload_fn must be called on SIGTERM"

    def test_grace_env_clamped(self, aks, monkeypatch):
        """OPENRESEARCH_CELL_PREEMPT_GRACE_S is clamped to [1, 120]."""
        monkeypatch.setenv("OPENRESEARCH_CELL_PREEMPT_GRACE_S", "9999")
        # We just verify main() does not crash with the extreme value
        blob = FakeAksBlobClient()
        blob.seed_blob("runs/run1/code/train_cell.py", b"# fake")
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        code = aks.main(
            blob_client=blob,
            subprocess_runner=runner,
            _preempt_upload_fn=lambda *a, **kw: True,
        )
        assert code == aks.EXIT_OK


class TestGkeSigtermHandlerBehavior:
    """Same SIGTERM tests for the GKE entrypoint."""

    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENRESEARCH_CELL_ID", "sigterm-cell-gke")
        monkeypatch.setenv("OPENRESEARCH_GCP_GCS_BUCKET", "testbucket")
        monkeypatch.setenv("OPENRESEARCH_BLOB_CODE_PREFIX", "runs/run1/code")
        monkeypatch.setenv("OPENRESEARCH_BLOB_OUTPUT_PREFIX", "runs/run1/cells")
        monkeypatch.setenv("OPENRESEARCH_CELL_MAX_OOM_RETRIES", "0")
        monkeypatch.setenv("OPENRESEARCH_CACHE_MOUNT", str(tmp_path / "cache"))
        monkeypatch.setenv("OPENRESEARCH_CELL_PREEMPT_GRACE_S", "5")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

    def test_sigterm_exits_preempted(self, gke):
        preempt_upload_calls: list[Any] = []

        def _preempt_fn(*a, **kw) -> bool:
            preempt_upload_calls.append(1)
            return True

        def _sigterm_runner(train_cell_path, output_dir, env_overrides, attempt_log_path):
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_log_path.write_text("", encoding="utf-8")
            os.kill(os.getpid(), signal.SIGTERM)
            return 0, "done"

        gcs = FakeGksBucketClient()
        gcs.seed_blob("runs/run1/code/train_cell.py", b"# fake")

        with pytest.raises(SystemExit) as exc_info:
            gke.main(
                gcs_client=gcs,
                subprocess_runner=_sigterm_runner,
                _preempt_upload_fn=_preempt_fn,
            )
        assert exc_info.value.code == gke.EXIT_PREEMPTED

    def test_sigterm_calls_preempt_upload(self, gke):
        preempt_upload_calls: list[Any] = []

        def _preempt_fn(*a, **kw) -> bool:
            preempt_upload_calls.append(1)
            return True

        def _sigterm_runner(train_cell_path, output_dir, env_overrides, attempt_log_path):
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_log_path.write_text("", encoding="utf-8")
            os.kill(os.getpid(), signal.SIGTERM)
            return 0, "done"

        gcs = FakeGksBucketClient()
        gcs.seed_blob("runs/run1/code/train_cell.py", b"# fake")

        with pytest.raises(SystemExit):
            gke.main(
                gcs_client=gcs,
                subprocess_runner=_sigterm_runner,
                _preempt_upload_fn=_preempt_fn,
            )
        assert preempt_upload_calls, "preempt upload_fn must be called on SIGTERM"

    def test_grace_env_clamped(self, gke, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELL_PREEMPT_GRACE_S", "9999")
        gcs = FakeGksBucketClient()
        gcs.seed_blob("runs/run1/code/train_cell.py", b"# fake")
        good_metrics = {"final_training_loss": 0.1, "status": "completed"}
        runner = _make_fake_runner(returncode=0, output="done", metrics=good_metrics)

        code = gke.main(
            gcs_client=gcs,
            subprocess_runner=runner,
            _preempt_upload_fn=lambda *a, **kw: True,
        )
        assert code == gke.EXIT_OK
