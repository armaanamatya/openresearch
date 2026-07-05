"""Tests for the cell service-orchestration + GPU-partition seam.

A cell may declare ``cell["services"]`` — a list of AUXILIARY GPU-consuming
services (e.g. a FAISS retrieval server holding a large index) that must run
ALONGSIDE training for the cell's whole lifetime, on a DISJOINT slice of the
cell's leased GPUs. Presence-gated: a cell WITHOUT a non-empty ``services``
list behaves byte-for-byte as before this feature existed (no partition, no
service subprocess, ``slot`` == every assigned GPU id) — exactly like the
existing ``command``/``metrics_source`` cell seams.

Covers, in order:

  * ``_cell_gpu_count`` — the new ``"auto"`` sentinel (lease every GPU).
  * ``_partition_cell_gpus`` — the pure GPU-partition helper.
  * ``_wait_services_ready`` — the four readiness kinds (http/port/log/sleep),
    the fire-and-forget no-readiness-key case, the dead-proc cross-cutting
    check, and overall-deadline clamping.
  * ``_start_cell_services`` / ``_stop_cell_services`` — real subprocess
    start/kill (a trivial ``sleep``-based service), env propagation
    (``CUDA_VISIBLE_DEVICES`` from the assigned slice, including the
    CPU-only empty-slice case).
  * ``run_matrix`` wiring — OFF-parity (no/empty ``services`` is untouched),
    the ON-path partition + start/wait/train/stop sequence, the
    service-setup-failure short-circuit, the misconfiguration warning, and
    the ALWAYS-teardown guarantee (services are stopped and GPUs released on
    success, on a training FAILURE, and even when the training callable
    RAISES an unexpected exception).

Subprocess for the TRAINING call is mocked via ``patch.object(gcr, "_run_cell_subprocess", ...)``
— mirrors the established pattern in ``tests/agents/rlm/test_gpu_cell_runner.py`` and
``tests/rlm/test_cell_command_seam.py``. ``_start_cell_services``/``_stop_cell_services``
themselves are exercised against REAL (but trivial, short-lived) subprocesses —
no real GPUs or torch required.
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import backend.agents.rlm.gpu_cell_runner as gcr
from backend.agents.rlm.gpu_cell_runner import run_matrix


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return a currently-unused TCP port on 127.0.0.1 (nothing listening)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_log_until(log_path: Path, needle: str, *, timeout_s: float) -> str:
    """Poll ``log_path`` until it contains ``needle`` or ``timeout_s`` elapses."""
    deadline = time.monotonic() + timeout_s
    content = ""
    while time.monotonic() < deadline:
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="replace")
            if needle in content:
                return content
        time.sleep(0.05)
    return content


class _AlwaysAliveProc:
    """Minimal stand-in for ``subprocess.Popen`` — never exited (poll() -> None)."""

    def poll(self):
        return None


class _OkHttpHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — stdlib-mandated method name
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002 — silence default stderr logging
        pass


@pytest.fixture
def http_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OkHttpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _ok_subprocess(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                    grad_checkpoint, timeout_s, log_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return 0, ""


# ---------------------------------------------------------------------------
# 1. _cell_gpu_count — "auto" support
# ---------------------------------------------------------------------------

class TestCellGpuCountAuto:
    def test_auto_lowercase_returns_total_gpus(self):
        assert gcr._cell_gpu_count({"gpus": "auto"}, 1, 4) == 4

    def test_auto_case_insensitive_and_whitespace_tolerant(self):
        assert gcr._cell_gpu_count({"gpus": "AUTO"}, 1, 4) == 4
        assert gcr._cell_gpu_count({"gpus": " Auto "}, 1, 4) == 4

    def test_int_gpus_behaviour_unchanged(self):
        assert gcr._cell_gpu_count({"gpus": 2}, 1, 4) == 2
        assert gcr._cell_gpu_count({"gpus": 99}, 1, 4) == 4  # still clamped to total

    def test_absent_gpus_uses_default_unchanged(self):
        assert gcr._cell_gpu_count({}, 1, 4) == 1
        assert gcr._cell_gpu_count({}, 2, 4) == 2

    def test_non_dict_cell_uses_default_unchanged(self):
        assert gcr._cell_gpu_count("not-a-dict", 3, 4) == 3


# ---------------------------------------------------------------------------
# 2. _partition_cell_gpus — pure GPU partition
# ---------------------------------------------------------------------------

class TestPartitionCellGpus:
    def test_one_service_reserves_leading_gpu(self):
        train, slices = gcr._partition_cell_gpus(["0", "1", "2", "3"], [{"gpus": 1}])
        assert train == ["1", "2", "3"]
        assert slices == [["0"]]

    def test_two_services_reserve_sequential_slices(self):
        train, slices = gcr._partition_cell_gpus(
            ["0", "1", "2", "3"], [{"gpus": 1}, {"gpus": 1}]
        )
        assert train == ["2", "3"]
        assert slices == [["0"], ["1"]]

    def test_service_gpus_exceeding_assigned_gives_all_to_training(self):
        train, slices = gcr._partition_cell_gpus(["0", "1"], [{"gpus": 5}])
        assert train == ["0", "1"]
        assert slices == []

    def test_service_gpus_equal_to_assigned_gives_all_to_training(self):
        train, slices = gcr._partition_cell_gpus(["0", "1"], [{"gpus": 2}])
        assert train == ["0", "1"]
        assert slices == []

    def test_zero_gpu_service_gets_empty_slice_all_to_training(self):
        train, slices = gcr._partition_cell_gpus(["0", "1", "2", "3"], [{"gpus": 0}])
        assert train == ["0", "1", "2", "3"]
        assert slices == [[]]

    def test_non_dict_service_entries_contribute_zero_gpus(self):
        train, slices = gcr._partition_cell_gpus(
            ["0", "1", "2", "3"], ["not-a-dict", {"gpus": 1}]
        )
        assert train == ["1", "2", "3"]
        assert slices == [[], ["0"]]

    def test_pure_does_not_mutate_inputs(self):
        assigned = ["0", "1"]
        services = [{"gpus": 1}]
        gcr._partition_cell_gpus(assigned, services)
        assert assigned == ["0", "1"]
        assert services == [{"gpus": 1}]


# ---------------------------------------------------------------------------
# 3. _wait_services_ready — readiness kinds
# ---------------------------------------------------------------------------

class TestWaitServicesReadyHttp:
    def test_ready_when_server_responds(self, http_server):
        port = http_server.server_address[1]
        handle = {"name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": None}
        service = {
            "name": "svc",
            "readiness": {
                "kind": "http", "url": f"http://127.0.0.1:{port}/",
                "timeout_s": 5, "interval_s": 0.1,
            },
        }
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is True
        assert reason is None

    def test_not_ready_times_out_when_nothing_listens(self):
        port = _free_port()
        handle = {"name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": None}
        service = {
            "name": "svc",
            "readiness": {
                "kind": "http", "url": f"http://127.0.0.1:{port}/",
                "timeout_s": 0.3, "interval_s": 0.1,
            },
        }
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is False
        assert "svc" in reason


class TestWaitServicesReadyPort:
    def test_ready_when_listening(self, http_server):
        port = http_server.server_address[1]
        handle = {"name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": None}
        service = {
            "name": "svc",
            "readiness": {
                "kind": "port", "host": "127.0.0.1", "port": port,
                "timeout_s": 5, "interval_s": 0.1,
            },
        }
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is True

    def test_not_ready_times_out_when_nothing_listens(self):
        port = _free_port()
        handle = {"name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": None}
        service = {
            "name": "svc",
            "readiness": {"kind": "port", "port": port, "timeout_s": 0.3, "interval_s": 0.1},
        }
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is False
        assert "svc" in reason


class TestWaitServicesReadyLog:
    def test_ready_when_pattern_present(self, tmp_path):
        log_path = tmp_path / "svc.log"
        log_path.write_text("startup...\nServer is READY\n", encoding="utf-8")
        handle = {"name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": log_path}
        service = {
            "name": "svc",
            "readiness": {"kind": "log", "pattern": "READY", "timeout_s": 2, "interval_s": 0.1},
        }
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is True
        assert reason is None

    def test_not_ready_times_out_when_pattern_absent(self, tmp_path):
        log_path = tmp_path / "svc2.log"
        log_path.write_text("still starting...\n", encoding="utf-8")
        handle = {"name": "svc2", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": log_path}
        service = {
            "name": "svc2",
            "readiness": {
                "kind": "log", "pattern": "READY", "timeout_s": 0.2, "interval_s": 0.05,
            },
        }
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is False
        assert "svc2" in reason


class TestWaitServicesReadySleep:
    def test_ready_after_seconds_elapsed(self):
        handle = {
            "name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": None,
            "started_monotonic": time.monotonic(),
        }
        service = {
            "name": "svc",
            "readiness": {"kind": "sleep", "seconds": 0.15, "timeout_s": 5, "interval_s": 0.05},
        }
        start = time.monotonic()
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        elapsed = time.monotonic() - start
        assert ok is True
        assert elapsed >= 0.15 - 0.01  # small tolerance for scheduling jitter


class TestWaitServicesReadyFireAndForget:
    def test_no_readiness_key_is_ready_immediately_when_alive(self):
        handle = {"name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": None}
        service = {"name": "svc"}  # no "readiness" key at all
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is True
        assert reason is None


class TestWaitServicesReadyDeadProc:
    def test_dead_proc_is_not_ready(self):
        proc = subprocess.Popen(["bash", "-lc", "exit 3"])
        proc.wait(timeout=5)
        handle = {"name": "svc", "proc": proc, "pgid": None, "log_path": None}
        service = {"name": "svc"}  # fire-and-forget shape, but the proc already died
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is False
        assert "svc" in reason
        assert "3" in reason

    def test_failed_start_proc_none_is_not_ready(self):
        handle = {
            "name": "svc", "proc": None, "pgid": None, "log_path": None,
            "error": "boom: could not launch",
        }
        service = {"name": "svc"}
        ok, reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=None)
        assert ok is False
        assert "svc" in reason
        assert "boom" in reason


class TestWaitServicesReadyOverallDeadline:
    def test_overall_deadline_clamps_a_long_per_service_timeout(self):
        port = _free_port()
        handle = {"name": "svc", "proc": _AlwaysAliveProc(), "pgid": None, "log_path": None}
        service = {
            "name": "svc",
            "readiness": {"kind": "port", "port": port, "timeout_s": 300, "interval_s": 0.05},
        }
        deadline = time.monotonic() + 0.2  # far sooner than the 300s per-service timeout
        start = time.monotonic()
        ok, _reason = gcr._wait_services_ready([service], [handle], deadline_monotonic=deadline)
        elapsed = time.monotonic() - start
        assert ok is False
        assert elapsed < 2.0


# ---------------------------------------------------------------------------
# 4. _start_cell_services / _stop_cell_services — real (trivial) subprocess
# ---------------------------------------------------------------------------

class TestStartAndStopCellServices:
    def test_service_starts_alive_and_stop_kills_it(self, tmp_path):
        services = [{"name": "sleepy", "command": "sleep 300", "gpus": 0}]
        handles = gcr._start_cell_services(
            services, [[]],
            code_root=tmp_path, output_dir=tmp_path, base_env=dict(os.environ),
        )
        try:
            assert len(handles) == 1
            h = handles[0]
            assert h["error"] is None
            assert h["proc"] is not None
            assert h["proc"].poll() is None  # alive
        finally:
            gcr._stop_cell_services(handles)

        assert handles[0]["proc"].poll() is not None  # dead after teardown

    def test_service_sees_partitioned_cuda_visible_devices(self, tmp_path):
        services = [{
            "name": "echoer", "gpus": 1,
            "command": "echo CVD=[$CUDA_VISIBLE_DEVICES]; sleep 300",
        }]
        handles = gcr._start_cell_services(
            services, [["3"]],
            code_root=tmp_path, output_dir=tmp_path, base_env=dict(os.environ),
        )
        try:
            content = _read_log_until(
                tmp_path / "service_echoer.log", "CVD=", timeout_s=5.0
            )
            assert "CVD=[3]" in content
        finally:
            gcr._stop_cell_services(handles)

    def test_cpu_only_service_gets_empty_cuda_visible_devices(self, tmp_path):
        services = [{
            "name": "cpuecho", "gpus": 0,
            "command": "echo CVD=[$CUDA_VISIBLE_DEVICES]; sleep 300",
        }]
        handles = gcr._start_cell_services(
            services, [[]],
            code_root=tmp_path, output_dir=tmp_path, base_env=dict(os.environ),
        )
        try:
            content = _read_log_until(
                tmp_path / "service_cpuecho.log", "CVD=", timeout_s=5.0
            )
            assert "CVD=[]" in content
        finally:
            gcr._stop_cell_services(handles)

    def test_stop_is_idempotent_and_fail_soft(self, tmp_path):
        services = [{"name": "sleepy2", "command": "sleep 300", "gpus": 0}]
        handles = gcr._start_cell_services(
            services, [[]],
            code_root=tmp_path, output_dir=tmp_path, base_env=dict(os.environ),
        )
        gcr._stop_cell_services(handles)
        # Calling it again on already-dead/closed handles must never raise.
        gcr._stop_cell_services(handles)

    def test_non_dict_service_entry_yields_no_proc_handle(self, tmp_path):
        handles = gcr._start_cell_services(
            ["not-a-dict"], [[]],
            code_root=tmp_path, output_dir=tmp_path, base_env=dict(os.environ),
        )
        assert len(handles) == 1
        assert handles[0]["proc"] is None
        assert handles[0]["error"]
        # Must never raise (fail-soft) even though there is nothing to kill.
        gcr._stop_cell_services(handles)


# ---------------------------------------------------------------------------
# 5. run_matrix wiring — OFF-parity (byte-identical without services)
# ---------------------------------------------------------------------------

class TestOffParityNoServices:
    def test_no_services_key_slot_is_all_assigned_gpus(self, tmp_path):
        # "gpus": 4 so the cell actually leases the WHOLE pool (a cell with no
        # "gpus" key defaults to gpus_per_cell=1 regardless of pool size — that
        # default is orthogonal to what this test is about).
        cells = [{"id": "c0", "gpus": 4}]
        captured = {}

        def _capture(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                     grad_checkpoint, timeout_s, log_path):
            captured["gpu_id"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _capture):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1", "2", "3"]
            )

        # _acquire_gpus pops from its free-list (LIFO) and makes no ordering
        # promise ("take any k — cards are identical") — so assert the SET of
        # leased ids, not a specific join order.
        assert set(captured["gpu_id"].split(",")) == {"0", "1", "2", "3"}
        assert results["c0"]["gpu"] == captured["gpu_id"]
        assert list(tmp_path.rglob("service_*.log")) == []

    def test_empty_services_list_is_also_byte_identical(self, tmp_path):
        cells = [{"id": "c0", "services": [], "gpus": 2}]
        captured = {}

        def _capture(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                     grad_checkpoint, timeout_s, log_path):
            captured["gpu_id"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _capture):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])

        assert set(captured["gpu_id"].split(",")) == {"0", "1"}

    def test_non_dict_services_entries_are_also_byte_identical(self, tmp_path):
        cells = [{"id": "c0", "services": ["not-a-dict", 42], "gpus": 2}]
        captured = {}

        def _capture(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                     grad_checkpoint, timeout_s, log_path):
            captured["gpu_id"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _capture):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])

        assert set(captured["gpu_id"].split(",")) == {"0", "1"}

    def test_no_services_never_calls_service_helpers(self, tmp_path):
        cells = [{"id": "c0"}]
        with (
            patch.object(gcr, "_run_cell_subprocess", _ok_subprocess),
            patch.object(gcr, "_start_cell_services") as mock_start,
            patch.object(gcr, "_wait_services_ready") as mock_wait,
            patch.object(gcr, "_stop_cell_services") as mock_stop,
        ):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])

        mock_start.assert_not_called()
        mock_wait.assert_not_called()
        mock_stop.assert_not_called()


# ---------------------------------------------------------------------------
# 6. run_matrix wiring — ON path (partition, start/wait/train/stop, failures)
# ---------------------------------------------------------------------------

class TestRunMatrixServicesWiring:
    def test_partition_applied_and_start_stop_called(self, tmp_path):
        # "gpus": 4 so the cell leases the WHOLE pool — otherwise it defaults
        # to gpus_per_cell=1, which the 1-GPU service would then consume
        # entirely (a genuine misconfiguration, exercised separately below).
        cells = [{
            "id": "c0", "gpus": 4,
            "services": [{"name": "faiss", "gpus": 1, "command": "true"}],
        }]
        captured = {}
        fake_handles = [
            {"name": "faiss", "proc": None, "pgid": None, "log_path": tmp_path / "x.log"}
        ]

        def _capture_train(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                            grad_checkpoint, timeout_s, log_path):
            captured["gpu_id"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with (
            patch.object(gcr, "_run_cell_subprocess", _capture_train),
            patch.object(gcr, "_start_cell_services", return_value=fake_handles) as mock_start,
            patch.object(gcr, "_wait_services_ready", return_value=(True, None)) as mock_wait,
            patch.object(gcr, "_stop_cell_services") as mock_stop,
        ):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1", "2", "3"]
            )

        # _acquire_gpus makes no ordering promise ("take any k — cards are
        # identical"), so assert the PARTITION INVARIANTS rather than a
        # specific join order: exactly 1 GPU reserved for the service, 3 for
        # training, and the two sets are disjoint and together cover all 4.
        train_ids = set(captured["gpu_id"].split(","))
        service_slice = mock_start.call_args.args[1][0]
        assert len(service_slice) == 1
        assert len(train_ids) == 3
        assert train_ids.isdisjoint(service_slice)
        assert train_ids | set(service_slice) == {"0", "1", "2", "3"}

        assert results["c0"]["status"] == "ok"
        assert results["c0"]["gpu"] == captured["gpu_id"]
        mock_start.assert_called_once()
        assert mock_start.call_args.args[0] == cells[0]["services"]
        mock_wait.assert_called_once()
        assert mock_wait.call_args.kwargs["deadline_monotonic"] is None
        mock_stop.assert_called_once_with(fake_handles)

    def test_service_not_ready_skips_training_and_tears_down(self, tmp_path):
        cells = [{"id": "c0", "services": [{"name": "faiss", "gpus": 1, "command": "true"}]}]
        train_called = []
        fake_handles = [
            {"name": "faiss", "proc": None, "pgid": None, "log_path": tmp_path / "x.log"}
        ]

        def _train(*args, **kwargs):
            train_called.append(True)
            return 0, ""

        with (
            patch.object(gcr, "_run_cell_subprocess", _train),
            patch.object(gcr, "_start_cell_services", return_value=fake_handles),
            patch.object(
                gcr, "_wait_services_ready",
                return_value=(False, "service faiss not ready: boom"),
            ),
            patch.object(gcr, "_stop_cell_services") as mock_stop,
        ):
            results = run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])

        assert train_called == []
        assert results["c0"]["status"] == "service_setup_failed"
        assert "not ready" in results["c0"]["error"]
        mock_stop.assert_called_once_with(fake_handles)

        manifest = json.loads((tmp_path / "c0" / "cell_manifest.json").read_text())
        assert manifest["status"] == "service_setup_failed"

    def test_teardown_happens_on_training_failure(self, tmp_path):
        cells = [{"id": "c0", "services": [{"name": "faiss", "gpus": 1, "command": "true"}]}]
        fake_handles = [
            {"name": "faiss", "proc": None, "pgid": None, "log_path": tmp_path / "x.log"}
        ]

        def _fail(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                  grad_checkpoint, timeout_s, log_path):
            output_dir.mkdir(parents=True, exist_ok=True)
            return 1, "SyntaxError: invalid syntax"

        with (
            patch.object(gcr, "_run_cell_subprocess", _fail),
            patch.object(gcr, "_start_cell_services", return_value=fake_handles),
            patch.object(gcr, "_wait_services_ready", return_value=(True, None)),
            patch.object(gcr, "_stop_cell_services") as mock_stop,
        ):
            results = run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])

        assert results["c0"]["status"] == "error"
        mock_stop.assert_called_once_with(fake_handles)

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_teardown_happens_even_if_training_raises(self, tmp_path):
        """The strongest form of the ALWAYS-teardown guarantee: services are
        stopped and GPUs released even when the training call raises an
        unexpected exception (defense in depth — in production
        ``_run_cell_subprocess`` never raises, it always returns a tuple)."""
        cells = [{"id": "c0", "services": [{"name": "faiss", "gpus": 1, "command": "true"}]}]
        fake_handles = [
            {"name": "faiss", "proc": None, "pgid": None, "log_path": tmp_path / "x.log"}
        ]

        def _boom(*args, **kwargs):
            raise RuntimeError("training subprocess exploded")

        with (
            patch.object(gcr, "_run_cell_subprocess", _boom),
            patch.object(gcr, "_start_cell_services", return_value=fake_handles),
            patch.object(gcr, "_wait_services_ready", return_value=(True, None)),
            patch.object(gcr, "_stop_cell_services") as mock_stop,
        ):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])

        mock_stop.assert_called_once_with(fake_handles)

    def test_misconfigured_services_logs_warning_and_trains_on_all_gpus(self, tmp_path, caplog):
        cells = [{
            "id": "c0", "gpus": 4,
            "services": [{"name": "faiss", "gpus": 4, "command": "true"}],
        }]
        captured = {}

        def _capture_train(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                            grad_checkpoint, timeout_s, log_path):
            captured["gpu_id"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with (
            patch.object(gcr, "_run_cell_subprocess", _capture_train),
            patch.object(gcr, "_start_cell_services", return_value=[]) as mock_start,
            patch.object(gcr, "_wait_services_ready", return_value=(True, None)),
            patch.object(gcr, "_stop_cell_services"),
            caplog.at_level("WARNING", logger="backend.agents.rlm.gpu_cell_runner"),
        ):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1", "2", "3"])

        assert set(captured["gpu_id"].split(",")) == {"0", "1", "2", "3"}  # ALL to training
        assert mock_start.call_args.args[1] == []
        assert any("misconfigured" in rec.message for rec in caplog.records)

    def test_zero_gpu_services_do_not_trigger_misconfig_warning(self, tmp_path, caplog):
        """A cell whose services are all CPU-only (gpus:0) also gives ALL GPUs
        to training (svc_total == 0), but this is legitimate, not a
        misconfiguration — no warning should fire."""
        cells = [{
            "id": "c0", "gpus": 2,
            "services": [{"name": "cpu_svc", "gpus": 0, "command": "true"}],
        }]
        captured = {}

        def _capture_train(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                            grad_checkpoint, timeout_s, log_path):
            captured["gpu_id"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with (
            patch.object(gcr, "_run_cell_subprocess", _capture_train),
            patch.object(gcr, "_start_cell_services", return_value=[]),
            patch.object(gcr, "_wait_services_ready", return_value=(True, None)),
            patch.object(gcr, "_stop_cell_services"),
            caplog.at_level("WARNING", logger="backend.agents.rlm.gpu_cell_runner"),
        ):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])

        assert set(captured["gpu_id"].split(",")) == {"0", "1"}
        assert not any("misconfigured" in rec.message for rec in caplog.records)

    def test_gpu_released_after_service_setup_failure_for_next_cell(self, tmp_path):
        cells = [
            {"id": "c0", "services": [{"name": "svc", "gpus": 1, "command": "true"}]},
            {"id": "c1"},
        ]
        train_calls = []

        def _train(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                   grad_checkpoint, timeout_s, log_path):
            train_calls.append((cell["id"], gpu_id))
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        fake_handles = [
            {"name": "svc", "proc": None, "pgid": None, "log_path": tmp_path / "x.log"}
        ]

        def _wait_ready(services, handles, *, deadline_monotonic):
            return False, "service svc not ready: boom"

        with (
            patch.object(gcr, "_run_cell_subprocess", _train),
            patch.object(gcr, "_start_cell_services", return_value=fake_handles),
            patch.object(gcr, "_wait_services_ready", side_effect=_wait_ready),
            patch.object(gcr, "_stop_cell_services"),
        ):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0"], max_parallel=1,
            )

        assert results["c0"]["status"] == "service_setup_failed"
        assert results["c1"]["status"] == "ok"  # only possible if GPU "0" was released
        assert train_calls == [("c1", "0")]


# ---------------------------------------------------------------------------
# 7. gpus: "auto" end-to-end through run_matrix
# ---------------------------------------------------------------------------

class TestRunMatrixAutoGpus:
    def test_auto_gpus_cell_leases_every_available_gpu(self, tmp_path):
        cells = [{"id": "c0", "gpus": "auto"}]
        captured = {}

        def _capture(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                     grad_checkpoint, timeout_s, log_path):
            captured["gpu_id"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _capture):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1", "2", "3"])

        assert set(captured["gpu_id"].split(",")) == {"0", "1", "2", "3"}


# ---------------------------------------------------------------------------
# 8. OPENRESEARCH_CELL_TRAIN_GPUS export in _run_cell_subprocess
# ---------------------------------------------------------------------------

class TestTrainGpusEnvExport:
    def _run(self, tmp_path: Path, gpu_id: str, command: str = "true") -> dict:
        import io

        code = tmp_path / "code"
        code.mkdir(parents=True, exist_ok=True)
        cell_script = code / "train_cell.py"
        output_dir = tmp_path / "c0"
        log_path = tmp_path / "c0.log"
        captured: dict = {}

        class _FakeProc:
            pid = 12345
            returncode = 0
            stdout = io.StringIO("")

            def wait(self, timeout=None):
                return 0

        fake_proc = _FakeProc()

        def _popen(cmd, **kwargs):
            captured["env"] = dict(kwargs["env"])
            return fake_proc

        with (
            patch("backend.agents.rlm.gpu_cell_runner.subprocess.Popen", _popen),
            patch("backend.agents.rlm.gpu_cell_runner._orphan_register"),
            patch("backend.agents.rlm.gpu_cell_runner._orphan_deregister"),
        ):
            gcr._run_cell_subprocess(
                cell={"id": "c0", "command": command},
                cell_script=str(cell_script),
                gpu_id=gpu_id,
                output_dir=output_dir,
                batch_scale=None,
                grad_checkpoint=False,
                timeout_s=None,
                log_path=log_path,
            )
        return captured

    def test_single_gpu_exports_count_one(self, tmp_path):
        # Command cell: the authors' launcher gets the training GPU count.
        captured = self._run(tmp_path, "0", command="true")
        assert captured["env"]["OPENRESEARCH_CELL_TRAIN_GPUS"] == "1"

    def test_multi_gpu_slot_exports_matching_count(self, tmp_path):
        captured = self._run(tmp_path, "1,2,3", command="true")
        assert captured["env"]["OPENRESEARCH_CELL_TRAIN_GPUS"] == "3"

    def test_default_path_does_not_export_train_gpus(self, tmp_path):
        # Byte-identity: the default train_cell.py path (no command) learns its
        # GPUs from CUDA_VISIBLE_DEVICES and must NOT gain a new env key.
        captured = self._run(tmp_path, "1,2,3", command="")
        assert "OPENRESEARCH_CELL_TRAIN_GPUS" not in captured["env"]
