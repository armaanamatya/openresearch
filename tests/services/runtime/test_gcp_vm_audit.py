"""Tests for the read-only GCP VM stray-billing audit (backend/services/runtime/gcp_vm_audit.py).

Hermetic: every test injects a fake runner returning a canned VmExecResult --
nothing here executes a real gcloud subprocess. This module must NEVER call
delete/stop, so there is no destructive path to test in the first place; the
tests instead pin the read-only argv shape and the RUNNING/STOPPED/NOT_FOUND
classification.
"""

from __future__ import annotations

import json

import pytest

from backend.services.runtime.gcp_vm_audit import (
    VmAuditTarget,
    VmExecResult,
    audit,
    build_describe_argv,
    count_active_local_gcp_runs,
    resolve_audit_targets,
)

_ENV_KEYS = (
    "OPENRESEARCH_GCP_PROJECT",
    "OPENRESEARCH_GCP_ZONE",
    "OPENRESEARCH_GCP_INSTANCE",
    "OPENRESEARCH_GCP_CPU_INSTANCE",
)


@pytest.fixture(autouse=True)
def _clear_ambient_gcp_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _target() -> VmAuditTarget:
    return VmAuditTarget(label="gpu", project="proj-x", zone="us-central1-b", instance="sdar-a100-od")


class TestResolveAuditTargets:
    def test_defaults_produce_gpu_and_distinct_cpu_staging_target(self):
        targets = resolve_audit_targets()
        assert len(targets) == 2
        labels = {t.label for t in targets}
        assert labels == {"gpu", "cpu-staging"}

    def test_explicit_cpu_instance_env_is_honored(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_GCP_CPU_INSTANCE", "my-cpu-box")
        targets = resolve_audit_targets()
        cpu = next(t for t in targets if t.label == "cpu-staging")
        assert cpu.instance == "my-cpu-box"


class TestBuildDescribeArgv:
    def test_argv_is_read_only_describe(self):
        argv = build_describe_argv(_target())
        assert argv[:4] == ["gcloud", "compute", "instances", "describe"]
        assert "delete" not in argv
        assert "stop" not in argv
        assert "sdar-a100-od" in argv
        assert "--format=value(status)" in argv


class TestAudit:
    def test_running_instance_with_no_local_runs_is_a_loud_warning(self, tmp_path):
        def runner(argv):
            return VmExecResult(returncode=0, stdout="RUNNING\n")

        findings = audit(runs_root=tmp_path, runner=runner, targets=[_target()])

        assert len(findings) == 1
        assert findings[0].level == "warn"
        assert findings[0].status == "RUNNING"
        assert "orphaned" in findings[0].message

    def test_running_instance_with_active_local_run_is_still_warn_but_notes_context(self, tmp_path):
        run_dir = tmp_path / "prj_active"
        run_dir.mkdir()
        (run_dir / "demo_status.json").write_text(
            json.dumps({"status": "running"}), encoding="utf-8"
        )

        def runner(argv):
            return VmExecResult(returncode=0, stdout="RUNNING\n")

        findings = audit(runs_root=tmp_path, runner=runner, targets=[_target()])

        assert findings[0].active_local_runs == 1
        assert "1 local run" in findings[0].message

    def test_stopped_instance_is_informational_not_warn(self, tmp_path):
        def runner(argv):
            return VmExecResult(returncode=0, stdout="TERMINATED\n")

        findings = audit(runs_root=tmp_path, runner=runner, targets=[_target()])

        assert findings[0].level == "info"
        assert findings[0].status == "TERMINATED"

    def test_not_found_instance_is_informational(self, tmp_path):
        def runner(argv):
            return VmExecResult(returncode=1, stderr="ERROR: (gcloud.compute.instances.describe) Could not fetch resource: - The resource was not found")

        findings = audit(runs_root=tmp_path, runner=runner, targets=[_target()])

        assert findings[0].level == "info"
        assert findings[0].status == "NOT_FOUND"

    def test_unexpected_error_is_reported_not_swallowed(self, tmp_path):
        def runner(argv):
            return VmExecResult(returncode=1, stderr="ERROR: (gcloud) permission denied")

        findings = audit(runs_root=tmp_path, runner=runner, targets=[_target()])

        assert findings[0].level == "error"
        assert "permission denied" in findings[0].message

    def test_never_calls_a_mutating_verb(self, tmp_path):
        """Guard: this module must stay read-only. Any runner call whose argv
        contains a mutating verb fails the test immediately."""
        calls: list[list[str]] = []

        def runner(argv):
            calls.append(argv)
            return VmExecResult(returncode=0, stdout="STOPPED\n")

        audit(runs_root=tmp_path, runner=runner, targets=[_target()])

        for argv in calls:
            assert not ({"delete", "stop", "reset", "suspend"} & set(argv)), argv


class TestCountActiveLocalGcpRuns:
    def test_counts_running_and_queued_only(self, tmp_path):
        for pid, status in (("a", "running"), ("b", "queued"), ("c", "completed"), ("d", "failed")):
            d = tmp_path / pid
            d.mkdir()
            (d / "demo_status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

        assert count_active_local_gcp_runs(tmp_path) == 2

    def test_missing_runs_root_returns_zero(self, tmp_path):
        assert count_active_local_gcp_runs(tmp_path / "does-not-exist") == 0

    def test_corrupt_status_file_is_skipped_not_fatal(self, tmp_path):
        d = tmp_path / "prj_bad"
        d.mkdir()
        (d / "demo_status.json").write_text("not json", encoding="utf-8")
        assert count_active_local_gcp_runs(tmp_path) == 0
