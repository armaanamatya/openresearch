"""Cross-boundary regression test: env-var CONTRACT between k8s_job_cell_runner.py
(the runner) and aks_cell_entrypoint.py (the in-Job wrapper).

This test programmatically verifies that every env var the entrypoint READS is
present in the set the runner INJECTS — the seam that mocked tests couldn't
catch (P0-fix-1 introduced a silent total failure because the names diverged).

Design:
    1. Build a Job manifest via the runner's manifest builder.
    2. Extract the set of env-var NAMES it injects into the container spec.
    3. Parse aks_cell_entrypoint.py source for every ``os.environ[...]`` /
       ``os.environ.get(...)`` key it reads.
    4. Assert: every key the entrypoint reads must appear in the injected set.
       (The runner may inject extras; the entrypoint may read extras injected by
       other means — but no key the entrypoint NEEDS must be missing from the
       runner's injection.)

This test is pure (no azure package, no K8s, no subprocess) and runs in <1s.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # tests/agents/rlm → tests/agents → tests → project root

_RUNNER_MOD_PATH = (
    _PROJECT_ROOT / "backend" / "agents" / "rlm" / "k8s_job_cell_runner.py"
)

_ENTRYPOINT_PATH = (
    _PROJECT_ROOT / "docker" / "aks-cell-base" / "aks_cell_entrypoint.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_injected_env_names(
    *,
    storage_account: str = "myacct",
    blob_container: str = "myctr",
    code_blob_prefix: str = "runs/r1/code",
    output_blob_prefix: str = "runs/r1/cells",
    cache_mount_path: str = "/mnt/reprolab-cache",
) -> set[str]:
    """Build a manifest via _build_job_manifest and return the injected env-var names."""
    import backend.agents.rlm.k8s_job_cell_runner as kjcr

    manifest = kjcr._build_job_manifest(
        job_name="test-contract-job",
        namespace="reprolab",
        service_account="reprolab-sa",
        node_pool_name="gpunodes",
        base_image="test-registry.io/img:v1",
        storage_account=storage_account,
        blob_container=blob_container,
        files_share="reprolab-cache",
        cell_id="cell-001",
        cell_params_json='{"model": "qwen3-1.7b"}',
        output_blob_prefix=output_blob_prefix,
        code_blob_prefix=code_blob_prefix,
        active_deadline_seconds=3600,
        max_oom_retries=2,
        fingerprint="fp-abc",
        now_iso="2026-06-08T00:00:00Z",
        gpu_plan=None,
        cache_mount_path=cache_mount_path,
    )
    env_list: list[dict[str, Any]] = (
        manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    )
    return {e["name"] for e in env_list}


def _extract_entrypoint_read_env_names() -> set[str]:
    """Parse aks_cell_entrypoint.py with AST to find every env var name the
    entrypoint reads via os.environ[...] or os.environ.get(...).

    Returns a set of string literal keys only (dynamic lookups are ignored).
    """
    source = _ENTRYPOINT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    keys: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            """Catch os.environ["KEY"]."""
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "os"
                and node.value.attr == "environ"
            ):
                # Python 3.9+: node.slice is the key directly.
                key_node = node.slice
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    keys.add(key_node.value)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            """Catch os.environ.get("KEY", ...) and os.environ.get("KEY")."""
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
                and node.func.value.attr == "environ"
            ):
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        keys.add(node.args[0].value)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return keys


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAzureEnvContract:
    """Runner's injected env vars must be a superset of the entrypoint's read env vars."""

    def test_every_entrypoint_read_var_is_injected(self):
        """Core contract: entrypoint must not read any var the runner does not inject.

        This is the regression test for P0-fix-1, which found that the runner
        was injecting REPROLAB_BLOB_ACCOUNT / REPROLAB_BLOB_CONTAINER while the
        entrypoint was reading REPROLAB_AZURE_STORAGE_ACCOUNT / REPROLAB_AZURE_BLOB_CONTAINER —
        causing every Blob I/O call to fail silently.
        """
        injected = _extract_injected_env_names()
        entrypoint_reads = _extract_entrypoint_read_env_names()

        # Some env vars the entrypoint reads are set by OTHER means (e.g. by the
        # trainer subprocess itself, or by the REPROLAB_CELL_* vars the runner also
        # injects but might be set by the Job spec in different ways).  We focus
        # on the Blob / storage / cell config variables the runner is responsible for.
        # Exclude vars that are legitimately set by other agents (HF_HOME, etc.)
        # or are the OOM shrink vars set WITHIN the entrypoint itself (not by runner).
        runner_responsible_prefixes = (
            "REPROLAB_AZURE_",
            "REPROLAB_BLOB_",
            "REPROLAB_CELL_",
            "REPROLAB_CACHE_",
            "REPROLAB_BOOTSTRAP_",
        )
        entrypoint_runner_vars = {
            k for k in entrypoint_reads
            if any(k.startswith(pfx) for pfx in runner_responsible_prefixes)
        }

        missing = entrypoint_runner_vars - injected
        assert not missing, (
            f"P0-fix-1 / env-contract regression: the entrypoint reads these env vars "
            f"that the runner does NOT inject:\n"
            + "\n".join(f"  {k}" for k in sorted(missing))
            + "\n\nThis would cause silent Blob I/O failures in every pod."
        )

    def test_old_mismatched_names_not_injected(self):
        """The pre-fix names (REPROLAB_BLOB_ACCOUNT, REPROLAB_BLOB_CONTAINER) must
        NOT appear in the injected set — they were the wrong names."""
        injected = _extract_injected_env_names()
        assert "REPROLAB_BLOB_ACCOUNT" not in injected, (
            "REPROLAB_BLOB_ACCOUNT is the old (wrong) name; entrypoint reads "
            "REPROLAB_AZURE_STORAGE_ACCOUNT.  Remove the old name from the manifest."
        )
        assert "REPROLAB_BLOB_CONTAINER" not in injected, (
            "REPROLAB_BLOB_CONTAINER is the old (wrong) name; entrypoint reads "
            "REPROLAB_AZURE_BLOB_CONTAINER.  Remove the old name from the manifest."
        )

    def test_correct_azure_names_are_injected(self):
        """The correct REPROLAB_AZURE_* names must appear in the injected set."""
        injected = _extract_injected_env_names()
        assert "REPROLAB_AZURE_STORAGE_ACCOUNT" in injected, (
            "Runner must inject REPROLAB_AZURE_STORAGE_ACCOUNT "
            "(the name the entrypoint reads)."
        )
        assert "REPROLAB_AZURE_BLOB_CONTAINER" in injected, (
            "Runner must inject REPROLAB_AZURE_BLOB_CONTAINER "
            "(the name the entrypoint reads)."
        )

    def test_cache_mount_injected(self):
        """REPROLAB_CACHE_MOUNT must be injected (entrypoint uses it to set HF_HOME etc.)."""
        injected = _extract_injected_env_names()
        assert "REPROLAB_CACHE_MOUNT" in injected, (
            "Runner must inject REPROLAB_CACHE_MOUNT so the entrypoint can set "
            "HF_HOME / PIP_CACHE_DIR under the Azure Files mount."
        )

    def test_injected_names_are_all_strings(self):
        """All injected env var names must be plain strings (not dicts, None, etc.)."""
        injected = _extract_injected_env_names()
        for name in injected:
            assert isinstance(name, str) and name, f"Injected env name is not a non-empty string: {name!r}"

    def test_entrypoint_ast_parse_succeeds(self):
        """Sanity: the entrypoint module must be parseable by the AST extractor."""
        keys = _extract_entrypoint_read_env_names()
        # Must find at least the core storage vars.
        assert "REPROLAB_AZURE_STORAGE_ACCOUNT" in keys, (
            "AST extractor should find REPROLAB_AZURE_STORAGE_ACCOUNT in entrypoint"
        )
        assert "REPROLAB_AZURE_BLOB_CONTAINER" in keys, (
            "AST extractor should find REPROLAB_AZURE_BLOB_CONTAINER in entrypoint"
        )
        assert "REPROLAB_CELL_ID" in keys

    def test_contract_with_gpu_plan_bound(self):
        """Contract must hold even when a gpu_plan is provided."""
        import backend.agents.rlm.k8s_job_cell_runner as kjcr

        class _FakePlan:
            short_name = "azure_a100_80"
            gpu_count = 1

        manifest = kjcr._build_job_manifest(
            job_name="test-contract-gpu",
            namespace="reprolab",
            service_account="reprolab-sa",
            node_pool_name="gpunodes",
            base_image="test-registry.io/img:v1",
            storage_account="myacct",
            blob_container="myctr",
            files_share="reprolab-cache",
            cell_id="cell-002",
            cell_params_json="{}",
            output_blob_prefix="runs/r1/cells",
            code_blob_prefix="runs/r1/code",
            active_deadline_seconds=3600,
            max_oom_retries=2,
            fingerprint=None,
            now_iso=None,
            gpu_plan=_FakePlan(),
        )
        env_list = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        injected = {e["name"] for e in env_list}
        entrypoint_reads = _extract_entrypoint_read_env_names()

        runner_responsible_prefixes = (
            "REPROLAB_AZURE_",
            "REPROLAB_BLOB_",
            "REPROLAB_CELL_",
            "REPROLAB_CACHE_",
            "REPROLAB_BOOTSTRAP_",
        )
        entrypoint_runner_vars = {
            k for k in entrypoint_reads
            if any(k.startswith(pfx) for pfx in runner_responsible_prefixes)
        }
        missing = entrypoint_runner_vars - injected
        assert not missing, (
            f"Contract broken with gpu_plan bound — missing vars: {missing!r}"
        )
