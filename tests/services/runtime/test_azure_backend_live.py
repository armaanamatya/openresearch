"""End-to-end smoke against real Azure.

Marked ``azure_live`` so it does NOT run in the default suite.  To run:

    REPROLAB_AZURE_LIVE_TESTS=1 \\
    REPROLAB_AZURE_SUBSCRIPTION_ID=... \\
    REPROLAB_AZURE_SSH_KEY_PATH=~/.ssh/azure_ed25519 \\
    pytest tests/services/runtime/test_azure_backend_live.py -m azure_live -v

Costs <$0.01 — provisions Standard_B1s (CPU), runs `echo hello`, destroys.
Total wall clock ~2-3 min including VM boot.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path

import pytest

from backend.services.runtime.azure_backend import AzureBackend
from backend.services.runtime.interface import SandboxConfig


pytestmark = [
    pytest.mark.azure_live,
    pytest.mark.skipif(
        not os.environ.get("REPROLAB_AZURE_LIVE_TESTS"),
        reason="REPROLAB_AZURE_LIVE_TESTS not set — opt-in, costs money",
    ),
]


@pytest.mark.asyncio
async def test_end_to_end_smoke():
    tmp = Path(tempfile.mkdtemp(prefix="azure-live-"))
    (tmp / "hello.txt").write_text("hi\n")
    backend = AzureBackend(
        subscription_id=os.environ["REPROLAB_AZURE_SUBSCRIPTION_ID"],
        region=os.environ.get("REPROLAB_AZURE_REGION", "eastus"),
        vm_size="Standard_B1s",  # cheapest CPU SKU, no quota req
        image="Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest",
        ssh_key_path=os.environ.get("REPROLAB_AZURE_SSH_KEY_PATH") or None,
        ssh_public_key=os.environ.get("REPROLAB_AZURE_SSH_PUBLIC_KEY", ""),
        max_boot_seconds=300,
    )
    config = SandboxConfig(
        project_id="live-smoke",
        run_id=uuid.uuid4().hex[:8],
        project_root=tmp,
    )
    sandbox = await backend.create_sandbox(config)
    try:
        result = await backend.exec(sandbox, "echo hello-from-azure", timeout=30)
        assert result.exit_code == 0, result.stderr
        assert "hello-from-azure" in result.stdout
    finally:
        await backend.destroy(sandbox)
