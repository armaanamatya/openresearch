"""Static contracts shared by the GKE and AKS durable-controller charts."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cloud", ["gcp", "azure"])
def test_foundry_secret_is_opt_in_and_projected_by_both_charts(cloud: str) -> None:
    chart = REPO_ROOT / "infra" / cloud / "helm"
    values = (chart / "values.yaml").read_text(encoding="utf-8")
    secret_class = (
        chart / "templates" / "orchestrator-secretproviderclass.yaml"
    ).read_text(encoding="utf-8")

    assert "azureFoundry:" in values
    assert "enabled: false" in values
    assert ".Values.orchestrator.azureFoundry.enabled" in secret_class
    assert "azure-foundry-api-key" in secret_class
    assert "AZURE_FOUNDRY_API_KEY" in secret_class


def test_gcp_foundry_secret_has_resource_output_wiring_and_iam() -> None:
    gcp = REPO_ROOT / "infra" / "gcp"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            gcp / "modules" / "secret_manager" / "main.tf",
            gcp / "modules" / "secret_manager" / "outputs.tf",
            gcp / "modules" / "identity" / "variables.tf",
            gcp / "modules" / "identity" / "main.tf",
            gcp / "main.tf",
        )
    )

    assert 'secret_id = "azure-foundry-api-key"' in combined
    assert "azure_foundry_api_key_secret_id" in combined
    assert "orchestrator_azure_foundry_key" in combined


def test_aks_orchestrator_can_read_shared_cache_pvc() -> None:
    role = (
        REPO_ROOT / "infra" / "azure" / "helm" / "templates" / "role.yaml"
    ).read_text(encoding="utf-8")

    assert 'resources: ["persistentvolumeclaims"]' in role
    assert 'verbs: ["get", "list"]' in role
