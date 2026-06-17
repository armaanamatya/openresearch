"""Guard the GCP cell-scheduling SKU ↔ node-pool invariant.

Cell K8s Jobs are placed with ``nodeSelector {reprolab/sku: <plan.short_name>}``
(k8s_job_cell_runner). Terraform provisions one node pool labeled
``reprolab/sku=<short_name>`` per entry in the ``gpu_skus`` variable. The SKU
resolver may only pick a ``short_name`` that has a matching provisioned pool, so
the invariant that must hold is::

    config.gcp_gpu_skus  ==  the set of reprolab/sku labels tfvars provisions

If they drift (the original bug: config defaulted to ``gcp_a100_80`` while the
TF default pool was ``gcp_a100_80x8``), every cell resolves to a label that
exists on no node → Pending → capacity_exhausted.

These tests are hermetic — pure file reads + the Settings default, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.config import Settings
from backend.services.runtime.gpu_catalog import CATALOG

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VARIABLES_TF = _REPO_ROOT / "infra" / "gcp" / "variables.tf"

_EXPECTED_DEFAULT = ["gcp_a100_80x8"]


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings_cache(monkeypatch):
    # A session-leaking .env (via a pytest plugin's load_dotenv) or a stale
    # shell export could shadow the Field default — clear both so we assert the
    # true code default hermetically.
    monkeypatch.delenv("OPENRESEARCH_GCP_GPU_SKUS", raising=False)
    monkeypatch.delenv("REPROLAB_GCP_GPU_SKUS", raising=False)
    _reset_settings_cache()
    try:
        yield
    finally:
        _reset_settings_cache()


def _tf_gpu_skus_short_names() -> set[str]:
    """Regex out the short_name values from the gpu_skus variable's default block.

    Deterministic text scan (no HCL parser): isolate the ``default = [ ... ]``
    block inside ``variable "gpu_skus" { ... }``, then pull every
    ``short_name = "<value>"`` string literal from it. The ``type`` block uses
    ``short_name = string`` (no quotes) so it never matches the quoted-literal
    pattern.
    """
    text = _VARIABLES_TF.read_text(encoding="utf-8")

    var_match = re.search(
        r'variable\s+"gpu_skus"\s*\{(?P<body>.*?)\n\}',
        text,
        re.DOTALL,
    )
    assert var_match, f'no `variable "gpu_skus"` block found in {_VARIABLES_TF}'

    default_match = re.search(
        r"default\s*=\s*\[(?P<block>.*?)\]",
        var_match.group("body"),
        re.DOTALL,
    )
    assert default_match, (
        f'no `default = [...]` block in the gpu_skus variable of {_VARIABLES_TF}'
    )

    return set(
        re.findall(
            r'short_name\s*=\s*"([^"]+)"',
            default_match.group("block"),
        )
    )


def test_config_default_is_the_x8_pool():
    """Settings().gcp_gpu_skus defaults to the TF default pool label."""
    s = Settings(_env_file=None)
    assert s.gcp_gpu_skus == _EXPECTED_DEFAULT, (
        "config.gcp_gpu_skus default must match the Terraform default pool "
        f"label; got {s.gcp_gpu_skus!r}, expected {_EXPECTED_DEFAULT!r}"
    )


def test_config_default_skus_are_valid_catalog_short_names():
    """Every default SKU is a real GCP catalog short_name (no typo'd label)."""
    gcp_short_names = {
        sku.short_name for sku in CATALOG if sku.short_name.startswith("gcp_")
    }
    s = Settings(_env_file=None)
    unknown = set(s.gcp_gpu_skus) - gcp_short_names
    assert not unknown, (
        f"config.gcp_gpu_skus contains short_names absent from the GCP catalog: "
        f"{sorted(unknown)} (known: {sorted(gcp_short_names)})"
    )


def test_config_default_equals_tf_provisioned_pools():
    """INVARIANT: config.gcp_gpu_skus == the reprolab/sku labels TF provisions.

    The SKU resolver only selects from config.gcp_gpu_skus and cells are
    placed by nodeSelector {reprolab/sku: <short_name>}; a config SKU with no
    matching provisioned pool (or a provisioned pool the resolver never picks)
    is a scheduling dead-end. This guard catches future drift between the
    Terraform `gpu_skus` default and the resolver menu.
    """
    tf_short_names = _tf_gpu_skus_short_names()
    s = Settings(_env_file=None)
    config_short_names = set(s.gcp_gpu_skus)
    assert config_short_names == tf_short_names, (
        "SKU↔pool drift: config.gcp_gpu_skus must equal the reprolab/sku labels "
        f"infra/gcp/variables.tf `gpu_skus` default provisions. "
        f"config={sorted(config_short_names)} vs tf={sorted(tf_short_names)}. "
        "Cells schedule via nodeSelector {reprolab/sku: <short_name>}, so a "
        "mismatch leaves every cell Pending → capacity_exhausted."
    )
