"""Unit tests for backend.services.runtime.asset_provisioning.

Tests idempotency, AssetSpec declaration, error propagation, and the run.py
hook gating predicate — all without any network calls, GPU, or ML library
dependencies.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.schemas import AssetSpec
from backend.services.runtime.asset_provisioning import (
    AssetProvisionError,
    check_assets,
    ensure_assets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(**kwargs) -> AssetSpec:
    defaults = dict(requirements_files=[], models=[], datasets=[], webshop=False)
    defaults.update(kwargs)
    return AssetSpec(**defaults)


# ---------------------------------------------------------------------------
# Idempotency: cache env vars are always set
# ---------------------------------------------------------------------------

def test_ensure_assets_sets_cache_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """ensure_assets always injects HF_HOME / PIP_CACHE_DIR / ENV_CACHE_DIR."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
    monkeypatch.delenv("OPENRESEARCH_ENV_CACHE_DIR", raising=False)

    spec = _make_spec()
    report = ensure_assets(spec, cache_root=tmp_path, prepare=False)

    import os
    assert os.environ["HF_HOME"] == str(tmp_path / "hf")
    assert os.environ["PIP_CACHE_DIR"] == str(tmp_path / "pip")
    assert os.environ["OPENRESEARCH_ENV_CACHE_DIR"] == str(tmp_path / "envs")
    assert report.ensured == []
    assert report.skipped == []
    assert report.failed == []


# ---------------------------------------------------------------------------
# Idempotency: a warm asset is skipped, not re-downloaded
# ---------------------------------------------------------------------------

def test_ensure_assets_skips_warm_webshop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When web_agent_site is already importable, webshop is skipped (not reinstalled)."""
    spec = _make_spec(webshop=True)

    with patch(
        "backend.services.runtime.asset_provisioning._module_exists",
        return_value=True,
    ):
        report = ensure_assets(spec, cache_root=tmp_path)

    assert "webshop:web_agent_site" in report.skipped
    assert report.ensured == []
    assert report.failed == []


def test_ensure_assets_skips_pip_when_already_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """pip install is idempotent: it runs regardless (pip itself skips cached),
    but the spec records it as 'ensured' (not skipped).  Verify subprocess call."""
    req = tmp_path / "requirements-test.txt"
    req.write_text("# empty\n", encoding="utf-8")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch(
            "backend.services.runtime.asset_provisioning._repo_root",
            return_value=tmp_path,
        ):
            spec = _make_spec(requirements_files=["requirements-test.txt"])
            report = ensure_assets(spec, cache_root=tmp_path)

    assert mock_run.called
    assert "requirements:requirements-test.txt" in report.ensured


# ---------------------------------------------------------------------------
# Required-asset failures raise AssetProvisionError
# ---------------------------------------------------------------------------

def test_ensure_assets_raises_on_pip_failure(tmp_path: Path):
    """A non-zero pip exit code raises AssetProvisionError."""
    req = tmp_path / "bad_req.txt"
    req.write_text("nonexistent-package-xyz==99.0\n", encoding="utf-8")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        with patch(
            "backend.services.runtime.asset_provisioning._repo_root",
            return_value=tmp_path,
        ):
            spec = _make_spec(requirements_files=["bad_req.txt"])
            with pytest.raises(AssetProvisionError, match="pip install failed"):
                ensure_assets(spec, cache_root=tmp_path)


def test_ensure_assets_raises_when_req_file_missing(tmp_path: Path):
    """A requirements file that doesn't exist raises AssetProvisionError."""
    with patch(
        "backend.services.runtime.asset_provisioning._repo_root",
        return_value=tmp_path,
    ):
        spec = _make_spec(requirements_files=["does-not-exist.txt"])
        with pytest.raises(AssetProvisionError, match="requirements file not found"):
            ensure_assets(spec, cache_root=tmp_path)


def test_ensure_assets_raises_on_model_download_failure(tmp_path: Path):
    """A failing snapshot_download raises AssetProvisionError."""
    spec = _make_spec(models=["Org/nonexistent-model"])

    with patch(
        "backend.services.runtime.asset_provisioning.warm_hf_models",
        side_effect=RuntimeError("404 Not Found"),
    ):
        with pytest.raises(AssetProvisionError, match="could not download model weights"):
            ensure_assets(spec, cache_root=tmp_path)


# ---------------------------------------------------------------------------
# Dataset failures are best-effort: recorded in report.failed, not raised
# ---------------------------------------------------------------------------

def test_ensure_assets_dataset_failure_is_non_fatal(tmp_path: Path):
    """A dataset load failure records in report.failed and does not raise."""
    spec = _make_spec(datasets=["nq_open", "hotpot_qa"])

    with patch(
        "backend.services.runtime.asset_provisioning.warm_datasets",
        return_value=["hotpot_qa"],  # hotpot_qa failed, nq_open succeeded
    ):
        report = ensure_assets(spec, cache_root=tmp_path)

    assert "dataset:hotpot_qa" in report.failed
    assert "dataset:nq_open" in report.ensured
    # No AssetProvisionError raised


def test_ensure_assets_all_datasets_fail_is_non_fatal(tmp_path: Path):
    """All dataset failures are non-fatal."""
    spec = _make_spec(datasets=["ds_a", "ds_b"])

    with patch(
        "backend.services.runtime.asset_provisioning.warm_datasets",
        return_value=["ds_a", "ds_b"],
    ):
        report = ensure_assets(spec, cache_root=tmp_path)

    assert set(report.failed) == {"dataset:ds_a", "dataset:ds_b"}
    assert report.ensured == []


# ---------------------------------------------------------------------------
# Declaration: SDAR paper hint carries the expected AssetSpec
# ---------------------------------------------------------------------------

def test_sdar_paper_hint_has_assets():
    """PAPER_HINTS['2605.15155'] carries the SDAR AssetSpec."""
    from backend.agents.prompts.paper_hints import PAPER_HINTS

    hint = PAPER_HINTS["2605.15155"]
    assert hint.assets is not None
    assert isinstance(hint.assets, AssetSpec)
    assert "backend/requirements-sdar.txt" in hint.assets.requirements_files
    assert "Qwen/Qwen3-1.7B" in hint.assets.models
    assert "Qwen/Qwen2.5-3B-Instruct" in hint.assets.models
    assert "Qwen/Qwen2.5-7B-Instruct" in hint.assets.models
    assert "nq_open" in hint.assets.datasets
    assert "hotpot_qa" in hint.assets.datasets
    assert hint.assets.webshop is True


def test_non_sdar_paper_hint_has_no_assets():
    """A paper hint without an assets declaration has assets=None."""
    from backend.agents.prompts.paper_hints import PAPER_HINTS

    # Adam paper — no assets declared
    hint = PAPER_HINTS["1412.6980"]
    assert hint.assets is None


def test_missing_paper_hint_returns_none():
    """lookup_paper_hint for an unknown paper returns None."""
    from backend.agents.prompts.paper_hints import lookup_paper_hint

    assert lookup_paper_hint("9999.99999") is None


# ---------------------------------------------------------------------------
# Hook gating predicate contract (tested in isolation)
# ---------------------------------------------------------------------------

def test_hook_gating_no_op_when_assets_is_none():
    """The hook predicate skips ensure_assets when hint.assets is None."""
    hint = MagicMock()
    hint.assets = None

    # Simulate the gating logic from run.py:
    #   if _preload_hint is not None and getattr(_preload_hint, "assets", None) is not None:
    should_run = hint is not None and getattr(hint, "assets", None) is not None
    assert should_run is False


def test_hook_gating_no_op_when_hint_is_none():
    """The hook predicate skips ensure_assets when hint is None."""
    hint = None
    should_run = hint is not None and getattr(hint, "assets", None) is not None
    assert should_run is False


def test_hook_gating_runs_when_assets_present():
    """The hook predicate fires when hint.assets is an AssetSpec."""
    hint = MagicMock()
    hint.assets = _make_spec(webshop=True)

    should_run = hint is not None and getattr(hint, "assets", None) is not None
    assert should_run is True


def test_hook_gating_flag_zero_disables(monkeypatch: pytest.MonkeyPatch):
    """OPENRESEARCH_PRELOAD_ASSETS=0 makes the gating predicate False."""
    import os
    monkeypatch.setenv("OPENRESEARCH_PRELOAD_ASSETS", "0")

    # Simulate the full gate from run.py:
    #   os.environ.get("OPENRESEARCH_PRELOAD_ASSETS", "1") != "0"
    flag_enabled = os.environ.get("OPENRESEARCH_PRELOAD_ASSETS", "1") != "0"
    assert flag_enabled is False


# ---------------------------------------------------------------------------
# AssetSpec model validation
# ---------------------------------------------------------------------------

def test_asset_spec_defaults():
    """AssetSpec with no args has sensible empty defaults."""
    spec = AssetSpec()
    assert spec.requirements_files == []
    assert spec.models == []
    assert spec.datasets == []
    assert spec.webshop is False


def test_asset_spec_full():
    """AssetSpec can be constructed with all fields."""
    spec = AssetSpec(
        requirements_files=["backend/requirements-sdar.txt"],
        models=["Qwen/Qwen3-1.7B"],
        datasets=["nq_open"],
        webshop=True,
    )
    assert spec.requirements_files == ["backend/requirements-sdar.txt"]
    assert spec.models == ["Qwen/Qwen3-1.7B"]
    assert spec.datasets == ["nq_open"]
    assert spec.webshop is True


# ---------------------------------------------------------------------------
# check_assets: GPU-free config resolve
# ---------------------------------------------------------------------------

def _inject_fake_transformers(from_pretrained_side_effect=None):
    """Return a context manager that injects a fake transformers module.

    This is needed because transformers is not installed in the dev venv, but
    check_assets does a lazy ``from transformers import AutoConfig`` inside the
    loop.  We inject a fake module into sys.modules so that import resolves
    without hitting the network.
    """
    fake_auto_config = MagicMock()
    if from_pretrained_side_effect is not None:
        fake_auto_config.from_pretrained.side_effect = from_pretrained_side_effect
    else:
        mock_cfg = MagicMock()
        mock_cfg.model_type = "qwen3"
        fake_auto_config.from_pretrained.return_value = mock_cfg

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = fake_auto_config  # type: ignore[attr-defined]
    # importlib.util.find_spec() raises ValueError when __spec__ is None on a
    # module that is already in sys.modules; give it a minimal spec so the
    # _module_exists("transformers") check inside check_assets succeeds.
    fake_transformers.__spec__ = importlib.util.spec_from_loader("transformers", loader=None)  # type: ignore[attr-defined]

    return patch.dict(sys.modules, {"transformers": fake_transformers})


def test_check_assets_resolve_success(tmp_path: Path):
    """check_assets emits a (resolve …, True, model_type) tuple on success."""
    spec = AssetSpec(models=["Qwen/Qwen3-1.7B"])

    with _inject_fake_transformers():
        results = check_assets(spec, tmp_path)

    assert ("resolve Qwen/Qwen3-1.7B", True, "qwen3") in results


def test_check_assets_resolve_failure(tmp_path: Path):
    """check_assets emits a (resolve …, False, …) tuple when from_pretrained raises."""
    spec = AssetSpec(models=["Qwen/Qwen3-1.7B"])

    with _inject_fake_transformers(from_pretrained_side_effect=KeyError("qwen3")):
        results = check_assets(spec, tmp_path)

    names_to_result = {name: (ok, detail) for name, ok, detail in results}
    assert "resolve Qwen/Qwen3-1.7B" in names_to_result
    ok, detail = names_to_result["resolve Qwen/Qwen3-1.7B"]
    assert ok is False
    assert "KeyError" in detail


def test_check_assets_no_models_no_resolve_entries(tmp_path: Path):
    """check_assets with an empty models list produces no 'resolve …' entries."""
    spec = AssetSpec(models=[])

    # No fake transformers needed — the loop body is never entered.
    results = check_assets(spec, tmp_path)

    resolve_entries = [name for name, _ok, _detail in results if name.startswith("resolve ")]
    assert resolve_entries == []
