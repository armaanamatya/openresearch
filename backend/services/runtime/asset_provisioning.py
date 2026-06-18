"""Idempotent asset pre-provisioning for heavy ML papers.

Ensures pip stacks, HuggingFace model weights, datasets, and the WebShop
package are installed/warm in a SHARED persistent cache before any GPU work
starts. Called by the harness (run.py) immediately before env provisioning on
``local`` sandbox runs, and by ``scripts/sdar_gcp_assets.py`` as its
implementation backend.

All heavy dependencies (torch, transformers, huggingface_hub, datasets) are
lazy-imported inside the functions that need them, so

    import backend.services.runtime.asset_provisioning

works without any ML libraries installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agents.schemas import AssetSpec


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class AssetProvisionError(Exception):
    """Raised when a REQUIRED asset cannot be ensured."""


@dataclass
class AssetReport:
    """Summary of one ensure_assets call."""

    ensured: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers (shared with sdar_gcp_assets.py callers)
# ---------------------------------------------------------------------------

def _module_exists(name: str) -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec(name) is not None


def _console_script_exists(name: str) -> bool:
    return Path(sys.executable).with_name(name).exists() or shutil.which(name) is not None


def _site_package_dirs() -> list[Path]:
    dirs: list[Path] = []
    for raw in site.getsitepackages() + [site.getusersitepackages()]:
        if raw:
            dirs.append(Path(raw))
    return dirs


def _add_repo_pth(name: str, repo_dir: Path) -> None:
    for sp in _site_package_dirs():
        if sp.exists() and os.access(sp, os.W_OK):
            (sp / f"{name}.pth").write_text(str(repo_dir.resolve()) + "\n", encoding="utf-8")
            return
    raise RuntimeError("no writable site-packages directory found for WebShop .pth")


def install_webshop(pip_cache_dir: Path, cache_root: Path) -> None:
    """Ensure web_agent_site is importable; clone + install if not.

    Mirrors the logic in ``scripts/sdar_gcp_assets.py::install_webshop``.
    Raises ``AssetProvisionError`` if the package is still missing after install.
    """
    if _module_exists("web_agent_site"):
        return

    repo_dir = cache_root / "webshop" / "WebShop"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_url = os.environ.get(
        "OPENRESEARCH_WEBSHOP_REPO_URL",
        "https://github.com/princeton-nlp/WebShop.git",
    )
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
        )
    req = repo_dir / "requirements.txt"
    if req.exists():
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--cache-dir", str(pip_cache_dir),
                "-r", str(req),
            ],
            check=True,
        )
    pyproject = repo_dir / "pyproject.toml"
    setup_py = repo_dir / "setup.py"
    if pyproject.exists() or setup_py.exists():
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--cache-dir", str(pip_cache_dir),
                "-e", str(repo_dir),
            ],
            check=True,
        )
    else:
        _add_repo_pth("openresearch_webshop", repo_dir)

    if not _module_exists("web_agent_site"):
        raise AssetProvisionError(
            "WebShop install completed but web_agent_site is still not importable; "
            "inspect the upstream repo layout under " + str(repo_dir)
        )


def warm_hf_models(models: list[str]) -> None:
    """Download model weights into HF_HOME (idempotent; HF skips cached files)."""
    from huggingface_hub import snapshot_download  # lazy

    allow = [
        "*.json",
        "*.model",
        "*.txt",
        "*.safetensors",
        "tokenizer*",
        "merges.txt",
        "vocab*",
    ]
    for model_id in models:
        snapshot_download(repo_id=model_id, allow_patterns=allow)


def warm_datasets(datasets: list[str], rows: int = 64) -> list[str]:
    """Touch a bounded prefix of each dataset to force cache materialization.

    Returns the list of dataset names that FAILED (for the caller to record).
    Best-effort — never raises.
    """
    from datasets import load_dataset  # lazy

    failed: list[str] = []
    for name in datasets:
        try:
            if name == "hotpot_qa":
                ds = load_dataset("hotpot_qa", "distractor", split="validation")
            elif name == "nq_open":
                ds = load_dataset("nq_open", split="validation")
            else:
                ds = load_dataset(name, split="validation")
            for idx, _row in enumerate(ds):
                if idx + 1 >= rows:
                    break
        except Exception:  # noqa: BLE001
            failed.append(name)
    return failed


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def ensure_assets(
    spec: "AssetSpec",
    *,
    cache_root: Path,
    prepare: bool = True,
) -> AssetReport:
    """Ensure all assets declared in *spec* are warm under *cache_root*.

    Sets ``HF_HOME``, ``PIP_CACHE_DIR``, and ``OPENRESEARCH_ENV_CACHE_DIR``
    in ``os.environ`` so every subprocess and library call sees the shared
    cache (this is safe because the harness is the only code running at the
    point this is called).

    When ``prepare=False`` the cache env vars are still set but no installs
    or downloads are performed (useful for testing the gating predicate).

    Required assets (requirements_files, webshop, models) raise
    ``AssetProvisionError`` on failure.  Datasets are best-effort and only
    recorded in ``report.failed``.
    """
    pip_cache = cache_root / "pip"
    hf_home = cache_root / "hf"
    env_cache = cache_root / "envs"

    # Set the shared cache dirs into the environment so subprocess pip calls
    # and HuggingFace library calls all land in the same place.
    pip_cache.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    env_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["PIP_CACHE_DIR"] = str(pip_cache)
    os.environ["OPENRESEARCH_ENV_CACHE_DIR"] = str(env_cache)

    report = AssetReport()

    if not prepare:
        return report

    repo_root = _repo_root()

    # (a) Requirements files — REQUIRED
    for rel_path in spec.requirements_files:
        req_file = repo_root / rel_path
        label = f"requirements:{rel_path}"
        if not req_file.exists():
            raise AssetProvisionError(
                f"requirements file not found: {req_file}"
            )
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--cache-dir", str(pip_cache),
                "-r", str(req_file),
            ],
            capture_output=False,
        )
        if result.returncode != 0:
            raise AssetProvisionError(
                f"pip install failed for {rel_path} (exit {result.returncode})"
            )
        report.ensured.append(label)

    # (b) WebShop — REQUIRED when requested
    if spec.webshop:
        label = "webshop:web_agent_site"
        if _module_exists("web_agent_site"):
            report.skipped.append(label)
        else:
            install_webshop(pip_cache, cache_root)  # raises AssetProvisionError on failure
            report.ensured.append(label)

    # (c) HF model weights — REQUIRED
    for model_id in spec.models:
        label = f"model:{model_id}"
        try:
            warm_hf_models([model_id])
            report.ensured.append(label)
        except Exception as exc:  # noqa: BLE001
            raise AssetProvisionError(
                f"could not download model weights for {model_id}: {exc}"
            ) from exc

    # (d) Datasets — BEST-EFFORT
    if spec.datasets:
        failed_ds = warm_datasets(spec.datasets)
        for name in spec.datasets:
            label = f"dataset:{name}"
            if name in failed_ds:
                report.failed.append(label)
            else:
                report.ensured.append(label)

    return report


def check_assets(
    spec: "AssetSpec",
    cache_root: Path,
) -> list[tuple[str, bool, str]]:
    """Check importability / presence of assets declared in *spec*.

    Returns a list of (name, ok, detail) tuples mirroring the Check list in
    ``scripts/sdar_gcp_assets.py::run_checks``.
    """
    results: list[tuple[str, bool, str]] = []

    # Standard ML stack modules
    for mod in (
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface_hub",
        "sentence_transformers",
        "faiss",
        "rank_bm25",
        "alfworld",
    ):
        results.append((f"import {mod}", _module_exists(mod), ""))

    results.append((
        "console alfworld-download",
        _console_script_exists("alfworld-download"),
        "",
    ))
    results.append((
        "import web_agent_site",
        _module_exists("web_agent_site"),
        "",
    ))

    # GPU-free config resolve: catches too-old transformers (KeyError: 'qwen3')
    # and typo'd HF repo ids before any GPU hours are leased.
    for model_id in getattr(spec, "models", []) or []:
        label = f"resolve {model_id}"
        try:
            from transformers import AutoConfig  # lazy  # noqa: PLC0415
            cfg = AutoConfig.from_pretrained(model_id)
            results.append((label, True, str(getattr(cfg, "model_type", "") or "")))
        except Exception as exc:  # noqa: BLE001
            results.append((label, False, type(exc).__name__ + ": " + str(exc)[:120]))

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (parent of ``backend/``)."""
    return Path(__file__).resolve().parents[3]
