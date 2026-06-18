#!/usr/bin/env python3
"""Prepare and validate SDAR assets on a GCP A100 VM.

The expensive reproduction must not discover missing datasets, model weights, or
environment packages after GPUs are leased. This script makes the SDAR asset
contract explicit and idempotent:

* install the SDAR runtime stack;
* warm HuggingFace model and dataset caches;
* provision ALFWorld/WebShop/Search-QA through EnvCacheManager;
* write an env file that the launch command can source.
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib
import os
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO / "runs" / ".cache" / "envs"
DEFAULT_HF_HOME = REPO / "runs" / ".cache" / "hf"
DEFAULT_PIP_CACHE = REPO / "runs" / ".cache" / "pip"
DEFAULT_ENV_FILE = REPO / "runs" / ".cache" / "sdar_gcp.env"
SDAR_REQUIREMENTS = REPO / "backend" / "requirements-sdar.txt"

DEFAULT_MODELS = (
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)
DEFAULT_DATASETS = ("nq_open", "hotpot_qa")
DEFAULT_ENVS = ("Search-QA", "ALFWorld", "WebShop")
WEBSHOP_REPO_URL = "https://github.com/princeton-nlp/WebShop.git"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(raw: str | None, default: Iterable[str]) -> list[str]:
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or REPO), env=env)


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


def install_requirements(args: argparse.Namespace, env: dict[str, str]) -> None:
    _run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--cache-dir",
        str(args.pip_cache_dir),
        "-r",
        str(SDAR_REQUIREMENTS),
    ], env=env)


def install_webshop(args: argparse.Namespace, env: dict[str, str]) -> None:
    if _module_exists("web_agent_site"):
        print("[OK] WebShop module already importable")
        return
    repo_dir = args.cache_dir / "webshop" / "WebShop"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_url = os.environ.get("OPENRESEARCH_WEBSHOP_REPO_URL", WEBSHOP_REPO_URL)
    if not repo_dir.exists():
        _run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], env=env)
    req = repo_dir / "requirements.txt"
    if req.exists():
        _run([sys.executable, "-m", "pip", "install", "--cache-dir", str(args.pip_cache_dir), "-r", str(req)], env=env)
    pyproject = repo_dir / "pyproject.toml"
    setup_py = repo_dir / "setup.py"
    if pyproject.exists() or setup_py.exists():
        _run([sys.executable, "-m", "pip", "install", "--cache-dir", str(args.pip_cache_dir), "-e", str(repo_dir)], env=env)
    else:
        _add_repo_pth("openresearch_webshop", repo_dir)
    if not _module_exists("web_agent_site"):
        raise RuntimeError(
            "WebShop install completed but web_agent_site is still not importable; "
            "inspect the upstream repo layout under " + str(repo_dir)
        )


def warm_hf_models(models: list[str]) -> None:
    from huggingface_hub import snapshot_download

    allow = [
        "*.json",
        "*.model",
        "*.txt",
        "*.safetensors",
        "tokenizer*",
        "merges.txt",
        "vocab*",
    ]
    for model in models:
        print(f"[prepare] warming HF model {model}", flush=True)
        snapshot_download(repo_id=model, allow_patterns=allow)


def warm_datasets(datasets: list[str], rows: int) -> None:
    from datasets import load_dataset

    for name in datasets:
        print(f"[prepare] warming dataset {name}", flush=True)
        if name == "hotpot_qa":
            ds = load_dataset("hotpot_qa", "distractor", split="validation")
        elif name == "nq_open":
            ds = load_dataset("nq_open", split="validation")
        else:
            ds = load_dataset(name, split="validation")
        # Touch a bounded prefix to force cache materialization without scanning
        # a full corpus during preflight.
        for idx, _row in enumerate(ds):
            if idx + 1 >= rows:
                break


def provision_envs(env_names: list[str]) -> dict[str, str]:
    from backend.services.runtime.env_cache import EnvCacheManager, provision_scope

    result = provision_scope(env_names, EnvCacheManager())
    try:
        if result.exclusions:
            details = ", ".join(f"{e.item}: {e.reason}" for e in result.exclusions)
            raise RuntimeError(f"environment provisioning exclusions: {details}")
        return dict(result.env_vars)
    finally:
        result.release()


def write_env_file(args: argparse.Namespace, values: dict[str, str]) -> None:
    path = args.env_file
    path.parent.mkdir(parents=True, exist_ok=True)
    # WebShop is ref-counted and intentionally released after prepare. Keep the
    # cache knobs, but let the real run acquire a fresh live lease.
    stable_values = {k: v for k, v in values.items() if k != "WEBSHOP_URL"}
    merged = {
        "OPENRESEARCH_ENV_CACHE_DIR": str(args.cache_dir.resolve()),
        "HF_HOME": str(args.hf_home.resolve()),
        "PIP_CACHE_DIR": str(args.pip_cache_dir.resolve()),
        "OPENRESEARCH_PROVISION_ENVS": ",".join(DEFAULT_ENVS),
        "OPENRESEARCH_FORCE_SANDBOX": "local",
        "OPENRESEARCH_DEFAULT_SANDBOX": "local",
        **stable_values,
    }
    text = "\n".join(f'export {key}="{value}"' for key, value in sorted(merged.items()))
    path.write_text(text + "\n", encoding="utf-8")
    print(f"[OK] wrote {path}")


def run_checks(args: argparse.Namespace, env_vars: dict[str, str] | None = None) -> list[Check]:
    checks = [
        Check("python", sys.version_info >= (3, 11), sys.version.split()[0]),
        Check("requirements file", SDAR_REQUIREMENTS.exists(), str(SDAR_REQUIREMENTS)),
    ]
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
        checks.append(Check(f"import {mod}", _module_exists(mod)))
    checks.append(Check("console alfworld-download", _console_script_exists("alfworld-download")))
    checks.append(Check("import web_agent_site", _module_exists("web_agent_site"), required=not args.allow_missing_webshop))

    alf_raw = (env_vars or {}).get("ALFWORLD_DATA") or os.environ.get("ALFWORLD_DATA", "")
    alf_dir = Path(alf_raw) if alf_raw else Path()
    checks.append(Check(
        "ALFWorld data",
        bool(alf_raw) and (alf_dir / "json_2.1.1").exists(),
        str(alf_dir) if alf_raw else "ALFWORLD_DATA unset",
    ))
    if args.require_gpu:
        try:
            import torch

            count = torch.cuda.device_count()
            checks.append(Check("CUDA GPUs", count >= args.min_gpus, f"visible={count}, required={args.min_gpus}"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("CUDA GPUs", False, repr(exc)))
    return checks


def print_checks(checks: list[Check]) -> int:
    failures = 0
    for chk in checks:
        label = "OK" if chk.ok else ("WARN" if not chk.required else "FAIL")
        print(f"[{label}] {chk.name}" + (f" - {chk.detail}" if chk.detail else ""))
        if not chk.ok and chk.required:
            failures += 1
    return failures


def build_env(args: argparse.Namespace) -> dict[str, str]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.hf_home.mkdir(parents=True, exist_ok=True)
    args.pip_cache_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OPENRESEARCH_ENV_CACHE_DIR": str(args.cache_dir.resolve()),
        "HF_HOME": str(args.hf_home.resolve()),
        "PIP_CACHE_DIR": str(args.pip_cache_dir.resolve()),
    }
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Install/warm assets before checking.")
    parser.add_argument("--check", action="store_true", help="Run checks. Default when --prepare is not passed.")
    parser.add_argument("--require-gpu", action="store_true", help="Require CUDA GPUs to be visible.")
    parser.add_argument("--min-gpus", type=int, default=8)
    parser.add_argument("--allow-missing-webshop", action="store_true", help="Warn instead of fail if WebShop is unavailable.")
    parser.add_argument("--skip-models", action="store_true", help="Do not download HF model snapshots.")
    parser.add_argument("--skip-datasets", action="store_true", help="Do not warm HF datasets.")
    parser.add_argument("--dataset-rows", type=int, default=64)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--pip-cache-dir", type=Path, default=DEFAULT_PIP_CACHE)
    args = parser.parse_args()

    if not args.prepare and not args.check:
        args.check = True

    env = build_env(args)
    os.environ.update(env)
    env_vars: dict[str, str] = {}

    if args.prepare:
        install_requirements(args, env)
        install_webshop(args, env)
        if not args.skip_models:
            warm_hf_models(_split_csv(os.environ.get("OPENRESEARCH_SDAR_HF_MODELS"), DEFAULT_MODELS))
        if not args.skip_datasets:
            warm_datasets(_split_csv(os.environ.get("OPENRESEARCH_SDAR_DATASETS"), DEFAULT_DATASETS), args.dataset_rows)
        env_vars = provision_envs(_split_csv(os.environ.get("OPENRESEARCH_SDAR_ENVS"), DEFAULT_ENVS))
        write_env_file(args, env_vars)

    if args.check or args.prepare:
        failures = print_checks(run_checks(args, env_vars))
        if failures:
            print(f"[RED] {failures} required preflight check(s) failed.")
            return 1
        print("[GREEN] SDAR/GCP assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
