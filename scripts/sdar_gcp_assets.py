#!/usr/bin/env python3
"""Prepare and validate SDAR assets on a GCP A100 VM.

The expensive reproduction must not discover missing datasets, model weights, or
environment packages after GPUs are leased. This script makes the SDAR asset
contract explicit and idempotent:

* install the SDAR runtime stack;
* warm HuggingFace model and dataset caches;
* provision ALFWorld/WebShop/Search-QA through EnvCacheManager;
* write an env file that the launch command can source.

Implementation is delegated to
``backend.services.runtime.asset_provisioning`` so the harness and this CLI
share one provisioning path.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
# Make the repo root importable when this file is run as a standalone script
# (e.g. `.venv/bin/python scripts/sdar_gcp_assets.py` on a VM where the repo is
# NOT pip-installed): Python puts scripts/ on sys.path[0], never the repo root,
# so the lazy `from backend... import` calls below would raise ModuleNotFoundError.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
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
# Envs allowed to fail without blocking the run (operator-overridable). WebShop's
# 2022 frozen stack + JVM + data corpus are fragile; ALFWorld/Search-QA are not.
DEFAULT_BEST_EFFORT_ENVS = ("WebShop",)


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


def provision_envs(
    env_names: list[str], *, best_effort: set[str] | None = None
) -> dict[str, str]:
    """Provision the named envs and return the env_vars of those that came up.

    Exclusion of a REQUIRED env raises (it gates the run). Exclusion of a
    BEST-EFFORT env (e.g. WebShop, whose 2022 frozen stack + JVM + data corpus are
    fragile) warns and is skipped — mirroring ``env_cache``'s own graceful
    degradation so one finicky env never blocks a multi-hour GPU run.
    """
    from backend.services.runtime.env_cache import EnvCacheManager, provision_scope

    best = {e.strip().lower() for e in (best_effort or set())}
    result = provision_scope(env_names, EnvCacheManager())
    try:
        required_failures = []
        for e in result.exclusions:
            if e.item.strip().lower() in best:
                print(f"[WARN] env {e.item} unavailable (best-effort, skipped): {e.reason}")
            else:
                required_failures.append(e)
        if required_failures:
            details = ", ".join(f"{e.item}: {e.reason}" for e in required_failures)
            raise RuntimeError(f"required environment provisioning failed: {details}")
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
        # Preflight already provisioned all assets; the run must not re-provision.
        "OPENRESEARCH_PRELOAD_ASSETS": "0",
        **stable_values,
    }
    # Propagate the dedicated WebShop interpreter path when the dedicated-venv
    # install path was used (set by install_webshop_dedicated via ensure_assets).
    webshop_python = os.environ.get("OPENRESEARCH_WEBSHOP_PYTHON", "")
    if webshop_python:
        merged["OPENRESEARCH_WEBSHOP_PYTHON"] = webshop_python
    text = "\n".join(f'export {key}="{value}"' for key, value in sorted(merged.items()))
    path.write_text(text + "\n", encoding="utf-8")
    print(f"[OK] wrote {path}")


def run_checks(args: argparse.Namespace, env_vars: dict[str, str] | None = None) -> list[Check]:
    from backend.services.runtime.asset_provisioning import (
        _console_script_exists,
        _module_exists,
        webshop_importable,
    )

    checks = [
        Check("python", sys.version_info >= (3, 11), sys.version.split()[0]),
        Check("requirements file", SDAR_REQUIREMENTS.exists(), str(SDAR_REQUIREMENTS)),
    ]
    # System binaries — aggregated so a missing toolchain dep surfaces in THIS one
    # preflight pass instead of serially mid-build. cmake/ninja/gcc build
    # textworld's C extensions (REQUIRED); java/javac back WebShop's pyserini JVM
    # (best-effort, since WebShop is best-effort).
    for binname, required in (
        ("cmake", True),
        ("ninja", True),
        ("gcc", True),
        ("java", False),
        ("javac", False),
    ):
        found = shutil.which(binname)
        checks.append(Check(f"binary {binname}", found is not None, found or "not on PATH", required=required))
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
    # Probe the WebShop interpreter, not the run venv: under the dedicated-venv
    # split web_agent_site lives in OPENRESEARCH_WEBSHOP_PYTHON, never here.
    # Best-effort by default (WebShop is a best-effort env) — reports status but
    # does not gate; --allow-missing-webshop is kept for explicit operator intent.
    checks.append(Check(
        "import web_agent_site",
        webshop_importable(),
        os.environ.get("OPENRESEARCH_WEBSHOP_PYTHON") or "run venv",
        required=False,
    ))

    alf_raw = (env_vars or {}).get("ALFWORLD_DATA") or os.environ.get("ALFWORLD_DATA", "")
    alf_dir = Path(alf_raw) if alf_raw else Path()
    checks.append(Check(
        "ALFWorld data",
        bool(alf_raw) and (alf_dir / "json_2.1.1").exists(),
        str(alf_dir) if alf_raw else "ALFWORLD_DATA unset",
    ))

    # GPU-free model config resolve: catches too-old transformers or bad repo ids
    # before any GPU hours are leased.
    models = _split_csv(os.environ.get("OPENRESEARCH_SDAR_HF_MODELS"), DEFAULT_MODELS)
    if not args.skip_models:
        from backend.agents.schemas import AssetSpec
        from backend.services.runtime.asset_provisioning import check_assets

        resolve_results = check_assets(
            AssetSpec(models=models),
            cache_root=args.pip_cache_dir.parent,
        )
        for name, ok, detail in resolve_results:
            if not name.startswith("resolve "):
                continue
            checks.append(Check(name, ok, detail, required=True))
    else:
        for model_id in models:
            checks.append(Check(f"resolve {model_id}", True, "skipped (--skip-models)", required=False))

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
        from backend.agents.schemas import AssetSpec
        from backend.services.runtime.asset_provisioning import (
            AssetProvisionError,
            ensure_assets,
        )

        # Build the SDAR AssetSpec for the requirements + models + datasets.
        # WebShop is handled separately (env provisioning needs the live lease).
        models = _split_csv(os.environ.get("OPENRESEARCH_SDAR_HF_MODELS"), DEFAULT_MODELS)
        datasets = (
            _split_csv(os.environ.get("OPENRESEARCH_SDAR_DATASETS"), DEFAULT_DATASETS)
            if not args.skip_datasets
            else []
        )
        spec = AssetSpec(
            requirements_files=["backend/requirements-sdar.txt"],
            models=models if not args.skip_models else [],
            datasets=datasets,
            webshop=True,
        )
        cache_root = args.pip_cache_dir.parent  # runs/.cache
        try:
            ensure_assets(spec, cache_root=cache_root, webshop_python_version="3.10")
        except AssetProvisionError as exc:
            print(f"[FAIL] asset provisioning: {exc}", file=sys.stderr)
            return 1

        env_vars = provision_envs(
            _split_csv(os.environ.get("OPENRESEARCH_SDAR_ENVS"), DEFAULT_ENVS),
            best_effort=set(
                _split_csv(
                    os.environ.get("OPENRESEARCH_SDAR_BEST_EFFORT_ENVS"),
                    DEFAULT_BEST_EFFORT_ENVS,
                )
            ),
        )
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
