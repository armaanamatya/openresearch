"""Pre-stage env-specific corpus assets into the provider-expected directory.

Feature-gated behind ``OPENRESEARCH_ASSET_RESOLVER_V2`` (same flag as
:class:`~backend.services.runtime.asset_resolver_v2.AssetResolverV2`).

**Flag OFF (default):** every public function in this module is a provable
no-op — :func:`build_default_resolver` returns ``None``, and
:func:`prestage_env_assets` returns ``[]`` immediately.  The wired callers
(``env_cache.provision_scope``) guard on the return value, so the flag-off
path is byte-identical to before.

**Flag ON:** :data:`ENV_ASSET_REGISTRY` maps canonical-lowercased env names to
their :class:`PrestageSpec` tuples.  :func:`build_default_resolver` constructs
a production :class:`~backend.services.runtime.asset_resolver_v2.AssetResolverV2`
from env vars.  :func:`prestage_env_assets` resolves each spec's asset and
copies it to the env-specific directory before the adapter runs.

Only :data:`ENV_ASSET_REGISTRY` contains per-env knowledge — adding a new
environment is a data-only change (one registry entry), not a code change.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from backend.services.runtime.asset_resolver_v2 import (
    AssetResolverV2,
    RealGcsStore,
    v2_enabled,
)
from backend.services.runtime.run_plan import RequiredAsset

if TYPE_CHECKING:
    from backend.services.runtime.asset_cache import AssetCache

log = logging.getLogger(__name__)

__all__ = [
    "PrestageSpec",
    "ENV_ASSET_REGISTRY",
    "build_default_resolver",
    "prestage_env_assets",
]

# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrestageSpec:
    """One corpus file an environment needs pre-staged before its adapter runs.

    ``asset``       — the :class:`~backend.services.runtime.run_plan.RequiredAsset`
                      to resolve (``kind`` / ``identifier``).
    ``env_dir_var`` — name of the env var that names the env's data directory
                      (e.g. ``"WEBSHOP_DATA_DIR"``).  If this var is unset at
                      staging time the file is not copied (skip, no raise).
    ``dest_subpath``— path relative to the env dir where the adapter expects the
                      file (e.g. ``"items_shuffle.json"``).
    """

    asset: RequiredAsset
    env_dir_var: str
    dest_subpath: str


# ---------------------------------------------------------------------------
# Registry — one entry per environment that needs corpus pre-staging.
# Keys: canonical-lowercased env name (matching EnvCacheManager.setup input).
# Data-only: add a new environment here, not in any function.
# ---------------------------------------------------------------------------

ENV_ASSET_REGISTRY: dict[str, tuple[PrestageSpec, ...]] = {
    "webshop": (
        PrestageSpec(
            asset=RequiredAsset(
                kind="dataset",
                identifier="gdrive:1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
            ),
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_shuffle.json",
        ),
        PrestageSpec(
            asset=RequiredAsset(
                kind="dataset",
                identifier="gdrive:1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
            ),
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_ins_v2.json",
        ),
    ),
}


# ---------------------------------------------------------------------------
# gdrive source factory (rclone-backed)
# ---------------------------------------------------------------------------


def _build_rclone_gdrive_source(
    remote: str,
) -> Callable[[str, Path], Path | None]:
    """Return a gdrive source callable that fetches ``gdrive:<id>`` via rclone.

    The returned callable matches the ``(identifier: str, dest: Path) -> Path | None``
    contract required by :class:`AssetResolverV2`.  It is fail-soft: any error
    or unhandleable identifier shape returns ``None``.
    """

    def _fetch(identifier: str, dest: Path) -> Path | None:
        if not identifier.startswith("gdrive:"):
            return None  # shape guard: not a gdrive id
        file_id = identifier.removeprefix("gdrive:")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(
                ["rclone", "copyto", f"{remote}:{file_id}", str(dest)],
                capture_output=True,
                timeout=3600,
            )
            if r.returncode != 0:
                return None
            return dest if dest.exists() else None
        except Exception:  # noqa: BLE001
            return None

    return _fetch


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_resolver() -> AssetResolverV2 | None:
    """Construct a production :class:`AssetResolverV2` from env vars.

    Returns ``None`` when:

    * ``OPENRESEARCH_ASSET_RESOLVER_V2`` is unset/falsy (flag-off — the
      default; all callers treat ``None`` as a provable no-op); or
    * any construction error occurs (fail-soft).

    GCS bucket: ``OPENRESEARCH_ASSET_GCS_BUCKET``, falling back to
    ``OPENRESEARCH_SDAR_REPORT_GCS`` (the report-upload bucket already
    configured on GCP deployments).  If neither is set, ``gcs_store=None``
    (GCS tier simply skipped).

    gdrive: built from ``OPENRESEARCH_ASSET_RCLONE_REMOTE`` if set; else
    ``gdrive_source=None`` (tier skipped).
    """
    if not v2_enabled():
        return None
    try:
        bucket = (
            os.environ.get("OPENRESEARCH_ASSET_GCS_BUCKET")
            or os.environ.get("OPENRESEARCH_SDAR_REPORT_GCS")
            or ""
        )
        gcs_store: RealGcsStore | None = RealGcsStore(bucket) if bucket else None

        gdrive_source: Callable[[str, Path], Path | None] | None = None
        rclone_remote = os.environ.get("OPENRESEARCH_ASSET_RCLONE_REMOTE")
        if rclone_remote:
            gdrive_source = _build_rclone_gdrive_source(rclone_remote)

        return AssetResolverV2(gcs_store=gcs_store, gdrive_source=gdrive_source)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Staging function
# ---------------------------------------------------------------------------


def prestage_env_assets(
    env_name: str,
    resolver: AssetResolverV2 | None,
    cache: "AssetCache",
    *,
    env_getter: Callable[[str, str], str] = os.environ.get,  # type: ignore[assignment]
    copier: Callable[[str, str], None] = shutil.copy2,  # type: ignore[assignment]
) -> list[str]:
    """Resolve and copy corpus assets for one environment.

    Returns the list of destination paths successfully staged.  Never raises
    — every per-spec body is wrapped in ``try/except``.

    **Flag-off or resolver=None → returns [] immediately (no-op).**

    ``env_getter`` and ``copier`` are injected to keep the function hermetically
    testable without touching the filesystem or real env vars.
    """
    # Flag-off fast path: resolver is None (returned by build_default_resolver)
    # or the flag was flipped OFF between construction and this call.
    if resolver is None or not v2_enabled():
        return []

    specs = ENV_ASSET_REGISTRY.get((env_name or "").strip().lower())
    if not specs:
        return []

    staged: list[str] = []
    for spec in specs:
        try:
            res = resolver.resolve(spec.asset, cache)
            if not (res.ok and res.local_path):
                continue
            env_dir_val = env_getter(spec.env_dir_var, "")
            if not env_dir_val:
                # env var not set → dest unknown; skip copy (no raise)
                continue
            dest = Path(env_dir_val) / spec.dest_subpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            copier(res.local_path, str(dest))
            staged.append(str(dest))
        except Exception:  # noqa: BLE001 — per-spec fail-soft
            pass

    if staged:
        log.debug(
            "prestage_env_assets(%r): staged %d file(s): %s",
            env_name,
            len(staged),
            staged,
        )
    return staged
