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
from env vars.  :func:`prestage_env_assets` resolves each spec's
:attr:`PrestageSpec.candidates` **in order** — the first candidate whose
resolve succeeds AND passes the spec's declared integrity pin
(:attr:`PrestageSpec.sha256` / :attr:`PrestageSpec.min_size_bytes`) wins and is
copied to the env-specific directory before the adapter runs; a candidate that
resolves but fails integrity is discarded (never copied) and the next
candidate is tried.

Only :data:`ENV_ASSET_REGISTRY` contains per-env knowledge — adding a new
environment, or a new fallback mirror for an existing one, is a data-only
change (one registry entry / one more candidate), not a code change.
"""

from __future__ import annotations

import hashlib
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

    ``env_dir_var``  — name of the env var that names the env's data directory
                        (e.g. ``"WEBSHOP_DATA_DIR"``).  If this var is unset at
                        staging time the file is not copied (skip, no raise).
    ``dest_subpath``  — path relative to the env dir where the adapter expects
                        the file (e.g. ``"items_shuffle.json"``).
    ``asset``         — convenience single-source constructor: a
                        :class:`~backend.services.runtime.run_plan.RequiredAsset`
                        to resolve.  Populates a single-element
                        :attr:`candidates` tuple when ``candidates`` itself is
                        left empty (kept for backward compatibility with the
                        original one-source-per-file shape).
    ``candidates``    — ordered tuple of :class:`RequiredAsset` mirrors to try;
                        the first one that resolves *and* passes the integrity
                        pin below wins.  Preferred over ``asset`` for any file
                        with more than one known source.
    ``sha256``        — expected sha256 hex digest of the staged file, verified
                        when set; a mismatch discards that candidate's file
                        (never copied to ``dest_subpath``) and the next
                        candidate is tried.
    ``min_size_bytes``— cheap sanity floor applied when no ``sha256`` is
                        declared (or in addition to it); a resolved file
                        smaller than this is rejected the same way.
    """

    env_dir_var: str
    dest_subpath: str
    asset: RequiredAsset | None = None
    candidates: tuple[RequiredAsset, ...] = ()
    sha256: str | None = None
    min_size_bytes: int | None = None

    def __post_init__(self) -> None:
        # Backward-compat alias: a single `asset=` populates `candidates`
        # when the caller didn't already supply an explicit candidate tuple.
        if not self.candidates and self.asset is not None:
            object.__setattr__(self, "candidates", (self.asset,))


# ---------------------------------------------------------------------------
# Registry — one entry per environment that needs corpus pre-staging.
# Keys: canonical-lowercased env name (matching EnvCacheManager.setup input).
# Data-only: add a new environment (or a new fallback mirror) here, not in
# any function.
# ---------------------------------------------------------------------------

# WebShop corpus mirrors, verified byte-identical (3-way, 2026-07-03) and
# anonymously fetchable, tried in this order before falling back to the
# operator-only gdrive ids (which need an authenticated rclone remote).
_WEBSHOP_HF_MIRRORS_FULL: tuple[str, ...] = (
    "YWZBrandon/webshop-data",
    "HongbangYuan/webshop",
    "quanwei0/webshop-minimal",
)
# The 1K subsets are only mirrored on the first two repos.
_WEBSHOP_HF_MIRRORS_1000: tuple[str, ...] = (
    "YWZBrandon/webshop-data",
    "HongbangYuan/webshop",
)

_ITEMS_SHUFFLE_SHA256 = "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8"
_ITEMS_SHUFFLE_SIZE = 5_479_720_229
_ITEMS_INS_V2_SHA256 = "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e"
_ITEMS_INS_V2_SIZE = 186_295_270
# 1K subsets: sizes only sanity-checked against the primary mirror (no
# independent cross-mirror checksum was verified for these).
_ITEMS_SHUFFLE_1000_SIZE = 4_467_013
_ITEMS_INS_V2_1000_SIZE = 147_099


def _hf_resolve_url(repo: str, filename: str) -> str:
    """Build a direct ``resolve/main`` download URL for an HF dataset repo file."""
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"


def _url_mirrors(repos: tuple[str, ...], filename: str) -> tuple[RequiredAsset, ...]:
    """One URL-kind candidate per repo, in the given order."""
    return tuple(
        RequiredAsset(kind="dataset", identifier=_hf_resolve_url(repo, filename))
        for repo in repos
    )


ENV_ASSET_REGISTRY: dict[str, tuple[PrestageSpec, ...]] = {
    "webshop": (
        PrestageSpec(
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_shuffle.json",
            candidates=_url_mirrors(_WEBSHOP_HF_MIRRORS_FULL, "items_shuffle.json")
            + (
                # Operator-only last resort: needs an authenticated rclone remote.
                RequiredAsset(
                    kind="dataset",
                    identifier="gdrive:1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
                ),
            ),
            sha256=_ITEMS_SHUFFLE_SHA256,
            min_size_bytes=_ITEMS_SHUFFLE_SIZE,
        ),
        PrestageSpec(
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_ins_v2.json",
            candidates=_url_mirrors(_WEBSHOP_HF_MIRRORS_FULL, "items_ins_v2.json")
            + (
                RequiredAsset(
                    kind="dataset",
                    identifier="gdrive:1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
                ),
            ),
            sha256=_ITEMS_INS_V2_SHA256,
            min_size_bytes=_ITEMS_INS_V2_SIZE,
        ),
        PrestageSpec(
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_shuffle_1000.json",
            candidates=_url_mirrors(_WEBSHOP_HF_MIRRORS_1000, "items_shuffle_1000.json"),
            min_size_bytes=_ITEMS_SHUFFLE_1000_SIZE,
        ),
        PrestageSpec(
            env_dir_var="WEBSHOP_DATA_DIR",
            dest_subpath="items_ins_v2_1000.json",
            candidates=_url_mirrors(_WEBSHOP_HF_MIRRORS_1000, "items_ins_v2_1000.json"),
            min_size_bytes=_ITEMS_INS_V2_1000_SIZE,
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
# Integrity verification
# ---------------------------------------------------------------------------

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB — stream large corpus files (5.5 GB) without buffering


def _verify_integrity(
    path: str, *, sha256: str | None, min_size_bytes: int | None
) -> bool:
    """Return ``True`` iff the file at ``path`` passes the declared integrity pins.

    No pins declared → trivially ``True`` (nothing to check).  The sha256 is
    computed by streaming the file in fixed-size chunks so a multi-GB corpus
    file is never loaded fully into memory.  Never raises — any I/O error
    while stat'ing or reading the file is treated as a failed check (the
    caller discards the candidate and tries the next one).
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return False

    if min_size_bytes is not None and size < min_size_bytes:
        return False

    if sha256:
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
        except OSError:
            return False
        if digest.hexdigest().lower() != sha256.strip().lower():
            return False

    return True


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

    For each :class:`PrestageSpec` in the environment's registry entry, tries
    :attr:`PrestageSpec.candidates` **in order**: the first candidate that
    resolves successfully AND passes the spec's declared integrity pin
    (:attr:`PrestageSpec.sha256` / :attr:`PrestageSpec.min_size_bytes`) is
    copied to ``<env_dir>/<dest_subpath>``; a candidate that resolves but
    fails the integrity check is discarded (never copied) and the next
    candidate is tried. Returns the list of destination paths successfully
    staged.  Never raises — every per-candidate body is wrapped in
    ``try/except``.

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
            env_dir_val = env_getter(spec.env_dir_var, "")
            if not env_dir_val:
                # env var not set → dest unknown; skip this spec entirely
                # (no candidate resolve attempted — no raise).
                continue
            dest = Path(env_dir_val) / spec.dest_subpath

            for candidate in spec.candidates:
                try:
                    res = resolver.resolve(candidate, cache)
                    if not (res.ok and res.local_path):
                        continue
                    if not _verify_integrity(
                        res.local_path,
                        sha256=spec.sha256,
                        min_size_bytes=spec.min_size_bytes,
                    ):
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    copier(res.local_path, str(dest))
                    staged.append(str(dest))
                    break  # first candidate to resolve + pass integrity wins
                except Exception:  # noqa: BLE001 — per-candidate fail-soft
                    continue
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
