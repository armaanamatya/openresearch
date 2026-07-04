"""GCS-first, multi-source, resumable asset resolution (AssetResolverV2).

Feature-gated behind ``OPENRESEARCH_ASSET_RESOLVER_V2`` (default OFF).  When
the flag is unset, every call to :meth:`AssetResolverV2.resolve` delegates to
an :class:`~backend.services.runtime.asset_resolver.AssetResolver` V1 instance
— behaviour is **byte-identical** to today.  When the flag is set, the
four-tier resolution chain applies:

  1. **GCS project-cache** (content-key addressed; hit → no network fetch)
  2. **HF Hub** (resumable ``snapshot_download`` with backoff)
  3. **Direct URL / mirror** (resumable: ``aria2c -c`` → ``curl -C -`` →
     ``urllib``; checksum-verified if declared on the asset)
  4. **Google Drive** via injected ``rclone`` remote (skip if unconfigured)
  5. → verified :class:`~backend.agents.rlm.exclusion.Exclusion` (never fake-ok)

Write-through: any non-GCS success → upload the fetched **file** to the GCS
cache under its content-key.  Directory results (HF snapshots) bypass the GCS
tier (HF's own local cache is the backstop for those).

Sources are tried in a **fixed priority order** (HF → URL → gdrive) regardless
of identifier shape.  Each source callable decides internally whether it can
handle a given identifier and returns ``None`` if not; a raise is treated as
``None`` at the chain level (fail-soft).

All sources and the GCS store are **constructor-injectable** so unit tests are
fully hermetic (no real GCS, no real network, no real HuggingFace Hub call).
The real GCS impl is lazy-imported inside :class:`RealGcsStore` so this module
never drags in ``google-cloud-storage`` at import time.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from backend.agents.rlm.exclusion import AXIS_DATASET
from backend.services.runtime.asset_resolver import (
    AssetResolver,
    ResolveResult,
    _fetch_failure_exclusion,
    _identifier_of,
    _safe_filename,
    _unresolved_exclusion,
    resolve_framework,
)
from backend.services.runtime.credential_broker import CredentialBroker

if TYPE_CHECKING:
    from backend.services.runtime.asset_cache import AssetCache
    from backend.services.runtime.run_plan import RequiredAsset

__all__ = [
    "GcsStore",
    "InMemoryGcsStore",
    "RealGcsStore",
    "AssetResolverV2",
    "content_key",
    "v2_enabled",
]

# ---------------------------------------------------------------------------
# Feature-flag
# ---------------------------------------------------------------------------

_ENV_FLAG = "OPENRESEARCH_ASSET_RESOLVER_V2"


def v2_enabled() -> bool:
    """True iff ``OPENRESEARCH_ASSET_RESOLVER_V2`` is set to a truthy value."""
    return os.environ.get(_ENV_FLAG, "").lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Stable content-key
# ---------------------------------------------------------------------------


def content_key(asset: "RequiredAsset") -> str:
    """Stable, collision-resistant sha256 key for a ``RequiredAsset``'s canonical coords.

    Incorporates ``kind + identifier + optional revision + optional checksum``
    (the latter two via ``getattr`` so standard ``RequiredAsset`` instances —
    which have neither field — still produce a deterministic, valid key).
    Two assets that differ in any coordinate produce distinct keys.  Never
    raises; worst-case returns a hash of empty bytes.
    """
    try:
        kind = str(getattr(asset, "kind", "") or "")
        identifier = _identifier_of(asset)
        revision = str(getattr(asset, "revision", None) or "")
        checksum = str(getattr(asset, "checksum", None) or "")
        raw = f"{kind}\n{identifier}\n{revision}\n{checksum}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001
        return hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# GcsStore protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class GcsStore(Protocol):
    """Minimal GCS project-cache interface (injected; never imported at module level).

    Implementations must be fail-soft internally — callers wrap every call in
    ``try/except`` so an unexpected raise becomes a tier miss, not a crash.

    Key format used by ``AssetResolverV2``:
        ``"assets/<content_key>/<filename>"``
    """

    def exists(self, key: str) -> bool:
        """Return ``True`` iff ``key`` exists in the cache."""
        ...

    def download(self, key: str, dest: Path) -> None:
        """Download the cached object to ``dest`` (file path, not directory)."""
        ...

    def upload(self, src: Path, key: str) -> None:
        """Upload ``src`` (a file) to the cache under ``key``."""
        ...


@dataclass
class InMemoryGcsStore:
    """Hermetic in-memory :class:`GcsStore` for unit tests (not thread-safe)."""

    _data: dict[str, bytes] = field(default_factory=dict)
    # Tracks every (key, bytes) pair uploaded; tests assert on this.
    upload_calls: list[tuple[str, bytes]] = field(default_factory=list)

    def exists(self, key: str) -> bool:
        return key in self._data

    def download(self, key: str, dest: Path) -> None:
        if key not in self._data:
            raise KeyError(f"InMemoryGcsStore: key not found: {key!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._data[key])

    def upload(self, src: Path, key: str) -> None:
        data = src.read_bytes() if (src.exists() and src.is_file()) else b""
        self._data[key] = data
        self.upload_calls.append((key, data))

    # Convenience for tests: seed a pre-existing cache entry.
    def seed(self, key: str, data: bytes) -> None:
        self._data[key] = data


class RealGcsStore:
    """Production :class:`GcsStore` backed by ``google-cloud-storage`` (lazy import).

    Pass ``bucket`` as ``gs://my-bucket`` or just ``my-bucket``.
    ``credentials`` is optional; the real SDK uses ADC when absent.
    """

    def __init__(self, bucket: str, *, credentials=None) -> None:
        self._bucket_name = bucket.removeprefix("gs://")
        self._credentials = credentials
        self.__client = None
        self.__bucket = None

    def _get_bucket(self):
        if self.__client is None:
            from google.cloud import storage  # lazy import
            self.__client = storage.Client(credentials=self._credentials)
            self.__bucket = self.__client.bucket(self._bucket_name)
        return self.__bucket

    def exists(self, key: str) -> bool:
        blob = self._get_bucket().blob(key)
        return blob.exists()

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = self._get_bucket().blob(key)
        blob.download_to_filename(str(dest))

    def upload(self, src: Path, key: str) -> None:
        blob = self._get_bucket().blob(key)
        blob.upload_from_filename(str(src))


# ---------------------------------------------------------------------------
# Default network source implementations
# ---------------------------------------------------------------------------


def _default_hf_source(identifier: str, dest: Path) -> Path | None:
    """HF Hub source: handles ``'owner/name'``-shaped identifiers; ``None`` otherwise.

    Applies exponential backoff (3 tries, 2^n s).  Returns ``None`` on any
    error or un-handleable shape (fail-soft contract of source callables).
    """
    lowered = identifier.lower()
    if "/" not in identifier or lowered.startswith("http"):
        return None  # shape guard: not an HF repo id
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            from huggingface_hub import snapshot_download  # lazy
            result = snapshot_download(repo_id=identifier)
            return Path(result) if result else None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    _ = last_exc  # logged elsewhere if needed
    return None


def _build_url_source() -> Callable[[str, Path], Path | None]:
    """Build a resumable URL source (aria2c → curl → urllib) with backoff."""

    def _fetch(url: str, dest: Path) -> Path | None:
        lowered = url.lower()
        if not (lowered.startswith("http://") or lowered.startswith("https://")):
            return None  # shape guard: not a URL
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _attempt_once() -> None:
            if shutil.which("aria2c"):
                r = subprocess.run(
                    ["aria2c", "-c", "--out", dest.name, "--dir", str(dest.parent), url],
                    capture_output=True, timeout=3600,
                )
                if r.returncode != 0:
                    raise RuntimeError(
                        f"aria2c rc={r.returncode}: {r.stderr[:200]!r}"
                    )
                return
            if shutil.which("curl"):
                r = subprocess.run(
                    ["curl", "-C", "-", "-L", "-o", str(dest), url],
                    capture_output=True, timeout=3600,
                )
                if r.returncode != 0:
                    raise RuntimeError(
                        f"curl rc={r.returncode}: {r.stderr[:200]!r}"
                    )
                return
            import urllib.request  # stdlib fallback
            urllib.request.urlretrieve(url, str(dest))  # noqa: S310

        for attempt in range(3):
            try:
                _attempt_once()
                return dest if dest.exists() else None
            except Exception:  # noqa: BLE001
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    return _fetch


# Module-level default URL source (built once; stateless).
_default_url_source: Callable[[str, Path], Path | None] = _build_url_source()


# ---------------------------------------------------------------------------
# Checksum verification helper
# ---------------------------------------------------------------------------


def _verify_checksum(path: Path, declared: str) -> None:
    """Verify ``path`` against a declared checksum.

    ``declared`` may be ``"sha256:<hexdigest>"`` or a raw hex string (sha256
    assumed).  Raises :class:`ValueError` on mismatch; no-op when empty.
    """
    if not declared:
        return
    algo, _, hexdigest = declared.partition(":")
    if not hexdigest:
        algo, hexdigest = "sha256", algo
    algo = algo.lower().replace("-", "")
    import hashlib as _hl
    h = _hl.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != hexdigest.lower():
        raise ValueError(
            f"checksum mismatch for {path.name}: "
            f"expected {hexdigest.lower()}, got {actual}"
        )


# ---------------------------------------------------------------------------
# AssetResolverV2
# ---------------------------------------------------------------------------


class AssetResolverV2:
    """GCS-first, multi-source, resumable asset resolver (feature-flagged V2).

    **Flag OFF** (default): every :meth:`resolve` call is forwarded to an
    :class:`~backend.services.runtime.asset_resolver.AssetResolver` V1 instance
    — strictly byte-identical to today's behaviour.

    **Flag ON**: the four-tier chain runs:

    1. Recipe registry (local, no download) — same as V1.
    2. GCS project-cache (content-key keyed).
    3. Ordered source probing: ``hf_source → url_source → gdrive_source``.
       Each source is called with ``(identifier, dest_path)`` and must return
       a :class:`~pathlib.Path` on success or ``None`` to fall through.  A
       raise is caught at the chain level and treated as ``None``.
    4. All tiers exhausted → a verified
       :class:`~backend.agents.rlm.exclusion.Exclusion` (never fake-ok).

    Write-through: a successful file fetch from any non-GCS source is uploaded
    to the GCS store under its content-key.  Directory results (e.g. HF
    snapshot) bypass write-through (HF's local cache is the backstop for those).
    """

    def __init__(
        self,
        *,
        gcs_store: "GcsStore | None" = None,
        broker: CredentialBroker | None = None,
        # Source callables: (identifier: str, dest: Path) -> Path | None.
        # Must NEVER raise — return None to signal "can't handle" or "not found".
        # None → built-in default (shape-checked + backoff).
        hf_source: "Callable[[str, Path], Path | None] | None" = None,
        url_source: "Callable[[str, Path], Path | None] | None" = None,
        gdrive_source: "Callable[[str, Path], Path | None] | None" = None,
        recipe_lookup: "Callable[[str], object | None] | None" = None,
    ) -> None:
        self._gcs = gcs_store
        self._broker = broker if broker is not None else CredentialBroker()
        self._hf_source: Callable[[str, Path], Path | None] = (
            hf_source if hf_source is not None else _default_hf_source
        )
        self._url_source: Callable[[str, Path], Path | None] = (
            url_source if url_source is not None else _default_url_source
        )
        # gdrive is optional — None means "tier skipped"
        self._gdrive_source: Callable[[str, Path], Path | None] | None = gdrive_source
        self._recipe_lookup: Callable[[str], object | None] | None = recipe_lookup
        # V1 fallback — created on first use in the flag-off path
        self._v1: AssetResolver | None = None

    # -- Public API ----------------------------------------------------------

    def resolve(self, asset: "RequiredAsset", cache: "AssetCache") -> ResolveResult:
        """Resolve one asset.  Never raises — failures become verified Exclusions."""
        if not v2_enabled():
            # Flag OFF: delegate to V1 for byte-identical behaviour.
            return self._get_v1().resolve(asset, cache)
        try:
            return self._resolve_v2(asset, cache)
        except Exception as exc:  # noqa: BLE001 — outer guard (resolve must not raise)
            try:
                exclusion = _fetch_failure_exclusion(asset, exc)
            except Exception:  # noqa: BLE001
                exclusion = None
            return ResolveResult(
                ok=False, asset=asset, exclusion=exclusion,
                detail=f"v2 resolve error: {exc}",
            )

    def resolve_all(
        self,
        assets: "list[RequiredAsset] | tuple[RequiredAsset, ...] | None",
        cache: "AssetCache",
    ) -> list[ResolveResult]:
        """Resolve every asset in order; one failure never aborts the rest."""
        return [self.resolve(a, cache) for a in (assets or [])]

    # -- Flag-off V1 fallback ------------------------------------------------

    def _get_v1(self) -> AssetResolver:
        if self._v1 is None:
            self._v1 = AssetResolver(broker=self._broker)
        return self._v1

    # -- V2 dispatch ---------------------------------------------------------

    def _resolve_v2(self, asset: "RequiredAsset", cache: "AssetCache") -> ResolveResult:
        # Gated-credential check (same as V1)
        if getattr(asset, "gated", False) and not self._broker.available("hf_token"):
            exclusion = self._broker.gated_exclusion(
                item=_identifier_of(asset), secret_name="hf_token", axis=AXIS_DATASET
            )
            return ResolveResult(
                ok=False, asset=asset, exclusion=exclusion,
                detail="gated: hf_token unavailable",
            )

        kind = getattr(asset, "kind", "")

        if kind == "framework":
            env_vars = resolve_framework(asset.identifier)
            return ResolveResult(
                ok=True, asset=asset, env_vars=dict(env_vars),
                detail="framework resolved (no download)",
            )

        if kind in ("image", "service"):
            # Not an AssetResolver concern — EnvironmentAdapters own these.
            return ResolveResult(ok=True, asset=asset, detail="handled elsewhere")

        if kind in ("dataset", "weights"):
            return self._resolve_downloadable(asset, cache)

        return ResolveResult(
            ok=False, asset=asset,
            exclusion=_unresolved_exclusion(asset, f"unknown asset kind '{kind}'"),
        )

    def _resolve_downloadable(
        self, asset: "RequiredAsset", cache: "AssetCache"
    ) -> ResolveResult:
        """Four-tier resolution for dataset/weights assets."""
        identifier = _identifier_of(asset)
        ck = content_key(asset)
        filename = _safe_filename(identifier) or "asset.bin"
        # GCS key: "assets/<content_key>/<filename>"
        gcs_key = f"assets/{ck}/{filename}"
        dest_dir = Path(getattr(cache, "cache_dir", Path("."))) / ck
        dest_path = dest_dir / filename

        # --- Tier 0: recipe registry (local, no download) ---
        recipe_lookup = self._recipe_lookup
        if recipe_lookup is None:
            from backend.services.runtime.asset_resolver import _default_recipe_lookup
            recipe_lookup = _default_recipe_lookup
        try:
            if recipe_lookup(identifier) is not None:
                return ResolveResult(
                    ok=True, asset=asset,
                    detail="resolved via dataset recipe (registry/torchvision, no download)",
                )
        except Exception:  # noqa: BLE001 — recipe error → fall through
            pass

        # --- Tier 1: GCS project-cache ---
        if self._gcs is not None:
            try:
                if self._gcs.exists(gcs_key):
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    self._gcs.download(gcs_key, dest_path)
                    return ResolveResult(
                        ok=True, asset=asset, local_path=str(dest_path),
                        detail=f"resolved via gcs cache (key={gcs_key})",
                    )
            except Exception:  # noqa: BLE001 — GCS miss / error → fall through
                pass

        # --- Tiers 2-4: ordered source probing (HF → URL → gdrive) ---
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        sources: list[Callable[[str, Path], Path | None]] = [
            self._hf_source,
            self._url_source,
        ]
        if self._gdrive_source is not None:
            sources.append(self._gdrive_source)

        for source_fn in sources:
            local: Path | None = None
            try:
                local = source_fn(identifier, dest_path)
            except Exception:  # noqa: BLE001 — raise == None in the priority chain
                local = None

            if local is not None:
                # Checksum verification if the asset declares one
                declared_checksum = str(getattr(asset, "checksum", None) or "")
                if declared_checksum:
                    try:
                        _verify_checksum(local, declared_checksum)
                    except Exception as exc:  # noqa: BLE001
                        return ResolveResult(
                            ok=False, asset=asset,
                            exclusion=_fetch_failure_exclusion(asset, exc),
                            detail=f"checksum mismatch: {exc}",
                        )
                # Write-through to GCS (file-only; directories bypass)
                self._write_through(local, gcs_key)
                return ResolveResult(
                    ok=True, asset=asset, local_path=str(local),
                    detail=f"resolved via source (gcs_key={gcs_key})",
                )

        # All tiers exhausted → verified Exclusion (never fake-ok)
        return ResolveResult(
            ok=False, asset=asset,
            exclusion=_unresolved_exclusion(
                asset,
                "all resolution tiers exhausted (recipe/gcs/hf/url/gdrive)",
            ),
        )

    def _write_through(self, src: Path, gcs_key: str) -> None:
        """Upload ``src`` to the GCS store under ``gcs_key``.

        Fail-soft: never raises into the caller.  Directory results (e.g. HF
        snapshots) are skipped — HF's local cache is the backstop for those.
        """
        if self._gcs is None:
            return
        try:
            if src.is_file():
                self._gcs.upload(src, gcs_key)
            # else: directory (HF snapshot) — write-through skipped intentionally
        except Exception:  # noqa: BLE001 — write-through failure is non-fatal
            pass
