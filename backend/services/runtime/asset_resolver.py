"""Generic AssetResolver — resolves Phase-1b ``RequiredAsset``s for ANY paper.

Phase 1d, Unit B of the credentials/assets/cpu-tier refactor (see
``docs/history/plans/2026-07-01-phase-1d-credentials-assets-cpu-tier.md``):
this module WRAPS the existing, proven fetchers (``dataset_recipes.find_recipe``
for the registry/torchvision path, ``huggingface_hub.snapshot_download`` for
weights/datasets keyed by an HF repo id, a plain URL fetch for everything else)
behind one small ``AssetResolver.resolve(asset, cache) -> ResolveResult``
dispatch. It does NOT rewrite ``asset_provisioning.py`` / ``dataset_recipes.py``
/ ``environment_detective.py`` — those stay exactly as they are; this is a new,
paper-agnostic seam that reuses them.

Fail-soft is the load-bearing behaviour, not an afterthought: a gated asset
without the matching credential, an asset that matches no known fetcher shape,
or a fetcher that raises, ALL resolve to a verified ``Exclusion`` (never an
unhandled exception out of ``resolve``/``resolve_all``, and never a fake
``ok=True``). This mirrors the harness-wide fairness principle in
``backend.agents.rlm.exclusion``: an asset the harness genuinely could not
acquire is excluded from scoring, not silently scored as if it worked.

``_FRAMEWORK_MATRIX`` is a data-driven, extendable python/cuda compatibility
table scoped to this module. It intentionally duplicates the SHAPE of
``backend.agents.environment_detective._FRAMEWORK_COMPATIBILITY`` (which stays
untouched — a different call site with a different purpose) rather than
importing it, so this module has no dependency on that agent-facing Dockerfile
generator.

Every fetcher is constructor-injectable so tests are fully hermetic (no real
network, no real HuggingFace Hub call); the defaults lazy-import their heavy
dependency so ``import backend.services.runtime.asset_resolver`` never drags in
``huggingface_hub``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from backend.agents.rlm.exclusion import AXIS_DATASET, KIND_ENV_SETUP_FAILED, Exclusion
from backend.services.runtime.credential_broker import CredentialBroker

if TYPE_CHECKING:
    from backend.services.runtime.asset_cache import AssetCache
    from backend.services.runtime.run_plan import RequiredAsset

__all__ = [
    "resolve_framework",
    "ResolveResult",
    "AssetResolver",
]

# ---------------------------------------------------------------------------
# Data-driven framework -> {version -> {python, cuda}} matrix.
#
# Ordered newest-version-first per framework: ``resolve_framework`` falls back
# to the FIRST entry (the newest known version) for an unknown/absent version,
# so this ordering is load-bearing, not cosmetic.
# ---------------------------------------------------------------------------
_FRAMEWORK_MATRIX: dict[str, dict[str, dict[str, str]]] = {
    "pytorch": {
        "2.5.1": {"python": "3.12", "cuda": "12.1"},
        "2.2.0": {"python": "3.11", "cuda": "12.1"},
        "2.1.0": {"python": "3.11", "cuda": "11.8"},
        "2.0.0": {"python": "3.10", "cuda": "11.7"},
    },
    "tensorflow": {
        "2.15.0": {"python": "3.11", "cuda": "12.2"},
        "2.14.0": {"python": "3.11", "cuda": "11.8"},
    },
    "jax": {
        "0.4.25": {"python": "3.11", "cuda": "12.1"},
    },
}

# Returned for a framework this matrix has never heard of — never an empty
# dict, never a raise.
_SAFE_DEFAULT_FRAMEWORK_ENV: dict[str, str] = {"python": "3.11", "cuda": "12.1"}


def resolve_framework(name: str, version: str | None = None) -> dict:
    """Data-driven framework -> ``{"python": ..., "cuda": ...}``; never raises.

    - known framework + known version -> that exact entry.
    - known framework + unknown/absent version -> the newest known version's
      entry (graceful fallback — still a real, non-empty compatibility pin).
    - unknown framework -> :data:`_SAFE_DEFAULT_FRAMEWORK_ENV` (a safe default,
      never an empty dict).
    """
    try:
        versions = _FRAMEWORK_MATRIX.get(str(name or "").strip().lower())
        if not versions:
            return dict(_SAFE_DEFAULT_FRAMEWORK_ENV)
        if version and version in versions:
            return dict(versions[version])
        newest = next(iter(versions.values()))
        return dict(newest)
    except Exception:  # noqa: BLE001 — this function must never raise
        return dict(_SAFE_DEFAULT_FRAMEWORK_ENV)


@dataclass(frozen=True)
class ResolveResult:
    """Outcome of resolving one ``RequiredAsset``."""

    ok: bool
    asset: "RequiredAsset"
    local_path: str | None = None
    env_vars: dict = field(default_factory=dict)
    exclusion: "Exclusion | None" = None
    detail: str = ""


def _identifier_of(asset: object) -> str:
    """Best-effort, never-raising read of ``asset.identifier``."""
    ident = getattr(asset, "identifier", "") or ""
    return str(ident).strip() or "unknown-asset"


def _unresolved_exclusion(asset: object, reason: str) -> Exclusion:
    """A verified exclusion for an asset that matched no known fetcher shape."""
    return Exclusion(
        item=_identifier_of(asset),
        axis=AXIS_DATASET,
        kind=KIND_ENV_SETUP_FAILED,
        reason=f"unresolved asset: {reason}",
        verified=True,
        evidence=reason,
    )


def _fetch_failure_exclusion(asset: object, exc: Exception) -> Exclusion:
    """A verified exclusion for an asset whose fetcher raised — fail-soft, never fake-ok."""
    return Exclusion(
        item=_identifier_of(asset),
        axis=AXIS_DATASET,
        kind=KIND_ENV_SETUP_FAILED,
        reason=f"fetch failed: {exc}",
        verified=True,
        evidence=str(exc)[:500],
    )


def _default_hf_snapshot(repo_id: str) -> str:
    """Lazy-imported default: a real HuggingFace Hub snapshot download."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id)


def _default_url_fetch(url: str, dest: str) -> str:
    """Lazy-imported default: a plain stdlib URL download."""
    import urllib.request

    urllib.request.urlretrieve(url, dest)  # noqa: S310 — operator-provided asset URL
    return dest


def _default_recipe_lookup(identifier: str):
    """Default recipe lookup: the existing, proven dataset-recipe registry."""
    from backend.agents.dataset_recipes import find_recipe

    return find_recipe(identifier)


def _safe_filename(url: str) -> str:
    """Best-effort filename for a URL-fetch destination; never raises/empty."""
    try:
        from urllib.parse import urlparse

        name = Path(urlparse(url).path).name
        return name or "download.bin"
    except Exception:  # noqa: BLE001
        return "download.bin"


class AssetResolver:
    """Resolves any :class:`RequiredAsset` into the shared :class:`AssetCache`.

    Reuses the existing fetchers (dataset-recipe registry, HF snapshot
    download, plain URL fetch) behind injected callables so it is fully
    hermetic under test. Dispatch is by ``asset.kind``; see the module
    docstring and the Phase-1d plan for the exact rules. Never raises: any
    fetcher failure or unresolvable shape becomes a verified ``Exclusion``.
    """

    def __init__(
        self,
        *,
        broker: CredentialBroker | None = None,
        hf_snapshot: Callable[[str], str] | None = None,
        url_fetch: Callable[[str, str], str] | None = None,
        recipe_lookup: Callable[[str], object | None] | None = None,
    ) -> None:
        self._broker = broker if broker is not None else CredentialBroker()
        self._hf_snapshot = hf_snapshot if hf_snapshot is not None else _default_hf_snapshot
        self._url_fetch = url_fetch if url_fetch is not None else _default_url_fetch
        self._recipe_lookup = (
            recipe_lookup if recipe_lookup is not None else _default_recipe_lookup
        )

    def resolve(self, asset: "RequiredAsset", cache: "AssetCache") -> ResolveResult:
        """Resolve one asset. Never raises — any failure becomes a verified Exclusion."""
        try:
            return self._resolve(asset, cache)
        except Exception as exc:  # noqa: BLE001 — resolve() must never raise (fail-soft contract)
            try:
                exclusion = _fetch_failure_exclusion(asset, exc)
            except Exception:  # noqa: BLE001 — even Exclusion construction must not raise
                exclusion = None
            return ResolveResult(
                ok=False, asset=asset, exclusion=exclusion, detail=f"resolve error: {exc}"
            )

    def resolve_all(
        self, assets: "list[RequiredAsset] | tuple[RequiredAsset, ...] | None", cache: "AssetCache"
    ) -> list[ResolveResult]:
        """Resolve every asset in order; one fetcher failure never aborts the rest."""
        return [self.resolve(a, cache) for a in (assets or [])]

    # -- dispatch -----------------------------------------------------------

    def _resolve(self, asset: "RequiredAsset", cache: "AssetCache") -> ResolveResult:
        if getattr(asset, "gated", False) and not self._broker.available("hf_token"):
            exclusion = self._broker.gated_exclusion(
                item=_identifier_of(asset), secret_name="hf_token", axis=AXIS_DATASET
            )
            return ResolveResult(
                ok=False, asset=asset, exclusion=exclusion, detail="gated: hf_token unavailable"
            )

        kind = getattr(asset, "kind", "")

        if kind in ("dataset", "weights"):
            return self._resolve_dataset_or_weights(asset, cache)

        if kind == "framework":
            env_vars = resolve_framework(asset.identifier)
            return ResolveResult(
                ok=True, asset=asset, env_vars=dict(env_vars), detail="framework resolved (no download)"
            )

        if kind in ("image", "service"):
            # Not an AssetResolver concern — EnvironmentAdapters own services/images.
            return ResolveResult(ok=True, asset=asset, detail="handled elsewhere")

        return ResolveResult(
            ok=False,
            asset=asset,
            exclusion=_unresolved_exclusion(asset, f"unknown asset kind '{kind}'"),
        )

    def _resolve_dataset_or_weights(self, asset: "RequiredAsset", cache: "AssetCache") -> ResolveResult:
        identifier = _identifier_of(asset)

        try:
            recipe = self._recipe_lookup(identifier)
        except Exception as exc:  # noqa: BLE001 — fetcher raise => fail-soft exclusion
            return ResolveResult(ok=False, asset=asset, exclusion=_fetch_failure_exclusion(asset, exc))
        if recipe is not None:
            return ResolveResult(
                ok=True, asset=asset, detail="resolved via dataset recipe (registry/torchvision, no download)"
            )

        lowered = identifier.lower()
        is_url = lowered.startswith("http://") or lowered.startswith("https://")

        # HF-repo-shaped id ("owner/name") — checked before the URL branch so a
        # bare repo id never falls through; a URL is excluded here even though
        # it also contains "/", since it has its own branch below.
        if "/" in identifier and not is_url:
            try:
                local_path = self._hf_snapshot(identifier)
            except Exception as exc:  # noqa: BLE001
                return ResolveResult(ok=False, asset=asset, exclusion=_fetch_failure_exclusion(asset, exc))
            return ResolveResult(
                ok=True, asset=asset, local_path=local_path, detail="resolved via hf_snapshot"
            )

        if is_url:
            dest = str(Path(getattr(cache, "cache_dir", Path("."))) / _safe_filename(identifier))
            try:
                local_path = self._url_fetch(identifier, dest)
            except Exception as exc:  # noqa: BLE001
                return ResolveResult(ok=False, asset=asset, exclusion=_fetch_failure_exclusion(asset, exc))
            return ResolveResult(
                ok=True, asset=asset, local_path=local_path, detail="resolved via url_fetch"
            )

        return ResolveResult(
            ok=False,
            asset=asset,
            exclusion=_unresolved_exclusion(
                asset, "no recipe match, not HF-repo-shaped, not a URL"
            ),
        )
