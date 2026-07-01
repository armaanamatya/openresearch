"""Typed required-assets extraction from a paper's reproduction contract (Phase 1b).

Deterministic, fail-soft, and UNWIRED: no live code path calls this yet (a later
Phase-1c ``ReproductionRun`` will). ``extract_required_assets`` resolves a flat,
deduped list of :class:`RequiredAsset` from whichever source is available, in
priority order:

1. ``contract`` -- a ``SemanticReproductionContract`` (richest, provenance-backed).
2. ``claim_map`` -- a legacy ``PaperClaimMap`` (planner output).
3. ``rubric`` -- a generated rubric dict; best-effort dataset/framework mention
   scan over leaf ``requirements`` text.

Bad, absent, or malformed input never raises -- the function always returns a
(possibly empty) list.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.agents.dataset_recipes import find_recipes_in_text

if TYPE_CHECKING:
    from backend.agents.resilience.budget import RunBudget
    from backend.agents.schemas import ScopeSpec

_FRAMEWORK_KEYWORDS = ("pytorch", "tensorflow", "jax")


@dataclass(frozen=True)
class RequiredAsset:
    """One externally-sourced thing a reproduction run needs before it trains."""

    kind: str  # "dataset" | "weights" | "image" | "service" | "framework"
    identifier: str
    gated: bool = False  # known/suspected to need a credential (best-effort hint)
    size_hint_gb: float | None = None


@dataclass(frozen=True)
class RunPlan:
    """The pre-lease plan a (Phase-1c) ReproductionRun assembles before leasing GPU."""

    paper_id: str = ""
    scope: "ScopeSpec | None" = None
    budget: "RunBudget | None" = None
    required_assets: tuple[RequiredAsset, ...] = ()


def _flatten_rubric_leaf_texts(node: object) -> list[str]:
    """Recursively collect every non-empty ``requirements`` string under ``children``."""
    texts: list[str] = []
    if not isinstance(node, dict):
        return texts
    requirements = node.get("requirements")
    if isinstance(requirements, str) and requirements.strip():
        texts.append(requirements)
    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            texts.extend(_flatten_rubric_leaf_texts(child))
    return texts


def _dedupe(assets: list[RequiredAsset]) -> list[RequiredAsset]:
    """Dedupe by (kind, identifier.casefold()), preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    deduped: list[RequiredAsset] = []
    for asset in assets:
        key = (asset.kind, asset.identifier.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(asset)
    return deduped


def extract_required_assets(
    *, contract=None, claim_map=None, rubric=None
) -> list[RequiredAsset]:
    """Extract a typed, deduped ``RequiredAsset`` list from whichever source is given.

    Resolution order (first non-None source wins): ``contract`` > ``claim_map`` >
    ``rubric``. Fail-soft: any internal error, or no source at all, returns
    ``[]`` -- this must never raise into a caller.
    """
    try:
        assets: list[RequiredAsset] = []

        if contract is not None:
            for identity in getattr(contract, "resource_identities", None) or []:
                kind = getattr(identity, "kind", None)
                identifier = getattr(identity, "identifier", None)
                if kind and identifier:
                    assets.append(RequiredAsset(kind=str(kind), identifier=str(identifier)))
            profile = getattr(contract, "capability_profile", None)
            for name in getattr(profile, "datasets", None) or []:
                if name:
                    assets.append(RequiredAsset(kind="dataset", identifier=str(name)))
            for name in getattr(profile, "frameworks", None) or []:
                if name:
                    assets.append(RequiredAsset(kind="framework", identifier=str(name)))
            for name in getattr(profile, "external_services", None) or []:
                if name:
                    assets.append(RequiredAsset(kind="service", identifier=str(name)))

        elif claim_map is not None:
            for requirement in getattr(claim_map, "datasets", None) or []:
                name = getattr(requirement, "name", None)
                if name:
                    assets.append(RequiredAsset(kind="dataset", identifier=str(name)))
            architecture = getattr(claim_map, "model_architecture", "") or ""
            if architecture.strip():
                assets.append(RequiredAsset(kind="weights", identifier=architecture.strip()))

        elif rubric is not None:
            for text in _flatten_rubric_leaf_texts(rubric):
                for recipe in find_recipes_in_text(text):
                    assets.append(
                        RequiredAsset(kind="dataset", identifier=recipe.canonical_name)
                    )
                lowered = text.lower()
                for framework in _FRAMEWORK_KEYWORDS:
                    if framework in lowered:
                        assets.append(RequiredAsset(kind="framework", identifier=framework))

        return _dedupe(assets)
    except Exception:
        return []


__all__ = ["RequiredAsset", "RunPlan", "extract_required_assets"]
