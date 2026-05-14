"""Resolve the rubric the rubric-verifier scores a reproduction against.

Two sources, one stable interface:
  * BundleRubricSource    — a vendored PaperBench bundle exists for this paper;
                            its rubric.json is the authoritative rubric.
  * GeneratedRubricSource — no bundle; the rubric-verifier agent derives a
                            PaperBench-style rubric from the paper claim map
                            (phase 1 of its two-phase prompt). The source
                            carries no pre-built rubric — load_rubric() is None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

# NOTE: `backend.evals.paperbench.bundle` is imported lazily inside the methods
# below. A module-level import pulls in `backend.evals/__init__.py`, which
# eagerly imports `backend.evals.runner` -> `backend.agents.orchestrator` — and
# the orchestrator imports this module, so an eager import here is a circular
# import. Deferring to call time breaks the cycle; `bundle` is only needed when
# a real PaperBench bundle is actually loaded.

RubricSourceKind = Literal["paperbench_bundle", "generated"]


@runtime_checkable
class RubricSource(Protocol):
    """Stable interface: where the rubric-verifier's rubric comes from."""

    kind: RubricSourceKind

    def load_rubric(self) -> dict[str, Any] | None:
        """Return the pre-built PaperBench rubric tree, or None when the agent
        must generate one."""
        ...


class BundleRubricSource:
    """The paper has a vendored PaperBench bundle — its rubric.json is authoritative."""

    kind: RubricSourceKind = "paperbench_bundle"

    def __init__(self, paperbench_root: str | Path, paper_id: str) -> None:
        self._root = Path(paperbench_root)
        self._paper_id = paper_id

    def load_rubric(self) -> dict[str, Any]:
        from backend.evals.paperbench.bundle import load_paperbench_bundle

        bundle = load_paperbench_bundle(self._root, self._paper_id)
        return bundle.rubric()


class GeneratedRubricSource:
    """No bundle — the rubric-verifier agent derives the rubric itself."""

    kind: RubricSourceKind = "generated"

    def load_rubric(self) -> None:
        return None


def resolve_rubric_source(
    paperbench_root: str | Path | None,
    paper_id: str | None,
) -> RubricSource:
    """Pick the rubric source for a paper.

    A BundleRubricSource is returned only when a vendored bundle directory
    exists AND validates; anything else (no root, no paper_id, missing or
    malformed bundle) degrades cleanly to GeneratedRubricSource.
    """
    if paperbench_root and paper_id:
        from backend.evals.paperbench.bundle import (
            PaperBenchBundleError,
            load_paperbench_bundle,
        )

        try:
            load_paperbench_bundle(paperbench_root, paper_id)
        except (PaperBenchBundleError, json.JSONDecodeError, OSError):
            # Missing, malformed, or unreadable bundle — degrade to generated.
            return GeneratedRubricSource()
        return BundleRubricSource(paperbench_root, paper_id)
    return GeneratedRubricSource()
