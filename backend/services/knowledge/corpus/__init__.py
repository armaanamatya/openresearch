"""Persistent cross-paper literature corpus (``runs/_corpus/``).

Phase 1 of the literature-corpus design: a global, content-addressed store of
papers related to reproduction targets, seeded from each target's own
reference list and Semantic Scholar citation graph. Consumed per-run as the
bounded ``prior_work_refs`` context manifest.

House rules (same contract as the sibling ``connectors`` package):
  - Flag-gated at the call site by ``OPENRESEARCH_LITERATURE_CORPUS``
    (network operations additionally require
    ``OPENRESEARCH_LITERATURE_GROUNDING``); always importable.
  - Fail-soft throughout — a broken corpus must never break a run.
  - Bounded everywhere: explicit ``MAX_*`` constants, never unbounded output.
"""

from backend.services.knowledge.corpus.builder import (
    BuildReport,
    TargetPaper,
    build_corpus,
    literature_corpus_enabled,
)
from backend.services.knowledge.corpus.manifest import build_prior_work_refs
from backend.services.knowledge.corpus.store import (
    CorpusStore,
    corpus_root,
    normalize_paper_id,
)

__all__ = [
    "BuildReport",
    "CorpusStore",
    "TargetPaper",
    "build_corpus",
    "build_prior_work_refs",
    "corpus_root",
    "literature_corpus_enabled",
    "normalize_paper_id",
]
