"""Phase-2 Context Engineer — deterministically assembles the minimal-correct
context window for one ``WorkCluster``.

Public API::

    from backend.agents.rdr.context_engineer import build_context
    ctx = build_context(cluster, paper=text, artifacts=done_artifacts)

See ``docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`` §6.
"""

from __future__ import annotations

import re
from dataclasses import replace

from backend.agents.rdr.models import (
    AgentContext,
    Artifacts,
    CitedSection,
    WorkCluster,
)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count as ``len(text) // 4`` (consistent with the spec)."""
    return len(text) // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Paper section splitting
# ---------------------------------------------------------------------------

# Recognises the following heading forms (at line start):
#   Markdown:    # Title / ## Title / ### Title
#   Numbered:    5 Title / 5. Title / 5.2 Title / 5.2.3 Title
#   Appendix:    Appendix E / Appendix E.1 Title / Appendix A.3.1 Title
#
# The pattern captures (level_chars, num_prefix, app_prefix, rest_of_title).
# We map these to a canonical heading string and numeric depth for comparison.

_HEADING_RE = re.compile(
    r"""
    ^                                      # start of line
    (?:
        (?P<md_hashes>\#{1,6})\s+(?P<md_title>.+?)              # ## Title
      | (?P<app_kw>Appendix)\s+(?P<app_num>[A-Z][0-9]*(?:\.\d+)*)\s*(?P<app_title>.*?)  # Appendix E.1 Title
      | (?P<sec_num>\d+(?:\.\d+)*)\.\s+(?P<dotted_title>.+?)     # 5. Title / 5.2. Sub
      | (?P<sec_num2>\d+(?:\.\d+)*)\s+(?P<plain_title>[A-Z].+?)  # 5 Title (title must start uppercase)
    )
    \s*$                                   # end of line (trailing space ok)
    """,
    re.VERBOSE | re.MULTILINE,
)


def _heading_depth(m: re.Match) -> int:  # type: ignore[type-arg]
    """Return a comparable depth int: lower = higher-level section."""
    if m.group("md_hashes"):
        return len(m.group("md_hashes"))
    if m.group("app_kw"):
        # Appendix at depth 1; E.1 → depth 2, etc.
        parts = m.group("app_num").split(".")
        return len(parts)
    num = m.group("sec_num") or m.group("sec_num2") or ""
    return len(num.split("."))


def _heading_label(m: re.Match) -> str:  # type: ignore[type-arg]
    """Return the canonical heading string, e.g. '5 Experiments'."""
    if m.group("md_hashes"):
        return m.group("md_title").strip()
    if m.group("app_kw"):
        app_num = m.group("app_num")
        app_title = (m.group("app_title") or "").strip()
        return f"Appendix {app_num} {app_title}".strip()
    if m.group("sec_num"):
        return f"{m.group('sec_num')} {m.group('dotted_title').strip()}"
    # sec_num2 / plain_title
    return f"{m.group('sec_num2')} {m.group('plain_title').strip()}"


def _split_sections(paper: str) -> list[tuple[str, str, int]]:
    """Split *paper* into ``(heading_label, body_text, depth)`` triples.

    The body is the text between this heading and the next heading of the
    same-or-higher level (same depth or shallower).  A leading preamble
    before the first heading is dropped (no heading → no section).
    """
    if not paper.strip():
        return []

    # Find all heading positions.
    matches: list[tuple[int, re.Match]] = []  # type: ignore[type-arg]
    for m in _HEADING_RE.finditer(paper):
        matches.append((m.start(), m))

    if not matches:
        return []

    sections: list[tuple[str, str, int]] = []
    for idx, (pos, m) in enumerate(matches):
        label = _heading_label(m)
        depth = _heading_depth(m)
        # body = text after this heading line up to the next heading of the
        # same or higher level (depth <= current depth); subsections are included.
        body_start = m.end()
        body_end = len(paper)
        for next_pos, next_m in matches[idx + 1:]:
            if _heading_depth(next_m) <= depth:
                body_end = next_pos
                break
        body = paper[body_start:body_end].strip()
        sections.append((label, body, depth))
    return sections


# ---------------------------------------------------------------------------
# Citation resolution helpers
# ---------------------------------------------------------------------------

# Parse citation strings like "Section 5", "Appendix E.1", "Table 2", "Figure 3"
_CITE_SECTION_RE = re.compile(
    r"^Section\s+(?P<num>\d+(?:\.\d+)*)$", re.IGNORECASE
)
_CITE_APPENDIX_RE = re.compile(
    r"^Appendix\s+(?P<num>[A-Z][0-9]*(?:\.\d+)*)$", re.IGNORECASE
)
_CITE_TABLE_RE = re.compile(r"^Table\s+(?P<num>\d+)$", re.IGNORECASE)
_CITE_FIGURE_RE = re.compile(r"^Figure\s+(?P<num>\d+)$", re.IGNORECASE)


def _section_num_matches_citation(label: str, cite_num: str) -> bool:
    """True if the section label's numeric prefix matches *cite_num*.

    "Section 5" matches a section labelled "5 Experiments" but NOT "5.2 Sub".
    "Section 5.2" matches "5.2 Sub" but NOT "5 Experiments".
    """
    # Extract the leading number from the label (handles "5 Title", "5.2 Title")
    label_m = re.match(r"^(\d+(?:\.\d+)*)", label.strip())
    if not label_m:
        return False
    label_num = label_m.group(1)
    # Exact numeric prefix match (cite_num == label_num)
    return label_num == cite_num


def _appendix_num_matches_citation(label: str, cite_num: str) -> bool:
    """True if the section label is an appendix with the right designator."""
    # Label looks like "Appendix E Title" or "Appendix E.1 Title"
    app_m = re.match(r"^Appendix\s+([A-Z][0-9]*(?:\.\d+)*)", label, re.IGNORECASE)
    if not app_m:
        return False
    return app_m.group(1).upper() == cite_num.upper()


def _resolve_citation(
    citation: str,
    sections: list[tuple[str, str, int]],
) -> tuple[str, str] | None:
    """Return ``(heading, body)`` for the first section matching *citation*, or None."""
    sec_m = _CITE_SECTION_RE.match(citation)
    if sec_m:
        cite_num = sec_m.group("num")
        for label, body, _ in sections:
            if _section_num_matches_citation(label, cite_num):
                return label, body
        return None

    app_m = _CITE_APPENDIX_RE.match(citation)
    if app_m:
        cite_num = app_m.group("num")
        for label, body, _ in sections:
            if _appendix_num_matches_citation(label, cite_num):
                return label, body
        return None

    tbl_m = _CITE_TABLE_RE.match(citation)
    if tbl_m:
        tbl_str = f"Table {tbl_m.group('num')}"
        for label, body, _ in sections:
            if re.search(re.escape(tbl_str), body, re.IGNORECASE):
                return label, body
        return None

    fig_m = _CITE_FIGURE_RE.match(citation)
    if fig_m:
        fig_str = f"Figure {fig_m.group('num')}"
        for label, body, _ in sections:
            if re.search(re.escape(fig_str), body, re.IGNORECASE):
                return label, body
        return None

    return None


# ---------------------------------------------------------------------------
# Stopwords for lexical retrieval fallback
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of is are was were be been being "
    "have has had do does did will would could should may might shall can "
    "with by from as its it its that this these those not no nor so".split()
)


def _word_tokens(text: str) -> set[str]:
    """Lowercased non-stopword words of ≥ 3 characters from *text*."""
    return {
        w
        for w in re.findall(r"[a-z]{3,}", text.lower())
        if w not in _STOPWORDS
    }


def _lexical_score(query_tokens: set[str], section_text: str) -> int:
    """Count of query tokens present in *section_text*."""
    section_tokens = _word_tokens(section_text)
    return len(query_tokens & section_tokens)


# ---------------------------------------------------------------------------
# Field builders
# ---------------------------------------------------------------------------


def _build_leaf_contract(cluster: WorkCluster) -> str:
    """Format the verbatim leaf-contract header + per-leaf entries."""
    n = len(cluster.leaves)
    total_w = cluster.weight
    lines = [
        f'This cluster ("{cluster.title}") is graded on {n} requirement{"s" if n != 1 else ""}'
        f" (total weight {total_w:.1f}).",
        "Implement so that each is satisfiable by an automated reproducibility judge.",
        "",
    ]
    for i, leaf in enumerate(cluster.leaves, start=1):
        lines.append(f"[{i}] (weight {leaf.weight:.1f}) {leaf.requirements}")
    return "\n".join(lines)


def _build_paper_sections(
    cluster: WorkCluster,
    sections: list[tuple[str, str, int]],
) -> list[CitedSection]:
    """Resolve citations + lexical fallback; return deduplicated ``CitedSection`` list."""
    included_labels: dict[str, CitedSection] = {}  # label → CitedSection (insertion-ordered)

    # 1. Resolve explicit citations from cluster.paper_citations.
    for citation in cluster.paper_citations:
        result = _resolve_citation(citation, sections)
        if result is None:
            continue
        heading, body = result
        if heading not in included_labels:
            included_labels[heading] = CitedSection(
                citation=citation, heading=heading, text=body
            )

    # 2. Lexical fallback for leaves with no citations.
    uncited_leaves = [lf for lf in cluster.leaves if not lf.paper_citations]
    if uncited_leaves and sections:
        # Build a combined query from all uncited leaves' requirements.
        combined = " ".join(lf.requirements for lf in uncited_leaves)
        query_tokens = _word_tokens(combined)
        if query_tokens:
            # Score sections not already included.
            best_label: str | None = None
            best_score = -1
            for label, body, _ in sections:
                if label in included_labels:
                    continue
                score = _lexical_score(query_tokens, f"{label} {body}")
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_label is not None and best_score > 0:
                # Find the body for best_label.
                body = next(b for lbl, b, _ in sections if lbl == best_label)
                # Short query: first 6 meaningful tokens joined
                short_query = " ".join(sorted(query_tokens)[:6])
                included_labels[best_label] = CitedSection(
                    citation=f"semantic:{short_query}",
                    heading=best_label,
                    text=body,
                )

    return list(included_labels.values())


def _build_dependency_artifacts(
    cluster: WorkCluster,
    artifacts: dict[str, Artifacts],
) -> dict[str, str]:
    """Merge .files dicts of depended-on clusters."""
    merged: dict[str, str] = {}
    for dep_id in cluster.depends_on:
        dep = artifacts.get(dep_id)
        if dep is None:
            continue
        merged.update(dep.files)
    return merged


def _build_prior_feedback(
    cluster: WorkCluster,
    prior_scores: dict | None,
) -> str | None:
    """Build repair-feedback string or return None."""
    if prior_scores is None:
        return None

    cluster_leaf_ids = {lf.id for lf in cluster.leaves}
    leaf_req_by_id = {lf.id: lf.requirements for lf in cluster.leaves}
    weak_threshold = 0.5

    leaf_score_entries: list[dict] = prior_scores.get("leaf_scores", [])
    weak = [
        entry
        for entry in leaf_score_entries
        if entry.get("id") in cluster_leaf_ids
        and entry.get("score", 1.0) < weak_threshold
    ]

    if not weak:
        overall = prior_scores.get("overall_score", "?")
        return (
            "=== REPAIR FEEDBACK ===\n"
            f"All leaves of this cluster scored adequately (overall_score={overall}).\n"
            "Re-attempt to improve further — make sure all requirements are fully met."
        )

    lines = ["=== REPAIR FEEDBACK ===", "The following leaves scored below threshold and need improvement:", ""]
    for entry in weak:
        leaf_id = entry["id"]
        justification = entry.get("justification", "")
        requirements = leaf_req_by_id.get(leaf_id, "")
        lines.append(f"Leaf {leaf_id}:")
        lines.append(f"  Requirements: {requirements}")
        lines.append(f"  Score: {entry.get('score', '?'):.2f}")
        lines.append(f"  Justification: {justification}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_working_summary(artifacts: dict[str, Artifacts]) -> str:
    """Compact project-structure summary from all completed artifacts."""
    if not artifacts:
        return ""

    all_paths: list[str] = []
    for art in artifacts.values():
        all_paths.extend(art.files.keys())
    all_paths = sorted(set(all_paths))

    lines = ["=== PROJECT STRUCTURE SO FAR ==="]
    for p in all_paths:
        lines.append(f"  {p}")

    notes = [(cid, art.notes) for cid, art in artifacts.items() if art.notes]
    if notes:
        lines.append("")
        lines.append("=== PRIOR CLUSTER NOTES ===")
        for cid, note in notes:
            lines.append(f"[{cid}] {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token budgeting
# ---------------------------------------------------------------------------

_TRUNCATION_MARKER = "\n... [truncated]"
_MIN_FILE_CHARS = 200  # never truncate a dep file below this length


def _trim_to_budget(
    leaf_contract: str,
    paper_sections: list[CitedSection],
    dependency_artifacts: dict[str, str],
    prior_feedback: str | None,
    working_summary: str,
    token_budget: int,
) -> tuple[list[CitedSection], dict[str, str], str]:
    """Return trimmed (paper_sections, dependency_artifacts, working_summary).

    Priority: drop/shorten working_summary first, then dep-artifact values,
    then paper_section bodies.  leaf_contract and prior_feedback are never trimmed.

    The trimming operates on mutable state containers (a list and a dict) so
    that ``_used()`` always reflects the current trimmed state, avoiding the
    classic closure-captures-name-not-value pitfall.
    """
    # Use single-element containers so mutations are seen by _used().
    _sections: list[list[CitedSection]] = [list(paper_sections)]
    _dep_art: list[dict[str, str]] = [dict(dependency_artifacts)]
    _summary: list[str] = [working_summary]

    def _used() -> int:
        total = estimate_tokens(leaf_contract)
        for cs in _sections[0]:
            total += estimate_tokens(cs.text)
        for v in _dep_art[0].values():
            total += estimate_tokens(v)
        total += estimate_tokens(prior_feedback or "")
        total += estimate_tokens(_summary[0])
        return total

    if _used() <= token_budget:
        return _sections[0], _dep_art[0], _summary[0]

    # (a) Drop working_summary entirely.
    _summary[0] = ""
    if _used() <= token_budget:
        return _sections[0], _dep_art[0], _summary[0]

    # (b) Truncate large dependency_artifact values, then drop whole files.
    dep = _dep_art[0]
    sorted_paths = sorted(dep.keys(), key=lambda p: -len(dep[p]))

    # First pass: truncate long files half, then quarter.
    for path in sorted_paths:
        if _used() <= token_budget:
            break
        content = dep[path]
        for fraction in (0.5, 0.25):
            new_len = max(_MIN_FILE_CHARS, int(len(content) * fraction))
            if new_len < len(content):
                dep[path] = content[:new_len] + _TRUNCATION_MARKER
                break

    if _used() <= token_budget:
        return _sections[0], _dep_art[0], _summary[0]

    # Second pass: drop whole files (largest-original first).
    for path in sorted_paths:
        if _used() <= token_budget:
            break
        dep.pop(path, None)

    if _used() <= token_budget:
        return _sections[0], _dep_art[0], _summary[0]

    # (c) Truncate paper_section bodies, then drop whole sections if still
    # over budget.  Track the *raw content length* (excluding the truncation
    # marker) per section so the halving loop terminates when every section is
    # already at _MIN_SECTION_CHARS.
    _MIN_SECTION_CHARS = 100
    _MARKER_LEN = len(_TRUNCATION_MARKER)
    secs = _sections[0]

    # Keep a parallel list of the current trimmed content lengths (not including
    # the marker) so we can detect when halving would have no further effect.
    content_lens: list[int] = []
    for cs in secs:
        # Strip trailing marker if it was already added (idempotent).
        body = cs.text[: -_MARKER_LEN] if cs.text.endswith(_TRUNCATION_MARKER) else cs.text
        content_lens.append(len(body))

    changed = True
    while _used() > token_budget and changed and secs:
        changed = False
        section_order = sorted(range(len(secs)), key=lambda i: -content_lens[i])
        for i in section_order:
            if _used() <= token_budget:
                break
            new_len = max(_MIN_SECTION_CHARS, content_lens[i] // 2)
            if new_len < content_lens[i]:
                content_lens[i] = new_len
                secs[i] = replace(
                    secs[i],
                    text=secs[i].text[:new_len] + _TRUNCATION_MARKER,
                )
                changed = True

    # If still over budget (all sections at minimum and budget is very tight),
    # drop sections one at a time, shortest body first (least information density).
    while _used() > token_budget and secs:
        drop_i = min(range(len(secs)), key=lambda i: len(secs[i].text))
        secs.pop(drop_i)

    return _sections[0], _dep_art[0], _summary[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_context(
    cluster: WorkCluster,
    *,
    paper: str,
    artifacts: dict[str, Artifacts],
    prior_scores: dict | None = None,
    token_budget: int = 50_000,
) -> AgentContext:
    """Deterministically assemble the minimal-correct context window for one cluster.

    Pure function — no I/O, no LLM, no external state.
    """
    # Pre-compute split sections once; shared by all resolution helpers.
    sections = _split_sections(paper)

    # 1. Leaf contract — never trimmed.
    leaf_contract = _build_leaf_contract(cluster)

    # 2. Paper sections with citation resolution + lexical fallback.
    paper_sections = _build_paper_sections(cluster, sections)

    # 3. Dependency artifacts.
    dependency_artifacts = _build_dependency_artifacts(cluster, artifacts)

    # 4. Prior feedback (repair pass).
    prior_feedback = _build_prior_feedback(cluster, prior_scores)

    # 5. Working summary.
    working_summary = _build_working_summary(artifacts)

    # Token budgeting — trim low-priority content to stay within budget.
    paper_sections, dependency_artifacts, working_summary = _trim_to_budget(
        leaf_contract=leaf_contract,
        paper_sections=paper_sections,
        dependency_artifacts=dependency_artifacts,
        prior_feedback=prior_feedback,
        working_summary=working_summary,
        token_budget=token_budget,
    )

    return AgentContext(
        cluster=cluster,
        leaf_contract=leaf_contract,
        paper_sections=paper_sections,
        dependency_artifacts=dependency_artifacts,
        prior_feedback=prior_feedback,
        working_summary=working_summary,
    )


__all__ = ["build_context", "estimate_tokens"]
