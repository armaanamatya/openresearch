"""Deterministic disk-only skill-library catalog loader (Release-1, part 5.A).

Parses the vendored ``backend/agents/rlm/skills/**/SKILL.md`` playbooks into
an in-memory catalog keyed by frontmatter ``name`` (NOT the containing
directory name — some skills alias their directory, e.g. ``ml-inference/vllm/``
carries frontmatter ``name: serving-llms-vllm``).

Pure library: zero LLM calls, zero network access, no side effects at import
time, no environment-variable reads. Callers decide when/whether to invoke
this and gate that decision behind a feature flag elsewhere — this module
simply parses whatever directory it is pointed at.

Ported from openscience's disk-only skill scan (``backend/cli/src/skill/skill.ts``
``compute()``/``addSkill``, LOCAL-DISK lines only — every network/cloud/user/
installed/config source in the original is deliberately out of scope here)
plus the "did you mean" fuzzy matcher and category-grouping shape from
``backend/cli/src/tool/skill.ts``.

Fail-soft throughout: an unreadable file, a missing/unbalanced frontmatter
fence, malformed YAML, non-mapping frontmatter, or a missing/non-string
``name`` is skipped with a ``logging.warning`` — this module never raises on
a malformed skill file.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"

# A leading ``---`` fence line, YAML content (captured, non-greedy), then a
# closing ``---`` fence line. Everything after the closing fence is the body.
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)

# Description substrings that mark a skill as a prompt-injection attempt.
# Mirrors skill.ts's ``addSkill`` injection block (case-insensitive substring
# match against the lowered description).
_INJECTION_MARKERS = ("always run this skill", "must always run")

# Body lines that read as injection directives, stripped from the loaded
# body text. Mirrors tool/skill.ts's line-183 sanitizer regex exactly
# (case-insensitive, per matching line, no trailing-newline consumption —
# that variant is only used in tool/skill.ts's separate cache-write path,
# which is out of scope here).
_INJECTION_LINE_RE = re.compile(
    r"^.*(?:always run this skill|must always run).*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class SkillMeta:
    """Parsed frontmatter metadata for one on-disk ``SKILL.md``."""

    name: str
    description: str
    category: str
    tags: tuple[str, ...]
    path: Path


# Per-resolved-directory memoization. Keyed by the RESOLVED absolute path so
# two different relative spellings of the same directory share one entry, and
# a tmp_path test directory never collides with the shipped skill library.
_CACHE: dict[Path, dict[str, SkillMeta]] = {}
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    """Drop all memoized catalogs. Test-only — production callers never need this."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _as_str(value: Any) -> str:
    """Best-effort stringify a frontmatter scalar. Never raises."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_tags(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(_as_str(t) for t in raw)
    # A lone scalar tag (e.g. a hand-written ``tags: solo``) — defensive,
    # never a crash.
    return (_as_str(raw),)


def _is_injection(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in _INJECTION_MARKERS)


def _split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Split ``text`` into ``(frontmatter_dict, body)``.

    ``frontmatter_dict`` is ``None`` on any parse failure (no fence found,
    malformed YAML, or YAML that doesn't parse to a mapping); ``body``
    degrades to the full input text in that case so callers always get a
    string back.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None, text
    body = text[match.end():]
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("skill_catalog: malformed YAML frontmatter: %s", exc)
        return None, body
    if not isinstance(data, dict):
        return None, body
    return data, body


def load_catalog(skills_dir: Path | None = None) -> dict[str, SkillMeta]:
    """Load (and memoize) the skill catalog rooted at ``skills_dir``.

    Default directory is the vendored ``backend/agents/rlm/skills/``. Globs
    ``**/SKILL.md`` recursively and keys the result by frontmatter ``name``.
    Fail-soft: any unreadable file / missing frontmatter fence / malformed
    YAML / non-mapping frontmatter / missing-or-non-string ``name`` is
    skipped with a ``logging.warning`` (never raised). A description
    containing an injection marker is blocked. A duplicate ``name`` across
    two files: last one wins (deterministic sorted-path glob order), warned.
    """
    directory = (skills_dir if skills_dir is not None else _DEFAULT_SKILLS_DIR).resolve()

    with _CACHE_LOCK:
        cached = _CACHE.get(directory)
        if cached is not None:
            return cached

    catalog: dict[str, SkillMeta] = {}
    if directory.is_dir():
        for md_path in sorted(directory.glob("**/SKILL.md")):
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("skill_catalog: failed to read %s: %s", md_path, exc)
                continue

            frontmatter, _body = _split_frontmatter(text)
            if frontmatter is None:
                logger.warning("skill_catalog: no usable frontmatter in %s", md_path)
                continue

            name = frontmatter.get("name")
            if not isinstance(name, str) or not name.strip():
                logger.warning("skill_catalog: missing/invalid name in %s", md_path)
                continue
            name = name.strip()

            description = _as_str(frontmatter.get("description"))
            if _is_injection(description):
                logger.warning(
                    "skill_catalog: blocked skill with injection pattern "
                    "(name=%s, path=%s)",
                    name, md_path,
                )
                continue

            category = _as_str(frontmatter.get("category"))
            tags = _coerce_tags(frontmatter.get("tags"))

            if name in catalog:
                logger.warning(
                    "skill_catalog: duplicate skill name %r (existing=%s, "
                    "duplicate=%s) — last one wins",
                    name, catalog[name].path, md_path,
                )

            catalog[name] = SkillMeta(
                name=name,
                description=description,
                category=category,
                tags=tags,
                path=md_path,
            )
    else:
        logger.warning("skill_catalog: skills directory does not exist: %s", directory)

    with _CACHE_LOCK:
        _CACHE[directory] = catalog
    return catalog


def get_skill_body(name: str, skills_dir: Path | None = None) -> str | None:
    """Return the sanitized markdown body for ``name``, or ``None`` if unknown.

    The catalog itself never holds body text (only frontmatter metadata), so
    this re-reads the source file and strips injection-directive lines
    (mirrors tool/skill.ts's line-183 sanitizer).
    """
    catalog = load_catalog(skills_dir)
    meta = catalog.get(name)
    if meta is None:
        return None
    try:
        text = meta.path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("skill_catalog: failed to re-read body for %s: %s", name, exc)
        return None
    _frontmatter, body = _split_frontmatter(text)
    return _INJECTION_LINE_RE.sub("", body).strip()


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _fuzzy_score(query: str, target: str) -> float:
    """Port of openscience's ``fuzzyScore`` (tool/skill.ts:14-30): exact match
    (case-insensitive) -> 1.0, substring containment either direction -> 0.8,
    else the bigram Dice coefficient (0.0 when either string has fewer than 2
    characters, i.e. no bigrams)."""
    q = query.lower()
    t = target.lower()
    if t == q:
        return 1.0
    if t in q or q in t:
        return 0.8
    qb = _bigrams(q)
    tb = _bigrams(t)
    if not qb or not tb:
        return 0.0
    shared = len(qb & tb)
    return (2 * shared) / (len(qb) + len(tb))


def fuzzy_did_you_mean(query: str, names: Iterable[str], *, limit: int = 5) -> list[str]:
    """Rank ``names`` against ``query`` by :func:`_fuzzy_score`; best first.

    Only positive-scoring names are returned (mirrors tool/skill.ts:137-143's
    "did you mean" hint, which filters ``score > 0`` after ranking). Ties are
    broken alphabetically: the upstream JS relies on a stable sort over
    ``Skill.all()``'s insertion order, which isn't a meaningful signal to
    preserve here, so this port picks the deterministic alternative (fixed
    alphabetical tiebreak) rather than depend on caller iteration order.
    """
    scored = [(name, _fuzzy_score(query, name)) for name in names]
    positive = [(name, score) for name, score in scored if score > 0]
    positive.sort(key=lambda pair: (-pair[1], pair[0]))
    return [name for name, _score in positive[:limit]]


def group_by_category(catalog: Mapping[str, SkillMeta]) -> dict[str, list[SkillMeta]]:
    """Group catalog entries by category; missing/empty category -> ``"other"``."""
    groups: dict[str, list[SkillMeta]] = {}
    for meta in catalog.values():
        cat = meta.category or "other"
        groups.setdefault(cat, []).append(meta)
    return groups
